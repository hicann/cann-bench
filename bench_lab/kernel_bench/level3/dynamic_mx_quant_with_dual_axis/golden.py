#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch
import numpy

"""
DynamicMxQuantWithDualAxis 算子 Torch Golden 参考实现

对输入张量同时在 -1 轴和 -2 轴执行 Microscaling (MX) 动态量化，
分别输出两路量化结果 y1/mxscale1 与 y2/mxscale2。

本实现复用单轴 DynamicMxQuant 的量化逻辑（见 dynamic_mx_quant），
分别调用 axis=-1 与 axis=-2，blocksize 固定为 32。

支持 3 种 scale 算法：
  - scale_alg=0: 基础共享指数（适用所有 dst_type）
  - scale_alg=1: FP8 块缩放 cuBLAS 风格（仅 FP8 类型）
  - scale_alg=2: FP4 自定义 max（仅 FP4_E2M1；当前 torch_npu Python 接口未暴露 dst_type_max，默认 0.0）

参考来源: /home/g00944280/canbranch/golden/DynamicMxQuantWithDualAxis/aclnnDynamicMxQuantWithDualAxis.py
"""

# dst_type 编码到格式字符串的映射
DST_TYPE_MAP = {
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    40: "float4_e2m1",
    41: "float4_e1m2",
}


def _get_dtype_range(dt):
    """获取数据类型的表示范围"""
    if "float4_e2m1" in str(dt):
        return -6.0, 6.0
    if "float4_e1m2" in str(dt):
        return -1.75, 1.75
    if "float8_e8m0" in str(dt):
        return 2.0 ** -127, 2.0 ** 127
    if "float8_e5m2" in str(dt):
        return -57344.0, 57344.0
    if "float8_e4m3fn" in str(dt):
        return -448.0, 448.0
    numpy_dtype = numpy.dtype(dt)
    if numpy_dtype.kind in "iu":
        info = numpy.iinfo(numpy_dtype)
    else:
        info = numpy.finfo(numpy_dtype)
    return info.min, info.max


def _mx_round_mantissa(fp_array: numpy.ndarray, round_mode: str):
    """
    对尾数进行舍入。
    - rint: 银行家舍入（tie to even）
    - round/nearest: 四舍五入（tie away from zero）
    - floor: 向负无穷舍入
    """
    if round_mode in ("rint", "even"):
        fp_array = numpy.rint(fp_array)
    elif round_mode in ("round", "nearest"):
        sign = numpy.signbit(fp_array)
        rounded_abs = numpy.floor(numpy.abs(fp_array) + numpy.array([0.5], dtype=fp_array.dtype))
        fp_array = numpy.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":
        fp_array = numpy.floor(fp_array)
    elif round_mode == "ceil":
        fp_array = numpy.ceil(fp_array)
    elif round_mode == "trunc":
        fp_array = numpy.trunc(fp_array)
    else:
        raise ValueError(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_calculate_share_exp(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str):
    """Algorithm 0: OCP 标准共享指数计算"""
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = _get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(numpy.log2(max_norm))
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True)
    res = numpy.floor(
        numpy.log2(fp_abs_max.astype(numpy.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
    ) - ele_emax
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_calculate_share_exp_1(fp_array: numpy.ndarray, scale_axis: int, mx_ele_dtype: str,
                               max_norm: float = None, subnormal: bool = True):
    """Algorithm 1/2: cuBLAS 风格 scale 计算（通过 FP32 位操作实现向上取整的 log2）

    Args:
        max_norm: 自定义最大值，None 时从 mx_ele_dtype 推导
        subnormal: 是否考虑 subnormal 条件（alg2 时为 False）
    """
    if max_norm is None:
        max_norm = _get_dtype_range(mx_ele_dtype)[1]
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True).astype(numpy.float32)
    s_fp32 = fp_abs_max / max_norm
    binary_ints = numpy.array(s_fp32.view(numpy.uint32))
    exponent_mask = numpy.uint32(0x7F800000)
    mantissa_mask = numpy.uint32(0x007FFFFF)
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(numpy.int16)
    mantissas = (binary_ints & mantissa_mask)
    # 如果尾数非零，指数向上取整
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    if subnormal:
        condition_2 = (exponents_int16 == 0) & (mantissas > 2 ** 22)
    else:
        condition_2 = False
    exponents_int16 = numpy.where((condition_1 | condition_2), exponents_int16 + 1, exponents_int16)
    res = (exponents_int16 - 127).astype(numpy.float32)
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_calculate_share_exp_dynamic_dtype_range(fp_array: numpy.ndarray, scale_axis: int,
                                                 mx_ele_dtype: str, max_norm: float,
                                                 subnormal: bool = False):
    """Algorithm 2 主路径: 基于 BF16 位操作计算 scale（dst_type_max=6/7 的尾轴场景）

    将 abs_max 转为 bfloat16，通过 BF16 的指数和尾数位判断是否需要向上取整。
    """
    from ml_dtypes import bfloat16 as bfloat16_type
    fp_abs_max = numpy.max(numpy.abs(fp_array), axis=scale_axis, keepdims=True).astype(bfloat16_type)
    binary_ints = numpy.array(fp_abs_max.view(numpy.uint16))
    exponent_mask = numpy.uint16(0x7F80)
    mantissa_mask = numpy.uint16(0x007F)
    # 提取指数部分
    exponents = (binary_ints & exponent_mask) >> 7
    exponents_int16 = exponents.astype(numpy.int16)
    # 提取尾数部分
    mantissas = (binary_ints & mantissa_mask).astype(numpy.uint16)
    # threshold 取决于 max_norm: 6 -> 0x0040, 7 -> 0x0060
    threshold = numpy.uint16(0x0040) if max_norm == 6 else numpy.uint16(0x0060)
    condition = mantissas > threshold
    exponents_int16_1 = numpy.where(condition, exponents_int16 + 1, exponents_int16)
    exponents_int16_1 -= 2
    res = (exponents_int16_1 - 127).astype(numpy.float32)
    res[exponents_int16 == 255] = float("inf")
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_reshape_to_blocks(fp_array: numpy.ndarray, axis: int, block_size: int):
    """将输入在 axis 维度 pad 到 block_size 整数倍，然后 reshape 为 [..., num_blocks, block_size, ...]"""
    fp_array = numpy.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size if pad_size > 0 else 0
    if pad_size > 0:
        fp_array = numpy.pad(fp_array, pad, 'constant')
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape


def _mx_quantize_to_element_format(fp_array: numpy.ndarray, share_exp: numpy.ndarray,
                                   mx_ele_dtype: str, round_mode: str):
    """将输入按 share_exp 缩放后，量化到目标 FP4/FP8 格式（精确模拟位宽约束）"""
    mx_dtype = str(mx_ele_dtype)
    exp_bits = 0
    mantissa_bits = 0
    if "float4_e2m1" in mx_dtype:
        exp_bits = 2
        mantissa_bits = 1
    elif "float4_e1m2" in mx_dtype:
        exp_bits = 1
        mantissa_bits = 2
    elif "float8_e4m3fn" in mx_dtype:
        exp_bits = 4
        mantissa_bits = 3
    elif "float8_e5m2" in mx_dtype:
        exp_bits = 5
        mantissa_bits = 2

    max_norm = _get_dtype_range(mx_dtype)[1]

    ret = fp_array / (2 ** share_exp)
    private_exp = numpy.floor(numpy.log2(numpy.abs(ret.astype(numpy.float32)) + (ret == 0))
                              ).astype(fp_array.dtype, copy=False)
    if "float8_e4m3fn" in mx_dtype or "float8_e5m2" in mx_dtype:
        min_exp = -(2 ** (exp_bits - 1)) + 2
    else:
        min_exp = -(2 ** (exp_bits - 1)) + exp_bits
    private_exp = private_exp.clip(min=min_exp)
    # Scale up so appropriate number of bits are in the integer portion
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    # Undo scaling
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    # Clamp to representable range
    numpy.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret


def _mx_undo_reshape_to_blocks(fp_array: numpy.ndarray, axis: int,
                               orig_shape: tuple, padded_shape: tuple):
    """撤销 reshape_to_blocks 的操作，恢复原始 shape"""
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    fp_array = numpy.squeeze(fp_array, axis=axis + 1)
    return fp_array


def _interleave(tensor: numpy.ndarray, axis: int, n_group: int = 2) -> numpy.ndarray:
    """在指定 axis 上做 interleave 重排（非尾轴时需要）"""
    length = tensor.shape[axis]
    if length % n_group != 0:
        raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")

    group_length = length // n_group
    shape = list(tensor.shape)
    new_shape = shape[:axis] + [group_length, 2] + shape[axis + 1:]
    reshaped = tensor.reshape(new_shape)
    transpose_order = (
        list(range(0, axis + 1)) +
        list(range(axis + 2, len(new_shape))) +
        [axis + 1]
    )
    transposed = reshaped.transpose(transpose_order)
    return transposed


def _pad_to_even(tensor: numpy.ndarray, axis: int) -> numpy.ndarray:
    """将 axis 维度 pad 到偶数长度（Cube 要求 scale 为偶数个）"""
    length = tensor.shape[axis]
    if length % 2 == 0:
        return tensor
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)
    padded_tensor = numpy.pad(tensor, pad_width, mode='constant', constant_values=2 ** -127)
    return padded_tensor


def dynamic_mx_quant(
    x: torch.Tensor,
    axis: int = -1,
    round_mode: str = "rint",
    dst_type: int = 40,
    blocksize: int = 32,
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对输入张量执行 Microscaling (MX) 动态量化（单轴）。

    Args:
        x: 输入张量，支持 float16/bfloat16，1-7D
        axis: 量化轴，默认 -1（最后一维）
        round_mode: 舍入模式，rint/floor/round
        dst_type: 输出类型编码 (35=FP8_E5M2, 36=FP8_E4M3FN, 40=FP4_E2M1, 41=FP4_E1M2)
        blocksize: 量化分组大小，32 的倍数
        scale_alg: Scale 算法 (0=基础共享指数, 1=FP8块缩放, 2=FP4自定义max)
        dst_type_max: 自定义量化范围上限，仅 scale_alg=2 有效

    Returns:
        (y, mxscale): 量化输出和对应的 scale 张量（均为 uint8 字节表示）
    """
    mx_ele_dtype = DST_TYPE_MAP[dst_type]

    # 转为 numpy 计算
    if x.dtype == torch.bfloat16:
        from ml_dtypes import bfloat16
        fp_array = x.to(torch.float32).numpy().astype(bfloat16)
    elif x.dtype == torch.float16:
        fp_array = x.numpy().astype(numpy.float16)
    else:
        fp_array = x.numpy().astype(numpy.float32)

    # 规范化 axis
    axis_norm = len(fp_array.shape) + axis if axis < 0 else axis

    # padding & reshape to block_size
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis_norm, blocksize)

    # 计算共享指数
    if scale_alg == 2:
        # Algorithm 2: FP4 自定义 max（仅 float4_e2m1）
        effective_max = dst_type_max if dst_type_max != 0.0 else 6.0
        if effective_max == 6 or effective_max == 7:
            # dst_type_max=6/7: 统一走 BF16 位操作路径，与 NPU 行为一致
            share_exp = _mx_calculate_share_exp_dynamic_dtype_range(
                fp_array, scale_axis=axis_norm + 1,
                mx_ele_dtype=mx_ele_dtype, max_norm=effective_max, subnormal=False)
        else:
            # dst_type_max 非 6/7 (如 8.0, 12.0): 走 FP32 位操作路径
            share_exp = _mx_calculate_share_exp_1(fp_array, scale_axis=axis_norm + 1,
                                                   mx_ele_dtype=mx_ele_dtype,
                                                   max_norm=effective_max, subnormal=False)
    elif scale_alg == 0 or (mx_ele_dtype in ("float4_e2m1", "float4_e1m2")):
        share_exp = _mx_calculate_share_exp(fp_array, scale_axis=axis_norm + 1,
                                            mx_ele_dtype=mx_ele_dtype)
    else:
        share_exp = _mx_calculate_share_exp_1(fp_array, scale_axis=axis_norm + 1,
                                               mx_ele_dtype=mx_ele_dtype)

    # 限制 scale 范围
    scale_emax = 2 ** (8 - 1) - 1  # E8M0: 127
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    # 量化元素
    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)

    # 恢复原始 shape
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis_norm, orig_shape, padded_shape)
    share_exp = numpy.squeeze(share_exp, axis=axis_norm + 1)

    # 构建 scale 数组 (2^share_exp)
    scale_array = 2 ** share_exp

    # NPU 会将 NaN cast 为 0
    ele_array = numpy.nan_to_num(ele_array, nan=0.0, copy=False)

    # 将 ele_array 转为目标 dtype 的 uint8 表示
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    # 构建 mxscale 输出（interleaved 格式）
    scale_array_pad = _pad_to_even(scale_array, axis=axis_norm)
    result_shape = list(scale_array_pad.shape) + [2]
    result_shape[axis_norm] = scale_array_pad.shape[axis_norm] // 2

    # 非尾轴需要 interleave
    if axis_norm != (len(fp_array.shape) - 2):  # -2 因为 reshape_to_blocks 多了一维
        scale_array_pad = _interleave(scale_array_pad, axis=axis_norm)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    # 转为 uint8（FP8_E8M0 的位表示）
    try:
        from en_dtypes import float8_e8m0
        scale_out = scale_array_pad.astype(float8_e8m0, copy=False)
        scale_uint8 = scale_out.view(numpy.uint8)
    except (ImportError, ModuleNotFoundError):
        # fallback: 手动编码 E8M0 = biased exponent of power-of-2
        scale_f32 = scale_array_pad.astype(numpy.float32)
        scale_uint8_vals = numpy.zeros(scale_f32.shape, dtype=numpy.uint8)
        valid = numpy.isfinite(scale_f32) & (scale_f32 > 0)
        log_vals = numpy.zeros_like(scale_f32)
        log_vals[valid] = numpy.log2(scale_f32[valid])
        biased = numpy.clip(numpy.round(log_vals) + 127, 0, 254).astype(numpy.uint8)
        scale_uint8_vals[valid] = biased[valid]
        scale_uint8_vals[~valid & (scale_f32 == 0)] = 0
        scale_uint8_vals[numpy.isnan(scale_f32)] = 255
        # -inf 对应 biased=0
        scale_uint8_vals[numpy.isneginf(scale_array_pad.astype(numpy.float32))] = 0
        scale_uint8 = scale_uint8_vals

    # 将 ele_array 转为目标量化 dtype。
    # 注意：对于 FP8，返回 torch.float8_e4m3fn / float8_e5m2 张量，与 NPU 输出 dtype 一致，
    # 这样框架才能按同 dtype 做数值比较；字节级等价性由 test_golden_vs_npu.py 内部处理。
    if mx_ele_dtype in ("float8_e4m3fn", "float8_e5m2"):
        if mx_ele_dtype == "float8_e4m3fn":
            from ml_dtypes import float8_e4m3fn
            ele_typed = ele_array.astype(float8_e4m3fn, copy=False)
            torch_dtype = torch.float8_e4m3fn
        else:
            from ml_dtypes import float8_e5m2
            ele_typed = ele_array.astype(float8_e5m2, copy=False)
            torch_dtype = torch.float8_e5m2
        y_uint8 = ele_typed.view(numpy.uint8).reshape(x.shape)
        y_tensor = torch.from_numpy(y_uint8.copy()).view(torch_dtype)
    elif mx_ele_dtype in ("float4_e2m1", "float4_e1m2"):
        # FP4 手动编码：将量化浮点值编码为 4-bit，两两打包成 uint8
        # NPU 打包格式: byte = low_nibble | (high_nibble << 4)
        # low_nibble = element[2i], high_nibble = element[2i+1]
        if mx_ele_dtype == "float4_e2m1":
            # E2M1: sign(1) exp(2) man(1), values: 0,0.5,1,1.5,2,3,4,6
            _val_to_code = {
                0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4, 3.0: 5, 4.0: 6, 6.0: 7,
                -0.0: 8, -0.5: 9, -1.0: 10, -1.5: 11, -2.0: 12, -3.0: 13, -4.0: 14, -6.0: 15,
            }
        else:
            # E1M2: sign(1) exp(1) man(2), values: 0,0.25,0.5,0.75,1,1.25,1.5,1.75
            _val_to_code = {
                0.0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4, 1.25: 5, 1.5: 6, 1.75: 7,
                -0.0: 8, -0.25: 9, -0.5: 10, -0.75: 11, -1.0: 12, -1.25: 13, -1.5: 14, -1.75: 15,
            }
        # 构建查找数组：用 float32 值索引
        ele_flat = ele_array.flatten().astype(numpy.float32)
        codes = numpy.zeros(len(ele_flat), dtype=numpy.uint8)
        for val, code in _val_to_code.items():
            mask = ele_flat == numpy.float32(val)
            # 处理 -0.0 == 0.0 的问题
            if val == 0.0 and code == 0:
                mask = mask & ~numpy.signbit(ele_flat)
            elif code == 8:  # -0.0
                mask = (ele_flat == 0.0) & numpy.signbit(ele_flat)
            codes[mask] = code

        # 打包: 每两个 FP4 code 打包为一个 uint8
        # NPU 格式: byte = codes[2i] | (codes[2i+1] << 4)
        codes_pairs = codes.reshape(-1, 2)
        packed = (codes_pairs[:, 0] | (codes_pairs[:, 1] << 4)).astype(numpy.uint8)
        y_uint8 = packed.reshape(x.shape[:-1] + (x.shape[-1] // 2,))
        y_tensor = torch.from_numpy(y_uint8.copy())
    else:
        # fallback
        y_tensor = torch.from_numpy(ele_array.astype(numpy.float32)).reshape(x.shape).to(x.dtype)

    mxscale_tensor = torch.from_numpy(scale_uint8)

    return y_tensor, mxscale_tensor


def dynamic_mx_quant_with_dual_axis(
    x: torch.Tensor,
    round_mode: str = "rint",
    dst_type: int = 40,
    scale_alg: int = 0,
    dst_type_max: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对输入张量同时在 -1 轴和 -2 轴执行 Microscaling (MX) 动态量化。

    Args:
        x: 输入张量，float16/bfloat16，2-7D
        round_mode: 舍入模式，默认 rint
        dst_type: 输出类型编码 (35=FP8_E5M2, 36=FP8_E4M3FN, 40=FP4_E2M1, 41=FP4_E1M2)
        scale_alg: Scale 算法 (0=基础共享指数, 1=FP8块缩放, 2=FP4自定义max)
        dst_type_max: 自定义量化范围上限，仅 scale_alg=2 有效；当前 NPU Python 接口未暴露，默认 0.0

    Returns:
        (y1, mxscale1, y2, mxscale2): 均为 uint8 字节表示
    """
    y1, mxscale1 = dynamic_mx_quant(
        x, axis=-1, round_mode=round_mode, dst_type=dst_type,
        blocksize=32, scale_alg=scale_alg, dst_type_max=dst_type_max,
    )
    y2, mxscale2 = dynamic_mx_quant(
        x, axis=-2, round_mode=round_mode, dst_type=dst_type,
        blocksize=32, scale_alg=scale_alg, dst_type_max=dst_type_max,
    )
    return y1, mxscale1, y2, mxscale2
