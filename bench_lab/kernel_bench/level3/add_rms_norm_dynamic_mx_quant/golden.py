#!/usr/bin/env python3
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

"""
AddRmsNormDynamicMxQuant 算子 Golden 参考实现（CPU）

两阶段实现：
1. add_rms_norm：PyTorch float32 计算 x = x1 + x2, rstd, y = x * rstd * gamma + beta。
2. dynamic_mx_quant：numpy + ml_dtypes/en_dtypes 模拟 MX 量化，输出 y 的 bit-pattern（uint8）
   与 mxscale（uint8）。

为适应 cann-bench 默认 FP64 golden 输入，函数通过 x1_dtype 参数将中间结果 cast 回原始 dtype，
再执行量化，从而与 NPU 的 FP16/BF16 计算路径对齐。
"""

from typing import Optional, Tuple

import numpy as np
import torch
from ml_dtypes import bfloat16, float8_e4m3fn, float8_e5m2


def _import_en_dtypes():
    """延迟导入 en_dtypes，缺失时给出清晰提示。"""
    try:
        import en_dtypes
        return en_dtypes
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "en_dtypes is needed for AddRmsNormDynamicMxQuant golden. "
            "Please install with `pip3 install en-dtypes`"
        ) from e


DATA_TYPE_INT_TO_STR = {
    35: "float8_e5m2",
    36: "float8_e4m3fn",
    40: "float4_e2m1",
    41: "float4_e1m2",
}


def _get_dtype_range(dt):
    """获取目标类型的值域。"""
    if "bfloat16" in str(dt):
        return -float.fromhex("0x1.FEp127"), float.fromhex("0x1.FEp127")
    if "uint4" in str(dt):
        return 0, 15
    if "int4" in str(dt):
        return -8, 7
    if "bool" in str(dt):
        return 0, 1
    if "float4_e2m1" in str(dt):
        return -float.fromhex("0x1.8p2"), float.fromhex("0x1.8p2")
    if "float4_e1m2" in str(dt):
        return -float.fromhex("0x1.Cp0"), float.fromhex("0x1.Cp0")
    if "float8_e8m0" in str(dt):
        return float.fromhex("0x1.p-127"), float.fromhex("0x1.p127")
    if "float8_e5m2" in str(dt):
        return -float.fromhex("0x1.Cp15"), float.fromhex("0x1.Cp15")
    if "float8_e4m3fn" in str(dt):
        return -float.fromhex("0x1.Cp8"), float.fromhex("0x1.Cp8")
    np_dtype = np.dtype(dt)
    if np_dtype.kind in "iu":
        info = np.iinfo(np_dtype)
    else:
        info = np.finfo(np_dtype)
    return info.min, info.max


def _mx_round_mantissa(fp_array: np.ndarray, round_mode: str):
    """MX 量化舍入。"""
    if round_mode in ("rint", "even"):
        fp_array = np.rint(fp_array)
    elif round_mode in ("round", "nearest"):
        sign = np.signbit(fp_array)
        rounded_abs = np.floor(np.abs(fp_array) + np.array([0.5], dtype=fp_array.dtype))
        fp_array = np.where(sign, -rounded_abs, rounded_abs)
    elif round_mode == "floor":
        fp_array = np.floor(fp_array)
    elif round_mode == "ceil":
        fp_array = np.ceil(fp_array)
    elif round_mode == "trunc":
        fp_array = np.trunc(fp_array)
    else:
        raise ValueError(f"Unrecognized round method {round_mode}")
    return fp_array


def _mx_calculate_share_exp(fp_array: np.ndarray, scale_axis: int, mx_ele_dtype: str):
    """OCP 标准 share_exp 计算。"""
    FP32_EXPONENT_BIAS = 127
    FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)
    max_norm = _get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(np.log2(max_norm))
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True)
    res = np.floor(
        np.log2(fp_abs_max.astype(np.float32) + FP32_MIN_NORMAL * (fp_abs_max == 0))
    ) - ele_emax
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_calculate_share_exp_cublas(fp_array: np.ndarray, scale_axis: int, mx_ele_dtype: str):
    """cuBLAS share_exp 计算，仅 FP8 有效。"""
    max_norm = _get_dtype_range(mx_ele_dtype)[1]
    ele_emax = int(np.log2(max_norm))
    fp_abs_max = np.max(np.abs(fp_array), axis=scale_axis, keepdims=True).astype(np.float32)
    s_fp32 = fp_abs_max / max_norm
    binary_ints = np.array(s_fp32.view(np.uint32))
    exponent_mask = np.uint32(0x7F800000)
    mantissa_mask = np.uint32(0x007FFFFF)
    exponents = (binary_ints & exponent_mask) >> 23
    exponents_int16 = exponents.astype(np.int16)
    mantissas = binary_ints & mantissa_mask
    condition_1 = (exponents_int16 > 0) & (exponents_int16 < 254) & (mantissas > 0)
    condition_2 = (exponents_int16 == 0) & (mantissas > 2 ** 22)
    exponents_int16 = np.where((condition_1 | condition_2), exponents_int16 + 1, exponents_int16)
    res = (exponents_int16 - 127).astype(np.float32)
    res[fp_abs_max == 0] = -float("inf")
    return res


def _mx_reshape_to_blocks(fp_array: np.ndarray, axis: int, block_size: int):
    """padding + reshape 为 block。"""
    fp_array = np.expand_dims(fp_array, axis=axis + 1)
    orig_shape = fp_array.shape
    pad = [[0, 0] for _ in range(len(orig_shape))]
    pad_size = orig_shape[axis] % block_size
    pad[axis][1] = block_size - pad_size
    if pad_size > 0:
        fp_array = np.pad(fp_array, pad, "constant")
    padded_shape = fp_array.shape
    reshape = list(padded_shape)
    reshape[axis + 1] = block_size
    reshape[axis] = reshape[axis] // block_size
    fp_array = fp_array.reshape(reshape)
    return fp_array, orig_shape, padded_shape


def _mx_quantize_to_element_format(
    fp_array: np.ndarray, share_exp: np.ndarray, mx_ele_dtype: str, round_mode: str
):
    """按 share_exp 量化到目标 FP8/FP4 类型。"""
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
    private_exp = np.floor(
        np.log2(np.abs(ret.astype(np.float32)) + (ret == 0))
    ).astype(fp_array.dtype, copy=False)
    if "float8_e4m3fn" in mx_dtype or "float8_e5m2" in mx_dtype:
        min_exp = -(2 ** (exp_bits - 1)) + 2
    else:
        min_exp = -(2 ** (exp_bits - 1)) + exp_bits
    private_exp = private_exp.clip(min=min_exp)
    ret = ret / (2 ** private_exp) * (2 ** mantissa_bits)
    ret = _mx_round_mantissa(ret, round_mode)
    ret = ret / (2 ** mantissa_bits) * (2 ** private_exp)
    np.clip(ret, a_min=-max_norm, a_max=max_norm, out=ret)
    return ret


def _mx_undo_reshape_to_blocks(
    fp_array: np.ndarray, axis: int, orig_shape: tuple, padded_shape: tuple
):
    """撤销 block reshape 与 padding。"""
    fp_array = fp_array.reshape(padded_shape)
    if tuple(padded_shape) != tuple(orig_shape):
        slices = [slice(0, x) for x in orig_shape]
        fp_array = fp_array[tuple(slices)]
    fp_array = np.squeeze(fp_array, axis=axis + 1)
    return fp_array


def _interleave(tensor: np.ndarray, axis: int, n_group: int = 2) -> np.ndarray:
    """沿 axis 做 2-way interleave。"""
    length = tensor.shape[axis]
    if length % n_group != 0:
        raise ValueError(f"Axis length ({length}) must be divisible by n_group ({n_group})")
    group_length = length // n_group
    shape = list(tensor.shape)
    new_shape = shape[:axis] + [group_length, n_group] + shape[axis + 1 :]
    reshaped = tensor.reshape(new_shape)
    transpose_order = list(range(0, axis + 1)) + list(range(axis + 2, len(new_shape))) + [axis + 1]
    return reshaped.transpose(transpose_order)


def _pad_to_even(tensor: np.ndarray, axis: int) -> np.ndarray:
    """沿 axis pad 到偶数长度，用于 scale 打包。"""
    if tensor.shape[axis] % 2 == 0:
        return tensor
    pad_width = [(0, 0)] * tensor.ndim
    pad_width[axis] = (0, 1)
    return np.pad(tensor, pad_width, mode="constant", constant_values=2 ** -127)


def add_rms_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add + RMSNorm 计算。"""
    gamma = gamma.type(torch.float32)
    x = torch.add(x1, x2)
    rstd = torch.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + eps)
    y = x * rstd * gamma
    if beta is not None:
        y += beta
    return y, x, rstd


def dynamic_mx_quant(
    fp_array: np.ndarray,
    mx_ele_dtype: str = "float4_e2m1",
    axis: int = -1,
    block_size: int = 32,
    round_mode: str = "rint",
    scale_alg: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """MX 量化：返回 (mxscale, ele_array)。"""
    if fp_array.dtype.name not in ("bfloat16", "float16", "float32", "float64"):
        raise RuntimeError(f"Unsupported input dtype: {fp_array.dtype.name}")
    if mx_ele_dtype not in ("float4_e2m1", "float4_e1m2", "float8_e4m3fn", "float8_e5m2"):
        raise NotImplementedError(f"Not support {mx_ele_dtype}")

    axis = len(fp_array.shape) + axis if axis < 0 else axis
    fp_array, orig_shape, padded_shape = _mx_reshape_to_blocks(fp_array, axis, block_size)

    if scale_alg == 0 or (mx_ele_dtype in ("float4_e2m1", "float4_e1m2")):
        share_exp = _mx_calculate_share_exp(fp_array, axis + 1, mx_ele_dtype)
    else:
        share_exp = _mx_calculate_share_exp_cublas(fp_array, axis + 1, mx_ele_dtype)

    scale_emax = 2 ** (8 - 1) - 1
    share_exp[share_exp > scale_emax] = float("NaN")
    share_exp[share_exp < -scale_emax] = -scale_emax

    ele_array = _mx_quantize_to_element_format(fp_array, share_exp, mx_ele_dtype, round_mode)
    ele_array = _mx_undo_reshape_to_blocks(ele_array, axis, orig_shape, padded_shape)
    share_exp = np.squeeze(share_exp, axis=axis + 1)

    en_dtypes = _import_en_dtypes()
    ele_dtype_np = getattr(en_dtypes, mx_ele_dtype) if hasattr(en_dtypes, mx_ele_dtype) else None
    if ele_dtype_np is None:
        if mx_ele_dtype == "float8_e4m3fn":
            ele_dtype_np = float8_e4m3fn
        elif mx_ele_dtype == "float8_e5m2":
            ele_dtype_np = float8_e5m2
        else:
            raise RuntimeError(f"Cannot map {mx_ele_dtype} to numpy dtype")

    scale_array = 2 ** share_exp
    if ele_array.dtype.name == "bfloat16":
        ele_array = ele_array.astype("float32", copy=False)

    ele_array = np.nan_to_num(ele_array, nan=0.0, copy=False)
    ele_array = ele_array.astype(ele_dtype_np, copy=False)

    scale_array_pad = _pad_to_even(scale_array, axis=axis)
    result_shape = list(scale_array_pad.shape)
    result_shape.append(2)
    result_shape[axis] = scale_array_pad.shape[axis] // 2
    if axis != (len(fp_array.shape) - 1):
        scale_array_pad = _interleave(scale_array_pad, axis=axis)
    scale_array_pad = scale_array_pad.reshape(result_shape)

    float8_e8m0 = getattr(en_dtypes, "float8_e8m0")
    scale_array = scale_array_pad.astype(float8_e8m0, copy=False)
    return scale_array, ele_array


def fp4_tensor_to_array(tensor: torch.Tensor) -> np.ndarray:
    """将打包的 FP4 uint8 张量解包为每元素 1 字节的 uint8 数组。"""
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    merged = tensor.detach().cpu().numpy().astype(np.uint8)
    merged_shape = merged.shape
    temp_shape = list(merged_shape[:-1]) + [merged_shape[-1], 1]
    temp_array = merged.reshape(temp_shape)
    low_bits = (temp_array & 0x0F).reshape(list(merged_shape[:-1]) + [merged_shape[-1], 1])
    high_bits = ((temp_array >> 4) & 0x0F).reshape(list(merged_shape[:-1]) + [merged_shape[-1], 1])
    unpacked = np.concatenate([low_bits, high_bits], axis=-1)
    original_shape = list(merged_shape[:-1]) + [merged_shape[-1] * 2]
    return unpacked.reshape(original_shape)


def _torch_to_numpy_dtype(torch_dtype: torch.dtype):
    """torch dtype -> numpy dtype。"""
    if torch_dtype == torch.float16:
        return np.float16
    if torch_dtype == torch.bfloat16:
        return bfloat16
    if torch_dtype == torch.float32:
        return np.float32
    if torch_dtype == torch.float64:
        return np.float64
    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def get_input(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    **attrs,
) -> list:
    """可选输入预处理：直接按顺序返回输入张量列表。"""
    return [x1, x2, gamma, beta]


def add_rms_norm_dynamic_mx_quant(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    epsilon: float = 1e-6,
    scale_alg: int = 0,
    round_mode: str = "rint",
    dst_type: int = 40,
    x1_dtype: str = "float16",
    output_rstd: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """AddRmsNormDynamicMxQuant CPU 参考实现。

    Args:
        x1/x2/gamma/beta: 输入张量（cann-bench 会默认提升为 FP64，函数内按 x1_dtype cast 回原始精度）
        epsilon: RMS 稳定系数
        scale_alg: 0=OCP, 1=cuBLAS
        round_mode: 舍入模式
        dst_type: 35/36/40/41
        x1_dtype: 原始计算 dtype，用于恢复 NPU 计算精度
        output_rstd: 算子层面是否输出 rstd；上层 NPU PTA 接口没有 output_rstd 参数，通过将输入 x1/x2 的 requires_grad 置为 True 来对应 output_rstd=true 的训练场景语义

    Returns:
        (y, x, mxscale, rstd):
            y: FP8 按 torch.float8_e4m3fn/float8_e5m2 返回；FP4 按 uint8 bit-pattern 返回
            x: 原始 dtype（x1_dtype）的残差
            mxscale: uint8 bit-pattern
            rstd: float32 的 RMS 倒数标准差；算子层面受 output_rstd 控制；上层 NPU PTA 接口没有 output_rstd 参数，通过输入 x1/x2 的 requires_grad 来对应，output_rstd=false 时返回空张量
    """
    torch_dtype = getattr(torch, x1_dtype)
    np_dtype = _torch_to_numpy_dtype(torch_dtype)

    # 1. 先提升到 float32 计算 add_rms_norm，与 NPU 内部精度路径对齐
    x1_f32 = x1.to(torch.float32)
    x2_f32 = x2.to(torch.float32)
    gamma_f32 = gamma.type(torch.float32)
    beta_f32 = beta.type(torch.float32) if beta is not None else None

    # 2. add_rms_norm 在 float32 中计算
    y_f32, x_out, rstd = add_rms_norm(x1_f32, x2_f32, gamma_f32, beta_f32, epsilon)

    # 3. 为量化将 y cast 到 numpy 原始 dtype
    if torch_dtype == torch.bfloat16:
        y_np = y_f32.detach().cpu().numpy().astype(bfloat16)
    else:
        y_np = y_f32.detach().cpu().numpy().astype(np_dtype)

    # 4. MX 量化
    dst_type_str = DATA_TYPE_INT_TO_STR.get(dst_type, "float4_e2m1")
    mxscale_np, y_quant_np = dynamic_mx_quant(
        y_np,
        mx_ele_dtype=dst_type_str,
        axis=-1,
        block_size=32,
        round_mode=round_mode,
        scale_alg=scale_alg,
    )

    # 5. y 输出：FP8 按 torch float8 返回，FP4 按 uint8 bit-pattern 返回
    if dst_type in (40, 41):  # FP4
        y_out = torch.from_numpy(y_quant_np.view(np.uint8).copy())
    else:  # FP8
        if dst_type_str == "float8_e4m3fn":
            y_torch_dtype = torch.float8_e4m3fn
        elif dst_type_str == "float8_e5m2":
            y_torch_dtype = torch.float8_e5m2
        else:
            raise ValueError(f"Unsupported dst_type: {dst_type}")
        y_bits = y_quant_np.view(np.uint8)
        y_out = torch.from_numpy(y_bits.copy()).view(y_torch_dtype)

    # 6. x 输出为原始 dtype
    x_out = x_out.to(torch_dtype).detach().cpu()

    # 7. mxscale 输出为 uint8 bit-pattern
    mxscale_out = torch.from_numpy(mxscale_np.view(np.uint8).copy())

    # 8. rstd 输出为 float32；output_rstd=False 时返回空张量与 NPU 对齐
    if output_rstd:
        rstd_out = rstd.to(torch.float32).detach().cpu()
    else:
        rstd_out = torch.empty(0, dtype=torch.float32)

    return y_out, x_out, mxscale_out, rstd_out
