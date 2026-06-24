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
import numpy as np

"""
AscendAntiQuantV2 算子 Torch Golden 参考实现

根据输入 scale 和可选 offset 对量化张量 x 进行反量化：
  - sqrt_mode=False: y = (x + offset) * scale      (offset 为 None 时省略加法)
  - sqrt_mode=True:  y = (x + offset) * scale^2    (offset 为 None 时省略加法)

当前 torch_npu.npu_anti_quant Python 接口未暴露 sqrt_mode 参数，
因此 cann-bench 用例中仅测试 sqrt_mode=False。

参考来源: /home/g00944280/canbranch/golden/AscendAntiQuantV2/aclnnAscendAntiQuantV2.py
"""


def _unpack_int4(x_int32: torch.Tensor) -> torch.Tensor:
    """将 INT32 打包的 INT4 数据解包为 INT8 张量（每 int32 含 8 个 int4）。"""
    x_np = x_int32.cpu().numpy().astype(np.int32)
    bytes_data = x_np.tobytes()
    uint8_data = np.frombuffer(bytes_data, dtype=np.uint8)
    # 每个 uint8 低 4 位和高 4 位各代表一个 int4
    shift = np.array([0, 4], dtype=np.uint8)
    int4_values = ((uint8_data.reshape(-1, 1) >> shift) & 0b00001111).astype(np.int8)
    # 将无符号 4-bit [0,15] 转为有符号 [-8,7]
    int4_values = np.where(int4_values > 7, int4_values - 16, int4_values).astype(np.int8)
    new_shape = tuple(x_np.shape[:-1]) + (x_np.shape[-1] * 8,)
    return torch.from_numpy(int4_values.reshape(new_shape))


def _pack_int4(x_int32: torch.Tensor) -> torch.Tensor:
    """将 INT32 概念张量（每 8 个 int4 值）打包成 INT32（每 int32 含 8 个 int4）。"""
    x_np = x_int32.to(torch.int64).cpu().numpy()
    # 映射到有符号 4-bit 范围后再取低 4 位
    int4_vals = (x_np & 0x0F).astype(np.uint8)
    # 按低 nibble 在前的方式每 8 个打包成一个 uint32
    flat = int4_vals.reshape(-1, 8)
    packed = np.zeros((flat.shape[0],), dtype=np.uint32)
    for i in range(8):
        packed |= (flat[:, i].astype(np.uint32) << (4 * i))
    new_shape = tuple(x_np.shape[:-1]) + (x_np.shape[-1] // 8,)
    return torch.from_numpy(packed.reshape(new_shape)).to(torch.int32)


def get_input(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor = None,
    dst_type: int = 1,
    sqrt_mode: bool = False,
    **kwargs,
) -> list:
    """输入预处理：打包 INT4 数据并约束 scale/offset 合法性。

    kernel_eval 在生成输入后会调用此函数，返回值会同时替换 golden 和 candidate
    的输入，保证对比公平。
    """
    if x.dtype == torch.int32:
        x = _pack_int4(x)

    # scale 必须为正数
    scale = scale.abs()
    if scale.numel() > 0:
        scale = scale.clamp_min(0.001)

    return [x, scale, offset]


def ascend_anti_quant_v2(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor = None,
    dst_type: int = 1,
    sqrt_mode: bool = False,
) -> torch.Tensor:
    """
    AscendAntiQuantV2 CPU golden 实现。

    Args:
        x: 量化输入张量，支持 int8 或 int32（int32 视为 int4 打包）
        scale: 一维 scale 张量
        offset: 可选一维 offset 张量
        dst_type: 输出类型，1=FLOAT16, 27=BFLOAT16
        sqrt_mode: scale 是否平方；当前 torch_npu Python 接口未暴露，保留兼容

    Returns:
        y: 反量化后的 float16/bfloat16 张量
    """
    if x.dtype == torch.int32:
        x = _unpack_int4(x)

    x_f32 = x.to(torch.float32)
    scale_f32 = scale.to(torch.float32)

    if offset is not None:
        offset_f32 = offset.to(torch.float32)
        x_f32 = x_f32 + offset_f32

    res = x_f32 * scale_f32
    if sqrt_mode:
        res = res * scale_f32

    if dst_type == 27:
        return res.to(torch.bfloat16)
    return res.to(torch.float16)
