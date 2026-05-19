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

"""
DynamicQuant 算子 Torch Golden 参考实现

per-token 对称动态量化 (沿 last-dim)，对齐 NPU torch_npu.npu_dynamic_quant 默认行为：
- 只支持沿 last-dim 量化，不暴露 axis
- 只支持 fp16 / bf16 输入；输出 int8 + per-token scale (fp32)
"""
from typing import Tuple


def dynamic_quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token 对称动态量化 (axis=-1, dtype_max=127→int8)。

    NPU API ``torch_npu.npu_dynamic_quant`` 返回 (y, scale) 双输出 —— scale
    是每 token 的反量化系数，下游算子（FFN W8A8 / KV cache 反量化）必须
    拿到 scale 才能还原浮点结果，所以 scale 是算子的本质输出之一。

    公式:
        scaleOut = row_max(abs(x)) / 127         # shape = x.shape[:-1]
        yOut     = round(x / scaleOut)            # shape = x.shape, int8

    Args:
        x: 输入张量 (fp16 / bf16)，shape ≥ 2 维

    Returns:
        y:     量化后张量 (int8, shape 与 x 一致)
        scale: per-token 反量化系数 (float32, shape = x.shape[:-1])
    """
    x_compute = x.to(torch.float32)
    abs_max = torch.max(torch.abs(x_compute), dim=-1, keepdim=True)[0]
    # clamp(min=1e-12) 防止全零输入触发 0/0 = NaN
    scale_out = abs_max.clamp(min=1e-12) / 127.0
    # clamp(-128, 127) 避免 |x/scale| > 127 时 int8 模 256 截断（罕见但理论可触发，
    # 例如下游 fuzz/数值边界场景）
    y = torch.clamp(torch.round(x_compute / scale_out), -128, 127).to(torch.int8)
    # scale 报给下游：去掉 keepdim 的最后一维，dtype 固定 float32
    scale = scale_out.squeeze(-1).to(torch.float32)
    return y, scale
