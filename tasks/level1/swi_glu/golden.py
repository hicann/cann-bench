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
import torch.nn.functional as F


def swi_glu(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """标准 SwiGLU 激活的 Torch Golden 参考实现 (P2 op).

    公式: output = silu(x0) * x1 = (x0 * sigmoid(x0)) * x1
    其中 x0, x1 = input.chunk(2, dim=dim).

    标准 SwiGLU (Shazeer 2020 / Llama / PaLM) 固定 Swish 的 beta = 1
    (即等价于 SiLU),没有可调 beta 参数。aclnnSwiGlu / torch_npu.npu_swiglu
    也是同样定义,本 spec 与之对齐。

    Args:
        input: 输入张量,dim 维度上 size 必须是偶数
        dim: 拆分维度,默认 -1

    Returns:
        output: 与 input 同 dtype/shape 但 dim 维度大小减半
    """
    # FP16/BF16 升精度计算,与 ACLNN 内部一致.
    out_dtype = input.dtype
    x = input.to(torch.float)
    x0, x1 = x.chunk(2, dim=dim)
    output = F.silu(x0) * x1
    return output.to(out_dtype)
