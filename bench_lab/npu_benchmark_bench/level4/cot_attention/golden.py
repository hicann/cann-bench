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

"""
CoTAttention 算子 Torch Golden 参考实现

Contextual Transformer (CoT) 注意力前向：key 分支用分组卷积提取 KxK 邻域上下文，
value 分支做 1x1 投影；两者与输入拼接后经两层 1x1 卷积生成 KxK 邻域注意力，
对 value 加权求和后与 key 分支残差相加。
公式:
    k1  = ReLU(DepthWiseConv_{K}(x))
    v   = Conv1x1(x)
    att = Conv1x1(ReLU(Conv1x1(cat([k1, x]))))
    att = mean_{K x K}(reshape(att, [B, C, K*K, H, W]))
    k2  = softmax_{H*W}(att) * v
    y   = k1 + k2
"""


def cot_attention(
    x: torch.Tensor,
    key_embed_weight: torch.Tensor,
    key_embed_bias: torch.Tensor,
    value_embed_weight: torch.Tensor,
    value_embed_bias: torch.Tensor,
    att_w1_weight: torch.Tensor,
    att_b1_bias: torch.Tensor,
    att_w2_weight: torch.Tensor,
    att_b2_bias: torch.Tensor,
    kernel_size: int = 3,
) -> torch.Tensor:
    """
    CoT 注意力模块前向

    公式:
        k1  = ReLU(DepthWiseConv_{K}(x))
        v   = Conv1x1(x)
        att = Conv1x1(ReLU(Conv1x1(cat([k1, x]))))
        att = mean_{K x K}(reshape(att, [B, C, K*K, H, W]))
        k2  = softmax_{H*W}(att) * v
        y   = k1 + k2

    Args:
        x: 输入特征图 [B, C, H, W]
        key_embed_weight: key 分支分组卷积权重 [C, C // groups, K, K]
        key_embed_bias: key 分支分组卷积偏置 [C]
        value_embed_weight: value 分支 1x1 卷积权重 [C, C, 1, 1]
        value_embed_bias: value 分支 1x1 卷积偏置 [C]
        att_w1_weight: 注意力第一层 1x1 卷积权重 [mid, 2C, 1, 1]
        att_b1_bias: 注意力第一层 1x1 卷积偏置 [mid]
        att_w2_weight: 注意力第二层 1x1 卷积权重 [K*K*C, mid, 1, 1]
        att_b2_bias: 注意力第二层 1x1 卷积偏置 [K*K*C]
        kernel_size: key 分支分组卷积核大小 K，padding = K // 2

    Returns:
        CoT 注意力输出 [B, C, H, W]
    """
    b, c, h, w = x.shape
    n = h * w
    k = int(kernel_size)
    padding = k // 2
    groups = 4 if c % 4 == 0 else (2 if c % 2 == 0 else 1)

    # 统一升 fp32 计算，避免 fp16/bf16 在长归约中溢出或精度损失
    x_f = x.float()
    k1 = F.conv2d(
        x_f, key_embed_weight.float(), key_embed_bias.float(), padding=padding, groups=groups
    )
    k1 = F.relu(k1)
    v = F.conv2d(x_f, value_embed_weight.float(), value_embed_bias.float()).view(b, c, n)

    y = torch.cat([k1, x_f], dim=1)
    att = F.conv2d(y, att_w1_weight.float(), att_b1_bias.float())
    att = F.relu(att)
    att = F.conv2d(att, att_w2_weight.float(), att_b2_bias.float())
    att = att.reshape(b, c, k * k, h, w).mean(dim=2).reshape(b, c, n)

    k2 = torch.softmax(att, dim=-1) * v
    k2 = k2.view(b, c, h, w)
    return (k1 + k2).to(dtype=x.dtype)
