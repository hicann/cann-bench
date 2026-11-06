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
Spatial Group Enhance (SGE) 算子 Torch Golden 参考实现

按通道分组，组内用全局平均池化特征与逐空间位置特征做相似度，再对每个样本的
g 个组各自做标准化，经逐组仿射后 sigmoid 得到空间门控，乘回输入。

源实现全程保持输入 dtype（内部不升精度）；标准化使用有偏估计之外的
torch.std 默认行为（unbiased=True），与源实现一致。
"""


def spatial_group_enhance(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    groups: int,
) -> torch.Tensor:
    """
    Spatial Group Enhance 前向

    Args:
        x: 输入特征图 [B, C, H, W]
        weight: 逐组缩放参数 [1, groups, 1, 1]
        bias: 逐组偏移参数 [1, groups, 1, 1]
        groups: 通道分组数，须整除 C

    Returns:
        增强后的特征图 [B, C, H, W]
    """
    b, c, h, w = x.shape

    x_reshaped = x.view(b * groups, -1, h, w)           # (b*g, c//g, h, w)
    avg_pool = F.adaptive_avg_pool2d(x_reshaped, 1)     # (b*g, c//g, 1, 1)
    xn = x_reshaped * avg_pool                          # (b*g, c//g, h, w)
    xn = xn.sum(dim=1, keepdim=True)                    # (b*g, 1, h, w)
    t = xn.view(b * groups, -1)                         # (b*g, h*w)
    t = t - t.mean(dim=1, keepdim=True)
    std = t.std(dim=1, keepdim=True) + 1e-5
    t = t / std
    t = t.view(b, groups, h, w)                         # (b, g, h, w)
    t = t * weight + bias
    t = t.view(b * groups, 1, h, w)                     # (b*g, 1, h, w)
    out = x_reshaped * torch.sigmoid(t)
    return out.view(b, c, h, w)
