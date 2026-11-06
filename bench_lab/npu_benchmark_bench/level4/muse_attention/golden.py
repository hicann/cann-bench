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

import math

import torch
import torch.nn.functional as F

"""
MUSE Attention 算子 Torch Golden 参考实现

MUSE (Multi-Scale) Attention 将标准多头缩放点积自注意力与多尺度卷积分支并联：
卷积分支对投影后的 value 依次做 depthwise + pointwise 一维卷积（核大小 1/3/5），
三路输出由可学习的 softmax 动态参数加权求和，再经自适应平均池化对齐到 query 长度，
最后与自注意力输出逐元素相加。

源实现全程保持输入 dtype（不做 fp32 升精度）。
"""


def muse_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    fc_q_weight: torch.Tensor,
    fc_q_bias: torch.Tensor,
    fc_k_weight: torch.Tensor,
    fc_k_bias: torch.Tensor,
    fc_v_weight: torch.Tensor,
    fc_v_bias: torch.Tensor,
    fc_o_weight: torch.Tensor,
    fc_o_bias: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias: torch.Tensor,
    conv3_depth_weight: torch.Tensor,
    conv3_depth_bias: torch.Tensor,
    conv3_point_weight: torch.Tensor,
    conv3_point_bias: torch.Tensor,
    conv5_depth_weight: torch.Tensor,
    conv5_depth_bias: torch.Tensor,
    conv5_point_weight: torch.Tensor,
    conv5_point_bias: torch.Tensor,
    dy_paras: torch.Tensor,
    d_model: int,
    d_k: int,
    d_v: int,
    h: int,
) -> torch.Tensor:
    """
    MUSE 多尺度注意力前向

    Args:
        queries: 查询 [B, nq, d_model]
        keys: 键 [B, nk, d_model]
        values: 值 [B, nk, d_model]
        fc_q_weight: Q 投影权重 [h*d_k, d_model]
        fc_q_bias: Q 投影偏置 [h*d_k]
        fc_k_weight: K 投影权重 [h*d_k, d_model]
        fc_k_bias: K 投影偏置 [h*d_k]
        fc_v_weight: V 投影权重 [h*d_v, d_model]
        fc_v_bias: V 投影偏置 [h*d_v]
        fc_o_weight: 输出投影权重 [d_model, h*d_v]
        fc_o_bias: 输出投影偏置 [d_model]
        conv1_weight: 1 核 pointwise 卷积权重 [d_model, h*d_v, 1]
        conv1_bias: 1 核 pointwise 卷积偏置 [d_model]
        conv3_depth_weight: 3 核 depthwise 卷积权重 [h*d_v, 1, 3]（groups=h*d_v）
        conv3_depth_bias: 3 核 depthwise 卷积偏置 [h*d_v]
        conv3_point_weight: 3 核 pointwise 卷积权重 [d_model, h*d_v, 1]
        conv3_point_bias: 3 核 pointwise 卷积偏置 [d_model]
        conv5_depth_weight: 5 核 depthwise 卷积权重 [h*d_v, 1, 5]（groups=h*d_v）
        conv5_depth_bias: 5 核 depthwise 卷积偏置 [h*d_v]
        conv5_point_weight: 5 核 pointwise 卷积权重 [d_model, h*d_v, 1]
        conv5_point_bias: 5 核 pointwise 卷积偏置 [d_model]
        dy_paras: 多尺度动态参数 [3]，源实现初始化为全 1
        d_model: 模型维度
        d_k: 每个 head 的 query/key 维度
        d_v: 每个 head 的 value 维度
        h: head 数

    Returns:
        输出 [B, nq, d_model]
    """
    b_s, nq = queries.shape[:2]
    nk = keys.shape[1]
    in_ch = h * d_v

    # ---------- 自注意力分支 ----------
    q = F.linear(queries, fc_q_weight, fc_q_bias).view(b_s, nq, h, d_k).permute(0, 2, 1, 3)
    k = F.linear(keys, fc_k_weight, fc_k_bias).view(b_s, nk, h, d_k).permute(0, 2, 3, 1)
    v = F.linear(values, fc_v_weight, fc_v_bias).view(b_s, nk, h, d_v).permute(0, 2, 1, 3)

    att = torch.matmul(q, k) / math.sqrt(d_k)
    att = F.softmax(att, dim=-1)

    out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, h * d_v)
    out = F.linear(out, fc_o_weight, fc_o_bias)

    # ---------- 多尺度卷积分支 ----------
    v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)

    # kernel=1：depthwise 退化为恒等映射，只有 pointwise 卷积
    conv1_out = F.conv1d(v2, conv1_weight, conv1_bias)
    # kernel=3：depthwise(groups=in_ch) + pointwise
    mid3 = F.conv1d(v2, conv3_depth_weight, conv3_depth_bias, padding=1, groups=in_ch)
    conv3_out = F.conv1d(mid3, conv3_point_weight, conv3_point_bias)
    # kernel=5：depthwise(groups=in_ch) + pointwise
    mid5 = F.conv1d(v2, conv5_depth_weight, conv5_depth_bias, padding=2, groups=in_ch)
    conv5_out = F.conv1d(mid5, conv5_point_weight, conv5_point_bias)

    # 动态参数 softmax 加权求和
    w = F.softmax(dy_paras, dim=0)
    out2 = w[0] * conv1_out + w[1] * conv3_out + w[2] * conv5_out

    # 序列长度对齐
    if nk != nq:
        out2 = F.adaptive_avg_pool1d(out2, nq)
    out2 = out2.permute(0, 2, 1)

    return out + out2
