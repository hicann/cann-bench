# #!/usr/bin/python3
# # coding=utf-8

# # ----------------------------------------------------------------------------------------------------------
# # Copyright (c) 2026 Huawei Technologies Co., Ltd.
# # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# # CANN Open Software License Agreement Version 2.0 (the "License").
# # Please refer to the License for details. You may not use this file except in compliance with the License.
# # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# # See LICENSE in the root of the software repository for the full text of the License.
# # ----------------------------------------------------------------------------------------------------------

import torch
from typing import Optional
import sys


def get_input(x_norm: torch.Tensor, grad_y: torch.Tensor, expert_idx: torch.Tensor, **kwargs):
    """对 expert_idx 做行内去重，避免 scatter_ 写入冲突导致精度不确定。"""
    M, K = expert_idx.shape
    N = x_norm.shape[1]
    # 每行从 [0, N) 中无放回采样 K 个索引
    rand_matrix = torch.rand(M, N)
    unique_idx = torch.argsort(rand_matrix, dim=1)[:, :K].to(torch.int32)
    return (x_norm, grad_y, unique_idx)


def moe_gating_top_k_backward(x_norm: torch.Tensor, grad_y: torch.Tensor, expert_idx: torch.Tensor,
                               renorm: int = 0, norm_type: int = 1,
                               routed_scaling_factor: float = 1.0, eps: float = 1e-20
):
    """
    MoE Gating Top-K 反向传播（普通函数实现），当前仅sigmoid模式

    参数:
        x_norm: 前向归一化得分 [M, N], float32 (sigmoid输出)
        grad_y: 上游梯度 [M, K], 数据类型支持float16、bfloat16、float32
        expert_idx: 前向选中的专家索引 [M, K], int32
        renorm: 未使用，保留参数
        norm_type: 归一化方式 (1-sigmoid, 否则-softmax)，当前仅支持1
        routed_scaling_factor: 最终权重的缩放因子, float32
        eps: 防止除零的小常数, float32

    返回:
        grad_x: [M, N] 输入得分矩阵的梯度, 数据类型与grad_y一致
    """
    # 转换为float32进行计算
    grad_y_fp32 = grad_y.double() if grad_y.dtype != torch.float32 else grad_y.clone()
    x_norm_fp32 = x_norm.double() if x_norm.dtype != torch.float32 else x_norm.clone()
    grad_y_scaled = grad_y_fp32 * routed_scaling_factor  # [M, K]

    # 步骤2: Gather前向归一化分数 (w')
    w_prime = x_norm_fp32.gather(1, expert_idx.long())  # [M, K]
    if norm_type == 1:
        D = w_prime.sum(dim=-1, keepdim=True) + eps  # [M, 1]

        inv_D = 1.0 / D  # [M, 1]

        w = w_prime * inv_D  # [M, K]
        beta = (w * grad_y_scaled).sum(dim=-1, keepdim=True)  # [M, 1]
        grad_w_prime = (grad_y_scaled - beta) * inv_D  # [M, K]
    else:
        # Softmax模式（预留）
        grad_w_prime = grad_y_scaled

    # 步骤3: Scatter回完整维度 [M, N]
    grad_norm_x = torch.zeros_like(x_norm_fp32)
    grad_norm_x.scatter_(1, expert_idx.long(), grad_w_prime)


    # 步骤4: Sigmoid反向: grad_x = x_norm * (1 - x_norm) * grad_norm_x
    if norm_type == 1:
        grad_x = x_norm_fp32 * (1.0 - x_norm_fp32) * grad_norm_x
    else:
        # Softmax反向（预留）
        grad_x = grad_norm_x

    # 转回原始数据类型
    return grad_x.to(grad_y.dtype)
    