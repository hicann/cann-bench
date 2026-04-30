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
from typing import List, Optional, Tuple, Union

"""
GRU 算子 Torch Golden 参考实现

对标 PyTorch torch.nn.GRU 接口，使用 TensorList 格式传递权重。
每层每方向的权重是独立的 tensor：
  - weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0 (forward)
  - weight_ih_l0_reverse, weight_hh_l0_reverse, ... (reverse, if bidirectional)
  - weight_ih_l1, ... (layer 1, if numLayers > 1)
"""

def gru(
    x: torch.Tensor,
    weight_ih: List[torch.Tensor],
    weight_hh: List[torch.Tensor],
    bias_ih: Optional[List[torch.Tensor]] = None,
    bias_hh: Optional[List[torch.Tensor]] = None,
    h0: Optional[torch.Tensor] = None,
    inputSize: int = 0,
    hiddenSize: int = 0,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GRU 前向计算（对标 PyTorch torch.nn.GRU）

    Args:
        x: 输入序列 (S, B, inputSize) 或 (B, S, inputSize) if batch_first
        weight_ih: TensorList，每层每方向一个 [3*hiddenSize, input_dim] tensor
        weight_hh: TensorList，每层每方向一个 [3*hiddenSize, hiddenSize] tensor
        bias_ih: TensorList?, 每层每方向一个 [3*hiddenSize] tensor
        bias_hh: TensorList?, 每层每方向一个 [3*hiddenSize] tensor
        h0: 初始隐藏状态 [numLayers*num_directions, B, hiddenSize]
        inputSize: 输入特征维度
        hiddenSize: 隐藏状态维度
        numLayers: 层数
        bias: 是否使用偏置
        batchFirst: 输入格式是否为 (batch, seq, feature)
        dropout: 层间 dropout
        bidirectional: 是否双向

    Returns:
        y: 输出序列
        hn: 最终隐藏状态
    """
    num_directions = 2 if bidirectional else 1
    gate_size = 3 * hiddenSize
    num_weights = numLayers * num_directions

    input_dtype = x.dtype
    target_device = x.device

    # 创建 GRU 模块（不移动到特定设备，因为 GRU 会自动跟随输入张量的设备）
    # 注意：当 golden 在 CPU fp64 运行时，GRU 保持在 CPU；
    # 当作为 AI 算子在 NPU 运行时，输入张量在 NPU，GRU 会自动在 NPU 上计算
    gru_layer = torch.nn.GRU(
        input_size=inputSize,
        hidden_size=hiddenSize,
        num_layers=numLayers,
        bias=False,  # 先创建无 bias 版本，后面根据需要添加
        batch_first=batchFirst,
        dropout=dropout if numLayers > 1 else 0.0,
        bidirectional=bidirectional
    ).to(input_dtype)  # 仅转换为正确的 dtype

    # 设置为 eval 模式，禁用 dropout（确保 CPU/NPU 输出一致）
    gru_layer.eval()

    # 计算每层的输入维度
    layer_inputs = [inputSize]
    for layer in range(1, numLayers):
        layer_inputs.append(hiddenSize * num_directions)

    # 设置权重参数（TensorList 格式）
    for layer in range(numLayers):
        layer_input = layer_inputs[layer]
        for d in range(num_directions):
            idx = layer * num_directions + d
            suffix = f"l{layer}" if d == 0 else f"l{layer}_reverse"

            # 从 TensorList 中取对应 tensor（保持原始 dtype）
            wi_data = weight_ih[idx][:gate_size, :layer_input]
            wh_data = weight_hh[idx][:gate_size, :hiddenSize]

            wi_param = torch.nn.Parameter(wi_data.to(target_device))
            wh_param = torch.nn.Parameter(wh_data.to(target_device))

            setattr(gru_layer, f'weight_ih_{suffix}', wi_param)
            setattr(gru_layer, f'weight_hh_{suffix}', wh_param)

            if bias and bias_ih is not None and bias_hh is not None:
                bi_data = bias_ih[idx][:gate_size]
                bh_data = bias_hh[idx][:gate_size]
                bi_param = torch.nn.Parameter(bi_data.to(target_device))
                bh_param = torch.nn.Parameter(bh_data.to(target_device))
                setattr(gru_layer, f'bias_ih_{suffix}', bi_param)
                setattr(gru_layer, f'bias_hh_{suffix}', bh_param)
            elif bias:
                # 有 bias 要求但没传入偏置，创建零偏置（使用输入 dtype）
                bi_param = torch.nn.Parameter(
                    torch.zeros(gate_size, dtype=input_dtype, device=target_device)
                )
                bh_param = torch.nn.Parameter(
                    torch.zeros(gate_size, dtype=input_dtype, device=target_device)
                )
                setattr(gru_layer, f'bias_ih_{suffix}', bi_param)
                setattr(gru_layer, f'bias_hh_{suffix}', bh_param)

    # 保持原始 dtype 进行计算（NPU 不支持 float32 GRU）
    if h0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        h0 = torch.zeros(numLayers * num_directions, batch_size, hiddenSize,
                         dtype=input_dtype, device=target_device)
    else:
        h0 = h0.to(input_dtype).to(target_device)

    y, hn = gru_layer(x, h0)

    return y, hn