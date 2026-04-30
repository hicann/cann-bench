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
LSTM 算子 Torch Golden 参考实现

对标 PyTorch torch.nn.LSTM 接口，使用 TensorList 格式传递权重。
每层每方向的权重是独立的 tensor：
  - weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0 (forward)
  - weight_ih_l0_reverse, weight_hh_l0_reverse, ... (reverse, if bidirectional)
  - weight_ih_l1, ... (layer 1, if numLayers > 1)
"""

def lstm(
    x: torch.Tensor,
    weight_ih: List[torch.Tensor],
    weight_hh: List[torch.Tensor],
    bias_ih: Optional[List[torch.Tensor]] = None,
    bias_hh: Optional[List[torch.Tensor]] = None,
    h0: Optional[torch.Tensor] = None,
    c0: Optional[torch.Tensor] = None,
    inputSize: int = 0,
    hiddenSize: int = 0,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False,
    projSize: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LSTM 前向计算（对标 PyTorch torch.nn.LSTM）

    Args:
        x: 输入序列 (S, B, inputSize) 或 (B, S, inputSize) if batch_first
        weight_ih: TensorList，每层每方向一个 [4*hiddenSize, input_dim] tensor
        weight_hh: TensorList，每层每方向一个 [4*hiddenSize, hiddenSize or projSize] tensor
        bias_ih: TensorList?, 每层每方向一个 [4*hiddenSize] tensor
        bias_hh: TensorList?, 每层每方向一个 [4*hiddenSize] tensor
        h0: 初始隐藏状态 [numLayers*num_directions, B, hiddenSize or projSize]
        c0: 初始细胞状态 [numLayers*num_directions, B, hiddenSize]
        inputSize: 输入特征维度
        hiddenSize: 隐藏状态维度
        numLayers: 层数
        bias: 是否使用偏置
        batchFirst: 输入格式是否为 (batch, seq, feature)
        dropout: 层间 dropout
        bidirectional: 是否双向
        projSize: 投影维度 (>0 时启用 LSTM with Projection)

    Returns:
        y: 输出序列
        hn: 最终隐藏状态
        cn: 最终细胞状态
    """
    num_directions = 2 if bidirectional else 1
    gate_size = 4 * hiddenSize  # LSTM: i, f, g, o
    effective_hidden = projSize if projSize > 0 else hiddenSize

    lstm_layer = torch.nn.LSTM(
        input_size=inputSize,
        hidden_size=hiddenSize,
        num_layers=numLayers,
        bias=bias,
        batch_first=batchFirst,
        dropout=dropout if numLayers > 1 else 0.0,
        bidirectional=bidirectional,
        proj_size=projSize if projSize > 0 else 0
    )

    input_dtype = x.dtype
    lstm_layer = lstm_layer.float().to(x.device)

    # 计算每层的输入维度
    layer_inputs = [inputSize]
    for layer in range(1, numLayers):
        if projSize > 0:
            layer_inputs.append(projSize * num_directions)
        else:
            layer_inputs.append(hiddenSize * num_directions)

    # 设置权重参数（TensorList 格式）
    with torch.no_grad():
        for layer in range(numLayers):
            layer_input = layer_inputs[layer]
            for d in range(num_directions):
                idx = layer * num_directions + d
                suffix = f"l{layer}" if d == 0 else f"l{layer}_reverse"

                # 从 TensorList 中取对应 tensor
                wi = weight_ih[idx][:gate_size, :layer_input]
                wh = weight_hh[idx][:gate_size, :effective_hidden]

                getattr(lstm_layer, f'weight_ih_{suffix}').copy_(wi.float())
                getattr(lstm_layer, f'weight_hh_{suffix}').copy_(wh.float())

                if bias and bias_ih is not None and bias_hh is not None:
                    bi = bias_ih[idx][:gate_size]
                    bh = bias_hh[idx][:gate_size]
                    getattr(lstm_layer, f'bias_ih_{suffix}').copy_(bi.float())
                    getattr(lstm_layer, f'bias_hh_{suffix}').copy_(bh.float())

    x_float = x.float()
    if h0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        h0 = torch.zeros(numLayers * num_directions, batch_size, effective_hidden,
                         dtype=torch.float32, device=x.device)
    else:
        h0 = h0.float()

    if c0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        c0 = torch.zeros(numLayers * num_directions, batch_size, hiddenSize,
                         dtype=torch.float32, device=x.device)
    else:
        c0 = c0.float()

    y, (hn, cn) = lstm_layer(x_float, (h0, c0))
    y = y.to(input_dtype)
    hn = hn.to(input_dtype)
    cn = cn.to(input_dtype)

    return y, hn, cn