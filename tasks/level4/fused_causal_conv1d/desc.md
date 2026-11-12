# FusedCausalConv1d 算子 API 描述

## 1. 算子简介

FusedCausalConv1d 算子对序列执行**因果一维卷积**，并融合 APC（Automatic Prefix Caching）、MTP（投机解码）、残差连接、原地更新、可选 silu 激活等特性。相较于标准 causal_conv1d，本算子新增了 APC 缓存复用、PD 混部、残差连接可选等功能。

**主要应用场景**：
- 大模型推理中的因果一维卷积（如 Mamba / State Space Model）
- 变长序列 / 投机解码 / APC 缓存场景下的融合卷积计算
- Transformer 推理加速中的融合卷积模块

**算子特征**：
- 难度等级：L4（Convolution）
- 10 个张量输入（含可选输入），2 个输出，8 个属性参数
- 支持 ND 格式输入
- 支持原地更新 `x`（`inplace=True`）

## 2. 算子定义

### 数学公式

设 $K$ 为卷积核宽度，$L$ 为当前 batch 的序列长度，$dim$ 为特征维度，$batchId$ 为当前处理的变长序列索引，$C$ 为 `conv_states` 的缓存空间长度（`conv_states.size(1)`），$B$ 为 APC 块大小（`block_size`）。对输入 $x$ 中的每一个 batch 执行如下操作：

**1）缓存读取**

缓存行索引：

$$
readCacheLine = \begin{cases}
cache\_indices[batchId, \; initial\_state\_idx[batchId]], & \text{APC 模式} \\
cache\_indices[batchId], & \text{非 APC 且 cache\_indices 存在} \\
batchId, & \text{其他}
\end{cases}
$$

- 首次计算（`num_computed_tokens[batchId] == 0`）：
  $$
  cachedState[i, dim] = 0, \quad 0 \leq i < K-1
  $$
  $$
  offset = 0
  $$

- 投机解码模式（`num_accepted_tokens` 存在）：
  $$
  offset = numAcceptedTokens[batchId] - 1
  $$
  $$
  cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
  $$

- 默认模式：
  $$
  offset = C - (K - 1)
  $$
  $$
  cachedState[i, dim] = convStates[readCacheLine][i, dim], \quad 0 \leq i < offset + K - 1
  $$

**2）缓存拼接**

$$
paddedInput[i, dim] =
\begin{cases}
cachedState[i, dim], & 0 \leq i < offset + K - 1 \\
x[i - (offset + K - 1), dim], & offset + K - 1 \leq i < offset + K - 1 + L
\end{cases}
$$

**3）缓存更新**

$$
Len = offset + K - 1 + L
$$
$$
M = \min(C, \; Len)
$$

$$
writeCacheLine = \begin{cases}
cache\_indices[batchId, \; idxLast], & \text{APC 模式} \\
cache\_indices[batchId], & \text{非 APC 且 cache\_indices 存在} \\
batchId, & \text{其他}
\end{cases}
$$

$$
convStates[writeCacheLine][C - M + i, dim] = paddedInput[Len - M + i, dim], \quad i = 0, 1, \dots, M-1
$$

**4）Offset 裁剪**

$$
x'[i, dim] = paddedInput[i + offset, dim], \quad 0 \leq i < K - 1 + L
$$

**5）APC 缓存填充（APC 模式下）**

$$
seqCompletedOffsetToken = numComputedTokens[batchId] \mod B
$$
$$
seqCompletedOffset = B - seqCompletedOffsetToken
$$
$$
seqEndOffset = (L - seqCompletedOffset) \mod B
$$

$$
lastFullBlockTokenIndex = \begin{cases}
L - seqEndOffset - B, & seqEndOffset = 0 \\
L - seqEndOffset, & \text{otherwise}
\end{cases}
$$

$$
nBlockToFill = idxLast - idxFirst
$$

对每个 $chunk = 0, 1, \dots, nBlockToFill - 1$：

$$
boundaryIdx = lastFullBlockTokenIndex - (nBlockToFill - chunk - 1) \times B
$$

$$
convStates[cacheIndices[batchId, \; idxFirst + chunk]][C-(K-1)+j, \; dim] = x'[boundaryIdx + j, \; dim], \quad j = 0, \dots, K-2
$$

**6）因果 1 维卷积**

$$
y[i, dim] = \sum_{k=0}^{K-1} w[k, dim] \cdot x'[i + k, dim], \quad i = 0, 1, \dots, L-1
$$

**7）零填充重置（`conv_mode == 1` 且 `num_computed_tokens` 不为空）**

$$
resetIdx = \min\!\Big(\max\!\big(K - 1 - numComputedTokens[batchId], \; 0\big), \; L\Big)
$$

$$
y[i, dim] = 0, \quad 0 \leq i < resetIdx
$$

**8）残差连接（可选）**

$$
y[i, dim] = x[i, dim] + y[i, dim]
$$

**9）激活函数（仅支持 silu，可选）**

$$
y[i, dim] = silu(y[i, dim])
$$

## 3. 接口规范

### 算子原型

```python
cann_bench.fused_causal_conv1d(
    Tensor x,
    Tensor weight,
    Tensor conv_states,
    Tensor query_start_loc,
    Tensor cache_indices,
    Tensor num_accepted_tokens,
    Tensor num_computed_tokens,
    Tensor block_idx_first_scheduled_token,
    Tensor block_idx_last_scheduled_token,
    Tensor initial_state_idx,
    int64 pad_slot_id=-1,
    int64 max_query_len=-1,
    int64 residual_connection=1,
    int64 block_size=128,
    int64 conv_mode=1,
    bool inplace=False,
    string activation="none",
    int64 run_mode=0
) -> (Tensor y, Tensor conv_states)
```

### 输入参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| x | 输入/输出 | 输入序列；`inplace=True` 时原地更新 | 2D `[cuSeqLen, dim]` 或 3D `[batch, seqLen, dim]`；`dim` 为 16 的倍数 | BFLOAT16/FLOAT16 | ND | 2D/3D | 支持（dim 维 stride=1） |
| weight | 输入 | 因果 1D 卷积核 | `K ∈ {3,4,5,6}`，不支持空 Tensor | 同 x | ND | `[K, dim]` | 支持 |
| conv_states | 输入/输出 | 历史 token 缓存，原地更新 | 不支持空 Tensor；`stateLen = K-1+m` | 同 x | ND | `[..., stateLen, dim]` | 支持（dim 维 stride=1） |
| query_start_loc | 可选输入 | 2D 场景下各序列起始偏移 | 2D 时不可省略；`query_start_loc[0]=0`，末项为 `cuSeqLen`；支持尾部空 batch | INT32 | ND | `[batch+1]`，batch ∈ [1, 4096] | 支持 |
| cache_indices | 可选输入 | 缓存 slot 索引 | 1D 表示非 APC；2D 表示开启 APC；APC 开启时不可省略 | INT32 | ND | `[batch]` 或 `[batch, maxNumBlocks]` | 支持 |
| initial_state_mode | 可选输入 | 每个序列的 padding 策略 | **暂不支持此字段** | INT32 | ND | `[batch]` | 支持 |
| bias | 可选输入 | 卷积偏置 | **暂不支持此字段** | 同 x | ND | `[dim]` | 支持 |
| num_accepted_tokens | 可选输入 | 投机解码每 batch 接受 token 数 | 值范围 `[1, seqlen]`；为空表示非投机模式 | INT32 | ND | `[batch]` | 支持 |
| num_computed_tokens | 可选输入 | 当前 batch 已处理 token 总数 | Pangu V2 首 token 时不能为空 | INT32 | ND | `[batch]` | 支持 |
| block_idx_first_scheduled_token | 可选输入 | APC 第一个调度 block 索引 | APC 开启时不能为空 | INT32 | ND | `[batch]` | 支持 |
| block_idx_last_scheduled_token | 可选输入 | APC 最后一个调度 block 索引 | APC 开启时不能为空 | INT32 | ND | `[batch]` | 支持 |
| initial_state_idx | 可选输入 | APC 初始 state 读取索引 | APC 开启时不能为空 | INT32 | ND | `[batch]` | 支持 |
| activation | 输入 | 激活函数类型 | 支持 `"none"` / `"silu"` / `"swish"` | STR | - | - | - |
| pad_slot_id | 输入 | 无效 slot ID，跳过对应 batch | 仅支持不参与计算的序列在 `x` 的开头或结尾 | INT64 | - | - | - |
| run_mode | 输入 | 0=prefill，1=decode | 历史遗留字段，暂不支持 | INT64 | - | - | - |
| max_query_len | 输入 | 最大序列长度 | -1 表示自动推导；≥9 视为 prefill，[1,8] 视为 decode | INT64 | - | - | - |
| residual_connection | 输入 | 是否残差连接 | 0=否，1=输出 y 与输入 x 相加 | INT64 | - | - | - |
| block_size | 输入 | APC block 大小 | 典型值 128/256 | INT64 | - | - | - |
| conv_mode | 输入 | 卷积实现模式 | 0=Qwen3-Next 社区版本；1=Pangu V2 | INT64 | - | - | - |
| inplace | 输入 | 是否原地更新 x | `True` 时结果写回 x；`False` 时通过 y 输出 | BOOL | - | - | - |

### 输出

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|--------|----------|------|---------|---------|---------|------------|-------------|
| y | 输出 | 卷积计算结果 | 与输入 `x` shape 一致 | 同 x | ND | 同输入 | 支持 |
| conv_states | 输出 | 更新后的缓存状态 | 与输入 `conv_states` shape 一致 | 同 x | ND | 同输入 | 支持 |

### 数据类型

| x / weight / conv_states dtype | index tensors dtype | y / conv_states dtype |
|-------------------------------|---------------------|-----------------------|
| bfloat16 | int32 | bfloat16 |
| float16 | int32 | float16 |

### 规则与约束

- **确定性计算**
  - 本算子默认为确定性实现，暂不支持非确定性实现。

- **输入值域范围**
  - 算子入参输入值域范围为 `inf`/`-inf`/`nan`，或输入值域/中间计算结果出现超过 float16/bfloat16 数据类型范围时，无法确定输出结果有效性。

- **输入 shape 限制**
  - `x`、`weight`、`conv_states` 不支持空 Tensor。
  - `dim` 必须是 16 的整数倍，范围 `[64, 16384]`。
  - `weight` 的第 0 维（卷积核宽度 $K$）仅支持 `{3, 4, 5, 6}`。
  - `conv_states` 的 `stateLen == K - 1 + m`：
    - prefill 场景下 `m = 0`；
    - decode / PD 混部场景下 `m ∈ [0, 7]`，对应 `num_accepted_tokens` 的值。
  - `cache_indices` 中的元素值不保证升序排列，值不重复（除非等于 `pad_slot_id`），非 pad 元素必须满足 `0 <= index < conv_states.shape[0]`。二维表的非 pad 索引在整个表中互异，不能只保证行内互异；输入构造器使用全局随机排列，并要求状态槽容量不少于表内元素总数。
  - `query_start_loc` 的元素值为升序，取值范围为 `0 ~ cuSeqLen`，首项必须为 0，末项必须等于 `cuSeqLen`。
  - `num_accepted_tokens` 的元素值取值范围为 `1 ~ seqlen`。
  - `x` 和 `conv_states` 在非连续情况下，最后一维（dim 维）的 stride 必须为 1。

- **其他输入参数限制**
  - `block_size` 必须大于等于 2。
  - APC 开启时（`cache_indices` 为 2D），必须提供 `block_idx_first_scheduled_token`、`block_idx_last_scheduled_token` 及 `initial_state_idx`，且对每个 batch $i$：
    - `initial_state_idx[i] <= block_idx_first_scheduled_token[i] + 1`
    - `initial_state_idx[i] <= block_idx_last_scheduled_token[i]`
    - `block_idx_first_scheduled_token[i] <= block_idx_last_scheduled_token[i]`
  - `num_accepted_tokens` 不为 None 时，值需满足：`num_accepted_tokens[i] >= 1`。
  - Pangu V2 模式（`conv_mode = 1`）下，首次运行 `num_computed_tokens` 不能为 None。

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `cuSeqLen`（2D 总 token 数） | 1 ~ 2097152 | 文档约束 |
| `batch`（batch 数） | 1 ~ 4096 | 文档约束 |
| `seqLen`（3D 单 batch 长度） | 1 ~ 2097152 | cases 实测范围 |
| `dim`（特征维度） | 64 ~ 16384 | 必须为 16 的倍数 |
| `K`（卷积核宽度） | 3 / 4 / 5 / 6 | - |
| `stateLen` | K-1 ~ K-1+7 | prefill 取 K-1，decode/PD 取 K-1+m |
| `block_size` | 128 / 256 | 典型值 |
| `conv_mode` | 0 / 1 | 0=Qwen3-Next，1=Pangu V2 |
| `residual_connection` | 0 / 1 | - |
| `inplace` | False / True | - |
| `activation` | "none" / "silu" / "swish" | 实际仅 silu 生效 |
| `max_query_len` | -1 或 1 ~ 1048576 | -1 表示自动推导 |
| `pad_slot_id` | -1 或有效 slot 外值 | 用于跳过无效 batch |

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}\left(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}}\right)
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max\left(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}}\right)
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

```python

# Portions of causal_conv1d_golden were adapted from Ascend/op-plugin.
# Reference source (comparison revision):
# https://github.com/Ascend/op-plugin/blob/3d6726621b6fc15cb1972522d9e3ec0b770bf03f/test/test_custom_ops/test_npu_fused_causal_conv1d.py
# The original adaptation revision was not recorded.
# This project adapts the wrapper, input preparation and reference integration.
# The upstream file has no separate license header; its repository LICENSE follows.
# Upstream license: https://github.com/Ascend/op-plugin/blob/3d6726621b6fc15cb1972522d9e3ec0b770bf03f/LICENSE
#
# rom PyTorch:
#
# Copyright (c) 2020, Huawei Technologies Co., Ltd
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
#    and IDIAP Research Institute nor the names of its contributors may be
#    used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import random
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# cann-bench Golden 入口
# ============================================================


def fused_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    pad_slot_id: int = -1,
    run_mode: int = 0,
    max_query_len: int = -1,
    residual_connection: int = 1,
    block_size: int = 128,
    conv_mode: int = 1,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FusedCausalConv1d 的 Torch Golden 参考实现。

    参数:
        x, weight, conv_states:           主输入张量
        query_start_loc, cache_indices:   变长/APC 辅助张量
        num_accepted_tokens:              投机解码接受 token 数
        num_computed_tokens:              已计算 token 数
        block_idx_first_scheduled_token:  APC 起始 block 索引
        block_idx_last_scheduled_token:   APC 结束 block 索引
        initial_state_idx:                APC 初始 state 索引
        activation:                       激活函数名 (None/"none"/"silu"/"swish")
        pad_slot_id:                      无效 slot ID
        run_mode:                         0=连续 prefill, 1=变长 prefill (golden 中不使用)
        max_query_len:                    最大序列长度
        residual_connection:              是否残差连接 (0/1)
        block_size:                       APC block 大小
        conv_mode:                        0=标准, 1=Pangu v2
        inplace:                          是否原地更新 x

    返回:
        (out, conv_states): 输出张量与更新后的 conv_states
    """
    act = None
    if activation is not None and activation.lower() not in ("none", ""):
        if activation.lower() in ("silu", "swish"):
            act = "silu"
        else:
            raise NotImplementedError(f"activation {activation} not supported")

    return causal_conv1d_golden(
        x=x,
        weight=weight,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        max_query_len=max_query_len,
        pad_slot_id=pad_slot_id,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        B_size=block_size,
        conv_mode=conv_mode,
        inplace=inplace,
        residual=bool(residual_connection),
        activation=act,
    )


# ============================================================
# 前置处理辅助：设置随机种子
# ============================================================


def set_seed(seed: int):
    """
    设置全局随机种子，保证跨机器确定性。

    参数:
        seed: 整数种子（通常为用例 id）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch_npu.npu.manual_seed_all / seed_all 仅在 NPU 环境下调用，
    # 此处保留接口，实际使用时按需启用。


# ============================================================
# 前置处理辅助：生成 query_start_loc
# ============================================================


def generate_query_start_loc(
    batch_size: int,
    total_tokens: int,
    max_query_len: int,
    contain_zero_seqlen_batch_num: int = 0,
) -> torch.Tensor:
    """
    生成满足约束的 query_start_loc，shape=(batch_size+1,), dtype=int32。

    约束：
    - 尾部 contain_zero_seqlen_batch_num 个 batch 的 seqLen 固定为 0
    - 其余 batch 中至少一个 seqLen = max_query_len
    - 所有 batch seqLen 之和 == total_tokens

    参数:
        batch_size:                    batch 总数
        total_tokens:                  总 token 数
        max_query_len:                 各 batch 序列长度上界
        contain_zero_seqlen_batch_num: 尾部固定 seqLen=0 的 batch 数

    返回:
        query_start_loc: torch.Tensor, shape=(batch_size+1,), dtype=int32
    """
    assert batch_size >= 1
    assert max_query_len <= total_tokens
    assert 0 <= contain_zero_seqlen_batch_num < batch_size

    nonzero_slots = batch_size - contain_zero_seqlen_batch_num
    assert nonzero_slots >= 1
    assert total_tokens <= nonzero_slots * max_query_len

    lengths = torch.zeros(batch_size, dtype=torch.int32)
    available_indices = torch.arange(nonzero_slots)

    # Zero-length batches must sit at the tail (assertion below). Choose
    # the max-length batch from the non-tail slots only, and recompute the
    # factual zero count: tokens left after Step 1 may be fewer than the
    # remaining nonzero slots, and those slots stay zero (e.g. total_tokens
    # == max_query_len with contain_zero estimated as 0).
    head_slots = max(nonzero_slots - contain_zero_seqlen_batch_num, 1)
    idx = int(torch.randint(0, head_slots, (1,)).item())
    lengths[idx] = max_query_len
    remaining = total_tokens - max_query_len
    contain_zero_seqlen_batch_num = batch_size - (1 + min(remaining, nonzero_slots - 1))
    # The recomputed zero count may exceed the caller estimate; if the
    # chosen idx now falls inside the (larger) tail-zero region, swap it
    # with the first non-tail slot so zero batches stay at the tail.
    if (
        contain_zero_seqlen_batch_num > 0
        and idx >= batch_size - contain_zero_seqlen_batch_num
    ):
        swap = available_indices[0].item()
        if swap == idx and len(available_indices) > 1:
            swap = available_indices[1].item()
        lengths[swap] = lengths[idx]
        lengths[idx] = 0
        idx = swap

    # Step 2: 其余非零 batch 先各放 1
    for i in available_indices.tolist():
        if i == idx:
            continue
        if remaining > 0:
            lengths[i] = 1
            remaining -= 1

    # Step 3: 随机补剩余 token
    while remaining > 0:
        candidates = torch.where(
            (lengths[:nonzero_slots] < max_query_len)
            & (torch.arange(nonzero_slots) != idx)
        )[0]
        i = candidates[torch.randint(0, len(candidates), (1,))].item()
        max_add = min(max_query_len - lengths[i].item(), remaining)
        add = torch.randint(1, max_add + 1, (1,)).item()
        lengths[i] += add
        remaining -= add

    assert (lengths == 0).sum().item() == contain_zero_seqlen_batch_num
    assert (lengths == max_query_len).sum().item() >= 1
    assert lengths.sum().item() == total_tokens

    if contain_zero_seqlen_batch_num > 0:
        assert torch.all(lengths[-contain_zero_seqlen_batch_num:] == 0)

    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32)
    query_start_loc[1:] = torch.cumsum(lengths, dim=0)
    return query_start_loc


# ============================================================
# 前置处理主函数（对应原 executor L477-488 + L542-726）
# ============================================================


def prepare_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor],
    cache_indices: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    num_computed_tokens: Optional[torch.Tensor],
    block_idx_first_scheduled_token: Optional[torch.Tensor],
    block_idx_last_scheduled_token: Optional[torch.Tensor],
    initial_state_idx: Optional[torch.Tensor],
    pad_slot_id: int,
    run_mode: int,
    max_query_len: int,
    block_size: int,
    case_id: int,
    device: str = "cpu",
) -> dict:
    """
    前置处理：推导元信息、随机约束化各输入张量、移到目标 device。

    对应原 executor init_by_input_data() 中：
    - L477-488: 推导 xdim / batch_size / m
    - L542-726: set_seed + 构造 query_start_loc / cache_indices /
                num_accepted_tokens / num_computed_tokens / APC 索引 /
                pad_slot_id 注入 + 移 device

    参数:
        x, weight, conv_states:      主输入张量（已从 kwargs 读取）
        query_start_loc:             原始 query_start_loc（可为 None）
        cache_indices:               slot 索引（1D 或 2D）
        num_accepted_tokens:         投机解码接受 token 数
        num_computed_tokens:         已计算 token 数（APC/Pangu v2）
        block_idx_first/last_...:   APC block 范围索引
        initial_state_idx:           APC 初始 state 读取索引
        pad_slot_id:                 无效 batch slot ID
        run_mode:                    0=连续 prefill；1=变长 prefill
        max_query_len:               最大序列长度（-1 表示自动推导）
        block_size:                  APC block 大小
        case_id:                     用例 id（用于 set_seed）
        device:                      目标 device 字符串（如 "cpu" 或 "npu:0"）

    返回:
        dict，包含经约束处理并移到 device 的全部输入张量和属性：
        {
          "x", "weight", "conv_states", "query_start_loc",
          "cache_indices", "num_accepted_tokens", "num_computed_tokens",
          "block_idx_first_scheduled_token", "block_idx_last_scheduled_token",
          "initial_state_idx", "pad_slot_id",
          "batch_size", "xdim", "m",
        }
    """
    # ── L477-488: 推导元信息 ──────────────────────────────────────────────
    xdim = x.ndim
    if xdim == 3:
        batch_size = x.shape[0]
    elif query_start_loc is not None:
        batch_size = query_start_loc.shape[0] - 1
    else:
        # query_start_loc 为 None 时，根据 total_tokens 与 max_query_len 推导 batch_size
        total_tokens = x.shape[0]
        if max_query_len is not None and max_query_len > 0:
            batch_size = max(1, (total_tokens + max_query_len - 1) // max_query_len)
        else:
            batch_size = 1
    width = weight.shape[0]
    m = max(0, conv_states.shape[1] - width + 1)

    # ── L542: 设置随机种子 ────────────────────────────────────────────────
    set_seed(case_id)

    # ── L544-546: 保证 pad_slot_id 有效（负数或超出 conv_states 范围）────
    if 0 <= pad_slot_id < conv_states.shape[0]:
        pad_slot_id = conv_states.shape[0] + 10

    # ── L551-603: 构造 query_start_loc 和 seq_lens ───────────────────────
    total_tokens = x.shape[0] if xdim == 2 else x.shape[0] * x.shape[1]

    if xdim == 2:
        if max_query_len is None or max_query_len == -1:
            max_query_len_act = (x.shape[0] + batch_size - 1) // batch_size
        else:
            max_query_len_act = max(
                max_query_len, (total_tokens + batch_size - 1) // batch_size
            )

        if run_mode == 0:
            # 连续 prefill：固定 total_tokens，不规整分配各 batch seqLen
            # Zero-length tail batches are only possible when the remaining
            # batches can still hold every token: keep the capacity
            # invariant total_tokens <= nonzero_slots * max_query_len.
            zero_num = 0
            if batch_size > 5 and (batch_size - 2) * max_query_len_act >= total_tokens:
                zero_num = 2
            query_start_loc = generate_query_start_loc(
                batch_size=batch_size,
                total_tokens=total_tokens,
                max_query_len=max_query_len_act,
                contain_zero_seqlen_batch_num=zero_num,
            ).to(torch.int32)
            seq_lens = torch.diff(query_start_loc)
        else:
            # 变长 prefill：随机 seqLen，允许尾部 0~2 个 batch seqLen=0
            if batch_size == 1:
                seq_lens = torch.tensor([max_query_len_act], dtype=torch.int32)
            else:
                seq_lens = torch.randint(
                    low=1,
                    high=max_query_len_act + 1,
                    size=(batch_size,),
                    dtype=torch.int32,
                )
            num_zero_tail = torch.randint(0, min(3, batch_size), ()).item()
            if num_zero_tail > 0:
                seq_lens[-num_zero_tail:] = 0
            query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32)
            query_start_loc[1:] = torch.cumsum(seq_lens, dim=0)
            cu_seq_len = query_start_loc[-1].item()
            # 重新生成 x（形状随 seqLen 变化）
            x = torch.empty((cu_seq_len, x.shape[1]), dtype=x.dtype).uniform_(-5.0, 5.0)
    else:
        # 3D 输入：每个 batch seqLen 固定
        seq_lens = torch.full((batch_size,), fill_value=x.shape[1], dtype=torch.int32)

    enAPC = cache_indices is not None and cache_indices.ndim == 2

    # Non-pad cache slots must be distinct across the complete block table.
    # Sharing across sequences would introduce order-dependent state writes.
    if cache_indices is not None:
        num_slots = conv_states.shape[0]
        required_slots = batch_size
        if cache_indices.ndim == 2:
            required_slots *= cache_indices.shape[1]
        elif cache_indices.ndim != 1:
            raise ValueError("cache_indices must have one or two dimensions")
        if num_slots < required_slots:
            raise ValueError(
                f"conv_states has {num_slots} slots but cache_indices requires "
                f"{required_slots} globally distinct slots"
            )
        cache_indices = torch.randperm(num_slots, dtype=torch.int32)[:required_slots]
        if enAPC:
            cache_indices = cache_indices.reshape(batch_size, -1)

    # ── L619-627: 约束化 num_accepted_tokens ─────────────────────────────
    if num_accepted_tokens is not None:
        for i in range(batch_size):
            if seq_lens[i] == 0:
                num_accepted_tokens[i] = torch.tensor([1], dtype=torch.int32)
            else:
                num_accepted_tokens[i] = torch.randint(
                    1, min(seq_lens[i].item(), m + 1) + 1, (1,)
                )

    # ── L629-636: 约束化 num_computed_tokens ─────────────────────────────
    if num_computed_tokens is not None:
        num_computed_tokens = torch.randint(
            low=0,
            high=2 * block_size + 1,
            size=(batch_size,),
            dtype=torch.int32,
        )

    # ── L637-658: APC 索引生成 ────────────────────────────────────────────
    if enAPC:
        seqlens_safe = torch.clamp(seq_lens, min=1)
        block_idx_first_scheduled_token = num_computed_tokens // block_size
        block_idx_last_scheduled_token = (
            num_computed_tokens + seqlens_safe - 1
        ) // block_size
        initial_state_index = []
        for batch_idx in range(batch_size):
            block_idx_first = block_idx_first_scheduled_token[batch_idx].item() + 1
            idx = torch.randint(low=0, high=max(block_idx_first, 1), size=(1,)).item()
            initial_state_index.append(idx)
        initial_state_idx_t = torch.tensor(initial_state_index, dtype=torch.int32)
        initial_state_idx = torch.minimum(
            initial_state_idx_t, block_idx_last_scheduled_token
        )

    # ── L660-674: pad_slot_id 无效 batch 注入 ────────────────────────────
    use_pad_slot_id = True
    if batch_size > 4 and use_pad_slot_id and cache_indices is not None:
        invalid_batch_choice = random.choice([0, 1, 2])
        batch_invalid_num = random.randint(0, batch_size // 4)
        batch_invalid_num_tail = random.randint(3 * batch_size // 4, batch_size - 1)
        if invalid_batch_choice == 0:
            cache_indices[:batch_invalid_num] = pad_slot_id
        elif invalid_batch_choice == 1:
            cache_indices[batch_invalid_num_tail:] = pad_slot_id
        else:
            cache_indices[:batch_invalid_num] = pad_slot_id
            cache_indices[batch_invalid_num_tail:] = pad_slot_id

    # ── L676-726: 所有张量移到目标 device ────────────────────────────────
    def to_dev(t):
        return t.to(device) if t is not None else None

    return {
        "x": to_dev(x),
        "weight": to_dev(weight),
        "conv_states": to_dev(conv_states),
        "query_start_loc": to_dev(query_start_loc),
        "cache_indices": to_dev(cache_indices),
        "num_accepted_tokens": to_dev(num_accepted_tokens),
        "num_computed_tokens": to_dev(num_computed_tokens),
        "block_idx_first_scheduled_token": to_dev(block_idx_first_scheduled_token),
        "block_idx_last_scheduled_token": to_dev(block_idx_last_scheduled_token),
        "initial_state_idx": to_dev(initial_state_idx),
        "pad_slot_id": pad_slot_id,
        "batch_size": batch_size,
        "xdim": xdim,
        "m": m,
    }


# ============================================================
# cann-bench 输入预处理（get_input）
# ============================================================


def get_input(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    pad_slot_id: int = -1,
    run_mode: int = 0,
    max_query_len: int = -1,
    residual_connection: int = 1,
    block_size: int = 128,
    conv_mode: int = 1,
    inplace: bool = False,
    **kwargs,
) -> tuple:
    """基于 prepare_inputs 为 cann-bench 生成合法且满足约束的输入张量。

    cases.yaml 中的 value_range 无法表达 query_start_loc 单调性、cache 索引有效范围等
    复杂约束，因此通过 get_input 在框架生成张量后重新约束化。
    """
    case_id = kwargs.get("case_id", 1)
    device = kwargs.get("device", "cpu")
    prepared = prepare_inputs(
        x=x,
        weight=weight,
        conv_states=conv_states,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accepted_tokens,
        num_computed_tokens=num_computed_tokens,
        block_idx_first_scheduled_token=block_idx_first_scheduled_token,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        pad_slot_id=pad_slot_id,
        run_mode=run_mode,
        max_query_len=max_query_len,
        block_size=block_size,
        case_id=case_id,
        device=device,
    )
    return (
        prepared["x"],
        prepared["weight"],
        prepared["conv_states"],
        prepared["query_start_loc"],
        prepared["cache_indices"],
        prepared["num_accepted_tokens"],
        prepared["num_computed_tokens"],
        prepared["block_idx_first_scheduled_token"],
        prepared["block_idx_last_scheduled_token"],
        prepared["initial_state_idx"],
    )


# ============================================================
# Golden 实现：因果卷积
# ============================================================


def causal_conv1d_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    max_query_len: int = -1,
    pad_slot_id: int = -1,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    num_computed_tokens: Optional[torch.Tensor] = None,
    block_idx_first_scheduled_token: Optional[torch.Tensor] = None,
    block_idx_last_scheduled_token: Optional[torch.Tensor] = None,
    initial_state_idx: Optional[torch.Tensor] = None,
    B_size: int = 128,
    conv_mode: int = 1,
    inplace: bool = False,
    residual: bool = True,
    activation: Optional[str] = None,
) -> tuple:
    """
    FusedCausalConv1d 的 CPU Golden 参考实现。

    参数:
        x:                               输入张量，shape=(cu_seq_len, dim) 或 (B, S, dim)
        weight:                          卷积核，shape=(width, dim)
        conv_states:                     KV 状态缓存，shape=(num_slots, state_len, dim)
        query_start_loc:                 各 batch 起始 token 位置，shape=(B+1,)
        cache_indices:                   cache slot 索引，shape=(B,) 或 (B, max_blocks)
        max_query_len:                   最大序列长度（-1 表示不限）
        pad_slot_id:                     无效 slot ID（跳过对应 batch）
        num_accepted_tokens:             投机解码每 batch 接受 token 数，shape=(B,)
        num_computed_tokens:             已计算 token 数（APC/Pangu v2），shape=(B,)
        block_idx_first_scheduled_token: APC 第一个调度 block 索引，shape=(B,)
        block_idx_last_scheduled_token:  APC 最后一个调度 block 索引，shape=(B,)
        initial_state_idx:               APC 初始 state 读取索引，shape=(B,)
        B_size:                          APC block 大小，默认 128
        conv_mode:                       1=Pangu v2（首 token 补零）；0=标准
        inplace:                         是否原地更新 x
        residual:                        是否加残差连接
        activation:                      激活函数（None 或 "silu"）

    返回:
        (out, conv_states): out shape 与 x 相同
    """
    # 处理 3D 输入
    if x.ndim == 3:
        flattened = True
        bsz, seq_len, dim = x.shape
        x = x.view(-1, dim)
        if query_start_loc is None:
            query_start_loc = torch.arange(
                start=0,
                end=(bsz + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=x.device,
            )
    else:
        flattened = False

    cu_seq_len, dim = x.shape
    batch_size = query_start_loc.shape[0] - 1
    width = weight.size(0)
    assert conv_states.size(1) >= width - 1

    apc_enabled = block_idx_last_scheduled_token is not None
    out = torch.zeros_like(x)

    for batch_idx in range(batch_size):
        start_idx = query_start_loc[batch_idx].item()
        end_idx = query_start_loc[batch_idx + 1].item()
        seq_len = end_idx - start_idx
        seq_x = x[start_idx:end_idx]

        if seq_len == 0:
            continue

        # APC 模式：计算各索引
        if apc_enabled:
            seq_completed_offset_token = num_computed_tokens[batch_idx].item() % B_size
            seq_completed_offset = B_size - seq_completed_offset_token
            seq_end_offset = (seq_len - seq_completed_offset) % B_size
            last_full_block_token_index = seq_len - seq_end_offset
            if seq_end_offset == 0:
                last_full_block_token_index -= B_size
            idx_first = block_idx_first_scheduled_token[batch_idx].item()
            idx_last = block_idx_last_scheduled_token[batch_idx].item()
            n_block_to_fill = idx_last - idx_first

            assert cache_indices is not None and cache_indices.ndim == 2
            read_cache_line = cache_indices[
                batch_idx, initial_state_idx[batch_idx]
            ].item()
            write_cache_line = cache_indices[batch_idx, idx_last].item()
        else:
            if cache_indices is not None:
                read_cache_line = cache_indices[batch_idx].item()
                write_cache_line = cache_indices[batch_idx].item()
            else:
                read_cache_line = batch_idx
                write_cache_line = batch_idx

        if read_cache_line == pad_slot_id:
            continue

        # Step 1: 读取历史 cache
        if num_computed_tokens is not None and num_computed_tokens[batch_idx] == 0:
            cached_state = torch.zeros((width - 1, dim), device=x.device, dtype=x.dtype)
            offset = 0
        else:
            if num_accepted_tokens is not None:
                accepted_tokens = num_accepted_tokens[batch_idx].item()
                assert 1 <= accepted_tokens <= seq_len
                offset = accepted_tokens - 1
            else:
                offset = conv_states.size(1) - (width - 1)
            cached_state = conv_states[read_cache_line][: offset + width - 1]

        padded_input = torch.cat([cached_state, seq_x], dim=0)

        # Step 2: 写入 running cache
        cache_len = min(conv_states.size(1), padded_input.size(0))
        conv_states[write_cache_line][-cache_len:] = padded_input[-cache_len:]

        padded_input = padded_input[offset:]

        # Step 2b: 写入 prefix cache（APC 模式）
        if apc_enabled:
            for chunk in range(n_block_to_fill):
                boundary_idx = (
                    last_full_block_token_index - (n_block_to_fill - chunk - 1) * B_size
                )
                assert boundary_idx > 0, (
                    "Sequence length, block_idx_first/last_scheduled_token mismatched. "
                )
                write_cache_line = cache_indices[batch_idx, idx_first + chunk]
                conv_states[write_cache_line][-(width - 1) :] = padded_input[
                    boundary_idx : boundary_idx + width - 1
                ]

        # Step 3: 因果卷积
        result = F.conv1d(
            padded_input.transpose(0, 1).unsqueeze(0),  # (1, dim, width-1+seq_len)
            weight.transpose(0, 1).unsqueeze(1),  # (dim, 1, width)
            bias=None,
            stride=1,
            padding=0,
            groups=dim,
        )
        result = result.squeeze(0).transpose(0, 1)
        # Pangu v2：将初始填充段置零
        if conv_mode == 1:
            assert num_computed_tokens is not None
            last_reset_idx = width - 1 - num_computed_tokens[batch_idx].item()
            last_reset_idx = min(max(last_reset_idx, 0), seq_len)
            result[:last_reset_idx] = 0

        result = result + seq_x if residual else result

        if activation is not None:
            if activation not in [None, "silu"]:
                raise NotImplementedError("activation 仅支持 None 或 'silu'")
            result = F.silu(result)

        out[start_idx:end_idx] = result
        if inplace:
            seq_x = out[start_idx:end_idx]
            x[start_idx:end_idx] = seq_x

    if inplace:
        return (x if not flattened else x.view(bsz, -1, dim), conv_states)
    return (out if not flattened else out.view(bsz, -1, dim), conv_states)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

dim = 512
width = 3
batch = 4
seq_len = 16
cu_seq_len = batch * seq_len
block_size = 128

dtype = torch.bfloat16
device = "npu"

x = torch.randn(cu_seq_len, dim, dtype=dtype, device=device)
weight = torch.randn(width, dim, dtype=dtype, device=device)
state_len = width - 1
conv_states = torch.randn(100, state_len, dim, dtype=dtype, device=device)

query_start_loc = torch.arange(0, cu_seq_len + 1, seq_len, dtype=torch.int32, device=device)
cache_indices = torch.arange(batch, dtype=torch.int32, device=device)
num_computed_tokens = torch.zeros(batch, dtype=torch.int32, device=device)

y, conv_states = cann_bench.fused_causal_conv1d(
    x, weight, conv_states,
    query_start_loc=query_start_loc,
    cache_indices=cache_indices,
    num_computed_tokens=num_computed_tokens,
    activation="silu",
    pad_slot_id=-1,
    max_query_len=-1,
    residual_connection=1,
    block_size=block_size,
    conv_mode=1,
    inplace=False,
)
```
