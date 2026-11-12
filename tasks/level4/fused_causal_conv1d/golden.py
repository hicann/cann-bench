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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------

"""
FusedCausalConv1d算子Torch Golden参考实现

公式: fused_causal_conv1d(...)
"""

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
