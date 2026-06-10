#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch



def _ceil(value):
    ivalue = int(value)
    return ivalue if float(value) == float(ivalue) else ivalue + 1


class _Math:
    @staticmethod
    def ceil(value):
        return _ceil(value)


math = _Math()
F = torch.nn.functional


class _NumpyAlias:
    ndarray = torch.Tensor
    dtype = torch.dtype
    float32 = torch.float32
    float64 = torch.float64
    int32 = torch.int32
    int64 = torch.int64
    inf = float("inf")


np = _NumpyAlias()

def esa_select_topk(query, key, actual_seq_q_len_optional=None, actual_seq_k_len_optional=None,
                    actual_cmp_seq_k_len_optional=None, blk_size=64, init_blk_num=2,
                    local_blk_num=4, topk=4, input_layout="TND", compress_blk_size=16):
    if input_layout != "TND":
        raise ValueError("CPU golden currently supports TND layout")
    native = query.device.type == "npu" and query.dtype in (torch.float16, torch.bfloat16)
    query = query.detach() if native else query.detach().cpu().to(torch.float32)
    key = key.detach() if native else key.detach().cpu().to(torch.float32)

    def _to_ints(value, default):
        if value is None:
            return list(default)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1).tolist()
        return [int(x) for x in value]

    q_prefix = _to_ints(actual_seq_q_len_optional, [query.shape[0]])
    if not q_prefix:
        q_prefix = [query.shape[0]]
    k_prefix = _to_ints(actual_seq_k_len_optional, [key.shape[1] * compress_blk_size] * len(q_prefix))
    if len(k_prefix) < len(q_prefix):
        fill = k_prefix[-1] if k_prefix else key.shape[1] * compress_blk_size
        k_prefix.extend([fill] * (len(q_prefix) - len(k_prefix)))
    if actual_cmp_seq_k_len_optional is None:
        prev = 0
        cmp_lens = [_ceil(max(int(end) - prev, 0) / int(compress_blk_size)) for end in k_prefix]
    else:
        cmp_lens = _to_ints(actual_cmp_seq_k_len_optional, [])
    if len(cmp_lens) < len(q_prefix):
        fill = cmp_lens[-1] if cmp_lens else key.shape[1]
        cmp_lens.extend([fill] * (len(q_prefix) - len(cmp_lens)))

    def _block_select(cur_query, cur_key, key_len):
        cs = max(int(blk_size) // int(compress_blk_size), 1)
        cmp_len = min(int(_ceil(key_len / int(blk_size)) * cs), cur_key.shape[0])
        cur_key = cur_key[:cmp_len]
        block_num = max(cmp_len // cs, 1)
        cur_key = cur_key[: block_num * cs].reshape(block_num, cs, key.shape[-2], key.shape[-1])
        cur_key = cur_key.permute(2, 0, 1, 3)

        n2 = key.shape[-2]
        group = max(cur_query.shape[1] // n2, 1)
        expanded_key = cur_key.repeat_interleave(group, dim=0)
        scores = torch.einsum("snd,nijd->snij", cur_query, expanded_key)
        scores = scores.reshape(cur_query.shape[0], n2, group, block_num, cs).sum(dim=2).max(dim=-1).values

        if int(init_blk_num) > 0:
            scores[:, :, : int(init_blk_num)] = torch.arange(
                int(init_blk_num), 0, -1, dtype=scores.dtype, device=scores.device
            ).view(1, 1, -1)

        row_blocks = torch.arange(scores.shape[0], device=scores.device) // int(blk_size)
        for block_idx in range(block_num):
            start_idx = max(0, block_idx - int(local_blk_num))
            end_idx = block_idx + 1
            dec = torch.arange(end_idx - start_idx - 1, -1, -1, dtype=scores.dtype, device=scores.device)
            rows = row_blocks == block_idx
            if rows.any():
                scores[rows, :, start_idx:end_idx] = dec.view(1, 1, -1)

        redundant_query_num = cur_query.shape[0] - int(blk_size) * block_num
        mask_block_num = block_num + 1 if redundant_query_num > 0 else block_num
        rows = torch.arange(mask_block_num, device=scores.device).view(-1, 1)
        cols = torch.arange(mask_block_num, device=scores.device).view(1, -1)
        block_mask = torch.zeros(mask_block_num, mask_block_num, dtype=scores.dtype, device=scores.device)
        block_mask.masked_fill_(torch.triu(torch.ones_like(block_mask, dtype=torch.bool), diagonal=1), float("inf"))
        max_value = torch.max(torch.abs(scores)) * 3
        block_mask.masked_fill_((rows > cols) & (cols < int(init_blk_num)), -max_value)
        max_value = torch.max(torch.abs(scores)) * 2
        local_mask = (cols >= (rows - int(local_blk_num)).clamp(min=0)) & (cols <= rows)
        block_mask.masked_fill_(local_mask, -max_value)
        block_mask = block_mask.repeat_interleave(int(blk_size), dim=0)
        scores = scores - block_mask[: scores.shape[0], :block_num].view(1, scores.shape[0], 1, block_num)[0]

        select_count = int(init_blk_num) + int(local_blk_num) + int(topk) + 1
        _, idx = torch.topk(scores, k=min(select_count, block_num), dim=-1)
        if idx.shape[-1] < select_count:
            idx = F.pad(idx, (0, select_count - idx.shape[-1]), value=-1)
        target = torch.arange(0, select_count, dtype=idx.dtype, device=idx.device)
        prefix_rows = min(select_count * int(blk_size), idx.shape[0])
        if prefix_rows > 0:
            idx[:prefix_rows, :, :] = target
        thresholds = (torch.arange(idx.shape[0], device=idx.device) // int(blk_size)).view(-1, 1, 1)
        idx = idx.masked_fill(idx > thresholds, -1)
        return idx.to(torch.int32)

    out = []
    q_start = 0
    k_start = 0
    select_count = int(init_blk_num) + int(local_blk_num) + int(topk) + 1
    for b, q_end in enumerate(q_prefix):
        q_end = int(q_end)
        cur_q = query[q_start:q_end]
        k_end = int(k_prefix[min(b, len(k_prefix) - 1)])
        key_len = max(k_end - k_start, 0)
        n2 = key.shape[-2]
        if key_len <= int(blk_size) * select_count:
            idx = torch.arange(select_count, dtype=torch.int32, device=query.device).expand(cur_q.shape[0], n2, select_count).clone()
            thresholds = (torch.arange(max(key_len - cur_q.shape[0], 0), key_len, device=query.device) // int(blk_size)).view(-1, 1, 1)
            if thresholds.numel() == idx.shape[0]:
                idx[idx > thresholds] = -1
            out.append(idx)
        else:
            cs = max(int(blk_size) // int(compress_blk_size), 1)
            padded_cmp_len = int(_ceil(key_len / int(blk_size)) * cs)
            cur_key = key[min(b, key.shape[0] - 1), : min(padded_cmp_len, key.shape[1])]
            init_pad_len = max(key_len - cur_q.shape[0], 0)
            padded_q = cur_q
            if init_pad_len > 0:
                padded_q = F.pad(padded_q, (0, 0, 0, 0, init_pad_len, 0))
            end_pad_len = int(_ceil(key_len / int(blk_size)) * int(blk_size)) - padded_q.shape[0]
            if end_pad_len > 0:
                padded_q = F.pad(padded_q, (0, 0, 0, 0, 0, end_pad_len), mode="constant", value=0)
            idx = _block_select(padded_q, cur_key, key_len)
            if init_pad_len > 0:
                idx = idx[init_pad_len:]
            if end_pad_len > 0:
                idx = idx[:-end_pad_len]
            out.append(idx)
        q_start = q_end
        k_start = k_end
    return torch.cat(out, dim=0) if out else torch.empty((0, key.shape[-2], select_count), dtype=torch.int32)


def ai_infra_esa_select_topk(
    query: torch.Tensor,
    key: torch.Tensor,
    blk_size: int = 64,
    init_blk_num: int = 2,
    local_blk_num: int = 4,
    topk: int = 4,
    input_layout: str = "TND",
    actual_seq_q_len_optional=None,
    actual_seq_k_len_optional=None,
    actual_cmp_seq_k_len_optional=None,
    compress_blk_size: int = 16,
    chunk_size=None,
    chunk_index=None,
    **kwargs,
) -> torch.Tensor:
    del chunk_size, chunk_index, kwargs
    return esa_select_topk(
        query,
        key,
        actual_seq_q_len_optional=actual_seq_q_len_optional,
        actual_seq_k_len_optional=actual_seq_k_len_optional,
        actual_cmp_seq_k_len_optional=actual_cmp_seq_k_len_optional,
        blk_size=blk_size,
        init_blk_num=init_blk_num,
        local_blk_num=local_blk_num,
        topk=topk,
        input_layout=input_layout,
        compress_blk_size=compress_blk_size,
    )


def get_input(query: torch.Tensor = None, key: torch.Tensor = None, **kwargs):
    chunk_size = kwargs.get("chunk_size")
    chunk_index = kwargs.get("chunk_index")
    input_layout = kwargs.get("input_layout", "TND")
    if query is not None and input_layout == "TND" and chunk_size and chunk_index:
        start = (int(chunk_index) - 1) * int(chunk_size)
        end = min(int(chunk_index) * int(chunk_size), query.shape[0])
        query = query[start:end]
        actual_seq_q_len = kwargs.get("actual_seq_q_len_optional")
        if actual_seq_q_len is not None:
            if isinstance(actual_seq_q_len, torch.Tensor):
                q_lens = [int(x) for x in actual_seq_q_len.detach().cpu().reshape(-1).tolist()]
            else:
                q_lens = [int(x) for x in actual_seq_q_len]
            if q_lens:
                query = query[: min(int(q_lens[-1]), query.shape[0])]
    if key is not None and input_layout == "TND":
        actual_cmp_seq_k_len = kwargs.get("actual_cmp_seq_k_len_optional")
        if actual_cmp_seq_k_len is not None:
            if isinstance(actual_cmp_seq_k_len, torch.Tensor):
                cmp_lens = [int(x) for x in actual_cmp_seq_k_len.detach().cpu().reshape(-1).tolist()]
            else:
                cmp_lens = [int(x) for x in actual_cmp_seq_k_len]
            batch = min(len(cmp_lens), key.shape[0])
            key = key[:batch]
            key = key.clone()
            for idx, cmp_len in enumerate(cmp_lens[:batch]):
                key[idx, int(cmp_len):] = 0
    return [query, key]
