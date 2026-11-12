#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""CPU scatter_block_update reference regression tests."""

import importlib.util
from pathlib import Path

import pytest
import torch


def _load(task, function):
    path = Path(__file__).resolve().parents[2] / "tasks" / "level3" / task / "golden.py"
    spec = importlib.util.spec_from_file_location(f"_indexing_{task}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)


block_update = _load("scatter_block_update", "scatter_block_update")


@pytest.mark.parametrize(
    "dtype", [torch.int8, torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_block_update_distinct_destinations(dtype, index_dtype, noncontiguous):
    data = torch.arange(60).reshape(3, 4, 5).to(dtype)
    indices = torch.tensor([[2, 3], [0, 1], [1, 0], [-1, -2]], dtype=index_dtype)
    updates = -torch.arange(1, 21).reshape(4, 5).to(dtype)
    if noncontiguous:
        data = data.transpose(1, 2).contiguous().transpose(1, 2)
        indices = indices.T.contiguous().T
        updates = updates.T.contiguous().T
        assert (
            not data.is_contiguous()
            and not indices.is_contiguous()
            and not updates.is_contiguous()
        )
    before = [value.clone() for value in (data, indices, updates)]
    expected = data.clone()
    expected[2, 3] = updates[0]
    expected[0, 1] = updates[1]
    expected[1, 0] = updates[2]
    expected[2, 2] = updates[3]
    actual = block_update(data, indices, updates)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.data_ptr() != data.data_ptr()
    for value, original in zip((data, indices, updates), before):
        torch.testing.assert_close(value, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "indices", [[[0, 1], [0, 1]], [[-1, -1], [2, 3]], [[2, -1], [-1, 3]]]
)
def test_block_update_duplicates_keep_last_write(indices):
    data = torch.zeros(3, 4, 5)
    indices = torch.tensor(indices)
    updates = torch.tensor([[1.0] * 5, [2.0] * 5])
    expected = data.clone()
    expected[indices[-1, 0], indices[-1, 1]] = 2
    torch.testing.assert_close(
        block_update(data, indices, updates), expected, rtol=0, atol=0
    )


def test_block_update_empty_still_clones():
    data = torch.ones(2, 3, 4)
    actual = block_update(data, torch.empty(0, 2, dtype=torch.int32), torch.empty(0, 4))
    assert torch.equal(actual, data) and actual.data_ptr() != data.data_ptr()


@pytest.mark.parametrize("destination", [[3, 0], [-4, 0], [0, 4], [0, -5]])
def test_block_update_out_of_bounds(destination):
    with pytest.raises(IndexError):
        block_update(
            torch.zeros(3, 4, 5), torch.tensor([destination]), torch.ones(1, 5)
        )


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("update_rows", [0, 1])
def test_block_update_missing_rows_raise_without_broadcast(index_dtype, update_rows):
    data = torch.zeros(3, 4, 5)
    indices = torch.tensor([[0, 0], [1, 1]], dtype=index_dtype)
    with pytest.raises(IndexError):
        block_update(data, indices, torch.ones(update_rows, 5))
    assert torch.count_nonzero(data) == 0


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_block_update_extra_rows_are_ignored(index_dtype):
    data = torch.zeros(3, 4, 5)
    indices = torch.tensor([[0, 0], [1, 1]], dtype=index_dtype)
    updates = torch.arange(15).reshape(3, 5).float()
    expected = data.clone()
    expected[0, 0] = updates[0]
    expected[1, 1] = updates[1]
    torch.testing.assert_close(
        block_update(data, indices, updates), expected, rtol=0, atol=0
    )


def test_block_update_flat_update_retains_index_error():
    with pytest.raises(IndexError):
        block_update(
            torch.zeros(3, 4, 5), torch.tensor([[0, 0], [1, 1]]), torch.ones(5)
        )
