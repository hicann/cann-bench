#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
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
import torch_npu
import cann_bench
import pytest


def test_sqrt_interface_exist():
    """
    Test that the 'cann_bench.sqrt' operator is present in torch.ops.
    """
    print(torch.ops.cann_bench.sqrt)
    assert hasattr(torch.ops.cann_bench, "sqrt"), "The 'sqrt' operator is not registered in the 'torch.ops.cann_bench' namespace."


SHAPES = [
    (1,),
    (3,),
    (10,),
    (100,),
    (1024,),
    (10000,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (100, 10),
    (256, 512),
    (5, 10, 15),
    (16, 32, 64),
    (32, 64, 128),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    (8, 3, 128, 128),
    (1000, 1000),
]

DTYPES = [
    torch.float32,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sqrt_operator(shape, dtype):
    """
    Test the functionality of the sqrt operator.

    Parameters:
        shape: Tensor shape
        dtype: Data type
    """
    a = torch.randn(*shape, dtype=dtype)
    a = torch.abs(a)

    expected = torch.sqrt(a)
    a_npu = a.npu()
    result_npu = torch.ops.cann_bench.sqrt(a_npu)
    result = result_npu.cpu()

    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), \
        f"Sqrt failed for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"

    print(f"Test passed: shape={shape}, dtype={dtype}")
