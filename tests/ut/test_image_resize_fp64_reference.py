# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""FP64 ResizeNormalize oracle must retain sub-FP32 feature/mean values."""

import importlib.util
from pathlib import Path
import pytest
import torch


def load_reference():
    root = Path(__file__).resolve().parents[2]
    tasks = root / "tasks"
    spec = importlib.util.spec_from_file_location(
        "resize_fp64_test", tasks / "level2/image_resize_normalize/golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.image_resize_normalize


@pytest.mark.parametrize("aligned", [False, True])
def test_oracle_preserves_feature_bits(aligned):
    eps = 2**-30
    x = torch.full((1, 1, 2, 3), 1 + eps, dtype=torch.float64)
    y = load_reference()(
        x,
        torch.ones(1, dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        3,
        5,
        aligned,
    )
    torch.testing.assert_close(y, torch.full_like(y, eps), rtol=0, atol=1e-15)


@pytest.mark.parametrize("aligned", [False, True])
def test_oracle_preserves_mean_bits(aligned):
    eps = 2**-30
    x = torch.ones(1, 1, 2, 3, dtype=torch.float64)
    y = load_reference()(
        x,
        torch.tensor([1 + eps], dtype=torch.float64),
        torch.ones(1, dtype=torch.float64),
        3,
        5,
        aligned,
    )
    torch.testing.assert_close(y, torch.full_like(y, -eps), rtol=0, atol=1e-15)
