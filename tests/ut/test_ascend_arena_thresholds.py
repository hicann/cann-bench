# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_THRESHOLD_TABLES = (
    "PRECISION_THRESHOLDS",
    "SMALL_VALUE_THRESHOLDS",
    "SMALL_VALUE_ERROR_THRESHOLDS",
    "CANCEL_BOUNDARY_THRESHOLDS",
    "CANCEL_ZERO_THRESHOLDS",
)


def test_arena_thresholds_match_kernel_eval(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "src/ascend_arena_e2e"))

    from ascend_arena_e2e.ascend_arena.evaluator import thresholds as arena
    from kernel_eval.utils import thresholds as kernel_eval

    for name in SHARED_THRESHOLD_TABLES:
        arena_values = getattr(arena, name)
        kernel_eval_values = getattr(kernel_eval, name)
        shared_dtypes = set(arena_values) & set(kernel_eval_values)

        assert shared_dtypes, f"{name} has no shared dtypes"
        assert {
            dtype: arena_values[dtype] for dtype in shared_dtypes
        } == {
            dtype: kernel_eval_values[dtype] for dtype in shared_dtypes
        }, name
