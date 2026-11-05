# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Final-rounding reference health must preserve FP32 math and reject bad data.

Each task's arithmetic/communication is covered by its independent numerical
tests. These regressions exercise the task hook through the actual checker,
including the sparse cancellation which previously invalidated valid runs.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from kernel_eval.checkers.mc2_three_way_checker import (
    check_mc2_references,
    compare_mc2_outputs,
)


LEVEL4 = Path(__file__).resolve().parents[2] / "bench_lab/kernel_bench/level4"
OPS = (
    "all_gather_matmul_v2",
    "allto_all_matmul",
    "matmul_allto_all",
    "allto_allv_quant_grouped_mat_mul",
    "quant_reduce_scatter",
    "quant_all_reduce",
)


def _load(op):
    spec = importlib.util.spec_from_file_location(
        op + "_health_test", LEVEL4 / op / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _outputs(op, primary, auxiliary):
    if op in ("all_gather_matmul_v2", "allto_all_matmul"):
        return primary, auxiliary
    if op == "allto_allv_quant_grouped_mat_mul":
        return primary, primary.clone(), auxiliary
    return primary


def _hook(monkeypatch, op, dtype="bfloat16"):
    module = _load(op)
    mathematical = torch.linspace(-64, 64, 8192, dtype=torch.float32) + 0.1234
    mathematical[:8] = 0.0002
    auxiliary = torch.tensor([1.0, 1.25, -0.5, 3.0])
    expected = _outputs(op, mathematical, auxiliary)
    monkeypatch.setattr(module, "mc2_cpu_golden", lambda *_: expected)
    attrs = {
        "is_quant": True,
        "mc2_accuracy_mode": "three_way",
        "out_dtype": dtype,
        "gather_output": True,
        "alltoall_out_flag": True,
    }
    math, staged = module.mc2_cpu_references({"world_size": 2}, {}, attrs)
    assert math is expected
    return mathematical, auxiliary, math, staged


@pytest.mark.parametrize("op", OPS)
def test_quant_final_cast_handles_sparse_cancellation(monkeypatch, op):
    cpu, auxiliary, math, staged = _hook(monkeypatch, op)
    native = cpu.bfloat16()
    native[:8] = 0.00022
    baseline = _outputs(op, native, auxiliary.to(torch.float8_e4m3fn))
    assert any(not r.passed for r in check_mc2_references(baseline, math, 0))
    results = check_mc2_references(baseline, math, 0, cascade_golden=staged)
    assert all(r.passed for r in results), [r.error_msg for r in results]
    assert all(
        r.passed
        for r in compare_mc2_outputs(baseline, baseline, math, 0, cascade_golden=staged)
    )
    torch.testing.assert_close(cpu[:8], torch.full((8,), 0.0002), rtol=0, atol=0)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize(
    "fault", ["missing_bias", "wrong_scale", "wrong_layout", "all_zero", "nonfinite"]
)
def test_bad_reference_and_bad_candidate_are_still_rejected(monkeypatch, op, fault):
    cpu, auxiliary, math, staged = _hook(monkeypatch, op)
    good = _outputs(op, cpu.bfloat16(), auxiliary.to(torch.float8_e4m3fn))
    wrong = {
        "missing_bias": lambda: cpu - 3.0,
        "wrong_scale": lambda: cpu * 2.0,
        "wrong_layout": lambda: cpu.flip(0),
        "all_zero": lambda: torch.zeros_like(cpu),
        "nonfinite": lambda: torch.full_like(cpu, float("nan")),
    }[fault]().bfloat16()
    corrupted = _outputs(op, wrong, auxiliary.to(torch.float8_e4m3fn))
    reference = check_mc2_references(corrupted, math, 0, cascade_golden=staged)
    assert any(r.metadata.get("failure_type") == "reference_invalid" for r in reference)
    candidate = compare_mc2_outputs(corrupted, good, math, 0, cascade_golden=staged)
    assert any(not r.passed for r in candidate)
    assert all(r.metadata.get("failure_type") != "reference_invalid" for r in candidate)


@pytest.mark.parametrize(
    "op", ["all_gather_matmul_v2", "quant_reduce_scatter", "quant_all_reduce"]
)
def test_float32_output_does_not_create_an_artificial_rounding_allowance(
    monkeypatch, op
):
    cpu, auxiliary, math, staged = _hook(monkeypatch, op, dtype="float32")
    if isinstance(math, tuple):
        for mathematical_leaf, staged_leaf in zip(math, staged):
            torch.testing.assert_close(mathematical_leaf, staged_leaf, rtol=0, atol=0)
    else:
        torch.testing.assert_close(math, staged, rtol=0, atol=0)
    wrong = _outputs(op, cpu + 1e-3, auxiliary.to(torch.float8_e4m3fn))
    results = check_mc2_references(wrong, math, 0, cascade_golden=staged)
    assert any(not r.passed for r in results)


@pytest.mark.parametrize(
    "op",
    ["all_gather_matmul_v2", "allto_all_matmul", "allto_allv_quant_grouped_mat_mul"],
)
def test_auxiliary_communication_output_is_not_rounded_or_ignored(monkeypatch, op):
    cpu, auxiliary, math, staged = _hook(monkeypatch, op)
    wrong_auxiliary = (auxiliary * 2).to(torch.float8_e4m3fn)
    wrong = _outputs(op, cpu.bfloat16(), wrong_auxiliary)
    assert any(
        not r.passed
        for r in check_mc2_references(wrong, math, 0, cascade_golden=staged)
    )
