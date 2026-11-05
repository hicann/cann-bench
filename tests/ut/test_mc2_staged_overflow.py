# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Expected dtype overflow must not invalidate an independent CPU cascade."""

import json

import pytest
import torch

from kernel_eval.checkers.mc2_three_way_checker import (
    check_mc2_references,
    compare_mc2_outputs,
)
from kernel_eval.utils.mc2_compare import error_metrics


@pytest.mark.parametrize("value", [65505.0, 65519.0])
def test_fp16_rounding_above_max_finite_does_not_invalidate_reference(value):
    golden = torch.tensor([value, -value], dtype=torch.float32)
    native = golden.half()
    assert torch.equal(native.float(), torch.tensor([65504.0, -65504.0]))
    result = compare_mc2_outputs(
        native, native, golden, 0, cascade_golden=native.float()
    )[0]
    assert result.passed, result.error_msg
    health = result.metadata["reference_check"]
    assert health["expected_overflow_count"] == 0
    assert health["finite_health_count"] == 2
    # Keep errors against the original FP32 math, not the rounded output.
    assert result.metadata["candidate_metrics"]["diff_big_max"] == value - 65504.0
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "dtype,boundary",
    [(torch.float16, 65520.0), (torch.bfloat16, float.fromhex("0x1.ffp127"))],
)
def test_overflow_masks_match_actual_cast_at_rounding_boundary(dtype, boundary):
    largest = torch.tensor(torch.finfo(dtype).max, dtype=torch.float32)
    threshold = torch.tensor(boundary, dtype=torch.float32)
    positive = torch.stack(
        (
            largest,
            torch.nextafter(largest, torch.tensor(float("inf"))),
            torch.nextafter(threshold, torch.tensor(0.0)),
            threshold,
            torch.nextafter(threshold, torch.tensor(float("inf"))),
        )
    )
    golden = torch.cat((positive, -positive))
    # Use torch's real cast as the independent oracle for the NumPy metrics.
    native = golden.to(dtype).float()
    assert torch.isfinite(native[:3]).all()
    assert torch.isposinf(native[3:5]).all()
    snapshot = golden.clone()
    metrics = error_metrics(
        native.numpy(), golden.numpy(), str(dtype).removeprefix("torch.")
    )
    assert metrics["num_total_inf"] == metrics["num_total_ninf"] == 2
    assert (
        metrics["err_total_inf"]
        == metrics["err_total_ninf"]
        == metrics["err_total_nan"]
        == 0
    )
    assert torch.equal(golden, snapshot)
    # BF16 near-max error squares exceed FP32 statistics range; this test
    # checks the cast masks, not acceptance of arbitrary extreme magnitudes.


@pytest.mark.parametrize("side", ["candidate", "native", "staged"])
@pytest.mark.parametrize("fault", ["extra_inf", "wrong_sign", "missing_inf", "nan"])
def test_rounding_boundary_keeps_exceptional_value_gates(side, fault):
    golden = torch.tensor([65505.0, -65505.0, 65520.0, -65520.0, 1.0])
    good = golden.half()
    broken = good.clone()
    if fault == "extra_inf":
        broken[0] = float("inf")
    elif fault == "wrong_sign":
        broken[2] = -float("inf")
    elif fault == "missing_inf":
        broken[2] = 65504.0
    else:
        broken[0] = float("nan")
    result = compare_mc2_outputs(
        broken if side == "candidate" else good,
        broken if side == "native" else good,
        golden,
        0,
        cascade_golden=(broken if side == "staged" else good).float(),
    )[0]
    assert not result.passed
    assert result.metadata["reference_valid"] is (side == "candidate")
    json.dumps(result.to_dict(), allow_nan=False)


def test_rounding_boundary_does_not_rescue_finite_candidate_error():
    golden = torch.tensor([65505.0, -65505.0, 65520.0, -65520.0, 1.0])
    native = golden.half()
    candidate = native.clone()
    candidate[0] = 65000.0
    result = compare_mc2_outputs(
        candidate, native, golden, 0, cascade_golden=native.float()
    )[0]
    assert result.metadata["reference_valid"]
    assert not result.passed and "diff_big_max" in result.metadata["failed_metrics"]
    assert result.metadata["cascade_metrics"]["diff_big_max"] == 1.0
    assert result.metadata["limits"]["diff_big_max"] == 10.0


@pytest.mark.parametrize(
    "dtype,large", [(torch.float16, 70000.0), (torch.bfloat16, 3.4e38)]
)
@pytest.mark.parametrize("finite_values", [[], [1.125, -2.25, 0.0, 0.0002]])
def test_expected_signed_overflow_keeps_finite_checks_and_json(
    dtype, large, finite_values
):
    golden = torch.tensor([large, -large, *finite_values], dtype=torch.float32)
    native = golden.to(dtype)
    staged = native.float()
    result = compare_mc2_outputs(native, native, golden, 0, cascade_golden=staged)[0]
    assert result.passed, result.error_msg
    health = result.metadata["reference_check"]
    assert health["finite_health_count"] == len(finite_values)
    assert health["expected_overflow_count"] == 2
    assert health["npu_stage_relative_rmse"] == 0.0
    assert result.metadata["candidate_metrics"]["err_total_inf"] == 0
    assert result.metadata["candidate_metrics"]["err_total_ninf"] == 0
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("side", ["candidate", "native", "staged"])
@pytest.mark.parametrize(
    "fault", ["wrong_sign", "missing_inf", "extra_inf", "nan", "wrong_position"]
)
def test_overflow_masks_cannot_hide_broken_outputs(side, fault):
    golden = torch.tensor([70000.0, -70000.0, 1.0, 2.0])
    good = golden.half()
    broken = good.clone()
    if fault == "wrong_sign":
        broken[0] = -float("inf")
    elif fault == "missing_inf":
        broken[0] = 65504.0
    elif fault == "extra_inf":
        broken[2] = float("inf")
    elif fault == "nan":
        broken[0] = float("nan")
    else:
        broken[0], broken[2] = good[2], good[0]
    result = compare_mc2_outputs(
        broken if side == "candidate" else good,
        broken if side == "native" else good,
        golden,
        0,
        cascade_golden=(broken if side == "staged" else good).float(),
    )[0]
    assert not result.passed
    assert result.metadata["reference_valid"] is (side == "candidate")
    if side != "candidate":
        assert result.metadata["failure_type"] == "reference_invalid"
        assert "candidate_metrics" not in result.metadata
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("side", ["native", "staged", "both"])
def test_huge_overflow_values_cannot_dilute_finite_reference_errors(side):
    golden = torch.tensor([1e30, -1e30, 100.0, 100.0])
    staged = golden.half().float()
    native = golden.half()
    if side in ("staged", "both"):
        staged[2:] += 0.5
    if side in ("native", "both"):
        native[2:] += 1.0
    result = check_mc2_references(native, golden, 0, cascade_golden=staged)[0]
    assert not result.passed
    assert result.metadata["failure_type"] == "reference_invalid"
    assert any(
        "relative_rmse" in name
        for name in result.metadata["reference_check"]["failed_metrics"]
    )


@pytest.mark.parametrize("finite_value,bad_value", [(100.0, 101.0), (0.0, 1e-4)])
def test_many_overflows_cannot_dilute_one_finite_error(finite_value, bad_value):
    golden = torch.full((4097,), 70000.0)
    golden[-1] = finite_value
    native = golden.half()
    native[-1] = bad_value
    # Even matching CPU/NPU errors must not pass by dividing their finite
    # error by the total output count or by a huge overflow normalization.
    result = check_mc2_references(native, golden, 0, cascade_golden=native.float())[0]
    assert not result.passed and not result.metadata["reference_valid"]
    health = result.metadata["reference_check"]
    assert health["finite_health_count"] == 1
    assert "cpu_staged_cascade.relative_rmse" in health["failed_metrics"]


@pytest.mark.parametrize("fault", ["zero", "double", "offset"])
def test_matching_overflow_masks_do_not_rescue_finite_candidate_errors(fault):
    golden = torch.tensor([70000.0, -70000.0, 1.125, -2.25, 0.0, 0.0002])
    native = golden.half()
    candidate = native.clone()
    candidate[2:] = {
        "zero": lambda: torch.zeros_like(native[2:]),
        "double": lambda: native[2:] * 2,
        "offset": lambda: native[2:] + 1,
    }[fault]()
    result = compare_mc2_outputs(
        candidate, native, golden, 0, cascade_golden=native.float()
    )[0]
    assert not result.passed and result.metadata["reference_valid"]
    assert result.metadata["failed_metrics"]


@pytest.mark.parametrize("value", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_mathematical_golden_is_still_invalid(value):
    golden = torch.tensor([value, 1.0])
    native = golden.half()
    result = check_mc2_references(native, golden, 0, cascade_golden=native.float())[0]
    assert not result.passed
    assert result.error_msg == "CPU FP32 reference contains nonfinite values"


def test_fp32_comparison_does_not_inherit_fp16_overflow_allowance():
    golden = torch.tensor([70000.0, 1.0])
    native = golden.half()
    result = check_mc2_references(
        native, golden, 0, "float32", cascade_golden=native.float()
    )[0]
    assert not result.passed and not result.metadata["reference_valid"]


def test_intermediate_overflow_with_finite_final_math_is_not_rescued():
    # A hypothetical later cancellation must not turn an intermediate Inf into
    # permission to ignore a finite mathematical result.
    golden = torch.tensor([100.0, 1.0])
    native = torch.tensor([float("inf"), 1.0], dtype=torch.float16)
    assert not check_mc2_references(native, golden, 0, cascade_golden=native.float())[
        0
    ].passed
