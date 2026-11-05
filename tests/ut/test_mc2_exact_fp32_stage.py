# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""An exact FP32 CPU stage needs a nonzero, independently fixed health bound."""

import json

import pytest
import torch

from kernel_eval.checkers.mc2_three_way_checker import (
    check_mc2_references,
    compare_mc2_outputs,
)
from kernel_eval.utils.mc2_compare import compare_three_way
from kernel_eval.utils.thresholds import PRECISION_THRESHOLDS


@pytest.mark.parametrize("count", [1, 65536])
@pytest.mark.parametrize("value", [-1.0, 1.0])
@pytest.mark.parametrize("direction", [float("-inf"), float("inf")])
def test_exact_fp32_stage_accepts_same_direction_ulp(count, value, direction):
    cpu = torch.full((count,), value)
    native = torch.nextafter(cpu, torch.full_like(cpu, direction))
    evidence = compare_three_way(native.numpy(), cpu.numpy(), cpu.numpy(), "float32")
    assert evidence["passed"] and evidence["accepted_by"] == "finite_tolerance"

    result = compare_mc2_outputs(native, native, cpu, 0, cascade_golden=cpu)[0]
    assert result.passed, result.error_msg
    health = result.metadata["reference_check"]
    assert health["accepted_by"] == "finite_tolerance"
    assert health["npu_stage_bias_fraction"] == 1.0
    assert health["failed_metrics"] == []
    assert result.metadata["reference_valid"]
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("count", [1, 65536])
def test_exact_fp32_stage_ratio_acceptance_stays_within_absolute_tolerance(count):
    cpu = torch.zeros(count)
    # FP32 squared errors underflow, so these nonzero errors pass the original
    # ratio branch. They are also well inside its independent absolute bound.
    native = torch.full_like(cpu, 1e-24)
    evidence = compare_three_way(native.numpy(), cpu.numpy(), cpu.numpy(), "float32")
    assert evidence["passed"] and evidence["accepted_by"] == "cascade_ratio"
    assert (native.abs() <= evidence["absolute_tolerance"]).all()
    result = check_mc2_references(native, cpu, 0, cascade_golden=cpu)[0]
    assert result.passed, result.error_msg
    assert result.metadata["reference_check"]["accepted_by"] == "cascade_ratio"

    outside = torch.full_like(cpu, 2e-9)
    result = check_mc2_references(outside, cpu, 0, cascade_golden=cpu)[0]
    assert not result.passed
    assert result.metadata["failure_type"] == "reference_invalid"


@pytest.mark.parametrize("count", [1, 65536])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_exact_fp32_stage_rejects_same_direction_error_needing_fallback(
    count, direction
):
    cpu = torch.ones(count)
    # Below the wider reference-precision limit but above the ordinary finite
    # tolerance: removing the bias gate entirely would accept this offset.
    native = cpu + direction * 2e-5
    evidence = compare_three_way(native.numpy(), cpu.numpy(), cpu.numpy(), "float32")
    assert not evidence["passed"]
    result = check_mc2_references(native, cpu, 0, cascade_golden=cpu)[0]
    assert not result.passed and not result.metadata["reference_valid"]
    health = result.metadata["reference_check"]
    assert (
        health["npu_stage_max_relative_to_rms"]
        < health["npu_stage_max_relative_to_rms_limit"]
    )
    assert "npu_vs_cpu_staged.bias_fraction" in health["failed_metrics"]


def test_same_direction_ulp_reference_does_not_relax_candidate_limits():
    cpu = torch.ones(4096)
    native = torch.nextafter(cpu, torch.full_like(cpu, float("inf")))
    candidate = cpu + 32 * (native - cpu)
    result = compare_mc2_outputs(candidate, native, cpu, 0, cascade_golden=cpu)[0]
    assert result.metadata["reference_valid"]
    assert not result.passed
    assert result.metadata.get("failure_type") != "reference_invalid"
    assert "diff_big_max" in result.metadata["failed_metrics"]
    assert result.metadata["limits"]["diff_big_max"] == 10 * float(native[0] - cpu[0])


def test_exact_fp32_fallback_preserves_small_value_count_gate():
    cpu = torch.cat((torch.zeros(100000), torch.ones(100000)))
    native = cpu.clone()
    native[:100000:2] = 2e-9
    native[1:100000:2] = -2e-9
    result = check_mc2_references(native, cpu, 0, cascade_golden=cpu)[0]
    assert not result.passed and not result.metadata["reference_valid"]
    assert "err_small_num" in result.metadata["reference_check"]["failed_metrics"]


@pytest.mark.parametrize("relative_error", [9.5e-8, 7.7e-7, 2e-5])
def test_exact_fp32_stage_accepts_bounded_native_gemm_roundoff(relative_error):
    cpu = torch.full((4096,), 100.0)
    native = cpu.clone()
    native[::2] *= 1 + relative_error
    native[1::2] *= 1 - relative_error
    result = compare_mc2_outputs(native, native, cpu, 0, cascade_golden=cpu)[0]
    assert result.passed, result.error_msg
    health = result.metadata["reference_check"]
    assert health["accepted_by"] == "fixed_fp32_reference_precision"
    assert health["npu_stage_relative_rmse_limit"] == PRECISION_THRESHOLDS["float32"]
    assert (
        health["npu_stage_max_relative_to_rms"]
        <= health["npu_stage_max_relative_to_rms_limit"]
    )
    assert health["cpu_stage_metrics"]["diff_rmse"] == 0
    assert health["npu_metrics"]["diff_rmse"] > 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_case25_scale_sparse_roundoff_remains_bounded():
    cpu = torch.full((4096,), 53.9)
    native = cpu.clone()
    native[0] += 0.004
    result = check_mc2_references(native, cpu, 0, cascade_golden=cpu)[0]
    assert result.passed, result.error_msg
    health = result.metadata["reference_check"]
    assert health["npu_stage_max_relative_to_rms"] < PRECISION_THRESHOLDS["float32"]


@pytest.mark.parametrize("corruption", ["systematic", "sparse", "wrong_inf", "nan"])
def test_exact_fp32_stage_rejects_broken_native_reference(corruption):
    cpu = torch.full((65536,), 100.0)
    native = cpu.clone()
    if corruption == "systematic":
        native += 0.1
    elif corruption == "sparse":
        # RMS alone would hide this one bad output in 65536 values.
        native[0] += 1.0
    elif corruption == "wrong_inf":
        native[0] = float("inf")
    else:
        native[0] = float("nan")
    result = check_mc2_references(native, cpu, 0, cascade_golden=cpu)[0]
    assert not result.passed and not result.metadata["reference_valid"]
    if corruption in ("systematic", "sparse"):
        assert result.metadata["reference_check"]["failed_metrics"]


def test_exact_fp32_stage_keeps_candidate_comparison_strict():
    cpu = torch.full((8192,), 100.0)
    native = cpu.clone()
    native[::2] += 0.002
    native[1::2] -= 0.002
    candidate = cpu + 1.0
    result = compare_mc2_outputs(candidate, native, cpu, 0, cascade_golden=cpu)[0]
    assert result.metadata["reference_valid"]
    assert not result.passed and result.metadata["failed_metrics"]
    assert result.metadata.get("failure_type") != "reference_invalid"


def test_nonexact_fp32_stage_keeps_original_health_limit():
    cpu = torch.full((4096,), 100.0)
    staged = cpu + 0.001
    native = staged + 0.001
    result = check_mc2_references(native, cpu, 0, cascade_golden=staged)[0]
    health = result.metadata["reference_check"]
    assert health["npu_stage_relative_rmse_limit"] == 2**-23
    assert not result.passed


def test_exact_bf16_stage_cannot_use_fp32_bound():
    cpu = torch.full((4096,), 100.0)
    native = (cpu + 2.0).bfloat16()
    result = check_mc2_references(native, cpu, 0, cascade_golden=cpu)[0]
    assert not result.passed
    assert result.metadata["reference_check"]["npu_stage_relative_rmse_limit"] == 2**-7
