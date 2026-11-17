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

"""
张量对比模块单元测试

测试对象：kernel_eval.utils.compare
"""

import json
import math

import pytest
import torch

from kernel_eval.utils import compare as comparison
from kernel_eval.utils.compare import compare_tensors


def _compare(candidate, native, dtype="float32"):
    return compare_tensors(candidate, torch.ones_like(candidate, dtype=torch.float64),
                           dtype, native_output=native)


def test_wrong_values_with_acceptable_count_ratio_are_rejected():
    # The old gate accepts 100 candidate errors / 100 native errors; magnitude matters now.
    native = torch.full((100,), 1.03125)
    candidate = native.clone()
    candidate[0] = 1.25
    result = _compare(candidate, native)
    assert result.normal_mare_ratio == pytest.approx(8)
    assert result.normal_mere_ratio < 1.5
    assert not result.normal_passed and not result.passed
    assert "正常值域三方比对失败" in result.output_results[0].error_msg


def test_mere_alone_can_reject():
    result = _compare(torch.full((100,), 1.0625), torch.full((100,), 1.03125))
    assert result.normal_mare_ratio == pytest.approx(2)
    assert result.normal_mere_ratio == pytest.approx(2)
    assert not result.normal_passed and not result.passed


@pytest.mark.parametrize("kind", ["mare", "mere"])
def test_values_just_inside_ratio_limits_pass(kind):
    native = torch.full((100,), 1.03125)
    candidate = torch.ones_like(native)
    if kind == "mare":
        candidate[0] = torch.nextafter(torch.tensor(1.15625), torch.tensor(1.0))
    else:
        candidate[:] = torch.nextafter(torch.tensor(1.046875), torch.tensor(1.0))
    result = _compare(candidate, native)
    assert result.normal_mare_ratio <= 5
    assert result.normal_mere_ratio <= 1.5
    assert result.normal_passed and result.passed


def test_no_native_retains_two_way_check():
    result = compare_tensors(torch.full((16,), 1.1), torch.ones(16, dtype=torch.float64))
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None
    assert not result.normal_passed and not result.passed


@pytest.mark.parametrize("candidate_value,native_value", [
    (1.0, 1.0),  # Overall pass, without the normal-region fallback.
    (1.03125, 1.03125),  # Normal-region fallback passes.
    (1.25, 1.03125),  # Normal-region fallback fails.
    (1.25, None),  # No native reference.
])
def test_serialized_results_omit_legacy_normal_counts(candidate_value, native_value):
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker

    candidate = torch.full((16,), candidate_value)
    golden = torch.ones(16, dtype=torch.float64)
    native = torch.full_like(candidate, native_value) if native_value is not None else None
    result = compare_tensors(candidate, golden, native_output=native)
    checked = RelativeErrorChecker().check(
        candidate, golden, "float32", 2**-13, native_outputs=native)
    removed_fields = {
        "normal_error_count", "normal_cpu_error_count",
        "normal_count_passed", "normal_count_gate_enabled",
    }

    def assert_clean(value):
        if isinstance(value, dict):
            assert removed_fields.isdisjoint(value)
            for item in value.values():
                assert_clean(item)
        elif isinstance(value, list):
            for item in value:
                assert_clean(item)

    assert_clean(result.to_dict())
    assert_clean(checked.to_dict())
    assert "normal_mare_ratio" in result.to_dict()
    assert "normal_mere_ratio" in checked.output_results[0].to_dict()
    expected_count = 0 if candidate_value == 1.0 else 16
    assert result.normal_total_count == expected_count
    assert checked.metadata["normal_total_count"] == expected_count
    assert checked.output_results[0].to_dict()["normal_total_count"] == expected_count


def test_multi_output_ratios_and_failure_category():
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker
    from kernel_eval.base.result import FAILURE_TYPE_PRECISION_MISMATCH
    ones = torch.ones(16)
    result = RelativeErrorChecker().check(
        [ones, ones * 1.25], [ones.double(), ones.double()], "float32", 2**-13,
        native_outputs=[ones, ones * 1.03125])
    assert not result.passed
    assert result.metadata["failure_type"] == FAILURE_TYPE_PRECISION_MISMATCH
    first, second = result.output_results
    assert first.passed and first.metadata["normal_mare_ratio"] is None
    assert not second.passed and second.metadata["normal_mare_ratio"] == pytest.approx(8)


def test_overall_pass_still_returns_before_normal_fallback(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("overall pass must not enter the normal-region fallback")

    monkeypatch.setattr(comparison, "_normal_error_ratio", unexpected)
    candidate = torch.full((100,), 1.0001)
    result = _compare(candidate, torch.ones_like(candidate))
    assert result.mere < 2**-13 and result.mare < 10 * 2**-13
    assert result.passed
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_normal_threshold_pass_skips_native_reduction_and_ratios(monkeypatch, dtype):
    from kernel_eval.utils.thresholds import get_threshold

    threshold = get_threshold(str(dtype).replace("torch.", ""))
    golden = torch.tensor([1.0, 1e-8, 1e-8], dtype=torch.float64)
    candidate = torch.tensor([1 + threshold, 1e-5, 1e-5], dtype=dtype)
    native = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    original_stats = comparison._MaskedStats
    reductions = []

    def record_stats(values, mask):
        stats = original_stats(values, mask)
        reductions.append(stats)
        return stats

    def unexpected(*args, **kwargs):
        pytest.fail("normal threshold pass must skip the three-way ratios")

    monkeypatch.setattr(comparison, "_MaskedStats", record_stats)
    monkeypatch.setattr(comparison, "_normal_error_ratio", unexpected)
    # Tiny values force the overall gate to fall back. The normal error meets
    # its dtype threshold, even though its MERE ratio against a perfect native
    # would exceed 1.5 for float16/bfloat16/float32.
    result = compare_tensors(candidate, golden, native_output=native)
    assert result.passed and result.normal_passed
    assert result.normal_total_count == 1
    assert 0 < result.normal_mare < threshold
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None
    assert result.output_results[0].metadata["threshold"] == threshold
    # Overall error, absolute diff, and candidate normal error only.
    assert len(reductions) == 3


def test_normal_threshold_check_honors_custom_dtype_threshold():
    golden = torch.tensor([1.0, 1e-8, 1e-8], dtype=torch.float64)
    candidate = torch.tensor([1 + 2**-13, 1e-5, 1e-5])
    native = torch.tensor([1.0, 0.0, 0.0])
    result = compare_tensors(candidate, golden, native_output=native,
                             custom_thresholds={"float32": 2**-15})
    assert not result.normal_passed and not result.passed
    assert result.normal_mare_ratio == pytest.approx(8)
    assert result.normal_mere_ratio == pytest.approx(8)


@pytest.mark.parametrize("boundary", ["mere", "mare"])
def test_normal_threshold_equality_still_uses_three_way_ratios(boundary):
    normal = torch.ones(16)
    normal[:16 if boundary == "mere" else 1] += 0.03125
    candidate = torch.cat([normal, torch.full((2,), 1e-5)])
    native = torch.cat([normal, torch.zeros(2)])
    golden = torch.cat([torch.ones(16), torch.full((2,), 1e-8)]).double()
    error = (normal.double() - 1).abs() / (1 + 1e-7)
    threshold = float(error.mean()) if boundary == "mere" else float(error.max()) / 10
    if boundary == "mare":
        assert float(error.max()) == 10 * threshold
    result = compare_tensors(candidate, golden, native_output=native,
                             custom_thresholds={"float32": threshold})
    assert result.normal_mare_ratio == pytest.approx(1)
    assert result.normal_mere_ratio == pytest.approx(1)
    assert result.normal_passed and result.passed


@pytest.mark.parametrize("region", ["small_value", "cancel"])
def test_normal_threshold_pass_does_not_bypass_other_region_failures(region):
    special = 0.0 if region == "small_value" else 1e-3
    golden = torch.tensor([1.0, special, special, special], dtype=torch.float64)
    candidate = torch.tensor([1 + 2**-13] + [1e-5 if region == "small_value" else 0.0] * 3)
    result = compare_tensors(candidate, golden, native_output=golden.float())
    assert result.normal_passed and not result.passed
    assert result.normal_total_count == 1
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None
    assert not getattr(result, region + "_passed")


def test_normal_statistics_match_naive_three_way_formulas():
    torch.manual_seed(19)
    golden = torch.ones(1027, dtype=torch.float64)
    native = torch.ones(1027) + torch.rand(1027) * 0.02
    candidate = torch.ones(1027) + torch.rand(1027) * 0.1
    result = compare_tensors(candidate, golden, native_output=native)
    error_n = (candidate.double() - golden).abs() / (golden.abs() + 1e-7)
    error_b = (native.double() - golden).abs() / (golden.abs() + 1e-7)
    assert result.normal_mare_ratio == pytest.approx(float(error_n.max() / error_b.max()))
    assert result.normal_mere_ratio == pytest.approx(float(error_n.mean() / error_b.mean()))


def test_empty_normal_region_does_not_fail():
    # Force the fallback while all values belong to the small-value region.
    golden = torch.full((16,), 1e-8, dtype=torch.float64)
    result = compare_tensors(torch.zeros(16), golden, native_output=torch.zeros(16))
    assert result.normal_passed
    assert result.normal_total_count == 0
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("fallback", [False, True])
def test_normal_total_count_only_runs_in_fallback(dtype, fallback):
    from kernel_eval.utils.thresholds import (
        get_cancel_boundary, get_cancel_zero_threshold, get_small_value_threshold,
    )

    target_dtype = getattr(torch, dtype)
    small = get_small_value_threshold(dtype)
    boundary = get_cancel_boundary(dtype)
    gold = torch.tensor([
        0, small / 2, small, -small, boundary / 2,
        boundary, -boundary, 1, float("nan"), float("inf"), -float("inf"),
    ], dtype=target_dtype)
    candidate = gold.clone()
    if fallback:
        candidate[7] = 2
    golden = gold.double()
    valid = torch.isfinite(golden) & torch.isfinite(candidate)
    small_mask = golden.abs() < small
    cancel_mask = ((golden.abs() >= small) & (golden.abs() < boundary)
                   & (candidate.double().abs() < get_cancel_zero_threshold(dtype)))
    expected = int((valid & ~small_mask & ~cancel_mask).sum()) if fallback else 0
    result = compare_tensors(candidate, golden, dtype, native_output=gold)
    assert result.normal_total_count == expected
    assert result.output_results[0].metadata["normal_total_count"] == expected
    assert (result.normal_mare_ratio is not None) == fallback


def test_normal_total_count_aggregates_only_compared_floating_outputs():
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker

    outputs = [torch.ones(3), torch.tensor([0.0, 1.25]), torch.ones(7),
               torch.ones(4, dtype=torch.int64)]
    goldens = [torch.ones(3).double(), torch.tensor([0.0, 1.0]).double(),
               torch.ones(7).double(), torch.ones(4, dtype=torch.int64)]
    native = [torch.ones(3), torch.tensor([0.0, 1.03125]), torch.ones(7),
              torch.ones(4, dtype=torch.int64)]
    result = RelativeErrorChecker().check(
        outputs, goldens, "float32", 2**-13, native_outputs=native, ignore_indices=[2])
    assert result.metadata["normal_total_count"] == 1
    assert result.output_results[0].metadata["normal_total_count"] == 0
    assert result.output_results[1].metadata["normal_total_count"] == 1
    assert result.output_results[3].metadata["normal_total_count"] == 0


def test_normal_total_count_excludes_matching_special_values_after_saturation():
    candidate = torch.tensor([float("inf"), 1, float("inf"), float("nan")], dtype=torch.float16)
    golden = torch.tensor([60000, 1, float("inf"), float("nan")], dtype=torch.float64)
    result = compare_tensors(candidate, golden, "float16")
    # One-sided Inf is saturated and compared as a finite normal value.
    assert result.normal_total_count == 2


@pytest.mark.parametrize("values", [[], [float("nan"), float("inf")]])
def test_normal_total_count_empty_or_all_invalid_is_zero(values):
    output = torch.tensor(values, dtype=torch.float32)
    result = compare_tensors(output, output.double())
    assert result.passed and result.normal_total_count == 0


def test_normal_native_reduction_allocation_is_bounded(monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Watch(TorchDispatchMode):
        largest = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for obj in tree_flatten(result)[0]:
                if isinstance(obj, torch.Tensor):
                    storage = obj.untyped_storage()
                    if storage.data_ptr() not in inputs:
                        self.largest = max(self.largest, storage.nbytes())
            return result

    original = comparison._MaskedStats
    watches = []

    def bounded_stats(values, mask):
        # Reductions: overall error, diff, candidate normal, then optional native normal.
        # Watch every reduction so this test does not depend on call order.
        nonlocal inputs
        inputs = {values.untyped_storage().data_ptr(), mask.untyped_storage().data_ptr()}
        watch = Watch()
        with watch:
            result = original(values, mask, chunk=256)
        watches.append(watch.largest)
        return result

    inputs = set()
    monkeypatch.setattr(comparison, "_MaskedStats", bounded_stats)
    result = _compare(torch.full((8192,), 1.25), torch.full((8192,), 1.03125))
    assert not result.passed
    assert len(watches) == 4
    assert max(watches) <= 256 * 8


@pytest.mark.parametrize("dtype,floor", [("float16", 2**-11), ("bfloat16", 2**-8),
                                         ("float32", 2**-14)])
def test_native_zero_error_uses_dtype_noise_floor(dtype, floor):
    candidate = torch.full((16,), 1 + 4 * floor, dtype=getattr(torch, dtype))
    native = torch.ones_like(candidate)
    result = compare_tensors(candidate, native.double(), native_output=native)
    assert result.normal_mare_ratio == pytest.approx(4 / (1 + 1e-7))
    assert result.normal_mere_ratio == pytest.approx(4 / (1 + 1e-7))
    assert not result.passed


def test_perfect_normal_region_skips_ratios_when_overall_fallback_is_needed():
    # Two inaccurate tiny values force the old overall gate to fall back.
    golden = torch.tensor([1.0, 1e-8, 1e-8], dtype=torch.float64)
    candidate = torch.tensor([1.0, 1e-5, 1e-5])
    result = compare_tensors(candidate, golden, native_output=torch.tensor([1.0, 0.0, 0.0]))
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None
    assert result.normal_passed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_strict_custom_threshold_tightens_ratio_floor_without_changing_regions(dtype):
    from kernel_eval.utils.thresholds import get_small_value_threshold, get_threshold

    dtype_name = str(dtype).replace("torch.", "")
    default = get_threshold(dtype_name)
    small = get_small_value_threshold(dtype_name)
    effective = default / 128
    # For fp32 this is the review's 1.00005 example. Low-precision outputs need
    # a representable deviation, but use the same strict/default threshold ratio.
    deviation = 5e-5 if dtype in (torch.float32, torch.float64) else default
    golden = torch.tensor([1.0, small / 2, small], dtype=torch.float64)
    candidate = golden.to(dtype).clone()
    candidate[0] = 1 + deviation
    native = golden.to(dtype)
    result = compare_tensors(candidate, golden, native_output=native,
                             custom_thresholds={dtype_name: effective})
    assert not result.passed and not result.normal_passed
    assert result.mare > 10 * effective
    # The small/cancel partition still uses the dtype's original small threshold.
    assert result.small_value_total_count == 1
    assert result.cancel_total_count == 1
    assert result.normal_total_count == 1
    floor = small * effective / default
    assert result.normal_mare_ratio == pytest.approx(result.normal_mare / floor)
    assert result.normal_mere_ratio == pytest.approx(result.normal_mare / floor)
    assert result.normal_mere_ratio > 1.5
    assert result.failure_type is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_looser_custom_threshold_does_not_increase_ratio_floor():
    native = torch.ones(16)
    result = compare_tensors(torch.full((16,), 1.125), native.double(), native_output=native,
                             custom_thresholds={"float32": 0.01})
    assert not result.passed
    expected = (0.125 / (1 + 1e-7)) / (2**-14)
    assert result.normal_mare_ratio == pytest.approx(expected)
    assert result.normal_mere_ratio == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["mare", "mere"])
@pytest.mark.parametrize("repeats", [1, 2, 3, 10, 100])
@pytest.mark.parametrize("chunk", [None, 32])
def test_exact_ratio_endpoints_pass_with_replication_and_chunking(monkeypatch, kind, repeats, chunk):
    if chunk is not None:
        original = comparison._MaskedStats
        monkeypatch.setattr(comparison, "_MaskedStats",
                            lambda values, mask: original(values, mask, chunk=chunk))
    candidate = torch.ones(100)
    if kind == "mare":
        candidate[0] = 1.15625  # 5 * native deviation of 0.03125
        limit = 5.0
    else:
        candidate[:] = 1.046875  # 1.5 * native deviation
        limit = 1.5
    candidate = candidate.repeat(repeats)
    result = _compare(candidate, torch.full_like(candidate, 1.03125))
    assert getattr(result, "normal_" + kind + "_ratio") == pytest.approx(limit, rel=1e-12)
    assert result.normal_passed and result.passed


@pytest.mark.parametrize("kind", ["mare", "mere"])
def test_ratio_just_outside_endpoint_still_fails(kind):
    candidate = torch.ones(100)
    if kind == "mare":
        candidate[0] = torch.nextafter(torch.tensor(1.15625), torch.tensor(2.0))
    else:
        candidate[:] = torch.nextafter(torch.tensor(1.046875), torch.tensor(2.0))
    result = _compare(candidate, torch.full_like(candidate, 1.03125))
    assert not result.normal_passed and not result.passed


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("position", [0, 9])
def test_nonfinite_native_invalidates_reference_and_serializes_as_null(monkeypatch, dtype, value, position):
    from kernel_eval.base.result import FAILURE_TYPE_REFERENCE_INVALID, get_accuracy_failure_type
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker

    original = comparison._MaskedStats
    monkeypatch.setattr(comparison, "_MaskedStats",
                        lambda values, mask: original(values, mask, chunk=4))
    native = torch.ones(10, dtype=dtype)
    native[position] = value
    candidate = torch.full_like(native, 100)
    golden = torch.ones(10, dtype=torch.float64)
    result = compare_tensors(candidate, golden, native_output=native)
    assert not result.normal_passed and not result.passed
    assert result.failure_type == FAILURE_TYPE_REFERENCE_INVALID
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None
    assert "同精度参考失效" in result.error_msg
    assert result.output_results[0].metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    json.dumps(result.to_dict(), allow_nan=False)
    checked = RelativeErrorChecker().check(candidate, golden, "float32", 2**-13, native_outputs=native)
    assert not checked.passed
    assert get_accuracy_failure_type(checked) == FAILURE_TYPE_REFERENCE_INVALID
    serialized = json.loads(json.dumps(checked.to_dict(), allow_nan=False))
    for fields in (serialized["metadata"], serialized["output_results"][0]):
        assert fields["normal_mare_ratio"] is None and fields["normal_mere_ratio"] is None


def test_finite_native_with_overflowing_error_statistics_is_invalid():
    from kernel_eval.base.result import FAILURE_TYPE_REFERENCE_INVALID

    result = _compare(torch.full((4,), 100.0), torch.full((4,), 1e308, dtype=torch.float64))
    assert not result.passed and result.failure_type == FAILURE_TYPE_REFERENCE_INVALID
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_multi_output_reference_invalid_is_preserved_alongside_precision_failure():
    from kernel_eval.base.result import FAILURE_TYPE_PRECISION_MISMATCH, FAILURE_TYPE_REFERENCE_INVALID
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker

    outputs = [torch.full((4,), 100.0), torch.full((4,), 100.0)]
    goldens = [torch.ones(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64)]
    native = [torch.ones(4), torch.tensor([1.0, 1.0, math.nan, 1.0])]
    checked = RelativeErrorChecker().check(outputs, goldens, "float32", 2**-13, native_outputs=native)
    assert not checked.passed
    assert checked.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    first, second = checked.output_results
    assert not first.passed and not second.passed
    assert first.metadata.get("failure_type", FAILURE_TYPE_PRECISION_MISMATCH) == FAILURE_TYPE_PRECISION_MISMATCH
    assert second.metadata["failure_type"] == FAILURE_TYPE_REFERENCE_INVALID
    assert second.metadata["normal_mare_ratio"] is None
    assert first.metadata["normal_mare_ratio"] is not None
    json.dumps(checked.to_dict(), allow_nan=False)


def test_nonfinite_native_does_not_disable_normal_threshold_short_circuit():
    golden = torch.tensor([1.0, 1e-8, 1e-8], dtype=torch.float64)
    candidate = torch.tensor([1 + 2**-13, 1e-5, 1e-5])
    native = torch.tensor([math.inf, 0.0, 0.0])
    result = compare_tensors(candidate, golden, native_output=native)
    assert result.passed and result.normal_passed
    assert result.normal_total_count == 1
    assert result.failure_type is None
    assert result.normal_mare_ratio is None and result.normal_mere_ratio is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_cpu_evaluator_and_final_report_preserve_invalid_reference(tmp_path, monkeypatch, value):
    from kernel_eval.base.result import FAILURE_TYPE_REFERENCE_INVALID
    from kernel_eval.benches.cann_spec import CannCaseSpec
    from kernel_eval.config import Config
    from kernel_eval.eval.evaluator import Evaluator
    from kernel_eval.eval.results import EvalCaseResult, EvalOperatorResult, summarize_case_results
    from kernel_eval.report.report_generator import ReportGenerator

    task = tmp_path / "tasks" / "level1" / "normal_probe"
    task.mkdir(parents=True)
    (task / "golden.py").write_text(
        "import torch\n"
        "def normal_probe(x: torch.Tensor):\n    return torch.ones_like(x)\n"
        "def normal_probe_bench(x: torch.Tensor):\n"
        "    output = torch.ones_like(x)\n"
        f"    output[0] = float({str(value)!r})\n"
        "    return output\n"
    )
    (task / "proto.yaml").write_text(
        "operator:\n  name: NormalProbe\n  difficulty: L1\n"
        "  inputs:\n  - name: x\n    dtype: [float32]\n"
        "  outputs:\n  - name: y\n    dtype: [float32]\n"
        "  schema: normal_probe(Tensor x) -> Tensor y\n"
    )
    (task / "cases.yaml").write_text("cases: []\n")
    config = Config(device_type="cpu", tasks_root=str(tmp_path / "tasks"),
                    reports_dir=str(tmp_path / "reports"), torch_op_guard_mode="off",
                    perf_freq_boost=False)
    case = CannCaseSpec(case_id="level1/normal_probe_1", case_num=1, operator="NormalProbe",
                        rel_path="level1/normal_probe", input_shapes=[[10]], dtypes=["float32"],
                        value_ranges=[[1, 2]], attrs={})
    result = Evaluator(config).evaluate_case(case, ai_op_func=lambda x: torch.full_like(x, 100))
    assert result.ai_run_result is not None, result.error_msg
    assert result.ai_run_result.success, result.error_msg
    assert not result.success and result.failure_type == FAILURE_TYPE_REFERENCE_INVALID
    assert "同精度参考失效" in result.error_msg
    result = EvalCaseResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False)))
    counts = summarize_case_results([result])
    assert counts.reference_invalid == 1 and counts.failed == 0
    operator = EvalOperatorResult(case.rel_path, case.operator, 1, 0, 0, 0, [result], 0.0, 0.0)
    monkeypatch.setenv("CANN_BENCH_SCORE_WEIGHTS", "2")
    monkeypatch.setattr("kernel_eval.report.report_generator.collect_setup_info", lambda config: {})
    generator = ReportGenerator(config.reports_dir, config=config)
    generator.add_operator_result(operator)
    report = generator.generate()
    assert report.overall_score is None
    assert report.summary["reference_invalid_cases"] == 1
    assert report.operators[0].cases[0].status == FAILURE_TYPE_REFERENCE_INVALID
    json.dumps(report.to_dict(), allow_nan=False)
    path = generator.save_json(report)
    saved = json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(f"invalid JSON number: {value}"))
    output = saved["operators"][0]["cases"][0]["accuracy"]["output_results"][0]
    assert output["normal_mare_ratio"] is None and output["normal_mere_ratio"] is None


@pytest.mark.parametrize("candidate,native", [(1.0, math.inf), (math.inf, 1.0), (math.inf, math.inf),
                                             (math.nan, 1.0), (1.0, math.nan), (1.0, 0.0)])
def test_unavailable_ratio_is_null_and_never_passes(candidate, native):
    ratio = comparison._normal_error_ratio(candidate, native, 5e-324)
    assert ratio is None
    assert not comparison._normal_ratio_passed(ratio, 5.0)
    assert "不可用" in comparison._normal_failure_message(ratio, ratio)
    assert json.dumps({"ratio": ratio}, allow_nan=False) == '{"ratio": null}'
