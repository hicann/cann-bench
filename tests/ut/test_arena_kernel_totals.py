# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import csv
import json
from pathlib import Path
from statistics import median

import pytest

from ascend_arena_e2e.ascend_arena.runners.kernel_timing import measurement_protocol, parse_kernel_totals, parse_kernel_batches
from ascend_arena_e2e.ascend_arena.runners.profile import ProfileRunner
from ascend_arena_e2e.ascend_arena.runners.benchmark import BenchmarkRunner
from kernel_eval.optimize import _build_report, _duration, _noise_band, _timing_scope, build_parser


def kernel_csv(path, rounds):
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Name", "Type", "Duration(us)", "Start Time(us)", "Stream ID"])
        writer.writeheader()
        start = 0
        for launches in rounds:
            writer.writerow({"Name": "clean", "Type": "_Z_CannBenchCacheClean", "Duration(us)": 2,
                             "Start Time(us)": start, "Stream ID": 1})
            start += 3
            for name, duration in launches:
                writer.writerow({"Name": name, "Type": name, "Duration(us)": duration,
                                 "Start Time(us)": start, "Stream ID": 1})
                start += duration + 1


def test_round_totals_keep_multiple_launches_and_correlated_kernel_times(tmp_path):
    path = tmp_path / "kernel_details.csv"
    rounds = [[("A", 5), ("A", 5), ("B", 30)]] * 3
    kernel_csv(path, rounds)
    result = parse_kernel_totals(path, warmup=10, repeat=3)
    assert result["task_duration_us"] == 40
    assert result["duration_samples_us"] == [40, 40, 40]
    assert all(len(r["launches"]) == 3 for r in result["rounds"])

    rounds = [[("A", 1), ("B", 100)], [("A", 1), ("B", 1)], [("A", 100), ("B", 1)]]
    kernel_csv(path, rounds)
    result = parse_kernel_totals(path, warmup=10, repeat=3)
    assert result["task_duration_us"] == 101
    assert result["task_duration_us"] != median([1, 1, 100]) + median([100, 1, 1])


def test_missing_or_wrong_boundaries_fail_instead_of_using_task_ids(tmp_path):
    path = tmp_path / "kernel_details.csv"
    kernel_csv(path, [[("A", 5)]])
    with pytest.raises(ValueError, match="boundary count"):
        parse_kernel_totals(path, warmup=10, repeat=30)
    path.write_text(path.read_text().replace("_Z_CannBenchCacheClean", "unrelated"))
    with pytest.raises(ValueError, match="boundary count"):
        parse_kernel_totals(path, warmup=10, repeat=1)


def test_level0_names_identify_boundaries_when_types_are_unavailable(tmp_path):
    path = tmp_path / "kernel_details.csv"
    kernel_csv(path, [[("A", 5), ("A", 5), ("B", 30)]] * 3)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    for row in rows:
        if "CannBenchCacheClean" in row["Type"]:
            row["Name"] = "_Z19CannBenchCacheCleanIDhEvPhS0_llj"
        row["Type"] = "N/A"
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = parse_kernel_totals(path, warmup=10, repeat=3)
    assert result["duration_samples_us"] == [40, 40, 40]
    assert all([k["name"] for k in r["launches"]] == ["A", "A", "B"] for r in result["rounds"])


def test_empty_invocation_and_invalid_duration_are_rejected(tmp_path):
    path = tmp_path / "kernel_details.csv"
    kernel_csv(path, [[]])
    with pytest.raises(ValueError, match="no target"):
        parse_kernel_totals(path, warmup=10, repeat=1)
    kernel_csv(path, [[("A", -1)]])
    with pytest.raises(ValueError, match="invalid launch"):
        parse_kernel_totals(path, warmup=10, repeat=1)


def metric(scope, value, protocol=None):
    row = {"duration_scope": scope, "task_duration_us": value, "noise_band": .01,
           "sample_count": 30, "duration_samples_us": [value] * 30}
    if protocol is not None:
        row["measurement_protocol"] = protocol
    return row


def test_primary_kernel_metric_and_secondary_call_latency_are_both_reported(tmp_path):
    protocol = measurement_protocol(10, 30)
    before = [metric("operator_call", 320, protocol), metric("kernel_total", 40, protocol)]
    after = [metric("operator_call", 300, protocol), metric("kernel_total", 20, protocol)]
    assert _duration(before) == 40
    assert _timing_scope(before) == "kernel_total"
    assert _noise_band(before) == .01
    _build_report(output_dir=tmp_path, run_name="test", operator="test", baseline={"case_01": before},
                  candidate={"case_01": after}, baseline_status={"success": True},
                  candidate_status={"success": True}, device_id=7)
    report = json.loads((tmp_path / "test.json").read_text())
    case = report["cases"][0]
    assert case["speedup"] == 2
    assert case["baseline_call_us"] == 320
    assert case["call_speedup"] == pytest.approx(320 / 300)
    for suffix in ("md", "html"):
        assert "完整调用延迟" in (tmp_path / f"test.{suffix}").read_text()


@pytest.mark.parametrize("old_scope", ["operator_call", "kernel_total"])
def test_old_baseline_or_changed_protocol_is_not_comparable(tmp_path, old_scope):
    _build_report(output_dir=tmp_path, run_name="test", operator="test",
                  baseline={"case_01": [metric(old_scope, 40)]},
                  candidate={"case_01": [metric("kernel_total", 20, measurement_protocol(10, 30))]},
                  baseline_status={"success": True}, candidate_status={"success": True}, device_id=7)
    case = json.loads((tmp_path / "test.json").read_text())["cases"][0]
    assert case["speedup"] is None
    assert case["timing_comparable"] is False


def test_default_sampling_and_complete_pair_requirement():
    runner = ProfileRunner(profiler="torch-profiler")
    benchmark = BenchmarkRunner(dependency_root=Path("."))
    assert (benchmark.profile_runner.warmup, benchmark.profile_runner.repeat) == (10, 30)
    args = build_parser().parse_args([])
    assert (runner.warmup, runner.repeat) == (args.warmup, args.repeat) == (10, 30)
    assert not runner._has_expected_profile_samples([metric("operator_call", 300)])
    assert runner._has_expected_profile_samples([metric("kernel_total", 10), metric("operator_call", 300)])


def test_different_helper_binaries_are_not_comparable(tmp_path):
    baseline_protocol = measurement_protocol(10, 30, helper_build={"binary_sha256": "old"})
    candidate_protocol = measurement_protocol(10, 30, helper_build={"binary_sha256": "new"})
    _build_report(output_dir=tmp_path, run_name="test", operator="test",
                  baseline={"case_01": [metric("kernel_total", 40, baseline_protocol)]},
                  candidate={"case_01": [metric("kernel_total", 20, candidate_protocol)]},
                  baseline_status={"success": True}, candidate_status={"success": True}, device_id=7)
    case = json.loads((tmp_path / "test.json").read_text())["cases"][0]
    assert case["speedup"] is None
    assert case["timing_comparable"] is False


def test_different_allocator_configs_are_not_comparable(tmp_path):
    before = measurement_protocol(10, 30, runtime_config={"PYTORCH_NPU_ALLOC_CONF": None})
    after = measurement_protocol(10, 30, runtime_config={"PYTORCH_NPU_ALLOC_CONF": "page_size:1g"})
    _build_report(output_dir=tmp_path, run_name="test", operator="test",
                  baseline={"case_01": [metric("kernel_total", 40, before)]},
                  candidate={"case_01": [metric("kernel_total", 20, after)]},
                  baseline_status={"success": True}, candidate_status={"success": True}, device_id=7)
    case = json.loads((tmp_path / "test.json").read_text())["cases"][0]
    assert case["speedup"] is None and case["timing_comparable"] is False


def test_batch_case_boundaries_skip_each_cases_warmup_and_keep_launch_sums(tmp_path):
    path = tmp_path / "kernel_details.csv"
    kernel_csv(path, [[("A", 999)]] * 10 + [[("A", 5), ("A", 5), ("B", 30)]] * 30
               + [[("C", 888)]] * 10 + [[("C", 7)]] * 30)
    result = parse_kernel_batches(path, case_ids=["case_01", "case_02"], warmup=10, repeat=30)
    assert result['case_01']['duration_samples_us'] == [40] * 30
    assert result['case_02']['duration_samples_us'] == [7] * 30
    assert all(len(r['launches']) == 3 for r in result['case_01']['rounds'])
    with pytest.raises(ValueError, match="boundary count"):
        parse_kernel_batches(path, case_ids=["case_01"], warmup=10, repeat=30)
    with pytest.raises(ValueError, match="unique"):
        parse_kernel_batches(path, case_ids=["case_01", "case_01"], warmup=10, repeat=30)
    path.write_text(path.read_text().replace(',5,', ',50000,', 1))
    with pytest.raises(ValueError, match="crosses"):
        parse_kernel_batches(path, case_ids=["case_01", "case_02"], warmup=10, repeat=30)


def test_invalid_profile_parse_is_reported_as_failure_with_raw_error(tmp_path):
    from concurrent.futures import Future
    from types import SimpleNamespace
    import subprocess
    from ascend_arena_e2e.ascend_arena.runners.profile import _PendingProfileParse
    future = Future()
    future.set_exception(ValueError("cache-clean boundary count 399 != expected 400"))
    pending = [_PendingProfileParse("case_01", tmp_path, subprocess.CompletedProcess("profile", 0, "", ""), future)]
    metrics, logs = {}, []
    passed = ProfileRunner(profiler="torch-profiler")._drain_completed_profile_parses(
        build_result=SimpleNamespace(op_name="test"), pending=pending, metrics_by_case=metrics,
        log_sections=logs, success=True, wait=True)
    assert passed is False and not metrics and not pending
    assert "boundary count 399" in logs[0]


@pytest.mark.parametrize('invalid_boundaries', [False, True])
def test_profile_run_uses_batch_collection_and_records_parse_failures(tmp_path, monkeypatch, invalid_boundaries):
    from contextlib import contextmanager
    from types import SimpleNamespace
    runner = ProfileRunner(profiler='torch-profiler')
    cases = [{'case_id': 'case_01'}, {'case_id': 'case_02'}]
    monkeypatch.setattr(runner, '_load_profile_cases', lambda directory: cases)

    @contextmanager
    def runtime(**kwargs):
        yield {}

    def collect(*, output_dir, cases, runtime, batch):
        assert batch is True and len(cases) == 2
        for case in cases:
            directory = output_dir / case['case_id']
            directory.mkdir()
            (directory / 'precision_verified.txt').write_text('ok\n')

    def extract(directory):
        if invalid_boundaries:
            raise ValueError('cache-clean boundary count 399 != expected 400')
        return [metric('kernel_total', 10), metric('operator_call', 100)]

    monkeypatch.setattr(runner, '_torch_profiler_runtime_context', runtime)
    monkeypatch.setattr(runner, '_execute_torch_profiler_cases', collect)
    monkeypatch.setattr(runner, '_extract_metrics', extract)
    build = SimpleNamespace(work_dir=tmp_path / 'work', standard_op_root=tmp_path,
                            success=True, extension_path=tmp_path / 'binding.so', op_name='test')
    result = runner.run(standard_dir=tmp_path, build_result=build,
                        correctness_result=SimpleNamespace(success=True), device_id=7)
    assert result.success is (not invalid_boundaries)
    assert len(result.metrics) == (0 if invalid_boundaries else 2)
    if invalid_boundaries:
        assert 'boundary count 399' in result.log_path.read_text()


def test_changed_case_shape_cannot_reuse_baseline_even_with_same_case_id(tmp_path):
    original = {'case_id': 'case_01', 'inputs': {'x': {'shape': [512, 1024], 'dtype': 'float32'}},
                'attrs': {'dim': {'value': -1}}}
    scaled = json.loads(json.dumps(original))
    scaled['inputs']['x']['shape'][0] = 4096
    before = measurement_protocol(10, 30, case=original)
    after = measurement_protocol(10, 30, case=scaled)
    assert before['case_spec_sha256'] != after['case_spec_sha256']
    _build_report(output_dir=tmp_path, run_name='shapes', operator='test',
                  baseline={'case_01': [metric('kernel_total', 10, before)]},
                  candidate={'case_01': [metric('kernel_total', 50, after)]},
                  baseline_status={'success': True}, candidate_status={'success': True}, device_id=7)
    case = json.loads((tmp_path / 'shapes.json').read_text())['cases'][0]
    assert case['timing_comparable'] is False and case['speedup'] is None
    reordered = dict(attrs=original['attrs'], inputs=original['inputs'], case_id='case_01',
                     meta={'description': 'Same workload'}, roles=['perf-anchored'])
    assert measurement_protocol(10, 30, case=reordered) == before
