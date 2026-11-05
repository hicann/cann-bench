# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MC2 lifecycle, scoring and metadata regression tests; no NPU required."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kernel_eval.eval.results import (
    EvalCaseResult,
    EvalOperatorResult,
    summarize_case_results,
)
from kernel_eval.eval.perf_eval import PerfResult
from kernel_eval.eval.op_runner import OpRunResult
from kernel_eval.staged_eval import _merge_results
from kernel_eval.utils.baseline_resolver import resolve_hardware

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "mc2_resource_model", ROOT / "docs/hap_thw_model/compute_mc2_a5.py"
)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def case(success=True, **kwargs):
    return EvalCaseResult(
        case_id="level4/allto_all_matmul_1",
        rel_path="level4/allto_all_matmul",
        operator="AlltoAllMatmul",
        case_num=1,
        success=success,
        **kwargs,
    )


def operator(item):
    return EvalOperatorResult(
        rel_path=item.rel_path,
        operator=item.operator,
        total_cases=1,
        passed_cases=int(item.success),
        failed_cases=int(not item.success),
        skipped_cases=0,
        results=[item],
        pass_rate=float(item.success),
        avg_speedup=0,
    )


def test_final_report_keeps_measured_mc2_baseline_and_score():
    correctness = case(t_hw_us=10)
    performance = case(
        t_hw_us=10,
        baseline_perf_us=200,
        perf_result=PerfResult(elapsed_us=100),
        golden_run_result=OpRunResult(success=True, elapsed_us=200, device="npu"),
    )
    merged = _merge_results([operator(correctness)], [operator(performance)])[0]
    assert merged.avg_speedup == 2
    assert merged.results[0].get_perf_score() == pytest.approx(190 / 280)
    assert merged.results[0].golden_run_result.elapsed_us == 200


def test_runtime_failure_is_failed_not_skipped_without_accuracy():
    summary = summarize_case_results(
        [case(False, failure_type="compile_runtime_error")]
    )
    assert (summary.failed, summary.skipped) == (1, 0)


@pytest.mark.parametrize("performance_ops", [[], [operator(case())]])
def test_missing_timing_has_explicit_final_diagnostic(performance_ops):
    merged = _merge_results([operator(case())], performance_ops)[0].results[0]
    assert merged.success
    assert merged.perf_result is None
    assert merged.perf_recheck["status"] == "perf_unmeasured"


def test_rank_with_delayed_teardown_can_finish_within_case_budget(
    monkeypatch, tmp_path
):
    from kernel_eval.eval import mc2_distributed_runner as runner

    class Process:
        exitcode = None

        def start(self):
            pass

        def is_alive(self):
            return self.exitcode is None

        def join(self, timeout):
            if timeout >= 12:
                self.exitcode = 0

        def terminate(self):
            self.exitcode = -15

        def kill(self):
            self.exitcode = -9

    queue = Mock()
    queue.get.side_effect = [
        {"rank": r, "passed": True, "elapsed_us": 100, "golden_elapsed_us": 200}
        for r in range(2)
    ]
    context = SimpleNamespace(Queue=lambda: queue, Process=lambda **kw: Process())
    monkeypatch.setattr(runner.torch.multiprocessing, "get_context", lambda _: context)
    monkeypatch.setattr(runner, "get_golden_loader", lambda *a, **kw: Mock())
    config = SimpleNamespace(
        tasks_root=tmp_path,
        reports_dir=tmp_path,
        warmup=1,
        repeat=1,
        enable_profiler=True,
        timeout_per_operator=90,
    )
    spec_case = SimpleNamespace(
        attrs={"world_size": 2},
        rel_path="level4/allto_all_matmul",
        operator="AlltoAllMatmul",
        case_num=1,
        input_shapes=[],
        dtypes=["float16"],
        value_ranges=[],
        baseline_perf_us=0,
        t_hw_us=10,
        get_case_id_str=lambda: "level4/allto_all_matmul_1",
    )
    result = runner.MC2DistributedEvaluator(config).evaluate_case(spec_case)
    assert result.success, result.error_msg
    assert result.get_speedup() == 2


def test_all_a5_cases_have_reproducible_nonzero_anchors_and_three_way():
    import json

    all_cases = list(model.a5_cases())
    assert len(all_cases) == 300
    assert set(model.PLATFORMS) == {"950pr", "950dt"}
    assert {op for op, _ in all_cases} == model.OPS
    for hw in model.PLATFORMS:
        data = json.loads(
            (model.BENCH / "metadata" / f"{hw}.json").read_text(encoding="utf-8")
        )
        for op, item in all_cases:
            assert item["attrs"]["mc2_accuracy_mode"] == "three_way"
            assert item["attrs"]["world_size"] == 2
            if hw not in model.supported_platforms(item):
                assert str(item["case_id"]) not in data["level4"].get(op, {})
                continue
            stored = data["level4"][op][str(item["case_id"])]["t_hw_us"]
            assert stored > 0
            assert stored == model.compute(op, item, hw)["t_hw_us"]


def test_cube_bound_has_independent_worked_example():
    # 1024^3 GEMM -> 2,147,483,648 operations; 486 TFLOPS -> 4.418690... us.
    item = {
        "input_shape": [[1024, 1024], [1024, 1024]],
        "dtype": ["float16", "float16"],
        "attrs": {"world_size": 2},
    }
    assert model.compute("matmul_allto_all", item, "950dt")["t_hw_us"] == 4.418690


def test_published_mc2_metadata_has_valid_entries_for_supported_cases():
    import json
    import math
    import yaml
    from kernel_eval.utils.case_eligibility import supports_case

    operators = model.OPS | {"matmul_reduce_scatter", "moe_distribute_dispatch_v2"}
    cases_by_op = {}
    for op in operators:
        cases_path = model.BENCH / "level4" / op / "cases.yaml"
        cases = yaml.safe_load(cases_path.read_text(encoding="utf-8"))["cases"]
        cases_by_op[op] = {str(case["case_id"]): case for case in cases}
    for path in (model.BENCH / "metadata").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        for op, entries in data.get("level4", {}).items():
            if op not in operators:
                continue
            for case_id, entry in entries.items():
                assert case_id in cases_by_op[op], (path.name, op, case_id)
                assert supports_case(cases_by_op[op][case_id]["attrs"], path.stem), (
                    path.name,
                    op,
                    case_id,
                )
                for field in ("baseline_perf_us", "t_hw_us"):
                    value = entry.get(field)
                    assert (
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(value)
                        and value > 0
                    ), (path.name, op, case_id, field, value)


@pytest.mark.parametrize("hardware", ["Ascend950DT_9582", "Ascend950PR_9582"])
def test_all_a5_mc2_baselines_are_loadable(hardware):
    import json
    import math
    from kernel_eval.utils.baseline_store import BaselineStore

    platform = resolve_hardware(hardware)
    data = json.loads(
        (model.BENCH / "metadata" / f"{platform}.json").read_text(encoding="utf-8")
    )
    store = BaselineStore(model.BENCH, hardware=hardware)
    store.load()
    for op, case in model.a5_cases():
        if platform not in model.supported_platforms(case):
            continue
        case_id = case["case_id"]
        entry = data["level4"][op][str(case_id)]
        assert "baseline_perf_us" in entry
        expected = entry["baseline_perf_us"]
        assert math.isfinite(expected) and expected > 0
        assert expected >= entry["t_hw_us"]
        assert store.get_perf(f"level4/{op}", case_id) == expected


def test_resource_regeneration_preserves_published_baselines(monkeypatch, tmp_path):
    import json

    cases = list(model.a5_cases())
    monkeypatch.setattr(model, "a5_cases", lambda bench: iter(cases))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    original = {}
    for hw in model.PLATFORMS:
        contents = (model.BENCH / "metadata" / f"{hw}.json").read_text(encoding="utf-8")
        original[hw] = json.loads(contents)
        (metadata / f"{hw}.json").write_text(contents, encoding="utf-8")
    model.generate(write=True, bench=tmp_path)
    for hw, expected in original.items():
        actual = json.loads((metadata / f"{hw}.json").read_text(encoding="utf-8"))
        # Model provenance may be regenerated; timings and other metadata must stay unchanged.
        for data in (actual, expected):
            data["_metadata"].pop("mc2_hap", None)
        assert actual == expected


def test_dt_model_is_not_pr_or_a2_fallback():
    assert resolve_hardware("Ascend950DT_9582") == "950dt"
    assert resolve_hardware("Ascend950DT_9591") == "950dt"
    assert resolve_hardware("Ascend950PR_9582") == "950pr"


@pytest.mark.parametrize(
    "hardware",
    [
        "950dt",
        "Ascend950DT_9582",
        "Ascend950DT_9582B",
        "Ascend950DT_9591",
        "950dt_9582",
    ],
)
def test_real_baseline_store_loads_dt_anchors(hardware):
    from kernel_eval.utils.baseline_store import BaselineStore

    store = BaselineStore(model.BENCH, hardware=hardware)
    store.load()
    assert store.hardware == "950dt"
    for op, item in model.a5_cases():
        expected = model.compute(op, item, "950dt")["t_hw_us"]
        assert store.get_t_hw(f"level4/{op}", item["case_id"]) == expected


def test_final_markdown_distinguishes_accuracy_and_missing_performance(tmp_path):
    from kernel_eval.base.result import AccuracyResult
    from kernel_eval.report.report_generator import ReportGenerator
    from kernel_eval.config import Config

    correct = case(
        accuracy_result=AccuracyResult(
            passed=True, metadata={"checker_name": "mc2_three_way"}
        )
    )
    failed_perf = case(
        False, failure_type="compile_runtime_error", error_msg="rank shutdown timed out"
    )
    merged = _merge_results([operator(correct)], [operator(failed_perf)])[0]
    config = Config()
    config.enable_profiler = True
    config.reports_dir = tmp_path
    generator = ReportGenerator(config=config)
    generator.add_operator_result(merged)
    report = generator.generate()
    markdown = generator._generate_markdown_content(report)
    assert report.passed_cases == 1
    assert report.summary["performance_measured_cases"] == 0
    assert report.summary["performance_unmeasured_cases"] == 1
    assert "性能未完成" in markdown
    assert "三方精度通过" in markdown
    assert "rank shutdown timed out" in markdown
