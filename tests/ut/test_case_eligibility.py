# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Hardware exclusions are execution selection, not failed candidate cases."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from kernel_eval import cli, staged_eval
from kernel_eval.benches.cann_loader import CannCaseLoader, CannTaskLoader
from kernel_eval.config import Config
from kernel_eval.eval.evaluator import Evaluator
from kernel_eval.eval.failure_synthesizer import FailureSynthesizer
from kernel_eval.eval.results import EvalCaseResult
from kernel_eval.utils.case_eligibility import (
    HardwareDetectionError,
    detect_execution_hardware,
    filter_cases_for_hardware,
    supports_case,
)


@pytest.fixture
def eligibility_bench(tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    for name, restrictions in (
        ("Mixed", [None, ["950pr", "950dt"], ["950dt"]]),
        ("DtOnly", [["950dt"]]),
    ):
        directory = tmp_path / name.lower()
        directory.mkdir()
        (directory / "proto.yaml").write_text(
            yaml.safe_dump(
                {
                    "operator": {
                        "name": name,
                        "difficulty": "L4",
                        "schema": f"{name.lower()}(Tensor x) -> Tensor",
                    }
                }
            ),
            encoding="utf-8",
        )
        (directory / "golden.py").write_text("", encoding="utf-8")
        cases = [
            {
                "operator": name,
                "case_id": index,
                "input_shape": [[2]],
                "dtype": ["float32"],
                "attrs": {"supported_devices": allowed} if allowed else {},
            }
            for index, allowed in enumerate(restrictions, 1)
        ]
        (directory / "cases.yaml").write_text(
            yaml.safe_dump({"cases": cases}), encoding="utf-8"
        )
    return tmp_path


@pytest.mark.parametrize(
    "hardware, expected",
    [(None, 4), ("950pr", 2), ("Ascend950DT_9582", 4), ("910b2", 1)],
)
def test_loader_filters_before_statistics_and_preserves_unmarked(
    eligibility_bench, hardware, expected
):
    loader = CannCaseLoader(str(eligibility_bench), target_hardware=hardware)
    assert len(loader.scan_all()) == expected
    assert loader.get_statistics()["total"] == expected
    assert supports_case({"supported_devices": ["Ascend950DT_9582"]}, "950dt")


def test_normal_and_staged_paths_select_same_eligible_cases(
    eligibility_bench, monkeypatch
):
    args = cli.create_parser().parse_args(["eval"])
    normal = cli._load_eval_cases(args, str(eligibility_bench), "", "950pr")
    monkeypatch.setattr(staged_eval, "detect_execution_hardware", lambda *args: "950pr")
    staged_args = staged_eval.create_parser().parse_args([])
    correct = staged_eval._load_cases(
        staged_args, str(eligibility_bench), None, filter_prefix=""
    )
    allowed = {staged_eval._case_spec_key(case) for case in correct}
    perf = staged_eval._load_cases(
        staged_args, str(eligibility_bench), None, filter_prefix="", allowlist=allowed
    )
    assert (
        [case.case_id for case in normal]
        == [case.case_id for case in correct]
        == [case.case_id for case in perf]
    )
    assert len(normal) == 2
    assert (
        staged_eval._load_cases(
            staged_args, str(eligibility_bench), [], filter_prefix=""
        )
        == []
    )


def test_loader_and_failure_results_preserve_baseline_source(eligibility_bench):
    from kernel_eval.eval.subprocess_utils import _synthesize_failure_cases

    metadata = eligibility_bench / "metadata"
    (metadata / "910b2.json").write_text(
        json.dumps(
            {
                "mixed": {"1": {"baseline_perf_us": 10, "t_hw_us": 1}},
            }
        ),
        encoding="utf-8",
    )
    loader = CannCaseLoader(str(eligibility_bench), target_hardware="950dt")
    case = loader.scan_by_operator("Mixed")[0]
    sources = case.metadata["baseline_source"]
    assert sources["baseline_perf_us"]["source_hardware"] == "910b2"
    restored = type(case).from_dict(case.to_dict())
    assert restored.metadata["baseline_source"] == sources
    failed = _synthesize_failure_cases([restored], "timeout", "timeout")[0]
    assert failed.baseline_source == sources
    spec = next(
        spec
        for spec in CannTaskLoader(str(eligibility_bench)).list_tasks()
        if spec.name == "Mixed"
    )
    compiled = FailureSynthesizer(loader).synthesize_compile_failure(
        spec, "build failed"
    )
    assert compiled.results[0].baseline_source == sources


def test_compile_synthesis_drops_only_proven_hardware_exclusions(eligibility_bench):
    loader = CannCaseLoader(str(eligibility_bench), target_hardware="950pr")
    synthesizer = FailureSynthesizer(loader)
    specs = {
        spec.name: spec for spec in CannTaskLoader(str(eligibility_bench)).list_tasks()
    }
    assert (
        synthesizer.synthesize_compile_failure(specs["DtOnly"], "build failed") is None
    )
    failure = synthesizer.synthesize_compile_failure(specs["Mixed"], "build failed")
    assert failure.total_cases == 2
    assert len(failure.results) == 2

    def filter_func(cases, selected):
        return [case for case in cases if case.case_num == selected["case_id"]]

    assert (
        synthesizer.synthesize_compile_failure(
            specs["Mixed"], "build failed", {"case_id": 3}, filter_func
        )
        is None
    )
    matcher = SimpleNamespace(
        find_operator_info=lambda name: specs.get(name),
        find_operator_info_by_snake=lambda name: specs.get(name),
    )
    assert (
        synthesizer.synthesize_all_compile_failures(
            matcher, SimpleNamespace(compile_errors={"DtOnly": "build failed"})
        )
        == []
    )
    assert (
        synthesizer.synthesize_all_compile_failures(
            matcher,
            SimpleNamespace(compile_errors={"<build>": "build failed"}),
            operator_filter=["DtOnly"],
        )
        == []
    )
    failures = synthesizer.synthesize_all_compile_failures(
        matcher, SimpleNamespace(compile_errors={"<build>": "build failed"})
    )
    assert len(failures) == 1 and failures[0].compilation_error


def test_staged_all_excluded_skips_build_and_workers(
    eligibility_bench, monkeypatch, tmp_path
):
    args = staged_eval.create_parser().parse_args(
        [
            "--task-dir",
            str(eligibility_bench),
            "--operator",
            "DtOnly",
            "--source-dir",
            "candidate",
            "--reports-dir",
            str(tmp_path / "reports"),
        ]
    )
    monkeypatch.setattr(staged_eval, "detect_execution_hardware", lambda *args: "950pr")
    monkeypatch.setattr(
        staged_eval, "_compile", lambda *args: pytest.fail("must not build")
    )
    monkeypatch.setattr(
        staged_eval,
        "_evaluate_cases",
        lambda *args, **kwargs: pytest.fail("must not schedule"),
    )
    reports = []
    monkeypatch.setattr(
        staged_eval, "_save_report", lambda *args, **kwargs: reports.append(args[2])
    )
    assert staged_eval.run(args) == 0
    assert reports == [[]]


def test_cli_all_excluded_skips_build_and_workers(eligibility_bench, monkeypatch):
    args = cli.create_parser().parse_args(
        ["eval", "--operator", "DtOnly", "--source-dir", "candidate"]
    )
    monkeypatch.setattr(cli, "detect_execution_hardware", lambda *args: "950pr")
    monkeypatch.setattr(
        "kernel_eval.eval.evaluator.Evaluator",
        lambda *args, **kwargs: pytest.fail("must not build"),
    )
    report = Mock()
    assert cli._cmd_eval_npu(args, str(eligibility_bench), "", Config(), report) == 0
    report.add_operator_result.assert_not_called()


def test_json_run_cases_filters_before_counting_or_calling_candidate(eligibility_bench):
    evaluator = object.__new__(Evaluator)
    evaluator.target_hardware = "950pr"
    evaluator.config = Config(perf_batch_cases=False)
    evaluator.operator_matcher = Mock()
    evaluator.incremental_output_path = None
    evaluator.device_manager = Mock()
    evaluator._cleanup_memory = Mock()
    seen = []

    def evaluate(case):
        seen.append(case.case_num)
        return EvalCaseResult(
            case_id=case.case_id,
            operator=case.operator,
            case_num=case.case_num,
            rel_path=case.rel_path,
            success=True,
        )

    evaluator.evaluate_case = evaluate
    specs = CannCaseLoader(str(eligibility_bench)).scan_by_operator("Mixed")
    # eval-child deserializes this same transport shape before calling run_cases.
    specs = [type(case).from_dict(case.to_dict()) for case in specs]
    result = evaluator.run_cases(specs, "Mixed", "mixed")
    assert seen == [1, 2]
    assert result.total_cases == result.passed_cases == 2


def test_direct_evaluator_excluded_selection_never_prepares_packages(eligibility_bench):
    evaluator = object.__new__(Evaluator)
    evaluator.case_loader = CannCaseLoader(
        str(eligibility_bench), target_hardware="950pr"
    )
    evaluator.package_manager = Mock()
    assert evaluator.evaluate_from_source("candidate", ["DtOnly"]).operators == []
    assert evaluator.evaluate_skip_build(["Mixed"], {"case_id": 3}).operators == []
    evaluator.package_manager.prepare_from_source.assert_not_called()
    evaluator.package_manager.prepare_skip_build.assert_not_called()


@pytest.mark.parametrize(
    "names", [["unknown"], [" unknown "], [""], ["Ascend950PR", "Ascend950DT"]]
)
def test_unknown_and_mixed_npu_are_environment_errors(monkeypatch, names):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device_count=lambda: len(names), get_device_name=lambda index: names[index]
        ),
        raising=False,
    )
    with pytest.raises(HardwareDetectionError):
        detect_execution_hardware("npu")
    assert detect_execution_hardware("cpu") is None


@pytest.mark.parametrize("name", ["custom-npu-model", "custom-device"])
def test_unmapped_custom_device_name_preserves_existing_platforms(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(device_count=lambda: 1, get_device_name=lambda index: name),
        raising=False,
    )
    hardware = detect_execution_hardware("npu")
    assert hardware == name
    assert supports_case({}, hardware)
    assert supports_case({"supported_devices": [name]}, hardware)
    assert not supports_case({"supported_devices": ["950pr"]}, hardware)


def test_detected_a3_product_names_select_only_legacy_gmm_cases(monkeypatch):
    names = ["Ascend910_9391", "Ascend910_9362", "Ascend910_9362B"]
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device_count=lambda: len(names), get_device_name=lambda index: names[index]
        ),
        raising=False,
    )
    hardware = detect_execution_hardware("npu")
    assert hardware == "910c"
    case_path = (
        Path(__file__).resolve().parents[2]
        / "bench_lab"
        / "kernel_bench"
        / "level4"
        / "grouped_mat_mul_allto_allv"
        / "cases.yaml"
    )
    cases = yaml.safe_load(case_path.read_text(encoding="utf-8"))["cases"]
    assert [
        case["case_id"] for case in filter_cases_for_hardware(cases, hardware)
    ] == list(range(1, 21))


def test_visible_hardware_aliases_resolve_without_physical_zero_assumptions(
    monkeypatch,
):
    names = ["Ascend950DT_9582", "950dt"]
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            device_count=lambda: len(names), get_device_name=lambda index: names[index]
        ),
        raising=False,
    )
    assert detect_execution_hardware("npu", 1) == "950dt"


def test_baseline_raw_yaml_filters_before_spawning(eligibility_bench, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "collect_baseline.py"
    spec = importlib.util.spec_from_file_location(
        "eligibility_collect_baseline", script
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    collector = object.__new__(module.BaselineCollector)
    collector.bench_root = eligibility_bench
    collector.target_hardware = "950pr"
    collector.device_manager = SimpleNamespace(get_device=lambda: "npu:0")
    monkeypatch.setattr(module.ref_registry, "get_ref", lambda path: object())
    seen = []
    collector._collect_one_case_in_subprocess = lambda path, cid, case: seen.append(
        cid
    ) or {"elapsed_us": 1.0}
    results = collector.collect_op("mixed")
    assert seen == [1, 2]
    assert len(results) == 2
    assert collector.collect_op("dtonly") == []
