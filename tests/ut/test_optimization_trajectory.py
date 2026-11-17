# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import json
from pathlib import Path

import pytest

from kernel_eval.optimization_trajectory import (
    _trace_events,
    build_trajectory_data,
    main,
    write_optimization_trajectory_report,
)


@pytest.mark.parametrize("phase,build,success,rc,expected", [
    ("candidate", False, False, 1, "compile_failed"),
    ("candidate", True, False, 1, "evaluation_failed"),
    ("candidate", True, True, 0, "success"),
    ("init", True, True, 0, "success"),
])
def test_trace_evaluation_status(tmp_path, phase, build, success, rc, expected):
    root = tmp_path / "kernel_eval"
    run = root / "run_1"
    result = run if phase == "init" else run / "results"
    result.mkdir(parents=True)
    (result / "status.json").write_text(json.dumps({"build_success": build, "success": success}))
    if phase != "init":
        (run / "run_1.json").write_text(json.dumps({
            "source_diff": {}, "cases": [], "summary": {"average_speedup": 1.25},
        }))
    start = {"id": "eval", "kind": "evaluation", "event": "start", "started_at": "2026-09-17T10:00:00+08:00"}
    end = dict(start, event="end", run_name="run_1", phase=phase, returncode=rc,
               ended_at="2026-09-17T10:00:03+08:00", duration_seconds=3)
    (root / "timing.jsonl").write_text("\n".join(map(json.dumps, [start, end, dict(end, id="repeat")])) + '\n{"id":')
    events = _trace_events(root)
    assert len(events) == 2  # Repeated evaluations remain visible without source changes.
    assert events[0]["status"] == expected
    assert events[0]["speedup"] == (1.25 if success and phase != "init" else None)


def test_trace_incomplete_and_agent_events(tmp_path):
    root = tmp_path / "kernel_eval"
    root.mkdir()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (root / "timing.jsonl").write_text(json.dumps({"id": "e", "kind": "evaluation", "event": "start"}))
    (artifact / "timing.jsonl").write_text(json.dumps({"id": "a", "kind": "agent", "event": "end", "duration_seconds": 10}))
    events = _trace_events(root)
    assert {item["kind"] for item in events} == {"evaluation", "agent"}
    assert all(item["status"] == "incomplete" and item["speedup"] is None for item in events)


@pytest.mark.parametrize("interrupted,expected", [(False, "compile_failed"), (True, "incomplete")])
def test_trace_comparison_baseline_failure_and_interruption(tmp_path, interrupted, expected):
    run = tmp_path / "comparison"
    baseline = run / "baseline_results"
    baseline.mkdir(parents=True)
    (baseline / "status.json").write_text(json.dumps({
        "build_success": False, "success": False, "interrupted": interrupted,
    }))
    (tmp_path / "timing.jsonl").write_text(json.dumps({
        "id": "e", "kind": "evaluation", "event": "end", "phase": "comparison",
        "run_name": run.name, "returncode": 1,
    }))
    assert _trace_events(tmp_path)[0]["status"] == expected


def _write_round(
    reports_root: Path,
    *,
    name: str,
    timestamp: str,
    diff: str,
    changed_file_count: int,
    cases: list[tuple[str, float, float]],
    success: bool,
) -> None:
    run_dir = reports_root / name
    run_dir.mkdir(parents=True)
    report = {
        "eval_code": name,
        "timestamp": timestamp,
        "operator": "test_op",
        "candidate_status": {"success": success},
        "baseline_status": {"success": success},
        "cases": [
            {"case_id": case_id, "baseline_us": baseline, "candidate_us": candidate}
            for case_id, baseline, candidate in cases
        ],
        "source_diff": {
            "changed_file_count": changed_file_count,
            "changed_files": (
                [{"path": "op_kernel/test.cpp", "status": "modified"}]
                if changed_file_count else []
            ),
            "unified_diff": diff,
        },
    }
    (run_dir / f"{name}.json").write_text(json.dumps(report), encoding="utf-8")


def test_trajectory_filters_unchanged_code_and_uses_initialized_baseline(tmp_path):
    init_results = tmp_path / "init_results"
    init_results.mkdir()
    (init_results / "status.json").write_text(json.dumps({"success": True}), encoding="utf-8")
    (init_results / "result.json").write_text(
        json.dumps(
            {
                "case_1": [{"task_duration_us": 100.0}],
                "case_2": [{"task_duration_us": 200.0}],
            }
        ),
        encoding="utf-8",
    )
    _write_round(
        tmp_path,
        name="run_0",
        timestamp="2026-08-24T10:00:00",
        diff="# No implementation source changes detected.\n",
        changed_file_count=0,
        cases=[("case_1", 100.0, 100.0), ("case_2", 200.0, 200.0)],
        success=True,
    )
    metadata_run = tmp_path / "run_metadata"
    metadata_run.mkdir()
    (metadata_run / "run_metadata.json").write_text(
        json.dumps(
            {
                "eval_code": "run_metadata",
                "timestamp": "2026-08-24T10:00:30",
                "operator": "test_op",
                "candidate_status": {"success": False},
                "baseline_status": {"success": True},
                "cases": [],
                "source_diff": {
                    "changed_file_count": 1,
                    "changed_files": [{"path": ".source", "status": "deleted"}],
                    "unified_diff": "--- a/.source\n+++ b/.source\n-old metadata\n",
                },
            }
        ),
        encoding="utf-8",
    )
    diff_a = "--- a/op_kernel/test.cpp\n+++ b/op_kernel/test.cpp\n@@ -1 +1 @@\n-old\n+a\n"
    _write_round(
        tmp_path,
        name="run_1",
        timestamp="2026-08-24T10:01:00",
        diff=diff_a,
        changed_file_count=1,
        cases=[("case_1", 110.0, 50.0), ("case_2", 190.0, 100.0)],
        success=True,
    )
    _write_round(
        tmp_path,
        name="run_2",
        timestamp="2026-08-24T10:02:00",
        diff=diff_a,
        changed_file_count=1,
        cases=[("case_1", 90.0, 0.0), ("case_2", 210.0, 0.0)],
        success=False,
    )
    diff_b = "--- a/op_kernel/test.cpp\n+++ b/op_kernel/test.cpp\n@@ -1 +1 @@\n-old\n+b\n"
    _write_round(
        tmp_path,
        name="run_3",
        timestamp="2026-08-24T10:03:00",
        diff=diff_b,
        changed_file_count=1,
        cases=[],
        success=False,
    )

    data = build_trajectory_data(tmp_path)

    assert data["generated_from_runs"] == 5
    assert [item["run_name"] for item in data["rounds"]] == ["run_1", "run_3"]
    assert data["rounds"][0]["speedup"] == pytest.approx(2.0)
    assert data["rounds"][1]["success"] is False
    assert data["rounds"][1]["speedup"] is None
    assert [item["candidate_us"] for item in data["rounds"][1]["cases"]] == [None, None]
    assert [item["baseline_us"] for item in data["rounds"][1]["cases"]] == [100.0, 200.0]
    assert [item["baseline_us"] for item in data["baselines"]] == [100.0, 200.0]
    assert all(item["run_name"] == "init_results" for item in data["baselines"])


def test_trajectory_splits_legacy_diff_and_writes_standalone_html(tmp_path):
    diff = (
        "--- a/op_host/tiling.cpp\n+++ b/op_host/tiling.cpp\n@@ -1 +1 @@\n-old host\n+new host\n"
        "--- a/op_kernel/kernel.cpp\n+++ b/op_kernel/kernel.cpp\n@@ -1 +1 @@\n-old kernel\n+new kernel\n"
    )
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    report = {
        "eval_code": "run_1",
        "timestamp": "2026-08-24T10:00:00",
        "operator": "test_op",
        "candidate_status": {"success": True},
        "baseline_status": {"success": True},
        "cases": [{"case_id": "case_1", "baseline_us": 10.0, "candidate_us": 5.0}],
        "source_diff": {
            "changed_file_count": 2,
            "changed_files": [
                {"path": "op_host/tiling.cpp", "status": "modified"},
                {"path": "op_kernel/kernel.cpp", "status": "modified"},
            ],
            "unified_diff": diff,
        },
    }
    (run_dir / "run_1.json").write_text(json.dumps(report), encoding="utf-8")

    data = build_trajectory_data(tmp_path)
    file_diffs = data["rounds"][0]["file_diffs"]
    assert [item["path"] for item in file_diffs] == ["op_host/tiling.cpp", "op_kernel/kernel.cpp"]
    assert "+new host" in file_diffs[0]["diff"]
    assert "+new kernel" in file_diffs[1]["diff"]

    output = write_optimization_trajectory_report(tmp_path)
    html = output.read_text(encoding="utf-8")
    assert output == tmp_path / "optimization_trajectory.html"
    assert "<style>" in html
    assert '<script id="trajectory-data" type="application/json">' in html
    assert "https://" not in html
    assert "run_1" in html
    assert "op_kernel/kernel.cpp" in html
    assert "观测轮次" not in html
    assert 'id="case-table"' not in html
    assert "value.toFixed(2)" in html
    assert "--- 优化前：" in html
    assert "+++ 优化后：" in html
    assert "FAILED · 0x" in html
    assert "失败 · 0x" in html


def test_main_writes_report(tmp_path, capsys):
    assert main([str(tmp_path)]) == 0
    output = tmp_path / "optimization_trajectory.html"
    assert output.is_file()
    assert capsys.readouterr().out.strip() == str(output)


@pytest.mark.parametrize("first_success", [True, False])
def test_scored_validation_completes_the_same_source_round(tmp_path, first_success):
    reports_root = tmp_path / "kernel_eval"
    reports_root.mkdir()
    baseline = reports_root / "init_results"
    baseline.mkdir()
    (baseline / "status.json").write_text(json.dumps({"success": True}))
    (baseline / "result.json").write_text(json.dumps({"case_01": [{"task_duration_us": 100}]}))
    events = []
    for index, (name, cases, success) in enumerate([
        ("correctness", [], first_success),
        ("full", [("case_01", 100, 50)], True),
        ("failed", [], False),
    ]):
        timestamp = f"2026-11-05T10:0{index}:00"
        _write_round(reports_root, name=name, timestamp=timestamp, diff="same implementation",
                     changed_file_count=1, cases=cases, success=success)
        result = reports_root / name / "results"
        result.mkdir()
        (result / "status.json").write_text(json.dumps({"build_success": True, "success": success}))
        events.append({"id": name, "kind": "evaluation", "event": "end", "phase": "candidate",
                       "run_name": name, "started_at": timestamp, "returncode": 0 if success else 1})
    (reports_root / "timing.jsonl").write_text("\n".join(map(json.dumps, events)))
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "timing.jsonl").write_text(json.dumps({"id": "agent", "kind": "agent",
        "event": "end", "started_at": "2026-11-05T10:03:00", "returncode": 124, "timed_out": True}))

    data = build_trajectory_data(reports_root)

    assert len(data["rounds"]) == 1
    assert data["rounds"][0]["run_name"] == "full"
    assert data["rounds"][0]["success"] is True
    assert data["rounds"][0]["speedup"] == pytest.approx(2)
    assert len([e for e in data["trace"] if e["kind"] == "evaluation"]) == 3
    assert data["trace"][-1]["returncode"] == 124
