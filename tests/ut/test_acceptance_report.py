# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel_eval.acceptance_report import write_acceptance_report


def _make_run(root: Path) -> None:
    operator = "demo_op"
    candidate = root / operator / "kernel_eval" / "ascend_arena_demo_op_run"
    (candidate / "results").mkdir(parents=True)
    (root / operator / "kernel_eval" / "init_results").mkdir(parents=True)
    (root / "e2e_summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-01-01T00:00:00+00:00",
                "operators": [operator],
                "idle_devices": [0],
                "busy_devices": [],
                "num_processes": 1,
                "success": True,
                "results": [
                    {
                        "operator": operator,
                        "device_id": 0,
                        "success": True,
                        "compile": True,
                        "correctness": True,
                        "performance": True,
                        "baseline_returncode": 0,
                        "candidate_returncode": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status = {"build_success": True, "profile_success": True, "profile_cases": 1}
    (root / operator / "kernel_eval" / "init_results" / "status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    (candidate / "results" / "status.json").write_text(json.dumps(status), encoding="utf-8")
    for stage in ("smoke", "complete", "profile"):
        (candidate / "results" / f"{stage}-correctness_result.json").write_text(
            json.dumps({"success": True, "case_results": [{"passed": True}]}), encoding="utf-8"
        )
    (candidate / "comparison.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case-1",
                        "baseline_us": 2.0,
                        "candidate_us": 1.0,
                        "speedup": 2.0,
                        "candidate_passed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (candidate / "ascend_arena_demo_op_run.json").write_text(
        json.dumps(
            {
                "summary": {"average_speedup": 2.0},
                "setup_info": {
                    "environment": {
                        "npu": "Ascend910B4 × 1",
                        "python": "3.10.20",
                        "cann": "9.0.0",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_acceptance_report_is_portable_and_copies_screenshot(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    screenshots = tmp_path / "screenshots"
    screenshots.mkdir()
    (screenshots / "summary.png").write_bytes(b"png")

    output = write_acceptance_report(
        run_root,
        command="python tests/run_ascend_arena_batch_e2e.py",
        screenshots_dir=screenshots,
    )
    report = output.read_text(encoding="utf-8")
    data = json.loads((output.parent / "report_data.json").read_text(encoding="utf-8"))

    assert "**PASS**" in report
    assert "demo_op" in report
    assert "2.0000x" in report
    assert "screenshots/summary.png" in report
    assert (output.parent / "screenshots" / "summary.png").is_file()
    assert (output.parent / "evidence" / "demo_op" / "comparison.json").is_file()
    assert data["evidence"]["summary"] == "evidence/e2e_summary.json"
    assert data["portable"] is True
    assert data["run_root"] == "run"
    assert not str(run_root) in json.dumps(data)
    assert data["environment"]["paths_are_host_local"] is True
    assert data["operators"][0]["candidate_status"] == "evidence/demo_op/candidate_status.json"
    assert not str(run_root) in report


def test_acceptance_report_marks_empty_run_incomplete(tmp_path):
    run_root = tmp_path / "empty"
    run_root.mkdir()
    output = write_acceptance_report(run_root)
    assert "**INCOMPLETE**" in output.read_text(encoding="utf-8")


def test_acceptance_report_accepts_single_operator_work_root(tmp_path):
    run_root = tmp_path / "operator" / "work"
    _make_run(run_root)
    # _make_run models a batch child; move its child contents into the direct
    # workspace layout used by a single operator report.
    child = run_root / "demo_op"
    for entry in child.iterdir():
        entry.rename(run_root / entry.name)
    child.rmdir()
    (run_root / "e2e_summary.json").unlink()
    output = write_acceptance_report(run_root)
    report = output.read_text(encoding="utf-8")
    assert "`demo_op`" in report
    assert "**PASS**" in report


def test_acceptance_report_keeps_failure_logs(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    candidate = run_root / "demo_op" / "kernel_eval" / "ascend_arena_demo_op_run"
    (candidate / "results" / "compile.log").write_text("candidate compile failure\n", encoding="utf-8")
    (candidate / "results" / "profile.log").write_text("candidate profile failure\n", encoding="utf-8")
    (run_root / "e2e_summary.json").write_text(
        json.dumps({"operators": ["demo_op"], "results": [{"operator": "demo_op"}]}),
        encoding="utf-8",
    )
    output = write_acceptance_report(run_root)
    data = json.loads((output.parent / "report_data.json").read_text(encoding="utf-8"))
    item = data["operators"][0]
    assert data["result"] == "PASS"
    assert item["status"] == "PASS"
    assert item["candidate_status"] == "evidence/demo_op/candidate_status.json"
    assert (output.parent / "evidence" / "demo_op" / "candidate_compile.log").is_file()
    assert (output.parent / "evidence" / "demo_op" / "candidate_profile.log").is_file()


def test_acceptance_report_marks_missing_candidate_incomplete(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    candidate = run_root / "demo_op" / "kernel_eval" / "ascend_arena_demo_op_run"
    (candidate / "results" / "status.json").unlink()
    for stage in ("smoke", "complete", "profile"):
        (candidate / "results" / f"{stage}-correctness_result.json").unlink()
    (run_root / "e2e_summary.json").write_text(
        json.dumps({"operators": ["demo_op"], "results": [{"operator": "demo_op"}]}),
        encoding="utf-8",
    )
    output = write_acceptance_report(run_root)
    data = json.loads((output.parent / "report_data.json").read_text(encoding="utf-8"))
    assert data["result"] == "INCOMPLETE"
    assert data["operators"][0]["status"] == "INCOMPLETE"


def test_summary_success_without_evidence_is_not_pass(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "e2e_summary.json").write_text(
        json.dumps({
            "operators": ["demo_op"],
            "results": [{
                "operator": "demo_op",
                "compile": True,
                "correctness": True,
                "performance": True,
                "success": True,
            }],
        }),
        encoding="utf-8",
    )
    data = json.loads((write_acceptance_report(run_root).parent / "report_data.json").read_text(encoding="utf-8"))
    assert data["result"] == "INCOMPLETE"
    assert data["operators"][0]["status"] == "INCOMPLETE"
    assert data["operators"][0]["evidence_complete"] is False


def test_missing_comparison_is_incomplete_until_profile_finishes(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    candidate = run_root / "demo_op" / "kernel_eval" / "ascend_arena_demo_op_run"
    (candidate / "comparison.json").unlink()
    data = json.loads((write_acceptance_report(run_root).parent / "report_data.json").read_text(encoding="utf-8"))
    assert data["result"] == "INCOMPLETE"
    assert data["operators"][0]["performance"] is None


def test_comparison_requires_positive_timing_and_passed_case(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    candidate = run_root / "demo_op" / "kernel_eval" / "ascend_arena_demo_op_run"
    (candidate / "comparison.json").write_text(
        json.dumps({
            "cases": [
                {"speedup": 2.0, "candidate_passed": None, "baseline_us": 2.0, "candidate_us": 1.0},
                {"speedup": 3.0, "candidate_passed": True, "baseline_us": 0.0, "candidate_us": 1.0},
            ]
        }),
        encoding="utf-8",
    )
    data = json.loads((write_acceptance_report(run_root).parent / "report_data.json").read_text(encoding="utf-8"))
    assert data["operators"][0]["comparison"]["valid_cases"] == 0
    assert data["operators"][0]["status"] == "FAIL"


def test_report_is_repository_feature_matrix_without_operator_expansion(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    output = write_acceptance_report(run_root)
    report = output.read_text(encoding="utf-8")
    data = json.loads((output.parent / "report_data.json").read_text(encoding="utf-8"))

    assert "仓库功能完整性测试报告" in report
    assert "README 功能验收矩阵" in report
    assert "## 5. 算子详情" not in report
    assert "case-1" not in report
    assert len(data["features"]) >= 16
    assert data["feature_result"] == "INCOMPLETE"
    assert data["operator_result"] == "PASS"


def test_feature_results_override_and_copy_evidence(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    (run_root / "feature-proof.txt").write_text("verified\n", encoding="utf-8")
    (run_root / "feature_results.json").write_text(
        json.dumps(
            {
                "features": {
                    "workspace_only": {
                        "status": "PASS",
                        "actual": "workspace-only completed",
                        "evidence": ["feature-proof.txt"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    output = write_acceptance_report(run_root)
    data = json.loads((output.parent / "report_data.json").read_text(encoding="utf-8"))
    feature = next(item for item in data["features"] if item["id"] == "workspace_only")

    assert feature["status"] == "PASS"
    assert feature["evidence"] == ["evidence/features/workspace_only/feature-proof.txt"]
    assert data["evidence"]["feature_results"] == "evidence/feature_results.json"
    assert (output.parent / "evidence" / "feature_results.json").is_file()
    assert (output.parent / "evidence" / "features" / "workspace_only" / "feature-proof.txt").is_file()


def test_feature_pass_without_actual_evidence_is_incomplete(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    (run_root / "feature_results.json").write_text(
        json.dumps({"features": {"docker_mode": {"status": "PASS"}}}),
        encoding="utf-8",
    )
    data = json.loads((write_acceptance_report(run_root).parent / "report_data.json").read_text(encoding="utf-8"))
    feature = next(item for item in data["features"] if item["id"] == "docker_mode")
    assert feature["status"] == "INCOMPLETE"


@pytest.mark.parametrize("failed_stage", ["baseline", "candidate", "returncode"])
def test_explicit_failure_overrides_successful_measurements(tmp_path, failed_stage):
    run_root = tmp_path / "run"
    _make_run(run_root)
    candidate = run_root / "demo_op/kernel_eval/ascend_arena_demo_op_run"
    if failed_stage == "returncode":
        summary = json.loads((run_root / "e2e_summary.json").read_text())
        summary["results"][0]["candidate_returncode"] = 1
        (run_root / "e2e_summary.json").write_text(json.dumps(summary))
    else:
        status_path = (candidate / "results/status.json" if failed_stage == "candidate"
                       else run_root / "demo_op/kernel_eval/init_results/status.json")
        status = json.loads(status_path.read_text())
        status["success"] = False
        status_path.write_text(json.dumps(status))
    data = json.loads((write_acceptance_report(run_root).parent / "report_data.json").read_text())
    assert data["operator_result"] == "FAIL"
    assert data["operators"][0]["success"] is False


@pytest.mark.parametrize("changes", [{"speedup": 0}, {"speedup": -2}, {"timing_comparable": False}])
def test_invalid_comparison_is_excluded(tmp_path, changes):
    run_root = tmp_path / "run"
    _make_run(run_root)
    path = run_root / "demo_op/kernel_eval/ascend_arena_demo_op_run/comparison.json"
    comparison = json.loads(path.read_text())
    comparison["cases"][0].update(changes)
    path.write_text(json.dumps(comparison))
    data = json.loads((write_acceptance_report(run_root).parent / "report_data.json").read_text())
    assert data["operators"][0]["comparison"]["valid_cases"] == 0
    assert data["operator_result"] != "PASS"


def test_missing_selected_operator_prevents_complete_pass(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    path = run_root / "e2e_summary.json"
    summary = json.loads(path.read_text())
    summary["operators"].append("missing_op")
    path.write_text(json.dumps(summary))
    data = json.loads((write_acceptance_report(run_root).parent / "report_data.json").read_text())
    assert data["operator_result"] == "INCOMPLETE"
    assert [item["operator"] for item in data["operators"]] == ["demo_op", "missing_op"]


@pytest.mark.parametrize("revision,dirty,status", [("pinned", "", "PASS"),
                                                   ("other", "", "FAIL"),
                                                   ("pinned", " M source.cpp", "FAIL")])
def test_skeleton_submodule_verification(tmp_path, monkeypatch, revision, dirty, status):
    import kernel_eval.acceptance_report as reports

    manifest = tmp_path / "ops-skeleton.json"
    manifest.write_text(json.dumps({"commit": "pinned", "distribution": "submodule"}))
    checkout = tmp_path / "ops-skeleton"
    checkout.mkdir()
    (checkout / ".git").touch()
    monkeypatch.setattr(reports, "_skeleton_manifest_path", lambda root: manifest)
    monkeypatch.setattr(reports, "_run_text", lambda command, **kw: revision if command[1] == "rev-parse" else dirty)
    skeleton = reports._skeleton_metadata(tmp_path)
    actual, _ = reports._infer_feature_status("skeleton_archive", run_root=tmp_path, environment={},
                                              skeleton=skeleton, batch={}, operators=[])
    assert actual == status
    assert skeleton["archive_sha256"] == ""


def test_workspace_baseline_stage_evidence_is_copied(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    baseline = run_root / "demo_op/kernel_eval/init_results"
    for stage in ("smoke", "complete", "profile"):
        (baseline / f"{stage}-correctness_result.json").write_text(json.dumps({"success": True}))
    report = write_acceptance_report(run_root)
    for stage in ("smoke", "complete", "profile"):
        assert (report.parent / f"evidence/demo_op/baseline_{stage}-correctness_result.json").is_file()


def test_explicit_not_applicable_feature_keeps_its_status(tmp_path):
    run_root = tmp_path / "run"
    _make_run(run_root)
    (run_root / "feature_results.json").write_text(json.dumps({"features": {
        "docker_mode": {"status": "N/A", "note": "Host evaluation"},
        "custom_check": {"status": "N/A", "note": "Outside the declared scope"},
    }}))
    data = json.loads((write_acceptance_report(run_root).parent / "report_data.json").read_text())
    for name in ["docker_mode", "custom_check"]:
        assert next(item for item in data["features"] if item["id"] == name)["status"] == "N/A"
