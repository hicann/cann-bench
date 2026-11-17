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

import pytest

from kernel_eval.batch_report import build_batch_report_data, write_batch_report


def test_batch_report_embeds_operator_trajectory_without_child_html_links(tmp_path):
    run_root = tmp_path / "run"
    reports = run_root / "angle_v2" / "work" / "kernel_eval" / "round-1"
    reports.mkdir(parents=True)
    (run_root / "config.json").write_text(
        json.dumps(
            {
                "parameters": {"operators": ["angle_v2"], "agent": "codex", "model": "gpt-test"},
                "workspaces": [{"operator": "angle_v2", "workspace": str(run_root / "angle_v2" / "work")}],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "progress.json").write_text(
        json.dumps({"operators": {"angle_v2": {"status": "completed", "device_id": 2}}}),
        encoding="utf-8",
    )
    (reports / "round-1.json").write_text(
        json.dumps(
            {
                "operator": "angle_v2",
                "eval_code": "round-1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "candidate_status": {"success": True},
                "baseline_status": {"success": True},
                "cases": [{"case_id": "case-1", "baseline_us": 20, "candidate_us": 10}],
                "source_diff": {
                    "changed_file_count": 1,
                    "unified_diff": "diff --git a/op.cpp b/op.cpp\n+optimized\n",
                    "file_diffs": [{"path": "op.cpp", "status": "modified", "diff": "+optimized"}],
                },
            }
        ),
        encoding="utf-8",
    )
    second_round = reports.parent / "round-2"
    second_round.mkdir()
    (second_round / "round-2.json").write_text(
        json.dumps(
            {
                "operator": "angle_v2",
                "eval_code": "round-2",
                "timestamp": "2026-01-01T00:01:00+00:00",
                "candidate_status": {"success": True},
                "baseline_status": {"success": True},
                "cases": [{"case_id": "case-1", "baseline_us": 20, "candidate_us": 16}],
                "source_diff": {
                    "changed_file_count": 1,
                    "unified_diff": "diff --git a/op.cpp b/op.cpp\n+slower-change\n",
                    "file_diffs": [{"path": "op.cpp", "status": "modified", "diff": "+slower-change"}],
                },
            }
        ),
        encoding="utf-8",
    )

    (reports.parent / "timing.jsonl").write_text(json.dumps({
        "id": "trace-test", "kind": "evaluation", "event": "start",
        "started_at": "2026-01-01T00:00:00+00:00",
    }))
    data = build_batch_report_data(run_root)
    report = write_batch_report(run_root)
    html = report.read_text(encoding="utf-8")

    assert data["operators"][0]["best_speedup"] == 2.0
    assert data["operators"][0]["best_rounds"] == [
        {"round": 1, "speedup": 2.0, "success": True},
        {"round": 2, "speedup": 2.0, "success": True},
    ]
    assert data["operators"][0]["status"] == "completed"
    assert '<script id="trajectory-data"' in data["operators"][0]["trajectory_html"]
    assert '"id":"trace-test"' in data["operators"][0]["trajectory_html"]
    assert 'id="execution-trace"' in data["operators"][0]["trajectory_html"]
    assert "angle_v2" in html
    assert "+optimized" in html
    assert "frame.srcdoc=op.trajectory_html" in html
    assert "button.addEventListener('click',()=>renderTrajectory(op,index))" in html
    assert "el('trajectories').replaceChildren(section)" in html
    assert "overflow-wrap:anywhere" in html
    assert "series.forEach((op,index)=>" in html
    assert "label.textContent=op.name" in html
    assert "value.toFixed(2)" in html
    assert "Math.ceil(maxRound/8)" in html
    assert "class','hover-guide'" in html
    assert "hit.addEventListener('pointermove'" in html
    assert "title.textContent=`第 ${round} 轮`" in html
    assert "Number(item.round)<=round" in html
    assert "toFixed(2)}x`:'N/A'" in html
    assert "整体优化情况（最优性能）" in html
    assert "rounds:op.best_rounds" in html
    assert "frame.setAttribute('scrolling','no')" in html
    assert "doc.documentElement.style.overflow='hidden'" in html
    assert "if(ops.length)renderTrajectory" not in html
    assert "optimization_trajectory.html" not in html


def test_manual_report_marks_workspace_with_rounds_as_reported(tmp_path):
    run_root = tmp_path / "manual"
    reports = run_root / "logit" / "work" / "kernel_eval" / "round-1"
    reports.mkdir(parents=True)
    (run_root / "config.json").write_text(
        json.dumps(
            {
                "parameters": {"operators": ["logit"]},
                "workspaces": [{"operator": "logit", "workspace": "/moved/from/another/server"}],
            }
        ),
        encoding="utf-8",
    )
    (run_root / "progress.json").write_text(
        json.dumps({"operators": {"logit": {"status": "prepared"}}}),
        encoding="utf-8",
    )
    (reports / "round-1.json").write_text(
        json.dumps(
            {
                "operator": "logit",
                "candidate_status": {"success": True},
                "baseline_status": {"success": True},
                "cases": [{"case_id": "case-1", "baseline_us": 2, "candidate_us": 1}],
                "source_diff": {
                    "changed_file_count": 1,
                    "unified_diff": "+optimized\n",
                    "file_diffs": [],
                },
            }
        ),
        encoding="utf-8",
    )

    data = build_batch_report_data(run_root)

    assert data["operators"][0]["status"] == "reported"
    assert data["operators"][0]["workspace"] == str(run_root / "logit" / "work")
    assert data["summary"]["completed"] == 1


def test_report_keeps_each_operators_actual_round_count(tmp_path, monkeypatch):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "config.json").write_text(
        json.dumps({"parameters": {"operators": ["short", "long"]}}), encoding="utf-8"
    )
    trajectories = {
        "short": [{"round": 1, "success": True, "speedup": 1.5}],
        "long": [
            {"round": 1, "success": True, "speedup": 1.0},
            {"round": 2, "success": True, "speedup": 1.2},
            {"round": 3, "success": True, "speedup": 1.1},
        ],
    }

    def fake_trajectory(path):
        operator = path.parent.parent.name
        return {"operator": operator, "rounds": trajectories[operator], "baselines": []}

    monkeypatch.setattr("kernel_eval.batch_report.build_trajectory_data", fake_trajectory)

    data = build_batch_report_data(run_root)

    assert data["operators"][0]["best_rounds"] == [
        {"round": 1, "speedup": 1.5, "success": True},
    ]
    assert data["operators"][1]["best_rounds"] == [
        {"round": 1, "speedup": 1.0, "success": True},
        {"round": 2, "speedup": 1.2, "success": True},
        {"round": 3, "speedup": 1.2, "success": True},
    ]


def test_report_uses_na_when_no_candidate_round_exists(tmp_path):
    run_root = tmp_path / "run"
    (run_root / "logit" / "work" / "kernel_eval").mkdir(parents=True)
    (run_root / "config.json").write_text(
        json.dumps({"parameters": {"operators": ["logit"]}}), encoding="utf-8"
    )

    data = build_batch_report_data(run_root)
    html = write_batch_report(run_root).read_text(encoding="utf-8")

    assert data["operators"][0]["best_speedup"] is None
    assert data["summary"]["mean_best_speedup"] is None
    assert "最佳 N/A" in html


def test_report_discovers_operator_from_workspace_metadata_without_config(tmp_path):
    run_root = tmp_path / "single"
    workspace = run_root / "logit" / "work"
    (workspace / "kernel_eval").mkdir(parents=True)
    (workspace / "workspace.json").write_text(
        json.dumps({"operator": "logit"}), encoding="utf-8"
    )

    data = build_batch_report_data(run_root)

    assert [item["name"] for item in data["operators"]] == ["logit"]
    assert data["operators"][0]["workspace"] == str(workspace)


@pytest.mark.parametrize("status,failed,timed_out", [("failed", 1, 0), ("timeout", 0, 1)])
def test_report_counts_failed_operators_and_failed_rounds_separately(tmp_path, monkeypatch, status, failed, timed_out):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "config.json").write_text(
        json.dumps({"parameters": {"operators": ["logit"]}}), encoding="utf-8"
    )
    (run_root / "progress.json").write_text(
        json.dumps({"operators": {"logit": {"status": status}}}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "kernel_eval.batch_report.build_trajectory_data",
        lambda path: {
            "operator": "logit",
            "baselines": [],
            "rounds": [
                {"round": 1, "success": False, "speedup": None},
                {"round": 2, "success": True, "speedup": 1.25},
                {"round": 3, "success": False, "speedup": None},
                {"round": 4, "success": True, "speedup": 1.1},
            ],
        },
    )

    data = build_batch_report_data(run_root)
    html = write_batch_report(run_root).read_text(encoding="utf-8")

    assert data["summary"]["failed_operators"] == failed
    assert data["summary"]["timed_out"] == timed_out
    assert data["operators"][0]["best_speedup"] == 1.25
    assert data["summary"]["failed_rounds"] == 2
    assert data["operators"][0]["best_rounds"] == [
        {"round": 1, "speedup": 0.0, "success": False},
        {"round": 2, "speedup": 1.25, "success": True},
        {"round": 3, "speedup": 0.0, "success": False},
        {"round": 4, "speedup": 1.25, "success": True},
    ]
    assert data["operators"][0]["rounds"][0]["speedup"] is None
    assert "0x · 失败" in html
    assert "失败算子" in html
    assert "失败轮次" in html
