# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import argparse
import json
import subprocess

import pytest

from tests import run_ascend_arena_batch_e2e as batch_e2e


def test_discovers_only_idle_devices(monkeypatch):
    mapping = """
    NPU ID  Chip ID  Chip Logic ID  Phy ID  Name
    6       0        0              12      Ascend910
    6       1        1              13      Ascend910
    """

    def fake_npu_smi(*arguments):
        if arguments == ("info", "-m"):
            output = mapping
        elif arguments[-1] == "0":
            output = "No process in device.\n"
        else:
            output = "1234 python 1024\n"
        return subprocess.CompletedProcess(["npu-smi", *arguments], 0, output, "")

    monkeypatch.setattr(batch_e2e, "_npu_smi", fake_npu_smi)

    assert batch_e2e.discover_idle_devices() == ([0], [1])


def test_parses_requested_devices():
    assert batch_e2e.parse_devices("4, 5,6") == [4, 5, 6]
    with pytest.raises(argparse.ArgumentTypeError):
        batch_e2e.parse_devices("4,4")
    with pytest.raises(argparse.ArgumentTypeError):
        batch_e2e.parse_devices("4,x")


def test_summarizes_compile_correctness_and_performance(tmp_path):
    reports = tmp_path / "kernel_eval"
    run = reports / "ascend_arena_expand_into_jagged_permute_20260901_000000_deadbeef"
    baseline = run / "baseline_results"
    results = run / "results"
    baseline.mkdir(parents=True)
    results.mkdir(parents=True)
    (baseline / "status.json").write_text(json.dumps({
        "success": True,
        "build_success": True,
        "profile_success": True,
        "profile_cases": 4,
    }))
    (results / "status.json").write_text(json.dumps({
        "success": True,
        "build_success": True,
        "profile_success": True,
        "profile_cases": 4,
    }))
    for result_dir in (baseline, results):
        for stage in ("smoke", "complete", "profile"):
            (result_dir / f"{stage}-correctness_result.json").write_text(json.dumps({
                "success": True,
                "case_results": [{"passed": True}],
            }))

    result = batch_e2e.summarize_operator(
        tmp_path,
        operator="expand_into_jagged_permute",
        device_id=3,
        baseline_returncode=0,
        candidate_returncode=0,
    )

    assert result["success"] is True
    assert result["compile"] is True
    assert result["correctness"] is True
    assert result["performance"] is True


def test_batch_evaluates_original_operator_with_one_full_evaluation(tmp_path, monkeypatch):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        return 0

    monkeypatch.setattr(batch_e2e, "_run_command", fake_run)
    monkeypatch.setattr(
        batch_e2e,
        "summarize_operator",
        lambda *args, **kwargs: {
            "operator": "expand_into_jagged_permute",
            "device_id": 3,
            "success": True,
            "compile": True,
            "correctness": True,
            "performance": True,
        },
    )

    batch_e2e._evaluate_operator("expand_into_jagged_permute", 3, tmp_path)

    assert len(commands) == 1
    assert "--run-optimize" in commands[0]
    assert "--init-only" not in commands[0]
    assert "--baseline-dir" not in commands[0]


def test_main_uses_only_requested_idle_devices(tmp_path, monkeypatch):
    selected = []
    monkeypatch.setattr(batch_e2e, "_operators", lambda: ["op_a", "op_b"])
    monkeypatch.setattr(batch_e2e, "discover_idle_devices", lambda: ([0, 2, 4], [1, 3]))

    def fake_run(operators, devices, output, mode):
        selected.extend(devices)
        return [{"operator": name, "success": True} for name in operators]

    monkeypatch.setattr(batch_e2e, "_run_batch", fake_run)
    output = tmp_path / "run"

    assert batch_e2e.main(["--devices", "4,0", "--output", str(output)]) == 0
    assert selected == [4, 0]
    summary = json.loads((output / "e2e_summary.json").read_text())
    assert summary["requested_devices"] == [4, 0]
    assert summary["devices"] == [4, 0]
    assert summary["num_processes"] == 2


def test_main_defaults_to_idle_devices(tmp_path, monkeypatch):
    selected = []
    monkeypatch.setattr(batch_e2e, "_operators", lambda: ["op_a", "op_b"])
    monkeypatch.setattr(batch_e2e, "discover_idle_devices", lambda: ([3, 5, 7], []))
    monkeypatch.setattr(
        batch_e2e,
        "_run_batch",
        lambda operators, devices, output, mode: selected.extend(devices)
        or [{"operator": name, "success": True} for name in operators],
    )
    output = tmp_path / "run"

    assert batch_e2e.main(["--output", str(output)]) == 0
    assert selected == [3, 5]
    summary = json.loads((output / "e2e_summary.json").read_text())
    assert summary["requested_devices"] is None
    assert summary["devices"] == [3, 5]


def test_main_rejects_requested_busy_device(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(batch_e2e, "_operators", lambda: ["op_a"])
    monkeypatch.setattr(batch_e2e, "discover_idle_devices", lambda: ([0], [1]))
    output = tmp_path / "run"

    assert batch_e2e.main(["--devices", "1", "--output", str(output)]) == 2
    assert "requested NPUs are busy or unavailable: 1" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("failure", [None, "prepare", "init", "candidate"])
def test_workspace_flow_and_failure_summary(tmp_path, monkeypatch, failure, arena_workspace_source):
    from auto_pipeline import cli

    commands = []
    work = tmp_path / "expand_into_jagged_permute/work"

    def artifacts(folder):
        folder.mkdir(parents=True)
        (folder / "status.json").write_text(json.dumps({
            "success": True, "build_success": True, "profile_success": True, "profile_cases": 1,
        }))
        for stage in ("smoke", "complete", "profile"):
            (folder / f"{stage}-correctness_result.json").write_text(json.dumps({
                "success": True, "case_results": [{"passed": True}],
            }))

    def run(command, **kwargs):
        commands.append(command)
        stage = "prepare" if "--workspace-only" in command else "init" if "init" in command else "candidate"
        if stage == "prepare":
            # Run the prepare CLI with the isolated source fixture.
            return cli.main(command[3:]) if failure != stage else 1
        assert kwargs["cwd"] == work
        assert command[0] == "./validate.sh" and "require-idle" in command
        if failure == stage:
            return 1
        folder = (work / "kernel_eval/init_results" if stage == "init" else
                  work / "kernel_eval/ascend_arena_expand_into_jagged_permute_20260920_000000_test/results")
        artifacts(folder)
        return 0

    monkeypatch.setattr(batch_e2e, "_run_command", run)
    result = batch_e2e._evaluate_operator("expand_into_jagged_permute", 1, tmp_path, "workspace")
    assert result["success"] is (failure is None)
    assert result["mode"] == "workspace"
    assert result["prepare_returncode"] == (1 if failure == "prepare" else 0)
    assert result["baseline_returncode"] == (None if failure == "prepare" else 1 if failure == "init" else 0)
    assert len(commands) == (1 if failure == "prepare" else 2 if failure == "init" else 3)
    assert result["candidate_returncode"] == (None if failure in {"prepare", "init"} else 1 if failure else 0)
    if failure is None:
        assert result["baseline_status"] == str(work / "kernel_eval/init_results/status.json")


def test_workspace_mode_reaches_scheduler_and_records_failures(tmp_path, monkeypatch):
    operators = ["expand_into_jagged_permute", "group_quant", "adaptive_avg_pool3d"]
    monkeypatch.setattr(batch_e2e, "_operators", lambda: operators)
    monkeypatch.setattr(batch_e2e, "discover_idle_devices", lambda: ([0, 1, 2, 3], []))

    def evaluate(operator, device_id, output, mode):
        assert mode == "workspace"
        assert device_id in (1, 2)
        return {"operator": operator, "device_id": device_id, "success": operator != "expand_into_jagged_permute"}

    monkeypatch.setattr(batch_e2e, "_evaluate_operator", evaluate)
    output = tmp_path / "batch"
    assert batch_e2e.main([
        "--mode", "workspace", "--devices", "1,2", "--output", str(output),
    ]) == 1
    summary = json.loads((output / "e2e_summary.json").read_text())
    assert summary["mode"] == "workspace"
    assert summary["devices"] == [1, 2]
    assert summary["num_processes"] == 2
    assert [item["operator"] for item in summary["results"]] == operators
    assert [item["success"] for item in summary["results"]] == [False, True, True]
