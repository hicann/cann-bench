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

import ascend_arena_e2e.agent_base as agent_base
import auto_pipeline.generator.ascend_arena.workspace as workspace_module
from scripts import batch_optimize as batch


@pytest.fixture(autouse=True)
def _skip_hardware_workspace_initialization(monkeypatch, arena_workspace_source):
    monkeypatch.setattr(workspace_module, "_run_initial_evaluation", lambda workspace: None)


def test_batch_prepares_manifest_progress_report_reuse_and_force(tmp_path, monkeypatch, capsys):
    output = tmp_path / "batch"
    initialized = []
    monkeypatch.setattr(
        workspace_module,
        "_run_initial_evaluation",
        lambda workspace: initialized.append(workspace.workdir),
    )
    args = [
        "--operators",
        "expand_into_jagged_permute",
        "--output",
        str(output),
        "--devices",
        "3",
        "--goal",
        "Make this operator faster.",
    ]

    assert batch.main(args) == 0
    output_text = capsys.readouterr().out
    assert "[batch] preparing 1 operator(s) on NPU 3" in output_text
    assert "[batch] finished; report:" in output_text
    workdir = output / "expand_into_jagged_permute" / "work"
    manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert manifest["parameters"]["operators"] == ["expand_into_jagged_permute"]
    assert manifest["parameters"]["devices"] == [3]
    assert manifest["parameters"]["batch_size"] == 1
    assert manifest["parameters"]["agent"] == ""
    assert manifest["parameters"]["agent_environment"] == {}
    assert manifest["workspaces"][0]["workspace"] == str(workdir)
    assert json.loads((output / "progress.json").read_text())["operators"]["expand_into_jagged_permute"]["status"] == "prepared"
    assert (output / "report.html").is_file()
    workspace_metadata = json.loads((workdir / "workspace.json").read_text(encoding="utf-8"))
    assert workspace_metadata["goal"] == "Make this operator faster."
    assert workspace_metadata["device_id"] == 3
    assert 'ASCEND_ARENA_DEVICE_ID="${ASCEND_ARENA_DEVICE_ID:-3}"' in (
        workdir / "validate.sh"
    ).read_text(encoding="utf-8")
    assert not (workdir / "AGENTS.md").exists()
    assert not (workdir / "REQUIRE.md").exists()
    assert not (workdir / "cases" / "complete_cases.json").exists()
    assert not (workdir / "cases" / "profile_cases.json").exists()
    assert (workdir / "expand_into_jagged_permute" / "op_host").is_dir()
    assert (workdir / "expand_into_jagged_permute" / "op_kernel").is_dir()
    assert not (workdir / "expand_into_jagged_permute" / "cases").exists()
    assert initialized == [workdir]

    marker = workdir / "expand_into_jagged_permute" / "agent_edit.txt"
    marker.write_text("preserve me\n", encoding="utf-8")
    assert batch.main(args) == 0
    capsys.readouterr()
    assert marker.is_file()
    manifest = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert manifest["workspaces"][0]["status"] == "existing"
    assert initialized == [workdir, workdir]

    assert batch.main([*args, "--force"]) == 0
    capsys.readouterr()
    assert not marker.exists()
    assert initialized == [workdir, workdir, workdir]


def test_incomplete_workspace_requires_force(tmp_path, capsys):
    output = tmp_path / "batch"
    (output / "expand_into_jagged_permute").mkdir(parents=True)

    assert batch.main(["--operators", "expand_into_jagged_permute", "--output", str(output)]) == 1
    assert "incomplete workspace" in capsys.readouterr().err


def test_batch_runs_one_worker_per_device_and_resumes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("kernel_eval.optimize.workspace_evaluation", lambda path: {"success": True})
    output = tmp_path / "batch-agent"
    calls = []

    class FakeResult:
        def __init__(self, workspace, ok):
            self.workspace = workspace
            self.ok = ok
            self.returncode = 0 if ok else 9
            self.log_file = workspace / "artifact" / "fake-agent.log"

        def to_dict(self):
            return {
                "agent": "fake-agent",
                "workspace": str(self.workspace),
                "returncode": 0 if self.ok else 9,
                "ok": self.ok,
            }

    class FakeAgent:
        def preflight(self, *, docker=False):
            pass

        name = "fake-agent"
        harness = "fake-harness"
        api_key = "private-key"
        model = "model-a"
        model_reasoning_effort = "high"
        base_url = ""
        environment_variables = {}

        def run(self, workspace, *, prompt, device_id, docker_image):
            workspace = Path(workspace)
            calls.append((workspace.parent.name, device_id, prompt, docker_image))
            return FakeResult(workspace, ok=workspace.parent.name != "expand_into_jagged_permute")

    monkeypatch.setattr(agent_base, "agent_from_args", lambda args: FakeAgent())
    args = [
        "--operators",
        "gather_elements_v2,expand_into_jagged_permute",
        "--output",
        str(output),
        "--devices",
        "2,5",
        "--agent",
        "fake-agent",
        "--agent-prompt",
        "Optimize each operator.",
    ]

    assert batch.main(args) == 1
    output_text = capsys.readouterr().out
    assert "[batch] starting agent fake-agent; pending: 2, resumed: 0" in output_text
    assert "[gather_elements_v2][NPU " in output_text
    assert "[expand_into_jagged_permute][NPU " in output_text
    assert {item[0]: item[1] for item in calls} == {"gather_elements_v2": 2, "expand_into_jagged_permute": 5}
    assert all(item[2:] == ("Optimize each operator.", "") for item in calls)
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert "private-key" not in (output / "config.json").read_text(encoding="utf-8")
    assert progress["operators"]["gather_elements_v2"]["status"] == "completed"
    assert progress["operators"]["expand_into_jagged_permute"]["status"] == "failed"
    assigned_devices = [
        json.loads((output / name / "work/workspace.json").read_text(encoding="utf-8"))["device_id"]
        for name in ("gather_elements_v2", "expand_into_jagged_permute")
    ]
    assert assigned_devices == [2, 5]

    calls.clear()
    assert batch.main(args) == 1
    capsys.readouterr()
    assert [(item[0], item[1]) for item in calls] == [("expand_into_jagged_permute", 5)]

    calls.clear()
    assert batch.main([*args, "--restart"]) == 1
    capsys.readouterr()
    assert {item[0]: item[1] for item in calls} == {"gather_elements_v2": 2, "expand_into_jagged_permute": 5}


def test_timeout_preserves_implementation_and_returns_timeout_status(tmp_path, monkeypatch, capsys):
    output = tmp_path / "timeout"

    class Agent:
        name = "timeout-test"
        harness = model = "test"
        model_reasoning_effort = base_url = ""
        environment_variables = {}

        def preflight(self, **kwargs):
            pass

        def run(self, workspace, **kwargs):
            path = Path(workspace)
            (path / "expand_into_jagged_permute" / "agent_edit.txt").write_text("candidate")
            return agent_base.AgentRunResult(
                self.name, self.harness, path, 124, path / "agent.log", timed_out=True,
                evaluation={"success": True},
            )

    monkeypatch.setattr(agent_base, "agent_from_args", lambda args: Agent())
    assert batch.main(["--operators", "expand_into_jagged_permute", "--agent", "timeout-test",
                       "--output", str(output)]) == 124
    text = capsys.readouterr().out
    assert "timeout (exit 124)" in text
    assert "failed (exit 124" not in text
    progress = json.loads((output / "progress.json").read_text())
    state = progress["operators"]["expand_into_jagged_permute"]
    assert state["status"] == "timeout"
    assert state["agent_run"]["evaluation"]["success"] is True
    assert (output / "expand_into_jagged_permute/work/expand_into_jagged_permute/agent_edit.txt").read_text() == "candidate"


@pytest.mark.parametrize("states,expected", [(["completed", "timeout"], 124), (["failed", "timeout"], 1)])
def test_batch_exit_distinguishes_timeout_from_failure(tmp_path, monkeypatch, states, expected):
    monkeypatch.setattr(batch, "optimize", lambda args: {"operators": {
        str(i): {"status": status} for i, status in enumerate(states)
    }})
    assert batch.main(["--output", str(tmp_path)]) == expected


def test_agent_starts_after_each_operator_baseline_on_same_device(tmp_path, monkeypatch):
    output = tmp_path / "pipelined"
    events = []

    class FakeResult:
        ok = True
        returncode = 0

        def __init__(self, workspace):
            self.log_file = Path(workspace) / "artifact" / "fake.log"

        def to_dict(self):
            return {"ok": True}

    class FakeAgent:
        def preflight(self, *, docker=False):
            pass

        name = "fake-agent"
        harness = "fake-harness"
        model = "model-a"
        model_reasoning_effort = "high"
        base_url = ""
        environment_variables = {}

        def build_docker_command(self, workspace, command, *, image, device_id, environment):
            events.append(("docker", Path(workspace).parent.name, tuple(command), image, device_id, environment))
            return ["docker", "init", Path(workspace).parent.name]

        def run(self, workspace, **kwargs):
            events.append(("agent", Path(workspace).parent.name))
            return FakeResult(workspace)

    monkeypatch.setattr(agent_base, "agent_from_args", lambda args: FakeAgent())
    monkeypatch.setattr(
        workspace_module,
        "_run_initial_evaluation",
        lambda workspace, command=None: events.append(
            ("init", workspace.workdir.parent.name, tuple(command) if command else None)
        ),
    )

    assert batch.main([
        "--operators", "gather_elements_v2,expand_into_jagged_permute",
        "--output", str(output),
        "--devices", "4",
        "--agent", "fake-agent",
        "--docker",
        "--docker-image", "test-image",
        "--restart",
    ]) == 0

    assert events == [
        ("docker", "gather_elements_v2", ("./validate.sh", "init"), "test-image", 4, {}),
        ("init", "gather_elements_v2", ("docker", "init", "gather_elements_v2")),
        ("agent", "gather_elements_v2"),
        ("docker", "expand_into_jagged_permute", ("./validate.sh", "init"), "test-image", 4, {}),
        ("init", "expand_into_jagged_permute", ("docker", "init", "expand_into_jagged_permute")),
        ("agent", "expand_into_jagged_permute"),
    ]


def test_batch_marks_running_operator_interrupted_on_ctrl_c(tmp_path, monkeypatch):
    output = tmp_path / "interrupted"

    class InterruptingAgent:
        def preflight(self, *, docker=False):
            pass

        name = "interrupting-agent"
        harness = "fake-harness"
        model = "model-a"
        model_reasoning_effort = ""
        base_url = ""
        environment_variables = {}

        def run(self, workspace, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(agent_base, "agent_from_args", lambda args: InterruptingAgent())

    assert batch.main([
        "--operators", "expand_into_jagged_permute",
        "--output", str(output),
        "--devices", "4",
        "--agent", "interrupting-agent",
    ]) == 130

    state = json.loads((output / "progress.json").read_text(encoding="utf-8"))["operators"]["expand_into_jagged_permute"]
    assert state["status"] == "interrupted"
    assert state["error"].startswith("KeyboardInterrupt:")


def test_device_parser_rejects_duplicates():
    parser = batch.create_parser()
    try:
        parser.parse_args(["--devices", "0,0"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("duplicate devices should be rejected")


def test_auth_failure_stops_next_operator_initialization(tmp_path, monkeypatch):
    from ascend_arena_e2e.agent_base import AgentRunResult
    events = []

    class Agent:
        name = "auth-test"
        harness = "test"
        model = "test"
        model_reasoning_effort = ""
        base_url = ""
        environment_variables = {}

        def preflight(self, **kwargs):
            pass

        def run(self, workspace, **kwargs):
            path = Path(workspace)
            events.append(("agent", path.parent.name))
            return AgentRunResult(self.name, self.harness, path, 1, path / "auth.log",
                                  error_kind="authentication_failed")

    monkeypatch.setattr(agent_base, "agent_from_args", lambda args: Agent())
    monkeypatch.setattr(workspace_module, "_run_initial_evaluation",
                        lambda work: events.append(("init", work.workdir.parent.name)))
    output = tmp_path / "auth"
    assert batch.main(["--operators", "expand_into_jagged_permute,gather_elements_v2", "--devices", "1",
                       "--agent", "auth-test", "--output", str(output)]) == 1
    assert events == [("init", "expand_into_jagged_permute"), ("agent", "expand_into_jagged_permute")]
    state = json.loads((output / "progress.json").read_text())["operators"]
    assert state["expand_into_jagged_permute"]["agent_run"]["error_kind"] == "authentication_failed"
    assert state["gather_elements_v2"]["status"] == "pending"


def test_negative_agent_concurrency_fails_before_preparation(tmp_path):
    with pytest.raises(SystemExit) as error:
        batch.main(["--agent-concurrency", "-1", "--output", str(tmp_path / "invalid")])
    assert error.value.code == 2
    assert not (tmp_path / "invalid").exists()


def test_auth_failure_stops_agent_waiting_for_slot(tmp_path, monkeypatch):
    import threading
    import time

    from ascend_arena_e2e.agent_base import AgentRunResult

    ready = threading.Barrier(2)
    calls = []

    class Agent:
        name = "auth-test"
        harness = "test"
        model = "test"
        model_reasoning_effort = ""
        base_url = ""
        environment_variables = {}

        def preflight(self, **kwargs):
            pass

        def run(self, workspace, **kwargs):
            path = Path(workspace)
            calls.append(path)
            time.sleep(0.05)
            return AgentRunResult(self.name, self.harness, path, 1, path / "auth.log",
                                  error_kind="authentication_failed")

    import kernel_eval.batch_report as reports
    original_report = reports.write_batch_report

    def write_report(*args, **kwargs):
        # Status persistence happens after the running agent releases its slot.
        if calls:
            time.sleep(0.05)
        return original_report(*args, **kwargs)

    monkeypatch.setattr(reports, "write_batch_report", write_report)
    monkeypatch.setattr(agent_base, "agent_from_args", lambda args: Agent())
    monkeypatch.setattr(workspace_module, "_run_initial_evaluation", lambda work: ready.wait(timeout=5))
    output = tmp_path / "auth-concurrent"
    assert batch.main(["--operators", "expand_into_jagged_permute,gather_elements_v2", "--devices", "1,2",
                       "--agent", "auth-test", "--agent-concurrency", "1", "--output", str(output)]) == 1
    assert len(calls) == 1
    state = json.loads((output / "progress.json").read_text())
    assert sorted(item["status"] for item in state["operators"].values()) == ["failed", "pending"]
    pending = next(item for item in state["operators"].values() if item["status"] == "pending")
    assert "started_at" not in pending
    config = json.loads((output / "config.json").read_text())
    assert config["parameters"]["agent_concurrency"] == 1
