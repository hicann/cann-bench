# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Regression coverage for Ascend Arena acceptance feedback (no NPU required)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ascend_arena_e2e.agent_base import OperatorAgent
from ascend_arena_e2e.agents.claude_code import ClaudeCodeAgent
from ascend_arena_e2e.devices import device_snapshot
from kernel_eval.batch_report import build_batch_report_data
from kernel_eval.optimize import _build_report, source_fingerprint, workspace_evaluation
from scripts import batch_optimize as batch


def workspace(tmp_path, operator="expand_into_jagged_permute"):
    work = tmp_path / operator / "work"
    (work / operator).mkdir(parents=True)
    (work / operator / "op.cpp").write_text("optimized")
    (work / "workspace.json").write_text(json.dumps({"operator": operator}))
    (work / "validate.sh").write_text("#!/bin/sh\n")
    return work


def evaluated(work, *, success=True):
    op = json.loads((work / "workspace.json").read_text())["operator"]
    name = f"ascend_arena_{op}_20260918_123456_test"
    root = work / "kernel_eval" / name
    root.mkdir(parents=True, exist_ok=True)
    report = {"operator": op, "baseline_status": {"success": True},
              "candidate_status": {"success": success},
              "summary": {"compared_cases": 1},
              "cases": [{"case_id": "case", "candidate_passed": success,
                         "baseline_us": 20, "candidate_us": 10, "speedup": 2}],
              "source_diff": {"candidate_sha256": source_fingerprint(work / op),
                              "changed_file_count": 1, "unified_diff": "+optimized"}}
    (root / f"{name}.json").write_text(json.dumps(report))
    return report


def test_final_source_must_match_latest_successful_evaluation(tmp_path):
    work = workspace(tmp_path)
    assert workspace_evaluation(work)["reason"] == "no_candidate_evaluation"
    evaluated(work, success=False)
    assert workspace_evaluation(work)["reason"] == "evaluation_failed"
    evaluated(work)
    assert workspace_evaluation(work)["success"]
    assert workspace_evaluation(work)["source_changed"] is True
    (work / "expand_into_jagged_permute/op.cpp").write_text("changed after validate")
    assert workspace_evaluation(work)["reason"] == "final_source_unvalidated"
    evaluated(work)
    assert workspace_evaluation(work)["success"]
    (work / "kernel_eval/ascend_arena_expand_into_jagged_permute_20260918_130000_interrupted").mkdir()
    assert workspace_evaluation(work)["reason"] == "incomplete_evaluation"


def test_unchanged_source_and_continuation_prompt(tmp_path):
    work = workspace(tmp_path)
    evaluated(work)
    path = next((work / "kernel_eval").glob("*/*.json"))
    report = json.loads(path.read_text())
    report["source_diff"]["changed_file_count"] = 0
    path.write_text(json.dumps(report))
    assert workspace_evaluation(work)["source_changed"] is False
    prompt = ClaudeCodeAgent().setup_prompt(work, "Validate and finish.")
    assert prompt.startswith("Validate and finish.")
    assert "Workspace continuation state:" in prompt and "validated" in prompt


def test_auth_error_classification(tmp_path):
    log = tmp_path / "claude.log"
    agent = ClaudeCodeAgent()
    for event in ({"type": "assistant", "api_error_status": 401},
                  {"type": "error", "message": {"api_error_status": 403}},
                  {"type": "result", "is_error": True, "result": "authentication_error"}):
        log.write_text(json.dumps(event))
        assert agent.classify_error(log) == "authentication_failed"
    log.write_text(json.dumps({"type": "user", "message": "401 in example source"}))
    assert agent.classify_error(log) == ""


def test_compact_artifacts_only_clean_success(tmp_path, monkeypatch):
    from kernel_eval.optimize import _cleanup_success_logs
    monkeypatch.setenv("ASCEND_ARENA_PROFILE_ARTIFACTS", "compact")
    (tmp_path / "profile").mkdir()
    (tmp_path / "profile/raw.db").write_text("raw")
    (tmp_path / "result.json").write_text("{}")
    _cleanup_success_logs(tmp_path, {"success": False})
    assert (tmp_path / "profile/raw.db").is_file()
    _cleanup_success_logs(tmp_path, {"success": True})
    assert not (tmp_path / "profile").exists()
    assert (tmp_path / "result.json").is_file()


def test_unknown_shell_subcommand_exits_before_dependencies():
    import subprocess
    result = subprocess.run(["bash", "scripts/run_auto_pipeline.sh", "not-a-command"],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "未知子命令" in result.stderr
    assert "cann_bench_utils" not in result.stdout


def test_baseline_process_cancellation(tmp_path):
    import subprocess
    OperatorAgent.reset_cancellation()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        with OperatorAgent.track_process(process):
            OperatorAgent.cancel_active_runs()
            assert process.wait(timeout=5) != 0
    finally:
        OperatorAgent.reset_cancellation()


def test_readme_skill_agent_fresh_instance(tmp_path, monkeypatch):
    import re
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    code = next(block for block in re.findall(r"```python\n(.*?)```", readme, re.S) if "class CannbotAgent" in block)
    code = code.replace("@register_agent", "")
    scope = {}
    exec(code, scope)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("Small changes and validate.")
    monkeypatch.setenv("CANNBOT_SKILL_PATH", str(skill / "SKILL.md"))
    agent = scope["CannbotAgent"]()
    agent.setup_agents_md(workspace(tmp_path))


def test_workspace_only_moved_directory_preserves_round_and_baseline(tmp_path):
    work = workspace(tmp_path)
    evaluated(work)
    moved = tmp_path / "renamed"
    work.parent.rename(moved)
    report = build_batch_report_data(tmp_path)
    op = report["operators"][0]
    assert op["name"] == "expand_into_jagged_permute"
    assert len(op["rounds"]) == 1 and len(op["baselines"]) == 1
    assert op["workspace"] == str(moved / "work")


def test_resume_rechecks_results_and_restart_preserves_unselected(tmp_path):
    work = workspace(tmp_path)
    state = {"operators": {"expand_into_jagged_permute": {"status": "completed"}, "other": {"status": "failed"}}}
    path = tmp_path / "progress.json"
    path.write_text(json.dumps(state))
    items = [{"operator": "expand_into_jagged_permute", "workspace": str(work)}]
    result = batch._initial_progress(path, items, run_agent=True, resume=True)
    assert result["operators"]["expand_into_jagged_permute"]["status"] == "pending"
    assert result["operators"]["other"] == state["operators"]["other"]
    evaluated(work)
    assert batch._initial_progress(path, items, run_agent=True, resume=True)["operators"]["expand_into_jagged_permute"]["status"] == "completed"
    assert batch._initial_progress(path, items, run_agent=True, resume=False)["operators"]["expand_into_jagged_permute"]["status"] == "pending"


def test_timeout_stops_silent_process_and_removes_pid(tmp_path):
    work = workspace(tmp_path)
    class SleepingAgent(OperatorAgent):
        name = "sleeping-test"
        def build_command(self, prompt):
            return [sys.executable, "-c", "import time; time.sleep(60)"]
    result = SleepingAgent(timeout_seconds=0.1).run(work)
    assert result.timed_out and result.returncode == 124 and not result.ok
    assert not (work / "artifact/agent.pid").exists()
    assert '"timed_out": true' in (work / "artifact/timing.jsonl").read_text()
    assert "agent exit code: 124" in result.log_file.read_text()
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            SleepingAgent(timeout_seconds=value)


def test_timeout_stops_detached_children(tmp_path):
    import psutil

    work = workspace(tmp_path)
    class ChildAgent(OperatorAgent):
        name = "child-test"
        def build_command(self, prompt):
            return [sys.executable, "-c",
                    "import subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True); "
                    "Path('child.pid').write_text(str(p.pid)); time.sleep(60)"]
    result = ChildAgent(timeout_seconds=1).run(work)
    assert result.timed_out
    pid = int((work / "child.pid").read_text())
    try:
        child = psutil.Process(pid)
        child.wait(timeout=5)
    except psutil.NoSuchProcess:
        pass
    except psutil.TimeoutExpired:
        assert child.status() == psutil.STATUS_ZOMBIE


def test_container_stop_timeout_keeps_id_for_recovery(tmp_path, monkeypatch):
    import subprocess
    cidfile = tmp_path / "container-id"
    cidfile.write_text("a" * 64)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker stop", 10)
    monkeypatch.setattr(subprocess, "run", timeout)
    OperatorAgent._stop_container(cidfile)
    assert cidfile.exists()


def test_claude_profile_uses_private_configuration(tmp_path):
    work = workspace(tmp_path)
    agent = ClaudeCodeAgent(api_key="secret", base_url="https://example.invalid")
    env = {"ANTHROPIC_AUTH_TOKEN": "personal-token", "CLAUDE_CONFIG_DIR": "/personal"}
    agent.setup_workspace_environment(work, env, docker=False)
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert env["CLAUDE_CONFIG_DIR"] == str(work / "artifact/claude-config")
    assert (Path(env["CLAUDE_CONFIG_DIR"]).stat().st_mode & 0o777) == 0o700
    command = agent.build_command("test")
    assert command[command.index("--setting-sources") + 1] == ""
    assert "secret" not in command


def test_success_log_cleanup_updates_status(tmp_path):
    from kernel_eval.optimize import _cleanup_success_logs

    status = {"success": True, "compile_log": "compile.log", "profile_log": "profile.log"}
    for name in ("compile.log", "profile.log"):
        (tmp_path / name).write_text("done")
    _cleanup_success_logs(tmp_path, status)
    stored = json.loads((tmp_path / "status.json").read_text())
    assert stored["compile_log"] is None and stored["profile_log"] is None
    assert not (tmp_path / "compile.log").exists()


def test_claude_bearer_token_passes_to_docker(tmp_path):
    work = workspace(tmp_path)
    agent = ClaudeCodeAgent(api_key="bearer-key", auth_type="bearer", base_url="https://example.invalid")
    env = agent.setup_environment({})
    agent.setup_workspace_environment(work, env, docker=True)
    assert env["ANTHROPIC_AUTH_TOKEN"] == "bearer-key"
    command = agent.build_docker_command(work, ["true"], image="test", device_id=1, environment=env)
    assert "ANTHROPIC_AUTH_TOKEN" in command
    assert "ANTHROPIC_API_KEY" not in command
    assert "bearer-key" not in command


@pytest.mark.parametrize("auth_type", ["api-key", "bearer"])
def test_claude_profile_credential_overrides_ambient_auth(tmp_path, auth_type):
    work = workspace(tmp_path)
    agent = ClaudeCodeAgent(api_key="profile-test-key", auth_type=auth_type)
    env = agent.setup_environment({"ANTHROPIC_AUTH_TOKEN": "ambient-test-token",
                                   "ANTHROPIC_API_KEY": "ambient-test-key"})
    agent.setup_workspace_environment(work, env, docker=False)
    selected, other = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY") if auth_type == "bearer" else (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    assert env[selected] == "profile-test-key"
    assert other not in env


def test_device_preflight_idle_busy_unknown_and_visible_mapping(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    calls = []
    def query(*args):
        calls.append(args)
        return SimpleNamespace(stdout="No process in device", stderr="")
    record = device_snapshot(0, mapping={3: (1, 0)}, query=query)
    assert record["state"] == "idle" and record["logic_id"] == 3
    assert calls == [("info", "-t", "proc-mem", "-i", "1", "-c", "0")]
    assert device_snapshot(0, mapping={3: (1, 0)}, query=lambda *a: SimpleNamespace(stdout="PID 123", stderr=""))["state"] == "busy"
    assert device_snapshot(0, mapping={}, query=query)["state"] == "unknown"


def test_median_samples_noise_and_versioned_reports(tmp_path):
    from ascend_arena_e2e.ascend_arena.runners.profile import ProfileRunner
    runner = ProfileRunner()
    metric = runner._duration_statistics([{"Task Duration(us)": str(x)} for x in [10, 11, 1000]])
    assert metric["task_duration_us"] == 11
    assert metric["duration_samples_us"] == [10, 11, 1000]
    assert runner._finalize_msprof_metric(metric)["noise_band"] == metric["noise_band"]
    _build_report(output_dir=tmp_path, run_name="run", operator="expand_into_jagged_permute", device_id=1,
                  baseline={"a": [{"task_duration_us": 10, "noise_band": .02}]},
                  candidate={"a": [{"task_duration_us": 9.99, "noise_band": .02}]},
                  baseline_status={"success": True}, candidate_status={"success": True})
    report = json.loads((tmp_path / "run.json").read_text())
    comparison = json.loads((tmp_path / "comparison.json").read_text())
    assert report["schema_version"] == comparison["schema_version"] == 1
    assert {"operator", "device", "summary", "source_diff", "candidate_status", "baseline_status", "cases"} <= report.keys()
    assert comparison["cases"][0]["noise_band"] == .04
    assert comparison["cases"][0]["significant"] is False
    assert report["measurement"]["significant"] is False
    schema = json.loads((Path(__file__).resolve().parents[2] / "docs/ascend_arena/report.schema.json").read_text())
    assert set(schema["required"]) <= report.keys()
    assert set(schema["properties"]["summary"]["required"]) <= report["summary"].keys()
    for case in report["cases"]:
        assert set(schema["properties"]["cases"]["items"]["required"]) <= case.keys()


def test_subset_restart_preserves_manifest_and_progress(tmp_path, monkeypatch, arena_workspace_source):
    import auto_pipeline.generator.ascend_arena.workspace as workspace_module
    monkeypatch.setattr(workspace_module, "_run_initial_evaluation", lambda *a: None)
    output = tmp_path / "batch"
    assert batch.main(["--operators", "expand_into_jagged_permute,group_quant", "--output", str(output)]) == 0
    assert batch.main(["--operators", "expand_into_jagged_permute", "--restart", "--output", str(output)]) == 0
    config = json.loads((output / "config.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    assert config["parameters"]["operators"] == ["expand_into_jagged_permute", "group_quant"]
    assert config["parameters"]["selected_operators"] == ["expand_into_jagged_permute"]
    assert len(config["invocations"]) == 2
    assert set(progress["operators"]) == {"expand_into_jagged_permute", "group_quant"}


def test_legacy_result_without_noise_is_unknown(tmp_path):
    _build_report(output_dir=tmp_path, run_name="legacy", operator="expand_into_jagged_permute", device_id=1,
                  baseline={"a": [{"task_duration_us": 20}]},
                  candidate={"a": [{"task_duration_us": 10}]},
                  baseline_status={"success": True}, candidate_status={"success": True})
    row = json.loads((tmp_path / "comparison.json").read_text())["cases"][0]
    assert row["speedup"] == 2 and row["noise_band"] is None and row["significant"] is None


def test_build_environment_avoids_internal_paths():
    root = Path(__file__).resolve().parents[2]
    cmake = (root / "src/cann_bench_utils/CMakeLists.txt").read_text()
    assert "set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)" in cmake
    build = (root / "src/ascend_arena_e2e/ascend_arena/runners/build.py").read_text()
    assert "Path(sys.executable).resolve().parent" not in build
