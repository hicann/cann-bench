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
import subprocess
import sys

import pytest

import ascend_arena_e2e.agent_base as agent_base
from ascend_arena_e2e.agent_base import (
    OperatorAgent,
    available_agents,
    create_agent,
    register_agent,
)
from ascend_arena_e2e.agents.claude_code import ClaudeCodeAgent
from ascend_arena_e2e.agents.codex import CodexAgent


def _workspace(tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "workspace.json").write_text(
        json.dumps({"operator": "example_op", "source_dir": str(workspace / "example_op"), "goal": "Go faster."}),
        encoding="utf-8",
    )
    (workspace / "validate.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    return workspace


def test_agent_failure_refreshes_closed_trace(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    agent = CodexAgent()

    def fail(*args, **kwargs):
        raise RuntimeError("agent unavailable")

    monkeypatch.setattr(agent, "_run", fail)
    with pytest.raises(RuntimeError, match="agent unavailable"):
        agent.run(workspace)
    events = [json.loads(line) for line in (workspace / "artifact/timing.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "end"
    html = (workspace / "kernel_eval/optimization_trajectory.html").read_text()
    assert events[-1]["ended_at"] in html


def test_builtin_agents_map_environment_and_commands(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    codex = CodexAgent(
        base_url="https://openai.example/v1",
        api_key="openai-key",
        model="gpt-test",
        model_reasoning_effort="high",
    )
    codex_env = codex.setup_environment({"PATH": "/bin"})

    assert {"codex", "claude-code"}.issubset(available_agents())
    assert codex_env["OPENAI_BASE_URL"] == "https://openai.example/v1"
    assert codex_env["OPENAI_API_KEY"] == "openai-key"
    assert codex_env["OPENAI_MODEL"] == "gpt-test"
    assert codex.build_command("optimize") == [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--json",
        "--config",
        'model_provider="openai-compatible"',
        "--config",
        'model_providers.openai-compatible.name="OpenAI compatible"',
        "--config",
        'model_providers.openai-compatible.base_url="https://openai.example/v1"',
        "--config",
        'model_providers.openai-compatible.env_key="OPENAI_API_KEY"',
        "--config",
        'model_providers.openai-compatible.wire_api="responses"',
        "--config",
        "model_providers.openai-compatible.supports_websockets=false",
        "--model",
        "gpt-test",
        "--config",
        'model_reasoning_effort="high"',
        "optimize",
    ]

    claude = ClaudeCodeAgent(
        base_url="https://anthropic.example",
        api_key="anthropic-key",
        model="sonnet",
        model_reasoning_effort="high",
    )
    claude_env = claude.setup_environment({})
    assert claude_env["ANTHROPIC_BASE_URL"] == "https://anthropic.example"
    assert claude_env["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert claude_env["ANTHROPIC_MODEL"] == "sonnet"
    assert claude.build_command("optimize") == [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--verbose",
        "--output-format",
        "stream-json",
        "--setting-sources",
        "",
        "--model",
        "sonnet",
        "--effort",
        "high",
        "optimize",
    ]

    assert CodexAgent().model == "gpt-5.4"
    assert ClaudeCodeAgent().model == "sonnet"

    monkeypatch.setenv("OPENAI_BASE_URL", "https://configured.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "configured-key")
    configured = CodexAgent()
    assert configured.base_url == "https://configured.example/v1"
    assert configured.api_key == "configured-key"


def test_agent_profile_loads_private_yaml_config(tmp_path, monkeypatch):
    config = tmp_path / ".env"
    config.write_text(
        "team-codex:\n"
        "  harness: codex\n"
        "  api_key: private-key\n"
        "  base_url: https://gateway.example/v1\n"
        "  model: gpt-private\n"
        "  model_reasoning_effort: high\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent_base, "AGENT_CONFIG_PATH", config)

    agent = create_agent("team-codex")

    assert agent.name == "team-codex"
    assert agent.harness == "codex"
    assert agent.api_key == "private-key"
    assert agent.base_url == "https://gateway.example/v1"
    assert agent.model == "gpt-private"
    assert agent.model_reasoning_effort == "high"
    assert agent.setup_environment({})["OPENAI_API_KEY"] == "private-key"


def test_claude_requires_a_successful_result_event(tmp_path):
    log_file = tmp_path / "claude.log"
    agent = ClaudeCodeAgent()

    log_file.write_text('{"type":"result","subtype":"success","is_error":false}\n', encoding="utf-8")
    assert agent.output_succeeded(log_file)

    log_file.write_text('{"type":"result","subtype":"error","is_error":true}\n', encoding="utf-8")
    assert not agent.output_succeeded(log_file)

    log_file.write_text("Execution error\n", encoding="utf-8")
    assert not agent.output_succeeded(log_file)


def test_codex_requires_a_completed_turn(tmp_path):
    log_file = tmp_path / "codex.log"
    agent = CodexAgent()

    log_file.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
    assert agent.output_succeeded(log_file)

    log_file.write_text('{"type":"turn.failed","error":{"message":"timeout"}}\n', encoding="utf-8")
    assert not agent.output_succeeded(log_file)

    log_file.write_text("ERROR: Reconnecting... 5/5\n", encoding="utf-8")
    assert not agent.output_succeeded(log_file)


def test_claude_filters_thinking_token_events():
    agent = ClaudeCodeAgent()

    assert not agent.should_log_output(
        '{"type":"system","subtype":"thinking_tokens","estimated_tokens":5}\n'
    )
    assert agent.should_log_output('{"type":"assistant","message":{"content":[]}}\n')
    assert agent.should_log_output("plain output\n")


def test_agent_can_override_setup_agents_md(tmp_path):
    workspace = _workspace(tmp_path)

    class CustomAgent(CodexAgent):
        def setup_agents_md(self, workspace):
            (workspace / "AGENTS.md").write_text("# Custom Instructions\n", encoding="utf-8")

    CustomAgent().setup_prompt(workspace)

    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == "# Custom Instructions\n"


def test_agent_runs_in_workspace_and_records_artifacts(tmp_path, capsys):
    workspace = _workspace(tmp_path)

    @register_agent
    class RecordingAgent(OperatorAgent):
        name = "test-recording-agent"
        environment_variables = {"api_key": "TEST_AGENT_API_KEY"}

        def build_command(self, prompt):
            script = (
                "import json, os, pathlib, sys; "
                "pathlib.Path('capture.json').write_text(json.dumps({"
                "'cwd': os.getcwd(), 'key': os.environ['TEST_AGENT_API_KEY'], "
                "'prompt': sys.argv[1], 'tail': sys.argv[2]})); "
                "print('live output', flush=True)"
            )
            return [sys.executable, "-c", script, prompt, "tail"]

    result = RecordingAgent(api_key="secret").run(workspace, prompt="private prompt")

    assert result.returncode == 0
    assert not result.ok
    assert result.evaluation["reason"] == "no_candidate_evaluation"
    capture = json.loads((workspace / "capture.json").read_text(encoding="utf-8"))
    assert capture == {
        "cwd": str(workspace.resolve()),
        "key": "secret",
        "prompt": "private prompt",
        "tail": "tail",
    }
    log = result.log_file.read_text(encoding="utf-8")
    assert not list((workspace / "artifact").glob("*-prompt.md"))
    assert "command:" in log
    assert "<prompt>" in log
    assert "prompt:\nprivate prompt" in log
    assert "agent output:\nlive output" in log
    assert "agent exit code: 0" in log
    agents_md = (workspace / "AGENTS.md").read_text(encoding="utf-8")
    assert agents_md.startswith("# Operator Optimization\n")
    assert "- Operator: `example_op`" in agents_md
    assert "- Goal: Go faster." in agents_md
    assert f"[{workspace.parent.name}][test-recording-agent] live output" in capsys.readouterr().out


def test_agent_validates_workspace_and_loads_module_class(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="workspace.json, validate.sh"):
        CodexAgent().get_workspace_path(tmp_path)

    module = tmp_path / "example_workspace_agent.py"
    module.write_text(
        "from ascend_arena_e2e.agent_base import OperatorAgent, register_agent\n"
        "@register_agent\n"
        "class ExampleAgent(OperatorAgent):\n"
        "    name = 'example-external-agent'\n"
        "    def build_command(self, prompt):\n"
        "        return ['example-agent', prompt]\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    agent = create_agent("example_workspace_agent:ExampleAgent", model="example-model")

    assert agent.name == "example-external-agent"
    assert agent.model == "example-model"
    assert create_agent("example-external-agent").name == "example-external-agent"


def test_docker_command_mounts_workspace_context_read_only(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    agent = CodexAgent(api_key="secret")
    agent.setup_prompt(workspace)
    config = tmp_path / ".env"
    config.write_text("codex: {}\n", encoding="utf-8")
    monkeypatch.setattr(agent_base, "AGENT_CONFIG_PATH", config)

    command = agent.build_docker_command(
        workspace,
        ["codex", "exec", "prompt"],
        image="agent-image",
        device_id=3,
        environment=agent.setup_environment({}),
    )
    joined = " ".join(command)

    assert command[:3] == ["docker", "run", "--rm"]
    assert command[command.index("--cidfile") + 1] == str(
        workspace / "artifact" / ".codex.container-id"
    )
    assert "--userns=host" in command
    assert command[command.index("--user") + 1] == f"{agent_base.os.getuid()}:{agent_base.os.getgid()}"
    assert not any(item.startswith("ASCEND_RT_VISIBLE_DEVICES=") for item in command)
    assert "ASCEND_ARENA_DEVICE_ID=0" in command
    assert "OPENAI_API_KEY" in command
    assert "secret" not in command
    assert f"src={workspace},dst=/workspace" in joined
    assert f"src={workspace / 'AGENTS.md'},dst=/workspace/AGENTS.md,readonly" in joined
    assert "src=/dev/null,dst=/cann-bench/src/ascend_arena_e2e/agents/.env,readonly" in joined
    assert command[-3:] == ["codex", "exec", "prompt"]


def test_docker_run_tells_agent_to_use_container_paths(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    captured = {}

    class DockerAgent(CodexAgent):
        def build_command(self, prompt):
            captured["prompt"] = prompt
            return [sys.executable, "-c", 'print(\'{"type":"turn.completed"}\')']

        def build_docker_command(self, workspace, command, **kwargs):
            return command

    result = DockerAgent().run(workspace, prompt="Validate.", device_id=4, docker_image="test-image")

    assert result.returncode == 0
    assert not result.ok
    assert "/workspace" in captured["prompt"]
    assert "/cann-bench" in captured["prompt"]
    assert "Do not use host absolute paths from workspace.json." in captured["prompt"]


def test_agent_stops_container_recorded_by_cidfile(tmp_path, monkeypatch):
    cidfile = tmp_path / "container-id"
    cidfile.write_text("a" * 64, encoding="utf-8")
    calls = []
    def stop(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(subprocess, "run", stop)

    OperatorAgent._stop_container(cidfile)

    assert calls == [["docker", "stop", "--time", "5", "a" * 64]]
    assert not cidfile.exists()


def test_invalid_agent_import_has_a_cli_friendly_error():
    with pytest.raises(ValueError, match="cannot load agent"):
        create_agent("missing_workspace_agent:MissingAgent")


def test_claude_root_preflight_and_container_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_base.shutil, 'which', lambda name: '/bin/' + name)
    monkeypatch.setattr(agent_base.os, 'geteuid', lambda: 0)
    monkeypatch.delenv('IS_SANDBOX', raising=False)
    agent = ClaudeCodeAgent()
    with pytest.raises(ValueError, match='non-root'):
        agent.preflight()
    agent.preflight(docker=True)
    command = agent.build_docker_command(
        _workspace(tmp_path), ['claude', '-p', 'test'], image='test-image', device_id=4,
        environment={'ASCEND_ARENA_CHIP': 'ascend910_93'},
    )
    assert command[:4] == ['docker', 'run', '--env', 'IS_SANDBOX=1']
    assert 'ASCEND_ARENA_CHIP' in command
    assert 'IS_SANDBOX' not in agent.setup_environment({})
