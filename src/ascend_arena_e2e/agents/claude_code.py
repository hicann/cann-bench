# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Claude Code CLI workspace agent."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ascend_arena_e2e.agent_base import OperatorAgent, register_agent


@register_agent
class ClaudeCodeAgent(OperatorAgent):
    name = "claude-code"
    executable = "claude"
    default_model = "sonnet"
    default_model_reasoning_effort = "high"
    environment_variables = {
        "base_url": "ANTHROPIC_BASE_URL",
        "api_key": "ANTHROPIC_API_KEY",
        "model": "ANTHROPIC_MODEL",
    }

    def preflight(self, *, docker: bool = False) -> None:
        super().preflight(docker=docker)
        if not docker and os.geteuid() == 0 and os.environ.get("IS_SANDBOX") != "1":
            raise ValueError("Claude Code requires a non-root user on the host; use --docker for isolated root execution")

    def build_docker_command(self, *args, **kwargs):
        command = super().build_docker_command(*args, **kwargs)
        command[2:2] = ["--env", "IS_SANDBOX=1"]
        return command

    def setup_workspace_environment(self, workspace, environment, *, docker):
        if self.api_key:
            selected, other = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY") if self.auth_type == "bearer" else (
                "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
            environment[selected] = self.api_key
            environment.pop(other, None)
        elif environment.get("ANTHROPIC_AUTH_TOKEN"):
            environment.pop("ANTHROPIC_API_KEY", None)
        if self.api_key or self.base_url:
            config = workspace / "artifact" / "claude-config"
            config.mkdir(mode=0o700, parents=True, exist_ok=True)
            environment["CLAUDE_CONFIG_DIR"] = "/workspace/artifact/claude-config" if docker else str(config)

    def build_command(self, prompt: str) -> list[str]:
        command = [
            self.executable,
            "-p",
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format",
            "stream-json",
        ]
        if self.api_key or self.base_url:
            # Project/local settings can also contain an env block overriding the profile.
            command.extend(["--setting-sources", ""])
        if self.model:
            command.extend(["--model", self.model])
        if self.model_reasoning_effort:
            command.extend(["--effort", self.model_reasoning_effort])
        return [*command, *self.extra_args, prompt]

    def output_succeeded(self, log_file: Path) -> bool:
        for line in reversed(log_file.read_text(encoding="utf-8").splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "result":
                return event.get("subtype") == "success" and not event.get("is_error", False)
        return False

    def classify_error(self, log_file: Path) -> str:
        for line in log_file.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            message = event.get("message") or {}
            status = event.get("api_error_status") or (message.get("api_error_status") if isinstance(message, dict) else None)
            if str(status) in {"401", "403"}:
                return "authentication_failed"
            if event.get("is_error") or event.get("type") == "error":
                text = json.dumps(event).lower()
                if "authentication_failed" in text or "authentication_error" in text or "invalid api key" in text:
                    return "authentication_failed"
        return ""

    def should_log_output(self, line: str) -> bool:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return True
        return not (isinstance(event, dict) and event.get("type") == "system"
                    and event.get("subtype") == "thinking_tokens")
