# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Codex CLI workspace agent."""

from __future__ import annotations

import json
from pathlib import Path

from ascend_arena_e2e.agent_base import OperatorAgent, register_agent


@register_agent
class CodexAgent(OperatorAgent):
    name = "codex"
    executable = "codex"
    default_model = "gpt-5.4"
    environment_variables = {
        "base_url": "OPENAI_BASE_URL",
        "api_key": "OPENAI_API_KEY",
        "model": "OPENAI_MODEL",
    }

    def build_command(self, prompt: str) -> list[str]:
        command = [
            self.executable,
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--json",
        ]
        if self.base_url:
            command.extend(
                [
                    "--config",
                    'model_provider="openai-compatible"',
                    "--config",
                    'model_providers.openai-compatible.name="OpenAI compatible"',
                    "--config",
                    f"model_providers.openai-compatible.base_url={json.dumps(self.base_url)}",
                    "--config",
                    'model_providers.openai-compatible.env_key="OPENAI_API_KEY"',
                    "--config",
                    'model_providers.openai-compatible.wire_api="responses"',
                    "--config",
                    "model_providers.openai-compatible.supports_websockets=false",
                ]
            )
        if self.model:
            command.extend(["--model", self.model])
        if self.model_reasoning_effort:
            effort = json.dumps(self.model_reasoning_effort)
            command.extend(["--config", f"model_reasoning_effort={effort}"])
        return [*command, *self.extra_args, prompt]

    def output_succeeded(self, log_file: Path) -> bool:
        for line in reversed(log_file.read_text(encoding="utf-8").splitlines()):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in {"turn.completed", "turn.failed"}:
                return event["type"] == "turn.completed"
        return False
