# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Prepare and run an Ascend Arena optimization workspace."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from auto_pipeline.core import AGENT_FAILED, Artifact, GeneratorInput, RunnerPrompt
from auto_pipeline.generator.ascend_arena.workspace import prepare_ascend_arena_workspace
from auto_pipeline.generator.base import Runner


class AscendArenaAgent:
    type = "ascend-arena"
    workspace_instructions = """# Ascend Arena Operator Optimization

- Operator: `{operator}`
- Editable source: `{source_dir}/`
- Goal: {goal}
- Do not modify `api_desc.md`, `reference.py`, `binding/`, `cases/`, `workspace.json`, or `validate.sh`.
- Run `./validate.sh` after meaningful changes and finish with a validated implementation.
"""

    def __init__(
        self,
        *,
        cann_bench_root: Path,
        runner: Runner,
        device_id: int | None = None,
        goal: str = "",
    ) -> None:
        self.cann_bench_root = Path(cann_bench_root).expanduser().resolve()
        self.runner = runner
        self.device_id = device_id
        self.goal = str(goal or "")

    def generate(self, task: GeneratorInput) -> Artifact:
        workdir = Path(task.workdir).expanduser().resolve()
        output_dir = Path(task.output_dir).expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if task.case is None:
                raise ValueError("Ascend Arena generator requires case metadata")
            workspace = prepare_ascend_arena_workspace(
                cann_bench_root=self.cann_bench_root,
                case=task.case,
                material=task.material,
                workdir=workdir,
                reports_dir=workdir / "kernel_eval",
                device_id=self.device_id,
                goal=self.goal,
                reuse_existing=True,
            )
            compile_entry = workspace.compile_entry
            operator_dir = workspace.operator_dir
            instructions = self.workspace_instructions.format(
                operator=task.case.operator,
                source_dir=operator_dir.name,
                goal=self.goal or "Improve performance without changing public behavior.",
            )
            (workdir / "AGENTS.md").write_text(instructions.rstrip() + "\n", encoding="utf-8")
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            return Artifact(
                status=AGENT_FAILED,
                workdir=output_dir,
                message=f"failed to prepare Ascend Arena workspace: {exc}",
                metadata={"ascend_arena_status": "prepare_failed"},
            )

        prompt = RunnerPrompt(
            text=(
                "Read AGENTS.md in the current workspace and complete the operator optimization. "
                "Work autonomously and keep all changes inside this workspace."
            ),
            cwd=workdir,
            output_dir=output_dir,
            timeout_sec=task.timeout_sec,
            env=dict(task.env),
            title=task.title,
            files={task_file.key: task_file.source_path for task_file in task.material.task_files},
            metadata={
                **dict(task.metadata),
                "generator": self.type,
                "operator_dir": str(operator_dir),
            },
        )
        runner_artifact = self.runner.run(prompt)
        metadata = {
            **dict(runner_artifact.metadata),
            "ascend_arena_status": "success" if runner_artifact.ok else runner_artifact.status,
            "compile_repo": compile_entry["compile-repo"],
            "compile_family": compile_entry.get("compile-family", ""),
            "workspace_operator_dir": str(operator_dir),
            "baseline_dir": str(workdir / "kernel_eval" / "init_results"),
        }
        if not runner_artifact.ok:
            return replace(runner_artifact, metadata=metadata)
        if not (operator_dir / "op_host").is_dir() or not (operator_dir / "op_kernel").is_dir():
            return replace(
                runner_artifact,
                status=AGENT_FAILED,
                message=f"generated operator is missing op_host/op_kernel: {operator_dir}",
                metadata={**metadata, "ascend_arena_status": "artifact_missing"},
            )
        return replace(
            runner_artifact,
            workdir=operator_dir,
            files={**dict(runner_artifact.files), "source_dir": operator_dir},
            metadata=metadata,
        )
