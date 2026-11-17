# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Prepare an editable Ascend Arena workspace without running an agent."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from auto_pipeline.core import CannBenchCase
from auto_pipeline.prompt.base import CaseMaterial

_EXCLUDED_WORKSPACE_CASES = {"cases/complete_cases.json", "cases/profile_cases.json"}


@dataclass(frozen=True)
class AscendArenaWorkspace:
    workdir: Path
    operator_dir: Path
    artifact_dir: Path
    validate_script: Path
    reports_dir: Path
    compile_entry: dict[str, Any]
    device_id: int | None

    def to_dict(self, *, case: CannBenchCase) -> dict[str, Any]:
        return {
            "status": "prepared",
            "benchmark": "ascend_arena",
            "operator": case.operator,
            "workspace": str(self.workdir),
            "source_dir": str(self.operator_dir),
            "artifact_dir": str(self.artifact_dir),
            "task_dir": str(case.task_dir),
            "workspace_task_dir": str(self.workdir),
            "validate_script": str(self.validate_script),
            "reports_dir": str(self.reports_dir),
            "compile_repo": self.compile_entry.get("compile-repo", ""),
            "compile_family": self.compile_entry.get("compile-family", ""),
            "device_id": self.device_id,
            "baseline_present": (self.reports_dir / "init_results" / "status.json").is_file(),
        }


def prepare_ascend_arena_workspace(
    *,
    cann_bench_root: Path,
    case: CannBenchCase,
    material: CaseMaterial,
    workdir: Path,
    reports_dir: Path,
    device_id: int | None = None,
    goal: str = "",
    reuse_existing: bool = False,
    skip_initial_evaluation: bool = False,
    initial_evaluation_command: Sequence[str] | None = None,
) -> AscendArenaWorkspace:
    cann_bench_root = Path(cann_bench_root).expanduser().resolve()
    workdir = Path(workdir).expanduser().resolve()
    reports_dir = Path(reports_dir).expanduser().resolve()
    e2e_root = _resolve_e2e_root(cann_bench_root)
    compile_entry = _compile_entry(e2e_root, material.op_name)
    operator_source = _operator_source_dir(e2e_root, compile_entry, case.task_dir)
    operator_dir = workdir / material.op_name

    metadata_path = workdir / "workspace.json"
    if reuse_existing and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("operator") != case.operator or previous.get("device_id") != device_id:
            raise ValueError("workspace operator/device differs; use a new output directory")

    if operator_dir.exists() and not reuse_existing:
        raise FileExistsError(f"workspace operator directory already exists: {operator_dir}")
    workdir.mkdir(parents=True, exist_ok=True)
    artifact_dir = workdir / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if not operator_dir.exists():
        shutil.copytree(operator_source, operator_dir, symlinks=True)

    for task_file in material.task_files:
        if task_file.target_name in _EXCLUDED_WORKSPACE_CASES:
            continue
        target = workdir / task_file.target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(task_file.source_path, target)
    for target_name in _EXCLUDED_WORKSPACE_CASES:
        (workdir / target_name).unlink(missing_ok=True)
    (workdir / "REQUIRE.md").unlink(missing_ok=True)
    validate_script = _write_validate_script(
        cann_bench_root=cann_bench_root,
        case=case,
        operator=material.op_name,
        operator_dir=operator_dir,
        workdir=workdir,
        reports_dir=reports_dir,
        device_id=device_id,
    )
    workspace = AscendArenaWorkspace(
        workdir=workdir,
        operator_dir=operator_dir,
        artifact_dir=artifact_dir,
        validate_script=validate_script,
        reports_dir=reports_dir,
        compile_entry=compile_entry,
        device_id=device_id,
    )
    metadata = {**workspace.to_dict(case=case), "goal": goal}
    (workdir / "workspace.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if skip_initial_evaluation:
        return workspace
    if initial_evaluation_command is None:
        _run_initial_evaluation(workspace)
    else:
        _run_initial_evaluation(workspace, initial_evaluation_command)
    return workspace


def _run_initial_evaluation(
    workspace: AscendArenaWorkspace,
    command: Sequence[str] | None = None,
) -> None:
    status_path = workspace.reports_dir / "init_results" / "status.json"
    if status_path.is_file():
        try:
            if json.loads(status_path.read_text(encoding="utf-8")).get("success") is True:
                return
        except (OSError, json.JSONDecodeError):
            pass
    from ascend_arena_e2e.agent_base import OperatorAgent

    argv = list(command) if command is not None else [str(workspace.validate_script), "init"]
    cidfile = Path(argv[argv.index("--cidfile") + 1]) if "--cidfile" in argv else None
    process = subprocess.Popen(
        argv,
        cwd=workspace.workdir,
        start_new_session=True,
    )
    with OperatorAgent.track_process(process, cidfile):
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            f"initial evaluation failed with exit code {returncode}; "
            f"see {workspace.reports_dir / 'init_results'}"
        )


def _resolve_e2e_root(cann_bench_root: Path) -> Path:
    candidates = [cann_bench_root / "src" / "ascend_arena_e2e", cann_bench_root]
    for candidate in candidates:
        if (candidate / "dependency").is_dir() and (candidate / "ops.json").is_file():
            return candidate
    raise ValueError("CannBench root must contain src/ascend_arena_e2e/dependency and ops.json")


def _compile_entry(e2e_root: Path, operator: str) -> dict[str, Any]:
    payload = json.loads((e2e_root / "ops.json").read_text(encoding="utf-8"))
    entry = payload.get(operator) if isinstance(payload, dict) else None
    if entry is None and isinstance(payload, dict):
        entry = next((value for key, value in payload.items() if str(key).lower() == operator.lower()), None)
    if not isinstance(entry, dict):
        raise ValueError(f"operator is absent from Ascend Arena ops.json: {operator}")
    return entry


def _operator_source_dir(e2e_root: Path, entry: dict[str, Any], task_dir: Path) -> Path:
    op_name = str(entry.get("compile-op-name") or entry.get("op") or "")
    task_source = task_dir / op_name
    if (task_source / "op_host").is_dir() and (task_source / "op_kernel").is_dir():
        return task_source
    raise ValueError(
        f"benchmark operator source directory not found: {task_source}; "
        "run python3 scripts/setup_ascend_arena_dependencies.py"
    )


def _write_validate_script(
    *,
    cann_bench_root: Path,
    case: CannBenchCase,
    operator: str,
    operator_dir: Path,
    workdir: Path,
    reports_dir: Path,
    device_id: int | None,
) -> Path:
    evaluation_script = cann_bench_root / "scripts" / "run_evaluation.sh"
    if not evaluation_script.is_file():
        raise ValueError(f"CannBench evaluation script not found: {evaluation_script}")
    # Keep editable source and reports in the movable workspace, but read the
    # complete benchmark assets from the checkout selected by CANN_BENCH_ROOT.
    default_root = _shell_double_quote_value(str(cann_bench_root))
    operator_arg = shlex.quote(operator)
    operator_segment = _shell_double_quote_value(operator)
    try:
        task_relative = case.task_dir.resolve().relative_to(cann_bench_root)
    except ValueError:
        task_arg = shlex.quote(str(case.task_dir.resolve()))
    else:
        task_segment = _shell_double_quote_value(task_relative.as_posix())
        task_arg = f'"$CANN_BENCH_ROOT/{task_segment}"'
    device_line = (
        f'ASCEND_ARENA_DEVICE_ID="${{ASCEND_ARENA_DEVICE_ID:-{_shell_double_quote_value(str(device_id))}}}"\n'
        if device_id is not None
        else ""
    )
    command = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f'CANN_BENCH_ROOT="${{CANN_BENCH_ROOT:-{default_root}}}"\n'
        f"{device_line}"
        'if [[ "${1:-}" == "only-html" ]]; then\n'
        '    export PYTHONPATH="$CANN_BENCH_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"\n'
        '    exec python -m kernel_eval.optimization_trajectory "$SCRIPT_DIR/kernel_eval"\n'
        "fi\n"
        'MODE_ARGS=()\n'
        'if [[ "${1:-}" == "init" ]]; then MODE_ARGS+=(--init-only); shift; fi\n'
        'exec "$CANN_BENCH_ROOT/scripts/run_evaluation.sh" --run_optimize '
        f'--source-dir "$SCRIPT_DIR/{operator_segment}" '
        f"--task-dir {task_arg} "
        f"--operator {operator_arg} "
        '--reports-dir "$SCRIPT_DIR/kernel_eval" '
        '--baseline-dir "$SCRIPT_DIR/kernel_eval/init_results" "${MODE_ARGS[@]}"'
    )
    path = workdir / "validate.sh"
    if device_id is not None:
        command += ' --device-id "$ASCEND_ARENA_DEVICE_ID"'
    command += ' "$@"'
    path.write_text(command + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _shell_double_quote_value(value: str) -> str:
    """Escape a value embedded inside a generated double-quoted assignment."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
