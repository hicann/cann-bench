"""PyPTO orchestrator agent integration.

This agent runs the real PyPTO/OpenCode workflow:

    opencode run --agent pypto-op-orchestrator <initial prompt>

The generated PyPTO artifacts stay in the configured PyPTO workspace and are
returned as an ``Artifact``. Submission normalization remains the converter
stage's responsibility.
"""

from __future__ import annotations

import os
import json
import shutil
import subprocess
import time
import hashlib
from pathlib import Path
from typing import Mapping, Optional

from auto_pipeline.generator.opencode import OpenCodeAgent
from auto_pipeline.generator.opencode.exporter import make_session_title
from auto_pipeline.prompt.base import CaseMaterial
from auto_pipeline.prompt.builders import case_material_prompt_context
from auto_pipeline.core import GeneratorInput
from auto_pipeline.core import (
    AGENT_FAILED,
    AGENT_SUCCESS,
    AGENT_TIMEOUT,
    Artifact,
    RunnerPrompt,
)
from auto_pipeline.core import render_prompt_file


DEFAULT_PYPTO_AGENT = "pypto-op-orchestrator"
DEFAULT_PERF_ROUND = 3
PERF_ROUND_ENV = "PYPTO_PERF_ROUND"
_REQUIRED_STAGES = tuple(str(index) for index in range(1, 8))
_WORKTREE_MARKER = ".auto_pipeline_pypto_worktree.json"
_ORCHESTRATOR_TEMPLATE = Path(__file__).with_name("templates") / "orchestrator.j2"


class PyptoOrchestratorAgent:
    """Runs PyPTO's native seven-stage orchestrator agent."""

    type = "pypto"

    def __init__(
        self,
        *,
        pypto_repo_root: Path,
        workdir_root: str = "custom",
        opencode_bin: str = "opencode",
        opencode_model: str = "",
        agent: str = DEFAULT_PYPTO_AGENT,
        output_format: str = "default",
        perf_round: int = DEFAULT_PERF_ROUND,
        device_id: Optional[int] = None,
        device_mode: str = "normal",
        skip_if_done: bool = True,
        worktree_root: Optional[Path] = None,
        worktree_ref: str = "HEAD",
        extra_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.pypto_repo_root = Path(pypto_repo_root).expanduser().resolve()
        self.workdir_root = str(workdir_root or "custom").strip("/")
        self.opencode_bin = str(opencode_bin or "opencode")
        self.opencode_model = str(opencode_model or "")
        self.agent = str(agent or DEFAULT_PYPTO_AGENT)
        self.output_format = str(output_format or "default")
        self.opencode_runner = OpenCodeAgent(
            opencode_bin=self.opencode_bin,
            skill=self.agent,
            model=self.opencode_model,
            output_format=self.output_format,
            dangerously_skip_permissions=True,
        )
        self.perf_round = _perf_round_from_environment(perf_round)
        self.device_id = device_id
        self.device_mode = str(device_mode or "normal")
        self.skip_if_done = bool(skip_if_done)
        self.worktree_root = Path(worktree_root).expanduser().resolve() if worktree_root else None
        self.worktree_ref = str(worktree_ref or "HEAD")
        self.extra_env = {
            str(key): str(value)
            for key, value in dict(extra_env or {}).items()
        }

    def generate(self, task: GeneratorInput) -> Artifact:
        prompt = RunnerPrompt(
            text="",
            cwd=task.workdir,
            output_dir=task.output_dir,
            timeout_sec=task.timeout_sec,
            env=dict(task.env),
            title=task.title,
            files={task_file.key: task_file.source_path for task_file in task.material.task_files},
            metadata=dict(task.metadata),
        )
        return self._run_material(task.material, prompt)

    def _run_material(self, task_info: CaseMaterial, prompt: RunnerPrompt) -> Artifact:
        prompt.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = prompt.output_dir / f"{self.type}.log"

        if not self.pypto_repo_root.is_dir():
            message = f"PyPTO repo root not found: {self.pypto_repo_root}"
            _write_single_line_log(log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
            )

        try:
            run_repo_root, workspace_metadata = self._prepare_run_repo_root(prompt, task_info)
        except OSError as exc:
            message = f"failed to prepare PyPTO workspace root: {exc}"
            _write_single_line_log(log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
            )

        op_dir = run_repo_root / self.workdir_root / task_info.op_name
        artifacts = _expected_artifact_paths(task_info.op_name, op_dir)
        op_dir_rel = f"{self.workdir_root}/{task_info.op_name}"
        benchmark_context = case_material_prompt_context(task_info, op_dir_rel)
        pypto_prompt = _render_pypto_prompt(
            op_name=task_info.op_name,
            op_dir_rel=op_dir_rel,
            bench_name=task_info.bench_name,
            case_task_files_text=str(benchmark_context.get("case_task_files_text") or "- <none>"),
            case_detail_sections=str(benchmark_context.get("case_detail_sections") or ""),
            device_mode=self.device_mode,
            pool_device_id=self.device_id,
            perf_round=self.perf_round,
        )
        state = _read_orchestrator_state(op_dir)
        missing = _missing_artifacts(artifacts)
        required_stages = _required_stages(self.perf_round)
        if self.skip_if_done and not missing and _state_all_stages_completed(state, required_stages):
            message = "PyPTO workflow already completed; skipped."
            _write_single_line_log(log_file, message)
            return Artifact(
                status=AGENT_SUCCESS,
                workdir=op_dir,
                message=message,
                files={"source_dir": op_dir, **artifacts},
                log_file=log_file,
                metadata={
                    "pypto_status": "skipped",
                    "orchestrator_state": state or {},
                    "pypto_perf_round": self.perf_round,
                    "pypto_required_stages": list(required_stages),
                },
            )

        try:
            self._prepare_pypto_workspace(task_info, op_dir)
        except OSError as exc:
            message = f"failed to prepare PyPTO workspace: {exc}"
            _write_single_line_log(log_file, message)
            return Artifact(status=AGENT_FAILED, workdir=op_dir, message=message, log_file=log_file)

        session_title = make_session_title(task_info.op_name, phase="pypto")
        extra_env = dict(self.extra_env)
        if self.device_id is not None:
            extra_env["TILE_FWK_DEVICE_ID"] = str(self.device_id)
        tmpdir = op_dir / ".tmp"
        opencode_result = self.opencode_runner.run_opencode(
            prompt,
            cwd=run_repo_root,
            prompt_text=pypto_prompt,
            log_name=f"{self.type}.log",
            session_title=session_title,
            extra_env=extra_env,
            tmpdir=tmpdir,
            live_bridge=True,
            export_session=True,
        )

        if not opencode_result.started:
            return Artifact(
                status=opencode_result.status,
                workdir=op_dir,
                message=opencode_result.message,
                log_file=opencode_result.log_file,
                metadata={
                    "opencode_live_bridge": dict(opencode_result.live_bridge),
                    "opencode_session": dict(opencode_result.session_export),
                },
            )

        state = _read_orchestrator_state(op_dir)
        missing = _missing_artifacts(artifacts)
        metadata = {
            "pypto_status": "",
            "returncode": opencode_result.returncode,
            "timed_out": opencode_result.timed_out,
            "opencode_permission_external_directory": "deny",
            "op_name": task_info.op_name,
            "prompt_file": str(opencode_result.prompt_file),
            "opencode_live_bridge": dict(opencode_result.live_bridge),
            "opencode_session": dict(opencode_result.session_export),
            "orchestrator_state": state or {},
            "missing_artifacts": missing,
            "pypto_perf_round": self.perf_round,
            "pypto_required_stages": list(required_stages),
            **workspace_metadata,
        }

        if opencode_result.timed_out:
            message = (
                f"PyPTO orchestrator timed out after {prompt.timeout_sec}s; "
                f"missing artifacts: {missing or '(none)'}"
            )
            _append_log_footer(opencode_result.log_file, message)
            return Artifact(
                status=AGENT_TIMEOUT,
                workdir=op_dir,
                message=message,
                files=_existing_files({"source_dir": op_dir, **artifacts}),
                log_file=opencode_result.log_file,
                metadata={**metadata, "pypto_status": "timeout"},
            )

        if missing:
            message = f"PyPTO orchestrator exited code={opencode_result.returncode}; missing artifacts: {missing}"
            _append_log_footer(opencode_result.log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=op_dir,
                message=message,
                files=_existing_files({"source_dir": op_dir, **artifacts}),
                log_file=opencode_result.log_file,
                metadata={**metadata, "pypto_status": "artifact_missing"},
            )

        if opencode_result.returncode != 0:
            message = f"PyPTO orchestrator exited code={opencode_result.returncode}, although artifacts are present"
            _append_log_footer(opencode_result.log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=op_dir,
                message=message,
                files={"source_dir": op_dir, **artifacts},
                log_file=opencode_result.log_file,
                metadata={**metadata, "pypto_status": "subprocess_error"},
            )

        if not _state_all_stages_completed(state, required_stages):
            stage_display = ",".join(required_stages)
            message = f"PyPTO orchestrator state did not reach required stages completed: {stage_display}"
            _append_log_footer(opencode_result.log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=op_dir,
                message=message,
                files={"source_dir": op_dir, **artifacts},
                log_file=opencode_result.log_file,
                metadata={**metadata, "pypto_status": "blocked"},
            )

        message = f"PyPTO artifacts present and orchestrator required stages completed: {','.join(required_stages)}"
        _append_log_footer(opencode_result.log_file, message)
        return Artifact(
            status=AGENT_SUCCESS,
            workdir=op_dir,
            message=message,
            files={"source_dir": op_dir, **artifacts},
            log_file=opencode_result.log_file,
            metadata={**metadata, "pypto_status": "success"},
        )

    def _prepare_pypto_workspace(self, task_info: CaseMaterial, op_dir: Path) -> None:
        op_dir.mkdir(parents=True, exist_ok=True)
        for task_file in task_info.task_files:
            shutil.copy2(task_file.source_path, op_dir / task_file.target_name)
        require_target = op_dir / "REQUIRE.md"
        if task_info.require_path and task_info.require_path.is_file():
            shutil.copy2(task_info.require_path, require_target)
        elif not require_target.is_file():
            require_target.write_text(task_info.require_text, encoding="utf-8")

    def _prepare_run_repo_root(
        self,
        prompt: RunnerPrompt,
        task_info: CaseMaterial,
    ) -> tuple[Path, dict[str, object]]:
        if self.worktree_root is None:
            return self.pypto_repo_root, {
                "pypto_repo_root": str(self.pypto_repo_root),
                "pypto_run_repo_root": str(self.pypto_repo_root),
                "pypto_isolated_worktree": False,
            }

        if _is_relative_to(self.worktree_root, self.pypto_repo_root):
            raise OSError(
                f"worktree_root must not be inside pypto_repo_root: {self.worktree_root}"
            )

        worktree_name = _worktree_name(prompt, task_info)
        worktree_dir = self.worktree_root / worktree_name
        metadata = {
            "base_repo": str(self.pypto_repo_root),
            "worktree": str(worktree_dir),
            "worktree_ref": self.worktree_ref,
            "workdir_root": self.workdir_root,
            "op_name": task_info.op_name,
            "prompt_output_dir": str(prompt.output_dir),
            "created_at": int(time.time()),
        }
        _create_clean_git_worktree(
            base_repo=self.pypto_repo_root,
            worktree_dir=worktree_dir,
            ref=self.worktree_ref,
            marker_payload=metadata,
        )
        return worktree_dir, {
            "pypto_repo_root": str(self.pypto_repo_root),
            "pypto_run_repo_root": str(worktree_dir),
            "pypto_isolated_worktree": True,
            "pypto_worktree_root": str(self.worktree_root),
            "pypto_worktree_ref": self.worktree_ref,
        }


def _normalize_perf_round(value: object) -> int:
    if value is None or str(value).strip() == "":
        return DEFAULT_PERF_ROUND
    perf_round = int(value)
    if perf_round < 0:
        raise ValueError(f"perf_round must be non-negative, got: {value!r}")
    return perf_round


def _perf_round_from_environment(default: object = DEFAULT_PERF_ROUND) -> int:
    value = os.environ.get(PERF_ROUND_ENV)
    if value is None or value == "":
        value = default
    return _normalize_perf_round(value)


def _render_pypto_prompt(
    *,
    op_name: str,
    op_dir_rel: str,
    bench_name: str,
    case_task_files_text: str,
    case_detail_sections: str,
    device_mode: str,
    pool_device_id: Optional[int],
    perf_round: int,
) -> str:
    return render_prompt_file(
        _ORCHESTRATOR_TEMPLATE,
        bench_name=bench_name,
        op_name=op_name,
        op_dir_rel=op_dir_rel,
        case_task_files_text=case_task_files_text,
        case_detail_sections=case_detail_sections,
        device_mode=device_mode,
        pool_device_id=pool_device_id,
        perf_round=perf_round,
    )


def _expected_artifact_paths(op_name: str, op_dir: Path) -> dict[str, Path]:
    names = [
        f"{op_name}_impl.py",
        f"{op_name}_golden.py",
        f"test_{op_name}.py",
        "SPEC.md",
    ]
    return {name: op_dir / name for name in names}


def _missing_artifacts(artifacts: Mapping[str, Path]) -> list[str]:
    return [name for name, path in artifacts.items() if not Path(path).is_file()]


def _existing_files(files: Mapping[str, Path]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for name, path in files.items():
        path = Path(path)
        if path.exists():
            out[name] = path
    return out


def _read_orchestrator_state(op_dir: Path) -> Optional[dict]:
    state_file = op_dir / ".orchestrator_state.json"
    if not state_file.is_file():
        return None
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _required_stages(perf_round: int) -> tuple[str, ...]:
    if int(perf_round) <= 0:
        return tuple(str(index) for index in range(1, 7))
    return _REQUIRED_STAGES


def _state_all_stages_completed(state: Optional[dict], required_stages: tuple[str, ...]) -> bool:
    if not isinstance(state, dict):
        return False
    stage_status = state.get("stage_status")
    if not isinstance(stage_status, dict):
        return False
    return all(str(stage_status.get(key)).lower() == "completed" for key in required_stages)


def _worktree_name(prompt: RunnerPrompt, task_info: CaseMaterial) -> str:
    task_fingerprint = "|".join(
        str(task_file.source_path)
        for task_file in task_info.task_files
    )
    seed = "|".join(
        [
            str(prompt.output_dir.expanduser().resolve()),
            task_info.bench_name,
            task_info.op_name,
            task_fingerprint,
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return "__".join(
        [
            _safe_path_name(task_info.bench_name),
            _safe_path_name(task_info.op_name),
            digest,
        ]
    )


def _create_clean_git_worktree(
    *,
    base_repo: Path,
    worktree_dir: Path,
    ref: str,
    marker_payload: Mapping[str, object],
) -> None:
    if not _is_git_work_tree(base_repo):
        raise OSError(f"pypto_repo_root is not a git work tree: {base_repo}")

    if worktree_dir.exists():
        marker = worktree_dir / _WORKTREE_MARKER
        if not marker.is_file():
            raise OSError(
                f"refusing to reuse existing path without {_WORKTREE_MARKER}: {worktree_dir}"
            )
        _remove_existing_worktree(base_repo=base_repo, worktree_dir=worktree_dir)

    worktree_dir.parent.mkdir(parents=True, exist_ok=True)
    _run_git(
        base_repo,
        ["worktree", "add", "--detach", str(worktree_dir), ref],
        action=f"create isolated worktree {worktree_dir}",
    )
    marker_payload = dict(marker_payload)
    marker_payload["marker"] = _WORKTREE_MARKER
    (worktree_dir / _WORKTREE_MARKER).write_text(
        json.dumps(marker_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _is_git_work_tree(path: Path) -> bool:
    try:
        result = _run_git(path, ["rev-parse", "--is-inside-work-tree"])
    except OSError:
        return False
    return result.stdout.strip() == "true"


def _remove_existing_worktree(*, base_repo: Path, worktree_dir: Path) -> None:
    try:
        _run_git(
            base_repo,
            ["worktree", "remove", "--force", str(worktree_dir)],
            action=f"remove previous isolated worktree {worktree_dir}",
        )
    except OSError:
        pass
    if worktree_dir.exists():
        shutil.rmtree(worktree_dir)


def _run_git(
    repo: Path,
    args: list[str],
    *,
    action: str = "run git",
) -> subprocess.CompletedProcess[str]:
    command = ["git", "-C", str(repo), *args]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise OSError(f"{action} failed: {detail}")
    return result


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _safe_path_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(value))
    return (safe.strip("._-") or "item")[:64]


def _write_single_line_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{message}\n", encoding="utf-8")


def _append_log_footer(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {message}\n")
