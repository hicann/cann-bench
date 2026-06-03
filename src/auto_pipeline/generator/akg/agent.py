"""AKG generator integration."""

from __future__ import annotations

import asyncio
import itertools
import os
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from auto_pipeline.core import GeneratorInput
from auto_pipeline.core import AGENT_FAILED, AGENT_SUCCESS, AGENT_TIMEOUT, Artifact, RunnerPrompt


_MAX_METADATA_STRING = 500
_MAX_METADATA_ITEMS = 8
_MAX_METADATA_DEPTH = 3
_AKG_RUNTIME_LOCK = threading.RLock()
_TASK_ID_COUNTER = itertools.count()


class AkgAgent:
    """Runs AKG's Python operator-generation workflows."""

    type = "akg-agent"

    def __init__(
        self,
        *,
        repo_root: Path | str,
        config_path: Optional[Path | str] = None,
        device_id: int = 0,
        backend: str = "ascend",
        arch: str = "ascend910b4",
        framework: str = "torch",
        codegen_target: str = "triton_ascend",
        workflow: str = "kernelgen_only_workflow",
        verify_timeout: Optional[int] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve() if config_path else None
        self.device_id = int(device_id)
        self.backend = str(backend)
        self.arch = str(arch)
        self.framework = str(framework)
        self.codegen_target = str(codegen_target)
        self.workflow = str(workflow)
        self.verify_timeout = verify_timeout
        self.env = {str(key): str(value) for key, value in dict(env or {}).items()}

    def generate(self, task: GeneratorInput) -> Artifact:
        metadata = dict(task.metadata)
        metadata.setdefault("op_name", task.material.op_name)
        if task.case is not None:
            metadata.setdefault("operator", task.case.operator)
            metadata.setdefault("task_dir", str(task.case.task_dir))
        prompt = RunnerPrompt(
            text="",
            cwd=task.workdir,
            output_dir=task.output_dir,
            timeout_sec=task.timeout_sec,
            env=dict(task.env),
            title=task.title,
            files={task_file.key: task_file.source_path for task_file in task.material.task_files},
            metadata=metadata,
        )
        return self.run(prompt)

    def run(self, prompt: RunnerPrompt) -> Artifact:
        prompt.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = prompt.output_dir / "akg-agent.log"

        try:
            return self._run(prompt, log_file)
        except (TimeoutError, asyncio.TimeoutError):
            timeout_sec = _timeout_sec(prompt.timeout_sec)
            message = f"AKG KernelGen timeout after {timeout_sec}s"
            _write_log(log_file, message + "\n" + traceback.format_exc())
            return Artifact(
                status=AGENT_TIMEOUT,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
                metadata={"akg_task_config": self._task_config_metadata()},
            )
        except Exception as exc:  # pragma: no cover - defensive integration boundary
            message = f"AKG KernelGen failed: {exc}"
            _write_log(log_file, message + "\n" + traceback.format_exc())
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
                metadata={"akg_task_config": self._task_config_metadata()},
            )

    def _run(self, prompt: RunnerPrompt, log_file: Path) -> Artifact:
        if not self.repo_root.is_dir():
            message = f"AKG repo root not found: {self.repo_root}"
            _write_log(log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
                metadata={"akg_task_config": self._task_config_metadata()},
            )

        python_root = _find_akg_python_root(self.repo_root)
        if python_root is None:
            message = f"AKG python root not found under: {self.repo_root}"
            _write_log(log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
                metadata={"akg_task_config": self._task_config_metadata()},
            )

        config_path = self.config_path or _default_config_path(python_root, self.workflow)
        if not config_path.is_file():
            message = f"AKG config path not found: {config_path}"
            _write_log(log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
                metadata={"akg_task_config": self._task_config_metadata()},
            )

        task_dir = _task_dir_from_prompt(prompt)
        bench_type = _bench_type_from_prompt(prompt)
        missing_files = _missing_task_files(task_dir, prompt=prompt, bench_type=bench_type)
        if missing_files:
            message = f"AKG task dir missing required files: {missing_files}"
            _write_log(log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
                metadata={"akg_task_config": self._task_config_metadata()},
            )

        merged_env = {**prompt.env, **self.env}
        timeout_sec = _timeout_sec(prompt.timeout_sec)
        deadline = _deadline(timeout_sec)
        with _AKG_RUNTIME_LOCK:
            with (
                _temporary_sys_path(python_root),
                _temporary_akg_modules(),
                _temporary_environ(merged_env),
            ):
                from akg_agents.core.worker.manager import register_local_worker
                from akg_agents.op.config.config_validator import load_config
                from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
                from akg_agents.utils.environment_check import check_env_for_task

                config = load_config(config_path=str(config_path))
                if bench_type == "cann":
                    config["cann_problem_dir"] = str(task_dir)
                config["bench_type"] = bench_type
                config["default_workflow"] = _inner_workflow_for(self.workflow)
                if self.verify_timeout is not None:
                    config["verify_timeout"] = self.verify_timeout
                if timeout_sec is not None:
                    config["workflow_timeout"] = timeout_sec

                check_env_for_task(
                    self.framework,
                    self.backend,
                    self.codegen_target,
                    config,
                    is_remote=False,
                )
                remaining = _remaining_timeout(deadline)
                _run_async(
                    register_local_worker([self.device_id], backend=self.backend, arch=self.arch),
                    timeout_sec=remaining,
                )
                if bench_type == "cann":
                    from akg_agents.op.utils.cann_utils import get_cann_task_desc_for_prompt

                    task_desc = get_cann_task_desc_for_prompt(task_dir)
                else:
                    task_desc = _stanford_task_desc_from_prompt(prompt, task_dir)

                op_name = _op_name_from_prompt(prompt, task_dir)
                task_id = f"auto_pipeline_{op_name}_{time.time_ns()}_{next(_TASK_ID_COUNTER)}"
                task = AIKGTask(
                    op_name=op_name,
                    task_desc=task_desc,
                    task_id=task_id,
                    backend=self.backend,
                    arch=self.arch,
                    dsl=self.codegen_target,
                    config=config,
                    framework=self.framework,
                    workflow=self.workflow,
                    bench_type=bench_type,
                )
                remaining = _remaining_timeout(deadline)
                _, success, final_state = _run_async(task.run(), timeout_sec=remaining)

        final_state = dict(final_state or {})
        code = str(final_state.get("coder_code") or "")
        metadata = {
            "akg_success": bool(success),
            "akg_final_state": _compact_final_state(final_state, task_id=task_id, op_name=op_name),
            "akg_task_config": self._task_config_metadata(),
        }

        if not success:
            reason = _failure_reason(final_state)
            message = "AKG KernelGen failed"
            if reason:
                message += f": {reason}"
            _write_log(log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
                metadata=metadata,
                output_text=code,
            )

        if not code:
            message = "AKG KernelGen returned empty coder_code"
            _write_log(log_file, message)
            return Artifact(
                status=AGENT_FAILED,
                workdir=prompt.output_dir,
                message=message,
                log_file=log_file,
                metadata=metadata,
            )

        code_file = prompt.output_dir / "akg_model.py"
        code_file.write_text(code, encoding="utf-8")
        message = "AKG KernelGen completed"
        _write_log(log_file, message)
        return Artifact(
            status=AGENT_SUCCESS,
            workdir=prompt.output_dir,
            message=message,
            files={"generated_code": code_file},
            log_file=log_file,
            metadata=metadata,
            output_text=code,
        )

    def _task_config_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "arch": self.arch,
            "codegen_target": self.codegen_target,
            "framework": self.framework,
            "workflow": self.workflow,
            "device_id": self.device_id,
        }


def _find_akg_python_root(repo_root: Path) -> Path | None:
    for candidate in (repo_root / "akg_agents" / "python", repo_root / "python"):
        if (candidate / "akg_agents").is_dir():
            return candidate
    return None


def _default_config_path(python_root: Path, workflow: str) -> Path:
    config_dir = python_root / "akg_agents" / "op" / "config"
    if workflow in {"adaptive_search", "adaptive_search_workflow"}:
        return config_dir / "triton_ascend_evolve_config.yaml"
    return config_dir / "triton_ascend_kernelgen_config.yaml"


def _inner_workflow_for(workflow: str) -> str:
    if workflow in {"adaptive_search", "adaptive_search_workflow"}:
        return "kernelgen_only_workflow"
    return workflow


def _task_dir_from_prompt(prompt: RunnerPrompt) -> Path:
    task_dir = prompt.metadata.get("task_dir")
    if task_dir:
        return Path(str(task_dir)).expanduser().resolve()
    task = prompt.files.get("task")
    if task:
        return Path(task).expanduser().resolve().parent
    proto = prompt.files.get("proto")
    if proto:
        return Path(proto).expanduser().resolve().parent
    return prompt.cwd.expanduser().resolve()


def _bench_type_from_prompt(prompt: RunnerPrompt) -> str:
    benchmark = str(
        prompt.metadata.get("benchmark")
        or prompt.metadata.get("bench_name")
        or ""
    ).strip().lower().replace("_", "-")
    if benchmark in {"stanford", "kernelbench", "kernel-bench"}:
        return "kernelbench"
    if prompt.files.get("task"):
        return "kernelbench"
    return "cann"


def _missing_task_files(task_dir: Path, *, prompt: RunnerPrompt, bench_type: str) -> list[str]:
    if bench_type == "kernelbench":
        task_path = _task_path_from_prompt(prompt, task_dir)
        if task_path is None or not task_path.is_file():
            return ["task_desc.py"]
        return []

    missing = [name for name in ("proto.yaml", "golden.py") if not (task_dir / name).is_file()]
    case_files = ("cases.yaml", "cases.yml", "cases.csv")
    if not any((task_dir / name).is_file() for name in case_files):
        missing.append("cases.yaml/cases.yml/cases.csv")
    return missing


def _task_path_from_prompt(prompt: RunnerPrompt, task_dir: Path) -> Path | None:
    task = prompt.files.get("task")
    if task:
        return Path(task).expanduser().resolve()
    candidate = task_dir / "task_desc.py"
    if candidate.is_file():
        return candidate.resolve()
    return None


def _stanford_task_desc_from_prompt(prompt: RunnerPrompt, task_dir: Path) -> str:
    task_path = _task_path_from_prompt(prompt, task_dir)
    if task_path is None or not task_path.is_file():
        raise FileNotFoundError(f"Stanford task_desc.py not found under: {task_dir}")
    return task_path.read_text(encoding="utf-8")


def _op_name_from_prompt(prompt: RunnerPrompt, task_dir: Path) -> str:
    return str(
        prompt.metadata.get("op_name")
        or prompt.metadata.get("operator")
        or task_dir.name
    )


def _compact_final_state(final_state: Mapping[str, Any], *, task_id: str, op_name: str) -> dict[str, Any]:
    payload = {
        "verifier_result": _compact_value(final_state.get("verifier_result")),
        "verifier_error": _compact_value(final_state.get("verifier_error")),
        "error": _compact_value(final_state.get("error")),
        "error_message": _compact_value(final_state.get("error_message")),
        "profile_res": _compact_value(final_state.get("profile_res")),
        "task_id": _compact_value(task_id),
        "op_name": _compact_value(op_name),
        "has_coder_code": bool(final_state.get("coder_code")),
    }
    if "task_kwargs" in final_state:
        task_kwargs = _compact_mapping(
            final_state["task_kwargs"],
            (
                "op_name",
                "task_desc",
                "task_id",
                "backend",
                "arch",
                "dsl",
                "framework",
                "workflow",
                "bench_type",
            ),
        )
        if "dsl" in task_kwargs:
            task_kwargs["codegen_target"] = task_kwargs.pop("dsl")
        payload["task_kwargs"] = task_kwargs
    if "task_config" in final_state:
        payload["task_config"] = _compact_mapping(
            final_state["task_config"],
            (
                "bench_type",
                "cann_problem_dir",
                "env_checked",
                "verify_timeout",
                "workflow_timeout",
                "default_workflow",
                "loaded_config_path",
            ),
        )
    return payload


def _failure_reason(final_state: Mapping[str, Any]) -> str:
    for key in ("error", "error_message", "verifier_error"):
        value = final_state.get(key)
        if value is None or (isinstance(value, str) and value == ""):
            continue
        compact = _compact_value(value)
        text = compact if isinstance(compact, str) else str(compact)
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if first_line:
            return first_line
    return ""


def _timeout_sec(value: Any) -> float | None:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _deadline(timeout_sec: float | None) -> float | None:
    if timeout_sec is None:
        return None
    return time.monotonic() + timeout_sec


def _remaining_timeout(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError()
    return remaining


async def _await_with_timeout(awaitable: Any, timeout_sec: float | None) -> Any:
    if timeout_sec is None:
        return await awaitable
    return await asyncio.wait_for(awaitable, timeout=timeout_sec)


def _run_async(awaitable: Any, *, timeout_sec: float | None = None) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_with_timeout(awaitable, timeout_sec))

    result: dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(_await_with_timeout(awaitable, timeout_sec))
        except BaseException as exc:  # pragma: no cover - re-raised in caller thread
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _compact_mapping(value: Any, keys: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: _compact_value(item)
        for key in keys
        if key in value
        for item in (value[key],)
    }


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) <= _MAX_METADATA_STRING:
            return value
        return value[:_MAX_METADATA_STRING] + "...<truncated>"
    if depth >= _MAX_METADATA_DEPTH:
        return _compact_summary(value)
    if isinstance(value, Mapping):
        return {
            str(key): _compact_value(item, depth=depth + 1)
            for key, item in list(value.items())[:_MAX_METADATA_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _compact_value(item, depth=depth + 1)
            for item in list(value)[:_MAX_METADATA_ITEMS]
        ]
    return _compact_summary(value)


def _compact_summary(value: Any) -> str:
    try:
        size = len(value)
    except TypeError:
        return f"<{type(value).__name__}>"
    return f"<{type(value).__name__} len={size}>"


@contextmanager
def _temporary_akg_modules() -> Iterator[None]:
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "akg_agents" or name.startswith("akg_agents.")
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in [
            module_name
            for module_name in sys.modules
            if module_name == "akg_agents" or module_name.startswith("akg_agents.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@contextmanager
def _temporary_sys_path(path: Path) -> Iterator[None]:
    original = list(sys.path)
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        sys.path[:] = original


@contextmanager
def _temporary_environ(env: Mapping[str, str]) -> Iterator[None]:
    original = dict(os.environ)
    try:
        os.environ.update({str(key): str(value) for key, value in env.items()})
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _write_log(log_file: Path, text: str) -> None:
    log_file.write_text(text.rstrip() + "\n", encoding="utf-8")
