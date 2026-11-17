# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Base class, registry, and configuration loader for workspace agents."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shlex
import shutil
import signal
import math
import subprocess
import threading
from contextlib import contextmanager
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Mapping, Sequence

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AGENT_CONFIG_PATH = Path(__file__).resolve().parent / "agents" / ".env"
DEFAULT_DOCKER_IMAGE = "cann-bench:cann9.1.0-910b-latest"
_AGENT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
_PROFILE_FIELDS = {"harness", "api_key", "base_url", "model", "model_reasoning_effort", "auth_type"}
_ACTIVE_RUNS: dict[int, tuple[subprocess.Popen[str], Path | None]] = {}
_ACTIVE_RUNS_LOCK = threading.Lock()
_RUNS_CANCELLED = threading.Event()


@dataclass(frozen=True)
class AgentRunResult:
    agent: str
    harness: str
    workspace: Path
    returncode: int
    log_file: Path
    timed_out: bool = False
    evaluation: dict[str, object] | None = None
    error_kind: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and (self.evaluation is None or self.evaluation.get("success") is True)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent": self.agent,
            "harness": self.harness,
            "workspace": str(self.workspace),
            "returncode": self.returncode,
            "ok": self.ok,
            "log_file": str(self.log_file),
            "timed_out": self.timed_out,
            "evaluation": self.evaluation,
            "error_kind": self.error_kind,
        }


class OperatorAgent(ABC):
    """Base class for a command-line agent operating inside one workspace."""

    name: ClassVar[str] = ""
    executable: ClassVar[str] = ""
    environment_variables: ClassVar[Mapping[str, str]] = {}
    default_model: ClassVar[str] = ""
    default_model_reasoning_effort: ClassVar[str] = ""

    def __init__(
        self,
        *,
        base_url: str = "",
        api_key: str = "",
        auth_type: str = "",
        model: str = "",
        model_reasoning_effort: str = "",
        profile_name: str = "",
        extra_args: Sequence[str] = (),
        timeout_seconds: float = 0,
    ) -> None:
        self.harness = type(self).name
        self.name = str(profile_name or self.harness)
        self.base_url = self._configured_value("base_url", base_url)
        self.api_key = self._configured_value("api_key", api_key)
        self.auth_type = auth_type or "api-key"
        if self.auth_type not in {"api-key", "bearer"}:
            raise ValueError("agent auth_type must be api-key or bearer")
        self.model = self._configured_value("model", model, self.default_model)
        self.model_reasoning_effort = self._configured_value(
            "model_reasoning_effort",
            model_reasoning_effort,
            self.default_model_reasoning_effort,
        )
        self.extra_args = [str(item) for item in extra_args]
        self.timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds < 0:
            raise ValueError("agent timeout must be a finite non-negative number of seconds")

    def _configured_value(self, attribute: str, explicit: str, default: str = "") -> str:
        variable = self.environment_variables.get(attribute, "")
        return str(explicit or os.environ.get(variable, "") or default)

    def preflight(self, *, docker: bool = False) -> None:
        """Check local launch requirements before collecting the baseline."""
        executable = "docker" if docker else self.executable
        if not shutil.which(executable):
            raise ValueError(f"agent executable not found: {executable}")
        if not docker:
            try:
                version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=10)
                print(f"[agent] {self.name}: {(version.stdout or version.stderr).strip().splitlines()[0]}", flush=True)
            except (OSError, subprocess.TimeoutExpired, IndexError):
                print(f"[agent] warning: cannot detect {self.name} CLI version", flush=True)

    def setup_environment(self, current: Mapping[str, str] | None = None) -> dict[str, str]:
        """Return the subprocess environment without mutating the parent process."""

        env = dict(os.environ if current is None else current)
        for attribute, variable in self.environment_variables.items():
            value = str(getattr(self, attribute, "") or "")
            if value:
                env[variable] = value
        return env

    def get_workspace_path(self, workspace: Path | str) -> Path:
        """Resolve and validate a prepared operator workspace."""

        path = Path(workspace).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"agent workspace does not exist: {path}")
        missing = [name for name in ("workspace.json", "validate.sh") if not (path / name).is_file()]
        if missing:
            raise ValueError(f"agent workspace is missing {', '.join(missing)}: {path}")
        return path

    def setup_agents_md(self, workspace: Path) -> None:
        """Write the default AGENTS.md; subclasses may override this method."""

        metadata = json.loads((workspace / "workspace.json").read_text(encoding="utf-8"))
        operator = str(metadata.get("operator") or workspace.parent.name)
        source_dir = Path(str(metadata.get("source_dir") or operator)).name
        goal = str(metadata.get("goal") or "Improve performance without changing public behavior. Every profile case must achieve at least a 5% performance improvement, with no performance regression in any case.")
        instructions = f"""# Operator Optimization
- Operator: `{operator}`
- Editable source: `{source_dir}/`
- Goal: {goal}
- Do not modify `api_desc.md`, `reference.py`, `binding/`, `cases/`, `workspace.json`, or `validate.sh`.
- Keep the implementation buildable for the existing Ascend C project.
- Baseline timings: `kernel_eval/init_results/result.json` (microseconds per case).
- Candidate feedback: `kernel_eval/ascend_arena_*/results/` for correctness and `kernel_eval/ascend_arena_*/comparison.json` for speedup; inspect both before the next edit.
- Start with one small, buildable change and validate it before further investigation.
- Treat gains within measured noise as inconclusive; check `noise_band` and `significant` in comparison.json.
- Run `./validate.sh` after meaningful changes and finish with a validated implementation.
"""
        (workspace / "AGENTS.md").write_text(instructions.rstrip() + "\n", encoding="utf-8")

    def setup_prompt(self, workspace: Path, prompt: str = "") -> str:
        """Write this agent's workspace instructions and build its prompt."""

        self.setup_agents_md(workspace)
        text = str(prompt or "").strip() or (
            "Read AGENTS.md and optimize the operator in this workspace. "
            "Run ./validate.sh after meaningful changes and finish with a validated implementation."
        )
        from kernel_eval.optimize import workspace_evaluation
        previous = workspace_evaluation(workspace)
        if previous.get("reason") != "no_candidate_evaluation":
            text += ("\n\nWorkspace continuation state: " + json.dumps(previous, ensure_ascii=False)
                     + "\nInspect existing source and evaluation results before making another change. "
                     "Validate the final source and finish after successful validation.")
        return text

    def setup_workspace_environment(self, workspace: Path, environment: dict[str, str], *, docker: bool) -> None:
        """Customize workspace-local CLI configuration without changing the user's settings."""

    @abstractmethod
    def build_command(self, prompt: str) -> list[str]:
        """Return the argv used to run this agent."""

    def output_succeeded(self, log_file: Path) -> bool:
        """Validate agent-specific output after a zero exit code."""

        return True

    def should_log_output(self, line: str) -> bool:
        """Return whether one subprocess output line should be logged and displayed."""

        return True

    def build_docker_command(
        self,
        workspace: Path,
        command: Sequence[str],
        *,
        image: str,
        device_id: int,
        environment: Mapping[str, str],
    ) -> list[str]:
        """Wrap an agent command in one disposable, single-device container."""

        docker = [
            "docker",
            "run",
            "--rm",
            "--cidfile",
            str(workspace / "artifact" / f".{self.name}.container-id"),
            "--ipc=host",
            "--userns=host",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
        ]
        for device in (f"/dev/davinci{device_id}", "/dev/davinci_manager", "/dev/devmm_svm", "/dev/hisi_hdc"):
            docker.extend(["--device", device])
        for source, target in (
            (Path("/usr/local/Ascend/driver"), "/usr/local/Ascend/driver"),
            (Path("/usr/local/dcmi"), "/usr/local/dcmi"),
            (Path("/usr/local/bin/npu-smi"), "/usr/local/bin/npu-smi"),
            (Path("/etc/ascend_install.info"), "/etc/ascend_install.info"),
        ):
            if source.exists():
                docker.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])

        docker.extend(
            [
                "--env",
                "ASCEND_ARENA_DEVICE_ID=0",
                "--env",
                "CANN_BENCH_ROOT=/cann-bench",
                "--env",
                "PYTHONPATH=/cann-bench/src",
            ]
        )
        for variable in dict.fromkeys([*self.environment_variables.values(), "ANTHROPIC_AUTH_TOKEN", "ASCEND_ARENA_CHIP", "CLAUDE_CONFIG_DIR"]):
            if environment.get(variable):
                docker.extend(["--env", variable])

        docker.extend(["--mount", f"type=bind,src={REPOSITORY_ROOT},dst=/cann-bench,readonly"])
        if AGENT_CONFIG_PATH.is_file():
            docker.extend(
                [
                    "--mount",
                    "type=bind,src=/dev/null,dst=/cann-bench/src/ascend_arena_e2e/agents/.env,readonly",
                ]
            )
        docker.extend(["--mount", f"type=bind,src={workspace},dst=/workspace"])
        for name in (
            "AGENTS.md",
            "api_desc.md",
            "reference.py",
            "workspace.json",
            "validate.sh",
            "cases",
            "binding",
        ):
            source = workspace / name
            if source.exists():
                docker.extend(["--mount", f"type=bind,src={source},dst=/workspace/{name},readonly"])
        return [*docker, "--workdir", "/workspace", image, *command]

    @staticmethod
    def _stop_container(cidfile: Path) -> None:
        try:
            container_id = cidfile.read_text(encoding="utf-8").strip()
        except OSError:
            return
        if re.fullmatch(r"[0-9a-f]{12,64}", container_id):
            try:
                stopped = subprocess.run(
                    ["docker", "stop", "--time", "5", container_id],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                )
                if stopped.returncode:
                    print(f"[agent] could not stop container {container_id}; retained {cidfile}", flush=True)
                    return
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"[agent] container stop failed ({type(exc).__name__}); retained {cidfile}", flush=True)
                return
        cidfile.unlink(missing_ok=True)

    @classmethod
    def cancel_active_runs(cls) -> None:
        """Stop active agent containers after a parent interruption."""

        with _ACTIVE_RUNS_LOCK:
            _RUNS_CANCELLED.set()
            active = list(_ACTIVE_RUNS.values())
        for process, cidfile in active:
            if cidfile is not None:
                cls._stop_container(cidfile)
            cls._stop_process(process)

    @staticmethod
    def reset_cancellation() -> None:
        _RUNS_CANCELLED.clear()

    @classmethod
    @contextmanager
    def track_process(cls, process, cidfile=None):
        with _ACTIVE_RUNS_LOCK:
            _ACTIVE_RUNS[id(process)] = (process, cidfile)
            cancelled = _RUNS_CANCELLED.is_set()
        try:
            if cancelled:
                if cidfile is not None:
                    cls._stop_container(cidfile)
                cls._stop_process(process)
            yield process
        finally:
            if process.poll() is None:
                if cidfile is not None:
                    cls._stop_container(cidfile)
                cls._stop_process(process)
            with _ACTIVE_RUNS_LOCK:
                _ACTIVE_RUNS.pop(id(process), None)

    def classify_error(self, log_file: Path) -> str:
        return ""

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        import psutil

        try:
            children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        for child in reversed(children):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # Children may outlive the CLI, including validators holding stdout open.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for child in children:
            try:
                if child.is_running():
                    child.kill()
            except psutil.NoSuchProcess:
                pass

    def run(
        self,
        workspace: Path | str,
        *,
        prompt: str = "",
        device_id: int | None = None,
        docker_image: str = "",
    ) -> AgentRunResult:
        from ascend_arena_e2e.timing import timed_run
        from dataclasses import replace
        from kernel_eval.optimize import workspace_evaluation
        from kernel_eval.optimization_trajectory import write_optimization_trajectory_report

        path = self.get_workspace_path(workspace)
        try:
            with timed_run(path / "artifact" / "timing.jsonl", "agent", path.parent.name) as timing:
                result = self._run(workspace, prompt=prompt, device_id=device_id, docker_image=docker_image)
                result = replace(result, evaluation=workspace_evaluation(path))
                timing["returncode"] = result.returncode if result.returncode else (0 if result.ok else 1)
                timing["timed_out"] = result.timed_out
        finally:
            write_optimization_trajectory_report(path / "kernel_eval")
        return result

    def _run(
        self,
        workspace: Path | str,
        *,
        prompt: str = "",
        device_id: int | None = None,
        docker_image: str = "",
    ) -> AgentRunResult:
        path = self.get_workspace_path(workspace)
        prompt_text = self.setup_prompt(path, prompt)
        if docker_image:
            prompt_text = (
                f"{prompt_text.rstrip()}\n\n"
                "This command runs inside Docker. Use /workspace as the workspace root "
                "and /cann-bench as the repository root. Do not use host absolute paths "
                "from workspace.json."
            )
        artifact_dir = path / "artifact"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        log_file = artifact_dir / f"{self.name}.log"

        environment = self.setup_environment()
        self.setup_workspace_environment(path, environment, docker=bool(docker_image))
        command = [str(item) for item in self.build_command(prompt_text)]
        cidfile = path / "artifact" / f".{self.name}.container-id" if docker_image else None
        if cidfile is not None:
            self._stop_container(cidfile)
        if docker_image:
            if device_id is None:
                raise ValueError("docker agent execution requires a device id")
            command = self.build_docker_command(
                path,
                command,
                image=docker_image,
                device_id=device_id,
                environment=environment,
            )
        elif device_id is not None:
            environment["ASCEND_ARENA_DEVICE_ID"] = str(device_id)
        executable = shutil.which(command[0]) if command else None
        expired = threading.Event()

        with log_file.open("w", encoding="utf-8") as log:
            display_command = ["<prompt>" if item == prompt_text else item for item in command]
            log.write(
                f"cwd: {path}\n"
                f"command: {shlex.join(display_command)}\n\n"
                f"prompt:\n{prompt_text.rstrip()}\n\n"
                "agent output:\n"
            )
            log.flush()
            if executable is None:
                log.write(f"agent executable not found: {command[0] if command else ''}\n")
                return AgentRunResult(self.name, self.harness, path, 127, log_file)
            command[0] = executable
            try:
                process = subprocess.Popen(
                    command,
                    cwd=path,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
                with _ACTIVE_RUNS_LOCK:
                    _ACTIVE_RUNS[id(process)] = (process, cidfile)
                    cancelled = _RUNS_CANCELLED.is_set()
                if cancelled:
                    self._stop_process(process)
                pid_file = artifact_dir / "agent.pid"
                pid_file.write_text(str(process.pid) + "\n", encoding="utf-8")
                def expire():
                    expired.set()
                    if cidfile is not None:
                        self._stop_container(cidfile)
                    self._stop_process(process)
                timer = threading.Timer(self.timeout_seconds, expire) if self.timeout_seconds else None
                if timer is not None:
                    timer.daemon = True
                    timer.start()
                try:
                    assert process.stdout is not None
                    for line in process.stdout:
                        if not self.should_log_output(line):
                            continue
                        log.write(line)
                        log.flush()
                        print(f"[{path.parent.name}][{self.name}] {line.rstrip()}", flush=True)
                    returncode = process.wait()
                finally:
                    if process.poll() is None:
                        self._stop_process(process)
                    if timer is not None:
                        timer.cancel()
                        timer.join()
                    pid_file.unlink(missing_ok=True)
                    with _ACTIVE_RUNS_LOCK:
                        _ACTIVE_RUNS.pop(id(process), None)
                    if cidfile is not None:
                        self._stop_container(cidfile)
            except OSError as exc:
                log.write(f"agent failed to start: {exc}\n")
                log.flush()
                returncode = 126
            if returncode == 0 and not self.output_succeeded(log_file):
                returncode = 1
            if expired.is_set():
                message = f"[{self.name}] timeout after {self.timeout_seconds:g}s"
                print(message, flush=True)
                log.write(f"\n{message}\n")
                returncode = 124
            log.write(f"\nagent exit code: {returncode}\n")
            log.flush()
        return AgentRunResult(self.name, self.harness, path, returncode, log_file, expired.is_set(),
                              error_kind=self.classify_error(log_file) if returncode else "")


_AGENTS: dict[str, type[OperatorAgent]] = {}


def register_agent(agent_type: type[OperatorAgent]) -> type[OperatorAgent]:
    if not isinstance(agent_type, type) or not issubclass(agent_type, OperatorAgent):
        raise TypeError("registered agent must inherit OperatorAgent")
    name = str(agent_type.name or "").strip().lower()
    if not name:
        raise ValueError("registered agent must define a non-empty name")
    if _AGENT_NAME.fullmatch(name) is None:
        raise ValueError("agent name may contain only letters, numbers, hyphens, and underscores")
    existing = _AGENTS.get(name)
    if existing is not None and existing is not agent_type:
        raise ValueError(f"agent is already registered: {name}")
    agent_type.name = name
    _AGENTS[name] = agent_type
    return agent_type


def available_agents() -> list[str]:
    return sorted(_AGENTS)


def load_agent_profiles(path: Path | None = None) -> dict[str, dict[str, str]]:
    config_path = AGENT_CONFIG_PATH if path is None else Path(path)
    if not config_path.is_file():
        return {}
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read agent config {config_path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"agent config must contain a YAML mapping: {config_path}")

    profiles = {}
    for raw_name, raw_profile in payload.items():
        name = str(raw_name or "").strip().lower()
        if _AGENT_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid agent profile name: {raw_name!r}")
        if not isinstance(raw_profile, dict):
            raise ValueError(f"agent profile must be a mapping: {name}")
        unknown = set(raw_profile) - _PROFILE_FIELDS
        if unknown:
            raise ValueError(f"unknown fields in agent profile {name}: {', '.join(sorted(unknown))}")
        profile = {key: str(raw_profile.get(key) or "").strip() for key in _PROFILE_FIELDS}
        if not profile["harness"]:
            raise ValueError(f"agent profile is missing harness: {name}")
        profiles[name] = profile
    return profiles


def _load_agent_type(spec: str) -> type[OperatorAgent]:
    name = str(spec or "").strip()
    registered = _AGENTS.get(name.lower())
    if registered is not None:
        return registered
    if ":" not in name:
        raise ValueError(f"unknown harness {name!r}; available: {', '.join(available_agents())}")
    module_name, class_name = name.rsplit(":", 1)
    if not module_name or not class_name:
        raise ValueError(f"invalid agent import path {name!r}; expected module:Class")
    try:
        candidate = getattr(importlib.import_module(module_name), class_name)
        return register_agent(candidate)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot load agent {name!r}: {exc}") from exc


def create_agent(name: str, **kwargs: object) -> OperatorAgent:
    requested = str(name or "").strip()
    profile_name = requested.lower()
    profile = load_agent_profiles().get(profile_name)
    harness = profile["harness"] if profile else requested
    values: dict[str, object] = {
        key: value for key, value in (profile or {}).items() if key != "harness" and value
    }
    values.update(kwargs)
    if profile:
        values["profile_name"] = profile_name
    agent_type = _load_agent_type(harness)
    try:
        return agent_type(**values)
    except TypeError as exc:
        raise ValueError(f"cannot initialize agent {requested!r}: {exc}") from exc


def add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--agent",
        help="agent profile from agents/.env, built-in harness name, or module:Class; omitted means prepare only",
    )
    parser.add_argument("--agent-prompt", default="", help="optional optimization prompt override")
    parser.add_argument("--agent-timeout", type=float, default=0, metavar="SECONDS",
                        help="maximum agent session time including evaluation; 0 means unlimited")
    parser.add_argument("--docker", action="store_true", help="run one operator per disposable Docker container")
    parser.add_argument("--docker-image", default=DEFAULT_DOCKER_IMAGE, help="agent execution image used by --docker")


def agent_from_args(args: argparse.Namespace) -> OperatorAgent | None:
    name = str(getattr(args, "agent", "") or "")
    return create_agent(name, timeout_seconds=getattr(args, "agent_timeout", 0)) if name else None


importlib.import_module("ascend_arena_e2e.agents")
