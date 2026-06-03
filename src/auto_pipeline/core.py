"""Core contracts and orchestration for the auto pipeline."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Protocol, TYPE_CHECKING

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

if TYPE_CHECKING:
    from auto_pipeline.prompt.base import CaseMaterial
    from auto_pipeline.converter.base import Converter
    from auto_pipeline.generator.base import Runner

AGENT_SUCCESS = "success"
AGENT_FAILED = "failed"
AGENT_NOT_FOUND = "not_found"
AGENT_TIMEOUT = "timeout"


@dataclass(frozen=True)
class RunnerPrompt:
    """Prompt material sent to a low-level runner."""

    text: str
    cwd: Path
    output_dir: Path
    timeout_sec: int = 7200
    env: Dict[str, str] = field(default_factory=dict)
    title: str = ""
    files: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Artifact:
    """Normalized artifact returned by a generator or runner."""

    status: str
    workdir: Path
    message: str = ""
    files: Dict[str, Path] = field(default_factory=dict)
    log_file: Optional[Path] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_text: str = ""

    @property
    def ok(self) -> bool:
        return self.status == AGENT_SUCCESS


@dataclass(frozen=True)
class Submission:
    """Submission directory accepted by cann-bench kernel_eval."""

    kind: str
    operator: str
    source_dir: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CannBenchCase:
    """Task context loaded from cann-bench tasks."""

    bench_name: str
    task_dir: Path
    operator: str
    rel_path: str
    files: Dict[str, Path]
    metadata: Dict[str, Any] = field(default_factory=dict)


TaskContext = CannBenchCase


@dataclass(frozen=True)
class EvalTarget:
    """Evaluation target, either one operator or a task directory."""

    bench_name: str
    task_dir: Path
    rel_path: str
    task_selector: str
    operator: str = ""
    files: Dict[str, Path] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def read_yaml_mapping(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def read_case_preview(path: Path, *, limit: int = 3) -> List[Dict[str, Any]]:
    """Read a small case preview for prompts without owning evaluation."""

    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        cases = data.get("cases") if isinstance(data, dict) else None
        if not isinstance(cases, list):
            return []
        return [case for case in cases[:limit] if isinstance(case, dict)]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for _, row in zip(range(limit), csv.DictReader(handle))]

    return []

class ParentProcessSignal(SystemExit):
    """Raised when the pipeline receives a terminating parent signal."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        self.signal_name = _signal_name(self.signum)
        super().__init__(128 + self.signum)


@contextlib.contextmanager
def cleanup_process_on_exit(
    process: subprocess.Popen,
    *,
    grace_sec: float = 10.0,
    on_signal: Optional[Callable[[int], None]] = None,
    match_environ: Optional[Mapping[str, str]] = None,
) -> Iterator[None]:
    """Terminate a child process family on timeout, Ctrl-C, SIGTERM, or errors."""

    tracker = _ProcessFamilyTracker(process, match_environ=match_environ)
    cleaned = False

    def cleanup_once() -> None:
        nonlocal cleaned
        if cleaned:
            return
        cleaned = True
        tracker.terminate(grace_sec=grace_sec)

    installed: dict[int, object] = {}

    def handler(signum: int, _frame: Optional[FrameType]) -> None:
        if on_signal is not None:
            on_signal(signum)
        cleanup_once()
        raise ParentProcessSignal(signum)

    if threading.current_thread() is threading.main_thread():
        for signum in _termination_signals():
            try:
                installed[signum] = signal.getsignal(signum)
                signal.signal(signum, handler)
            except (OSError, RuntimeError, ValueError):
                installed.pop(signum, None)

    tracker.start()
    try:
        yield
    except BaseException:
        if process.poll() is None:
            cleanup_once()
        raise
    finally:
        cleanup_once()
        tracker.stop()
        for signum, previous in installed.items():
            try:
                signal.signal(signum, previous)
            except (OSError, RuntimeError, ValueError):
                pass


def terminate_process_family(
    process: subprocess.Popen,
    *,
    grace_sec: float = 10.0,
    pgid: Optional[int] = None,
    descendants: Optional[dict[int, int]] = None,
) -> None:
    """SIGTERM then SIGKILL a process group plus any snapshotted descendants."""

    root_pid = int(process.pid)
    tracked_descendants = dict(descendants or {})
    tracked_descendants.update(_descendant_snapshot(root_pid))
    if pgid is None:
        pgid = _safe_getpgid(root_pid)
    group_targeted = pgid is not None and pgid != os.getpgrp()

    _send_group_or_process(root_pid, pgid, signal.SIGTERM)
    _send_signal_to_snapshotted_pids(tracked_descendants, signal.SIGTERM)

    deadline = time.monotonic() + max(float(grace_sec), 0.0)
    while time.monotonic() < deadline:
        if (
            process.poll() is not None
            and not _snapshot_has_live_pids(tracked_descendants)
            and not (group_targeted and _pgid_has_members(pgid))
        ):
            break
        time.sleep(0.1)

    if process.poll() is None or (group_targeted and _pgid_has_members(pgid)):
        _send_group_or_process(root_pid, pgid, signal.SIGKILL)
    _send_signal_to_snapshotted_pids(tracked_descendants, signal.SIGKILL)

    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


class _ProcessFamilyTracker:
    def __init__(
        self,
        process: subprocess.Popen,
        *,
        match_environ: Optional[Mapping[str, str]],
    ) -> None:
        self.process = process
        self.root_pid = int(process.pid)
        self.pgid = _safe_getpgid(self.root_pid)
        self.match_environ = {
            str(key): str(value)
            for key, value in dict(match_environ or {}).items()
            if str(key) and str(value)
        }
        self.descendants: dict[int, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.refresh()
        if not Path("/proc").is_dir():
            return
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def terminate(self, *, grace_sec: float) -> None:
        self.refresh()
        with self._lock:
            descendants = dict(self.descendants)
        terminate_process_family(
            self.process,
            grace_sec=grace_sec,
            pgid=self.pgid,
            descendants=descendants,
        )

    def refresh(self) -> None:
        snapshot = _descendant_snapshot(self.root_pid)
        snapshot.update(_environ_match_snapshot(self.match_environ))
        if not snapshot:
            return
        with self._lock:
            self.descendants.update(snapshot)

    def _poll(self) -> None:
        while not self._stop.wait(0.5):
            self.refresh()
            if self.process.poll() is not None:
                self.refresh()
                return


def _termination_signals() -> tuple[int, ...]:
    signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)
    return tuple(signals)


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def _safe_getpgid(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _pgid_has_members(pgid: Optional[int]) -> bool:
    if pgid is None:
        return False
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            pid = int(stat_path.parent.name)
            if pid == os.getpid():
                continue
            if os.getpgid(pid) == pgid:
                stat = _read_proc_stat(pid)
                if stat is None or stat[2] != "Z":
                    return True
        except (OSError, ValueError):
            continue
    return False


def _send_group_or_process(pid: int, pgid: Optional[int], signum: int) -> None:
    try:
        if pgid is not None and pgid != os.getpgrp():
            os.killpg(pgid, signum)
        else:
            os.kill(pid, signum)
    except ProcessLookupError:
        return
    except OSError:
        try:
            os.kill(pid, signum)
        except OSError:
            pass


def _send_signal_to_snapshotted_pids(snapshot: dict[int, int], signum: int) -> None:
    for pid, start_time in snapshot.items():
        if not _pid_matches_start_time(pid, start_time):
            continue
        try:
            os.kill(pid, signum)
        except OSError:
            pass


def _snapshot_has_live_pids(snapshot: dict[int, int]) -> bool:
    return any(_pid_matches_start_time(pid, start_time) for pid, start_time in snapshot.items())


def _descendant_snapshot(root_pid: int) -> dict[int, int]:
    entries: dict[int, tuple[int, int, str]] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            pid = int(stat_path.parent.name)
            stat = _read_proc_stat(pid)
        except (OSError, ValueError):
            continue
        if stat is not None:
            entries[pid] = stat

    children: dict[int, list[int]] = {}
    for pid, (ppid, _start_time, _state) in entries.items():
        children.setdefault(ppid, []).append(pid)

    out: dict[int, int] = {}
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        entry = entries.get(pid)
        if entry is None:
            continue
        _ppid, start_time, _state = entry
        out[pid] = start_time
        stack.extend(children.get(pid, []))
    return out


def _environ_match_snapshot(match_environ: Mapping[str, str]) -> dict[int, int]:
    if not match_environ:
        return {}
    out: dict[int, int] = {}
    current_pid = os.getpid()
    for environ_path in Path("/proc").glob("[0-9]*/environ"):
        try:
            pid = int(environ_path.parent.name)
            if pid == current_pid:
                continue
            content = environ_path.read_bytes()
        except (OSError, ValueError):
            continue
        env = _parse_environ(content)
        if any(env.get(key) != value for key, value in match_environ.items()):
            continue
        stat = _read_proc_stat(pid)
        if stat is not None and stat[2] != "Z":
            out[pid] = stat[1]
    return out


def _parse_environ(content: bytes) -> dict[str, str]:
    env: dict[str, str] = {}
    for item in content.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        try:
            env[key.decode("utf-8", errors="surrogateescape")] = value.decode(
                "utf-8",
                errors="surrogateescape",
            )
        except UnicodeError:
            continue
    return env


def _pid_matches_start_time(pid: int, start_time: int) -> bool:
    stat = _read_proc_stat(pid)
    return stat is not None and stat[1] == start_time and stat[2] != "Z"


def _read_proc_stat(pid: int) -> Optional[tuple[int, int, str]]:
    try:
        text = Path("/proc") / str(pid) / "stat"
        content = text.read_text(encoding="utf-8")
        fields = content.rsplit(") ", 1)[1].split()
        state = fields[0]
        ppid = int(fields[1])
        start_time = int(fields[19])
    except (IndexError, OSError, ValueError):
        return None
    return ppid, start_time, state

@lru_cache(maxsize=None)
def _prompt_environment(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def render_prompt_file(path: Path, **context: Any) -> str:
    template_path = Path(path).expanduser().resolve()
    return _prompt_environment(str(template_path.parent)).get_template(template_path.name).render(**context).strip()


def render_case_preview_json(case_preview: object) -> str:
    return json.dumps(case_preview or [], ensure_ascii=False, indent=2)

DEFAULT_CANN_BENCH_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_TASK_FILES = ("proto.yaml", "golden.py")
_CASE_FILENAMES = ("cases.yaml", "cases.yml", "cases.csv")


@dataclass(frozen=True)
class CannBenchEvalResult:
    """Result returned by a kernel_eval subprocess."""

    returncode: int
    command: List[str]
    reports_dir: Path
    stdout: str = ""
    stderr: str = ""
    report_files: List[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CannBenchClient:
    """Thin boundary for all direct interactions with cann-bench."""

    def __init__(
        self,
        cann_bench_root: Optional[Path] = None,
        *,
        python_executable: str = sys.executable,
        env: Optional[Mapping[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> None:
        self.cann_bench_root = Path(cann_bench_root or DEFAULT_CANN_BENCH_ROOT).expanduser().resolve()
        self.python_executable = python_executable
        self.extra_env = dict(env or {})
        self.timeout_sec = timeout_sec

    @property
    def tasks_root(self) -> Path:
        return self.cann_bench_root / "tasks"

    def load_case(self, bench_name: str, task_selector: str) -> CannBenchCase:
        """Load one cann-bench operator case into the benchmark pipeline."""

        if bench_name.lower() == "stanford":
            return self._load_stanford_case(bench_name, task_selector)

        task_dir = self._resolve_task_dir(task_selector)
        self._validate_task_dir(task_dir)

        proto_path = task_dir / "proto.yaml"
        proto = read_yaml_mapping(proto_path)
        operator_meta = proto.get("operator") if isinstance(proto, dict) else {}
        if not isinstance(operator_meta, dict):
            operator_meta = {}

        cases_path = self._find_cases_file(task_dir)
        operator = str(operator_meta.get("name") or task_dir.name)
        files = {
            "proto": proto_path,
            "golden": task_dir / "golden.py",
            "cases": cases_path,
        }
        desc_path = task_dir / "desc.md"
        if desc_path.is_file():
            files["desc"] = desc_path

        metadata: Dict[str, Any] = {
            "proto": proto,
            "operator": operator_meta,
            "schema": operator_meta.get("schema"),
            "category": operator_meta.get("category"),
            "difficulty": operator_meta.get("difficulty"),
            "case_format": cases_path.suffix.lstrip("."),
            "case_preview": read_case_preview(cases_path),
            "task_selector": task_selector,
        }
        return CannBenchCase(
            bench_name=bench_name,
            task_dir=task_dir,
            operator=operator,
            rel_path=self._task_rel_path(task_dir),
            files=files,
            metadata=metadata,
        )

    def load_eval_target(self, bench_name: str, task_selector: str) -> EvalTarget:
        """Load an evaluation target, accepting one operator or a directory."""

        task_dir = self._resolve_task_dir(task_selector)
        if self._is_single_task_dir(task_dir):
            case = self.load_case(bench_name, task_selector)
            return EvalTarget(
                bench_name=case.bench_name,
                task_dir=case.task_dir,
                rel_path=case.rel_path,
                task_selector=task_selector,
                operator=case.operator,
                files=case.files,
                metadata={**case.metadata, "multi_operator": False},
            )

        return EvalTarget(
            bench_name=bench_name,
            task_dir=task_dir,
            rel_path=self._task_rel_path(task_dir),
            task_selector=task_selector,
            metadata={
                "task_selector": task_selector,
                "multi_operator": True,
                "task_count": len(self._discover_task_dirs(task_dir)),
            },
        )

    def build_eval_command(
        self,
        *,
        bench_name: str,
        source_dir: Path,
        task_selector: str,
        reports_dir: Path,
        device_id: Optional[int] = None,
        extra_args: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Build the cann-bench kernel_eval command without executing it."""

        extra_args_list = list(extra_args or [])
        eval_task_selector = task_selector
        auto_operator = None
        if bench_name.lower() == "stanford":
            eval_task_selector, auto_operator = self._stanford_eval_selector(task_selector)

        command = [
            self.python_executable,
            "-m",
            "kernel_eval.cli",
            "eval",
            "--bench-name",
            bench_name,
            "--source-dir",
            str(Path(source_dir)),
            "--task-dir",
            eval_task_selector,
            "--reports-dir",
            str(Path(reports_dir)),
        ]
        if device_id is not None:
            command.extend(["--device-id", str(device_id)])
        if auto_operator and "--operator" not in extra_args_list:
            command.extend(["--operator", auto_operator])
        if extra_args_list:
            command.extend(extra_args_list)
        return command

    def eval_submission(
        self,
        *,
        bench_name: str,
        source_dir: Path,
        task_selector: str,
        reports_dir: Path,
        device_id: Optional[int] = None,
        extra_args: Optional[Iterable[str]] = None,
    ) -> CannBenchEvalResult:
        """Run kernel_eval for an already prepared cann-bench source_dir."""

        source_dir = Path(source_dir).expanduser().resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"source_dir not found: {source_dir}")

        reports_dir = Path(reports_dir).expanduser().resolve()
        reports_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_eval_command(
            bench_name=bench_name,
            source_dir=source_dir,
            task_selector=task_selector,
            reports_dir=reports_dir,
            device_id=device_id,
            extra_args=extra_args,
        )
        completed = _run_captured_process(
            command,
            cwd=str(self.cann_bench_root),
            env=self._build_env(),
            timeout=self.timeout_sec,
        )
        return CannBenchEvalResult(
            returncode=completed.returncode,
            command=command,
            reports_dir=reports_dir,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            report_files=self._collect_report_files(reports_dir),
        )

    def _resolve_task_dir(self, task_selector: str) -> Path:
        selector = Path(task_selector).expanduser()
        candidates = []
        if selector.is_absolute():
            candidates.append(selector)
        else:
            candidates.extend([self.cann_bench_root / selector, self.tasks_root / selector])

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_dir():
                return resolved
        display = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"cann-bench task not found: {task_selector}; tried: {display}")

    def _load_stanford_case(self, bench_name: str, task_selector: str) -> CannBenchCase:
        py_path = self._resolve_stanford_task_file(task_selector)
        operator = self._stanford_operator_name(py_path.stem)
        task_id = self._stanford_task_id(py_path)
        return CannBenchCase(
            bench_name=bench_name,
            task_dir=py_path.parent,
            operator=operator,
            rel_path=task_id,
            files={"task": py_path},
            metadata={
                "task_selector": task_selector,
                "task_id": task_id,
                "py_path": str(py_path),
                "operator": {"name": operator},
                "schema": "",
            },
        )

    def _stanford_eval_selector(self, task_selector: str) -> tuple:
        py_path = self._resolve_stanford_task_file(task_selector)
        return str(self._stanford_bench_root_for(py_path)), self._stanford_operator_name(py_path.stem)

    def _resolve_stanford_task_file(self, task_selector: str) -> Path:
        selector = Path(task_selector).expanduser()
        candidates = []
        if selector.is_absolute():
            candidates.append(selector)
        else:
            candidates.append(self.cann_bench_root / selector)
            default_root = self.cann_bench_root / "thirdparty" / "KernelBench" / "KernelBench"
            if selector.suffix == ".py":
                candidates.append(default_root / selector)
            else:
                candidates.append(default_root / f"{selector}.py")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.is_file() and resolved.suffix == ".py":
                return resolved
        display = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"stanford task file not found: {task_selector}; tried: {display}")

    def _stanford_task_id(self, py_path: Path) -> str:
        bench_root = self._stanford_bench_root_for(py_path)
        try:
            return str(py_path.with_suffix("").relative_to(bench_root))
        except ValueError:
            return py_path.stem

    def _stanford_bench_root_for(self, py_path: Path) -> Path:
        default_root = (self.cann_bench_root / "thirdparty" / "KernelBench" / "KernelBench").resolve()
        try:
            py_path.relative_to(default_root)
            return default_root
        except ValueError:
            if py_path.parent.name.startswith("level"):
                return py_path.parent.parent
            return py_path.parent

    def _stanford_operator_name(self, stem: str) -> str:
        name = re.sub(r"^\d+_", "", stem)
        name = re.sub(r"_\d+$", "", name)
        return "".join(part.capitalize() for part in name.split("_") if part)

    def _validate_task_dir(self, task_dir: Path) -> None:
        missing = [name for name in _REQUIRED_TASK_FILES if not (task_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"invalid cann-bench task dir: {task_dir}; missing: {', '.join(missing)}"
            )
        self._find_cases_file(task_dir)

    def _is_single_task_dir(self, task_dir: Path) -> bool:
        return all((task_dir / name).is_file() for name in _REQUIRED_TASK_FILES) and any(
            (task_dir / name).is_file() for name in _CASE_FILENAMES
        )

    def _discover_task_dirs(self, root: Path) -> List[Path]:
        return sorted(path for path in root.rglob("*") if path.is_dir() and self._is_single_task_dir(path))

    def _find_cases_file(self, task_dir: Path) -> Path:
        for name in _CASE_FILENAMES:
            path = task_dir / name
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"invalid cann-bench task dir: {task_dir}; missing cases.yaml/cases.yml/cases.csv"
        )

    def _task_rel_path(self, task_dir: Path) -> str:
        for root in (self.tasks_root, self.cann_bench_root):
            try:
                return str(task_dir.relative_to(root))
            except ValueError:
                continue
        return task_dir.name

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        pythonpath = []
        src_dir = self.cann_bench_root / "src"
        if src_dir.is_dir():
            pythonpath.append(str(src_dir))
        if env.get("PYTHONPATH"):
            pythonpath.append(env["PYTHONPATH"])
        if pythonpath:
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
        env.update(self.extra_env)
        return env

    def _collect_report_files(self, reports_dir: Path) -> List[Path]:
        if not reports_dir.is_dir():
            return []
        return sorted(
            path
            for path in reports_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".md", ".csv"}
        )


def _run_captured_process(
    command: List[str],
    *,
    cwd: str,
    env: Dict[str, str],
    timeout: Optional[int],
) -> subprocess.CompletedProcess[str]:
    env = dict(env)
    cleanup_token = uuid.uuid4().hex
    env["AUTO_PIPELINE_SUBPROCESS_TOKEN"] = cleanup_token
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with cleanup_process_on_exit(
        process,
        match_environ={"AUTO_PIPELINE_SUBPROCESS_TOKEN": cleanup_token},
    ):
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            terminate_process_family(process)
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(
                cmd=exc.cmd,
                timeout=exc.timeout,
                output=stdout,
                stderr=stderr,
            ) from exc
    return subprocess.CompletedProcess(
        args=command,
        returncode=process.returncode if process.returncode is not None else -9,
        stdout=stdout or "",
        stderr=stderr or "",
    )

@dataclass(frozen=True)
class GeneratorInput:
    """Structured case material passed from the benchmark layer to a generator."""

    case: Optional[CannBenchCase]
    material: CaseMaterial
    workdir: Path
    output_dir: Path
    timeout_sec: int = 7200
    env: Dict[str, str] = field(default_factory=dict)
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class Generator(Protocol):
    """Generator that can produce an artifact from a benchmark case."""

    type: str

    def generate(self, task: GeneratorInput) -> Artifact:
        ...


class PromptGenerator:
    """Generic generator wrapper for runners that only know prompt execution."""

    def __init__(self, runner: Runner) -> None:
        self.runner = runner
        self.type = getattr(runner, "type", "runner")

    def generate(self, task: GeneratorInput) -> Artifact:
        return self.runner.run(_build_generic_prompt(task))


def _build_generic_prompt(task: GeneratorInput) -> RunnerPrompt:
    task_files = "\n".join(
        f"- {task_file.key}: {task_file.source_path}"
        for task_file in task.material.task_files
    ) or "- <none>"
    text = "\n".join(
        [
            f"请为 benchmark `{task.material.bench_name}` 的算子 `{task.material.op_name}` 生成实现。",
            "",
            "任务输入文件:",
            task_files,
            "",
            "任务需求:",
            task.material.require_text,
        ]
    )
    return RunnerPrompt(
        text=text,
        cwd=task.workdir,
        output_dir=task.output_dir,
        timeout_sec=task.timeout_sec,
        env=dict(task.env),
        title=task.title,
        files={task_file.key: task_file.source_path for task_file in task.material.task_files},
        metadata=dict(task.metadata),
    )

@dataclass(frozen=True)
class PipelineRunResult:
    """Result of one benchmark pipeline run."""

    case: CannBenchCase
    submission: Submission
    eval_result: CannBenchEvalResult
    generator_prompt_file: Optional[Path] = None
    converter_prompt_file: Optional[Path] = None
    generated_artifact: Optional[Artifact] = None
    conversion_artifact: Optional[Artifact] = None
    status: str = "success"
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "success" and self.eval_result.ok


class BenchmarkPipeline:
    """Orchestrates Case -> Generator -> Convert -> Eval."""

    def __init__(self, *, bench_name: str, client: CannBenchClient) -> None:
        self.bench_name = str(bench_name or "cann")
        self.client = client

    def run_from_artifact(
        self,
        *,
        selector: str,
        artifact: Artifact,
        converter: Converter,
        submission_dir: Path,
        reports_dir: Path,
        conversion_runner: Optional[Runner] = None,
        conversion_workdir: Optional[Path] = None,
        device_id: Optional[int] = None,
        extra_eval_args: Optional[Iterable[str]] = None,
    ) -> PipelineRunResult:
        """Resume after a generator has already produced an artifact."""

        case = self.client.load_case(self.bench_name, selector)
        return self._convert_and_eval(
            case=case,
            generated_artifact=artifact,
            converter=converter,
            submission_dir=submission_dir,
            reports_dir=reports_dir,
            conversion_runner=conversion_runner,
            conversion_workdir=conversion_workdir,
            device_id=device_id,
            extra_eval_args=extra_eval_args,
        )

    def run_case(
        self,
        *,
        selector: str,
        generator: Generator,
        converter: Converter,
        workdir: Path,
        submission_dir: Path,
        reports_dir: Path,
        conversion_runner: Optional[Runner] = None,
        conversion_workdir: Optional[Path] = None,
        timeout_sec: int = 7200,
        env: Optional[Mapping[str, str]] = None,
        device_id: Optional[int] = None,
        extra_eval_args: Optional[Iterable[str]] = None,
    ) -> PipelineRunResult:
        case = self.client.load_case(self.bench_name, selector)
        generator_input = self._build_generator_input(
            case=case,
            generator=generator,
            workdir=workdir,
            timeout_sec=timeout_sec,
            env=env,
        )
        generated_artifact = generator.generate(generator_input)
        generator_prompt_file = generator_input.output_dir / "PROMPT.md"
        if not generator_prompt_file.is_file():
            generator_prompt_file = None

        if not generated_artifact.ok:
            message = generated_artifact.message or "generator failed"
            return self._failed_result(
                case=case,
                submission_kind=self.bench_name,
                submission_dir=submission_dir,
                reports_dir=reports_dir,
                status="generation_failed",
                message=message,
                generated_artifact=generated_artifact,
                generator_prompt_file=generator_prompt_file,
            )

        return self._convert_and_eval(
            case=case,
            generated_artifact=generated_artifact,
            converter=converter,
            submission_dir=submission_dir,
            reports_dir=reports_dir,
            conversion_runner=conversion_runner,
            conversion_workdir=conversion_workdir,
            device_id=device_id,
            extra_eval_args=extra_eval_args,
            generator_prompt_file=generator_prompt_file,
        )

    def _convert_and_eval(
        self,
        *,
        case: CannBenchCase,
        generated_artifact: Artifact,
        converter: Converter,
        submission_dir: Path,
        reports_dir: Path,
        conversion_runner: Optional[Runner],
        conversion_workdir: Optional[Path],
        device_id: Optional[int],
        extra_eval_args: Optional[Iterable[str]],
        generator_prompt_file: Optional[Path] = None,
    ) -> PipelineRunResult:
        conversion = converter.convert(
            self.bench_name,
            case,
            generated_artifact,
            output_dir=submission_dir,
            runner=conversion_runner,
            workdir=conversion_workdir,
        )
        if not conversion.ok or conversion.submission is None:
            message = conversion.artifact.message or "converter failed"
            return self._failed_result(
                case=case,
                submission_kind=self.bench_name,
                submission_dir=submission_dir,
                reports_dir=reports_dir,
                status="conversion_failed",
                message=message,
                generated_artifact=generated_artifact,
                conversion_artifact=conversion.artifact,
                generator_prompt_file=generator_prompt_file,
                converter_prompt_file=conversion.prompt_file,
            )

        eval_result = self.client.eval_submission(
            bench_name=self.bench_name,
            source_dir=conversion.submission.source_dir,
            task_selector=str(case.metadata.get("task_selector") or case.rel_path),
            reports_dir=reports_dir,
            device_id=device_id,
            extra_args=extra_eval_args,
        )
        return PipelineRunResult(
            case=case,
            submission=conversion.submission,
            eval_result=eval_result,
            generator_prompt_file=generator_prompt_file,
            converter_prompt_file=conversion.prompt_file,
            generated_artifact=generated_artifact,
            conversion_artifact=conversion.artifact if conversion_runner is not None else None,
            status=AGENT_SUCCESS,
        )

    def _build_generator_input(
        self,
        *,
        case: CannBenchCase,
        generator: Generator,
        workdir: Path,
        timeout_sec: int,
        env: Optional[Mapping[str, str]],
    ) -> GeneratorInput:
        workdir = Path(workdir).expanduser().resolve()
        material = build_case_material(case)
        return GeneratorInput(
            case=case,
            material=material,
            workdir=workdir,
            output_dir=workdir / "artifact",
            timeout_sec=timeout_sec,
            env={str(key): str(value) for key, value in dict(env or {}).items()},
            title=f"{generator.type}:{case.operator}",
            metadata={
                "generator": generator.type,
                "benchmark": self.bench_name,
                "operator": case.operator,
                "rel_path": case.rel_path,
                "task_dir": str(case.task_dir),
                "schema": case.metadata.get("schema") or "",
                "case_preview": case.metadata.get("case_preview"),
            },
        )

    def _failed_result(
        self,
        *,
        case: CannBenchCase,
        submission_kind: str,
        submission_dir: Path,
        reports_dir: Path,
        status: str,
        message: str,
        generated_artifact: Optional[Artifact] = None,
        conversion_artifact: Optional[Artifact] = None,
        generator_prompt_file: Optional[Path] = None,
        converter_prompt_file: Optional[Path] = None,
    ) -> PipelineRunResult:
        return PipelineRunResult(
            case=case,
            submission=Submission(
                kind=submission_kind,
                operator=case.operator,
                source_dir=Path(submission_dir).expanduser().resolve(),
                metadata={
                    "generated_status": generated_artifact.status if generated_artifact else "",
                    "conversion_status": conversion_artifact.status if conversion_artifact else "",
                },
            ),
            eval_result=CannBenchEvalResult(
                returncode=1,
                command=[],
                reports_dir=Path(reports_dir).expanduser().resolve(),
                stderr=message,
            ),
            generator_prompt_file=generator_prompt_file,
            converter_prompt_file=converter_prompt_file,
            generated_artifact=generated_artifact,
            conversion_artifact=conversion_artifact,
            status=status,
            message=message,
        )


def build_case_material(case: CannBenchCase) -> CaseMaterial:
    from auto_pipeline.prompt.registry import build_case_material as _build_case_material

    return _build_case_material(case)


def create_converter(source_generator: str, target_benchmark: str, cfg: Mapping[str, Any]) -> Converter:
    from auto_pipeline.converter.registry import create_converter as _create_converter

    return _create_converter(source_generator, target_benchmark, cfg)


def create_generator(generator_type: str, cfg: Mapping[str, Any]) -> Generator:
    from auto_pipeline.generator.registry import create_generator as _create_generator

    return _create_generator(generator_type, cfg)


def create_runner(runner_type: str, cfg: Mapping[str, Any]) -> Runner:
    from auto_pipeline.generator.registry import create_runner as _create_runner

    return _create_runner(runner_type, cfg)

DEFAULT_OUTPUT_ROOT = Path("benchmark_runs")
DEFAULT_TIMEOUT_SEC = 7200
DEFAULT_OP_TIMEOUT_SEC = 3600
PERF_SOURCE_ENV = "CANN_BENCH_PERF_SOURCE"
TILE_LIB_ENV = "PTO_TILE_LIB_CODE_PATH"


@dataclass(frozen=True)
class ConfiguredTask:
    name: str
    selector: str
    root_dir: Path

    @property
    def workdir(self) -> Path:
        return self.root_dir / "work"

    @property
    def convert_dir(self) -> Path:
        return self.root_dir / "convert"

    @property
    def submission_dir(self) -> Path:
        return self.root_dir / "submission"

    @property
    def reports_dir(self) -> Path:
        return self.root_dir / "kernel_eval"

    @property
    def result_path(self) -> Path:
        return self.root_dir / "benchmark_result.json"


@dataclass(frozen=True)
class SimpleConfig:
    output: Path
    bench_name: str
    bench_root: Path
    agent_type: str
    workspace: Path
    agent_options: Mapping[str, Any]
    model: str
    devices: tuple[int, ...]
    parallel: int
    tasks: tuple[ConfiguredTask, ...]

    @property
    def batch_report_path(self) -> Path:
        return self.output / "batch_result.json"


def run_from_config(config_path: Path, *, runtime: Optional[Mapping[str, Any]] = None) -> int:
    cfg = read_yaml_mapping(Path(config_path))
    entries = run_cases_from_mapping(cfg, config_path=Path(config_path), runtime=runtime)
    return 0 if entries and all(entry.get("ok") for entry in entries) else 1


def run_cases_from_mapping(
    cfg: Mapping[str, Any],
    *,
    config_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    runtime: Optional[Mapping[str, Any]] = None,
) -> list:
    spec = _parse_config(cfg, config_path=config_path, runtime=runtime)
    report = Path(report_path).expanduser().resolve() if report_path else spec.batch_report_path
    if len(set(spec.devices)) != len(spec.devices):
        raise ValueError("devices must not contain duplicate device ids")
    if spec.parallel > 1 and not spec.devices:
        raise ValueError("devices is required when parallel is greater than 1")

    if len(spec.tasks) == 1 or spec.parallel <= 1:
        entries = _run_cases_serial(spec, report_path=report)
    else:
        max_workers = min(len(spec.tasks), spec.parallel, len(spec.devices) or spec.parallel)
        entries = _run_cases_with_device_pool(spec, max_workers=max_workers, report_path=report)

    write_batch_report(entries, report, output=spec.output)
    return entries


def run_from_mapping(
    cfg: Mapping[str, Any],
    *,
    config_path: Optional[Path] = None,
    runtime: Optional[Mapping[str, Any]] = None,
) -> PipelineRunResult:
    spec = _parse_config(cfg, config_path=config_path, runtime=runtime)
    if len(spec.tasks) != 1:
        raise ValueError("run_from_mapping requires exactly one benchmark task; use run_cases_from_mapping")
    device_id = spec.devices[0] if spec.devices else None
    result = _run_task(spec, spec.tasks[0], device_id=device_id)
    write_pipeline_report(result, spec.tasks[0].result_path)
    return result


def _run_cases_serial(spec: SimpleConfig, *, report_path: Path) -> list:
    entries = []
    for index, task in enumerate(spec.tasks):
        device_id = _device_for_index(index, spec.devices)
        entries.append(_run_one_task_entry(spec, task, device_id=device_id))
        write_batch_report(entries, report_path, output=spec.output)
    return entries


def _run_cases_with_device_pool(spec: SimpleConfig, *, max_workers: int, report_path: Path) -> list:
    if max_workers <= 0:
        raise ValueError("parallel must be positive")

    entries = [_pending_entry(task) for task in spec.tasks]
    write_batch_report(entries, report_path, output=spec.output)

    pending = deque(range(len(spec.tasks)))
    available_devices = deque(spec.devices or tuple(range(max_workers)))
    active: dict[Any, tuple[int, Optional[int]]] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while pending or active:
            while pending and available_devices and len(active) < max_workers:
                index = pending.popleft()
                device_id = available_devices.popleft()
                task = spec.tasks[index]
                entries[index] = _running_entry(task, device_id=device_id)
                write_batch_report(entries, report_path, output=spec.output)
                future = executor.submit(_run_task_child, spec, task, device_id)
                active[future] = (index, device_id)

            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                index, device_id = active.pop(future)
                task = spec.tasks[index]
                try:
                    entry = future.result()
                except BaseException as exc:
                    entry = _exception_entry(task, exc, device_id=device_id)
                entries[index] = entry
                write_batch_report(entries, report_path, output=spec.output)
                available_devices.append(device_id)

    return entries


def _run_one_task_entry(
    spec: SimpleConfig,
    task: ConfiguredTask,
    *,
    device_id: Optional[int],
) -> dict[str, Any]:
    try:
        result = _run_task(spec, task, device_id=device_id)
        write_pipeline_report(result, task.result_path)
        return _result_entry(task, result, device_id=device_id)
    except Exception as exc:
        return _exception_entry(task, exc, device_id=device_id)


def _run_task_child(spec: SimpleConfig, task: ConfiguredTask, device_id: Optional[int]) -> dict[str, Any]:
    return _run_one_task_entry(spec, task, device_id=device_id)


def _run_task(spec: SimpleConfig, task: ConfiguredTask, *, device_id: Optional[int]) -> PipelineRunResult:
    client = CannBenchClient(
        spec.bench_root,
        python_executable=sys.executable,
        env=_eval_env(
            agent_type=spec.agent_type,
            bench_name=spec.bench_name,
        ),
        timeout_sec=DEFAULT_TIMEOUT_SEC,
    )
    pipeline = BenchmarkPipeline(bench_name=spec.bench_name, client=client)

    converter = create_converter(spec.agent_type, spec.bench_name, _converter_config(spec))
    conversion_runner = _conversion_runner(spec)
    conversion_workdir = task.convert_dir if conversion_runner is not None else None
    generator = create_generator(spec.agent_type, _generator_config(spec, device_id=device_id))

    return pipeline.run_case(
        selector=task.selector,
        generator=generator,
        converter=converter,
        workdir=task.workdir,
        submission_dir=task.submission_dir,
        reports_dir=task.reports_dir,
        conversion_runner=conversion_runner,
        conversion_workdir=conversion_workdir,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        env=_agent_env(spec),
        device_id=device_id,
        extra_eval_args=build_eval_args(_default_eval_config()),
    )


def _conversion_runner(spec: SimpleConfig):
    if _normalize_name(spec.agent_type) == "pypto":
        cfg = {"model": spec.model} if spec.model else {}
        return create_runner("opencode", cfg)
    return None


def _generator_config(spec: SimpleConfig, *, device_id: Optional[int]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "repo_root": spec.workspace,
        **dict(spec.agent_options),
    }
    env = _agent_env(spec)
    if env:
        cfg["env"] = env
    if device_id is not None:
        cfg["device_id"] = device_id
    if _normalize_name(spec.agent_type) == "pypto":
        if spec.model:
            cfg["model"] = spec.model
        cfg["worktree_root"] = spec.output / "pypto_worktrees"
        if device_id is not None:
            cfg["device_mode"] = "pool"
    return cfg


def _converter_config(spec: SimpleConfig) -> dict[str, Any]:
    return {
        "timeout_sec": DEFAULT_TIMEOUT_SEC,
        "env": _agent_env(spec),
    }


def _default_eval_config() -> dict[str, Any]:
    return {
        "no_subprocess_isolation": True,
        "op_timeout_sec": DEFAULT_OP_TIMEOUT_SEC,
        "verbose": True,
    }


def _agent_env(spec: SimpleConfig) -> Dict[str, str]:
    return {}


def _result_entry(
    task: ConfiguredTask,
    result: PipelineRunResult,
    *,
    device_id: Optional[int],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": task.name,
        "selector": task.selector,
        "ok": result.ok,
        "status": result.status,
        "result_file": str(task.result_path),
        "result": _pipeline_result_payload(result),
    }
    if device_id is not None:
        entry["device_id"] = device_id
    return entry


def _pending_entry(task: ConfiguredTask) -> dict[str, Any]:
    return {
        "name": task.name,
        "selector": task.selector,
        "ok": None,
        "status": "pending",
    }


def _running_entry(task: ConfiguredTask, *, device_id: Optional[int]) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": task.name,
        "selector": task.selector,
        "ok": None,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if device_id is not None:
        entry["device_id"] = device_id
    return entry


def _exception_entry(
    task: ConfiguredTask,
    exc: BaseException,
    *,
    device_id: Optional[int],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": task.name,
        "selector": task.selector,
        "ok": False,
        "status": "error",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "result_file": str(task.result_path),
    }
    if device_id is not None:
        entry["device_id"] = device_id
    return entry


def _parse_config(
    cfg: Mapping[str, Any],
    *,
    config_path: Optional[Path],
    runtime: Optional[Mapping[str, Any]] = None,
) -> SimpleConfig:
    _reject_legacy_sections(cfg)
    _reject_unknown_keys(cfg, {"agent", "benchmark"})

    agent_cfg = _mapping(cfg.get("agent"), "agent")
    benchmark_cfg = _mapping(cfg.get("benchmark"), "benchmark")
    _reject_unknown_keys(agent_cfg, _allowed_agent_keys(), prefix="agent")
    _reject_unknown_keys(benchmark_cfg, {"name", "root", "tasks"}, prefix="benchmark")
    runtime_cfg = _runtime_options(runtime)

    agent_type = str(agent_cfg.get("type") or "").strip()
    if not agent_type:
        raise ValueError("agent.type is required")
    workspace = runtime_cfg.get("workspace")
    if not workspace:
        raise ValueError("--workspace is required")

    bench_name = str(benchmark_cfg.get("name") or "cann")
    _validate_environment(agent_type=agent_type)
    bench_root = _resolve_path(benchmark_cfg.get("root") or DEFAULT_CANN_BENCH_ROOT)
    output = _resolve_output(runtime_cfg.get("output"))
    selectors = _task_selectors(benchmark_cfg)
    tasks = _configured_tasks(selectors, output)
    devices = tuple(_int_list(runtime_cfg.get("devices")))
    parallel = _int_or_default(runtime_cfg.get("parallel"), len(devices) if devices else 1)
    if parallel <= 0:
        raise ValueError("parallel must be positive")

    return SimpleConfig(
        output=output,
        bench_name=bench_name,
        bench_root=bench_root,
        agent_type=agent_type,
        workspace=_resolve_path(workspace),
        agent_options=_agent_options(agent_cfg),
        model=str(runtime_cfg.get("model") or ""),
        devices=devices,
        parallel=parallel,
        tasks=tuple(tasks),
    )


def _allowed_agent_keys() -> set[str]:
    return {
        "type",
        "backend",
        "arch",
        "framework",
        "codegen_target",
        "workflow",
        "verify_timeout",
        "config_path",
    }


def _agent_options(agent_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    options = {
        str(key): value
        for key, value in agent_cfg.items()
        if key not in {"type"}
    }
    if options.get("config_path"):
        options["config_path"] = _resolve_path(options["config_path"])
    return options


def _runtime_options(runtime: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    cfg = dict(runtime or {})
    _reject_unknown_keys(
        cfg,
        {
            "output",
            "workspace",
            "model",
            "devices",
            "parallel",
        },
        prefix="runtime",
    )
    return cfg


def _reject_legacy_sections(cfg: Mapping[str, Any]) -> None:
    legacy = [
        key
        for key in (
            "generator",
            "convert",
            "eval",
            "cases",
            "pipeline",
            "artifact",
            "bench",
            "dsl",
            "generation",
            "submission",
            "converter",
            "agent_output",
            "source_dir",
        )
        if cfg.get(key) is not None
    ]
    if legacy:
        raise ValueError(
            "unsupported legacy config sections: "
            + ", ".join(legacy)
            + "; use agent/benchmark in YAML and CLI args for runtime values"
        )


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], *, prefix: str = "") -> None:
    unknown = sorted(str(key) for key in value if str(key) not in allowed)
    if unknown:
        location = f"{prefix}." if prefix else ""
        raise ValueError(f"unsupported config keys: {', '.join(location + key for key in unknown)}")


def _task_selectors(benchmark_cfg: Mapping[str, Any]) -> list[str]:
    raw_tasks = benchmark_cfg.get("tasks")
    if raw_tasks is None:
        raise ValueError("benchmark.tasks is required")
    tasks = _string_list(raw_tasks)
    if not tasks:
        raise ValueError("benchmark.tasks must not be empty")
    return tasks


def _configured_tasks(selectors: list[str], output: Path) -> list[ConfiguredTask]:
    used: set[str] = set()
    tasks = []
    for selector in selectors:
        base = _selector_leaf_name(selector)
        slug = _unique_slug(base, selector, used)
        tasks.append(ConfiguredTask(name=base, selector=selector, root_dir=output / slug))
    return tasks


def _selector_leaf_name(selector: str) -> str:
    normalized = selector.replace("\\", "/").rstrip("/")
    leaf = normalized.rsplit("/", 1)[-1] if normalized else "task"
    if "." in leaf:
        leaf = leaf.rsplit(".", 1)[0]
    return _safe_path_name(leaf) or "task"


def _unique_slug(base: str, selector: str, used: set[str]) -> str:
    slug = base
    if slug in used:
        slug = f"{base}_{uuid.uuid5(uuid.NAMESPACE_URL, selector).hex[:8]}"
    used.add(slug)
    return slug


def _safe_path_name(value: str) -> str:
    return "_".join(part for part in str(value).replace("\\", "/").split("/") if part).strip("_")


def _resolve_output(value: object) -> Path:
    if value is None or value == "":
        value = DEFAULT_OUTPUT_ROOT / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    return _resolve_path(value)


def _resolve_path(value: object) -> Path:
    return Path(str(value)).expanduser().resolve()


def _device_for_index(index: int, devices: tuple[int, ...]) -> Optional[int]:
    if not devices:
        return None
    return devices[index % len(devices)]


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _validate_environment(*, agent_type: str) -> None:
    if _normalize_name(agent_type) == "pypto" and not os.environ.get(TILE_LIB_ENV, "").strip():
        raise ValueError(f"{TILE_LIB_ENV} must be set in the environment for pypto")


def write_pipeline_report(result: PipelineRunResult, output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        **_pipeline_result_payload(result),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def write_batch_report(entries: list, output_path: Path, *, output: Optional[Path] = None) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = [entry for entry in entries if entry.get("ok") is not None]
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output": str(output) if output is not None else str(output_path.parent),
        "ok": bool(entries) and len(completed) == len(entries) and all(entry.get("ok") for entry in entries),
        "total_cases": len(entries),
        "completed_cases": len(completed),
        "passed_cases": sum(1 for entry in entries if entry.get("ok") is True),
        "failed_cases": sum(1 for entry in entries if entry.get("ok") is False),
        "running_cases": sum(1 for entry in entries if entry.get("status") == "running"),
        "pending_cases": sum(1 for entry in entries if entry.get("status") == "pending"),
        "cases": entries,
    }
    _atomic_write_text(output_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return output_path


def read_yaml_mapping(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def build_eval_args(eval_cfg: Mapping[str, Any]) -> list:
    args = []
    value_options = {
        "case_id": "--case-id",
        "device": "--device",
        "operator": "--operator",
        "processes_per_card": "--processes-per-card",
        "timeout_per_operator": "--timeout-per-operator",
        "warmup": "--warmup",
        "repeat": "--repeat",
        "profiler_level": "--profiler-level",
        "op_timeout_sec": "--op-timeout-sec",
        "eval_code": "--eval-code",
        "output": "--output",
    }
    for key, option in value_options.items():
        value = eval_cfg.get(key)
        if value is not None and value != "":
            args.extend([option, str(value)])

    flag_options = {
        "no_perf": "--no-perf",
        "no_subprocess_isolation": "--no-subprocess-isolation",
        "no_iterative_compile": "--no-iterative-compile",
        "verbose": "--verbose",
    }
    for key, option in flag_options.items():
        if _optional_bool(eval_cfg.get(key)) is True:
            args.append(option)

    args.extend(_string_list(eval_cfg.get("extra_args")))
    return args


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if value is None:
        raise ValueError(f"{name} is required")
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    raise ValueError(f"expected list or string, got {value!r}")


def _string_mapping(value: object) -> Dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("expected mapping")
    return {str(key): str(val) for key, val in value.items()}


def _int_list(value: object) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.replace("\n", ",").split(",")]
        result: list[int] = []
        for part in raw_items:
            if part:
                result.extend(_int_token(part))
        return result
    if isinstance(value, Iterable):
        return [int(item) for item in value]
    raise ValueError(f"expected integer list or comma-separated string, got {value!r}")


def _int_token(value: str) -> list[int]:
    if "-" not in value:
        return [int(value)]
    start_text, end_text = value.split("-", 1)
    start = int(start_text)
    end = int(end_text)
    if end < start:
        raise ValueError(f"invalid integer range: {value!r}")
    return list(range(start, end + 1))


def _int_or_default(value: object, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _optional_bool(value: object) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _eval_env(
    *,
    agent_type: str,
    bench_name: str,
) -> Dict[str, str]:
    env = {}
    if _is_pypto_cann_eval(agent_type=agent_type, bench_name=bench_name):
        env[PERF_SOURCE_ENV] = "trace_view"
    return env


def _is_pypto_cann_eval(*, agent_type: str, bench_name: str) -> bool:
    return (
        str(agent_type or "").strip().lower().replace("-", "_") == "pypto"
        and str(bench_name or "").strip().lower() == "cann"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _pipeline_result_payload(result: PipelineRunResult) -> Dict[str, Any]:
    return {
        "status": result.status,
        "message": result.message,
        "ok": result.ok,
        "case": _task_to_dict(result.case),
        "generator_prompt_file": str(result.generator_prompt_file) if result.generator_prompt_file else "",
        "generated_artifact": _artifact_to_dict(result.generated_artifact),
        "converter_prompt_file": str(result.converter_prompt_file) if result.converter_prompt_file else "",
        "conversion_artifact": _artifact_to_dict(result.conversion_artifact),
        "submission": {
            "kind": result.submission.kind,
            "operator": result.submission.operator,
            "source_dir": str(result.submission.source_dir),
            "metadata": result.submission.metadata,
        },
        "kernel_eval": {
            "returncode": result.eval_result.returncode,
            "command": result.eval_result.command,
            "reports_dir": str(result.eval_result.reports_dir),
            "report_files": [str(path) for path in result.eval_result.report_files],
            "stdout": result.eval_result.stdout,
            "stderr": result.eval_result.stderr,
        },
    }


def _task_to_dict(task: object) -> Dict[str, Any]:
    return {
        "bench_name": getattr(task, "bench_name", ""),
        "task_dir": str(getattr(task, "task_dir", "")),
        "operator": getattr(task, "operator", ""),
        "rel_path": getattr(task, "rel_path", ""),
        "files": {key: str(path) for key, path in getattr(task, "files", {}).items()},
        "metadata": getattr(task, "metadata", {}),
    }


def _artifact_to_dict(artifact: Optional[Artifact]) -> Dict[str, Any]:
    if artifact is None:
        return {}
    return {
        "status": artifact.status,
        "message": artifact.message,
        "workdir": str(artifact.workdir),
        "files": {key: str(path) for key, path in artifact.files.items()},
        "log_file": str(artifact.log_file) if artifact.log_file else "",
        "metadata": artifact.metadata,
        "output_text": artifact.output_text,
    }


def _jsonable(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): _jsonable(val) for key, val in value.items()}
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return [_jsonable(item) for item in value]
        return str(value)
