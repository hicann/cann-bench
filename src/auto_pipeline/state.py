"""Runtime registry and monitor helpers for auto_pipeline runs."""

from __future__ import annotations

import json
import os
import signal
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

DEFAULT_CANN_BENCH_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR_NAME = ".auto_pipeline"
RUNS_DIR_NAME = "runs"


def utc_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def new_run_id() -> str:
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def state_root(cann_bench_root: Optional[Path] = None) -> Path:
    return Path(cann_bench_root or DEFAULT_CANN_BENCH_ROOT).expanduser().resolve() / STATE_DIR_NAME


def runs_root(cann_bench_root: Optional[Path] = None) -> Path:
    return state_root(cann_bench_root) / RUNS_DIR_NAME


def run_dir(run_id: str, cann_bench_root: Optional[Path] = None) -> Path:
    return runs_root(cann_bench_root) / str(run_id)


def run_file(run_id: str, cann_bench_root: Optional[Path] = None) -> Path:
    return run_dir(run_id, cann_bench_root) / "run.json"


def tasks_dir(run_id: str, cann_bench_root: Optional[Path] = None) -> Path:
    return run_dir(run_id, cann_bench_root) / "tasks"


def task_file(run_id: str, task_id: str, cann_bench_root: Optional[Path] = None) -> Path:
    return tasks_dir(run_id, cann_bench_root) / f"{safe_name(task_id)}.json"


def process_identity(pid: Optional[int] = None) -> dict[str, int]:
    pid = os.getpid() if pid is None else int(pid)
    out: dict[str, int] = {"pid": pid}
    pgid = _safe_getpgid(pid)
    if pgid is not None:
        out["pgid"] = pgid
    stat = _read_proc_stat(pid)
    if stat is not None:
        out["pid_start_time"] = stat[1]
    return out


def safe_name(value: object) -> str:
    text = str(value or "").replace("\\", "/").strip("/")
    out = "_".join(part for part in text.split("/") if part)
    return out or "task"


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def jsonable(value: object) -> object:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(key): jsonable(val) for key, val in value.items()}
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return [jsonable(item) for item in value]
        return str(value)


def upsert_run(run_id: str, payload: Mapping[str, Any], *, cann_bench_root: Optional[Path] = None) -> dict[str, Any]:
    now = utc_ts()
    path = run_file(run_id, cann_bench_root)
    current = read_json(path)
    merged = {
        **current,
        **dict(payload),
        "run_id": str(run_id),
        "updated_at": now,
    }
    if not merged.get("created_at"):
        merged["created_at"] = now
    write_json_atomic(path, merged)
    return merged


def update_run(run_id: str, payload: Mapping[str, Any], *, cann_bench_root: Optional[Path] = None) -> dict[str, Any]:
    return upsert_run(run_id, payload, cann_bench_root=cann_bench_root)


def update_task(
    run_id: str,
    task_id: str,
    payload: Mapping[str, Any],
    *,
    cann_bench_root: Optional[Path] = None,
) -> dict[str, Any]:
    now = utc_ts()
    path = task_file(run_id, task_id, cann_bench_root)
    current = read_json(path)
    merged = {
        **current,
        **dict(payload),
        "run_id": str(run_id),
        "task_id": str(task_id),
        "updated_at": now,
    }
    if not merged.get("created_at"):
        merged["created_at"] = now
    write_json_atomic(path, merged)
    return merged


def list_runs(*, cann_bench_root: Optional[Path] = None) -> list[dict[str, Any]]:
    root = runs_root(cann_bench_root)
    if not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/run.json")):
        data = read_json(path)
        if data:
            data.setdefault("run_id", path.parent.name)
            data["tasks"] = list_tasks(str(data["run_id"]), cann_bench_root=cann_bench_root)
            data["summary"] = summarize_tasks(data["tasks"])
            runs.append(data)
    runs.sort(key=lambda item: str(item.get("created_at") or item.get("updated_at") or ""), reverse=True)
    return runs


def list_tasks(run_id: str, *, cann_bench_root: Optional[Path] = None) -> list[dict[str, Any]]:
    root = tasks_dir(run_id, cann_bench_root)
    if not root.is_dir():
        return []
    declared_order = _declared_task_order(run_id, cann_bench_root=cann_bench_root)
    tasks: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        data = read_json(path)
        if data:
            data.setdefault("task_id", path.stem)
            tasks.append(data)
    tasks.sort(key=lambda task: _task_order_key(task, declared_order))
    return tasks


def _declared_task_order(run_id: str, *, cann_bench_root: Optional[Path] = None) -> dict[str, int]:
    run = read_json(run_file(run_id, cann_bench_root))
    declared = run.get("tasks_declared")
    if not isinstance(declared, list):
        return {}

    order: dict[str, int] = {}
    for index, item in enumerate(declared):
        if not isinstance(item, Mapping):
            continue
        for key in ("task_id", "name", "selector"):
            value = item.get(key)
            if value not in (None, ""):
                order.setdefault(str(value), index)
    return order


def _task_order_key(task: Mapping[str, Any], declared_order: Mapping[str, int]) -> tuple[int, str]:
    raw_index = task.get("task_index")
    try:
        return (int(raw_index), str(task.get("task_id") or task.get("name") or ""))
    except (TypeError, ValueError):
        pass

    for key in ("task_id", "name", "selector"):
        value = task.get(key)
        if value not in (None, "") and str(value) in declared_order:
            return (declared_order[str(value)], str(task.get("task_id") or task.get("name") or ""))

    return (len(declared_order) + 1, str(task.get("task_id") or task.get("name") or ""))


def summarize_tasks(tasks: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "total": 0,
        "pending": 0,
        "running": 0,
        "success": 0,
        "failed": 0,
        "timeout": 0,
        "killed": 0,
        "unknown": 0,
    }
    for task in tasks:
        counts["total"] += 1
        status = str(task.get("status") or "unknown")
        ok = task.get("ok")
        if status in counts:
            counts[status] += 1
        elif ok is True:
            counts["success"] += 1
        elif ok is False:
            counts["failed"] += 1
        else:
            counts["unknown"] += 1
    return counts


def find_run(
    *,
    run_id: str = "",
    output: object = None,
    latest: bool = False,
    cann_bench_root: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    runs = list_runs(cann_bench_root=cann_bench_root)
    if latest and runs:
        return runs[0]
    if run_id:
        for run in runs:
            if str(run.get("run_id")) == str(run_id):
                return run
    if output is not None and str(output).strip():
        try:
            target = Path(str(output)).expanduser().resolve()
        except OSError:
            target = Path(str(output))
        for run in runs:
            raw = run.get("output")
            if not raw:
                continue
            try:
                candidate = Path(str(raw)).expanduser().resolve()
            except OSError:
                candidate = Path(str(raw))
            if candidate == target:
                return run
    return None


def mark_run_killed(run_id: str, *, signum: int, target: str, cann_bench_root: Optional[Path] = None) -> None:
    update_run(
        run_id,
        {
            "status": "killed",
            "killed_at": utc_ts(),
            "kill_signal": signum,
            "kill_target": target,
        },
        cann_bench_root=cann_bench_root,
    )


def mark_task_killed(
    run_id: str,
    task_id: str,
    *,
    signum: int,
    target: str,
    cann_bench_root: Optional[Path] = None,
) -> None:
    update_task(
        run_id,
        task_id,
        {
            "status": "killed",
            "stage": "killed",
            "ok": False,
            "killed_at": utc_ts(),
            "kill_signal": signum,
            "kill_target": target,
        },
        cann_bench_root=cann_bench_root,
    )


def kill_run(
    run: Mapping[str, Any],
    *,
    grace_sec: float = 5.0,
    on_signal_start: Optional[Callable[[], None]] = None,
) -> dict[str, Any]:
    return _kill_target(
        run,
        label=f"run:{run.get('run_id')}",
        grace_sec=grace_sec,
        on_signal_start=on_signal_start,
    )


def _kill_target(
    target: Mapping[str, Any],
    *,
    label: str,
    grace_sec: float,
    on_signal_start: Optional[Callable[[], None]] = None,
) -> dict[str, Any]:
    pid = _optional_int(target.get("pid"))
    pgid = _optional_int(target.get("pgid"))
    pid_start_time = _optional_int(target.get("pid_start_time"))
    sent: list[str] = []
    errors: list[str] = []
    root_alive = False
    root_identity_known = pid is not None and pid_start_time is not None
    if pid is not None:
        if pid_start_time is None:
            errors.append("missing pid_start_time; refusing pid/pgid signal to avoid killing an unrelated process")
        else:
            root_status = _pid_identity_status(pid, pid_start_time)
            if root_status == "alive":
                root_alive = True
            elif root_status == "not-found":
                sent.append(f"{label}:pid:{pid}:not-found")
            else:
                errors.append(
                    f"pid identity mismatch for {pid}; refusing pid/pgid signal to avoid killing an unrelated process"
                )

    descendants = _descendant_snapshot(pid) if root_alive and pid is not None else {}
    env_matches = _env_snapshots(_cleanup_matchers(target))
    snapshots = {**descendants, **env_matches}
    group_targeted = bool(root_alive and pgid is not None and pgid != os.getpgrp())
    pid_targeted = bool(root_alive and pid is not None and not group_targeted and pid != os.getpid())

    if not group_targeted and not pid_targeted and not snapshots and not root_identity_known:
        errors.append("missing safe kill target")

    if on_signal_start is not None and not errors and (group_targeted or pid_targeted or snapshots):
        on_signal_start()

    _send_safe_targets(
        label=label,
        signum=signal.SIGTERM,
        pid=pid if pid_targeted else None,
        pgid=pgid if group_targeted else None,
        snapshots=snapshots,
        sent=sent,
        errors=errors,
    )

    deadline = time.monotonic() + max(float(grace_sec), 0.0)
    while time.monotonic() < deadline:
        if not _has_live_targets(
            pid=pid if pid_targeted else None,
            pid_start_time=pid_start_time,
            pgid=pgid if group_targeted else None,
            snapshots=snapshots,
        ):
            break
        time.sleep(0.1)

    if _has_live_targets(
        pid=pid if pid_targeted else None,
        pid_start_time=pid_start_time,
        pgid=pgid if group_targeted else None,
        snapshots=snapshots,
    ):
        _send_safe_targets(
            label=label,
            signum=signal.SIGKILL,
            pid=pid if pid_targeted else None,
            pgid=pgid if group_targeted else None,
            snapshots=snapshots,
            sent=sent,
            errors=errors,
        )

    return {"ok": not errors, "sent": sent, "errors": errors}


def _optional_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _send_safe_targets(
    *,
    label: str,
    signum: signal.Signals,
    pid: Optional[int],
    pgid: Optional[int],
    snapshots: Mapping[int, int],
    sent: list[str],
    errors: list[str],
) -> None:
    if pgid is not None:
        try:
            os.killpg(pgid, signum)
            sent.append(f"{label}:pgid:{pgid}:{signum.name}")
        except ProcessLookupError:
            sent.append(f"{label}:pgid:{pgid}:not-found:{signum.name}")
        except OSError as exc:
            errors.append(f"{signum.name}/pgid:{pgid}: {exc}")

    if pid is not None:
        try:
            os.kill(pid, signum)
            sent.append(f"{label}:pid:{pid}:{signum.name}")
        except ProcessLookupError:
            sent.append(f"{label}:pid:{pid}:not-found:{signum.name}")
        except OSError as exc:
            errors.append(f"{signum.name}/pid:{pid}: {exc}")

    for snapshot_pid, start_time in sorted(snapshots.items()):
        if snapshot_pid == os.getpid() or not _pid_matches_start_time(snapshot_pid, start_time):
            continue
        try:
            os.kill(snapshot_pid, signum)
            sent.append(f"{label}:snapshot-pid:{snapshot_pid}:{signum.name}")
        except ProcessLookupError:
            sent.append(f"{label}:snapshot-pid:{snapshot_pid}:not-found:{signum.name}")
        except OSError as exc:
            errors.append(f"{signum.name}/snapshot-pid:{snapshot_pid}: {exc}")


def _has_live_targets(
    *,
    pid: Optional[int],
    pid_start_time: Optional[int],
    pgid: Optional[int],
    snapshots: Mapping[int, int],
) -> bool:
    if pid is not None and pid_start_time is not None and _pid_matches_start_time(pid, pid_start_time):
        return True
    if pgid is not None and _pgid_has_members(pgid):
        return True
    return any(_pid_matches_start_time(snapshot_pid, start_time) for snapshot_pid, start_time in snapshots.items())


def _cleanup_matchers(target: Mapping[str, Any]) -> list[dict[str, str]]:
    matchers: list[dict[str, str]] = []
    run_id = target.get("run_id")
    if run_id not in (None, ""):
        matchers.append({"AUTO_PIPELINE_RUN_ID": str(run_id)})

    raw = target.get("cleanup_env")
    for matcher in _iter_cleanup_env(raw):
        if matcher:
            matchers.append(matcher)
    return _dedupe_matchers(matchers)


def _iter_cleanup_env(value: object) -> Iterable[dict[str, str]]:
    if not value:
        return []
    if isinstance(value, Mapping):
        if all(not isinstance(item, Mapping) for item in value.values()):
            return [_string_env(value)]
        out: list[dict[str, str]] = []
        for item in value.values():
            if isinstance(item, Mapping):
                out.append(_string_env(item))
        return out
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        out = []
        for item in value:
            if isinstance(item, Mapping):
                out.append(_string_env(item))
        return out
    return []


def _string_env(value: Mapping[str, object]) -> dict[str, str]:
    return {str(key): str(val) for key, val in value.items() if str(key) and val not in (None, "")}


def _dedupe_matchers(matchers: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for matcher in matchers:
        normalized = tuple(sorted((str(key), str(value)) for key, value in matcher.items() if key and value))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(dict(normalized))
    return out


def _env_snapshots(matchers: Iterable[Mapping[str, str]]) -> dict[int, int]:
    out: dict[int, int] = {}
    for matcher in matchers:
        out.update(_environ_match_snapshot(matcher))
    return out


def _pid_identity_status(pid: int, start_time: int) -> str:
    stat = _read_proc_stat(pid)
    if stat is None or stat[2] == "Z":
        return "not-found"
    if stat[1] != start_time:
        return "mismatch"
    return "alive"


def _pid_matches_start_time(pid: int, start_time: int) -> bool:
    return _pid_identity_status(pid, start_time) == "alive"


def _descendant_snapshot(root_pid: int) -> dict[int, int]:
    entries: dict[int, tuple[int, int, str]] = {}
    for stat_path in _safe_proc_glob("[0-9]*/stat"):
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
        _ppid, start_time, state = entry
        if state != "Z":
            out[pid] = start_time
        stack.extend(children.get(pid, []))
    return out


def _environ_match_snapshot(match_environ: Mapping[str, str]) -> dict[int, int]:
    if not match_environ:
        return {}
    out: dict[int, int] = {}
    current_pid = os.getpid()
    for environ_path in _safe_proc_glob("[0-9]*/environ"):
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


def _pgid_has_members(pgid: Optional[int]) -> bool:
    if pgid is None:
        return False
    for stat_path in _safe_proc_glob("[0-9]*/stat"):
        try:
            pid = int(stat_path.parent.name)
            if pid == os.getpid():
                continue
            if os.getpgid(pid) == pgid:
                stat = _read_proc_stat(pid)
                if stat is not None and stat[2] != "Z":
                    return True
        except (OSError, ValueError):
            continue
    return False


def _safe_getpgid(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _safe_proc_glob(pattern: str) -> Iterable[Path]:
    try:
        yield from Path("/proc").glob(pattern)
    except OSError:
        return


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


def _read_proc_stat(pid: int) -> Optional[tuple[int, int, str]]:
    try:
        content = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        fields = content.rsplit(") ", 1)[1].split()
        state = fields[0]
        ppid = int(fields[1])
        start_time = int(fields[19])
    except (IndexError, OSError, ValueError):
        return None
    return ppid, start_time, state
