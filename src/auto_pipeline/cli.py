"""CLI for cann-bench benchmark pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Mapping, Optional

from auto_pipeline import state as pipeline_state
from auto_pipeline.core import ParentProcessSignal
from auto_pipeline.core import read_yaml_mapping, run_cases_from_mapping, run_from_config

_COMMANDS = {"run", "list", "monitor", "kill", "retry"}
_TERMINAL_RUN_STATUSES = {"success", "failed", "error", "killed"}
_CLEAR_SCREEN = "\033[2J\033[H"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auto_pipeline.cli",
        description="Run and monitor the cann-bench benchmark generation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_run_args(subparsers.add_parser("run", help="run a pipeline"))
    _add_list_args(subparsers.add_parser("list", help="list registered pipeline runs"))
    _add_monitor_args(subparsers.add_parser("monitor", help="show a monitor table for one run"))
    _add_kill_args(subparsers.add_parser("kill", help="kill a run or task"))
    _add_retry_args(subparsers.add_parser("retry", help="manually retry/resume one task"))
    return parser


def create_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m auto_pipeline.cli",
        description="Run the cann-bench benchmark generation pipeline.",
    )
    _add_run_args(parser)
    return parser


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="pipeline YAML config")
    parser.add_argument("--workspace", required=True, help="runtime agent source workspace")
    parser.add_argument("--model", help="model for PyPTO generation and conversion")
    parser.add_argument("--output", help="runtime output root")
    parser.add_argument("--devices", help="device ids, e.g. 0,1 or 1-7")
    parser.add_argument("--parallel", type=int, help="maximum number of benchmark tasks to run in parallel")
    parser.add_argument("--gen-timeout", type=int, help="generation/OpenCode timeout in seconds")
    parser.add_argument("--eval-timeout", type=int, help="kernel eval/verify timeout in seconds")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--foreground", action="store_true", help="run the pipeline in the foreground")
    group.add_argument("--background", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-monitor", action="store_true", help="do not open monitor after background start")
    parser.add_argument("--monitor-interval", type=float, default=2.0, help="auto-monitor refresh interval seconds")
    parser.add_argument("--run-id", help=argparse.SUPPRESS)


def _add_list_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="print JSON")


def _add_monitor_args(parser: argparse.ArgumentParser) -> None:
    _add_run_selector_args(parser)
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--once", action="store_true", help="render one snapshot instead of watching")
    parser.add_argument("--interval", type=float, default=2.0, help="monitor refresh interval seconds")


def _add_kill_args(parser: argparse.ArgumentParser) -> None:
    _add_run_selector_args(parser)
    parser.add_argument("--grace-sec", type=float, default=5.0, help="seconds between SIGTERM and SIGKILL")


def _add_retry_args(parser: argparse.ArgumentParser) -> None:
    _add_run_selector_args(parser)
    parser.add_argument(
        "--task",
        required=True,
        action="append",
        help="task id/name to retry/resume; repeat or use comma-separated values",
    )
    parser.add_argument("--devices", help="override device ids, e.g. 0,1 or 1-7")
    parser.add_argument("--parallel", type=int, help="maximum retry tasks to run in parallel")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--foreground", action="store_true", help="run the retry in the foreground")
    group.add_argument("--background", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-monitor", action="store_true", help="do not open monitor after background start")
    parser.add_argument("--monitor-interval", type=float, default=2.0, help="auto-monitor refresh interval seconds")
    parser.add_argument("--retry-run-id", help=argparse.SUPPRESS)
    parser.add_argument("--retry-output", help=argparse.SUPPRESS)


def _add_run_selector_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-id", help="registered run id")
    group.add_argument("--output", help="pipeline output directory")
    group.add_argument("--latest", action="store_true", help="select latest registered run")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and (argv[0] in _COMMANDS or argv[0] in {"-h", "--help"}):
            parser = create_parser()
            args = parser.parse_args(argv)
        else:
            parser = create_run_parser()
            args = parser.parse_args(argv)
            args.command = "run"

        if args.command == "run":
            return _main_run(args, argv)
        if args.command == "list":
            return _main_list(args)
        if args.command == "monitor":
            return _main_monitor(args)
        if args.command == "kill":
            return _main_kill(args)
        if args.command == "retry":
            return _main_retry(args)
        parser.error("missing command")
        return 2
    except ParentProcessSignal as exc:
        print(f"received {exc.signal_name}; child processes terminated", file=sys.stderr)
        return int(exc.code)
    except KeyboardInterrupt:
        print("interrupted; child processes terminated", file=sys.stderr)
        return 130


def _main_run(args: argparse.Namespace, argv: list[str]) -> int:
    run_id = args.run_id or pipeline_state.new_run_id()
    if not args.foreground:
        return _start_background(args, run_id=run_id)
    runtime = _runtime_from_args(args, run_id=run_id, command=_display_command(argv, run_id=run_id))
    return run_from_config(Path(args.config), runtime=runtime)


def _runtime_from_args(args: argparse.Namespace, *, run_id: str, command: str = "") -> dict[str, object]:
    return {
        key: value
        for key, value in {
            "workspace": args.workspace,
            "model": args.model,
            "output": args.output,
            "devices": args.devices,
            "parallel": args.parallel,
            "gen_timeout": args.gen_timeout,
            "eval_timeout": args.eval_timeout,
            "run_id": run_id,
            "command": command,
        }.items()
        if value is not None and value != ""
    }


def _start_background(args: argparse.Namespace, *, run_id: str) -> int:
    output = args.output or str(pipeline_state.DEFAULT_CANN_BENCH_ROOT / "benchmark_runs" / run_id)
    child_args = _run_args(args, run_id=run_id, output=output, foreground=True)
    command = [sys.executable, "-m", "auto_pipeline.cli", "run", *child_args]
    run_dir = pipeline_state.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "background.log"
    log = log_file.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    try:
        pgid = os.getpgid(process.pid)
    except OSError:
        pgid = process.pid
    process_info = pipeline_state.process_identity(process.pid)
    pipeline_state.upsert_run(
        run_id,
        {
            **process_info,
            "status": "running",
            "pid": process.pid,
            "pgid": pgid,
            "output": str(Path(output).expanduser().resolve()),
            "workspace": str(Path(args.workspace).expanduser().resolve()),
            "config_path": str(Path(args.config).expanduser().resolve()),
            "model": args.model or "",
            "devices": args.devices or "",
            "parallel": args.parallel,
            "gen_timeout_sec": args.gen_timeout,
            "eval_timeout_sec": args.eval_timeout,
            "command": shlex.join(command),
            "background_log": str(log_file),
        },
    )
    if args.no_monitor:
        _print_background_info(run_id=run_id, pid=process.pid, output=Path(output), log_file=log_file)
        return 0
    selector = argparse.Namespace(run_id=run_id, output=None, latest=False, json=False)
    return _watch_run(
        selector,
        interval=args.monitor_interval,
        clear=True,
        stop_when_terminal=True,
    )


def _print_background_info(*, run_id: str, pid: int, output: Path, log_file: Path) -> None:
    print(f"run_id: {run_id}")
    print(f"pid: {pid}")
    print(f"output: {output.expanduser().resolve()}")
    print(f"log: {log_file}")


def _run_args(args: argparse.Namespace, *, run_id: str, output: str, foreground: bool) -> list[str]:
    out = ["--config", str(args.config), "--workspace", str(args.workspace), "--output", output, "--run-id", run_id]
    if args.model:
        out.extend(["--model", str(args.model)])
    if args.devices:
        out.extend(["--devices", str(args.devices)])
    if args.parallel is not None:
        out.extend(["--parallel", str(args.parallel)])
    if args.gen_timeout is not None:
        out.extend(["--gen-timeout", str(args.gen_timeout)])
    if args.eval_timeout is not None:
        out.extend(["--eval-timeout", str(args.eval_timeout)])
    if foreground:
        out.append("--foreground")
    return out


def _display_command(argv: list[str], *, run_id: str) -> str:
    command = [sys.executable, "-m", "auto_pipeline.cli", *argv]
    if "--run-id" not in argv:
        command.extend(["--run-id", run_id])
    return shlex.join(command)


def _main_list(args: argparse.Namespace) -> int:
    runs = pipeline_state.list_runs()
    if args.json:
        print(json.dumps(runs, indent=2, ensure_ascii=False))
        return 0
    print(_format_runs_table(runs))
    return 0


def _main_monitor(args: argparse.Namespace) -> int:
    if args.once:
        return _render_monitor(args, clear=False)
    return _watch_run(args, interval=args.interval, clear=not args.json, stop_when_terminal=True)


def _watch_run(
    args: argparse.Namespace,
    *,
    interval: float,
    clear: bool,
    stop_when_terminal: bool,
) -> int:
    try:
        while True:
            code = _render_monitor(args, clear=clear)
            if code != 0:
                return code
            run = _select_run(args)
            if stop_when_terminal and run is not None and _run_is_terminal(run):
                return 0
            time.sleep(max(float(interval), 0.1))
    except KeyboardInterrupt:
        print("monitor closed; background run is untouched", file=sys.stderr)
        return 0


def _render_monitor(args: argparse.Namespace, *, clear: bool) -> int:
    run = _select_run(args)
    if run is None:
        print("no matching auto_pipeline run found", file=sys.stderr)
        return 1
    if clear:
        _clear_screen()
    if args.json:
        print(json.dumps(run, indent=2, ensure_ascii=False))
    else:
        print(_format_monitor_table(run))
    sys.stdout.flush()
    return 0


def _clear_screen() -> None:
    sys.stdout.write(_CLEAR_SCREEN)


def _run_is_terminal(run: dict[str, object]) -> bool:
    return str(run.get("status") or "").lower() in _TERMINAL_RUN_STATUSES


def _main_kill(args: argparse.Namespace) -> int:
    run = _select_run(args)
    if run is None:
        print("no matching auto_pipeline run found", file=sys.stderr)
        return 1
    run_id = str(run.get("run_id"))
    result = pipeline_state.kill_run(
        run,
        grace_sec=args.grace_sec,
        on_signal_start=lambda: pipeline_state.mark_run_killed(
            run_id,
            signum=signal_number(),
            target="run",
        ),
    )
    pipeline_state.mark_run_killed(run_id, signum=signal_number(), target="run")
    for task in run.get("tasks") or []:
        if isinstance(task, dict) and str(task.get("status")) == "running":
            pipeline_state.mark_task_killed(run_id, str(task.get("task_id") or task.get("name")), signum=signal_number(), target="run")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


def signal_number() -> int:
    return int(signal.SIGTERM)


def _main_retry(args: argparse.Namespace) -> int:
    run = _select_run(args)
    if run is None:
        print("no matching auto_pipeline run found", file=sys.stderr)
        return 1
    task_names = _retry_task_names(args.task)
    selected_tasks = []
    for task_name in task_names:
        task = _select_task(run, task_name)
        if task is None:
            print(f"task not found: {task_name}", file=sys.stderr)
            return 1
        selected_tasks.append(task)
    config_path = run.get("config_path")
    if not config_path:
        print("selected run has no config_path; cannot retry", file=sys.stderr)
        return 1
    cfg = read_yaml_mapping(Path(str(config_path)))
    benchmark = cfg.setdefault("benchmark", {})
    if not isinstance(benchmark, dict):
        print("invalid benchmark config; cannot retry", file=sys.stderr)
        return 1
    benchmark["tasks"] = [str(task.get("selector") or task.get("task_id") or task.get("name")) for task in selected_tasks]
    devices = args.devices if args.devices else _retry_devices(selected_tasks, run)
    parallel = args.parallel if args.parallel is not None else 1
    retry_output, retry_run_id = _retry_destination(run, args)
    if not args.foreground:
        return _start_retry_background(
            args,
            source_run=run,
            task_names=task_names,
            selected_tasks=selected_tasks,
            retry_run_id=retry_run_id,
            retry_output=retry_output,
            devices=devices,
            parallel=parallel,
        )
    _stage_retry_sources(run, selected_tasks, retry_output)
    runtime = {
        "workspace": run.get("workspace"),
        "model": run.get("model") or "",
        "output": str(retry_output),
        "devices": _format_devices(devices),
        "parallel": parallel,
        "gen_timeout": run.get("gen_timeout_sec"),
        "eval_timeout": run.get("eval_timeout_sec"),
        "run_id": retry_run_id,
        "command": f"manual retry source_run={run.get('run_id')} task={','.join(task_names)}",
        "reuse_generated": True,
    }
    runtime = {key: value for key, value in runtime.items() if value not in (None, "", [])}
    entries = run_cases_from_mapping(cfg, config_path=Path(str(config_path)), runtime=runtime)
    return 0 if entries and all(entry.get("ok") for entry in entries) else 1


def _start_retry_background(
    args: argparse.Namespace,
    *,
    source_run: dict[str, object],
    task_names: list[str],
    selected_tasks: list[dict[str, object]],
    retry_run_id: str,
    retry_output: Path,
    devices: object,
    parallel: int,
) -> int:
    command = [
        sys.executable,
        "-m",
        "auto_pipeline.cli",
        "retry",
        *_retry_args(
            args,
            task_names=task_names,
            retry_run_id=retry_run_id,
            retry_output=retry_output,
            devices=devices,
            parallel=parallel,
            foreground=True,
        ),
    ]
    run_dir = pipeline_state.run_dir(retry_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "background.log"
    log = log_file.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    try:
        pgid = os.getpgid(process.pid)
    except OSError:
        pgid = process.pid
    pipeline_state.upsert_run(
        retry_run_id,
        {
            **pipeline_state.process_identity(process.pid),
            "status": "running",
            "pid": process.pid,
            "pgid": pgid,
            "output": str(retry_output.expanduser().resolve()),
            "workspace": str(source_run.get("workspace") or ""),
            "config_path": str(source_run.get("config_path") or ""),
            "model": str(source_run.get("model") or ""),
            "devices": _format_devices(devices),
            "parallel": parallel,
            "gen_timeout_sec": source_run.get("gen_timeout_sec"),
            "eval_timeout_sec": source_run.get("eval_timeout_sec"),
            "command": shlex.join(command),
            "background_log": str(log_file),
            "source_run_id": str(source_run.get("run_id") or ""),
            "source_output": str(source_run.get("output") or ""),
            "retry_tasks": task_names,
            "tasks_declared": [
                {
                    "task_id": str(task.get("task_id") or name),
                    "task_index": index,
                    "name": str(task.get("name") or name),
                    "selector": str(task.get("selector") or name),
                    "source_result_file": str(task.get("result_file") or ""),
                }
                for index, (name, task) in enumerate(zip(task_names, selected_tasks))
            ],
        },
    )
    if args.no_monitor:
        _print_background_info(run_id=retry_run_id, pid=process.pid, output=retry_output, log_file=log_file)
        return 0
    selector = argparse.Namespace(run_id=retry_run_id, output=None, latest=False, json=False)
    return _watch_run(
        selector,
        interval=args.monitor_interval,
        clear=True,
        stop_when_terminal=True,
    )


def _retry_args(
    args: argparse.Namespace,
    *,
    task_names: list[str],
    retry_run_id: str,
    retry_output: Path,
    devices: object,
    parallel: int,
    foreground: bool,
) -> list[str]:
    out = []
    if getattr(args, "run_id", None):
        out.extend(["--run-id", str(args.run_id)])
    elif getattr(args, "output", None):
        out.extend(["--output", str(args.output)])
    elif getattr(args, "latest", False):
        out.append("--latest")
    for task_name in task_names:
        out.extend(["--task", task_name])
    formatted_devices = _format_devices(devices)
    if formatted_devices:
        out.extend(["--devices", formatted_devices])
    out.extend(["--parallel", str(parallel)])
    out.extend(["--retry-run-id", retry_run_id, "--retry-output", str(retry_output.expanduser().resolve())])
    if foreground:
        out.append("--foreground")
    return out


def _retry_task_names(values: object) -> list[str]:
    raw_values = values if isinstance(values, list) else [values]
    names = []
    for value in raw_values:
        for part in str(value or "").replace("\n", ",").split(","):
            name = part.strip()
            if name:
                names.append(name)
    return names


def _retry_devices(tasks: list[dict[str, object]], run: dict[str, object]) -> object:
    devices = [task.get("device_id") for task in tasks if task.get("device_id") is not None]
    return devices if devices else run.get("devices")


def _retry_destination(run: dict[str, object], args: argparse.Namespace) -> tuple[Path, str]:
    if getattr(args, "retry_run_id", None) and getattr(args, "retry_output", None):
        return Path(str(args.retry_output)).expanduser().resolve(), str(args.retry_run_id)
    source_output = Path(str(run.get("output") or "")).expanduser().resolve()
    parent = source_output.parent if source_output.name else Path.cwd()
    source_run_id = str(run.get("run_id") or "run")
    suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    retry_run_id = f"retry_{source_run_id}_{suffix}"
    return parent / retry_run_id, retry_run_id


def _stage_retry_sources(run: dict[str, object], tasks: list[dict[str, object]], retry_output: Path) -> None:
    retry_output.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        source_root = _source_task_root(run, task)
        if source_root is None:
            continue
        task_name = str(task.get("name") or task.get("task_id") or source_root.name)
        target_root = retry_output / task_name
        target_root.mkdir(parents=True, exist_ok=True)
        result_file = source_root / "benchmark_result.json"
        if result_file.is_file():
            shutil.copy2(result_file, target_root / "benchmark_result.json")
        source_artifact = source_root / "work" / "artifact"
        target_artifact = target_root / "work" / "artifact"
        if source_artifact.is_dir():
            target_artifact.mkdir(parents=True, exist_ok=True)
            for name in ("akg_model.py", "akg-agent.log"):
                source = source_artifact / name
                if source.is_file():
                    shutil.copy2(source, target_artifact / name)


def _source_task_root(run: dict[str, object], task: dict[str, object]) -> Optional[Path]:
    result_file = task.get("result_file")
    if result_file not in (None, ""):
        path = Path(str(result_file)).expanduser().resolve().parent
        if path.is_dir():
            return path
    output = run.get("output")
    task_name = task.get("name") or task.get("task_id")
    if output not in (None, "") and task_name not in (None, ""):
        path = Path(str(output)).expanduser().resolve() / str(task_name)
        if path.is_dir():
            return path
    return None


def _format_devices(value: object) -> str:
    if value in (None, "", []):
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def _select_run(args: argparse.Namespace) -> Optional[dict[str, object]]:
    return pipeline_state.find_run(
        run_id=getattr(args, "run_id", None) or "",
        output=getattr(args, "output", None),
        latest=bool(getattr(args, "latest", False) or not getattr(args, "run_id", None) and not getattr(args, "output", None)),
    )


def _select_task(run: dict[str, object], task_id: str) -> Optional[dict[str, object]]:
    for task in run.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        names = {str(task.get("task_id") or ""), str(task.get("name") or ""), str(task.get("selector") or "")}
        if str(task_id) in names:
            return task
    return None


def _format_runs_table(runs: list[dict[str, object]]) -> str:
    if not runs:
        return "No auto_pipeline runs registered."
    rows = [["RUN_ID", "STATUS", "ELAPSED", "UPDATED", "PID", "SUMMARY", "OUTPUT"]]
    for run in runs:
        summary = run.get("summary") if isinstance(run.get("summary"), dict) else {}
        summary_text = (
            f"T={summary.get('total', 0)} R={summary.get('running', 0)} "
            f"S={summary.get('success', 0)} F={summary.get('failed', 0)}"
        )
        rows.append([
            str(run.get("run_id") or ""),
            str(run.get("status") or ""),
            _format_elapsed_for_record(run),
            _format_display_timestamp(run.get("updated_at")),
            str(run.get("pid") or ""),
            summary_text,
            str(run.get("output") or ""),
        ])
    return _format_table(rows)


def _format_monitor_table(run: dict[str, object]) -> str:
    tasks = [task for task in (run.get("tasks") or []) if isinstance(task, dict)]
    now = time.time()
    header = [
        f"run_id = {run.get('run_id') or ''}",
        f"status = {run.get('status') or ''}",
        f"pid = {run.get('pid') or ''}",
        f"output = {run.get('output') or ''}",
        f"log = {run.get('background_log') or ''}",
        "",
    ]
    rows = [["TASK", "STATUS", "GEN", "CONVERT", "EVAL", "TOTAL", "STAGE", "DEVICE", "UPDATED", "TOKENS"]]
    for task in tasks:
        usage = task.get("token_usage") if isinstance(task.get("token_usage"), dict) else {}
        rows.append([
            str(task.get("task_id") or task.get("name") or ""),
            str(task.get("status") or ""),
            _format_stage_elapsed_for_task(task, "generation", now=now),
            _format_stage_elapsed_for_task(task, "conversion", now=now),
            _format_stage_elapsed_for_task(task, "eval", now=now),
            _format_task_total_elapsed(task, now=now),
            str(task.get("stage") or ""),
            str(task.get("device_id") if task.get("device_id") is not None else ""),
            _format_display_timestamp(task.get("updated_at")),
            str(usage.get("total", 0)),
        ])
    return "\n".join(header + [_format_table(rows)])


_MONITOR_TIMED_STAGES = ("generation", "conversion", "eval")


def _format_stage_elapsed_for_task(
    task: Mapping[str, object],
    stage: str,
    *,
    now: Optional[float] = None,
) -> str:
    seconds = _stage_elapsed_seconds_for_task(task, stage, now=now)
    if seconds is None:
        return "00:00:00"
    return _format_duration_seconds(seconds)


def _format_task_total_elapsed(task: Mapping[str, object], *, now: Optional[float] = None) -> str:
    total = 0
    has_timing = False
    for stage in _MONITOR_TIMED_STAGES:
        seconds = _stage_elapsed_seconds_for_task(task, stage, now=now)
        if seconds is not None:
            total += seconds
            has_timing = True
    if not has_timing:
        return "00:00:00"
    return _format_duration_seconds(total)


def _stage_elapsed_seconds_for_task(
    task: Mapping[str, object],
    stage: str,
    *,
    now: Optional[float] = None,
) -> Optional[int]:
    raw_stage_times = task.get("stage_times")
    if not isinstance(raw_stage_times, Mapping):
        return None
    raw_stage = raw_stage_times.get(stage)
    if not isinstance(raw_stage, Mapping):
        return None
    start = _parse_utc_timestamp(raw_stage.get("started_at"))
    if start is None:
        return None
    end = _parse_utc_timestamp(raw_stage.get("ended_at"))
    if end is None and str(task.get("stage") or "") == stage and str(task.get("status") or "") == "running":
        end = time.time() if now is None else now
    if end is None:
        return None
    return max(0, int(end - start))


def _format_duration_seconds(seconds: int) -> str:
    hours, remainder = divmod(max(0, int(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_elapsed_for_record(record: Mapping[str, object]) -> str:
    start = _parse_utc_timestamp(record.get("started_at") or record.get("created_at"))
    if start is None:
        return "00:00:00"
    end = _elapsed_end_timestamp(record)
    if end is None:
        end = time.time()
    return _format_duration_seconds(max(0, int(end - start)))


def _elapsed_end_timestamp(record: Mapping[str, object]) -> Optional[float]:
    for key in ("completed_at", "killed_at", "finished_at"):
        value = _parse_utc_timestamp(record.get(key))
        if value is not None:
            return value
    status = str(record.get("status") or "").lower()
    if status in _TERMINAL_RUN_STATUSES or status not in {"", "pending", "running", "queued"}:
        return _parse_utc_timestamp(record.get("updated_at"))
    return None


def _format_display_timestamp(value: object) -> str:
    timestamp = _parse_utc_timestamp(value)
    if timestamp is None:
        return str(value or "")
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _parse_utc_timestamp(value: object) -> Optional[float]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _format_table(rows: list[list[str]]) -> str:
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        line = "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        lines.append(line.rstrip())
        if row_index == 0:
            lines.append("  ".join("-" * width for width in widths).rstrip())
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
