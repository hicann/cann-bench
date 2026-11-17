#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Prepare, run, resume, and report a batch of Ascend Arena optimizations."""

from __future__ import annotations

import argparse
import json
import queue
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
REGISTRY = SRC / "ascend_arena_e2e" / "ops.json"
TASKS = ROOT / "bench_lab" / "ascend_arena_bench"
_OUTPUT_LOCK = threading.Lock()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_operators(value: str) -> list[str]:
    operators = [item.strip() for item in value.split(",") if item.strip()]
    if not operators:
        raise argparse.ArgumentTypeError("--operators cannot be empty")
    return operators


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--devices must be comma-separated non-negative integers") from exc
    if not devices or any(device < 0 for device in devices) or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("--devices must contain unique non-negative device ids")
    return devices


def create_parser() -> argparse.ArgumentParser:
    from ascend_arena_e2e.agent_base import add_agent_arguments

    parser = argparse.ArgumentParser(description="Batch optimize Ascend Arena operators with resumable progress.")
    parser.add_argument(
        "--operators",
        type=parse_operators,
        help="comma-separated operator names; defaults to every operator in ops.json",
    )
    parser.add_argument("--devices", type=parse_devices, default="0", help="device ids, for example 0,1,2,3")
    parser.add_argument(
        "--output",
        type=Path,
        help="run root; defaults to benchmark_runs/<agent>-<model> or benchmark_runs/manual",
    )
    parser.add_argument("--goal", default="", help="optimization goal used by the selected agent")
    parser.add_argument("--restart", action="store_true", help="run every selected operator without resuming progress")
    parser.add_argument("--force", action="store_true", help="recreate selected operator workspaces")
    parser.add_argument("--agent-concurrency", type=int, default=0,
                        help="maximum concurrent agents; 0 uses the number of selected devices")
    add_agent_arguments(parser)
    return parser


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _print_status(message: str) -> None:
    with _OUTPUT_LOCK:
        print(message, flush=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_registry() -> dict[str, dict[str, Any]]:
    payload = _load_json(REGISTRY)
    if not payload or not all(isinstance(entry, dict) for entry in payload.values()):
        raise ValueError(f"expected an operator mapping in {REGISTRY}")
    return payload


def _select_operators(requested: list[str] | None, registry: dict[str, dict[str, Any]]) -> list[str]:
    if requested is None:
        return list(registry)
    names = {name.lower(): name for name in registry}
    unknown = [name for name in requested if name.lower() not in names]
    if unknown:
        raise ValueError(f"unknown operators: {', '.join(unknown)}")
    selected = [names[name.lower()] for name in requested]
    if len(selected) != len(set(selected)):
        raise ValueError("operator names must be unique")
    return selected


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-_") or "default"


def _git_revision() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _existing_workspace(task_root: Path, operator: str) -> dict[str, Any]:
    workdir = task_root / "work"
    workspace_json = workdir / "workspace.json"
    required = (workspace_json, workdir / "validate.sh", workdir / operator)
    if not all(path.exists() for path in required):
        raise FileExistsError(f"incomplete workspace already exists: {task_root}; pass --force to recreate it")
    payload = _load_json(workspace_json)
    if payload.get("operator") != operator:
        raise ValueError(f"invalid workspace metadata: {workspace_json}")
    return {**payload, "status": "existing", "workspace": str(workdir),
            "source_dir": str(workdir / operator), "task_root": str(task_root)}


def _prepare_workspace(
    args: argparse.Namespace,
    output: Path,
    operator: str,
    device_id: int,
    agent: Any | None = None,
) -> dict[str, Any]:
    from auto_pipeline.core import CannBenchClient, build_case_material
    from auto_pipeline.generator.ascend_arena import workspace as workspace_module

    client = CannBenchClient(ROOT)
    task_root = output / operator
    initial_command = None
    if args.docker:
        assert agent is not None
        initial_command = agent.build_docker_command(
            task_root / "work",
            ["./validate.sh", "init"],
            image=args.docker_image,
            device_id=device_id,
            environment={},
        )
    if task_root.exists():
        if args.force:
            shutil.rmtree(task_root)
        else:
            existing = _existing_workspace(task_root, operator)
            if existing.get("device_id") != device_id:
                raise ValueError("workspace device differs; use the original device or a new output directory")
            workdir = Path(existing["workspace"])
            existing_workspace = SimpleNamespace(
                workdir=workdir,
                validate_script=workdir / "validate.sh",
                reports_dir=workdir / "kernel_eval",
            )
            if initial_command is None:
                workspace_module._run_initial_evaluation(existing_workspace)
            else:
                workspace_module._run_initial_evaluation(existing_workspace, initial_command)
            return existing
    case = client.load_case("ascend_arena", str(TASKS / operator))
    if case.operator != operator:
        raise ValueError(f"operator mismatch: registry={operator}, task={case.operator}")
    workspace = workspace_module.prepare_ascend_arena_workspace(
        cann_bench_root=ROOT,
        case=case,
        material=build_case_material(case),
        workdir=task_root / "work",
        reports_dir=task_root / "work" / "kernel_eval",
        device_id=device_id,
        goal=args.goal,
        initial_evaluation_command=initial_command,
    )
    return {**workspace.to_dict(case=case), "task_root": str(task_root)}


def _prepare_workspaces(args: argparse.Namespace, output: Path, operators: list[str]) -> list[dict[str, Any]]:
    return [
        _prepare_workspace(
            args,
            output,
            operator,
            args.devices[index % len(args.devices)],
        )
        for index, operator in enumerate(operators)
    ]


def _initial_progress(
    path: Path,
    workspaces: list[dict[str, Any]],
    *,
    run_agent: bool,
    resume: bool,
) -> dict[str, Any]:
    from kernel_eval.optimize import workspace_evaluation

    previous = _load_json(path).get("operators", {})
    previous = previous if isinstance(previous, dict) else {}
    operators = dict(previous)
    for workspace in workspaces:
        name = str(workspace["operator"])
        state = previous.get(name, {}) if isinstance(previous.get(name), dict) else {}
        if resume and state.get("status") == "completed" and workspace_evaluation(workspace["workspace"])["success"]:
            operators[name] = state
        else:
            operators[name] = {
                "status": "pending" if run_agent else "prepared",
                "workspace": str(workspace["workspace"]),
                "device_id": workspace.get("device_id"),
            }
    return {"schema_version": 1, "updated_at": _now(), "operators": operators}


def _run_agents(
    args: argparse.Namespace,
    output: Path,
    workspaces: list[dict[str, Any]],
    progress: dict[str, Any],
    config: dict[str, Any],
) -> None:
    from ascend_arena_e2e.agent_base import OperatorAgent, agent_from_args
    from kernel_eval.batch_report import write_batch_report

    pending = {device_id: queue.Queue() for device_id in args.devices}
    by_name = {str(item["operator"]): item for item in workspaces}
    for index, (name, state) in enumerate(progress["operators"].items()):
        if name not in by_name:
            continue
        if state.get("status") != "completed":
            device_id = state.get("device_id")
            if device_id not in pending:
                device_id = by_name[name].get("device_id")
            if device_id not in pending:
                device_id = args.devices[index % len(args.devices)]
            pending[device_id].put(name)
    lock = threading.Lock()
    interrupted = threading.Event()
    stop_scheduling = threading.Event()
    OperatorAgent.reset_cancellation()
    concurrency = getattr(args, "agent_concurrency", 0) or len(args.devices)
    if concurrency < 1:
        raise ValueError("--agent-concurrency must be non-negative")
    slots = threading.Semaphore(concurrency)

    def update(name: str, values: dict[str, Any]) -> None:
        with lock:
            progress["operators"][name].update(values)
            progress["updated_at"] = _now()
            _write_json(output / "progress.json", progress)
            write_batch_report(output)

    def worker(device_id: int) -> None:
        agent = agent_from_args(args)
        assert agent is not None
        device_queue = pending[device_id]
        while not interrupted.is_set() and not stop_scheduling.is_set():
            try:
                name = device_queue.get_nowait()
            except queue.Empty:
                return
            workspace = by_name[name]
            try:
                update(name, {"status": "initializing", "device_id": device_id})
                _print_status(f"[{name}][NPU {device_id}] initializing baseline")
                prepared = _prepare_workspace(args, output, name, device_id, agent)
                if interrupted.is_set():
                    return
                with lock:
                    workspace.clear()
                    workspace.update(prepared)
                    config["updated_at"] = _now()
                    _write_json(output / "config.json", config)
                with slots:
                    if interrupted.is_set() or stop_scheduling.is_set():
                        update(name, {"status": "pending"})
                        return
                    update(name, {"status": "running", "started_at": _now()})
                    log_file = Path(workspace["workspace"]) / "artifact" / f"{agent.name}.log"
                    _print_status(f"[{name}][NPU {device_id}] started; log: {log_file}")
                    result = agent.run(
                        workspace["workspace"],
                        prompt=args.agent_prompt,
                        device_id=device_id,
                        docker_image=args.docker_image if args.docker else "",
                    )
                    if getattr(result, "error_kind", "") == "authentication_failed":
                        stop_scheduling.set()
                if interrupted.is_set():
                    return
                update(
                    name,
                    {
                        "status": "timeout" if getattr(result, "timed_out", False) else "completed" if result.ok else "failed",
                        "finished_at": _now(),
                        "agent_run": result.to_dict(),
                    },
                )
                reason = getattr(result, "error_kind", "") or (getattr(result, "evaluation", None) or {}).get("reason", "")
                status = ("timeout (exit 124)" if getattr(result, "timed_out", False)
                          else "completed" if result.ok else f"failed (exit {result.returncode}; {reason})")
                _print_status(f"[{name}][NPU {device_id}] {status}; log: {result.log_file}")
                if getattr(result, "error_kind", "") == "authentication_failed":
                    _print_status("[batch] authentication failed; stopping new tasks; fix credentials and resume")
            except Exception as exc:
                if interrupted.is_set():
                    return
                update(
                    name,
                    {
                        "status": "failed",
                        "finished_at": _now(),
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                _print_status(f"[{name}][NPU {device_id}] failed: {type(exc).__name__}: {exc}")
            finally:
                device_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(args.devices)) as executor:
        futures = [executor.submit(worker, device_id) for device_id in args.devices]
        try:
            for future in futures:
                future.result()
        except BaseException as exc:
            interrupted.set()
            OperatorAgent.cancel_active_runs()
            for name, state in progress["operators"].items():
                if state.get("status") in {"initializing", "running"}:
                    update(
                        name,
                        {
                            "status": "interrupted",
                            "finished_at": _now(),
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
            raise


def optimize(args: argparse.Namespace) -> dict[str, Any]:
    from ascend_arena_e2e.agent_base import agent_from_args
    from kernel_eval.batch_report import write_batch_report

    registry = _load_registry()
    operators = _select_operators(args.operators, registry)
    agent = agent_from_args(args)
    if agent is not None:
        agent.preflight(docker=bool(args.docker))
    if args.docker and agent is None:
        raise ValueError("--docker requires --agent")
    output = (
        args.output.expanduser().resolve()
        if args.output
        else ROOT / "benchmark_runs" / (_slug(f"{agent.name}-{agent.model}") if agent is not None else "manual")
    )
    _print_status(
        f"[batch] preparing {len(operators)} operator(s) on NPU {','.join(map(str, args.devices))}; output: {output}"
    )
    output.mkdir(parents=True, exist_ok=True)
    previous_config = _load_json(output / "config.json")
    if agent is None:
        workspaces = _prepare_workspaces(args, output, operators)
    else:
        previous_workspaces = {
            str(item.get("operator")): item
            for item in previous_config.get("workspaces", [])
            if isinstance(item, dict) and item.get("operator")
        }
        workspaces = []
        for index, operator in enumerate(operators):
            task_root = output / operator
            workspaces.append({
                **previous_workspaces.get(operator, {}),
                "operator": operator,
                "workspace": str(task_root / "work"),
                "task_root": str(task_root),
                "device_id": args.devices[index % len(args.devices)],
            })
    config = {
        "schema_version": 1,
        "created_at": previous_config.get("created_at") or _now(),
        "updated_at": _now(),
        "repository_root": str(ROOT),
        "repository_revision": _git_revision(),
        "registry": str(REGISTRY),
        "parameters": {
            "operators": operators,
            "output": str(output),
            "devices": args.devices,
            "batch_size": len(args.devices),
            "goal": args.goal,
            "agent": agent.name if agent is not None else "",
            "harness": agent.harness if agent is not None else "",
            "model": agent.model if agent is not None else "",
            "model_reasoning_effort": agent.model_reasoning_effort if agent is not None else "",
            "base_url": agent.base_url if agent is not None else "",
            "agent_environment": dict(agent.environment_variables) if agent is not None else {},
            "docker": bool(args.docker),
            "docker_image": args.docker_image if args.docker else "",
            "command": args.command,
            "agent_timeout_seconds": getattr(args, "agent_timeout", 0),
            "agent_concurrency": getattr(args, "agent_concurrency", 0),
        },
        "workspaces": workspaces,
    }
    # Keep unselected operators visible when rerunning a subset in the same directory.
    retained = [item for item in previous_config.get("workspaces", [])
                if isinstance(item, dict) and item.get("operator") not in operators]
    config["workspaces"] = retained + workspaces
    config["parameters"]["selected_operators"] = operators
    config["parameters"]["operators"] = list(dict.fromkeys(
        [*previous_config.get("parameters", {}).get("operators", []), *operators]))
    config["invocations"] = [*previous_config.get("invocations", []),
                             {"at": _now(), "parameters": dict(config["parameters"])}]
    _write_json(output / "config.json", config)
    progress = _initial_progress(
        output / "progress.json",
        workspaces,
        run_agent=agent is not None,
        resume=not args.restart and not args.force,
    )
    _write_json(output / "progress.json", progress)
    write_batch_report(output)
    if agent is not None:
        pending_count = sum(progress["operators"][name].get("status") != "completed" for name in operators)
        completed_count = len(operators) - pending_count
        _print_status(f"[batch] starting agent {agent.name}; pending: {pending_count}, resumed: {completed_count}")
        _run_agents(args, output, workspaces, progress, config)
    report = write_batch_report(output)
    _print_status(f"[batch] finished; report: {report}")
    return {
        "output": str(output),
        "config": str(output / "config.json"),
        "progress": str(output / "progress.json"),
        "report": str(report),
        "operators": progress["operators"],
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.agent_concurrency < 0:
        parser.error("--agent-concurrency must be non-negative")
    args.command = shlex.join(["python", "scripts/batch_optimize.py", *argv])
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def interrupt_on_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_on_sigterm)
    try:
        try:
            result = optimize(args)
        except KeyboardInterrupt:
            print("batch optimize interrupted", file=sys.stderr)
            return 130
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"batch optimize failed: {exc}", file=sys.stderr)
            return 1
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        from ascend_arena_e2e.agent_base import OperatorAgent
        OperatorAgent.reset_cancellation()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    statuses = {item.get("status") for item in result["operators"].values()}
    if statuses & {"failed", "interrupted"}:
        return 1
    return 124 if "timeout" in statuses else 0


if __name__ == "__main__":
    raise SystemExit(main())
