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


"""Validate every registered Ascend Arena operator on selected or idle NPUs."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from ascend_arena_e2e.devices import parse_device_mapping, npu_smi as _npu_smi
from ascend_arena_e2e.devices import discover_idle_devices as _discover_idle_devices
REGISTRY = ROOT / "src" / "ascend_arena_e2e" / "ops.json"
TASKS = ROOT / "bench_lab" / "ascend_arena_bench"
EVALUATE = ROOT / "scripts" / "run_evaluation.sh"
_PRINT_LOCK = threading.Lock()


def _print(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}




def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--devices must be comma-separated non-negative integers"
        ) from exc
    if not devices or any(device < 0 for device in devices) or len(devices) != len(set(devices)):
        raise argparse.ArgumentTypeError("--devices must contain unique non-negative device ids")
    return devices


def discover_idle_devices() -> tuple[list[int], list[int]]:
    return _discover_idle_devices(query=_npu_smi, log=_print)


def _operators() -> list[str]:
    registry = _read_json(REGISTRY)
    if not registry:
        raise RuntimeError(f"operator registry is empty or invalid: {REGISTRY}")
    operators = list(registry)
    missing = [name for name in operators if not (TASKS / name / name).is_dir()]
    if missing:
        raise RuntimeError("operator source directories are missing: " + ", ".join(missing))
    return operators


def _run_command(command: Sequence[str], *, operator: str, device_id: int, log: Any,
                 cwd: Path = ROOT) -> int:
    log.write(f"$ {shlex.join(command)}\n")
    log.flush()
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "ASCEND_ARENA_DEVICE_ID": str(device_id),
             "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log.write(line)
        log.flush()
        _print(f"[{operator}][NPU {device_id}] {line.rstrip()}")
    return process.wait()


def _correctness_passed(results_dir: Path) -> bool:
    for stage in ("smoke", "complete", "profile"):
        payload = _read_json(results_dir / f"{stage}-correctness_result.json")
        cases = payload.get("case_results")
        if (
            payload.get("success") is not True
            or not isinstance(cases, list)
            or not cases
            or not all(isinstance(case, dict) and case.get("passed") is True for case in cases)
        ):
            return False
    return True


def summarize_operator(
    operator_root: Path,
    *,
    operator: str,
    device_id: int,
    baseline_returncode: int | None,
    candidate_returncode: int | None,
    mode: str = "source",
) -> dict[str, Any]:
    reports = operator_root / "work/kernel_eval" if mode == "workspace" else operator_root / "kernel_eval"
    runs = sorted(path for path in reports.glob(f"ascend_arena_{operator}_*") if path.is_dir())
    latest_run = runs[-1] if runs else None
    candidate_path = latest_run / "results/status.json" if latest_run else None
    if candidate_path is not None and not candidate_path.is_file():
        candidate_path = None
    baseline_path = (
        latest_run / "baseline_results/status.json"
        if latest_run and mode == "source" else reports / "init_results/status.json"
    )
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path) if candidate_path else {}
    results_dir = candidate_path.parent if candidate_path else reports / "missing-results"
    baseline_results_dir = baseline_path.parent

    compile_ok = (
        baseline.get("build_success") is True
        and candidate.get("build_success") is True
    )
    correctness_ok = _correctness_passed(baseline_results_dir) and _correctness_passed(results_dir)
    performance_ok = (
        baseline.get("profile_success") is True
        and int(baseline.get("profile_cases") or 0) > 0
        and candidate.get("profile_success") is True
        and int(candidate.get("profile_cases") or 0) > 0
    )
    success = (
        baseline_returncode == 0
        and candidate_returncode == 0
        and compile_ok
        and correctness_ok
        and performance_ok
    )
    return {
        "operator": operator,
        "mode": mode,
        "device_id": device_id,
        "success": success,
        "compile": compile_ok,
        "correctness": correctness_ok,
        "performance": performance_ok,
        "baseline_returncode": baseline_returncode,
        "candidate_returncode": candidate_returncode,
        "baseline_status": str(baseline_path),
        "candidate_status": str(candidate_path) if candidate_path else "",
        "log": str(operator_root / "evaluation.log"),
    }


def _evaluate_operator(operator: str, device_id: int, output: Path, mode: str = "source") -> dict[str, Any]:
    operator_root = output / operator
    reports = operator_root / "kernel_eval"
    operator_root.mkdir(parents=True)
    common = [
        str(EVALUATE),
        "--run-optimize",
        "--source-dir", str(TASKS / operator / operator),
        "--task-dir", str(TASKS / operator),
        "--operator", operator,
        "--reports-dir", str(reports),
        "--device-id", str(device_id),
    ]

    _print(f"[{operator}][NPU {device_id}] started")
    with (operator_root / "evaluation.log").open("w", encoding="utf-8") as log:
        prepare_returncode = None
        candidate_returncode = None
        if mode == "workspace":
            prepare_returncode = _run_command(
                [sys.executable, "-m", "auto_pipeline.cli", "prepare", "--operator", operator,
                 "--device-id", str(device_id), "--output", str(output), "--workspace-only", "--resume"],
                operator=operator, device_id=device_id, log=log,
            )
            evaluation_returncode = None
            if prepare_returncode == 0:
                work = operator_root / "work"
                evaluation_returncode = _run_command(
                    ["./validate.sh", "init", "--device-check", "require-idle"],
                    operator=operator, device_id=device_id, log=log, cwd=work,
                )
                if evaluation_returncode == 0:
                    candidate_returncode = _run_command(
                        ["./validate.sh", "--device-check", "require-idle"],
                        operator=operator, device_id=device_id, log=log, cwd=work,
                    )
        else:
            evaluation_returncode = _run_command(
                common, operator=operator, device_id=device_id, log=log
            )
            candidate_returncode = evaluation_returncode

    result = summarize_operator(
        operator_root,
        operator=operator,
        device_id=device_id,
        baseline_returncode=evaluation_returncode,
        candidate_returncode=candidate_returncode,
        mode=mode,
    )
    result["prepare_returncode"] = prepare_returncode
    state = "PASSED" if result["success"] else "FAILED"
    _print(
        f"[{operator}][NPU {device_id}] {state} "
        f"compile={result['compile']} correctness={result['correctness']} "
        f"performance={result['performance']}"
    )
    return result


def _run_batch(operators: list[str], devices: list[int], output: Path, mode: str = "source") -> list[dict[str, Any]]:
    tasks: queue.Queue[str] = queue.Queue()
    for operator in operators:
        tasks.put(operator)
    results: list[dict[str, Any]] = []
    result_lock = threading.Lock()

    def device_worker(device_id: int) -> None:
        while True:
            try:
                operator = tasks.get_nowait()
            except queue.Empty:
                return
            try:
                result = _evaluate_operator(operator, device_id, output, mode)
            except Exception as exc:
                _print(f"[{operator}][NPU {device_id}] FAILED: {type(exc).__name__}: {exc}")
                result = {
                    "operator": operator,
                    "device_id": device_id,
                    "success": False,
                    "compile": False,
                    "correctness": False,
                    "performance": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            with result_lock:
                results.append(result)
            tasks.task_done()

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(device_worker, device_id) for device_id in devices]
        for future in futures:
            future.result()
    return sorted(results, key=lambda item: operators.index(str(item["operator"])))


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate all Ascend Arena operators using selected or automatically discovered idle NPUs."
    )
    parser.add_argument("--mode", choices=("source", "workspace"), default="source",
                        help="source: evaluate original and candidate; workspace: prepare, validate.sh init, validate.sh")
    parser.add_argument(
        "--output",
        type=Path,
        help="new result directory; defaults to benchmark_runs/ascend-arena-e2e-<timestamp>",
    )
    parser.add_argument(
        "--devices",
        type=parse_devices,
        help="comma-separated logical NPU ids, for example 0,1,2,3; defaults to all idle NPUs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        operators = _operators()
        idle_devices, busy_devices = discover_idle_devices()
        if not idle_devices:
            raise RuntimeError("no idle NPU is available")
        requested_devices = args.devices
        unavailable = [device for device in requested_devices or [] if device not in idle_devices]
        if unavailable:
            raise RuntimeError(
                "requested NPUs are busy or unavailable: " + ",".join(map(str, unavailable))
            )
        devices = (requested_devices or idle_devices)[: len(operators)]
        num_processes = len(devices)
        output = (
            args.output.expanduser().resolve()
            if args.output
            else ROOT / "benchmark_runs" / f"ascend-arena-e2e-{datetime.now():%Y%m%d-%H%M%S}"
        )
        if output.exists():
            raise RuntimeError(f"output directory already exists: {output}")
        output.mkdir(parents=True)
    except (OSError, RuntimeError) as exc:
        print(f"batch evaluation failed: {exc}", file=sys.stderr)
        return 2

    _print(f"[devices] idle: {','.join(map(str, idle_devices))}")
    _print(f"[devices] busy or unavailable: {','.join(map(str, busy_devices)) or 'none'}")
    _print(f"[devices] requested: {','.join(map(str, requested_devices)) if requested_devices else 'auto'}")
    _print(f"[devices] selected: {','.join(map(str, devices))}")
    _print(
        f"[batch] mode={args.mode} operators={len(operators)} num_processes={num_processes} "
        f"output={output}"
    )
    results = _run_batch(operators, devices, output, args.mode)
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "operators": operators,
        "mode": args.mode,
        "requested_devices": requested_devices,
        "devices": devices,
        "idle_devices": idle_devices,
        "busy_devices": busy_devices,
        "num_processes": num_processes,
        "success": all(item.get("success") is True for item in results),
        "results": results,
    }
    summary_path = output / "e2e_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print(f"[batch] {'PASSED' if summary['success'] else 'FAILED'}; summary: {summary_path}")
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
