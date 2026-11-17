#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------



"""Initialize or evaluate an Ascend Arena operator in an ephemeral sandbox."""

from __future__ import annotations

import hashlib
import math
import tarfile
import argparse
import difflib
import html
import importlib.util
import json
import os
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable

from ascend_arena_e2e.ascend_arena.utils.paths import resolve_compile_repo

PROTECTED = {"api_desc.md", "reference.py", "binding", "cases", "op_info.json"}
IMPLEMENTATION_IGNORED = {".git", "__pycache__", "docs", "tests"}
WORKSPACE_RUNTIME_IGNORED = {
    "artifact",
    "kernel_eval",
    "AGENTS.md",
    "REQUIRE.md",
    "validate.sh",
    "workspace.json",
}
_SOURCE_DIFF_IGNORED_DIRS = IMPLEMENTATION_IGNORED | {"build", "dist"}
_SOURCE_DIFF_IGNORED_FILES = {".source"}
_SOURCE_DIFF_IGNORED_SUFFIXES = {".a", ".o", ".pyc", ".so"}
_STAGE_LOG_NAMES = (
    "compile.log",
    "complete-correctness.log",
    "profile-correctness.log",
    "profile.log",
    "smoke-correctness.log",
)


class _EvaluationInterrupted(Exception):
    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


def _interrupt_evaluation(signum: int, _frame: Any) -> None:
    raise _EvaluationInterrupted(signum)


def _write_status_payload(path: Path, status: Dict[str, Any]) -> None:
    from kernel_eval.eval.evaluator import _write_json_atomic

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name == "status.json":
        _update_stage_status(path.parent, status)
    _write_json_atomic(str(path), status)


def _update_stage_status(stage_dir: Path, status: Dict[str, Any]) -> None:
    """Describe completed, skipped and unavailable stages using retained evidence."""
    def entry(state: str, filename: str, log_name: str) -> Dict[str, Any]:
        return {
            "status": state,
            "success": state == "passed" if state in {"passed", "failed"} else None,
            "result_json": filename if (stage_dir / filename).is_file() else None,
            "log": log_name if (stage_dir / log_name).is_file() else None,
        }

    compile_path = stage_dir / "compile_result.json"
    if not compile_path.exists() and not status.get("interrupted") and "build_success" in status:
        _write_status_payload(compile_path, {"success": status["build_success"]})
    compiled = _load_json(compile_path).get("success")
    state = ("passed" if compiled is True else "failed" if compiled is False
             else "interrupted" if status.get("interrupted") else "unknown")
    stages = {"compile": entry(state, "compile_result.json", "compile.log")}
    if compiled is not None:
        status["build_success"] = compiled
    blocked = state != "passed"
    requested = status.get("requested_stages", ["smoke", "complete", "profile"])
    for phase, name in (("smoke", "smoke"), ("complete", "complete"), ("profile", "profile_correctness")):
        filename = f"{phase}-correctness_result.json"
        data = _load_json(stage_dir / filename)
        cases = data.get("case_results", [])
        passed = sum(case.get("passed") is True for case in cases)
        if phase not in requested or data.get("reason") in {
            "skipped_not_requested", "skipped_profile_correctness_not_requested",
        }:
            state = "skipped"
        elif data:
            state = "passed" if data.get("success") is True and cases and passed == len(cases) else "failed"
        elif blocked:
            state = "not_run"
        else:
            state = "interrupted" if status.get("interrupted") else "unknown"
        stages[name] = entry(state, filename, f"{phase}-correctness.log")
        stages[name].update(total_cases=len(cases), passed_cases=passed, failed_cases=len(cases) - passed)
        if data.get("reason") or data.get("error"):
            stages[name]["message"] = data.get("error") or data["reason"]
        blocked = blocked or state not in {"passed", "skipped"}

    message = status.get("profile_message", "")
    if "profile" not in requested or message in {"skipped_performance_disabled", "skipped_not_requested"}:
        state = "skipped"
    elif blocked:
        state = "not_run"
    elif status.get("interrupted"):
        state = "interrupted"
    elif status.get("profile_success") is False:
        state = "failed"
    elif status.get("profile_success") is True and _load_json(stage_dir / "result.json"):
        state = "passed"
    else:
        state = "unknown"
    stages["performance"] = entry(state, "result.json", "profile.log")
    stages["performance"].update(total_cases=status.get("profile_cases", 0), message=message)
    if state in {"skipped", "not_run"}:
        stages["performance"]["result_json"] = None
    status["stages"] = stages
    status["all_stages_passed"] = all(phase["status"] == "passed" for phase in stages.values())
    for key, name in (("compile_log", "compile"), ("profile_log", "performance")):
        if key in status:
            status[key] = stages[name]["log"]


def _record_stage_completion(stage_dir: Path, event: Dict[str, Any]) -> None:
    filenames = {
        "build_finished": "compile_result.json",
        "smoke_correctness_finished": "smoke-correctness_result.json",
        "complete_correctness_finished": "complete-correctness_result.json",
        "correctness_finished": "profile-correctness_result.json",
    }
    filename = filenames.get(event.get("label"))
    if filename and not (stage_dir / filename).is_file():
        _write_status_payload(stage_dir / filename, {
            "success": event["success"], "op_name": event.get("op_name"),
            "reason": event.get("reason", ""), "case_results": [],
        })


def _resolve_env_script(explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"env-script does not exist: {path}")
        return path

    current_home = os.environ.get("ASCEND_HOME_PATH", "")
    candidates = []
    if current_home:
        candidates.append(Path(current_home) / "set_env.sh")
    user_root = Path.home() / "Ascend"
    candidates.extend(sorted(user_root.glob("cann-*/set_env.sh"), reverse=True))
    candidates.extend(sorted(Path("/usr/local/Ascend").glob("cann-*/set_env.sh"), reverse=True))
    for script in candidates:
        if script.is_file():
            return script
    return None


def _copy_tree(source: Path, target: Path, *, ignore_names: Iterable[str] = ()) -> None:
    target.mkdir(parents=True, exist_ok=True)
    ignored = IMPLEMENTATION_IGNORED | set(ignore_names)
    for child in source.iterdir():
        if child.name in ignored:
            continue
        destination = target / child.name
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(child, destination)


def _copy_benchmark_tree(source: Path, target: Path, *, operator: str) -> None:
    """Copy benchmark assets while ignoring a prepared workspace's extras."""
    _copy_tree(source, target, ignore_names={operator, *WORKSPACE_RUNTIME_IGNORED})


def _copy_dependency_tree(source: Path, target: Path, *, compile_repo: str) -> None:
    """Copy only the dependency repository needed by this operator."""
    target.mkdir(parents=True, exist_ok=True)
    repo = resolve_compile_repo(source, compile_repo)
    opbase = source / "ops-base"
    if compile_repo == "ops-skeleton" and (opbase / ".git").exists():
        if " " in str(opbase.resolve()):
            # CANN's generated compiler commands cannot quote include paths containing spaces.
            shutil.copytree(opbase, target / "ops-base", symlinks=True)
            if (opbase / ".git").is_file():
                git_dir = subprocess.check_output(
                    ["git", "-C", str(opbase), "rev-parse", "--absolute-git-dir"], text=True,
                ).strip()
                (target / "ops-base/.git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
        else:
            (target / "ops-base").symlink_to(opbase.resolve(), target_is_directory=True)
    archive = source / f"{compile_repo}.tar.gz"
    if archive.is_file() and not (repo / "CMakeLists.txt").is_file():
        manifest = json.loads((source / f"{compile_repo}.json").read_text())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
            raise ValueError(f"skeleton archive checksum mismatch: {archive}")
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                path = Path(member.name)
                if (path.is_absolute() or ".." in path.parts or not path.parts
                        or path.parts[0] != compile_repo or not (member.isfile() or member.isdir())):
                    raise ValueError(f"unsafe skeleton archive member: {member.name}")
            if hasattr(tarfile, "data_filter"):
                bundle.extractall(target, filter="data")
            else:
                bundle.extractall(target)  # Members already restricted to regular files/directories.
        return
    if not (repo / "CMakeLists.txt").is_file():
        raise FileNotFoundError(
            f"dependency repository not found: {repo}; "
            "run python3 scripts/setup_ascend_arena_dependencies.py from the project root"
        )
    shutil.copytree(repo, target / compile_repo, symlinks=True, ignore=shutil.ignore_patterns(".git"))


def _overlay(source: Path, target: Path, *, reject_protected: bool = True) -> None:
    if not source.is_dir():
        raise ValueError(f"source-dir must be a directory: {source}")
    for child in source.iterdir():
        if child.is_symlink():
            raise ValueError(f"source-dir must not contain symlinks: {child}")
        if child.name in IMPLEMENTATION_IGNORED:
            continue
        if child.name in PROTECTED and reject_protected:
            raise ValueError(f"source-dir must not override benchmark asset: {child.name}")
        if child.name in PROTECTED:
            continue
        destination = target / child.name
        if destination.exists() and destination.is_dir() and not child.is_dir():
            shutil.rmtree(destination)
        if destination.exists() and destination.is_file() and child.is_dir():
            destination.unlink()
        if child.is_dir():
            shutil.copytree(child, destination, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(child, destination)


def _find_benchmark_op(task_root: Path, operator: str) -> Path:
    direct = task_root / operator
    if (direct / "cases/proto.yaml").exists():
        return direct
    if (task_root / "cases/proto.yaml").exists():
        return task_root
    for proto in task_root.rglob("proto.yaml"):
        op_dir = proto.parent.parent if proto.parent.name == "cases" else proto.parent
        if op_dir.name.lower() == operator.lower() and (op_dir / "reference.py").exists():
            return op_dir
    raise FileNotFoundError(f"Arena benchmark operator not found: {operator} under {task_root}")


def _resolve_source_dir(source: Path, operator: str) -> Path:
    """Accept either a direct implementation directory or an operator wrapper."""
    nested = source / operator
    if (nested / "op_host").is_dir() and (nested / "op_kernel").is_dir():
        return nested
    return source


def _load_compile_entry(ops_json: Path, operator: str, dependency_root: Path) -> tuple[str, str | None, str]:
    payload: Any = {}
    if ops_json.exists():
        try:
            payload = json.loads(ops_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid ops registry: {ops_json}: {exc}") from exc
    entry = payload.get(operator) if isinstance(payload, dict) else None
    if entry is None and isinstance(payload, dict):
        entry = next((value for key, value in payload.items() if str(key).lower() == operator.lower()), None)
    if isinstance(entry, dict):
        repo = str(entry.get("compile-repo") or "common")
        family = str(entry.get("compile-family") or "") or None
        op_name = str(entry.get("compile-op-name") or operator)
        return repo, family, op_name

    for repo_dir in sorted(dependency_root.glob("ops-*")):
        for op_dir in repo_dir.rglob(operator):
            if op_dir.is_dir() and (op_dir / "op_host").is_dir() and (op_dir / "op_kernel").is_dir():
                try:
                    family = op_dir.parent.relative_to(repo_dir).as_posix()
                except ValueError:
                    family = ""
                return repo_dir.name, family or None, op_dir.name
    raise ValueError(f"operator {operator} is absent from {ops_json} and dependency repositories")


def _preflight_dependencies(
    *, project_root: Path, dependency_root: Path, compile_repo: str, env_script: Path | None
) -> None:
    missing_modules = [
        name for name in ("pybind11", "torch", "torch_npu", "yaml")
        if importlib.util.find_spec(name) is None
    ]
    missing_tools = [name for name in ("cmake", "make", "g++") if shutil.which(name) is None]
    if env_script is None and shutil.which("bisheng") is None:
        missing_tools.append("bisheng (source the CANN set_env.sh first)")
    if not missing_modules and not missing_tools:
        return

    install_script = dependency_root / compile_repo / "install_deps.sh"
    details = []
    if missing_modules:
        details.append("missing Python modules: " + ", ".join(missing_modules))
    if missing_tools:
        details.append("missing build tools: " + ", ".join(missing_tools))
    hints = [f"install Python dependencies: python -m pip install -r {project_root / 'requirements.txt'}"]
    if install_script.is_file():
        hints.append(f"install {compile_repo} system dependencies: bash {install_script}")
    raise RuntimeError("; ".join([*details, *hints]))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _timing_record(records):
    if isinstance(records, list):
        for scope in ("kernel_total", "operator_call"):
            for record in records:
                if isinstance(record, dict) and record.get("duration_scope") == scope:
                    return record
    return {}


def _timing_scope(records: Any) -> str:
    return _timing_record(records).get("duration_scope", "kernel")


def _measurement_protocol(records):
    return _timing_record(records).get("measurement_protocol")


def _operator_call_duration(records):
    calls = [r for r in records or [] if isinstance(r, dict) and r.get("duration_scope") == "operator_call"]
    return _duration(calls) or None


def _duration(records: Any) -> float:
    if isinstance(records, list):
        primary = _timing_record(records)
        if primary:
            records = [primary]
        elif len(records) > 1 and any(isinstance(item, dict) and item.get("op_name") for item in records):
            # Kernel groups do not identify invocation boundaries or stream dependencies.
            return 0.0
    values = []
    if isinstance(records, list):
        for item in records:
            if not isinstance(item, dict):
                continue
            for key in ("task_duration_us", "duration_us", "elapsed_us"):
                value = item.get(key)
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    values.append(value)
                break
    return float(statistics.median(values)) if values else 0.0


def _duration_map(payload: Dict[str, Any]) -> Dict[str, float | None]:
    return {str(case_id): _duration(records) or None for case_id, records in payload.items()}


def _noise_band(records):
    if not isinstance(records, list) or not records:
        return None
    primary = _timing_record(records)
    if primary:
        records = [primary]
    bands = [item.get("noise_band") for item in records if isinstance(item, dict)]
    if len(bands) != len(records) or any(not isinstance(band, (int, float)) or not math.isfinite(band)
                                       or band < 0 for band in bands):
        return None
    return max(bands)


def _implementation_files(root: Path) -> Dict[str, Path]:
    files = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in _SOURCE_DIFF_IGNORED_DIRS for part in relative.parts):
            continue
        if relative.name in _SOURCE_DIFF_IGNORED_FILES:
            continue
        if path.suffix in _SOURCE_DIFF_IGNORED_SUFFIXES:
            continue
        files[relative.as_posix()] = path
    return files


def _diff_lines(path: Path) -> list[str] | None:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return None


def source_fingerprint(source_dir: Path) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(_implementation_files(source_dir).items()):
        digest.update(name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def workspace_evaluation(workspace: Path | str) -> Dict[str, Any]:
    """Require the latest complete performance evaluation to match the final source."""
    workspace = Path(workspace)
    metadata = _load_json(workspace / "workspace.json")
    operator = metadata.get("operator")
    runs = [run for run in (workspace / "kernel_eval").glob(f"ascend_arena_{operator}_*")
            if run.is_dir()]
    if not runs:
        return {"success": False, "reason": "no_candidate_evaluation"}
    run = max(runs, key=lambda item: item.name)
    run_name = run.name
    report = _load_json(run / f"{run_name}.json")
    if report.get("operator") != operator:
        return {"success": False, "reason": "incomplete_evaluation", "run_name": run_name}
    summary = report.get("summary", {})
    cases = report.get("cases", [])
    success = (report.get("candidate_status", {}).get("success") is True
               and report.get("baseline_status", {}).get("success") is True
               and bool(cases) and summary.get("compared_cases") == len(cases)
               and all(case.get("candidate_passed") and isinstance(case.get("speedup"), (int, float))
                       and math.isfinite(case["speedup"]) and case["speedup"] > 0 for case in cases))
    source = workspace / str(metadata.get("operator", ""))
    saved = report.get("source_diff", {})
    if saved.get("candidate_sha256"):
        matches = source.is_dir() and source_fingerprint(source) == saved["candidate_sha256"]
    else:
        task = Path(str(metadata.get("task_dir", "")))
        original = task / str(metadata.get("operator", ""))
        matches = (original.is_dir() and source.is_dir() and
                   _collect_source_diff(baseline_dir=original, candidate_dir=source)["unified_diff"]
                   == saved.get("unified_diff"))
    return {"success": bool(success and matches), "run_name": run_name,
            "source_changed": bool(saved["changed_file_count"]) if "changed_file_count" in saved else None,
            "source_matches": bool(matches),
            "reason": "validated" if success and matches else
                      "final_source_unvalidated" if not matches else "evaluation_failed"}


def _collect_source_diff(*, baseline_dir: Path, candidate_dir: Path) -> Dict[str, Any]:
    """Collect a reviewable implementation diff without retaining the sandbox."""
    baseline_files = _implementation_files(baseline_dir)
    candidate_files = _implementation_files(candidate_dir)
    changed_files = []
    file_diffs = []
    diff = []
    for relative in sorted(set(baseline_files) | set(candidate_files)):
        before_path = baseline_files.get(relative)
        after_path = candidate_files.get(relative)
        if before_path is None:
            status = "added"
            before_lines, after_lines = [], _diff_lines(after_path)
        elif after_path is None:
            status = "deleted"
            before_lines, after_lines = _diff_lines(before_path), []
        else:
            before_lines, after_lines = _diff_lines(before_path), _diff_lines(after_path)
            if before_lines == after_lines:
                continue
            status = "modified"

        changed_files.append({"path": relative, "status": status})
        if before_lines is None or after_lines is None:
            file_diff = [
                f"diff --git a/{relative} b/{relative}",
                f"Binary files a/{relative} and b/{relative} differ",
            ]
        else:
            file_diff = list(difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=f"a/{relative}",
                tofile=f"b/{relative}",
                lineterm="",
            ))
        diff.extend(file_diff)
        file_diffs.append({
            "path": relative,
            "status": status,
            "diff": "\n".join(file_diff) + "\n",
        })

    diff_text = "\n".join(diff)
    if diff_text:
        diff_text += "\n"
    else:
        diff_text = "# No implementation source changes detected.\n"
    return {
        "candidate_sha256": source_fingerprint(candidate_dir),
        "baseline_sha256": source_fingerprint(baseline_dir),
        "changed_file_count": len(changed_files),
        "added_files": [item["path"] for item in changed_files if item["status"] == "added"],
        "deleted_files": [item["path"] for item in changed_files if item["status"] == "deleted"],
        "modified_files": [item["path"] for item in changed_files if item["status"] == "modified"],
        "changed_files": changed_files,
        "file_diffs": file_diffs,
        "unified_diff": diff_text,
    }


def _profile_result(run_result: Any, output: Path) -> Dict[str, Any]:
    payload = _load_json(output)
    result_path = run_result.profile_result.output_dir.parent / "result.json"
    if not payload:
        payload = _load_json(result_path)
    if not payload and isinstance(run_result.profile_result.metrics, dict):
        payload = run_result.profile_result.metrics
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _stage_artifact_paths(stage_dir: Path, result_path: Path) -> Dict[str, Path]:
    return {
        "BENCHMARK_RESULT_JSON_PATH": result_path.resolve(),
        "BENCHMARK_COMPILE_LOG_PATH": stage_dir / "compile.log",
        "BENCHMARK_SMOKE_CORRECTNESS_LOG_PATH": stage_dir / "smoke-correctness.log",
        "BENCHMARK_SMOKE_CORRECTNESS_RESULT_JSON_PATH": stage_dir / "smoke-correctness_result.json",
        "BENCHMARK_COMPLETE_CORRECTNESS_LOG_PATH": stage_dir / "complete-correctness.log",
        "BENCHMARK_COMPLETE_CORRECTNESS_RESULT_JSON_PATH": stage_dir / "complete-correctness_result.json",
        "BENCHMARK_PROFILE_CORRECTNESS_LOG_PATH": stage_dir / "profile-correctness.log",
        "BENCHMARK_PROFILE_CORRECTNESS_RESULT_JSON_PATH": stage_dir / "profile-correctness_result.json",
        "BENCHMARK_PROFILE_LOG_PATH": stage_dir / "profile.log",
    }


def _copy_profile_artifacts(source_dir: Path, stage_dir: Path) -> None:
    """Keep profiler output after the temporary sandbox is removed."""
    if not source_dir.is_dir():
        return
    destination = stage_dir / "profile"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source_dir, destination, symlinks=True)


def _run_stage_in_process(*, standard_dir: Path, operator: str, dependency_root: Path, run_root: Path,
               chip: str, device_id: int, compile_repo: str, compile_family: str | None,
               compile_op_name: str, env_script: Path | None, warmup: int, repeat: int,
               profile_backend: str, compile_jobs: int, isolated_build: bool,
               profile_performance: bool, stages: frozenset[str],
               result_path: Path) -> tuple[Any, Dict[str, Any]]:
    from ascend_arena_e2e.ascend_arena.runners.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(
        dependency_root=dependency_root,
        env_script=env_script,
        profile_warmup=warmup,
        profile_repeat=repeat,
        profile_backend=profile_backend,
        compile_repo=compile_repo,
        compile_family=compile_family,
        compile_op_name=compile_op_name,
        compile_jobs=compile_jobs,
        isolated_build=isolated_build,
        profile_requires_correctness=stages != frozenset({"profile"}),
        profile_performance=profile_performance,
    )
    stage_dir = result_path.parent.resolve()
    artifact_paths = _stage_artifact_paths(stage_dir, result_path)
    previous = {name: os.environ.get(name) for name in artifact_paths}
    for name, path in artifact_paths.items():
        os.environ[name] = str(path)
    try:
        result = runner.run(
            standard_dir=standard_dir,
            op_name=operator,
            chip_name=chip,
            run_root=run_root,
            device_id=device_id,
            reload_build=False,
            stages=stages,
        )
        _copy_profile_artifacts(result.profile_result.output_dir, stage_dir)
        return result, _profile_result(result, result_path)
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


_STAGE_WORKER_MARKER = "__CANN_BENCH_STAGE_RESULT__"


def _stage_spec_from_kwargs(**kwargs: Any) -> dict[str, Any]:
    spec = dict(kwargs)
    spec["standard_dir"] = str(spec["standard_dir"])
    spec["dependency_root"] = str(spec["dependency_root"])
    spec["run_root"] = str(spec["run_root"])
    spec["result_path"] = str(spec["result_path"])
    spec["env_script"] = str(spec["env_script"]) if spec["env_script"] is not None else ""
    spec["stages"] = sorted(spec["stages"])
    return spec


def _stage_result_metadata(result: Any, payload: Dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "build_success": bool(result.build_result.success),
        "profile_success": bool(result.profile_result.success),
        "compile_log": str(result.build_result.log_path),
        "profile_log": str(result.profile_result.log_path),
        "profile_message": result.profile_result.message,
        "profile_cases": len(payload),
    }


def _stage_worker_main(spec_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["standard_dir"] = Path(spec["standard_dir"])
    spec["dependency_root"] = Path(spec["dependency_root"])
    spec["run_root"] = Path(spec["run_root"])
    spec["result_path"] = Path(spec["result_path"])
    spec["env_script"] = Path(spec["env_script"]) if spec["env_script"] else None
    spec["stages"] = frozenset(spec["stages"])
    result, payload = _run_stage_in_process(**spec)
    print(_STAGE_WORKER_MARKER + json.dumps(_stage_result_metadata(result, payload), ensure_ascii=False), flush=True)
    return 0


def _print_stage_progress(event: dict[str, Any], context: str) -> None:
    names = {
        "build": "编译", "smoke_correctness": "smoke 小测试集正确性",
        "complete_correctness": "complete 完备测试集正确性",
        "correctness": "profile 性能测试集正确性", "profile": "性能采集",
        "profile_case": "性能用例",
    }
    label = event.get("label", "")
    stage, _, state = label.rpartition("_")
    if stage not in names or state not in {"start", "finished", "retry"}:
        return
    # ProfileRunner supplies the total; omit the outer runner's duplicate start.
    if label == "profile_start" and "case_count" not in event:
        return
    outcome = (
        "开始" if state == "start" else
        "重试" if state == "retry" else
        ("通过" if event.get("success") else "失败")
    )
    detail = str(event.get("case_id", ""))
    if "case_count" in event:
        detail += f" 用例数={event['case_count']}"
    if "passed_count" in event:
        detail += f" 通过={event['passed_count']} 失败={event['case_count'] - event['passed_count']}"
    for duration in event.get("durations_us", []):
        detail += f" {float(duration):.4f} μs"
    if event.get("reason"):
        detail += f" ({event['reason']})"
    if state == "finished" and not event.get("success") and event.get("log_path"):
        detail += f" 日志={event['log_path']}"
    print(f"[{context}] {names[stage]} {outcome} {detail}".rstrip(), flush=True)


def _run_stage(*, standard_dir: Path, operator: str, dependency_root: Path, run_root: Path,
               chip: str, device_id: int, compile_repo: str, compile_family: str | None,
               compile_op_name: str, env_script: Path | None, warmup: int, repeat: int,
               profile_backend: str, compile_jobs: int, isolated_build: bool,
               profile_performance: bool, stages: frozenset[str],
               result_path: Path) -> tuple[Any, Dict[str, Any]]:
    """Run one candidate/baseline stage in a fresh process.

    CANN's custom-op registry and loaded vendor libraries are process-global, so
    clearing Python modules cannot isolate a candidate from the baseline. The
    worker process gives each stage a fresh ACL/CANN runtime as well.
    """
    stage_dir = result_path.parent.resolve()
    stage_dir.mkdir(parents=True, exist_ok=True)
    # Reinitializing a baseline must not reuse results from its previous run.
    for filename in ("compile_result.json", "status.json", result_path.name, *_STAGE_LOG_NAMES,
                     *(f"{phase}-correctness_result.json" for phase in ("smoke", "complete", "profile"))):
        (stage_dir / filename).unlink(missing_ok=True)
    spec_path = stage_dir / ".stage_request.json"
    spec_path.write_text(json.dumps(_stage_spec_from_kwargs(
        standard_dir=standard_dir,
        operator=operator,
        dependency_root=dependency_root,
        run_root=run_root,
        chip=chip,
        device_id=device_id,
        compile_repo=compile_repo,
        compile_family=compile_family,
        compile_op_name=compile_op_name,
        env_script=env_script,
        warmup=warmup,
        repeat=repeat,
        profile_backend=profile_backend,
        compile_jobs=compile_jobs,
        isolated_build=isolated_build,
        profile_performance=profile_performance,
        stages=stages,
        result_path=result_path,
    ), ensure_ascii=False), encoding="utf-8")
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")) if part
        )
        metadata = None
        process_log = stage_dir / "stage-process.log"
        context = f"{operator}/{run_root.name}/NPU:{device_id}"
        with process_log.open("w", encoding="utf-8") as log, subprocess.Popen(
            [sys.executable, "-m", "kernel_eval.optimize", "--stage-worker", str(spec_path)],
            cwd=str(Path(__file__).resolve().parents[2]), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        ) as proc:
            from ascend_arena_e2e.worker import stream_worker
            timeout = float(os.environ.get("ASCEND_ARENA_STAGE_TIMEOUT", "7200"))
            try:
                for line in stream_worker(proc, timeout):
                    log.write(line)
                    log.flush()
                    if line.startswith(_STAGE_WORKER_MARKER):
                        metadata = json.loads(line[len(_STAGE_WORKER_MARKER):])
                    elif line.startswith("[progress] "):
                        try:
                            event = json.loads(line[len("[progress] "):])
                            _record_stage_completion(stage_dir, event)
                            _print_stage_progress(event, context)
                        except (ValueError, TypeError):
                            pass
            except (TimeoutError, subprocess.TimeoutExpired) as exc:
                metadata = None
                log.write(f"[TIMEOUT] {exc}\n")
                print(f"[{context}] 评测超时：{exc}；日志={process_log}", flush=True)
            if proc.returncode != 0:
                metadata = None
        if metadata is not None and metadata.get("success") and proc.returncode == 0:
            process_log.unlink(missing_ok=True)
        payload = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        if metadata is None:
            error = f"stage worker failed (exit={proc.returncode}); see {process_log}"
            print(f"[{context}] 评测失败：{error}", flush=True)
            metadata = {
                "success": False,
                "build_success": _load_json(stage_dir / "compile_result.json").get("success", False),
                "profile_success": False,
                "compile_log": str(stage_dir / "compile.log"),
                "profile_log": str(stage_dir / "profile.log"),
                "profile_message": error,
                "profile_cases": len(payload),
            }
        build_result = SimpleNamespace(
            success=metadata["build_success"],
            log_path=Path(metadata["compile_log"]),
        )
        profile_result = SimpleNamespace(
            success=metadata["profile_success"],
            log_path=Path(metadata["profile_log"]),
            output_dir=stage_dir / "profile",
            message=metadata["profile_message"],
        )
        result = SimpleNamespace(
            success=metadata["success"],
            build_result=build_result,
            profile_result=profile_result,
        )
        return result, payload
    finally:
        spec_path.unlink(missing_ok=True)


def _write_status(
    path: Path,
    result: Any,
    payload: Dict[str, Any],
    *,
    stage: str,
    requested_stages: frozenset[str],
    source_dir: Path | None = None,
) -> Dict[str, Any]:
    build = result.build_result
    profile = result.profile_result
    status = {
        "success": bool(result.success),
        "stage": stage,
        "requested_stages": sorted(requested_stages),
        "build_success": bool(build.success),
        "profile_success": bool(profile.success),
        "compile_log": build.log_path.name,
        "profile_log": profile.log_path.name,
        "profile_message": profile.message,
        "profile_cases": len(payload),
    }
    if source_dir is not None:
        status["source_dir"] = str(source_dir.resolve())
    _write_status_payload(path, status)
    return status


def _cleanup_success_logs(stage_dir: Path, status: Dict[str, Any]) -> None:
    if status.get("success") is not True or any(
        phase["status"] not in {"passed", "skipped"} for phase in status.get("stages", {}).values()
    ):
        return
    for name in _STAGE_LOG_NAMES:
        (stage_dir / name).unlink(missing_ok=True)
    if os.environ.get("ASCEND_ARENA_PROFILE_ARTIFACTS", "all") == "compact":
        profile = stage_dir / "profile"
        if profile.is_dir():
            shutil.rmtree(profile)
    for key in ("compile_log", "profile_log"):
        if status.get(key) in _STAGE_LOG_NAMES:
            status[key] = None
    _write_status_payload(stage_dir / "status.json", status)


def _correctness_stage_summaries(output_dir: Path) -> list[Dict[str, Any]]:
    summaries = []
    for stage in ("smoke", "complete", "profile"):
        path = output_dir / "results" / f"{stage}-correctness_result.json"
        if not path.is_file():
            continue
        payload = _load_json(path)
        raw_cases = payload.get("case_results", [])
        cases = raw_cases if isinstance(raw_cases, list) else []
        passed_cases = sum(
            isinstance(item, dict) and item.get("passed") is True for item in cases
        )
        summaries.append({
            "stage": stage,
            "success": payload.get("success") is True,
            "total_cases": len(cases),
            "passed_cases": passed_cases,
            "failed_cases": len(cases) - passed_cases,
        })
    return summaries


def _build_report(*, output_dir: Path, run_name: str, operator: str,
                  baseline: Dict[str, Any], candidate: Dict[str, Any],
                  baseline_status: Dict[str, Any], candidate_status: Dict[str, Any],
                  device_id: int, source_diff: Dict[str, Any] | None = None) -> None:
    source_diff = source_diff or {
        "changed_file_count": 0,
        "added_files": [],
        "deleted_files": [],
        "modified_files": [],
        "changed_files": [],
        "file_diffs": [],
        "unified_diff": "# No implementation source changes detected.\n",
    }
    source_diff_text = str(source_diff["unified_diff"])
    correctness_only = (
        baseline_status.get("skipped") is True
        and baseline_status.get("reason") == "performance_disabled"
    )
    evaluation_mode = "correctness_only" if correctness_only else "performance_comparison"
    correctness_stages = _correctness_stage_summaries(output_dir) if correctness_only else []
    baseline_times = _duration_map(baseline)
    candidate_times = _duration_map(candidate)
    comparison = []
    for case_id in sorted(set(baseline_times) | set(candidate_times)):
        before = baseline_times.get(case_id)
        after = candidate_times.get(case_id)
        candidate_passed = after is not None and after > 0
        bands = [_noise_band(baseline.get(case_id)), _noise_band(candidate.get(case_id))]
        noise_band = sum(bands) if all(band is not None for band in bands) else None
        scope = _timing_scope(baseline.get(case_id))
        protocol = _measurement_protocol(baseline.get(case_id))
        comparable = (scope == _timing_scope(candidate.get(case_id))
                      and protocol == _measurement_protocol(candidate.get(case_id)))
        before_call = _operator_call_duration(baseline.get(case_id))
        after_call = _operator_call_duration(candidate.get(case_id))
        speedup = before / after if before and after and comparable else None
        comparison.append({
            "case_id": case_id,
            "baseline_us": before,
            "candidate_us": after,
            "speedup": speedup,
            "timing_scope": scope if comparable else "mixed",
            "timing_comparable": comparable,
            "measurement_protocol": protocol if comparable else None,
            "baseline_call_us": before_call,
            "candidate_call_us": after_call,
            "call_speedup": before_call / after_call if before_call and after_call and comparable else None,
            "noise_band": noise_band,
            "significant": (speedup - 1 > noise_band) if speedup and noise_band is not None
                            and candidate_status.get("success") is True and baseline_status.get("success") is True else None,
            "candidate_passed": candidate_passed,
            "candidate_profiled": candidate_passed,
        })
    scopes = {item["timing_scope"] for item in comparison}
    timing_scope = next(iter(scopes)) if len(scopes) == 1 else "mixed"
    if correctness_only:
        timing_scope = "not_run"
    timing_label = {"kernel_total": "逐轮 kernel 总时间中位数（设备工作量）", "operator_call": "整次算子调用", "kernel": "kernel 明细",
                    "mixed": "不同计时口径（加速比不可用）", "not_run": "未执行性能采集"}[timing_scope]
    (output_dir / "comparison.json").write_text(
        json.dumps({
            "schema_version": 1,
            "noise_method": "sum_of_max_relative_sample_deviations; descriptive_not_confidence_interval",
            "benchmark": "ascend_arena",
            "evaluation_mode": evaluation_mode,
            "performance_status": (
                "not_run" if correctness_only
                else "completed" if candidate_status.get("success") is True
                else "failed"
            ),
            "metric": None if correctness_only else "dependency_baseline_speedup",
            "cases": comparison,
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Keep the CannBench report formats and report speedup directly.
    from ._version import FRAMEWORK_VERSION, TASKS_VERSION
    from .config import Config
    from .report.setup_info import collect_setup_info

    valid_speedups = [item["speedup"] for item in comparison if item["speedup"] is not None]
    if correctness_only:
        total_cases = sum(item["total_cases"] for item in correctness_stages)
        passed_count = sum(item["passed_cases"] for item in correctness_stages)
        pass_rate = passed_count / total_cases if total_cases else None
        compared_cases = None
        average_speedup = None
    else:
        total_cases = len(comparison)
        passed_count = sum(bool(item["candidate_passed"]) for item in comparison)
        pass_rate = passed_count / total_cases if total_cases else 0.0
        compared_cases = len(valid_speedups)
        average_speedup = statistics.mean(valid_speedups) if valid_speedups else None
    config = Config(
        tasks_root=str(output_dir), reports_dir=str(output_dir),
        bench_name="ascend_arena", device_id=device_id,
    )
    report = {
        "schema_version": 1,
        "framework_version": FRAMEWORK_VERSION,
        "tasks_version": TASKS_VERSION,
        "eval_code": run_name,
        "timestamp": datetime.now().isoformat(),
        "device": f"npu:{device_id}",
        "benchmark": "ascend_arena",
        "operator": operator,
        "evaluation_mode": evaluation_mode,
        "score_applicable": False,
        "candidate_status": candidate_status,
        "baseline_status": baseline_status,
        "summary": {
            "total_cases": total_cases,
            "passed_cases": passed_count,
            "failed_cases": total_cases - passed_count,
            "pass_rate": pass_rate,
            "compared_cases": compared_cases,
            "average_speedup": average_speedup,
        },
        "correctness_stages": correctness_stages,
        "setup_info": collect_setup_info(config),
        "cases": comparison,
        "source_diff": source_diff,
        "measurement": {
            "latency_scope": timing_scope,
            "protocol": comparison[0].get("measurement_protocol") if comparison else None,
            "secondary_scope": "operator_call_unprofiled" if timing_scope == "kernel_total" else None,
            "noise_band": max((item["noise_band"] for item in comparison), default=None)
                          if comparison and all(item["noise_band"] is not None for item in comparison) else None,
            "significant": all(item["significant"] for item in comparison)
                           if comparison and all(item["significant"] is not None for item in comparison) else None,
            "method": "all_cases_exceed_observed_noise; not_statistical_significance",
        },
    }
    (output_dir / f"{run_name}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = report["summary"]
    pass_rate_text = (
        f"{summary['pass_rate']:.2%}" if summary["pass_rate"] is not None else "N/A"
    )
    compared_cases_text = (
        str(summary["compared_cases"])
        if summary["compared_cases"] is not None else "未执行"
    )
    average_speedup_text = (
        f"{summary['average_speedup']:.4f}x"
        if summary["average_speedup"] is not None else ("未执行" if correctness_only else "N/A")
    )
    def latency_text(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "N/A"

    def speedup_text(value: float | None) -> str:
        return f"{value:.4f}x" if value is not None else "N/A"

    markdown_lines = [
        "# 算子评测报告", "", f"**评测代号**: {run_name}",
        f"**评测时间**: {report['timestamp']}", f"**设备**: npu:{device_id}",
        "", "## 概览", "", "| 指标 | 数值 |", "|---|---:|",
        f"| 算子 | {operator} |", f"| 评测模式 | {evaluation_mode} |",
        f"| 总用例数 | {summary['total_cases']} |",
        f"| 通过用例数 | {summary['passed_cases']} |",
        f"| 通过率 | {pass_rate_text} |",
        f"| 有效性能对比用例数 | {compared_cases_text} |",
        f"| 平均加速比 | {average_speedup_text} |",
    ]
    if not correctness_only:
        markdown_lines.append(f"| 时延口径 | {timing_label} |")
    if correctness_only:
        markdown_lines.extend([
            "", "## 正确性阶段详情", "",
            "| 阶段 | 状态 | 总用例数 | 通过 | 失败 |",
            "|---|---|---:|---:|---:|",
        ])
        for item in correctness_stages:
            status = "通过" if item["success"] else "失败"
            markdown_lines.append(
                f"| {item['stage']} | {status} | {item['total_cases']} | "
                f"{item['passed_cases']} | {item['failed_cases']} |"
            )
    else:
        markdown_lines.extend([
            "", "## 用例详情", "",
            "| 用例 ID | 状态 | 原实现 (us) | 优化实现 (us) | 加速比 |",
            "|---|---|---:|---:|---:|",
        ])
        for item in comparison:
            status = "通过" if item["candidate_passed"] else "失败"
            markdown_lines.append(
                f"| {item['case_id']} | {status} | {latency_text(item['baseline_us'])} | "
                f"{latency_text(item['candidate_us'])} | {speedup_text(item['speedup'])} |"
            )
    call_rows = [item for item in comparison if item["baseline_call_us"] or item["candidate_call_us"]]
    if timing_scope == "kernel_total" and call_rows:
        markdown_lines.extend(["", "## 完整调用延迟（独立测量，profiler 关闭）", "",
                               "| 用例 ID | 原实现 (us) | 优化实现 (us) | 调用加速比 |",
                               "|---|---:|---:|---:|"])
        for item in call_rows:
            markdown_lines.append(f"| {item['case_id']} | {latency_text(item['baseline_call_us'])} | "
                                  f"{latency_text(item['candidate_call_us'])} | {speedup_text(item['call_speedup'])} |")
    markdown_lines.extend([
        "", "## 优化前后源码 Diff", "",
        f"- 变更文件数：{source_diff['changed_file_count']}",
    ])
    if source_diff["changed_files"]:
        markdown_lines.extend(["", "### 变更文件", ""])
        markdown_lines.extend(
            f"- `{item['status']}` `{item['path']}`" for item in source_diff["changed_files"]
        )
        markdown_lines.extend(["", "### 完整 Diff", "", "```diff", source_diff_text.rstrip(), "```"])
    else:
        markdown_lines.extend(["", "未检测到候选实现相对 dependency 原始实现的源码变化。"])
    (output_dir / f"{run_name}.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(item["case_id"])),
            "通过" if item["candidate_passed"] else "失败",
            latency_text(item["baseline_us"]),
            latency_text(item["candidate_us"]),
            speedup_text(item["speedup"]),
        )
        for item in comparison
    )
    correctness_rows = "\n".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(str(item["stage"])),
            "通过" if item["success"] else "失败",
            item["total_cases"], item["passed_cases"], item["failed_cases"],
        )
        for item in correctness_stages
    )
    call_section = ""
    if timing_scope == "kernel_total" and call_rows:
        call_html = "".join("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            html.escape(item["case_id"]), latency_text(item["baseline_call_us"]),
            latency_text(item["candidate_call_us"]), speedup_text(item["call_speedup"])) for item in call_rows)
        call_section = ("<h2>完整调用延迟（独立测量，profiler 关闭）</h2>"
                        "<table><thead><tr><th>用例 ID</th><th>原实现 (us)</th><th>优化实现 (us)</th>"
                        "<th>调用加速比</th></tr></thead><tbody>" + call_html + "</tbody></table>")
    diff_rows = "\n".join(
        "<tr><td>{}</td><td>{}</td></tr>".format(
            html.escape(item["status"]), html.escape(item["path"])
        )
        for item in source_diff["changed_files"]
    ) or "<tr><td colspan=\"2\">未检测到源码变化</td></tr>"
    diff_section = f"""<h2>优化前后源码 Diff</h2>
<p>变更文件数：{source_diff['changed_file_count']}</p>
<table><thead><tr><th>状态</th><th>文件</th></tr></thead><tbody>{diff_rows}</tbody></table>
<details><summary>查看完整 Diff</summary><pre>{html.escape(source_diff_text)}</pre></details>"""
    if correctness_only:
        result_section = f"""<p>评测模式：{evaluation_mode}；正确性通过率：{pass_rate_text}；性能对比：未执行</p>
<table><thead><tr><th>阶段</th><th>状态</th><th>总用例数</th><th>通过</th><th>失败</th></tr></thead><tbody>{correctness_rows}</tbody></table>"""
    else:
        result_section = f"""<p>评测模式：{evaluation_mode}；时延口径：{timing_label}；通过率：{pass_rate_text}；平均加速比：{average_speedup_text}</p>
<table><thead><tr><th>用例 ID</th><th>状态</th><th>原实现 (us)</th><th>优化实现 (us)</th><th>加速比</th></tr></thead><tbody>{rows}</tbody></table>"""
    html_report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(run_name)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#222}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ddd;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}pre{{overflow:auto;padding:12px;background:#f6f8f7;border:1px solid #ddd}}</style></head>
<body><h1>算子评测报告</h1><p><b>评测代号：</b>{html.escape(run_name)}<br><b>设备：</b>npu:{device_id}</p>
<h2>{html.escape(operator)}</h2>
{result_section}
{call_section}{diff_section}</body></html>\n"""
    (output_dir / f"{run_name}.html").write_text(html_report, encoding="utf-8")

    from .optimization_trajectory import write_optimization_trajectory_report

    write_optimization_trajectory_report(output_dir.parent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare an Ascend Arena operator with its dependency implementation.")
    parser.add_argument("--stage-worker", default="", help=argparse.SUPPRESS)
    parser.add_argument("--source-dir")
    parser.add_argument("--task-dir")
    parser.add_argument("--operator")
    parser.add_argument("--reports-dir", default="")
    parser.add_argument("--baseline-dir", default="", help="reuse an initialized baseline; by default evaluate both implementations")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="evaluate only the original implementation, including correctness and performance",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--chip", default=os.environ.get("ASCEND_ARENA_CHIP", "ascend910b"))
    parser.add_argument("--dependency-root", default="")
    parser.add_argument("--ops-json", default="")
    parser.add_argument("--env-script", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--device-check", choices=("warn", "require-idle"), default="warn",
                        help="record NPU occupancy; require-idle rejects busy or unknown devices")
    parser.add_argument("--profile-backend", choices=("msprof", "msprof-op", "torch-profiler"), default="torch-profiler")
    parser.add_argument("--profile-artifacts", choices=("all", "compact"),
                        default=os.environ.get("ASCEND_ARENA_PROFILE_ARTIFACTS", "all"),
                        help="compact keeps result JSON but removes raw profiler data after success")
    parser.add_argument("--compile-jobs", type=int, default=16)
    parser.add_argument("--isolated-build", action="store_true")
    parser.add_argument(
        "--no-perf",
        action="store_true",
        help="运行 smoke、complete 和 profile 正确性，但跳过性能采集与 baseline",
    )
    parser.add_argument(
        "--no-clean-sandbox",
        action="store_true",
        help="保留评测 sandbox 及编译中间产物，便于调试",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage_worker:
        return _stage_worker_main(Path(args.stage_worker))
    os.environ["ASCEND_ARENA_PROFILE_ARTIFACTS"] = args.profile_artifacts
    from ascend_arena_e2e.timing import timed_run
    from .optimization_trajectory import write_optimization_trajectory_report

    reports_root = (Path(args.reports_dir).resolve() if args.reports_dir else
                    Path(__file__).resolve().parents[2] / "reports/ascend_arena")
    try:
        with timed_run(reports_root / "timing.jsonl", "evaluation", args.operator) as timing:
            timing["phase"] = "init" if args.init_only else "candidate" if args.baseline_dir else "comparison"
            args.timing = timing
            timing["returncode"] = _evaluate(args)
            return timing["returncode"]
    finally:
        # Include the closing timing event, including failed/preflight runs.
        write_optimization_trajectory_report(reports_root)


def _evaluate(args: argparse.Namespace) -> int:
    if args.repeat < 1 or args.warmup < 0:
        raise SystemExit("repeat must be positive and warmup must be non-negative")
    missing = [name for name in ("source-dir", "task-dir", "operator") if not getattr(args, name.replace("-", "_"))]
    if missing:
        raise SystemExit("missing required arguments: " + ", ".join(f"--{name}" for name in missing))
    supplied_source = Path(args.source_dir)
    if supplied_source.is_symlink() and not supplied_source.exists():
        raise SystemExit("benchmark sources are not initialized; run python3 scripts/setup_ascend_arena_dependencies.py")
    source_dir = _resolve_source_dir(supplied_source.resolve(), args.operator)
    task_root = Path(args.task_dir).resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"source-dir does not exist: {source_dir}")
    benchmark_op = _find_benchmark_op(task_root, args.operator)
    project_root = Path(__file__).resolve().parents[2]
    dependency_root = Path(args.dependency_root).resolve() if args.dependency_root else project_root / "src/ascend_arena_e2e/dependency"
    ops_json = Path(args.ops_json).resolve() if args.ops_json else project_root / "src/ascend_arena_e2e/ops.json"
    compile_repo, compile_family, compile_op_name = _load_compile_entry(ops_json, args.operator, dependency_root)
    if not dependency_root.is_dir():
        raise SystemExit(f"dependency-root does not exist: {dependency_root}")
    reports_root = Path(args.reports_dir).resolve() if args.reports_dir else project_root / "reports/ascend_arena"
    baseline_dir = Path(args.baseline_dir).resolve() if args.baseline_dir else reports_root / "init_results"
    if args.baseline_dir and not args.init_only and not args.no_perf:
        baseline_status = _load_json(baseline_dir / "status.json")
        if baseline_status.get("success") is not True or not (baseline_dir / "result.json").is_file():
            raise SystemExit(
                f"initialized baseline not found: {baseline_dir}; "
                "build the workspace again or run validate.sh init"
            )
    stamp = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_name = f"ascend_arena_{args.operator}_{stamp}"
    output_dir = reports_root / run_name
    compare_both = not args.init_only and not args.no_perf and not args.baseline_dir
    if compare_both:
        baseline_dir = output_dir / "baseline_results"
    args.timing["run_name"] = baseline_dir.name if args.init_only else run_name
    env_script = _resolve_env_script(args.env_script)
    try:
        _preflight_dependencies(
            project_root=project_root,
            dependency_root=dependency_root,
            compile_repo=compile_repo,
            env_script=env_script,
        )
    except RuntimeError as exc:
        raise SystemExit(f"dependency preflight failed: {exc}") from exc

    from ascend_arena_e2e.devices import device_snapshot
    occupancy = device_snapshot(args.device_id)
    evidence_dir = baseline_dir if args.init_only else output_dir
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "device_preflight.json").write_text(json.dumps(occupancy, indent=2) + "\n")
    print(f"[device] NPU {args.device_id}: {occupancy['state']}; snapshot: {evidence_dir / 'device_preflight.json'}", flush=True)
    if occupancy["state"] != "idle":
        if getattr(args, "device_check", "warn") == "require-idle":
            raise SystemExit("NPU is busy or occupancy is unknown; use an idle device before evaluation")
        print("[device] warning: shared or unknown device occupancy may contaminate timings", flush=True)

    sandbox = (reports_root / ".sandboxes" / f"{run_name}_{time.time_ns()}" if args.no_clean_sandbox
               else Path(tempfile.mkdtemp(prefix=f"cannbench_arena_{args.operator}_")))
    print(json.dumps({
        "run_dir": str(baseline_dir if args.init_only else output_dir),
        "sandbox": str(sandbox),
        "sandbox_retained": bool(args.no_clean_sandbox),
        "compile_repo": compile_repo,
        "compile_family": compile_family,
    }, ensure_ascii=False))
    sandbox_dependency = sandbox / "dependency"
    baseline_standard = sandbox / "standard" / "init" / args.operator
    candidate_standard = sandbox / "standard" / "candidate" / args.operator
    active_stage: Dict[str, Any] = {}
    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt_evaluation)
    try:
        sandbox.mkdir(parents=True, exist_ok=True)
        _copy_dependency_tree(dependency_root, sandbox_dependency, compile_repo=compile_repo)
        baseline_standard.mkdir(parents=True, exist_ok=True)
        # A prepared workspace stores the editable implementation next to the
        # copied benchmark assets.  Exclude that implementation and runtime
        # outputs when using the workspace root as the benchmark task root.
        _copy_benchmark_tree(benchmark_op, baseline_standard, operator=args.operator)
        # Benchmark assets own the original implementation; the compile
        # repository contains only the shared build skeleton.
        original_op = _resolve_source_dir(benchmark_op, args.operator)
        if not original_op.is_dir():
            raise FileNotFoundError(f"benchmark operator source directory not found: {original_op}")
        _overlay(original_op, baseline_standard, reject_protected=False)
        if args.init_only or compare_both:
            baseline_stages = frozenset({"smoke", "complete", "profile"})
            baseline_dir.mkdir(parents=True, exist_ok=True)
            active_stage = {
                "path": baseline_dir / "status.json",
                "stage": "baseline",
                "requested_stages": baseline_stages,
                "source_dir": original_op,
            }
            baseline_result, baseline_payload = _run_stage(
                standard_dir=baseline_standard, operator=args.operator, dependency_root=sandbox_dependency,
                run_root=sandbox / "runs" / "init", chip=args.chip, device_id=args.device_id,
                compile_repo=compile_repo, compile_family=compile_family, compile_op_name=compile_op_name,
                env_script=env_script, warmup=args.warmup, repeat=args.repeat,
                profile_backend=args.profile_backend, compile_jobs=args.compile_jobs,
                isolated_build=args.isolated_build, profile_performance=True,
                stages=baseline_stages, result_path=baseline_dir / "result.json",
            )
            (baseline_dir / "result.json").write_text(
                json.dumps(baseline_payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            baseline_status = _write_status(
                baseline_dir / "status.json", baseline_result, baseline_payload,
                stage="baseline", requested_stages=baseline_stages, source_dir=original_op,
            )
            active_stage = {}
            _cleanup_success_logs(baseline_dir, baseline_status)
            if args.init_only or not baseline_result.success:
                return 0 if baseline_result.success else 1

        candidate_standard.parent.mkdir(parents=True, exist_ok=True)
        _copy_tree(baseline_standard, candidate_standard, ignore_names={
            path.name for path in baseline_standard.iterdir() if path.name not in PROTECTED
        })
        _overlay(source_dir, candidate_standard)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results").mkdir(exist_ok=True)
        source_diff = _collect_source_diff(
            baseline_dir=original_op, candidate_dir=source_dir
        )

        candidate_stages = frozenset({"smoke", "complete", "profile"})
        active_stage = {
            "path": output_dir / "results/status.json",
            "stage": "candidate",
            "requested_stages": candidate_stages,
            "source_dir": source_dir,
        }
        candidate_result, candidate_payload = _run_stage(
            standard_dir=candidate_standard, operator=args.operator, dependency_root=sandbox_dependency,
            run_root=sandbox / "runs" / "candidate", chip=args.chip, device_id=args.device_id,
            compile_repo=compile_repo, compile_family=compile_family, compile_op_name=compile_op_name,
            env_script=env_script, warmup=args.warmup, repeat=args.repeat,
            profile_backend=args.profile_backend, compile_jobs=args.compile_jobs, isolated_build=args.isolated_build,
            profile_performance=not args.no_perf,
            stages=candidate_stages,
            result_path=output_dir / "results/result.json",
        )
        (output_dir / "results/result.json").write_text(json.dumps(candidate_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        candidate_status = _write_status(
            output_dir / "results/status.json", candidate_result, candidate_payload,
            stage="candidate", requested_stages=candidate_stages,
            source_dir=source_dir,
        )
        active_stage = {}
        _cleanup_success_logs(output_dir / "results", candidate_status)
        if args.no_perf:
            baseline_payload = {}
            baseline_status = {
                "success": True,
                "stage": "baseline",
                "skipped": True,
                "reason": "performance_disabled",
                "source_dir": str(original_op.resolve()),
            }
            _build_report(output_dir=output_dir, run_name=run_name, operator=args.operator,
                          baseline=baseline_payload, candidate=candidate_payload,
                          baseline_status=baseline_status, candidate_status=candidate_status,
                          device_id=args.device_id, source_diff=source_diff)
            return 0 if candidate_result.success else 1
        baseline_payload = _load_json(baseline_dir / "result.json")
        baseline_status = _load_json(baseline_dir / "status.json")
        _build_report(output_dir=output_dir, run_name=run_name, operator=args.operator,
                      baseline=baseline_payload, candidate=candidate_payload,
                      baseline_status=baseline_status, candidate_status=candidate_status,
                      device_id=args.device_id, source_diff=source_diff)
        return 0 if candidate_result.success else 1
    except _EvaluationInterrupted as exc:
        signal_name = signal.Signals(exc.signum).name
        returncode = 128 + exc.signum
        if active_stage:
            interrupted_status = {
                "success": False,
                "stage": active_stage["stage"],
                "requested_stages": sorted(active_stage["requested_stages"]),
                "build_success": False,
                "profile_success": False,
                "compile_log": "compile.log",
                "profile_log": "profile.log",
                "profile_cases": 0,
                "interrupted": True,
                "signal": signal_name,
                "returncode": returncode,
                "message": f"evaluation interrupted by {signal_name}",
                "profile_message": f"evaluation interrupted by {signal_name}",
                "source_dir": str(Path(active_stage["source_dir"]).resolve()),
            }
            _write_status_payload(active_stage["path"], interrupted_status)
            print(
                f"[{args.operator}] 评测中断：{signal_name}；状态={active_stage['path']}",
                flush=True,
            )
        return returncode
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if not args.no_clean_sandbox:
            shutil.rmtree(sandbox, ignore_errors=True)


if __name__ == "__main__":
    exit_code = main()
    # torch_npu may abort during interpreter teardown after all final
    # artifacts have been flushed.  Preserve both success and failure return
    # codes without running that teardown path.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
