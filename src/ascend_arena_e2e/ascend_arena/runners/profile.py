# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import os
import shlex
import shutil
import stat
import statistics
import math
import subprocess
import sys
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ascend_arena_e2e.ascend_arena.evaluator.policy import PrecisionPolicy, load_precision_policy
from ascend_arena_e2e.ascend_arena.runners.artifacts import resolve_benchmark_artifact_path
from ascend_arena_e2e.ascend_arena.runners.build import BuildResult
from ascend_arena_e2e.ascend_arena.runners.cases import load_test_cases, select_cases
from ascend_arena_e2e.ascend_arena.runners.correctness import (
    CorrectnessResult,
    CorrectnessRunner,
    PreparedCaseEvaluation,
    _load_torch_npu,
    compare_outputs_for_case,
)
from ascend_arena_e2e.ascend_arena.runners.metrics.mappings import (
    AVERAGED_FIELDS,
    CACHE_CLEAR_OP_SPECS,
    DROPPED_NULL_KEYS,
    GROUP_KEY_COLUMNS,
    GROUPED_KEYS,
    NON_PERF_COLUMNS,
    PRESERVED_FIELD_ALIASES,
    PRESERVED_FIELDS,
    TOP_LEVEL_BASIC_KEYS,
)
from ascend_arena_e2e.ascend_arena.runners.kernel_timing import HELPER_KERNELS, measurement_protocol, parse_kernel_totals, parse_kernel_batches
from ascend_arena_e2e.ascend_arena.utils.progress import emit_progress
from ascend_arena_e2e.ascend_arena.runners.opp_env import isolated_runtime_opp_env, prepare_runtime_opp_env
from ascend_arena_e2e.ascend_arena.runners._proc_utils import DEFAULT_PROFILE_TIMEOUT_SEC, run_shell_with_timeout


PROFILE_PRECISION_FAILURE_MARKER = "profile_precision_failed:"
PROFILE_PRECISION_FAILURE_FILENAME = "precision_failure.txt"
PROFILE_PRECISION_SUCCESS_FILENAME = "precision_verified.txt"


@dataclass
class ProfileResult:
    success: bool
    op_name: str
    log_path: Path
    output_dir: Path
    metrics: dict[str, list[dict[str, Any]]]
    message: str


@dataclass
class _PendingProfileParse:
    case_id: str
    output_dir: Path
    proc: subprocess.CompletedProcess[str]
    future: Future[list[dict[str, Any]]]


@contextmanager
def measure_operator_call(torch, samples, inner_iters=1):
    start, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
    start.record()
    yield
    end.record()
    end.synchronize()
    samples.append(float(start.elapsed_time(end)) * 1000 / inner_iters)


def write_operator_timings(output_dir, samples, repeat):
    values = samples[-repeat:]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("operator call timings must be finite and positive")
    center = statistics.median(values)
    metric = {"duration_scope": "operator_call", "task_duration_us": center,
              "duration_samples_us": values, "sample_count": len(values),
              "duration_statistic": "median",
              "noise_band": max(abs(value / center - 1) for value in values) if len(values) >= 3 else None}
    (Path(output_dir) / "operator_timing.json").write_text(json.dumps(metric) + "\n", encoding="utf-8")


class ProfileRunner:

    def __init__(
        self,
        *,
        env_script: Path | None = None,
        warmup: int = 10,
        repeat: int = 30,
        inner_iters: int = 1,
        profiler: str = "msprof",
        case_filter: str = "all",
        profile_batch: bool = True,
    ) -> None:
        self.env_script = env_script
        self.warmup = warmup
        self.repeat = repeat
        self.inner_iters = max(1, int(inner_iters))
        self.profiler = profiler
        self.case_filter = case_filter
        self.profile_batch = profile_batch

    def run(self, *, standard_dir: Path, build_result: BuildResult, correctness_result: CorrectnessResult, device_id: int = 0) -> ProfileResult:
        log_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / "profile.log")
        output_dir = build_result.work_dir.parent / "profile"
        result_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / "result.json")
        self._prepare_output_dir(output_dir)
        self._remove_result_json(result_path)
        if not build_result.success or build_result.extension_path is None:
            log_path.write_text("build_failed\n", encoding="utf-8")
            emit_progress("profile_finished", op_name=build_result.op_name, success=False, reason="build_failed")
            return ProfileResult(False, build_result.op_name, log_path, output_dir, {}, "build_failed")
        if not correctness_result.success:
            log_path.write_text("correctness_failed\n", encoding="utf-8")
            emit_progress("profile_finished", op_name=build_result.op_name, success=False, reason="correctness_failed")
            return ProfileResult(False, build_result.op_name, log_path, output_dir, {}, "correctness_failed")

        metrics_by_case: dict[str, list[dict[str, Any]]] = {}
        log_sections: list[str] = []
        cases = self._load_profile_cases(build_result.standard_op_root / "cases")
        emit_progress(
            "profile_start",
            op_name=build_result.op_name,
            case_count=len(cases),
        )
        success = True
        pending: list[_PendingProfileParse] = []
        with ThreadPoolExecutor(max_workers=self._profile_parse_worker_count(len(cases))) as executor:
            with self._torch_profiler_runtime_context(
                standard_dir=build_result.standard_op_root,
                build_result=build_result,
                device_id=device_id,
            ) as torch_profiler_runtime:
                collection_proc = None
                if self._profiler_mode() == "torch-profiler":
                    try:
                        self._execute_torch_profiler_cases(
                            output_dir=output_dir, cases=cases, runtime=torch_profiler_runtime,
                            batch=self.profile_batch,
                        )
                        collection_proc = subprocess.CompletedProcess("torch-profiler:collection", 0, "", "")
                    except Exception:
                        collection_proc = subprocess.CompletedProcess("torch-profiler:collection", 1, "", traceback.format_exc())
                for case in cases:
                    case_id = str(case["case_id"])
                    case_output_dir = output_dir / case_id
                    case_output_dir.mkdir(parents=True, exist_ok=True)
                    if self._profiler_mode() != "torch-profiler":
                        emit_progress("profile_case_start", op_name=build_result.op_name, case_id=case_id)
                    if self._profiler_mode() == "torch-profiler":
                        proc = collection_proc
                    else:
                        command = self._build_command(
                            standard_dir=build_result.standard_op_root,
                            build_result=build_result,
                            output_dir=case_output_dir,
                            device_id=device_id,
                            case=case,
                        )
                        proc = run_shell_with_timeout(command, timeout=DEFAULT_PROFILE_TIMEOUT_SEC)
                    pending.append(
                        _PendingProfileParse(
                            case_id=case_id,
                            output_dir=case_output_dir,
                            proc=proc,
                            future=executor.submit(self._extract_metrics, case_output_dir),
                        )
                    )
                    success = self._drain_completed_profile_parses(
                        build_result=build_result,
                        pending=pending,
                        metrics_by_case=metrics_by_case,
                        log_sections=log_sections,
                        success=success,
                        wait=False,
                    )
                success = self._drain_completed_profile_parses(
                    build_result=build_result,
                    pending=pending,
                    metrics_by_case=metrics_by_case,
                    log_sections=log_sections,
                    success=success,
                    wait=True,
                )

        log_path.write_text("\n\n".join(log_sections) + ("\n" if log_sections else ""), encoding="utf-8")
        self._write_result_json(result_path, metrics_by_case)
        emit_progress(
            "profile_finished",
            op_name=build_result.op_name,
            success=success,
            case_count=len(cases),
            profiled_case_count=len(metrics_by_case),
            log_path=log_path,
        )
        return ProfileResult(success, build_result.op_name, log_path, output_dir, metrics_by_case, f"profiled_cases={len(metrics_by_case)}")

    def _profile_parse_worker_count(self, case_count: int) -> int:
        if case_count <= 1:
            return 1
        cpu_count = os.cpu_count() or 1
        return max(1, min(case_count, cpu_count, 4))

    def _drain_completed_profile_parses(
        self,
        *,
        build_result: BuildResult,
        pending: list[_PendingProfileParse],
        metrics_by_case: dict[str, list[dict[str, Any]]],
        log_sections: list[str],
        success: bool,
        wait: bool,
    ) -> bool:
        while pending:
            next_item = pending[0]
            if not wait and not next_item.future.done():
                break
            try:
                case_metrics = next_item.future.result()
            except Exception:
                case_metrics = []
                next_item.proc.stderr = (next_item.proc.stderr or "") + "\n" + traceback.format_exc()
            # 中文说明：
            # msprof/torch-profiler 在 teardown 阶段偶尔返回非零。只有精度复核
            # 已完成且采样数量完整时才保留该兼容语义，退出码仍写入日志供审计。
            process_output = f"{next_item.proc.stdout or ''}\n{next_item.proc.stderr or ''}"
            precision_failed = (
                PROFILE_PRECISION_FAILURE_MARKER in process_output
                or (next_item.output_dir / PROFILE_PRECISION_FAILURE_FILENAME).is_file()
            )
            precision_verified = (next_item.output_dir / PROFILE_PRECISION_SUCCESS_FILENAME).is_file()
            metrics_complete = self._has_expected_profile_samples(case_metrics)
            case_success = bool(case_metrics) and metrics_complete and precision_verified and not precision_failed
            if case_success:
                metrics_by_case[next_item.case_id] = case_metrics
            nonzero_exit_with_metrics = next_item.proc.returncode != 0 and bool(case_metrics)
            log_sections.append(
                "\n".join(
                    [
                        (
                            f"case_id={next_item.case_id} success={case_success} "
                            f"exit_code={next_item.proc.returncode} "
                            f"nonzero_exit_with_metrics={nonzero_exit_with_metrics} "
                            f"precision_failed={precision_failed} "
                            f"precision_verified={precision_verified} "
                            f"metrics_complete={metrics_complete} "
                            f"output_dir={next_item.output_dir}"
                        ),
                        next_item.proc.stdout or "",
                        next_item.proc.stderr or "",
                    ]
                ).strip()
            )
            success = success and case_success
            emit_progress(
                "profile_case_finished",
                op_name=build_result.op_name,
                case_id=next_item.case_id,
                success=case_success,
                exit_code=next_item.proc.returncode,
                nonzero_exit_with_metrics=nonzero_exit_with_metrics,
                durations_us=[m["task_duration_us"] for m in case_metrics if m.get("task_duration_us") is not None],
            )
            pending.pop(0)
        return success

    def _has_expected_profile_samples(self, metrics: list[dict[str, Any]]) -> bool:
        if self._profiler_mode() == "torch-profiler":
            return all(any(metric.get("duration_scope") == scope and metric.get("sample_count") == self.repeat
                           for metric in metrics) for scope in ("kernel_total", "operator_call"))
        expected = self.repeat if self._profiler_mode() == "msprof-op" else self.repeat * self.inner_iters
        for metric in metrics:
            try:
                if int(metric.get("sample_count", 0)) >= expected:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    def _build_command(
        self,
        *,
        standard_dir: Path,
        build_result: BuildResult,
        output_dir: Path,
        device_id: int,
        case: dict[str, Any],
    ) -> str:
        script = self._write_profile_script(
            standard_dir=standard_dir,
            build_result=build_result,
            output_dir=output_dir,
            device_id=device_id,
            case=case,
        )
        runtime_env = prepare_runtime_opp_env(
            op_run_root=build_result.work_dir.parent,
            custom_opp_root=build_result.custom_opp_root,
            env=os.environ,
        )
        command_parts = []
        if self.env_script is not None:
            command_parts.append(f"source {shlex.quote(str(self.env_script))}")
        custom_opp_path = runtime_env.get("ASCEND_CUSTOM_OPP_PATH")
        if custom_opp_path:
            command_parts.append(f"export ASCEND_CUSTOM_OPP_PATH={shlex.quote(custom_opp_path)}")
        runtime_opp_path = runtime_env.get("ASCEND_OPP_PATH")
        if runtime_opp_path:
            command_parts.append(f"export ASCEND_OPP_PATH={shlex.quote(runtime_opp_path)}")
        application = shlex.join([sys.executable, str(script)])
        if self._profiler_mode() == "torch-profiler":
            command_parts.append(application)
        elif self._profiler_mode() == "msprof-op":
            profile_command = (
                f"{self._profiler_command()} "
                f"--warm-up={self.warmup} "
                f"--launch-count={self.repeat} "
                f"--output={shlex.quote(str(output_dir))} "
                f"--application={shlex.quote(application)}"
            )
            command_parts.append(profile_command)
        else:
            profile_command = (
                f"{self._profiler_command()} "
                f"--output={shlex.quote(str(output_dir))} "
                f"--application={shlex.quote(application)}"
            )
            command_parts.append(profile_command)
        return " && ".join(command_parts)

    def _profiler_mode(self) -> str:
        parts = shlex.split(self.profiler)
        if not parts:
            raise ValueError("profiler command must not be empty")
        if parts[0] == "torch-profiler":
            return "torch-profiler"
        if parts[0] == "msprof-op":
            return "msprof-op"
        if Path(parts[0]).name == "msprof" and len(parts) > 1 and parts[1] == "op":
            return "msprof-op"
        return "msprof"

    @contextmanager
    def _torch_profiler_runtime_context(
        self,
        *,
        standard_dir: Path,
        build_result: BuildResult,
        device_id: int,
    ):
        if self._profiler_mode() != "torch-profiler":
            yield None
            return

        with self._sourced_env_context():
            with isolated_runtime_opp_env(
                op_run_root=build_result.work_dir.parent,
                custom_opp_root=build_result.custom_opp_root,
            ):
                with self._temporary_sys_path(build_result.build_dir):
                    torch = importlib.import_module("torch")
                    self._maybe_set_torch_threads(torch)
                    torch_npu = _load_torch_npu()
                    torch.npu.set_device(device_id)
                    correctness_runner = CorrectnessRunner(use_subprocess=False)
                    yield {
                        "torch": torch,
                        "torch_npu": torch_npu,
                        "binding_module": self._load_module("profile_binding_module", standard_dir / "binding" / "module.py"),
                        "reference_module": self._load_module("profile_reference_module", standard_dir / "reference.py"),
                        "inputs_module": self._load_module("profile_prepare_inputs_module", standard_dir / "cases" / "prepare_inputs.py"),
                        "prepare_case_evaluation": correctness_runner.prepare_case_evaluation,
                        "policy": load_precision_policy(standard_dir / "cases" / "proto.yaml"),
                        "device": torch.device(f"npu:{device_id}"),
                    }

    @contextmanager
    def _sourced_env_context(self):
        if self.env_script is None:
            yield
            return
        original_env = dict(os.environ)
        sourced_env = self._capture_env_from_script(self.env_script, original_env)
        os.environ.clear()
        os.environ.update(sourced_env)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(original_env)

    def _capture_env_from_script(self, env_script: Path, base_env: dict[str, str]) -> dict[str, str]:
        command = (
            f"source {shlex.quote(str(env_script))} >/dev/null 2>&1 && "
            f"{shlex.quote(sys.executable)} -c "
            f"\"import json, os; print(json.dumps(dict(os.environ)))\""
        )
        proc = subprocess.run(
            ["/bin/bash", "-lc", command],
            check=False,
            text=True,
            capture_output=True,
            env=base_env,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"failed to source env script {env_script}: {proc.stderr.strip() or proc.stdout.strip() or proc.returncode}"
            )
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"failed to decode environment exported by {env_script}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid environment payload exported by {env_script}")
        return {str(key): str(value) for key, value in payload.items()}

    def _maybe_set_torch_threads(self, torch: Any) -> None:
        threads = os.environ.get("BENCHMARK_TORCH_NUM_THREADS")
        if not threads:
            return
        try:
            torch.set_num_threads(max(1, int(threads)))
        except Exception:
            pass

    @contextmanager
    def _temporary_sys_path(self, path: Path):
        path_str = str(path)
        added = False
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            added = True
        try:
            yield
        finally:
            if added and path_str in sys.path:
                sys.path.remove(path_str)

    def _load_module(self, module_name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        standard_dir = str(path.parent.parent)
        added_standard_dir = False
        if standard_dir not in sys.path:
            sys.path.insert(0, standard_dir)
            added_standard_dir = True
        try:
            spec.loader.exec_module(module)
        finally:
            if added_standard_dir and standard_dir in sys.path:
                sys.path.remove(standard_dir)
        return module

    def _run_torch_profiler_case_in_process(
        self,
        *,
        standard_dir: Path,
        build_result: BuildResult,
        output_dir: Path,
        device_id: int,
        case: dict[str, Any],
        runtime: dict[str, Any] | None,
    ) -> subprocess.CompletedProcess[str]:
        del standard_dir, build_result, device_id
        if runtime is None:
            raise RuntimeError("torch-profiler runtime context was not initialized")
        try:
            self._execute_torch_profiler_case(output_dir=output_dir, case=case, runtime=runtime)
            if not self._has_expected_profile_samples(self._extract_metrics(output_dir)):
                emit_progress(
                    "profile_case_retry",
                    case_id=str(case.get("case_id", "")),
                    reason="missing_profile_samples",
                )
                self._prepare_output_dir(output_dir)
                self._execute_torch_profiler_case(output_dir=output_dir, case=case, runtime=runtime)
            return subprocess.CompletedProcess(
                args=f"torch-profiler:{case.get('case_id')}",
                returncode=0,
                stdout="",
                stderr="",
            )
        except Exception:
            return subprocess.CompletedProcess(
                args=f"torch-profiler:{case.get('case_id')}",
                returncode=1,
                stdout="",
                stderr=traceback.format_exc(),
            )

    def _execute_torch_profiler_case(
        self,
        *,
        output_dir: Path,
        case: dict[str, Any],
        runtime: dict[str, Any],
    ) -> None:
        self._execute_torch_profiler_cases(output_dir=output_dir.parent, cases=[case], runtime=runtime,
                                           batch=False, case_output_dirs=[output_dir])

    def _execute_torch_profiler_cases(self, *, output_dir, cases, runtime, batch=True, case_output_dirs=None):
        from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean
        from cann_bench_utils.build_info import check_build_compatibility
        from kernel_eval.eval.input_pool import InputPool, InputPoolConfig

        if not cases or len({str(case["case_id"]) for case in cases}) != len(cases):
            raise ValueError("profiler collection requires unique nonempty cases")
        torch, torch_npu = runtime["torch"], runtime["torch_npu"]
        device = runtime["device"]
        binding = runtime["binding_module"]
        helper_source = Path(__file__).resolve().parents[3] / "cann_bench_utils"
        helper_build = check_build_compatibility(
            source_root=helper_source if (helper_source / "CMakeLists.txt").is_file() else None)
        runtime_config = {name: os.environ.get(name) for name in (
            "PYTORCH_NPU_ALLOC_CONF", "PYTORCH_NO_NPU_MEMORY_CACHING", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING",
        )}
        pool_config = InputPoolConfig(max_pool_size=self.warmup + self.repeat)
        if "conditioning_tensors" not in runtime:
            runtime["conditioning_tensors"] = (
                torch.rand((10240, 10240), dtype=torch.float16, device=device),
                torch.rand((10240, 10240), dtype=torch.float16, device=device),
                torch.rand((96, 1024, 1024), dtype=torch.float16, device=device),
            )
        mm1, mm2, cache_input = runtime["conditioning_tensors"]

        def prepare(case):
            return runtime["prepare_case_evaluation"](
                row=case, reference_module=runtime["reference_module"], inputs_module=runtime["inputs_module"],
                device=device, seed=0, policy=runtime["policy"])

        def clear_cache():
            cann_bench_cache_clean(cache_input)
            torch.npu.synchronize(device)

        # Both session modes use the same allocation order and keep all pools alive.
        states = []
        directories = case_output_dirs or [output_dir / str(case["case_id"]) for case in cases]
        for case, directory in zip(cases, directories):
            directory.mkdir(parents=True, exist_ok=True)
            prepared = prepare(case)
            states.append(dict(case=case, directory=directory, prepared=prepared,
                               model=binding.ModelNew(*prepared.init_inputs_device).to(device),
                               call_model=binding.ModelNew(*prepared.init_inputs_device).to(device),
                               call_pool=InputPool(prepared.inputs_device, self.warmup + self.repeat, pool_config),
                               profile_pool=InputPool(prepared.inputs_device, self.warmup + self.repeat, pool_config)))
        torch.npu.synchronize(device)
        for state in states:
            cann_bench_warmup(mm1, mm2)
            torch.npu.synchronize(device)
            samples = []
            for _ in range(self.warmup + self.repeat):
                clear_cache()
                inputs = state["call_pool"].get_next()
                with measure_operator_call(torch, samples, self.inner_iters):
                    for _ in range(self.inner_iters):
                        state["call_actual"] = state["call_model"](*inputs)
            write_operator_timings(state["directory"], samples, self.repeat)
            pool = state["profile_pool"]
            protocol = measurement_protocol(self.warmup, self.repeat, self.inner_iters, helper_build,
                {"policy": "clone_rotation", "size": pool.size(), "max_pool_size": pool.config.max_pool_size,
                 "max_memory_mb": pool.config.max_memory_mb}, runtime_config, case=state["case"])
            protocol.update(version="arena_kernel_total_v6", profiler_session={
                "mode": "batch" if batch else "per_case", "case_ids": [str(c["case_id"]) for c in cases],
                "warmup_recorded": True, "input_lifetime": "all_cases_retained_through_collection"})
            timing_path = state["directory"] / "operator_timing.json"
            timing = json.loads(timing_path.read_text())
            timing["measurement_protocol"] = protocol
            timing_path.write_text(json.dumps(timing) + "\n")

        api = torch_npu.profiler
        config = api._ExperimentalConfig(profiler_level=api.ProfilerLevel.Level0,
            aic_metrics=api.AiCMetrics.AiCoreNone, export_type=api.ExportType.Text, data_simplification=False)
        groups = [states] if batch else [[state] for state in states]
        for group in groups:
            trace_dir = output_dir / "batch_session" if batch else group[0]["directory"] / "profiler_session"
            with api.profile(activities=[api.ProfilerActivity.NPU],
                schedule=api.schedule(wait=0, warmup=0, active=len(group) * (self.warmup + self.repeat), repeat=1),
                on_trace_ready=api.tensorboard_trace_handler(str(trace_dir)), record_shapes=False,
                profile_memory=False, with_stack=False, experimental_config=config) as prof:
                for state in group:
                    emit_progress("profile_case_start", case_id=str(state["case"]["case_id"]))
                    cann_bench_warmup(mm1, mm2)
                    torch.npu.synchronize(device)
                    for _ in range(self.warmup + self.repeat):
                        clear_cache()
                        inputs = state["profile_pool"].get_next()
                        for _ in range(self.inner_iters):
                            state["actual"] = state["model"](*inputs)
                        torch.npu.synchronize(device)
                        prof.step()
            torch.npu.synchronize(device)
            paths = list(trace_dir.rglob("kernel_details.csv"))
            if len(paths) != 1:
                raise ValueError(f"profiler session requires one kernel_details.csv, found {len(paths)}")
            ids = [str(state["case"]["case_id"]) for state in group]
            for state in group:
                manifest = {"csv": os.path.relpath(paths[0], state["directory"]), "case_ids": ids,
                            "case_id": str(state["case"]["case_id"]), "mode": "batch" if batch else "per_case"}
                (state["directory"] / "profile_session.json").write_text(json.dumps(manifest) + "\n")

        # Fresh correctness inputs avoid repeated mutations from in-place operators.
        for state in states:
            fresh = prepare(state["case"])
            model = binding.ModelNew(*fresh.init_inputs_device).to(device)
            pool = InputPool(fresh.inputs_device, 1, pool_config)
            actual = model(*pool.get_next())
            torch.npu.synchronize(device)
            self._verify_profile_output(state["directory"], state["case"], fresh, actual, runtime["policy"])

    def _verify_profile_output(
        self,
        output_dir: Path,
        case: dict[str, Any],
        prepared: PreparedCaseEvaluation,
        actual: Any,
        policy: PrecisionPolicy,
    ) -> None:
        compare = compare_outputs_for_case(
            case,
            prepared.golden,
            actual,
            native=prepared.native,
            policy=policy,
            max_abs_error=0.0,
            max_rel_error=0.0,
        )
        if not compare.passed:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / PROFILE_PRECISION_FAILURE_FILENAME).write_text(compare.message + "\n", encoding="utf-8")
            raise RuntimeError(f"{PROFILE_PRECISION_FAILURE_MARKER}{compare.message}")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / PROFILE_PRECISION_SUCCESS_FILENAME).write_text("ok\n", encoding="utf-8")

    def _profiler_command(self) -> str:
        parts = shlex.split(self.profiler)
        if not parts:
            raise ValueError("profiler command must not be empty")
        if parts[0] == "msprof-op":
            parts = ["msprof", "op", *parts[1:]]
        return " ".join(shlex.quote(part) for part in parts)

    def _write_profile_script(
        self,
        *,
        standard_dir: Path,
        build_result: BuildResult,
        output_dir: Path,
        device_id: int,
        case: dict[str, Any],
    ) -> Path:
        script = output_dir / "run_profile_case.py"
        if self._profiler_mode() == "torch-profiler":
            self._write_torch_profiler_script(
                script=script,
                standard_dir=standard_dir,
                build_result=build_result,
                output_dir=output_dir,
                device_id=device_id,
                case=case,
            )
        else:
            self._write_msprof_script(
                script=script,
                standard_dir=standard_dir,
                build_result=build_result,
                output_dir=output_dir,
                device_id=device_id,
                case=case,
            )
        script.chmod(0o700)
        return script

    def _write_msprof_script(
        self,
        *,
        script: Path,
        standard_dir: Path,
        build_result: BuildResult,
        output_dir: Path,
        device_id: int,
        case: dict[str, Any],
    ) -> None:
        # msprof-op can replay kernels, so its application events are not call latency measurements.
        call_block = "actual = model(*inputs_device)\ntorch_npu.npu.synchronize()\n"
        timing_write = ""
        if self._profiler_mode() == "msprof":
            call_block = (
                f"for _ in range({self.warmup + self.repeat}):\n"
                "    _clear_npu_cache()\n"
                f"    with measure_operator_call(torch, call_samples, {self.inner_iters}):\n"
                f"        for _ in range({self.inner_iters}):\n"
                "            actual = model(*inputs_device)\n"
                "    torch_npu.npu.synchronize()\n"
            )
            timing_write = f"write_operator_timings({str(output_dir)!r}, call_samples, {self.repeat})\n"
        cache_clear_block = (
            "# Cache clearing: boost NPU frequency + clear L2 cache\n"
            "_mm1 = torch.rand((10240, 10240), dtype=torch.float16, device=device)\n"
            "_mm2 = torch.rand((10240, 10240), dtype=torch.float16, device=device)\n"
            "_reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16, device=device)\n"
            "\n"
            "def _clear_npu_cache():\n"
            "    torch.matmul(_mm1, _mm2)\n"
            "    torch_npu.npu.synchronize(device)\n"
            "    torch.max(_reduce_input)\n"
            "    torch_npu.npu.synchronize(device)\n"
            "\n"
            "_clear_npu_cache()\n"
        )
        script.write_text(
            "import importlib.util, json, sys, time, torch, torch_npu\n"
            "from pathlib import Path\n"
            "from ascend_arena_e2e.ascend_arena.runners.profile import measure_operator_call, write_operator_timings\n"
            "from ascend_arena_e2e.ascend_arena.evaluator.policy import load_precision_policy\n"
            "from ascend_arena_e2e.ascend_arena.runners.correctness import CorrectnessRunner, compare_outputs_for_case\n"
            f"sys.path.insert(0, {str(build_result.build_dir)!r})\n"
            "def load(name, path):\n"
            "    spec = importlib.util.spec_from_file_location(name, path)\n"
            "    mod = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(mod)\n"
            "    return mod\n"
            f"mod = load('mod', {str(standard_dir / 'binding' / 'module.py')!r})\n"
            f"ref = load('ref', {str(standard_dir / 'reference.py')!r})\n"
            f"inp = load('inp', {str(standard_dir / 'cases' / 'prepare_inputs.py')!r})\n"
            f"case = json.loads({json.dumps(json.dumps(case, ensure_ascii=False))})\n"
            f"torch.npu.set_device({device_id})\n"
            f"device = torch.device('npu:{device_id}')\n"
            "runner = CorrectnessRunner(use_subprocess=False)\n"
            f"policy = load_precision_policy({str(standard_dir / 'cases' / 'proto.yaml')!r})\n"
            "prepared = runner.prepare_case_evaluation(row=case, reference_module=ref, inputs_module=inp, device=device, seed=0, policy=policy)\n"
            "inputs_device = prepared.inputs_device\n"
            "model = mod.ModelNew(*prepared.init_inputs_device).to(device)\n"
            "actual = object()\n"
            "call_samples = []\n"
            f"{cache_clear_block}"
            f"{call_block}"
            f"{timing_write}"
            "torch_npu.npu.synchronize(device)\n"
            "compare = compare_outputs_for_case(case, prepared.golden, actual, native=prepared.native, policy=policy, max_abs_error=0.0, max_rel_error=0.0)\n"
            f"if not compare.passed:\n    Path({str(output_dir / PROFILE_PRECISION_FAILURE_FILENAME)!r}).write_text(compare.message + '\\n', encoding='utf-8')\n    raise RuntimeError({PROFILE_PRECISION_FAILURE_MARKER!r} + compare.message)\n"
            f"Path({str(output_dir / PROFILE_PRECISION_SUCCESS_FILENAME)!r}).write_text('ok\\n', encoding='utf-8')\n"
            "time.sleep(1.0)\n",
            encoding="utf-8",
        )

    def _write_torch_profiler_script(
        self,
        *,
        script: Path,
        standard_dir: Path,
        build_result: BuildResult,
        output_dir: Path,
        device_id: int,
        case: dict[str, Any],
    ) -> None:
        script.write_text(
            "import json\nfrom pathlib import Path\n"
            "from types import SimpleNamespace\n"
            "from ascend_arena_e2e.ascend_arena.runners.profile import ProfileRunner\n"
            f"runner = ProfileRunner(env_script={('Path(' + repr(str(self.env_script)) + ')') if self.env_script else 'None'}, "
            f"warmup={self.warmup}, repeat={self.repeat}, inner_iters={self.inner_iters}, profiler='torch-profiler')\n"
            f"build = SimpleNamespace(work_dir=Path({str(getattr(build_result, 'work_dir', build_result.build_dir / 'work'))!r}), "
            f"build_dir=Path({str(build_result.build_dir)!r}), custom_opp_root="
            f"{('Path(' + repr(str(build_result.custom_opp_root)) + ')') if getattr(build_result, 'custom_opp_root', None) else 'None'})\n"
            f"case = json.loads({json.dumps(json.dumps(case, ensure_ascii=False))})\n"
            f"with runner._torch_profiler_runtime_context(standard_dir=Path({str(standard_dir)!r}), "
            f"build_result=build, device_id={device_id}) as runtime:\n"
            f"    runner._execute_torch_profiler_case(output_dir=Path({str(output_dir)!r}), case=case, runtime=runtime)\n",
            encoding="utf-8",
        )

    def _load_profile_cases(self, cases_dir: Path) -> list[dict[str, Any]]:
        from ascend_arena_e2e.ascend_arena.runners.cases import resolve_cases_path
        path = resolve_cases_path(cases_dir, self.case_filter)
        if not path.exists():
            raise ValueError(f"no cases file found in: {cases_dir}")
        cases = select_cases(load_test_cases(path), self.case_filter)
        if not cases:
            raise ValueError(f"no cases matched filter '{self.case_filter}' in: {cases_dir}")
        return cases

    def _prepare_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in [output_dir, *output_dir.parents]:
            mode = stat.S_IMODE(path.stat().st_mode)
            if mode & stat.S_IWGRP:
                try:
                    path.chmod(mode & ~stat.S_IWGRP)
                except PermissionError:
                    # A report directory may live below a shared system
                    # directory such as /tmp. Do not change its permissions.
                    continue

    def _remove_result_json(self, path: Path) -> None:
        if path.exists():
            path.unlink()

    def _is_cache_clear_op(
        self,
        op_name: str | None,
        op_type: str | None,
        input_shapes: str | None = None,
        input_data_types: str | None = None,
    ) -> bool:
        op_name = op_name or ""
        op_type = op_type or ""
        if any(token in op_type or token in op_name for token in HELPER_KERNELS):
            return True
        shapes_norm = (input_shapes or "").strip().strip('"')
        dtypes_norm = (input_data_types or "").strip().strip('"')
        for spec in CACHE_CLEAR_OP_SPECS:
            spec_type = spec.get("op_type") or ""
            spec_prefix = spec.get("op_name_prefix") or ""
            spec_shapes = spec.get("input_shapes")
            spec_dtypes = spec.get("input_data_types")
            if spec_type and op_type and op_type != spec_type:
                continue
            if spec_prefix and op_name and not op_name.startswith(spec_prefix):
                continue
            if spec_shapes is not None and shapes_norm != spec_shapes:
                continue
            if spec_dtypes is not None and dtypes_norm != spec_dtypes:
                continue
            return True
        return False

    def _extract_metrics(self, output_dir: Path) -> list[dict[str, Any]]:
        timing = output_dir / "operator_timing.json"
        call_metrics = [json.loads(timing.read_text())] if timing.is_file() else []
        device_info = self._extract_device_info(output_dir)
        totals = []
        if self._profiler_mode() == "torch-profiler":
            if not call_metrics or "helper_build" not in call_metrics[0].get("measurement_protocol", {}):
                return []
            protocol = call_metrics[0]["measurement_protocol"]
            options = dict(warmup=self.warmup, repeat=self.repeat, inner_iters=self.inner_iters,
                           helper_build=protocol["helper_build"], input_pool=protocol.get("input_pool"),
                           runtime_config=protocol.get("runtime_config"))
            session = output_dir / "profile_session.json"
            if session.is_file():
                manifest = json.loads(session.read_text())
                total = parse_kernel_batches(output_dir / manifest["csv"], case_ids=manifest["case_ids"],
                    session_mode=manifest["mode"], **options)[manifest["case_id"]]
                total["measurement_protocol"] = protocol
                total["raw_session"] = manifest
            else:
                csv_paths = list(output_dir.rglob("kernel_details.csv"))
                if len(csv_paths) != 1:
                    return []
                total = parse_kernel_totals(csv_paths[0], **options)
            (output_dir / "kernel_total_timing.json").write_text(json.dumps(total) + "\n")
            totals = [total]
        for path in sorted(output_dir.rglob("op_summary_*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    continue
                rows = list(reader)
                if not rows:
                    continue
                return self._aggregate_op_summary(rows, device_info) + totals + call_metrics
        opprof_metrics = self._extract_msprof_op_metrics(output_dir, device_info)
        if opprof_metrics:
            return opprof_metrics + totals + call_metrics
        return totals + call_metrics

    def _extract_device_info(self, output_dir: Path) -> dict[str, Any]:
        info: dict[str, Any] = {}
        for path in sorted(output_dir.rglob("info.json.*")):
            if path.suffix == ".done" or path.name.endswith(".done"):
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            entries = payload.get("DeviceInfo") or []
            if not entries:
                continue
            entry = entries[0]
            info["device_id"] = self._parse_numeric_or_keep(entry.get("id"))
            info["aic_frequency_mhz"] = self._parse_numeric_or_keep(entry.get("aic_frequency"))
            info["aiv_frequency_mhz"] = self._parse_numeric_or_keep(entry.get("aiv_frequency"))
            break
        return info

    def _extract_msprof_op_metrics(self, output_dir: Path, device_info: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        metrics: list[dict[str, Any]] = []
        basic_paths = sorted(output_dir.rglob("OpBasicInfo*.csv"))
        for basic_path in basic_paths:
            basic_rows = self._read_csv_rows(basic_path)
            if not basic_rows:
                continue
            row = basic_rows[0]
            op_type = row.get("Op Type") or ""
            op_name = row.get("Op Name") or ""
            input_shapes = row.get("Input Shapes")
            input_data_types = row.get("Input Data Types")
            if self._is_cache_clear_op(op_name, op_type, input_shapes, input_data_types):
                continue
            metric: dict[str, Any] = {
                "op_name": row.get("Op Name"),
                "sample_count": len(basic_rows),
                "op_type": row.get("Op Type"),
                "device_id": self._parse_numeric_or_keep(row.get("Device Id")),
                "pid": self._parse_numeric_or_keep(row.get("Pid")),
                "current_freq": self._parse_numeric_or_keep(row.get("Current Freq")),
                "rated_freq": self._parse_numeric_or_keep(row.get("Rated Freq")),
                "aic_frequency_mhz": (device_info or {}).get("aic_frequency_mhz"),
                "aiv_frequency_mhz": (device_info or {}).get("aiv_frequency_mhz"),
                "task_type": None,
                "hf32_eligible": None,
                "input_shapes": None,
                "input_data_types": None,
                "input_formats": None,
                "output_shapes": None,
                "output_data_types": None,
                "output_formats": None,
                "block_dim": self._first_present_value(row, ("Block Dim", "Block Num")),
                "mix_block_dim": self._first_present_value(row, ("Mix Block Dim", "Mix Block Num")),
            }
            for field_key, column in AVERAGED_FIELDS.items():
                metric[field_key] = self._average_column(basic_rows, column)
            metric.update(self._duration_statistics(basic_rows))
            detail_rows = self._merge_msprof_op_detail_metrics(metric, basic_path.parent)
            metric.update(self._build_grouped_metrics(metric, detail_rows))
            metrics.append(self._finalize_msprof_metric(metric))
        return metrics

    def _merge_msprof_op_detail_metrics(self, metric: dict[str, Any], opprof_dir: Path) -> dict[str, list[dict[str, str]]]:
        detail_rows: dict[str, list[dict[str, str]]] = {}
        for path in sorted(opprof_dir.glob("*.csv")):
            if path.name.startswith("OpBasicInfo"):
                continue
            rows = self._read_csv_rows(path)
            if not rows:
                continue
            detail_rows[path.stem.split("_")[0]] = rows
            for field_key, column in AVERAGED_FIELDS.items():
                if metric.get(field_key) is not None:
                    continue
                value = self._average_column(rows, column)
                if value is not None:
                    metric[field_key] = value
        self._merge_msprof_op_summary_metrics(metric, detail_rows)
        return detail_rows

    def _drop_none_fields(self, metric: dict[str, Any]) -> dict[str, Any]:
        if self._profiler_mode() == "msprof":
            return dict(metric)
        cleaned = dict(metric)
        for key in DROPPED_NULL_KEYS:
            if cleaned.get(key) is None:
                cleaned.pop(key, None)
        return cleaned

    def _finalize_msprof_metric(self, metric: dict[str, Any]) -> dict[str, Any]:
        cleaned = {key: metric.get(key) for key in TOP_LEVEL_BASIC_KEYS if key in metric}
        for key in ("duration_samples_us", "duration_statistic", "noise_band"):
            if key in metric:
                cleaned[key] = metric[key]
        for group_name in GROUPED_KEYS:
            if group_name == "OpBasicInfo":
                continue
            cleaned[group_name] = metric.get(group_name, {})
        return self._drop_none_fields(cleaned)

    def _build_grouped_metrics(
        self,
        metric: dict[str, Any],
        detail_rows: dict[str, list[dict[str, str]]],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for group_name, keys in GROUPED_KEYS.items():
            if group_name == "OpBasicInfo":
                continue
            rows = detail_rows.get(group_name, [])
            group = self._build_raw_detail_group(rows)
            for key in keys:
                if key not in metric:
                    continue
                value = self._summarize_group_value(rows, key, metric.get(key))
                if value is not None:
                    group[key] = value
            grouped[group_name] = group
        return grouped

    def _build_raw_detail_group(self, rows: list[dict[str, str]]) -> dict[str, Any]:
        if not rows:
            return {}
        group: dict[str, Any] = {}
        for column in rows[0]:
            if not column or column.lower() in NON_PERF_COLUMNS:
                continue
            value = self._summarize_column(rows, column)
            group[self._normalize_metric_name(column)] = value
        return group

    def _summarize_column(self, rows: list[dict[str, str]], column: str) -> Any:
        values = [self._parse_float(row.get(column)) for row in rows]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            return None
        return self._summarize_numeric_values(numeric_values)

    def _summarize_group_value(
        self,
        rows: list[dict[str, str]],
        field_key: str,
        fallback: Any,
    ) -> Any:
        column = AVERAGED_FIELDS.get(field_key)
        if not rows or column is None:
            return fallback
        values = [self._parse_float(row.get(column)) for row in rows]
        numeric_values = [value for value in values if value is not None]
        if not numeric_values:
            return fallback
        return self._summarize_numeric_values(numeric_values)

    def _summarize_numeric_values(self, numeric_values: list[float]) -> float | dict[str, float]:
        first = numeric_values[0]
        if len(numeric_values) > 1 and any(value != first for value in numeric_values[1:]):
            return {
                "avg": round(sum(numeric_values) / len(numeric_values), 6),
                "min": round(min(numeric_values), 6),
                "max": round(max(numeric_values), 6),
            }
        return round(first, 6)

    def _normalize_metric_name(self, column: str) -> str:
        normalized = column.strip()
        for src, dst in (
            ("(GB/s)", "_gbps"),
            ("(KB)", "_kb"),
            ("(us)", "_us"),
            ("(%)", "_pct"),
            ("(estimate)", "_estimate"),
            ("/", "_per_"),
            (" ", "_"),
            ("-", "_"),
        ):
            normalized = normalized.replace(src, dst)
        normalized = normalized.replace("(", "_").replace(")", "")
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        return normalized.strip("_").lower()

    def _build_unmapped_metrics(self, basic_row: dict[str, str], detail_rows: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, Any]]:
        unmapped = {
            "OpBasicInfo": self._build_unmapped_group(
                set(basic_row.keys()),
                {
                    "Op Name",
                    "Op Type",
                    "Task Duration(us)",
                    "Task Wait Time(us)",
                    "Block Dim",
                    "Block Num",
                    "Mix Block Dim",
                    "Mix Block Num",
                    "Device Id",
                    "Pid",
                    "Current Freq",
                    "Rated Freq",
                },
            )
        }
        detail_used_columns = {
            "PipeUtilization": {"aic_total_cycles", "aic_mac_time(us)", "aic_mac_ratio", "aic_scalar_time(us)", "aic_scalar_ratio", "aic_mte1_time(us)", "aic_mte1_ratio", "aic_mte2_time(us)", "aic_mte2_ratio", "aic_fixpipe_time(us)", "aic_fixpipe_ratio", "aic_icache_miss_rate", "aiv_time(us)", "aiv_total_cycles", "aiv_vec_time(us)", "aiv_vec_ratio", "aiv_scalar_time(us)", "aiv_scalar_ratio", "aiv_mte2_time(us)", "aiv_mte2_ratio", "aiv_mte3_time(us)", "aiv_mte3_ratio", "aiv_icache_miss_rate", "aiv_time(us)", "aic_time(us)", "aiv_mte2_active_bw(GB/s)", "aiv_mte3_active_bw(GB/s)", "aic_mte2_active_bw(GB/s)", "aic_mte3_active_bw(GB/s)"},
            "Memory": {"GM_to_UB_datas(KB)", "UB_to_GM_datas(KB)", "read_main_memory_datas(KB)", "write_main_memory_datas(KB)", "GM_to_UB_bw_usage_rate(%)", "UB_to_GM_bw_usage_rate(%)", "aiv_gm_to_ub_bw(GB/s)", "aiv_ub_to_gm_bw(GB/s)", "aiv_mte2_instructions", "aic_mte2_instructions"},
            "MemoryUB": {"aiv_ub_read_bw_vector(GB/s)", "aiv_ub_write_bw_vector(GB/s)", "aiv_ub_read_bw_scalar(GB/s)", "aiv_ub_write_bw_scalar(GB/s)"},
            "MemoryL0": set(),
            "L2Cache": {"aiv_total_hit_rate(%)", "aiv_read_hit_rate(%)", "aiv_write_hit_rate(%)", "aic_total_hit_rate(%)", "aic_read_hit_rate(%)", "aic_write_hit_rate(%)", "aiv_write_cache_hit", "aiv_write_cache_miss_allocate", "aic_write_cache_hit", "aic_write_cache_miss_allocate"},
            "ResourceConflictRatio": {"aiv_vec_total_cflt_ratio", "aiv_vec_bankgroup_cflt_ratio", "aiv_vec_bank_cflt_ratio", "aiv_vec_resc_cflt_ratio", "aiv_vec_mte_cflt_ratio", "aiv_vec_wait_ratio", "aiv_mte2_wait_ratio", "aiv_mte3_wait_ratio", "aic_vec_total_cflt_ratio", "aic_vec_bankgroup_cflt_ratio", "aic_vec_bank_cflt_ratio", "aic_vec_resc_cflt_ratio", "aic_vec_mte_cflt_ratio", "aic_vec_wait_ratio", "aic_mte2_wait_ratio", "aic_mte3_wait_ratio"},
            "ArithmeticUtilization": {"aiv_vec_fp32_ratio", "aiv_vec_fp16_ratio", "aiv_vec_int32_ratio", "aiv_vec_int16_ratio", "aiv_vec_misc_ratio", "aiv_vec_fops", "cube_utilization(%)"},
        }
        for group_name, rows in detail_rows.items():
            fieldnames = set(rows[0].keys()) if rows else set()
            unmapped[group_name] = self._build_unmapped_group(fieldnames, detail_used_columns.get(group_name, set()))
        return unmapped

    def _build_unmapped_group(self, fieldnames: set[str], used_columns: set[str]) -> dict[str, Any]:
        columns = sorted(name for name in fieldnames if name and name not in used_columns)
        return {"count": len(columns), "columns": columns}

    def _parse_numeric_or_keep(self, value: str | None) -> int | float | str | None:
        parsed = self._parse_float(value)
        if parsed is None:
            return value
        return int(parsed) if parsed.is_integer() else parsed

    def _merge_msprof_op_summary_metrics(self, metric: dict[str, Any], detail_rows: dict[str, list[dict[str, str]]]) -> None:
        pipe_rows = detail_rows.get("PipeUtilization")
        prefix = self._detect_core_prefix(pipe_rows or [])
        if pipe_rows:
            core_time = self._average_column(pipe_rows, f"{prefix}_time(us)")
            metric["core_prefix"] = prefix
            metric["core_time_us"] = self._round_value(core_time, 6)
            metric["vec_ratio_pct"] = self._scaled_average(pipe_rows, f"{prefix}_vec_ratio")
            metric["scalar_ratio_pct"] = self._scaled_average(pipe_rows, f"{prefix}_scalar_ratio")
            metric["mte2_ratio_pct"] = self._scaled_average(pipe_rows, f"{prefix}_mte2_ratio")
            metric["mte3_ratio_pct"] = self._scaled_average(pipe_rows, f"{prefix}_mte3_ratio")
            metric["icache_miss_pct"] = self._scaled_average(pipe_rows, f"{prefix}_icache_miss_rate")
            metric["mte2_active_bw_gbps"] = self._round_value(self._average_column(pipe_rows, f"{prefix}_mte2_active_bw(GB/s)"), 6)
            metric["mte3_active_bw_gbps"] = self._round_value(self._average_column(pipe_rows, f"{prefix}_mte3_active_bw(GB/s)"), 6)
            duration = metric.get("task_duration_us")
            if duration is not None and core_time is not None:
                overhead = max(0.0, duration - core_time)
                metric["head_overhead_us"] = round(overhead, 6)
                metric["head_overhead_pct"] = round(overhead / duration * 100, 4) if duration else 0.0

        mem_rows = detail_rows.get("Memory")
        if mem_rows:
            gm_to_ub_total = self._sum_column(mem_rows, "GM_to_UB_datas(KB)")
            ub_to_gm_total = self._sum_column(mem_rows, "UB_to_GM_datas(KB)")
            metric["gm_to_ub_kb_total"] = self._round_value(gm_to_ub_total, 6)
            metric["ub_to_gm_kb_total"] = self._round_value(ub_to_gm_total, 6)
            metric["read_main_memory_kb_total"] = self._round_value(self._sum_column(mem_rows, "read_main_memory_datas(KB)"), 6)
            metric["write_main_memory_kb_total"] = self._round_value(self._sum_column(mem_rows, "write_main_memory_datas(KB)"), 6)
            metric["gm_to_ub_bw_usage_pct"] = self._round_value(self._average_column(mem_rows, "GM_to_UB_bw_usage_rate(%)"), 6)
            metric["ub_to_gm_bw_usage_pct"] = self._round_value(self._average_column(mem_rows, "UB_to_GM_bw_usage_rate(%)"), 6)
            metric["gm_to_ub_avg_bw_gbps"] = self._round_value(self._average_column(mem_rows, "aiv_gm_to_ub_bw(GB/s)"), 6)
            metric["ub_to_gm_avg_bw_gbps"] = self._round_value(self._average_column(mem_rows, "aiv_ub_to_gm_bw(GB/s)"), 6)
            mte2_instr = self._sum_column(mem_rows, "aiv_mte2_instructions") + self._sum_column(mem_rows, "aic_mte2_instructions")
            if gm_to_ub_total is not None and mte2_instr:
                metric["avg_mte2_transfer_kb"] = round(gm_to_ub_total / mte2_instr, 6)

        ub_rows = detail_rows.get("MemoryUB")
        if ub_rows:
            metric["ub_read_bw_vector_gbps"] = self._round_value(self._average_column(ub_rows, "aiv_ub_read_bw_vector(GB/s)"), 6)
            metric["ub_write_bw_vector_gbps"] = self._round_value(self._average_column(ub_rows, "aiv_ub_write_bw_vector(GB/s)"), 6)
            metric["ub_read_bw_scalar_gbps"] = self._round_value(self._average_column(ub_rows, "aiv_ub_read_bw_scalar(GB/s)"), 6)
            metric["ub_write_bw_scalar_gbps"] = self._round_value(self._average_column(ub_rows, "aiv_ub_write_bw_scalar(GB/s)"), 6)

        l2_rows = detail_rows.get("L2Cache")
        if l2_rows:
            metric["l2_total_hit_rate_pct"] = self._round_value(self._average_column(l2_rows, f"{prefix}_total_hit_rate(%)"), 6)
            metric["l2_read_hit_rate_pct"] = self._round_value(self._average_column(l2_rows, f"{prefix}_read_hit_rate(%)"), 6)
            metric["l2_write_hit_rate_pct"] = self._round_value(self._average_column(l2_rows, f"{prefix}_write_hit_rate(%)"), 6)
            metric["write_cache_hit_total"] = self._round_value(self._sum_column(l2_rows, f"{prefix}_write_cache_hit"), 6)
            metric["write_cache_miss_allocate_total"] = self._round_value(self._sum_column(l2_rows, f"{prefix}_write_cache_miss_allocate"), 6)

        rc_rows = detail_rows.get("ResourceConflictRatio")
        if rc_rows:
            metric["vec_total_cflt_pct"] = self._scaled_average(rc_rows, f"{prefix}_vec_total_cflt_ratio")
            metric["bankgroup_cflt_pct"] = self._scaled_average(rc_rows, f"{prefix}_vec_bankgroup_cflt_ratio")
            metric["bank_cflt_pct"] = self._scaled_average(rc_rows, f"{prefix}_vec_bank_cflt_ratio")
            metric["resc_cflt_pct"] = self._scaled_average(rc_rows, f"{prefix}_vec_resc_cflt_ratio")
            metric["mte_cflt_pct"] = self._scaled_average(rc_rows, f"{prefix}_vec_mte_cflt_ratio")
            metric["vec_wait_pct"] = self._scaled_average(rc_rows, f"{prefix}_vec_wait_ratio")
            metric["mte2_wait_pct"] = self._scaled_average(rc_rows, f"{prefix}_mte2_wait_ratio")
            metric["mte3_wait_pct"] = self._scaled_average(rc_rows, f"{prefix}_mte3_wait_ratio")

        arith_rows = detail_rows.get("ArithmeticUtilization")
        if arith_rows:
            metric["vec_fp32_pct"] = self._scaled_average(arith_rows, "aiv_vec_fp32_ratio")
            metric["vec_fp16_pct"] = self._scaled_average(arith_rows, "aiv_vec_fp16_ratio")
            metric["vec_int32_pct"] = self._scaled_average(arith_rows, "aiv_vec_int32_ratio")
            metric["vec_int16_pct"] = self._scaled_average(arith_rows, "aiv_vec_int16_ratio")
            metric["vec_misc_pct"] = self._scaled_average(arith_rows, "aiv_vec_misc_ratio")
            metric["vec_fops_per_core"] = self._round_value(self._average_column(arith_rows, "aiv_vec_fops"), 6)

    def _detect_core_prefix(self, rows: list[dict[str, str]]) -> str:
        if not rows:
            return "aiv"
        aiv_time = self._parse_float(rows[0].get("aiv_time(us)")) or 0.0
        aic_time = self._parse_float(rows[0].get("aic_time(us)")) or 0.0
        return "aic" if aic_time > aiv_time else "aiv"

    def _scaled_average(self, rows: list[dict[str, str]], column: str, scale: float = 100.0, digits: int = 4) -> float | None:
        value = self._average_column(rows, column)
        if value is None:
            return None
        return round(value * scale, digits)

    def _sum_column(self, rows: list[dict[str, str]], column: str) -> float:
        return sum(self._parse_float(row.get(column)) or 0.0 for row in rows)

    def _round_value(self, value: float | None, digits: int) -> float | None:
        if value is None:
            return None
        return round(value, digits)

    def _read_csv_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return []
            return list(reader)

    def _aggregate_op_summary(self, rows: list[dict[str, str]], device_info: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
        order: list[tuple[str, ...]] = []
        ignored_op_names = {"N/A", "", None}
        for row in rows:
            op_name = row.get("Op Name")
            if op_name in ignored_op_names:
                continue
            op_type = row.get("OP Type") or ""
            if self._is_cache_clear_op(
                op_name,
                op_type,
                row.get("Input Shapes"),
                row.get("Input Data Types"),
            ):
                continue
            key = tuple((row.get(col) or "") for col in GROUP_KEY_COLUMNS)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)

        aggregated: list[dict[str, Any]] = []
        for key in order:
            selected_rows = groups[key]
            if not selected_rows:
                continue
            metric_rows = self._select_measured_rows(selected_rows)
            metrics: dict[str, Any] = {
                "op_name": metric_rows[0].get("Op Name"),
                "sample_count": len(metric_rows),
                "device_id": self._parse_numeric_or_keep(
                    metric_rows[0].get("Device_id") or metric_rows[0].get("Device Id")
                ),
                "aic_frequency_mhz": (device_info or {}).get("aic_frequency_mhz"),
                "aiv_frequency_mhz": (device_info or {}).get("aiv_frequency_mhz"),
            }
            for field_key, column in PRESERVED_FIELDS.items():
                metrics[field_key] = metric_rows[0].get(column)
            for field_key, columns in PRESERVED_FIELD_ALIASES.items():
                metrics[field_key] = self._first_present_value(metric_rows[0], columns)
            for field_key, column in AVERAGED_FIELDS.items():
                metrics[field_key] = self._average_column(metric_rows, column)
            metrics.update(self._duration_statistics(metric_rows))
            metrics.update(self._build_grouped_metrics(metrics, {}))
            aggregated.append(self._finalize_msprof_metric(metrics))
        return aggregated

    def _select_measured_rows(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        window = self.repeat * self.inner_iters
        if len(rows) > window:
            return rows[-window:]
        return rows

    def _duration_statistics(self, rows):
        samples = [self._parse_float(row.get("Task Duration(us)")) for row in rows]
        samples = [value for value in samples if value is not None and math.isfinite(value) and value > 0]
        if not samples:
            return {}
        center = statistics.median(samples)
        # Preserve every sample for audit; the median reduces isolated spike influence.
        noise = max(abs(value / center - 1) for value in samples) if len(samples) >= 3 else None
        return {"task_duration_us": center, "duration_samples_us": samples,
                "duration_statistic": "median", "noise_band": noise}

    def _first_present_value(self, row: dict[str, str], columns: tuple[str, ...]) -> str | None:
        for column in columns:
            value = row.get(column)
            if value is not None:
                return value
        return None

    def _average_column(self, rows: list[dict[str, str]], column: str) -> float | None:
        values = []
        for row in rows:
            value = self._parse_float(row.get(column))
            if value is not None:
                values.append(value)
        if not values:
            return None
        return sum(values) / len(values)

    def _parse_float(self, value: str | None) -> float | None:
        if value is None:
            return None
        text = str(value).replace("%", "").strip()
        if not text or text.upper() == "N/A":
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _write_result_json(self, path: Path, metrics: dict[str, list[dict[str, Any]]]) -> None:
        path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
