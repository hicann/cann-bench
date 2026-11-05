#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import logging
import os
import socket
import tempfile
import time
import traceback
from datetime import timedelta
from queue import Empty
from typing import Any, Callable, Dict, List, Optional

import torch

from ..base.models import CaseSpec
from ..base.result import (
    FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
    FAILURE_TYPE_REFERENCE_INVALID,
    get_accuracy_failure_type,
    AccuracyResult,
)
from ..registry.golden_registry import get_golden_loader
from ..utils import str_to_torch_dtype
from ..utils.baseline_resolver import resolve_hardware
from ..utils.case_eligibility import supports_case
from ..utils.compare import compare_tensors
from ..utils.thresholds import PRECISION_THRESHOLDS
from ..checkers.relative_error_checker import (
    RelativeErrorOutputResult,
    _convert_to_output_result,
)
from ..checkers.mc2_three_way_checker import (
    MC2ReferenceInvalidError,
    MC2ThreeWayOutputResult,
    check_mc2_references,
    compare_mc2_outputs,
)
from .mc2_cpu_collectives import CPUCollectives
from .op_runner import OpRunResult
from .perf_eval import PerfResult
from .results import EvalCaseResult


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _restore_mc2_visibility(world_size: int) -> None:
    """Restore the physical card set before spawning MC2 rank processes.

    torch_npu may rewrite ASCEND_RT_VISIBLE_DEVICES to logical IDs when the
    eval-child initializes a device. BENCH_DEVICE_VISIBILITY is the original
    physical device set saved by the coordinator; without restoration, later spawned
    ranks in parallel lanes can silently start on cards 0,1 again.
    """
    saved = os.environ.get("BENCH_DEVICE_VISIBILITY", "")
    if not saved:
        return
    devices = saved.split(",")
    if (
        len(devices) < world_size
        or any(not item.isdigit() for item in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError(
            f"Invalid BENCH_DEVICE_VISIBILITY for {world_size} MC2 ranks: {saved!r}"
        )
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = saved


class MC2DistributedEvaluator:
    """Run MC2 communication-fused operators with one process per rank.

    The normal cann-bench path is single-process. MC2 cases still need an HCCL
    process group, but their golden logic now lives in each task's golden.py via
    the ``mc2_distributed_golden`` hook.
    """

    def __init__(self, config):
        self.config = config
        self.golden_loader = get_golden_loader(
            "cann", bench_root=str(self.config.tasks_root)
        )

    def evaluate_case(
        self, case: CaseSpec, custom_thresholds: Optional[Dict[str, float]] = None
    ) -> EvalCaseResult:
        attrs = case.attrs or {}
        world_size = int(attrs.get("world_size", attrs.get("ep_world_size", 1)))
        if world_size < 2:
            raise ValueError("MC2 distributed evaluation requires at least two ranks")
        case_id_str = case.get_case_id_str()
        baseline_source = dict(
            (getattr(case, "metadata", None) or {}).get("baseline_source") or {}
        )
        port = _find_free_port()

        # Resolve the task-local golden module in the parent process so missing
        # golden.py or hook failures are reported before spawning ranks.
        try:
            self.golden_loader.get_mc2_distributed_golden(case.rel_path)
        except Exception as exc:
            if attrs.get("mc2_accuracy_mode") != "three_way":
                raise
            return EvalCaseResult(
                case_id=case_id_str,
                rel_path=case.rel_path,
                operator=case.operator,
                case_num=case.case_num,
                success=False,
                error_msg=f"MC2 reference loading failed: {exc}",
                baseline_perf_us=case.baseline_perf_us,
                t_hw_us=case.t_hw_us,
                baseline_source=baseline_source,
                failure_type=FAILURE_TYPE_REFERENCE_INVALID,
            )

        case_payload = {
            "case_id": case_id_str,
            "rel_path": case.rel_path,
            "operator": case.operator,
            "case_num": case.case_num,
            "input_shapes": case.input_shapes,
            "dtypes": case.dtypes,
            "value_ranges": case.value_ranges,
            "attrs": attrs,
            "tasks_root": str(self.config.tasks_root),
            "reports_dir": str(self.config.reports_dir),
            "profiler_level": getattr(self.config, "profiler_level", "Level1"),
            # Leave time for terminating ranks and writing the child report
            # before the outer coordinator's per-case timeout expires.
            "timeout": max(
                1.0,
                float(getattr(self.config, "timeout_per_operator", 300) or 300) - 30,
            ),
        }

        _restore_mc2_visibility(world_size)
        ctx = torch.multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        processes = []
        for rank in range(world_size):
            proc = ctx.Process(
                target=_mc2_rank_entry,
                args=(
                    rank,
                    world_size,
                    port,
                    case_payload,
                    self.config.warmup,
                    self.config.repeat,
                    bool(self.config.enable_profiler),
                    custom_thresholds or {},
                    queue,
                ),
            )
            proc.start()
            processes.append(proc)

        rank_results = []
        terminated_ranks = []
        deadline = time.monotonic() + case_payload["timeout"]
        # Drain while workers are alive: joining before reading can deadlock
        # when per-rank three-way evidence fills the multiprocessing pipe.
        try:
            while len(rank_results) < world_size and time.monotonic() < deadline:
                try:
                    rank_results.append(queue.get(timeout=0.2))
                except Empty:
                    if all(not proc.is_alive() for proc in processes):
                        break
            for proc in processes:
                # Profiler/Python teardown may outlive result delivery. Share the
                # case deadline rather than killing a successful rank after 5s.
                proc.join(timeout=max(0, deadline - time.monotonic()))
        finally:
            for rank, proc in enumerate(processes):
                if proc.is_alive():
                    terminated_ranks.append(rank)
                    proc.terminate()
            for proc in processes:
                proc.join(timeout=5)
            queue.close()
            for proc in processes:
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=5)

        process_errors = [
            f"rank {idx} exited with code {proc.exitcode}"
            for idx, proc in enumerate(processes)
            if proc.exitcode not in (0, None)
        ]
        rank_errors = [r.get("error") for r in rank_results if r.get("error")]
        if terminated_ranks:
            phase = (
                "rank teardown"
                if len(rank_results) == world_size
                else "rank execution/results"
            )
            process_errors.append(
                f"case deadline exceeded during {phase}; terminated ranks {terminated_ranks}"
            )
        if process_errors or rank_errors or len(rank_results) != world_size:
            error_msg = "; ".join(process_errors + rank_errors)
            if len(rank_results) != world_size:
                error_msg = (error_msg + "; " if error_msg else "") + (
                    f"only collected {len(rank_results)}/{world_size} rank results"
                )
            return EvalCaseResult(
                case_id=case_id_str,
                rel_path=case.rel_path,
                operator=case.operator,
                case_num=case.case_num,
                success=False,
                error_msg=f"MC2 distributed evaluation failed: {error_msg}",
                baseline_perf_us=case.baseline_perf_us,
                baseline_source=baseline_source,
                t_hw_us=case.t_hw_us,
                failure_type=(
                    FAILURE_TYPE_REFERENCE_INVALID
                    if any(
                        r.get("failure_type") == FAILURE_TYPE_REFERENCE_INVALID
                        for r in rank_results
                    )
                    else FAILURE_TYPE_COMPILE_RUNTIME_ERROR
                ),
            )

        rank_results.sort(key=lambda item: int(item.get("rank", 0)))
        passed = all(bool(r.get("passed")) for r in rank_results)
        dtype = next(
            (str(r.get("dtype")) for r in rank_results if r.get("dtype")), None
        )
        dtype = dtype or (case.dtypes[0] if case.dtypes else "float32")
        threshold = next(
            (
                float(r.get("threshold"))
                for r in rank_results
                if r.get("threshold") is not None
            ),
            None,
        )
        threshold = (
            threshold
            if threshold is not None
            else _threshold_for_dtype(dtype, custom_thresholds or {})
        )

        # 还原各 rank 的逐输出结果（rank 侧已序列化为 RelativeErrorOutputResult dict）
        output_results = []
        for rank_result in rank_results:
            for item in rank_result.get("output_results", []):
                output_cls = (
                    MC2ThreeWayOutputResult
                    if attrs.get("mc2_accuracy_mode") == "three_way"
                    else RelativeErrorOutputResult
                )
                output_results.append(output_cls.from_dict(item))

        # 跨 rank 聚合指标，写入 metadata（与 RelativeErrorChecker 的 metadata 字段对齐）
        metadata = {
            "checker_name": "relative_error",
            "mc2_distributed": True,
            "world_size": world_size,
            "dtype": dtype,
            "mere": max((float(r.get("mere", 0.0)) for r in rank_results), default=0.0),
            "mare": max((float(r.get("mare", 0.0)) for r in rank_results), default=0.0),
            "max_diff": max(
                (float(r.get("max_diff", 0.0)) for r in rank_results), default=0.0
            ),
            "mean_diff": max(
                (float(r.get("mean_diff", 0.0)) for r in rank_results), default=0.0
            ),
            "mismatch_count": sum(
                int(r.get("mismatch_count", 0)) for r in rank_results
            ),
            "total_count": sum(int(r.get("total_count", 0)) for r in rank_results),
            "mismatch_ratio": max(
                (float(r.get("mismatch_ratio", 0.0)) for r in rank_results), default=0.0
            ),
            "small_value_error_count": sum(
                int(r.get("small_value_error_count", 0)) for r in rank_results
            ),
            "small_value_cpu_error_count": sum(
                int(r.get("small_value_cpu_error_count", 0)) for r in rank_results
            ),
            "small_value_total_count": sum(
                int(r.get("small_value_total_count", 0)) for r in rank_results
            ),
            "cancel_error_count": sum(
                int(r.get("cancel_error_count", 0)) for r in rank_results
            ),
            "cancel_cpu_error_count": sum(
                int(r.get("cancel_cpu_error_count", 0)) for r in rank_results
            ),
            "cancel_total_count": sum(
                int(r.get("cancel_total_count", 0)) for r in rank_results
            ),
        }

        if attrs.get("mc2_accuracy_mode") == "three_way":
            threshold = None
            metadata = {
                "checker_name": "mc2_three_way",
                "mc2_distributed": True,
                "world_size": world_size,
                "dtype": dtype,
                "reference": "CPU FP32 + Gloo",
                "baseline": "NPU cascade + HCCL",
                "standard": "mc2_test/script/new_compare.py:Result.check_result",
            }
            if any(
                r.metadata.get("failure_type") == FAILURE_TYPE_REFERENCE_INVALID
                for r in output_results
            ):
                metadata["failure_type"] = FAILURE_TYPE_REFERENCE_INVALID
            elif any(r.metadata.get("failure_type") for r in output_results):
                metadata["failure_type"] = FAILURE_TYPE_COMPILE_RUNTIME_ERROR

        measurement = {
            "warmup": max(0, self.config.warmup),
            "repeat": max(1, self.config.repeat),
            "timing_statistic": "maximum_across_ranks_of_mean_wall_time_us",
            "clock": "time.perf_counter",
            "cold_start_excluded": self.config.warmup > 0,
            "includes_final_barrier": True,
            "measured": bool(self.config.enable_profiler),
        }
        metadata["measurement"] = measurement

        # MoeDistributeDispatchV2 的输出顺序与 golden 可能因路由实现差异而不同，
        # 但若所有误差指标恒为 0（逐元素完全一致），仍判为通过。该特判内化在
        # runner 内部，仅用于 legacy 模式，不能覆盖 three_way 的独立判定。
        if (
            not passed
            and attrs.get("mc2_accuracy_mode", "legacy") == "legacy"
            and case.operator == "MoeDistributeDispatchV2"
            and int(metadata["mismatch_count"]) == 0
            and float(metadata["mere"]) == 0.0
            and float(metadata["mare"]) == 0.0
            and float(metadata["max_diff"]) == 0.0
            and float(metadata["mean_diff"]) == 0.0
        ):
            passed = True

        accuracy_result = AccuracyResult(
            passed=passed,
            threshold=threshold,
            error_msg="; ".join(
                f"rank {r.get('rank')}: {r.get('compare_error')}"
                for r in rank_results
                if r.get("compare_error")
            )
            or None,
            output_results=output_results,
            metadata=metadata,
        )

        elapsed_us = max(float(r.get("elapsed_us", 0.0)) for r in rank_results)
        golden_elapsed_us = max(
            float(r.get("golden_elapsed_us", 0.0)) for r in rank_results
        )
        baseline_perf_us = (
            case.baseline_perf_us if case.baseline_perf_us > 0 else golden_elapsed_us
        )
        if case.baseline_perf_us <= 0 and golden_elapsed_us > 0:
            baseline_source["baseline_perf_us"] = {"source": "runtime_reference"}

        rank0_perf = next(
            (r.get("perf_result") for r in rank_results if r.get("rank") == 0), None
        )
        perf_result = None
        if elapsed_us > 0:
            perf_result = PerfResult(elapsed_us=elapsed_us)
            perf_result.metadata.update(measurement)
            if rank0_perf:
                perf_result.op_times = rank0_perf.get("op_times", {}) or {}
                perf_result.error_msg = rank0_perf.get("error")
                perf_result.metadata["repeat"] = int(
                    rank0_perf.get("_repeat", max(1, self.config.repeat))
                )
                perf_result.metadata["warmup_used"] = self.config.warmup > 0

        return EvalCaseResult(
            case_id=case_id_str,
            rel_path=case.rel_path,
            operator=case.operator,
            case_num=case.case_num,
            success=passed,
            accuracy_result=accuracy_result,
            perf_result=perf_result if passed else None,
            golden_run_result=OpRunResult(
                success=True, elapsed_us=golden_elapsed_us, device="npu"
            ),
            ai_run_result=OpRunResult(
                success=passed, elapsed_us=elapsed_us, device="npu"
            ),
            error_msg=None if passed else accuracy_result.error_msg,
            baseline_perf_us=baseline_perf_us,
            baseline_source=baseline_source,
            t_hw_us=case.t_hw_us,
            failure_type=get_accuracy_failure_type(accuracy_result),
        )


def _mc2_rank_entry(
    rank: int,
    world_size: int,
    port: int,
    case_payload: Dict[str, Any],
    warmup: int,
    repeat: int,
    enable_perf: bool,
    custom_thresholds: Dict[str, float],
    queue,
) -> None:
    try:
        _restore_mc2_visibility(world_size)
        result = _run_rank(
            rank,
            world_size,
            port,
            case_payload,
            warmup,
            repeat,
            enable_perf,
            custom_thresholds,
        )
    except Exception as exc:
        result = {
            "rank": rank,
            "passed": False,
            "error": f"{exc}\n{traceback.format_exc()}",
            "failure_type": (
                FAILURE_TYPE_REFERENCE_INVALID
                if isinstance(exc, MC2ReferenceInvalidError)
                else FAILURE_TYPE_COMPILE_RUNTIME_ERROR
            ),
        }
    queue.put(result)


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    case_payload: Dict[str, Any],
    warmup: int,
    repeat: int,
    enable_perf: bool,
    custom_thresholds: Dict[str, float],
) -> Dict[str, Any]:
    import torch.distributed as dist
    import torch_npu
    from torch.distributed.distributed_c10d import _get_default_group

    attrs = case_payload["attrs"]
    mode = attrs.get("mc2_accuracy_mode", "legacy")
    if mode not in ("legacy", "three_way"):
        raise ValueError(f"Unknown MC2 accuracy mode: {mode}")
    three_way = mode == "three_way"
    operator = case_payload["operator"]
    if torch_npu.npu.device_count() < world_size:
        raise RuntimeError(
            f"MC2 needs {world_size} visible NPUs; got {torch_npu.npu.device_count()}"
        )
    supported_devices = attrs.get("supported_devices")
    device_name = torch_npu.npu.get_device_name(rank)
    if not supports_case(attrs, device_name):
        raise RuntimeError(
            f"{operator} supports {supported_devices}, got {device_name}"
        )
    torch_npu.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    # A2 MC2 does not support this variable. Scope the default to actual A5
    # hardware and preserve any explicit user configuration.
    if resolve_hardware(device_name) in ("950pr", "950dt"):
        os.environ.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=timedelta(seconds=case_payload.get("timeout", 600)),
    )

    auxiliary_groups = []
    try:
        hcomm_info = (
            _get_default_group()
            ._get_backend(torch.device("npu"))
            .get_hccl_comm_name(rank)
        )
        candidate_hcomm_info = hcomm_info
        if attrs.get("mc2_isolated_candidate_group", False):
            # Quantized reduction operators require an exclusive communication
            # domain. Only their candidate receives this name; reference
            # collectives and measurement/profiler barriers keep the default.
            candidate_group = dist.new_group(
                backend="hccl",
                timeout=timedelta(seconds=case_payload.get("timeout", 600)),
            )
            auxiliary_groups.append(candidate_group)
            candidate_hcomm_info = candidate_group._get_backend(
                torch.device("npu")
            ).get_hccl_comm_name(rank)
        loader = get_golden_loader("cann", bench_root=case_payload["tasks_root"])
        golden_module = loader._load_module(case_payload["rel_path"])
        golden_hook = loader.get_mc2_distributed_golden(
            case_payload["rel_path"], required=True
        )

        rank_ctx = {
            "rank": rank,
            "world_size": world_size,
            "device": device,
            "dist": dist,
            "torch_npu": torch_npu,
            "hcomm_info": hcomm_info,
            "case_id": case_payload["case_id"],
            "case_num": case_payload["case_num"],
            "rel_path": case_payload["rel_path"],
            "operator": operator,
            "attrs": attrs,
        }

        if hasattr(golden_module, "mc2_make_rank_inputs"):
            inputs = golden_module.mc2_make_rank_inputs(rank_ctx, case_payload)
        else:
            inputs = _make_rank_inputs(case_payload, rank, device)

        candidate = _load_candidate(operator)
        candidate_ctx = dict(rank_ctx, hcomm_info=candidate_hcomm_info)
        if hasattr(golden_module, "mc2_call_candidate"):

            def candidate_fn():
                return golden_module.mc2_call_candidate(
                    candidate, candidate_ctx, inputs, attrs
                )
        else:

            def candidate_fn():
                return _call_builtin_candidate(
                    candidate, operator, candidate_ctx, inputs, attrs
                )

        def golden_fn():
            return golden_hook(rank_ctx, inputs, attrs)

        cpu_output = None
        cpu_cascade_output = None
        if three_way:
            cpu_hook = getattr(golden_module, "mc2_cpu_golden", None)
            if not callable(cpu_hook):
                raise MC2ReferenceInvalidError(
                    "three_way requires a task-local mc2_cpu_golden hook"
                )
            try:
                cpu_group = dist.new_group(
                    backend="gloo",
                    timeout=timedelta(seconds=case_payload.get("timeout", 600)),
                )
            except Exception as exc:
                raise MC2ReferenceInvalidError(
                    f"CPU reference group creation failed: {exc}"
                ) from exc
            auxiliary_groups.append(cpu_group)
            cpu_ctx = dict(
                rank_ctx,
                device=torch.device("cpu"),
                dist=CPUCollectives(dist, cpu_group, rank, world_size),
            )
            cpu_inputs = _clone_outputs_for_compare(inputs)
            try:
                # Trusted task hook may additionally emulate intermediate rounding
                # for reference health. G for three-way accuracy stays FP32.
                reference_hook = getattr(golden_module, "mc2_cpu_references", None)
                if callable(reference_hook):
                    cpu_output, cpu_cascade_output = reference_hook(
                        cpu_ctx, cpu_inputs, attrs
                    )
                else:
                    cpu_output = cpu_hook(cpu_ctx, cpu_inputs, attrs)
            except Exception as exc:
                raise MC2ReferenceInvalidError(
                    f"CPU reference execution failed: {exc}"
                ) from exc
            # Keep CPU reference construction out of performance measurement.
            del cpu_inputs

        try:
            golden_output = golden_fn()
            torch.npu.synchronize(device)
            golden_output = _clone_outputs_for_compare(golden_output)
        except Exception as exc:
            if three_way:
                raise MC2ReferenceInvalidError(
                    f"NPU cascade execution failed: {exc}"
                ) from exc
            raise
        if three_way:
            reference_results = check_mc2_references(
                golden_output,
                cpu_output,
                rank,
                attrs.get("mc2_compare_dtype"),
                cascade_golden=cpu_cascade_output,
            )
            invalid = torch.tensor(
                [int(any(not r.passed for r in reference_results))], dtype=torch.int32
            )
            # All ranks make the same decision before candidate collectives or
            # performance loops, even when only one rank has a broken reference.
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=cpu_group)
            if invalid.item():
                reason = "; ".join(
                    r.error_msg for r in reference_results if not r.passed
                )
                if not reason:
                    reason = "MC2 reference invalid on another rank"
                    reference_results.append(
                        MC2ThreeWayOutputResult(
                            len(reference_results),
                            f"rank{rank}.reference",
                            error_msg=reason,
                            metadata={
                                "rank": rank,
                                "failure_type": FAILURE_TYPE_REFERENCE_INVALID,
                            },
                        )
                    )
                return {
                    "rank": rank,
                    "passed": False,
                    "compare_error": reason,
                    "output_results": [r.to_dict() for r in reference_results],
                    "failure_type": FAILURE_TYPE_REFERENCE_INVALID,
                }
        torch.npu.synchronize(device)
        candidate_output = candidate_fn()
        torch.npu.synchronize(device)
        candidate_output = _clone_outputs_for_compare(candidate_output)
        torch.npu.synchronize(device)
        if three_way:
            checked = compare_mc2_outputs(
                candidate_output,
                golden_output,
                cpu_output,
                rank,
                attrs.get("mc2_compare_dtype"),
                cascade_golden=cpu_cascade_output,
            )
            comparison = {
                "passed": all(r.passed for r in checked),
                "dtype": next((r.dtype for r in checked if r.dtype), "float32"),
                "threshold": 0.0,  # Per-metric limits are stored in each output, not this legacy scalar.
                "compare_error": "; ".join(r.error_msg for r in checked if r.error_msg),
                "output_results": [r.to_dict() for r in checked],
            }
        else:
            compare_result = compare_tensors(
                candidate_output,
                golden_output,
                case_payload["dtypes"][0],
                _threshold_for_dtype(case_payload["dtypes"][0], custom_thresholds),
                custom_thresholds=custom_thresholds,
            )
            comparison = {
                key: getattr(compare_result, key)
                for key in (
                    "passed",
                    "dtype",
                    "threshold",
                    "mere",
                    "mare",
                    "max_diff",
                    "mean_diff",
                    "mismatch_count",
                    "total_count",
                    "mismatch_ratio",
                    "small_value_error_count",
                    "small_value_cpu_error_count",
                    "small_value_total_count",
                    "cancel_error_count",
                    "cancel_cpu_error_count",
                    "cancel_total_count",
                )
            }
            comparison["dtype"] = str(comparison["dtype"])
            comparison["compare_error"] = compare_result.error_msg
            comparison["output_results"] = _rank_output_results(compare_result, rank)

        elapsed_us = 0.0
        golden_elapsed_us = 0.0
        perf_payload = None
        if enable_perf:
            elapsed_us = _measure(candidate_fn, dist, device, warmup, repeat)
            reference_timing_error = ""
            try:
                golden_elapsed_us = _measure(golden_fn, dist, device, warmup, repeat)
            except Exception as exc:
                if not three_way:
                    raise
                reference_timing_error = f"MC2 reference timing failed: {exc}"
            if three_way:
                invalid = torch.tensor(
                    [int(bool(reference_timing_error))], dtype=torch.int32
                )
                try:
                    # No healthy rank may enter profiler collectives while a
                    # peer exits after a reference-only timing failure.
                    dist.all_reduce(invalid, op=dist.ReduceOp.MAX, group=cpu_group)
                except Exception as exc:
                    raise MC2ReferenceInvalidError(
                        reference_timing_error
                        or f"MC2 reference timing synchronization failed: {exc}"
                    ) from exc
                if invalid.item():
                    raise MC2ReferenceInvalidError(
                        reference_timing_error
                        or "MC2 reference timing failed on another rank"
                    )
            perf_payload = _profile_candidate_rank0(
                candidate_fn,
                dist,
                device,
                case_payload,
                rank,
                warmup,
                repeat,
            )

        return {
            "rank": rank,
            **comparison,
            "elapsed_us": elapsed_us,
            "golden_elapsed_us": golden_elapsed_us,
            "perf_result": perf_payload,
        }
    finally:
        # Release subgroups before the default group, including partial setup
        # failures. One backend's teardown failure must not skip the others or
        # replace the original operator error.
        for group in reversed(auxiliary_groups):
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _clone_outputs_for_compare(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return output.detach().cpu().clone()
    if isinstance(output, tuple):
        return tuple(_clone_outputs_for_compare(item) for item in output)
    if isinstance(output, list):
        return [_clone_outputs_for_compare(item) for item in output]
    if isinstance(output, dict):
        return {key: _clone_outputs_for_compare(item) for key, item in output.items()}
    return output


def _make_rank_inputs(case_payload: Dict[str, Any], rank: int, device: torch.device):
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    attrs = case_payload["attrs"]
    seed = int(attrs.get("seed", 1))
    weight_same = int(attrs.get("weight_same", 0))

    # Distinguishable rank inputs prevent local repetition from passing as AllGather.
    x1_seed = (
        seed + rank * 1000003 if attrs.get("mc2_accuracy_mode") == "three_way" else seed
    )
    x1 = _make_tensor(
        shapes[0], dtypes[0], ranges[0] if ranges else None, x1_seed, device
    )
    x2_seed = seed + rank * 2 if weight_same else seed
    x2 = _make_tensor(
        shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, x2_seed, device
    )
    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(
            shapes[2],
            dtypes[2],
            ranges[2] if len(ranges) > 2 else None,
            rank * 3,
            device,
        )
    return {"x1": x1, "x2": x2, "bias": bias}


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device: torch.device):
    if shape is None:
        return None
    torch.manual_seed(int(seed))
    dtype = str_to_torch_dtype(dtype_name)
    if value_range is None:
        value_range = [0, 1]
    lo, hi = value_range
    if dtype.is_floating_point:
        tensor = (
            torch.empty(shape, dtype=torch.float32)
            .uniform_(float(lo), float(hi))
            .to(dtype)
        )
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def _load_candidate(operator: str):
    name_map = {
        "MatmulReduceScatter": "matmul_reduce_scatter",
        "MatmulReduceScatterV2": "matmul_reduce_scatter_v2",
        "MatmulAllReduce": "matmul_all_reduce",
        "AlltoAllMatmul": "all_to_all_matmul",
        "MatmulAlltoAll": "matmul_all_to_all",
        "AllGatherMatmul": "all_gather_matmul",
        "AllGatherMatmulV2": "all_gather_matmul_v2",
        "AlltoAllvGroupedMatMul": "allto_allv_grouped_mat_mul",
        "AlltoAllvQuantGroupedMatMul": "allto_allv_quant_grouped_mat_mul",
        "GroupedMatMulAlltoAllv": "grouped_mat_mul_allto_allv",
        "QuantAllReduce": "quant_all_reduce",
        "QuantReduceScatter": "quant_reduce_scatter",
        "QuantGroupedMatMulAlltoAllv": "quant_grouped_mat_mul_allto_allv",
        "MoeDistributeDispatchV2": "moe_distribute_dispatch_v2",
    }
    if operator not in name_map:
        raise RuntimeError(f"Unsupported MC2 operator: {operator}")
    func_name = name_map[operator]
    try:
        import cann_bench

        if hasattr(cann_bench, func_name):
            return getattr(cann_bench, func_name)
    except ImportError:
        pass

    if hasattr(torch.ops, "cann_bench") and hasattr(torch.ops.cann_bench, func_name):
        return getattr(torch.ops.cann_bench, func_name)
    raise AttributeError(
        f"Cannot find cann_bench.{func_name} or torch.ops.cann_bench.{func_name}"
    )


def _call_builtin_candidate(
    candidate,
    operator: str,
    ctx: Dict[str, Any],
    inputs: Dict[str, Any],
    attrs: Dict[str, Any],
):
    if operator == "MatmulReduceScatter":
        return _call_matmul_reduce_scatter(candidate, ctx, inputs, attrs)
    if operator == "MatmulAllReduce":
        return _call_matmul_all_reduce(candidate, ctx, inputs, attrs)
    if operator == "AllGatherMatmul":
        return _call_all_gather_matmul(candidate, ctx, inputs, attrs)
    raise RuntimeError(f"{operator} must provide mc2_call_candidate in task golden.py")


def _call_matmul_reduce_scatter(
    candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    try:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            reduce_op=attrs.get("reduce_op", "sum"),
            bias=inputs.get("bias"),
            is_trans_b=bool(attrs.get("is_trans_b", False)),
        )
    except TypeError:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            attrs.get("reduce_op", "sum"),
            inputs.get("bias"),
            bool(attrs.get("is_trans_b", False)),
        )


def _call_matmul_all_reduce(
    candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    try:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            reduce_op=attrs.get("reduce_op", "sum"),
            bias=inputs.get("bias"),
            is_trans_b=bool(attrs.get("is_trans_b", False)),
        )
    except TypeError:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            attrs.get("reduce_op", "sum"),
            inputs.get("bias"),
            bool(attrs.get("is_trans_b", False)),
        )


def _call_all_gather_matmul(
    candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    try:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            bias=inputs.get("bias"),
            gather_output=bool(attrs.get("gather_output", False)),
            is_trans_b=bool(attrs.get("is_trans_b", False)),
        )
    except TypeError:
        return candidate(
            inputs["x1"],
            inputs["x2"],
            ctx["hcomm_info"],
            ctx["world_size"],
            inputs.get("bias"),
            bool(attrs.get("gather_output", False)),
            bool(attrs.get("is_trans_b", False)),
        )


def _measure(
    fn: Callable, dist, device: torch.device, warmup: int, repeat: int
) -> float:
    for _ in range(max(0, warmup)):
        dist.barrier()
        fn()
        torch.npu.synchronize(device)

    dist.barrier()
    start = time.perf_counter()
    for _ in range(max(1, repeat)):
        fn()
    torch.npu.synchronize(device)
    dist.barrier()
    return (time.perf_counter() - start) * 1_000_000 / max(1, repeat)


def _profile_candidate_rank0(
    fn: Callable,
    dist,
    device: torch.device,
    case_payload: Dict[str, Any],
    rank: int,
    warmup: int,
    repeat: int,
) -> Optional[Dict[str, Any]]:
    """Profile rank0 while all ranks execute the same collective calls."""
    rel_path = case_payload["rel_path"]
    case_num = str(case_payload["case_num"])
    prof_dir = os.path.join(
        case_payload["reports_dir"], "prof_data", rel_path, case_num, f"rank{rank}"
    )
    if rank == 0:
        os.makedirs(prof_dir, exist_ok=True)
        result = PerfResult()
        try:
            _profile_rank0_loop(
                fn,
                prof_dir,
                warmup,
                repeat,
                case_payload.get("profiler_level", "Level1"),
            )
        except Exception as exc:
            # Replaying a whole loop after a partial profiler failure would
            # issue extra collectives on rank0 and deadlock the other ranks.
            raise RuntimeError(f"MC2 rank0 profiling failed: {exc}") from exc
        dist.barrier()
        # profiling 数据已落盘到 prof_dir 归档供事后分析；此处不解析 kernel_details.csv：
        # MC2 评分用各 rank 的 wall-clock elapsed_us（来自 _measure），不依赖 op_times。
        csv_file = _find_kernel_details_csv(prof_dir)
        if csv_file is None and result.error_msg is None:
            result.error_msg = "no kernel_details.csv produced"
        return {
            "elapsed_us": float(result.elapsed_us),
            "op_times": result.op_times,
            "error": result.error_msg,
            "_repeat": repeat,
            "warmup_used": False,
        }

    for _ in range(max(0, warmup) + max(1, repeat)):
        fn()
        torch.npu.synchronize(device)
    dist.barrier()
    return None


def _profile_rank0_loop(
    fn: Callable, prof_dir: str, warmup: int, repeat: int, profiler_level: str
) -> None:
    import torch_npu

    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
    os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

    level_map = {
        "Level1": torch_npu.profiler.ProfilerLevel.Level1,
        "Level2": torch_npu.profiler.ProfilerLevel.Level2,
    }
    level = level_map.get(profiler_level, torch_npu.profiler.ProfilerLevel.Level1)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=level,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    )

    original_basic_config = logging.basicConfig
    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)
    sink_file = tempfile.NamedTemporaryFile(
        mode="w+", prefix="kernel_eval_mc2_profiler_", suffix=".log", delete=False
    )
    try:

        def _silent_basic_config(**kwargs):
            kwargs["level"] = logging.ERROR
            kwargs["force"] = True
            return original_basic_config(**kwargs)

        logging.basicConfig = _silent_basic_config
        os.dup2(sink_file.fileno(), 1)
        os.dup2(sink_file.fileno(), 2)
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=warmup, active=repeat, repeat=1
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
        ) as prof:
            for _ in range(max(0, warmup) + max(1, repeat)):
                fn()
                prof.step()
        try:
            from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import (
                MultiProcessPool,
            )

            MultiProcessPool().close_pool(wait=True)
        except Exception:
            pass
    finally:
        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        sink_file.close()
        logging.basicConfig = original_basic_config
        try:
            os.unlink(sink_file.name)
        except OSError:
            pass


def _find_kernel_details_csv(prof_dir: str) -> Optional[str]:
    direct = os.path.join(prof_dir, "kernel_details.csv")
    if os.path.isfile(direct):
        return direct
    for root, _, files in os.walk(prof_dir):
        if "kernel_details.csv" in files:
            return os.path.join(root, "kernel_details.csv")
    return None


def _rank_output_results(compare_result, rank: int) -> List[Dict[str, Any]]:
    """将本 rank 的逐输出结果转为可跨进程序列化的 dict（父进程用 from_dict 还原）。"""
    result = []
    for sr in compare_result.output_results:
        sr.name = f"rank{rank}:output{sr.index}"
        result.append(_convert_to_output_result(sr).to_dict())
    return result


def _threshold_for_dtype(dtype: str, custom_thresholds: Dict[str, float]) -> float:
    if dtype in custom_thresholds:
        return float(custom_thresholds[dtype])
    return float(PRECISION_THRESHOLDS.get(dtype, 0.001))
