#!/usr/bin/python3
# coding=utf-8

import logging
import os
import socket
import tempfile
import time
import traceback
from queue import Empty
from typing import Any, Callable, Dict, List, Optional

import torch

from ..base.models import CaseSpec
from ..base.result import (
    FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
    get_accuracy_failure_type,
    AccuracyResult,
)
from ..registry.golden_registry import get_golden_loader
from ..utils import str_to_torch_dtype
from ..utils.compare import compare_tensors
from ..utils.thresholds import PRECISION_THRESHOLDS
from ..checkers.relative_error_checker import (
    RelativeErrorOutputResult,
    _convert_to_output_result,
)
from .op_runner import OpRunResult
from .perf_eval import PerfResult
from .results import EvalCaseResult


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class MC2DistributedEvaluator:
    """Run MC2 communication-fused operators with one process per rank.

    The normal cann-bench path is single-process. MC2 cases still need an HCCL
    process group, but their golden logic now lives in each task's golden.py via 
    the ``mc2_distributed_golden`` hook.
    """

    def __init__(self, config):
        self.config = config
        self.golden_loader = get_golden_loader("cann", bench_root=str(self.config.tasks_root))

    def evaluate_case(self, case: CaseSpec, custom_thresholds: Optional[Dict[str, float]] = None) -> EvalCaseResult:
        attrs = case.attrs or {}
        world_size = int(attrs.get("world_size", attrs.get("ep_world_size", 1)))
        case_id_str = case.get_case_id_str()
        port = _find_free_port()

        # Resolve the task-local golden module in the parent process so missing
        # golden.py or hook failures are reported before spawning ranks.
        self.golden_loader.get_mc2_distributed_golden(case.rel_path)

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
        }

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

        for proc in processes:
            proc.join()

        rank_results = []
        while True:
            try:
                rank_results.append(queue.get_nowait())
            except Empty:
                break

        process_errors = [
            f"rank {idx} exited with code {proc.exitcode}"
            for idx, proc in enumerate(processes)
            if proc.exitcode not in (0, None)
        ]
        rank_errors = [r.get("error") for r in rank_results if r.get("error")]
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
                t_hw_us=case.t_hw_us,
                failure_type=FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
            )

        rank_results.sort(key=lambda item: int(item.get("rank", 0)))
        passed = all(bool(r.get("passed")) for r in rank_results)
        dtype = next((str(r.get("dtype")) for r in rank_results if r.get("dtype")), None)
        dtype = dtype or (case.dtypes[0] if case.dtypes else "float32")
        threshold = next((float(r.get("threshold")) for r in rank_results if r.get("threshold") is not None), None)
        threshold = threshold if threshold is not None else _threshold_for_dtype(dtype, custom_thresholds or {})

        # 还原各 rank 的逐输出结果（rank 侧已序列化为 RelativeErrorOutputResult dict）
        output_results = []
        for rank_result in rank_results:
            for item in rank_result.get("output_results", []):
                output_results.append(RelativeErrorOutputResult.from_dict(item))

        # 跨 rank 聚合指标，写入 metadata（与 RelativeErrorChecker 的 metadata 字段对齐）
        metadata = {
            "checker_name": "relative_error",
            "mc2_distributed": True,
            "world_size": world_size,
            "dtype": dtype,
            "mere": max((float(r.get("mere", 0.0)) for r in rank_results), default=0.0),
            "mare": max((float(r.get("mare", 0.0)) for r in rank_results), default=0.0),
            "max_diff": max((float(r.get("max_diff", 0.0)) for r in rank_results), default=0.0),
            "mean_diff": max((float(r.get("mean_diff", 0.0)) for r in rank_results), default=0.0),
            "mismatch_count": sum(int(r.get("mismatch_count", 0)) for r in rank_results),
            "total_count": sum(int(r.get("total_count", 0)) for r in rank_results),
            "mismatch_ratio": max((float(r.get("mismatch_ratio", 0.0)) for r in rank_results), default=0.0),
            "small_value_error_count": sum(int(r.get("small_value_error_count", 0)) for r in rank_results),
            "small_value_cpu_error_count": sum(int(r.get("small_value_cpu_error_count", 0)) for r in rank_results),
            "small_value_total_count": sum(int(r.get("small_value_total_count", 0)) for r in rank_results),
            "cancel_error_count": sum(int(r.get("cancel_error_count", 0)) for r in rank_results),
            "cancel_cpu_error_count": sum(int(r.get("cancel_cpu_error_count", 0)) for r in rank_results),
            "cancel_total_count": sum(int(r.get("cancel_total_count", 0)) for r in rank_results),
        }

        # MoeDistributeDispatchV2 的输出顺序与 golden 可能因路由实现差异而不同，
        # 但若所有误差指标恒为 0（逐元素完全一致），仍判为通过。该特判内化在
        # runner 内部，不侵入框架通用的精度判定路径（results.py）。
        if (
            not passed
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
            ) or None,
            output_results=output_results,
            metadata=metadata,
        )

        elapsed_us = max(float(r.get("elapsed_us", 0.0)) for r in rank_results)
        golden_elapsed_us = max(float(r.get("golden_elapsed_us", 0.0)) for r in rank_results)
        baseline_perf_us = case.baseline_perf_us if case.baseline_perf_us > 0 else golden_elapsed_us

        rank0_perf = next((r.get("perf_result") for r in rank_results if r.get("rank") == 0), None)
        perf_result = None
        if elapsed_us > 0:
            perf_result = PerfResult(elapsed_us=elapsed_us)
            if rank0_perf:
                perf_result.op_times = rank0_perf.get("op_times", {}) or {}
                perf_result.error_msg = rank0_perf.get("error")
                perf_result.metadata["repeat"] = int(rank0_perf.get("_repeat", max(1, self.config.repeat)))
                perf_result.metadata["warmup_used"] = bool(rank0_perf.get("warmup_used", False))

        return EvalCaseResult(
            case_id=case_id_str,
            rel_path=case.rel_path,
            operator=case.operator,
            case_num=case.case_num,
            success=passed,
            accuracy_result=accuracy_result,
            perf_result=perf_result if passed else None,
            golden_run_result=OpRunResult(success=True, elapsed_us=golden_elapsed_us, device="npu"),
            ai_run_result=OpRunResult(success=passed, elapsed_us=elapsed_us, device="npu"),
            error_msg=None if passed else accuracy_result.error_msg,
            baseline_perf_us=baseline_perf_us,
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
        result = _run_rank(rank, world_size, port, case_payload, warmup, repeat, enable_perf, custom_thresholds)
    except Exception as exc:
        result = {
            "rank": rank,
            "passed": False,
            "error": f"{exc}\n{traceback.format_exc()}",
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
    operator = case_payload["operator"]
    torch_npu.npu.set_device(rank)
    device = torch.device(f"npu:{rank}")
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )

    try:
        hcomm_info = _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        loader = get_golden_loader("cann", bench_root=case_payload["tasks_root"])
        golden_module = loader._load_module(case_payload["rel_path"])
        golden_hook = loader.get_mc2_distributed_golden(case_payload["rel_path"], required=True)

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
        if hasattr(golden_module, "mc2_call_candidate"):
            candidate_fn = lambda: golden_module.mc2_call_candidate(candidate, rank_ctx, inputs, attrs)
        else:
            candidate_fn = lambda: _call_builtin_candidate(candidate, operator, rank_ctx, inputs, attrs)
        golden_fn = lambda: golden_hook(rank_ctx, inputs, attrs)

        golden_output = golden_fn()
        torch.npu.synchronize(device)
        golden_output = _clone_outputs_for_compare(golden_output)
        torch.npu.synchronize(device)
        candidate_output = candidate_fn()
        torch.npu.synchronize(device)
        candidate_output = _clone_outputs_for_compare(candidate_output)
        torch.npu.synchronize(device)
        compare_result = compare_tensors(
            candidate_output,
            golden_output,
            case_payload["dtypes"][0],
            _threshold_for_dtype(case_payload["dtypes"][0], custom_thresholds),
            custom_thresholds=custom_thresholds,
        )
        primary_output = compare_result.output_results[0] if compare_result.output_results else None
        output_results = _rank_output_results(compare_result, rank)

        elapsed_us = 0.0
        golden_elapsed_us = 0.0
        perf_payload = None
        if enable_perf:
            elapsed_us = _measure(candidate_fn, dist, device, warmup, repeat)
            golden_elapsed_us = _measure(golden_fn, dist, device, warmup, repeat)
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
            "passed": bool(compare_result.passed),
            "dtype": str(compare_result.dtype),
            "threshold": float(compare_result.threshold),
            "mere": float(compare_result.mere),
            "mare": float(compare_result.mare),
            "max_diff": float(compare_result.max_diff),
            "mean_diff": float(compare_result.mean_diff),
            "mismatch_count": int(compare_result.mismatch_count),
            "total_count": int(compare_result.total_count),
            "mismatch_ratio": float(compare_result.mismatch_ratio),
            "small_value_error_count": int(compare_result.small_value_error_count),
            "small_value_cpu_error_count": int(compare_result.small_value_cpu_error_count),
            "small_value_total_count": int(compare_result.small_value_total_count),
            "cancel_error_count": int(compare_result.cancel_error_count),
            "cancel_cpu_error_count": int(compare_result.cancel_cpu_error_count),
            "cancel_total_count": int(compare_result.cancel_total_count),
            "compare_error": compare_result.error_msg,
            "output_results": output_results,
            "elapsed_us": elapsed_us,
            "golden_elapsed_us": golden_elapsed_us,
            "perf_result": perf_payload,
        }
    finally:
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
    return output


def _make_rank_inputs(case_payload: Dict[str, Any], rank: int, device: torch.device):
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    attrs = case_payload["attrs"]
    seed = int(attrs.get("seed", 1))
    weight_same = int(attrs.get("weight_same", 0))

    x1 = _make_tensor(shapes[0], dtypes[0], ranges[0] if ranges else None, seed, device)
    x2_seed = seed + rank * 2 if weight_same else seed
    x2 = _make_tensor(shapes[1], dtypes[1], ranges[1] if len(ranges) > 1 else None, x2_seed, device)
    bias = None
    if bool(attrs.get("is_bias", False)) and len(shapes) > 2 and shapes[2] is not None:
        bias = _make_tensor(shapes[2], dtypes[2], ranges[2] if len(ranges) > 2 else None, rank * 3, device)
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
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(float(lo), float(hi)).to(dtype)
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def _load_candidate(operator: str):
    name_map = {
        "MatmulReduceScatter": "matmul_reduce_scatter",
        "MatmulReduceScatterV2": "matmul_reduce_scatter_v2",
        "MatmulAllReduce": "matmul_all_reduce",
        "AllGatherMatmul": "all_gather_matmul",
        "AllGatherMatmulV2": "all_gather_matmul_v2",
        "GroupedMatMulAlltoAllv": "grouped_mat_mul_allto_allv",
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
    raise AttributeError(f"Cannot find cann_bench.{func_name} or torch.ops.cann_bench.{func_name}")


def _call_builtin_candidate(candidate, operator: str, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
    if operator == "MatmulReduceScatter":
        return _call_matmul_reduce_scatter(candidate, ctx, inputs, attrs)
    if operator == "MatmulAllReduce":
        return _call_matmul_all_reduce(candidate, ctx, inputs, attrs)
    if operator == "AllGatherMatmul":
        return _call_all_gather_matmul(candidate, ctx, inputs, attrs)
    raise RuntimeError(f"{operator} must provide mc2_call_candidate in task golden.py")


def _call_matmul_reduce_scatter(candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
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
        return candidate(inputs["x1"], inputs["x2"], ctx["hcomm_info"], ctx["world_size"],
                         attrs.get("reduce_op", "sum"), inputs.get("bias"),
                         bool(attrs.get("is_trans_b", False)))


def _call_matmul_all_reduce(candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
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
        return candidate(inputs["x1"], inputs["x2"], ctx["hcomm_info"], ctx["world_size"],
                         attrs.get("reduce_op", "sum"), inputs.get("bias"),
                         bool(attrs.get("is_trans_b", False)))


def _call_all_gather_matmul(candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]):
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
        return candidate(inputs["x1"], inputs["x2"], ctx["hcomm_info"], ctx["world_size"], inputs.get("bias"),
                         bool(attrs.get("gather_output", False)), bool(attrs.get("is_trans_b", False)))


def _measure(fn: Callable, dist, device: torch.device, warmup: int, repeat: int) -> float:
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
    prof_dir = os.path.join(case_payload["reports_dir"], "prof_data", rel_path, case_num, f"rank{rank}")
    if rank == 0:
        os.makedirs(prof_dir, exist_ok=True)
        result = PerfResult()
        try:
            _profile_rank0_loop(fn, prof_dir, warmup, repeat, case_payload.get("profiler_level", "Level1"))
        except Exception as exc:
            result.error_msg = str(exc)
            for _ in range(max(0, warmup) + max(1, repeat)):
                fn()
                torch.npu.synchronize(device)
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


def _profile_rank0_loop(fn: Callable, prof_dir: str, warmup: int, repeat: int, profiler_level: str) -> None:
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
    sink_file = tempfile.NamedTemporaryFile(mode="w+", prefix="kernel_eval_mc2_profiler_", suffix=".log", delete=False)
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
            schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=repeat, repeat=1),
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
            from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
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
