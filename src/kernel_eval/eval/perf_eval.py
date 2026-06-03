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

"""
性能评测器

职责：
1. NPU 模式下使用 torch_npu.profiler 采集性能数据
2. 支持 NPU 升频 + L2 cache 清空，保证测量一致性
3. 非 profiler 路径不做墙钟计时，perf_result 由调用侧置空
4. 按 CANN_BENCH_PERF_SOURCE 解析 kernel_details.csv 或 trace_view.json
5. 归档 profiling 中间目录到 reports/prof_data/{rel_path}/{caseid}/

参考evaluation/core/profiler_manager.py
"""

import csv
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import traceback
from typing import Optional, Dict, Any, Tuple, List, Callable

import torch


_logger = logging.getLogger(__name__)

from ..utils.device_manager import DeviceManager
from ..config import Config, get_config
from .input_pool import InputPool
from ..base.result import PerfResult, compute_speedup


PERF_SOURCE_ENV = "CANN_BENCH_PERF_SOURCE"
PERF_SOURCE_KERNEL_DETAILS = "kernel_details"
PERF_SOURCE_TRACE_VIEW = "trace_view"

# Warmup kernel 精确形状特征（用于过滤）
# F111: 硬编码形状会在 CANN / 驱动升级改 warmup 实现时静默失效（过滤命中率=0
# 意味着 warmup 时间混入了实测，性能数据偏低）。允许 env 覆盖 +
# _maybe_warn_warmup_filter_inactive 校验整体命中率。
WARMUP_MATMUL_SHAPE = os.environ.get(
    "CANN_BENCH_WARMUP_MATMUL_SHAPE", '"10240,10240;10240,10240"'
)
WARMUP_REDUCE_SHAPE = os.environ.get(
    "CANN_BENCH_WARMUP_REDUCE_SHAPE", '"96,1024,1024;3"'
)

# 跟踪 warmup 过滤命中率：在某轮评测开始时清零，结束时检查
_WARMUP_FILTER_STATS = {"checked": 0, "matched": 0}


def _maybe_warn_warmup_filter_inactive() -> None:
    """F111: 若过滤被频繁调用但命中率=0，提示 warmup 形状可能已过期。

    调用方在每轮评测结束时调用。only warn once per process。
    """
    if _WARMUP_FILTER_STATS["checked"] >= 20 and _WARMUP_FILTER_STATS["matched"] == 0:
        if not getattr(_maybe_warn_warmup_filter_inactive, "_warned", False):
            _logger.warning(
                "perf_eval: warmup-shape filter checked %d kernels but matched 0; "
                "CANN/驱动可能已升级 warmup 形状（旧值 MATMUL=%r / REDUCE=%r）。"
                "建议设置 env CANN_BENCH_WARMUP_MATMUL_SHAPE / "
                "CANN_BENCH_WARMUP_REDUCE_SHAPE 覆盖。",
                _WARMUP_FILTER_STATS["checked"], WARMUP_MATMUL_SHAPE, WARMUP_REDUCE_SHAPE,
            )
            _maybe_warn_warmup_filter_inactive._warned = True


def _perf_source_from_env() -> str:
    raw = os.environ.get(PERF_SOURCE_ENV, PERF_SOURCE_KERNEL_DETAILS)
    value = str(raw).strip().lower().replace("-", "_")
    if value in ("", "kernel", "kernel_detail", PERF_SOURCE_KERNEL_DETAILS, "csv"):
        return PERF_SOURCE_KERNEL_DETAILS
    if value in ("trace", PERF_SOURCE_TRACE_VIEW, "pypto"):
        return PERF_SOURCE_TRACE_VIEW

    _logger.warning(
        "unsupported %s=%r; falling back to %s",
        PERF_SOURCE_ENV,
        raw,
        PERF_SOURCE_KERNEL_DETAILS,
    )
    return PERF_SOURCE_KERNEL_DETAILS


# ---------------------------------------------------------------------------
# 独立 profiling 辅助（供 parse_trace_view_prof 运行算子 + 采集用）
# ---------------------------------------------------------------------------

def _profile_standalone(fn, prof_dir: str, warmup: int, repeat: int) -> None:
    """Run *fn* with torch_npu.profiler, writing trace output to *prof_dir*.

    This is a self-contained profiling helper that does NOT depend on
    PerfEvaluator instance state (no freq_boost, no warmup tensors).
    """
    import logging
    import torch_npu

    # Suppress profiler parser logs
    og_basicConfig = logging.basicConfig
    logging.basicConfig = lambda **kw: og_basicConfig(**{**kw, "level": logging.ERROR, "force": True})
    try:
        for name in ['', 'torch', 'torch_npu', 'torch_npu.profiler', 'ascend', 'profiler']:
            lg = logging.getLogger(name)
            lg.setLevel(logging.ERROR)
            lg.handlers = []
            lg.addHandler(logging.NullHandler())

        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        sink_file = tempfile.NamedTemporaryFile(
            mode='w+', prefix='trace_profiler_', suffix='.log', delete=False
        )
        sink_fd = sink_file.fileno()

        try:
            os.dup2(sink_fd, 1)
            os.dup2(sink_fd, 2)

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            )

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
                fn_exc = None
                for i in range(warmup + repeat):
                    try:
                        fn()
                    except BaseException as e:
                        fn_exc = e
                        prof.step()
                        break
                    prof.step()
                if fn_exc is not None:
                    raise fn_exc

            # Wait for profiler async parsing
            try:
                from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
                pool = MultiProcessPool()
                pool.close_pool(wait=True)
            except Exception:
                pass

        finally:
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            logging.basicConfig = og_basicConfig
            sink_file.close()
            try:
                os.unlink(sink_file.name)
            except OSError:
                pass

    finally:
        logging.basicConfig = og_basicConfig


class PerfEvaluator:
    """NPU 性能评测器

    使用 torch_npu.profiler 采集 NPU 性能数据。
    默认使用 Level1（47列CSV），支持 Level2 配置。
    通过解析 kernel_details.csv 获取设备内核时间，使用精确形状匹配过滤 warmup kernel。
    每次测量前执行 MatMul + ReduceMax 升频并清空 L2 cache。

    使用方法：
        config = Config(profiler_level="Level1")
        perf_eval = PerfEvaluator(config=config, device_manager=device_mgr)
        outputs, perf_result = perf_eval.run_profiled(case_id, func, *args)
    """

    def __init__(self, config: Config = None, device_manager: DeviceManager = None,
                 warmup: int = 3, repeat: int = 5, archive_prof: bool = True,
                 freq_boost: bool = True):
        """
        Args:
            config: 配置对象（含 profiler_level）
            device_manager: 设备管理器
            warmup: 预热次数
            repeat: 采集次数
            archive_prof: 是否归档profiling数据
            freq_boost: 是否启用NPU升频清cache
        """
        self.config = config or get_config()
        self.device_manager = device_manager
        self.warmup = warmup
        self.repeat = repeat
        self.archive_prof = archive_prof
        self.freq_boost = freq_boost

        # 性能数据归档目录
        self.prof_data_dir = os.path.join(self.config.reports_dir, "prof_data")

        # Warmup tensors（升频清cache）
        self._warmup_tensors: Optional[Tuple] = None

    def _prepare_warmup_tensors(self):
        """准备升频清cache的tensors

        Matrix + reduce tensors sized to cover typical AI-core/L2 footprints.
        Pinned to the configured NPU — bare ``.npu()`` would go to current
        device 0 and either hijack the wrong card or fail outright when the
        runner is using a different device.
        """
        if self._warmup_tensors is None and self.freq_boost:
            device = (self.device_manager.get_device()
                      if self.device_manager is not None else "npu")
            mm1 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
            mm2 = torch.rand((10240, 10240), dtype=torch.float16).to(device)
            reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).to(device)
            self._warmup_tensors = (mm1, mm2, reduce_input)

    def _boost_freq_and_clear_cache(self):
        """NPU升频 + 清L2 cache (仅在测量窗口前调用一次)

        执行 MatMul + ReduceMax 以：
        1. 提升 NPU 频率到稳定状态
        2. 清空 L2 cache，保证测量一致性

        Sync targets the warmup tensor's actual device — ``torch.npu.synchronize()``
        with no arg syncs the current device, which can disagree with the
        device the warmup tensors live on.

        Wrapped in ``TorchOpGuard.pause()`` so warmup matmul does not trip
        the guard's forbidden-API counter (the entire run_ai_op is wrapped
        in a guard upstream; warmup is not candidate computation).
        """
        if self._warmup_tensors is not None:
            from ..security.torch_op_guard import TorchOpGuard
            with TorchOpGuard.pause():
                mm1, mm2, reduce_input = self._warmup_tensors
                torch.matmul(mm1, mm2)
                torch.npu.synchronize(mm1.device)
                torch.max(reduce_input)
                torch.npu.synchronize(mm1.device)

    def _clear_cache(self):
        """清空 L2 cache (在每次测量 step 前调用，保证测量间 cache 状态一致)

        Wrapped in ``TorchOpGuard.pause()`` for the same reason as
        ``_boost_freq_and_clear_cache``.
        """
        if self._warmup_tensors is not None:
            from ..security.torch_op_guard import TorchOpGuard
            with TorchOpGuard.pause():
                _, _, reduce_input = self._warmup_tensors
                torch.max(reduce_input)
                torch.npu.synchronize(reduce_input.device)

    def _profile(self, fn: Callable, prof_dir: str, warmup: int, repeat: int):
        """Execute warmup + repeat calls with NPU profiler.

        使用 config.profiler_level（Level1 或 Level2）采集性能数据。
        Level1/Level2 产出 kernel_details.csv（47列），包含 Input Shapes 用于精确过滤。

        性能优化：频率提升仅在测量窗口前执行一次（而非每个 step），
        L2 cache 清理仅在测量 step 前执行（warmup step 跳过）。
        原先 (warmup+repeat) × (MatMul+ReduceMax) → 1 × (MatMul+ReduceMax) + repeat × ReduceMax。
        """
        import logging
        import os
        import sys
        import torch_npu

        # Suppress profiler parser logs via multiple mechanisms:
        # 1. Set environment variables before any process spawns
        os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
        os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'

        # 2. Monkey-patch logging.basicConfig to force ERROR level
        original_basicConfig = logging.basicConfig
        def _silent_basicConfig(**kwargs):
            kwargs['level'] = logging.ERROR
            kwargs['force'] = True
            return original_basicConfig(**kwargs)
        logging.basicConfig = _silent_basicConfig

        # 3. Pre-configure all loggers
        for name in ['', 'torch', 'torch_npu', 'torch_npu.profiler', 'ascend', 'profiler']:
            lg = logging.getLogger(name)
            lg.setLevel(logging.ERROR)
            lg.handlers = []
            lg.addHandler(logging.NullHandler())

        # 获取 profiler_level，支持 Level1 和 Level2
        profiler_level = getattr(self.config, 'profiler_level', 'Level1')
        level_map = {
            'Level1': torch_npu.profiler.ProfilerLevel.Level1,
            'Level2': torch_npu.profiler.ProfilerLevel.Level2,
        }
        level = level_map.get(profiler_level, torch_npu.profiler.ProfilerLevel.Level1)

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=level,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        )

        # 频率提升 + 初始 cache 清理（仅在测量窗口前执行一次）
        if self.freq_boost:
            self._boost_freq_and_clear_cache()

        # Save original stdout/stderr file descriptors
        # 使用 os.dup2 在系统级别重定向，影响所有子进程
        # 重定向到 tempfile 而非 /dev/null：profiler 退出后扫描其中的 NPU
        # 驱动 / Runtime 错误（AICPU 异常 / Tiling 错误等），避免静默丢失。
        import tempfile
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        sink_file = tempfile.NamedTemporaryFile(
            mode='w+', prefix='kernel_eval_profiler_', suffix='.log', delete=False
        )
        sink_fd = sink_file.fileno()

        try:
            # Redirect stdout and stderr to the temp file (NOT /dev/null)
            os.dup2(sink_fd, 1)
            os.dup2(sink_fd, 2)

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
                # F032: fn() exceptions used to escape the with-block, so
                # `prof.__exit__` ran in an unfinished state — step counters
                # mismatched and the kernel-details CSV could be truncated
                # or never written. Catch per-iteration, advance the step
                # counter so the profiler exits cleanly, then re-raise so
                # the caller surfaces the failure normally.
                fn_exc: Optional[BaseException] = None
                for i in range(warmup + repeat):
                    if self.freq_boost and i >= warmup:
                        self._clear_cache()
                    try:
                        fn()
                    except BaseException as e:
                        fn_exc = e
                        prof.step()
                        break
                    prof.step()
                if fn_exc is not None:
                    raise fn_exc

            # 等待 profiler 解析完成（在恢复 stdout/stderr 之前）
            # 解析器进程在 profiler context 退出后开始工作
            try:
                from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
                pool = MultiProcessPool()
                pool.close_pool(wait=True)  # 等待解析完成
            except Exception as e:
                # F047: 不再静默吞 — debug 级日志（用户/CI 通过 LOG_LEVEL 控制）
                _logger.debug("MultiProcessPool close_pool(wait=True) failed: %s", e)

        finally:
            # Restore original stdout/stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            sink_file.close()
            logging.basicConfig = original_basicConfig

            # 扫描 profiler 期间被捕获的 stderr/stdout，找出 NPU 关键错误关键词
            try:
                with open(sink_file.name, 'r', errors='replace') as f:
                    captured = f.read()
                # F110: 旧版只匹配已知关键词（AICPU/EZ1001 等），新型 NPU 错误
                # (OOM, Malloc failed, timeout 等) 会被静默删除。补通用正则兜底，
                # 任一含 error/fail/exception/traceback 关键词的行都计入 hits。
                _NPU_KNOWN_ERRORS = ('AICPU exception', 'Inner error', 'Runtime error',
                                     'EZ1001', 'EZ9999', 'aicore error',
                                     'kernel launch failed', 'failed to launch')
                _GENERIC_ERROR_RE = re.compile(
                    r'\b(error|fail(?:ed|ure)?|exception|traceback|malloc|oom|timeout)\b',
                    re.IGNORECASE,
                )
                hits = []
                for line in captured.splitlines():
                    if any(kw.lower() in line.lower() for kw in _NPU_KNOWN_ERRORS):
                        hits.append(line)
                    elif _GENERIC_ERROR_RE.search(line):
                        hits.append(line)
                if hits:
                    print(f"[WARN] Profiler 期间捕获 NPU 关键错误（{len(hits)} 条），完整日志: {sink_file.name}", flush=True)
                    for h in hits[:5]:
                        print(f"    {h.strip()}", flush=True)
                else:
                    # 无错误：直接删 tempfile
                    try:
                        os.unlink(sink_file.name)
                    except OSError as e:
                        # F047: tempfile 清理失败也 log 一下
                        _logger.debug("tempfile unlink %s failed: %s", sink_file.name, e)
            except Exception as e:
                # F047: 不再静默吞 — 读 sink_file 失败时至少留痕
                _logger.debug("perf_eval profiler-log scan failed: %s", e)

    def run_profiled(self, case_id: str, func: Callable, *args,
                     warmup: int = None, repeat: int = None,
                     inputs: List = None, use_input_pool: bool = False,
                     **kwargs) -> Tuple[Any, PerfResult]:
        """Profile func with warmup + repeat steps, return (outputs, result).

        NPU path uses torch_npu.profiler (Level1/Level2)，解析 kernel_details.csv。
        CPU fallback measures wall-clock time when ``self.config.enable_profiler`` is False.

        Args:
            case_id: case identifier (used in PerfResult and archive path).
            func: the callable under test.
            *args, **kwargs: forwarded to func.
            warmup: warmup steps (default: instance setting).
            repeat: measurement steps (default: instance setting).
            inputs: tensor list for InputPool rotation (prevents data-ptr
                    caching).  Ignored unless *use_input_pool* is True.
            use_input_pool: cycle inputs through InputPool.

        Returns:
            (last_outputs, PerfResult) — op_times / elapsed_us 直接填充完毕。
        """
        warmup = warmup or self.warmup
        repeat = repeat or self.repeat

        result = PerfResult(
            metadata={
                'case_id': case_id,
                '_repeat': repeat,
                'warmup_used': self.freq_boost,
                'freq_boost': self.freq_boost,
            }
        )

        if not self.config.enable_profiler:
            # 不应到达此处(run_profiled 仅在 enable_profiler 时被调用);保留防御性早退,
            # 不做墙钟计时——非 profiler 路径由 op_runner 直接产出 perf_result=None。
            return None, result

        if self.freq_boost:
            self._prepare_warmup_tensors()

        rel_path, caseid = self._parse_case_id(case_id)
        if self.archive_prof:
            prof_dir = os.path.join(self.prof_data_dir, rel_path, caseid)
            os.makedirs(prof_dir, exist_ok=True)
            # 清理上次评测遗留的时间戳子目录，避免读取到脏数据
            try:
                for entry in os.listdir(prof_dir):
                    entry_path = os.path.join(prof_dir, entry)
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path, ignore_errors=True)
            except OSError as e:
                _logger.debug("profiler archive dir cleanup skipped for %s: %s", prof_dir, e)
        else:
            prof_dir = tempfile.mkdtemp(prefix="cann_prof_")

        last_outputs = None

        try:
            # 注意：torch_npu.profiler 内部使用全局单例 ProcessPoolExecutor
            # 多线程并发调用会导致 Bus error，因此：
            # - 多线程模式应使用 _measure_simple（简单计时）
            # - 单线程/进程隔离模式可使用 profiler
            #
            # 当前实现：当 enable_profiler=True 时使用 profiler（需确保单线程执行）
            # 多线程并行评测应设置 enable_profiler=False
            # profiler parser logs 已在 _profile 内部抑制
            if inputs and use_input_pool:
                pool = InputPool(inputs, warmup + repeat)
                def _fn():
                    nonlocal last_outputs
                    last_outputs = func(*pool.get_next())
            else:
                def _fn():
                    nonlocal last_outputs
                    last_outputs = func(*args, **kwargs)

            self._profile(_fn, prof_dir, warmup, repeat)

            # Clean up InputPool now (trace already written to disk).
            if inputs and use_input_pool:
                pool.clear()

        except Exception as e:
            result.error_msg = f"{type(e).__name__}: {e}"
            result.metadata["profile_exception_type"] = type(e).__name__
            result.metadata["profile_exception"] = str(e)
            result.metadata["profile_exception_traceback"] = traceback.format_exc()

        try:
            perf_source = _perf_source_from_env()
            result.metadata["perf_source"] = perf_source

            if perf_source == PERF_SOURCE_TRACE_VIEW:
                if not self._normalize_trace_view_result(result, prof_dir) and not result.error_msg:
                    result.error_msg = "no valid trace_view metrics produced"
            else:
                # Locate kernel_details.csv — check common locations first.
                csv_file = None

                # 1) Directly in prof_dir
                direct = os.path.join(prof_dir, "kernel_details.csv")
                if os.path.isfile(direct):
                    csv_file = direct
                else:
                    # 2) One level down (torch_npu wraps in a timestamped subdir)
                    try:
                        for entry in os.listdir(prof_dir):
                            candidate = os.path.join(prof_dir, entry, "kernel_details.csv")
                            if os.path.isfile(candidate):
                                csv_file = candidate
                                break
                    except OSError:
                        pass

                    # 3) Fallback: deeper walk (should rarely be needed)
                    if csv_file is None:
                        for root, dirs, files in os.walk(prof_dir):
                            for f in files:
                                if f == "kernel_details.csv":
                                    csv_file = os.path.join(root, f)
                                    break
                            if csv_file:
                                break

                if csv_file:
                    op_times, total_kernel_us = self._parse_kernel_details_csv(csv_file)
                    self._normalize_result(result, op_times, total_kernel_us)
                elif not result.error_msg:
                    result.error_msg = "no kernel_details.csv produced"

        finally:
            # Clean up temp dir (non-archive mode).
            if not self.archive_prof and os.path.isdir(prof_dir):
                try:
                    shutil.rmtree(prof_dir, ignore_errors=True)
                except OSError:
                    pass

        return last_outputs, result

    def _parse_case_id(self, case_id: str) -> Tuple[str, str]:
        """Parse case_id like ``level2/scatter_1`` into ``(rel_path, case_num)``.

        New format: {rel_path}_{case_num}
        Old format (fallback): L2_Scatter_1 -> (level2/Scatter, 1)

        Returns:
            (rel_path, case_num) for prof_data archive path.
        """
        # Try new format first: rel_path_case_num
        # e.g., "level2/scatter_1" -> ("level2/scatter", "1")
        parts = case_id.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], parts[1]

        # Fallback: old format L2_Scatter_1 -> level2/Scatter, 1
        m = re.match(r"^L(?P<level>\d+)_(?P<op>.+)_(?P<case>\d+)$", case_id)
        if m:
            return f"level{m['level']}/{m['op']}", m["case"]

        return case_id or "unknown", "0"

    def _normalize_result(self, result: PerfResult,
                           op_times: Dict[str, Dict[str, float]],
                           total_kernel_us: float):
        """Normalize op_times and populate result.

        Note: _parse_kernel_details_csv 已经返回中位数，无需再除以 repeat。
        """
        for cat in op_times:
            for name in op_times[cat]:
                op_times[cat][name] = round(op_times[cat][name], 2)
        total_kernel_us = round(total_kernel_us, 2)
        result.op_times = op_times
        result.elapsed_us = total_kernel_us
        result.metadata["elapsed_us_source"] = "kernel_details.total_kernel_us"

    def _normalize_trace_view_result(self, result: PerfResult, prof_dir: str) -> bool:
        trace_result = self.parse_trace_view_prof(prof_dir)
        prof_metrics = trace_result.get("prof", {}) if isinstance(trace_result, dict) else {}
        if not isinstance(prof_metrics, dict):
            return False

        try:
            elapsed_us = float(prof_metrics.get("aicore_e2e", 0))
        except (TypeError, ValueError):
            return False
        if elapsed_us <= 0:
            return False

        normalized_prof = {}
        for name, value in prof_metrics.items():
            if isinstance(value, (int, float)):
                normalized_prof[name] = round(float(value), 2)
            else:
                normalized_prof[name] = value

        result.op_times = {PERF_SOURCE_TRACE_VIEW: normalized_prof}
        result.elapsed_us = round(elapsed_us, 2)
        result.metadata["elapsed_us_source"] = "trace_view.aicore_e2e"
        return True

    def wait_all(self):
        """兼容旧接口，当前为同步解析，无需等待。"""

    def shutdown(self):
        """清理warmup tensors 并强制关闭 profiler 进程池"""
        if self._warmup_tensors is not None:
            del self._warmup_tensors
            self._warmup_tensors = None

        # 强制关闭 profiler ProcessPoolExecutor，不等待 fork 子进程
        try:
            from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
            pool = MultiProcessPool()
            # close_pool(wait=False) 不等待 fork 子进程完成，直接关闭
            pool.close_pool(wait=False)
        except Exception as e:
            # F047: 不再静默吞 — debug 日志便于排查 profiler shutdown 失败
            _logger.debug("MultiProcessPool close_pool(wait=False) failed: %s", e)

    def _parse_kernel_details_csv(self, csv_file: str) -> Tuple[Dict[str, Dict[str, float]], float]:
        """解析 kernel_details.csv，提取 NPU kernel 执行时间（取中位数）

        Level1/Level2 产出的 kernel_details.csv 包含 47 列，包括：
        - Step Id: 步骤ID（warmup阶段可能为空，需要过滤）
        - Type: 算子类型
        - Input Shapes: 输入形状（用于精确过滤 warmup kernel）
        - Duration(us): 执行时间

        Warmup kernel 过滤：
        1. 排除 Step Id 为空的 kernel（profiler 内部记录）
        2. 精确形状匹配 MatMulV3/ReduceMax 升频清cache kernel

        性能指标计算：
        - 对每个 kernel，收集所有 Step 的时间值，取中位数（减少异常值影响）
        - 总时间为所有 kernel 中位数之和
        """
        if not csv_file or not os.path.exists(csv_file):
            return {}, 0.0

        # 按 Step 分组收集每个 kernel 的时间
        step_kernel_times: Dict[str, Dict[str, List[float]]] = {}

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # 获取 Step Id，过滤无效/空值（warmup阶段残留）
                    step_id = row.get('Step Id', '').strip()
                    if not step_id:
                        continue

                    # 获取执行时间
                    duration_str = row.get('Duration(us)', '0')
                    try:
                        duration = float(duration_str)
                    except (ValueError, TypeError):
                        continue

                    if duration <= 0:
                        continue

                    # 获取算子名称和输入形状
                    op_type = row.get('Type', '')
                    input_shapes = row.get('Input Shapes', '')

                    # 精确过滤 warmup kernel（升频清cache）
                    if self._is_warmup_kernel(op_type, input_shapes):
                        continue

                    # 记录 kernel 名称
                    name = row.get('Name', op_type)

                    # 按 step 和 kernel 收集时间列表
                    if step_id not in step_kernel_times:
                        step_kernel_times[step_id] = {}
                    if name not in step_kernel_times[step_id]:
                        step_kernel_times[step_id][name] = []
                    step_kernel_times[step_id][name].append(duration)

        except Exception as parse_err:
            # CSV 文件为空、列缺失、编码错误等都会进到这里。
            # 静默吞掉会让外层得到 elapsed_us=0 且无任何提示——必须可见。
            print(f"[WARN] kernel_details.csv 解析失败 ({csv_file}): "
                  f"{type(parse_err).__name__}: {parse_err}")

        if not step_kernel_times:
            return {}, 0.0

        # 收集每个 kernel 在所有 Step 中的时间，然后取中位数
        all_kernel_times: Dict[str, List[float]] = {}
        for step_id, kernels in step_kernel_times.items():
            for name, times in kernels.items():
                # 每个 kernel 在该 step 的总时间（可能有多个同名 kernel）
                step_total = sum(times)
                if name not in all_kernel_times:
                    all_kernel_times[name] = []
                all_kernel_times[name].append(step_total)

        # 计算每个 kernel 的时间中位数
        device_kernels: Dict[str, float] = {}
        total_kernel_us = 0.0

        for name, times in all_kernel_times.items():
            median_time = self._median(times)
            device_kernels[name] = round(median_time, 2)
            total_kernel_us += median_time

        op_times = {}
        if device_kernels:
            op_times["device_kernels"] = device_kernels

        return op_times, total_kernel_us

    def _median(self, values: List[float]) -> float:
        """计算中位数"""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        if n % 2 == 1:
            return sorted_vals[n // 2]
        else:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2

    def _is_warmup_kernel(self, op_type: str, input_shapes: str) -> bool:
        """判断是否为 warmup kernel（通过精确形状匹配）

        Warmup kernel 特征（可通过 env 覆盖）：
        - MatMulV3: Input Shapes = WARMUP_MATMUL_SHAPE
        - ReduceMax: Input Shapes = WARMUP_REDUCE_SHAPE

        F111: 同步更新命中统计；若整轮命中率=0 由 _maybe_warn_warmup_filter_inactive
        触发 warning，提示 CANN/驱动升级可能让形状失效。
        """
        if not op_type or not input_shapes:
            return False

        # 仅对预期的 op_type 统计命中率（避免 op_type=Cast/Add 等非 warmup 候选拉低统计）
        if op_type in ('MatMulV3', 'ReduceMax'):
            _WARMUP_FILTER_STATS["checked"] += 1

        # 精确匹配 MatMulV3 warmup
        if op_type == 'MatMulV3' and WARMUP_MATMUL_SHAPE in input_shapes:
            _WARMUP_FILTER_STATS["matched"] += 1
            return True

        # 精确匹配 ReduceMax warmup
        if op_type == 'ReduceMax' and WARMUP_REDUCE_SHAPE in input_shapes:
            _WARMUP_FILTER_STATS["matched"] += 1
            return True

        return False

    @staticmethod
    def parse_trace_view_prof(log_path: str = None, op_func=None, *op_args,
                              warmup: int = 3, repeat: int = 5,
                              **op_kwargs) -> Dict[str, Dict[str, float]]:
        """按 PyPTO trace_view.json 口径解析性能数据。

        两种模式：
        A) 仅解析已有数据（旧行为）:
           PerfEvaluator.parse_trace_view_prof("/path/to/profiling")
        B) 运行算子 + 采集 + 解析（新增）:
           PerfEvaluator.parse_trace_view_prof(op_func=ReLU_wrapper, x=x_tensor)

        输入目录要求与参考 parse_prof 保持一致：
        ``log_path/<ascend*>/ASCEND_PROFILER_OUTPUT/trace_view.json``。
        返回结构保持为 ``{"prof": {...}}``。
        """
        prof_dir = None

        if op_func is not None:
            # --- Mode B: 运行算子 + 采集性能 ---
            prof_dir = tempfile.mkdtemp(prefix="trace_prof_")
            try:
                if op_args:
                    def _fn():
                        op_func(*op_args, **op_kwargs)
                else:
                    def _fn():
                        op_func(**op_kwargs)

                _profile_standalone(_fn, prof_dir, warmup, repeat)
                log_path = prof_dir
            except Exception as e:
                _logger.warning("parse_trace_view_prof profiling failed: %s", e)
                if prof_dir and os.path.isdir(prof_dir):
                    shutil.rmtree(prof_dir, ignore_errors=True)
                return {"prof": {}}

        # --- Parse trace_view.json ---
        if not log_path or not os.path.isdir(log_path):
            return {"prof": {}}

        profilingdir = ""
        pathlisttemp = os.listdir(log_path)
        for dirname in pathlisttemp:
            if "ascend" in dirname:
                profilingdir = dirname
                break

        trace_view_json = os.path.join(
            log_path, profilingdir, "ASCEND_PROFILER_OUTPUT", "trace_view.json"
        )
        prof_per = {"prof": {}}
        if not os.path.isfile(trace_view_json):
            if prof_dir and os.path.isdir(prof_dir):
                shutil.rmtree(prof_dir, ignore_errors=True)
            return prof_per

        with open(trace_view_json, "r") as f:
            trace_view_data = json.load(f)

        perf_data = {"aicore_e2e": [], "aicpu_kernel": []}
        for data in trace_view_data:
            name = data.get("name", "")
            if name == "KERNEL_AICPU":
                data["end_time"] = float(data["ts"]) + data["dur"]
                perf_data["aicpu_kernel"].append(data)
            if "tilefwk" in name or "PYPTO" in name:
                data["end_time"] = float(data["ts"]) + data["dur"]
                perf_data["aicore_e2e"].append(data)

        # 过滤离群点，解析 aicore_e2e 时间
        perf_data_filter = {"aicore_e2e": [], "aicpu_kernel": []}
        if not perf_data["aicore_e2e"]:
            if prof_dir and os.path.isdir(prof_dir):
                shutil.rmtree(prof_dir, ignore_errors=True)
            return prof_per
        min_aicore_dur = min([data["dur"] for data in perf_data["aicore_e2e"]])
        aicore_e2e_time_list = []
        for sample in perf_data["aicore_e2e"]:
            if (sample["dur"] - min_aicore_dur) < max(50, 1.5 * min_aicore_dur):
                perf_data_filter["aicore_e2e"].append(sample)
                aicore_e2e_time_list.append([float(sample["ts"]), sample["end_time"]])
        aicore_e2e_list = [data["dur"] for data in perf_data_filter["aicore_e2e"]]
        aicore_e2e_list.sort()
        aicore_e2e = round(sum(aicore_e2e_list) / len(aicore_e2e_list), 2)

        # 解析 aicore_e2e 抖动时间，取后 40 轮数据，(max(data) - min(data)) / min(data)
        aicore_e2e_jitter_list = [
            data["dur"] for data in perf_data["aicore_e2e"]
        ][-40:]
        aicore_e2e_jitter = (
            (max(aicore_e2e_jitter_list) - min(aicore_e2e_jitter_list))
            / min(aicore_e2e_jitter_list)
        )

        # 解析 aicpu_kernel 时间
        for data in perf_data["aicpu_kernel"]:
            s = float(data["ts"])
            e = data["end_time"]
            for aicore_e2e_time in aicore_e2e_time_list:
                if s <= aicore_e2e_time[0] and e >= aicore_e2e_time[-1]:
                    aicore_e2e_time.append(e)
                    break
        gap_list = []
        for data in aicore_e2e_time_list:
            gap = 0 if len(data) == 2 else max(data[-1] - data[-2], 0)
            gap_list.append(gap)

        prof_per["prof"]["aicore_e2e"] = aicore_e2e
        prof_per["prof"]["aicpukernel_gap"] = round(sum(gap_list) / len(gap_list), 2)
        prof_per["prof"]["aicore_e2e_jitter"] = round(aicore_e2e_jitter, 2)

        if prof_dir and os.path.isdir(prof_dir):
            shutil.rmtree(prof_dir, ignore_errors=True)
        return prof_per
