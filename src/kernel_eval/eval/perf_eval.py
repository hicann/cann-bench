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
4. 定位 profiler 产出文件（CSV + trace_view），交由 PerfMetricStrategy 解析
5. 归档 profiling 中间目录到 reports/prof_data/{rel_path}/{caseid}/

参考evaluation/core/profiler_manager.py
"""

import csv
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import traceback
from typing import Optional, Dict, Any, Tuple, List, Callable

import torch

_logger = logging.getLogger(__name__)

from ..utils.device_manager import DeviceManager
from ..config import Config, get_config
from .input_pool import InputPool, CallInputPool
from .restored_inputs import RestoredCall, PerfIntegrityError
from ..base.result import PerfResult, compute_speedup
from ..base.perf_strategy import (
    PerfMetricStrategy,
    ProfFileLocations,
    parse_csv_kernel_batches,
)


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
    文件定位后交由 PerfMetricStrategy 解析性能指标。
    每次测量前执行 MatMul + ReduceMax 升频并清空 L2 cache。

    使用方法：
        config = Config(profiler_level="Level1")
        perf_eval = PerfEvaluator(config=config, device_manager=device_mgr,
                                  perf_metric_strategy=KernelDetailsStrategy())
        outputs, perf_result = perf_eval.run_profiled(case_id, func, *args)
    """

    def __init__(self, config: Config = None, device_manager: DeviceManager = None,
                 warmup: int = 3, repeat: int = 5, archive_prof: bool = True,
                 freq_boost: bool = True, perf_metric_strategy: PerfMetricStrategy = None):
        """
        Args:
            config: 配置对象（含 profiler_level、perf_metric_strategy_override）
            device_manager: 设备管理器
            warmup: 预热次数
            repeat: 采集次数
            archive_prof: 是否归档profiling数据
            freq_boost: 是否启用NPU升频清cache
            perf_metric_strategy: 性能指标解析策略（负责文件解析，不 fallback）
        """
        self.config = config or get_config()
        self.device_manager = device_manager
        self.warmup = warmup
        self.repeat = repeat
        self.archive_prof = archive_prof
        self.freq_boost = freq_boost
        self.perf_metric_strategy = perf_metric_strategy

        # 性能指标策略：从 Config 获取策略名，通过 registry 获取实例
        # Config.perf_metric_strategy_override 由 CLI --perf-metric-strategy 设置；
        # 若为 None 则使用默认 "kernel_details"。
        strategy_name = self.config.perf_metric_strategy_override or "kernel_details"
        from ..registry.perf_strategy_registry import get_perf_metric_strategy
        self.perf_metric_strategy = get_perf_metric_strategy(strategy_name)

        # ACL 逐算子模式：单算子 benchmark 场景下 ACL 是正确模式
        if getattr(self.config, 'enable_acl_launch_mode', True):
            os.environ.setdefault("ASCEND_LAUNCH_MODE", "ACL")

        # msprof export 开关：仅在 MsProfSummaryStrategy 下启用
        self._need_msprof_export = (
            self.perf_metric_strategy.get_strategy_name() == "msprof_summary"
            and getattr(self.config, 'enable_msprof_export', True)
        )

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

        Must be called OUTSIDE TorchOpGuard/DeviceResidencyGuard context.
        Idempotent (safe to call multiple times).
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

        V3 Anti-Cheat: 强制使用自定义算子 cann_bench_warmup / cann_bench_cache_clean
        替代内置 torch.matmul / torch.max，支持完全禁用内置 kernel 树。

        执行：
        1. cann_bench_warmup - 提升 NPU 频率到稳定状态
        2. cann_bench_cache_clean - 清空 L2 cache，保证测量一致性

        Profiling Type: CannBenchWarmup / CannBenchCacheClean (用于过滤)

        Sync targets the warmup tensor's actual device — ``torch.npu.synchronize()``
        with no arg syncs the current device, which can disagree with the
        device the warmup tensors live on.

        Wrapped in ``TorchOpGuard.pause()`` so warmup ops do not trip
        the guard's forbidden-API counter (the entire run_ai_op is wrapped
        in a guard upstream; warmup is not candidate computation).
        """
        if self._warmup_tensors is not None:
            from ..security.torch_op_guard import TorchOpGuard
            from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean
            with TorchOpGuard.pause():
                mm1, mm2, reduce_input = self._warmup_tensors
                cann_bench_warmup(mm1, mm2)
                torch.npu.synchronize(mm1.device)
                cann_bench_cache_clean(reduce_input)
                torch.npu.synchronize(mm1.device)

    def _clear_cache(self):
        """清空 L2 cache (在每次测量 step 前调用，保证测量间 cache 状态一致)

        V3 Anti-Cheat: 强制使用 cann_bench_cache_clean 替代 torch.max

        Wrapped in ``TorchOpGuard.pause()`` for the same reason as
        ``_boost_freq_and_clear_cache``.
        """
        if self._warmup_tensors is not None:
            from ..security.torch_op_guard import TorchOpGuard
            from cann_bench_utils import cann_bench_cache_clean
            with TorchOpGuard.pause():
                _, _, reduce_input = self._warmup_tensors
                cann_bench_cache_clean(reduce_input)
                torch.npu.synchronize(reduce_input.device)

    def _synchronize_profile_step(self) -> None:
        """Wait for candidate NPU work before advancing profiler step.

        Direct-launch kernels are asynchronous with respect to the Python
        return path. Advancing ``prof.step()`` before the stream is idle can
        leave long-running candidate kernels outside the active profiler
        window, which later looks like an empty ``kernel_details.csv``.
        """
        if self.device_manager is not None:
            self.device_manager.synchronize()
        elif hasattr(torch, "npu"):
            torch.npu.synchronize()

    def _run_profile_step(self, fn: Callable, prof) -> Optional[BaseException]:
        """Run one scheduled profiler step and return any deferred exception."""
        try:
            fn()
            self._synchronize_profile_step()
        except BaseException as e:
            prof.step()
            return e
        prof.step()
        return None

    def _profile(self, fn: Callable, prof_dir: str, warmup: int, repeat: int,
                 include_host: bool = False, *, run_preflight: bool = True,
                 condition_steps: bool = True):
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
        # 已移至 op_runner.run_ai_op 在 guard 之前执行

        # 预检（pre-flight）：每个 profiler session 启动前，先不带 profiler 执行 fn()。
        # 若 fn() 抛异常（EZ1001 dtype 不支持等），直接 raise 不启动 profiler —
        # 避免破损的 profiler session 残留 "CANN path ''" parser 进程，
        # 污染下一个 case 的 profiler session（导致 N/A 性能数据）。
        # 条件性 pre-flight（仅在上一个 case 失败时）无法防止首次失败进入 profiler，
        # 因此必须无条件执行。开销：1 次 fn() call，profiler 本身跑 8 次 (3+5)，
        # pre-flight 仅增 ~12%。
        #
        # TODO: 后续优化提升性能 — 重构 evaluator 流程为先跑精度（不带 profiler），
        # 精度通过的 case 才继续跑性能验证（带 profiler），避免重复执行 fn()。
        # 精度验证自然成为 profiler 的门卫，无需额外 pre-flight。
        if run_preflight:
            try:
                fn()
                self._synchronize_profile_step()
            except BaseException:
                _logger.info("perf_eval: fn() pre-flight failed — skipping profiler session")
                raise

        # 诊断日志：记录 profiler session 开始前的 PROF_* 目录状态
        pre_prof_dirs = [e for e in os.listdir(prof_dir)
                         if e.startswith("PROF") and os.path.isdir(os.path.join(prof_dir, e))]
        _logger.info("perf_eval: _profile start — prof_dir=%s, pre-existing PROF dirs: %s",
                     prof_dir, pre_prof_dirs)

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

            activities = [torch_npu.profiler.ProfilerActivity.NPU]
            if include_host:
                # Batch anti-cheat needs host ranges and AscendCL memcpy APIs
                # on the same clock. Keep ordinary per-case profiling NPU-only
                # so its established performance path is unchanged; CPU events
                # here are attribution markers and are never included in time.
                activities.insert(0, torch_npu.profiler.ProfilerActivity.CPU)

            with torch_npu.profiler.profile(
                activities=activities,
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
                    if condition_steps and self.freq_boost and i >= warmup:
                        self._clear_cache()
                    fn_exc = self._run_profile_step(fn, prof)
                    if fn_exc is not None:
                        break
                if fn_exc is not None:
                    raise fn_exc

            # 等待 profiler async parser 完成 CSV 写入
            time.sleep(1.0)

            # 诊断日志：记录 profiler session 完成后的 PROF_* 目录状态
            post_prof_dirs = [e for e in os.listdir(prof_dir)
                              if e.startswith("PROF") and os.path.isdir(os.path.join(prof_dir, e))]
            _logger.info("perf_eval: _profile done — prof_dir=%s, PROF dirs after session: %s",
                         prof_dir, post_prof_dirs)

        finally:
            # F741: 移除此处的 close_pool 调用，避免多次关闭进程池
            # 原问题：每个 case 执行完都关闭进程池，可能导致后续的解析子进程无法正常运行
            # 修复：只在 shutdown() 中统一关闭一次进程池
            #
            # 注释掉的原代码：
            # try:
            #     from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
            #     pool = MultiProcessPool()
            #     pool.close_pool(wait=True)
            #     time.sleep(0.5)
            # except Exception as e:
            #     _logger.debug("perf_eval: finally close_pool(wait=True) failed: %s", e)

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

    def _profile_batch(self, fns: List[Callable], prof_dir: str,
                       warmup: int, repeat: int) -> None:
        """Profile all case callables in one session with NPU kernel timing.

        The existing profiler implementation is reused with a zero-warmup
        global schedule. Its unconditional pre-flight call is turned into a
        no-op because every queued case has already passed the performance
        stage's unprofiled correctness recheck.
        """
        steps_per_case = warmup + repeat
        total_steps = len(fns) * steps_per_case
        if total_steps <= 0:
            return

        class _BatchDispatcher:
            def __init__(self):
                self.preflight_pending = True
                self.position = 0

            def __call__(self):
                if self.preflight_pending:
                    self.preflight_pending = False
                    return None
                case_index = self.position // steps_per_case
                step_index = self.position % steps_per_case
                fn = fns[case_index]
                # record_function and AscendCL@aclrtMemcpy are both host-side
                # trace events.  Their containment relationship is therefore
                # stable; unlike the old cache-clean mapping it never compares
                # a host timestamp with a device timestamp.
                range_name = (
                    f"CannBenchBatchStep/case={case_index}/step={step_index}"
                )
                with torch.autograd.profiler.record_function(range_name):
                    output = fn()
                self.position += 1
                if self.position % steps_per_case == 0:
                    release = getattr(fn, 'release', None)
                    if callable(release):
                        release()
                return output

        # Match the established per-case path: H2D input migration and input
        # pool cloning are setup work, not candidate-op work.  Preparing them
        # before the profiler session prevents those copies from tripping the
        # per-case aclrtMemcpy anti-cheat threshold and keeps elapsed kernels
        # scoped to the submitted operator.
        for fn in fns:
            prepare = getattr(fn, 'prepare', None)
            if callable(prepare):
                prepare()

        dispatcher = _BatchDispatcher()
        self._profile(
            dispatcher, prof_dir, warmup=0, repeat=total_steps,
            include_host=True,
        )
        if dispatcher.position != total_steps:
            raise RuntimeError(
                f"batch profiler executed {dispatcher.position}/{total_steps} steps"
            )

    @staticmethod
    def _batch_has_excessive_memcpy(
        api_statistic_path: Optional[str],
        trace_view_path: Optional[str],
        n_cases: int,
        warmup: int,
        repeat: int,
    ) -> Tuple[bool, str, Optional[List[int]]]:
        """Apply the existing ``aclrtMemcpy <= 5`` limit to every case.

        ``api_statistic.csv`` only exposes a session-wide count.  Every batch
        invocation is therefore wrapped in a ``CannBenchBatchStep`` host range.
        ``AscendCL@aclrtMemcpy`` is also a host event, so containment assigns
        each memcpy to exactly one case/step without crossing clock domains.
        Warmup ranges are discarded to reproduce the ordinary per-case
        profiler's active-window semantics exactly.

        The third return value contains the zero-based case indexes that need
        per-case verification.  ``None`` means the trace is structurally
        unsafe, so the caller must fail closed and verify the whole batch.
        """
        if not api_statistic_path or not os.path.isfile(api_statistic_path):
            return False, "", []
        api_counts: Dict[str, int] = {}
        try:
            with open(api_statistic_path, 'r', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    api_name = str(row.get('API Name') or '')
                    if not api_name.startswith('aclrtMemcpy'):
                        continue
                    api_counts[api_name] = (
                        api_counts.get(api_name, 0)
                        + int(row.get('Count') or 0)
                    )
        except Exception as exc:
            return True, f"api_statistic.csv parse failed: {exc}", None

        # KernelDetailsStrategy applies the threshold independently to every
        # aclrtMemcpy* API row. Preserve that rule instead of adding different
        # memcpy variants together in batch mode.
        if not api_counts or max(api_counts.values()) <= 5:
            return False, "", []
        aggregate_count = sum(api_counts.values())
        if not trace_view_path or not os.path.isfile(trace_view_path):
            return True, (
                f"aggregate aclrtMemcpy count {aggregate_count} exceeds 5, "
                "but trace_view.json is unavailable for per-case validation"
            ), None

        try:
            # Batch traces are normally small (the 20-case canary is < 1 MB).
            # Refuse unexpectedly large input rather than risking evaluator OOM.
            trace_size = os.path.getsize(trace_view_path)
            if trace_size > 100 * 1024 * 1024:
                return True, (
                    f"trace_view.json too large for anti-cheat parsing: "
                    f"{trace_size / 1024 / 1024:.1f} MB"
                ), None
            with open(trace_view_path, 'r', encoding='utf-8') as f:
                trace = json.load(f)

            steps_per_case = warmup + repeat
            expected_ranges = n_cases * steps_per_case
            if steps_per_case <= 0:
                return True, "batch profiler has no case steps", None

            range_pattern = re.compile(
                r'^CannBenchBatchStep/case=(\d+)/step=(\d+)$'
            )
            ranges = []
            seen_ranges = set()
            for event in trace:
                match = range_pattern.match(str(event.get('name') or ''))
                if not match:
                    continue
                case_index = int(match.group(1))
                step_index = int(match.group(2))
                timestamp = event.get('ts')
                duration = event.get('dur')
                if timestamp in (None, '') or duration in (None, ''):
                    return True, (
                        "trace batch host range is missing ts/dur "
                        f"(case[{case_index}], step[{step_index}])"
                    ), None
                if not (0 <= case_index < n_cases):
                    return True, (
                        f"trace batch host range case[{case_index}] outside "
                        f"batch size {n_cases}"
                    ), None
                if not (0 <= step_index < steps_per_case):
                    return True, (
                        f"trace batch host range step[{step_index}] outside "
                        f"case step count {steps_per_case}"
                    ), None
                key = (case_index, step_index)
                if key in seen_ranges:
                    return True, (
                        "duplicate trace batch host range "
                        f"case[{case_index}], step[{step_index}]"
                    ), None
                seen_ranges.add(key)
                start = float(timestamp)
                end = start + float(duration)
                if end < start:
                    return True, (
                        "negative trace batch host range duration "
                        f"case[{case_index}], step[{step_index}]"
                    ), None
                ranges.append({
                    'case': case_index,
                    'step': step_index,
                    'start': start,
                    'end': end,
                    'pid': event.get('pid'),
                    'tid': event.get('tid'),
                })

            if len(ranges) != expected_ranges:
                return True, (
                    f"trace batch host range count {len(ranges)} != "
                    f"expected {expected_ranges}"
                ), None

            expected_keys = {
                (case_index, step_index)
                for case_index in range(n_cases)
                for step_index in range(steps_per_case)
            }
            if seen_ranges != expected_keys:
                return True, "trace batch host range coverage is incomplete", None

            ranges.sort(key=lambda item: (item['start'], item['end']))
            for previous, current in zip(ranges, ranges[1:]):
                if current['start'] < previous['end']:
                    return True, (
                        "trace batch host ranges overlap "
                        f"(case[{previous['case']}]/step[{previous['step']}] "
                        f"and case[{current['case']}]/step[{current['step']}])"
                    ), None

            memcpy_events = [
                event
                for event in trace
                if str(event.get('name') or '').split('@')[-1].startswith(
                    'aclrtMemcpy'
                )
                and event.get('ts') not in (None, '')
            ]
            trace_api_counts: Dict[str, int] = {}
            for event in memcpy_events:
                api_name = str(event.get('name') or '').split('@')[-1]
                trace_api_counts[api_name] = (
                    trace_api_counts.get(api_name, 0) + 1
                )
            if trace_api_counts != api_counts:
                return True, (
                    "trace aclrtMemcpy counts do not match api_statistic "
                    f"(trace={trace_api_counts}, api={api_counts})"
                ), None

            per_case: List[Dict[str, int]] = [
                {} for _ in range(n_cases)
            ]
            for event in memcpy_events:
                timestamp = float(event['ts'])
                containing = [
                    item for item in ranges
                    if item['start'] <= timestamp < item['end']
                ]
                same_thread = [
                    item for item in containing
                    if item['pid'] == event.get('pid')
                    and item['tid'] == event.get('tid')
                ]
                if len(same_thread) == 1:
                    owner = same_thread[0]
                elif len(containing) == 1:
                    owner = containing[0]
                else:
                    return True, (
                        "aclrtMemcpy does not belong to exactly one batch "
                        f"host range (matches={len(containing)}, ts={timestamp})"
                    ), None
                if owner['step'] < warmup:
                    continue
                api_name = str(event.get('name') or '').split('@')[-1]
                counts = per_case[owner['case']]
                counts[api_name] = counts.get(api_name, 0) + 1

            offenders = [
                (index, api_name, count)
                for index, counts in enumerate(per_case)
                for api_name, count in counts.items()
                if count > 5
            ]
            if offenders:
                details = ", ".join(
                    f"case[{index}].{api_name}={count}"
                    for index, api_name, count in offenders
                )
                return True, (
                    f"per-case aclrtMemcpy threshold exceeded ({details}; "
                    "limit=5)"
                ), sorted({index for index, _, _ in offenders})
        except Exception as exc:
            return True, f"trace_view.json memcpy validation failed: {exc}", None
        return False, "", []

    def run_profiled_batch(self, cases: List[Tuple[str, Callable]],
                           warmup: int = None, repeat: int = None
                           ) -> Dict[str, PerfResult]:
        """Measure multiple correctness-passed cases in one profiler session.

        The method is deliberately fail-closed: missing files, delimiter
        mismatches, non-monotonic device order, memcpy trace ambiguity, or
        an empty case slice produce error results. The OpRunner caller then
        invokes the unchanged per-case profiler path for those cases.
        """
        if getattr(self.config, "perf_restore_inputs", True):
            raise PerfIntegrityError("restored-input profiling cannot use opaque batch callables")
        warmup = self.warmup if warmup is None else warmup
        repeat = self.repeat if repeat is None else repeat
        batch_id = f"{os.getpid()}-{time.time_ns()}"
        results = {
            case_id: PerfResult(metadata={
                'case_id': case_id,
                '_repeat': repeat,
                'warmup_used': self.freq_boost,
                'freq_boost': self.freq_boost,
                'perf_batch_id': batch_id,
                'perf_batch_size': len(cases),
            })
            for case_id, _ in cases
        }
        if not cases:
            return results

        def _fail_all(message: str) -> Dict[str, PerfResult]:
            for result in results.values():
                result.elapsed_us = 0.0
                result.error_msg = message
                result.metadata['perf_batch_failed'] = True
                result.metadata['perf_collection_failed'] = True
            return results

        if not self.config.enable_profiler:
            return _fail_all("batch profiler requested while profiler is disabled")
        strategy_name = (
            self.perf_metric_strategy.get_strategy_name()
            if self.perf_metric_strategy else ""
        )
        if strategy_name != 'kernel_details':
            return _fail_all(
                f"batch profiler requires kernel_details strategy, got {strategy_name or 'none'}"
            )

        first_rel_path, _ = self._parse_case_id(cases[0][0])
        if self.archive_prof:
            prof_dir = os.path.join(
                self.prof_data_dir, "_batched", first_rel_path, batch_id,
            )
            os.makedirs(prof_dir, exist_ok=True)
            self._clean_prof_dir_contents(prof_dir)
        else:
            prof_dir = tempfile.mkdtemp(prefix="cann_prof_batch_")

        try:
            try:
                self._profile_batch(
                    [fn for _, fn in cases], prof_dir, warmup, repeat,
                )
                prof_files = self._locate_prof_files(prof_dir)
                if not prof_files.csv_path:
                    return _fail_all(
                        "batch profiler did not produce kernel_details.csv"
                    )

                excessive_memcpy, memcpy_reason, memcpy_fallback_indices = (
                    self._batch_has_excessive_memcpy(
                        prof_files.api_statistic_path,
                        prof_files.trace_view_path,
                        n_cases=len(cases),
                        warmup=warmup,
                        repeat=repeat,
                    )
                )
                if excessive_memcpy and memcpy_fallback_indices is None:
                    return _fail_all(
                        "batch anti-cheat requires per-case fallback: "
                        f"{memcpy_reason}"
                    )
                memcpy_fallback_set = set(memcpy_fallback_indices or [])

                parsed = parse_csv_kernel_batches(
                    prof_files.csv_path,
                    n_cases=len(cases),
                    warmup=warmup,
                    repeat=repeat,
                )
            except Exception as exc:
                return _fail_all(
                    f"batch profiler invariant failed: {type(exc).__name__}: {exc}"
                )

            for index, ((case_id, _), kernel_data) in enumerate(zip(cases, parsed)):
                result = results[case_id]
                if index in memcpy_fallback_set:
                    result.error_msg = (
                        "batch anti-cheat requires per-case fallback: "
                        f"{memcpy_reason}"
                    )
                    result.metadata['perf_batch_failed'] = True
                    result.metadata['perf_collection_failed'] = False
                    continue
                total_us = float(kernel_data.get('total_kernel_us') or 0.0)
                device_kernels = kernel_data.get('device_kernels') or {}
                if total_us <= 0 or not device_kernels:
                    result.error_msg = (
                        "batch profiler found no valid NPU kernels for this case"
                    )
                    result.metadata['perf_batch_failed'] = True
                    result.metadata['perf_collection_failed'] = False
                    continue
                result.elapsed_us = total_us
                result.op_times = {'device_kernels': device_kernels}
                result.metadata.update({
                    'perf_batch_used': True,
                    'perf_batch_index': index,
                    'elapsed_us_source': 'kernel_details.batch_total_kernel_us',
                    'data_source': 'kernel_details_csv',
                    'trace_view_supplement_skipped': True,
                })
            return results
        finally:
            for _, fn in cases:
                release = getattr(fn, 'release', None)
                if callable(release):
                    try:
                        release()
                    except Exception:
                        pass
            if not self.archive_prof and os.path.isdir(prof_dir):
                shutil.rmtree(prof_dir, ignore_errors=True)

    def run_profiled(self, case_id: str, func: Callable, *args,
                     warmup: int = None, repeat: int = None,
                     inputs: List = None, use_input_pool: bool = False,
                     input_factory=None, validator=None,
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
            use_input_pool: rotate input from *inputs* each step if True.

        Returns:
            (outputs, PerfResult) where outputs = func(*args, **kwargs) result
            from the final repeat-step iteration, and PerfResult holds parsed
            perf metrics or error state.
        """
        # F748: 移除设备锁（DeviceLock）
        # 原因：threading.Lock 只在进程内有效，多卡并行是多进程架构。
        # 设备隔离（ASCEND_RT_VISIBLE_DEVICES）已经从根本上解决了资源竞争。
        if getattr(self.config, "perf_restore_inputs", True):
            return self._run_restored_profiled(
                case_id, func, tuple(inputs) if inputs is not None else args, kwargs,
                warmup=warmup, repeat=repeat,
                input_factory=input_factory, validator=validator,
            )
        if input_factory is not None or validator is not None:
            raise PerfIntegrityError("performance hooks require perf_restore_inputs")
        return self._run_profiled_impl(case_id, func, *args,
                                       warmup=warmup, repeat=repeat,
                                       inputs=inputs, use_input_pool=use_input_pool,
                                       **kwargs)

    @staticmethod
    def _parse_restored_step(prof_files):
        """Sum every kernel in a capture containing exactly one candidate call.

        No name-based exclusions: preparation and cache conditioning happened
        before capture, so even a candidate named CannBenchCopy must be counted.
        Task ID is a launch ID, not an invocation ID; grouping by it would lose
        work when a candidate launches a kernel multiple times.
        """
        import math

        if not prof_files.csv_path or not os.path.isfile(prof_files.csv_path):
            raise PerfIntegrityError("restored-input capture has no kernel_details.csv")
        kernels = {}
        with open(prof_files.csv_path, encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                name = str(row.get("Name") or row.get("Type") or "").strip()
                duration = float(row.get("Duration(us)") or 0)
                if not name or not math.isfinite(duration) or duration < 0:
                    raise PerfIntegrityError("invalid kernel row in restored-input capture")
                kernels[name] = kernels.get(name, 0.0) + duration
        if not kernels or sum(kernels.values()) <= 0:
            raise PerfIntegrityError("restored-input capture contains no positive candidate kernel time")
        memcpy_counts = {}
        if not prof_files.api_statistic_path or not os.path.isfile(prof_files.api_statistic_path):
            raise PerfIntegrityError("restored-input capture lacks api_statistic.csv for memcpy validation")
        with open(prof_files.api_statistic_path, encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {'API Name', 'Count'}.issubset(reader.fieldnames or []):
                raise PerfIntegrityError("invalid api_statistic.csv schema in restored-input capture")
            for row in reader:
                api_name = str(row.get("API Name") or "")
                if api_name.startswith("aclrtMemcpy"):
                    count = int(row.get("Count") or 0)
                    if count < 0:
                        raise PerfIntegrityError("invalid memcpy count in restored-input capture")
                    memcpy_counts[api_name] = memcpy_counts.get(api_name, 0) + count
        return kernels, memcpy_counts

    def _run_restored_profiled(self, case_id, func, args, kwargs, *,
                               warmup=None, repeat=None, input_factory=None,
                               validator=None):
        """One capture per invocation, with trusted preparation and I/O retirement."""
        from ..security.torch_op_guard import TorchOpGuard

        warmup = self.warmup if warmup is None else warmup
        repeat = self.repeat if repeat is None else repeat
        result = PerfResult(metadata={
            'case_id': case_id, '_repeat': repeat, 'perf_restore_inputs': True,
            'perf_protocol': 'restored_io_v2', 'perf_batch_used': False,
            'warmup_used': self.freq_boost, 'freq_boost': self.freq_boost,
            'prepared_calls': 0, 'measured_calls': 0,
            'prepare_wall_us': 0.0, 'validation_wall_us': 0.0,
            'io_cleanup_wall_us': 0.0, 'io_scrubbed_bytes': 0, 'io_retired_calls': 0,
            'io_address_policy': 'previous_generation_quarantine',
            'perf_restore_invocations': [],
            'candidate_memcpy_count': 0, 'candidate_memcpy_counts': {},
        })
        started = time.perf_counter()
        call = None
        last_outputs = None
        prof_dir = None
        try:
            if not self.config.enable_profiler or warmup < 0 or repeat < 1:
                raise PerfIntegrityError("restored inputs require profiler, warmup >= 0 and repeat >= 1")
            if self.perf_metric_strategy.get_strategy_name() != 'kernel_details':
                raise PerfIntegrityError("restored inputs currently require kernel_details strategy")
            if self.device_manager is not None and not self.device_manager.is_npu_mode():
                raise PerfIntegrityError("restored inputs require NPU profiling")
            rel_path, case_num = self._parse_case_id(case_id)
            if self.archive_prof:
                prof_dir = os.path.join(self.prof_data_dir, rel_path, case_num,
                                        f'restored-{os.getpid()}-{time.time_ns()}')
                os.makedirs(prof_dir, exist_ok=False)
            else:
                prof_dir = tempfile.mkdtemp(prefix='cann_prof_restored_')
            before = time.perf_counter()
            with TorchOpGuard.pause():
                call = RestoredCall(func, args, kwargs,
                                    input_factory=input_factory, validator=validator,
                                    synchronize=self._synchronize_profile_step)
                self._synchronize_profile_step()
                if self.freq_boost:
                    self._prepare_warmup_tensors()
            result.metadata['snapshot_wall_us'] = (time.perf_counter() - before) * 1e6
            kernel_steps = []
            memcpy_counts = {}
            phases = ['preflight'] + ['warmup'] * warmup + ['measured'] * repeat
            for index, phase in enumerate(phases):
                invocation = {'index': index, 'phase': phase, 'candidate_elapsed_us': None,
                              'candidate_memcpy_counts': None}
                result.metadata['perf_restore_invocations'].append(invocation)
                before = time.perf_counter()
                with TorchOpGuard.pause():
                    call.prepare(index, phase)
                    self._synchronize_profile_step()
                    if self.freq_boost and phase == 'measured':
                        self._clear_cache()
                result.metadata['prepare_wall_us'] += (time.perf_counter() - before) * 1e6
                result.metadata['prepared_calls'] += 1
                result.metadata['input_bytes'] = call.input_bytes

                def invoke():
                    call()

                if phase == 'measured':
                    step_dir = os.path.join(prof_dir, f'measured-{len(kernel_steps)}')
                    os.makedirs(step_dir)
                    self._profile(invoke, step_dir, warmup=0, repeat=1,
                                  run_preflight=False, condition_steps=False)
                    result.metadata['measured_calls'] += 1
                else:
                    invoke()
                    self._synchronize_profile_step()
                before = time.perf_counter()
                with TorchOpGuard.pause():
                    call.complete(call.io.output, index, phase)
                result.metadata['validation_wall_us'] += (time.perf_counter() - before) * 1e6
                if phase == 'measured':
                    self.wait_all()
                    kernels, counts = self._parse_restored_step(self._locate_prof_files(step_dir))
                    invocation['candidate_memcpy_counts'] = counts
                    # Match KernelDetailsStrategy and batch attribution: each API
                    # has its own case-wide budget over all measured repeats.
                    for api_name, count in counts.items():
                        memcpy_counts[api_name] = memcpy_counts.get(api_name, 0) + count
                    result.metadata['candidate_memcpy_counts'] = dict(memcpy_counts)
                    result.metadata['candidate_memcpy_count'] = sum(memcpy_counts.values())
                    for api_name, count in memcpy_counts.items():
                        if count > 5:
                            raise PerfIntegrityError(
                                f"excessive aclrtMemcpy ({api_name}={count} >5 across measured candidate windows)"
                            )
                    kernel_steps.append(kernels)
                    invocation['candidate_elapsed_us'] = sum(kernels.values())
                before = time.perf_counter()
                with TorchOpGuard.pause():
                    last_outputs, cleared = call.retire(preserve_output=index == len(phases) - 1)
                cleanup_us = (time.perf_counter() - before) * 1e6
                invocation['io_cleanup_wall_us'] = cleanup_us
                invocation['io_scrubbed_bytes'] = cleared
                result.metadata['io_cleanup_wall_us'] += cleanup_us
                result.metadata['io_scrubbed_bytes'] += cleared
                result.metadata['io_retired_calls'] += 1
                # Visible current/previous storage only; excludes pristine caller
                # inputs, final snapshot, allocator reserves and private workspaces.
                result.metadata['io_peak_tracked_bytes'] = call.io.peak_tracked_bytes
            # Aggregate complete calls, preserving correlations between kernels.
            # Summing per-kernel medians can combine mutually exclusive paths or
            # undercount calls even when every invocation has the same kernel set.
            step_totals = [sum(step.values()) for step in kernel_steps]
            names = set().union(*(step.keys() for step in kernel_steps))
            # Conditional per-kernel medians remain diagnostic only: absent
            # kernels have no sample, and these values must not be added for scoring.
            medians = {name: statistics.median(step[name] for step in kernel_steps if name in step)
                       for name in sorted(names)}
            result.elapsed_us = round(statistics.median(step_totals), 2)
            result.op_times = {'device_kernels': {name: round(value, 2) for name, value in medians.items()}}
            result.metadata.update({
                'elapsed_us_source': 'kernel_details.restored_per_invocation',
                'data_source': 'kernel_details_csv',
                'step_kernel_us': step_totals,
                'elapsed_us_aggregation': 'median_of_invocation_totals',
                'device_kernels_aggregation': 'median_when_present_diagnostic_only',
                'profiler_sessions': repeat,
                'validated_calls': repeat if validator is not None else 0,
            })
        except Exception as exc:
            # No retry or batch fallback: integrity failures are sticky. A caller
            # can rerun an infrastructure failure with the same strict protocol.
            last_outputs = None
            result.elapsed_us = 0.0
            result.error_msg = f'{type(exc).__name__}: {exc}'
            result.metadata.update({
                'perf_integrity_failed': True, 'perf_collection_failed': False,
                'profile_exception_type': type(exc).__name__,
                'profile_exception_traceback': traceback.format_exc(),
            })
        finally:
            if call is not None:
                try:
                    with TorchOpGuard.pause():
                        call.release()
                except Exception as exc:  # noqa: BLE001 - cleanup failure invalidates the case
                    last_outputs = None
                    result.elapsed_us = 0.0
                    result.error_msg = (result.error_msg or '') + f'; I/O cleanup failed: {type(exc).__name__}: {exc}'
                    result.metadata.update(perf_integrity_failed=True, perf_collection_failed=False,
                                           io_cleanup_failed=True)
            result.metadata['protocol_wall_us'] = (time.perf_counter() - started) * 1e6
            if prof_dir is not None and not self.archive_prof:
                shutil.rmtree(prof_dir, ignore_errors=True)
        return last_outputs, result

    def _run_profiled_impl(self, case_id: str, func: Callable, *args,
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
            self._clean_prof_dir_contents(prof_dir)
        else:
            prof_dir = tempfile.mkdtemp(prefix="cann_prof_")

        max_retries = getattr(self.config, 'profiler_max_retries', 0)
        retry_delay = getattr(self.config, 'profiler_retry_delay', 1.0)

        last_outputs = None
        retry_reasons: List[str] = []

        try:
            # 注意：torch_npu.profiler 内部使用全局单例 ProcessPoolExecutor
            # 多线程并发调用会导致 Bus error，因此：
            # - 多线程模式应使用 _measure_simple（简单计时）
            # - 单线程/进程隔离模式可使用 profiler
            #
            # 当前实现：当 enable_profiler=True 时使用 profiler（需确保单线程执行）
            # 多线程并行评测应设置 enable_profiler=False
            # profiler parser logs 已在 _profile 内部抑制
            for attempt in range(1 + max_retries):
                if attempt > 0:
                    _logger.warning(
                        "perf_eval: case %s — retry %d/%d "
                        "(prev elapsed_us=%.2f, error=%s)",
                        case_id, attempt, max_retries,
                        result.elapsed_us, result.error_msg,
                    )
                    retry_reasons.append(result.error_msg or "elapsed_us<=0")
                    self._clean_prof_dir_contents(prof_dir)
                    if retry_delay > 0:
                        time.sleep(retry_delay)
                    result.elapsed_us = 0.0
                    result.op_times = {}
                    result.error_msg = None
                    for _mk in ("profile_exception_type", "profile_exception",
                                "profile_exception_traceback",
                                "data_source", "elapsed_us_source",
                                "perf_collection_failed"):
                        result.metadata.pop(_mk, None)

                pool = None
                if use_input_pool and inputs:
                    # 位置参数入池路径（兼容按 *args 位置列表调用的场景）
                    pool = InputPool(inputs, warmup + repeat)
                    def _fn():
                        nonlocal last_outputs
                        last_outputs = func(*pool.get_next())
                elif use_input_pool and (args or kwargs):
                    # 防作弊（仅性能阶段）：对实际调用的 args/kwargs 张量做 clone 轮换，
                    # 有限地址池会绕回；严格恢复输入由 restored-input 路径处理。
                    pool = CallInputPool(args, kwargs, warmup + repeat)
                    def _fn():
                        nonlocal last_outputs
                        rot_args, rot_kwargs = pool.get_next()
                        last_outputs = func(*rot_args, **rot_kwargs)
                else:
                    def _fn():
                        nonlocal last_outputs
                        last_outputs = func(*args, **kwargs)

                exec_exception = None
                try:
                    self._profile(_fn, prof_dir, warmup, repeat)
                    if self._need_msprof_export:
                        self._export_msprof_summary(prof_dir)
                except Exception as e:
                    exec_exception = e
                    result.error_msg = f"{type(e).__name__}: {e}"
                    result.metadata["profile_exception_type"] = type(e).__name__
                    result.metadata["profile_exception"] = str(e)
                    result.metadata["profile_exception_traceback"] = traceback.format_exc()
                    result.metadata["perf_collection_failed"] = True
                finally:
                    if pool is not None:
                        try:
                            pool.clear()
                        except Exception:
                            pass

                if exec_exception is None:
                    try:
                        prof_files = self._locate_prof_files(prof_dir)

                        prof_dirs = [e for e in os.listdir(prof_dir)
                                     if e.startswith("PROF")
                                     and os.path.isdir(os.path.join(prof_dir, e))]
                        _logger.info(
                            "perf_eval: case %s — prof_files: csv=%s, "
                            "trace_view=%s, ascend_output=%s, PROF dirs: %s",
                            case_id, prof_files.csv_path,
                            prof_files.trace_view_path,
                            prof_files.ascend_output_dir, prof_dirs,
                        )

                        if self.perf_metric_strategy:
                            result = self.perf_metric_strategy.parse(
                                prof_files, result)
                        else:
                            result.elapsed_us = 0.0
                            result.error_msg = (
                                "no perf_metric_strategy configured")

                        _logger.info(
                            "perf_eval: case %s — strategy=%s, "
                            "elapsed_us=%.2f, data_source=%s, error_msg=%s",
                            case_id,
                            self.perf_metric_strategy.get_strategy_name()
                            if self.perf_metric_strategy else "none",
                            result.elapsed_us,
                            result.metadata.get("data_source", "?"),
                            result.error_msg,
                        )
                    except Exception as e:
                        result.error_msg = (
                            f"parse error: {type(e).__name__}: {e}")
                        _logger.warning(
                            "perf_eval: case %s — parse exception: %s",
                            case_id, e,
                        )

                # --- retry decision ---
                if result.elapsed_us > 0:
                    if attempt > 0:
                        result.metadata["profiler_retries_used"] = attempt
                        result.metadata["profiler_retry_reasons"] = (
                            retry_reasons)
                    break

                # No valid perf data — decide whether to retry.
                # Retry only when the op produced outputs (Host side dispatched
                # the kernel). If last_outputs is None the op itself failed to
                # execute, so retrying the profiler won't help.
                if last_outputs is None:
                    _logger.warning(
                        "perf_eval: case %s — no host launch detected "
                        "(last_outputs=None), skipping retry",
                        case_id,
                    )
                    break

                if attempt >= max_retries:
                    _logger.warning(
                        "perf_eval: case %s — profiler retries exhausted "
                        "(%d attempts), elapsed_us=%.2f, error=%s",
                        case_id, 1 + max_retries, result.elapsed_us,
                        result.error_msg,
                    )
                    if max_retries > 0:
                        result.metadata["profiler_retries_exhausted"] = True
                        result.metadata["profiler_retry_reasons"] = (
                            retry_reasons)

        finally:
            if not self.archive_prof and os.path.isdir(prof_dir):
                try:
                    shutil.rmtree(prof_dir, ignore_errors=True)
                except OSError:
                    pass

        return last_outputs, result

    def _clean_prof_dir_contents(self, prof_dir: str) -> None:
        """Remove profiler output subdirectories/files from *prof_dir*.

        Called before the first attempt (to clear stale data from prior runs)
        and between retry attempts (to ensure each attempt starts clean).
        """
        try:
            for entry in os.listdir(prof_dir):
                entry_path = os.path.join(prof_dir, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path, ignore_errors=True)
                else:
                    try:
                        os.remove(entry_path)
                    except OSError:
                        pass
        except OSError as e:
            _logger.debug("prof_dir cleanup skipped for %s: %s", prof_dir, e)

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

    def _locate_prof_files(self, prof_dir: str) -> ProfFileLocations:
        """定位 profiler 产出文件，不做任何解析。

        搜索策略：
        1. 三层搜索 kernel_details.csv（direct → one-level → walk）
        2. 从 CSV 路径推算 ASCEND_PROFILER_OUTPUT 目录 + trace_view.json
        3. Fallback: 找 ascend* 子目录 → ASCEND_PROFILER_OUTPUT/

        Returns:
            ProfFileLocations（csv_path, trace_view_path, ascend_output_dir）
        """
        csv_file = None
        ascend_output_dir = None

        # 三层 CSV 搜索
        direct = os.path.join(prof_dir, "kernel_details.csv")
        if os.path.isfile(direct):
            csv_file = direct
        else:
            try:
                for entry in os.listdir(prof_dir):
                    candidate = os.path.join(prof_dir, entry, "kernel_details.csv")
                    if os.path.isfile(candidate):
                        csv_file = candidate
                        break
            except OSError:
                pass

            if csv_file is None:
                for root, dirs, files in os.walk(prof_dir):
                    for f in files:
                        if f == "kernel_details.csv":
                            csv_file = os.path.join(root, f)
                            break
                    if csv_file:
                        break

        # 从 CSV 路径推算 ASCEND_PROFILER_OUTPUT + trace_view
        if csv_file:
            ascend_output_dir = os.path.dirname(csv_file)

        # Fallback: 只找 ascend* 目录（无 CSV 时）
        if not ascend_output_dir:
            try:
                for entry in os.listdir(prof_dir):
                    if "ascend" in entry:
                        candidate_dir = os.path.join(prof_dir, entry, "ASCEND_PROFILER_OUTPUT")
                        if os.path.isdir(candidate_dir):
                            ascend_output_dir = candidate_dir
                            break
            except OSError:
                pass

        # 确定 trace_view.json 路径
        trace_view_path = None
        if ascend_output_dir:
            tv_candidate = os.path.join(ascend_output_dir, "trace_view.json")
            if os.path.isfile(tv_candidate):
                trace_view_path = tv_candidate

        # 搜索 msprof 导出的 op_summary CSV
        msprof_summary_paths = []
        for root, dirs, files in os.walk(prof_dir):
            for f in files:
                if f.startswith("op_summary_") and f.endswith(".csv"):
                    msprof_summary_paths.append(os.path.join(root, f))

        # 定位 api_statistic.csv（与 kernel_details.csv 同目录）
        api_statistic_path = None
        if ascend_output_dir:
            api_candidate = os.path.join(ascend_output_dir, "api_statistic.csv")
            if os.path.isfile(api_candidate):
                api_statistic_path = api_candidate
        if not api_statistic_path:
            for root, dirs, files in os.walk(prof_dir):
                for f in files:
                    if f == "api_statistic.csv":
                        api_statistic_path = os.path.join(root, f)
                        break
                if api_statistic_path:
                    break

        return ProfFileLocations(
            ascend_output_dir=ascend_output_dir,
            csv_path=csv_file,
            trace_view_path=trace_view_path,
            prof_dir=prof_dir,
            msprof_summary_paths=sorted(msprof_summary_paths),
            api_statistic_path=api_statistic_path,
        )

    # --- msprof export 方法（MsProfSummaryStrategy 专用）---

    def _export_msprof_summary(self, prof_dir: str) -> List[str]:
        """torch_npu.profiler session 完成后，调用 msprof.py export summary。

        与 TTK MsProfiler._analysis_profile_data 一致：
        python3 msprof.py export summary -dir <prof_dir> --format csv

        msprof export 读取 PROF 原始数据，导出 op_summary_*.csv（包含所有 kernel，
        不受 Level1 过滤限制）。产出放在 PROF_*/mindstudio_profiler_output/ 目录下。
        """
        msprof_path = self._resolve_msprof_path()
        if not msprof_path:
            _logger.debug("msprof.py not found, skip msprof export")
            return []

        # msprof.py export 需要指向 PROF_* 子目录
        prof_subdirs = []
        try:
            for entry in os.listdir(prof_dir):
                if entry.startswith("PROF_") and os.path.isdir(
                    os.path.join(prof_dir, entry)):
                    prof_subdirs.append(os.path.join(prof_dir, entry))
        except OSError:
            if os.path.basename(prof_dir).startswith("PROF_"):
                prof_subdirs = [prof_dir]

        if not prof_subdirs:
            _logger.debug("No PROF_* subdirectory found in %s, skip msprof export",
                          prof_dir)
            return []

        all_summary_paths = []
        for prof_subdir in prof_subdirs:
            try:
                proc = subprocess.run(
                    ["python3", msprof_path, "export", "summary",
                     "-dir", prof_subdir, "--format", "csv"],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.returncode != 0:
                    _logger.warning(
                        "msprof.py export summary failed for %s: rc=%d, stderr=%s",
                        prof_subdir, proc.returncode, proc.stderr[:200],
                    )
                    continue
            except Exception as e:
                _logger.warning("msprof.py export summary exception for %s: %s",
                                prof_subdir, e)
                continue

            for root, dirs, files in os.walk(prof_subdir):
                for f in files:
                    if f.startswith("op_summary_") and f.endswith(".csv"):
                        all_summary_paths.append(os.path.join(root, f))

        if all_summary_paths:
            _logger.info("msprof export found %d op_summary CSV(s)",
                         len(all_summary_paths))
        else:
            _logger.debug("msprof export produced no op_summary CSVs")

        return sorted(all_summary_paths)

    def _resolve_msprof_path(self) -> Optional[str]:
        """定位 CANN msprof.py（与 TTK 一致的搜索路径）。

        搜索顺序：ASCEND_OPP_PATH → ASCEND_HOME_PATH → ASCEND_TOOLKIT_HOME
        """
        candidates = []
        opp_path = os.getenv("ASCEND_OPP_PATH", "")
        if opp_path:
            candidates.append(os.path.normpath(os.path.join(
                opp_path, "..", "tools", "profiler",
                "profiler_tool", "analysis", "msprof", "msprof.py")))
        for env_var in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
            home = os.getenv(env_var, "")
            if home:
                candidates.append(os.path.normpath(os.path.join(
                    home, "tools", "profiler",
                    "profiler_tool", "analysis", "msprof", "msprof.py")))
        for path in candidates:
            if os.path.isfile(path):
                return path
        return None

    # --- 以下方法已迁移到 PerfMetricStrategy，保留仅用于兼容性 ---

    def wait_all(self):
        """兼容旧接口，当前为同步解析，无需等待。"""

    def shutdown(self):
        """清理warmup tensors 并关闭 profiler 进程池

        F735: 修复多卡并行场景下最后一个 case 性能数据丢失问题。

        根因：最后一个 case 执行完后，profiler 数据还在异步解析中，
        但 shutdown() 使用 wait=False 立即强制关闭进程池，导致解析未完成。

        修复：改为 wait=True，等待进程池完成所有解析任务后再关闭。
        """
        if self._warmup_tensors is not None:
            del self._warmup_tensors
            self._warmup_tensors = None

        # 等待 profiler 进程池完成所有解析任务
        try:
            from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
            pool = MultiProcessPool()
            # F735: 改为 wait=True，确保最后一个 case 的 profiler 数据解析完成
            _logger.debug("perf_evaluator.shutdown: waiting for profiler pool to complete...")
            pool.close_pool(wait=True)
            _logger.debug("perf_evaluator.shutdown: profiler pool closed successfully")
        except Exception as e:
            _logger.debug("MultiProcessPool close_pool(wait=True) failed: %s", e)


    @staticmethod
    def parse_trace_view_prof(log_path=None, op_func=None, *op_args,
                              warmup=3, repeat=5,
                              **op_kwargs):
        """Compatible shim: delegates to TraceViewStrategy.parse().

        Two modes (same as original):
        A) Parse existing data:
           PerfEvaluator.parse_trace_view_prof("/path/to/profiling")
        B) Run op + collect + parse:
           PerfEvaluator.parse_trace_view_prof(op_func=ReLU_wrapper, x=x_tensor)

        Note: now delegates to TraceViewStrategy.
        If trace_view.json has no tilefwk/PYPTO events (Level1 default),
        returns {"prof": {}} - same behavior as original (no fallback).
        """
        from ..base.perf_strategy import TraceViewStrategy

        strategy = TraceViewStrategy()
        prof_dir = None

        if op_func is not None:
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

        if not log_path or not os.path.isdir(log_path):
            return {"prof": {}}

        # Locate files
        prof_files = PerfEvaluator._locate_prof_files_static(log_path)

        # Delegate to strategy
        result = PerfResult(metadata={"case_id": "parse_trace_view_prof"})
        result = strategy.parse(prof_files, result)

        # Convert to old return format {"prof": {...}}
        if result.elapsed_us > 0 and result.op_times.get("trace_view"):
            trace_data = result.op_times["trace_view"]
            if prof_dir and os.path.isdir(prof_dir):
                shutil.rmtree(prof_dir, ignore_errors=True)
            return {"prof": dict(trace_data)}

        if prof_dir and os.path.isdir(prof_dir):
            shutil.rmtree(prof_dir, ignore_errors=True)
        return {"prof": {}}

    @staticmethod
    def _locate_prof_files_static(prof_dir):
        """Static version of _locate_prof_files (for parse_trace_view_prof)"""
        csv_file = None
        ascend_output_dir = None

        direct = os.path.join(prof_dir, "kernel_details.csv")
        if os.path.isfile(direct):
            csv_file = direct
        else:
            try:
                for entry in os.listdir(prof_dir):
                    candidate = os.path.join(prof_dir, entry, "kernel_details.csv")
                    if os.path.isfile(candidate):
                        csv_file = candidate
                        break
            except OSError:
                pass

            if csv_file is None:
                for root, dirs, files in os.walk(prof_dir):
                    for f in files:
                        if f == "kernel_details.csv":
                            csv_file = os.path.join(root, f)
                            break
                    if csv_file:
                        break

        if csv_file:
            ascend_output_dir = os.path.dirname(csv_file)

        if not ascend_output_dir:
            try:
                for entry in os.listdir(prof_dir):
                    if "ascend" in entry:
                        candidate_dir = os.path.join(prof_dir, entry, "ASCEND_PROFILER_OUTPUT")
                        if os.path.isdir(candidate_dir):
                            ascend_output_dir = candidate_dir
                            break
            except OSError:
                pass

        trace_view_path = None
        if ascend_output_dir:
            tv_candidate = os.path.join(ascend_output_dir, "trace_view.json")
            if os.path.isfile(tv_candidate):
                trace_view_path = tv_candidate

        return ProfFileLocations(
            ascend_output_dir=ascend_output_dir,
            csv_path=csv_file,
            trace_view_path=trace_view_path,
            prof_dir=prof_dir,
        )
