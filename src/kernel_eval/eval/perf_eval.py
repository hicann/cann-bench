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

import logging
import os
import shutil
import tempfile
import time
import traceback
from typing import Optional, Dict, Any, Tuple, List, Callable

import torch


_logger = logging.getLogger(__name__)

from ..utils.device_manager import DeviceManager
from ..config import Config, get_config
from .input_pool import InputPool
from ..base.result import PerfResult, compute_speedup
from ..base.perf_strategy import PerfMetricStrategy, ProfFileLocations


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
                try:
                    torch.matmul(mm1, mm2)
                    torch.npu.synchronize(mm1.device)
                    torch.max(reduce_input)
                    torch.npu.synchronize(mm1.device)
                except RuntimeError:
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
                try:
                    torch.max(reduce_input)
                    torch.npu.synchronize(reduce_input.device)
                except RuntimeError:
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
        try:
            fn()
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

            # ACL 硬件复位等待：保留极小缓冲防调度抖动。
            # 注意：sleep(0.5) / sleep(5.0) 均无法解决 PROF_* 缺失问题
            # （根因是 fn() 异常导致 parser pool 残留而非 ACL 时间）。
            time.sleep(0.1)

            # 诊断日志：记录 profiler session 完成后的 PROF_* 目录状态
            post_prof_dirs = [e for e in os.listdir(prof_dir)
                              if e.startswith("PROF") and os.path.isdir(os.path.join(prof_dir, e))]
            _logger.info("perf_eval: _profile done — prof_dir=%s, PROF dirs after session: %s",
                         prof_dir, post_prof_dirs)

        finally:
            # 等待 profiler 解析子进程池完成 — 必须在 finally 中以确保 fn() 异常时也能执行。
            # 当 fn() 抛异常（EZ1001 等），raise fn_exc 跳过 with 块后面的代码，
            # 导致 parser 子进程池以空路径("CANN path ''")残留，污染下一个 profiler session。
            # close_pool(wait=True) 等待子进程完成（空路径时它们会快速失败退出），
            # 确保 pool 完全清理后再启动下一个 session。
            try:
                from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
                pool = MultiProcessPool()
                pool.close_pool(wait=True)
            except Exception as e:
                _logger.debug("perf_eval: finally close_pool(wait=True) failed: %s", e)

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
            # 算子执行已失败，无需再解析 perf 文件——直接返回，保留原始异常信息
            # 清理临时目录后返回（确保非 archive 模式下 temp dir 不残留）
            if not self.archive_prof and os.path.isdir(prof_dir):
                try:
                    shutil.rmtree(prof_dir, ignore_errors=True)
                except OSError:
                    pass
            return last_outputs, result

        try:
            # 定位 profiler 产出文件（只返回路径，不做解析）
            prof_files = self._locate_prof_files(prof_dir)

            # 诊断日志：记录文件定位结果
            prof_dirs = [e for e in os.listdir(prof_dir)
                         if e.startswith("PROF") and os.path.isdir(os.path.join(prof_dir, e))]
            _logger.info(
                "perf_eval: case %s — prof_files: csv=%s, trace_view=%s, "
                "ascend_output=%s, PROF dirs: %s",
                case_id, prof_files.csv_path, prof_files.trace_view_path,
                prof_files.ascend_output_dir, prof_dirs,
            )

            # 交由 strategy 解析性能数据
            if self.perf_metric_strategy:
                result = self.perf_metric_strategy.parse(prof_files, result)
            else:
                result.elapsed_us = 0.0
                result.error_msg = "no perf_metric_strategy configured"

            _logger.info(
                "perf_eval: case %s — strategy=%s, elapsed_us=%.2f, "
                "data_source=%s, error_msg=%s",
                case_id,
                self.perf_metric_strategy.get_strategy_name() if self.perf_metric_strategy else "none",
                result.elapsed_us,
                result.metadata.get("data_source", "?"),
                result.error_msg,
            )

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

        return ProfFileLocations(
            ascend_output_dir=ascend_output_dir,
            csv_path=csv_file,
            trace_view_path=trace_view_path,
            prof_dir=prof_dir,
        )

    # --- 以下方法已迁移到 PerfMetricStrategy，保留仅用于兼容性 ---

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
