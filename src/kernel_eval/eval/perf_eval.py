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
1. NPU 模式下使用 torch_npu.profiler 采集 kernel-only 性能数据
2. 支持 NPU 升频 + L2 cache 清空，保证测量一致性
3. CPU 模式下使用简单计时
4. 归档 profiling 中间目录到 reports/prof_data/{level}/{op_name}/{caseid}/

参考evaluation/core/profiler_manager.py
"""

import json
import os
import re
import shutil
import tempfile
import time
import torch
from contextlib import contextmanager
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field

from ..utils.device_manager import DeviceManager
from ..config import get_config
from .input_pool import InputPool


# Warmup kernels关键词（需要过滤，大小写不敏感）
# Level0 trace 中 dequeue 事件名为 aclnn API 级（如 aclnnMatmul、aclnnMax），
# 非详细 kernel 名（如 ReduceMaxAiCore），需同时覆盖两种命名
_WARMUP_KERNEL_KEYWORDS = ("matmul", "max", "reducemax", "reduced")


@dataclass
class PerfResult:
    """性能采集结果"""
    case_id: str
    elapsed_us: float = 0          # Kernel-only时间（平均）
    op_times: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error: Optional[str] = None
    _repeat: int = 1
    warmup_used: bool = False      # 是否使用了升频清cache


@contextmanager
def _suppress_cann_profiler_errors():
    """Suppress CANN profiler's internal parser-failure ERROR messages.

    torch_npu's Level0 profiling collects minimal CANN data — not enough
    for msprof --export=on to generate MindStudio timeline CSVs.
    CANNTimelineParser polls for those CSVs in an infinite loop and is
    eventually marked FAILED by ConcurrentTasksManager, cascading to 8
    downstream parsers (RelationParser, CANNAnalyzeParser, etc.).  The
    failures are harmless: CANNExportParser still generates trace_view.json
    with dequeue events (cat="dequeue") that carry NPU kernel times.

    Because the parsers run in a ProcessPoolExecutor child (forked on
    Linux), Python-level monkey-patching has no effect — the child gets a
    fresh interpreter.  Replacing parent fd 1 before the fork is the only
    reliable way.
    """
    saved_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)
    try:
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)


class PerfEvaluator:
    """NPU 性能评测器

    使用 torch_npu.profiler 采集 NPU kernel-only 性能数据。
    通过解析 chrome trace JSON 中的 dequeue 事件获取设备内核时间。
    每次测量前执行 MatMul + ReduceMax 升频并清空 L2 cache。

    使用方法：
        perf_eval = PerfEvaluator(enabled=True, device_manager=device_mgr)
        outputs, perf_result = perf_eval.run_profiled(case_id, func, *args)
    """

    def __init__(self, enabled: bool = False, device_manager: DeviceManager = None,
                 warmup: int = 3, repeat: int = 5, archive_prof: bool = True,
                 freq_boost: bool = True):
        """
        Args:
            enabled: 是否启用profiler
            device_manager: 设备管理器
            warmup: 预热次数
            repeat: 采集次数
            archive_prof: 是否归档profiling数据
            freq_boost: 是否启用NPU升频清cache
        """
        self.enabled = enabled
        self.device_manager = device_manager
        self.warmup = warmup
        self.repeat = repeat
        self.archive_prof = archive_prof
        self.freq_boost = freq_boost

        # 性能数据归档目录
        config = get_config()
        self.prof_data_dir = os.path.join(config.reports_dir, "prof_data")

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
        """NPU升频 + 清L2 cache

        执行 MatMul + ReduceMax 以：
        1. 提升 NPU 频率到稳定状态
        2. 清空 L2 cache，保证测量一致性

        Sync targets the warmup tensor's actual device — ``torch.npu.synchronize()``
        with no arg syncs the current device, which can disagree with the
        device the warmup tensors live on.
        """
        if self._warmup_tensors is not None:
            mm1, mm2, reduce_input = self._warmup_tensors
            torch.matmul(mm1, mm2)
            torch.npu.synchronize(mm1.device)
            torch.max(reduce_input)
            torch.npu.synchronize(mm1.device)

    def _profile(self, fn: Callable, prof_dir: str, warmup: int, repeat: int):
        """Execute warmup + repeat calls with NPU profiler.  Trace written to
        prof_dir — caller is responsible for locating trace_view.json and
        kicking off parsing."""
        import torch_npu

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
        )

        with _suppress_cann_profiler_errors():
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
                for _ in range(warmup + repeat):
                    if self.freq_boost:
                        self._boost_freq_and_clear_cache()
                    fn()
                    prof.step()

    def run_profiled(self, case_id: str, func: Callable, *args,
                     warmup: int = None, repeat: int = None,
                     inputs: List = None, use_input_pool: bool = False,
                     **kwargs) -> Tuple[Any, PerfResult]:
        """Profile func with warmup + repeat steps, return (outputs, result).

        NPU path uses torch_npu.profiler (Level0 kernel-only trace).  CPU
        fallback measures wall-clock time with ``time.perf_counter()`` when
        ``self.enabled`` is False.

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

        result = PerfResult(case_id=case_id, _repeat=repeat, warmup_used=self.freq_boost)

        if not self.enabled:
            self._measure_simple(result, func, inputs, warmup, repeat,
                                 use_input_pool, args, kwargs)
            return None, result

        if self.freq_boost:
            self._prepare_warmup_tensors()

        level, op_name, caseid = self._parse_case_id(case_id)
        if self.archive_prof:
            prof_dir = os.path.join(self.prof_data_dir, level, op_name, caseid)
            os.makedirs(prof_dir, exist_ok=True)
        else:
            prof_dir = tempfile.mkdtemp(prefix="cann_prof_")

        last_outputs = None

        try:
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
            result.error = str(e)

        # Locate trace_view.json and parse synchronously.
        trace_file = None
        for root, dirs, files in os.walk(prof_dir):
            for f in files:
                if f == "trace_view.json":
                    trace_file = os.path.join(root, f)
                    break
            if trace_file:
                break

        if trace_file:
            op_times, total_kernel_us = self._parse_trace_file(trace_file)
            self._normalize_result(result, op_times, total_kernel_us)
        else:
            result.error = "no trace_view.json produced"

        # Clean up temp dir (non-archive mode).
        if not self.archive_prof and os.path.isdir(prof_dir):
            try:
                shutil.rmtree(prof_dir, ignore_errors=True)
            except OSError:
                pass

        return last_outputs, result

    def _measure_simple(self, result: PerfResult, func: Callable,
                        inputs: List, warmup: int, repeat: int,
                        use_input_pool: bool, args: tuple, kwargs: dict):
        """CPU fallback: wall-clock time via time.perf_counter()."""
        if inputs and use_input_pool:
            pool = InputPool(inputs, warmup + repeat)
            def fn():
                return func(*pool.get_next())
        else:
            pool = None
            def fn():
                return func(*args, **kwargs)

        try:
            for _ in range(warmup):
                fn()
        except Exception:
            pass  # warmup errors don't invalidate measurement

        times = []
        for _ in range(repeat):
            if hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.synchronize()
            t0 = time.perf_counter()
            fn()
            if hasattr(torch, 'npu') and torch.npu.is_available():
                torch.npu.synchronize()
            times.append((time.perf_counter() - t0) * 1_000_000)

        if pool:
            pool.clear()

        result.elapsed_us = round(sum(times) / len(times), 2) if times else 0

    def _parse_case_id(self, case_id: str) -> Tuple[str, str, str]:
        """Split a CaseInfo.get_case_id_str() like ``L2_Gcd_5`` into
        ``("level2", "Gcd", "5")`` for use as the prof_data archive path.
        Falls back to safe defaults so an unexpected format doesn't crash
        the profiler — we'd rather lose the archive layout than the run.
        """
        m = re.match(r"^L(?P<level>\d+)_(?P<op>.+)_(?P<case>\d+)$", case_id)
        if not m:
            return "level_unknown", case_id or "unknown", "0"
        return f"level{m['level']}", m["op"], m["case"]

    def _normalize_result(self, result: PerfResult,
                           op_times: Dict[str, Dict[str, float]],
                           total_kernel_us: float):
        """Normalize op_times by repeat count and populate result."""
        repeat = result._repeat
        if repeat > 1:
            for cat in op_times:
                for name in op_times[cat]:
                    op_times[cat][name] = round(op_times[cat][name] / repeat, 2)
            total_kernel_us = round(total_kernel_us / repeat, 2)
        else:
            for cat in op_times:
                for name in op_times[cat]:
                    op_times[cat][name] = round(op_times[cat][name], 2)
            total_kernel_us = round(total_kernel_us, 2)
        result.op_times = op_times
        result.elapsed_us = total_kernel_us

    def wait_all(self):
        """兼容旧接口，当前为同步解析，无需等待。"""

    def shutdown(self):
        """清理warmup tensors"""
        if self._warmup_tensors is not None:
            del self._warmup_tensors
            self._warmup_tensors = None

    def _parse_trace_file(self, trace_file: str) -> Tuple[Dict[str, Dict[str, float]], float]:
        """解析 chrome trace JSON，提取 CANN NPU 执行时间

        依据 CANN 事件分类（cat 字段）：
        - cat="dequeue" 事件的 dur 字段 = NPU 内核执行时间
        - cat="cpu_op" 事件 = CPU 侧 API 调用时间（仅供参考）
        """
        if not trace_file or not os.path.exists(trace_file):
            return {}, 0.0

        host_ops: Dict[str, float] = {}
        device_kernels: Dict[str, float] = {}
        total_kernel_us = 0.0

        try:
            with open(trace_file, 'r') as f:
                content = f.read()

            # Fix for potentially incomplete JSON (torch_npu profiler sometimes doesn't close the array)
            content = content.strip()
            if not content.endswith(']'):
                content = content + ']'

            data = json.loads(content)
            events = data if isinstance(data, list) else data.get('traceEvents', [])

            for event in events:
                if event.get('ph') != 'X':
                    continue

                dur = event.get('dur', 0)
                if dur <= 0:
                    continue

                name = event.get('name', '')
                if not name:
                    continue

                # 提取 dequeue 事件（NPU 内核执行时间）
                # cat="dequeue" 是 CANN 权威分类，不依赖算子命名前缀（aclnn/acl/tbe 等）
                if event.get('cat') == 'dequeue':
                    # 过滤warmup kernels（大小写不敏感）
                    if any(kw in name.lower() for kw in _WARMUP_KERNEL_KEYWORDS):
                        continue

                    device_kernels[name] = device_kernels.get(name, 0) + dur
                    total_kernel_us += dur

                # 可选：记录CPU侧aclnn调用时间
                elif event.get('cat') == 'cpu_op':
                    if any(kw in name.lower() for kw in _WARMUP_KERNEL_KEYWORDS):
                        continue
                    host_ops[name] = host_ops.get(name, 0) + dur

        except Exception:
            pass

        op_times = {}
        if host_ops:
            op_times["host_ops"] = host_ops
        if device_kernels:
            op_times["device_kernels"] = device_kernels

        return op_times, total_kernel_us