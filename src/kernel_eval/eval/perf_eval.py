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
4. 异步解析 profiling 数据，解析完成后自动清理 trace 文件
5. 归档 profiling 中间目录到 reports/prof_data/{level}/{op_name}/{caseid}/

参考evaluation/core/profiler_manager.py
"""

import json
import os
import re
import shutil
import tempfile
import time
import torch
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field

from ..utils.device_manager import DeviceManager
from ..config import get_config
from .input_pool import InputPool


# Warmup kernels关键词（需要过滤）
_WARMUP_KERNEL_KEYWORDS = ("MatMul", "ReduceMax", "ReduceD")
# 需要跳过的事件前缀
_SKIP_EVENT_PREFIXES = ("empty_tensor", "profiler", "torch_to_npu",
                        "HostToDevice", "Free", "Computing")


@dataclass
class PerfResult:
    """性能采集结果"""
    case_id: str
    elapsed_us: float = 0          # Kernel-only时间（平均）
    op_times: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error: Optional[str] = None
    _repeat: int = 1
    warmup_used: bool = False      # 是否使用了升频清cache


class PerfEvaluator:
    """NPU 性能评测器

    使用 torch_npu.profiler 采集 NPU kernel-only 性能数据。
    通过解析 chrome trace JSON 中的无 cat 字段事件获取设备内核时间。
    每次测量前执行 MatMul + ReduceMax 升频并清空 L2 cache。

    使用方法：
        perf_eval = PerfEvaluator(enabled=True, device_manager=device_mgr)
        perf_result = perf_eval.measure_kernel_us(fn, inputs, warmup=3, repeat=5)
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
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="perf_parse")
        self._pending: List[Tuple[Future, PerfResult, str, Optional[str]]] = []

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

    def measure_kernel_us(
        self,
        fn: Callable,
        inputs: List[Any] = None,
        warmup: int = None,
        repeat: int = None,
        use_input_pool: bool = True
    ) -> PerfResult:
        """
        测量 kernel-only 时间（Profiler方式）

        Args:
            fn: 待测量的零参数callable（或通过inputs构建）
            inputs: 输入张量列表（用于创建输入池）
            warmup: 预热次数（默认使用配置）
            repeat: 采集次数（默认使用配置）
            use_input_pool: 是否使用输入池（防止data_ptr缓存攻击）

        Returns:
            PerfResult: 测量结果
        """
        if not self.enabled:
            # 简单计时
            return self._measure_simple(fn, inputs, warmup or self.warmup, repeat or self.repeat)

        warmup = warmup or self.warmup
        repeat = repeat or self.repeat

        result = PerfResult(case_id="perf", _repeat=repeat, warmup_used=self.freq_boost)

        # 创建临时profiler目录
        prof_dir = tempfile.mkdtemp(prefix="kernel_perf_")

        try:
            import torch_npu

            # 准备升频tensors
            if self.freq_boost:
                self._prepare_warmup_tensors()

            # 创建输入池（防止data_ptr缓存攻击）
            if inputs and use_input_pool:
                pool = InputPool(inputs, warmup + repeat)
                pooled_fn = lambda: fn(*pool.get_next())
            else:
                pool = None
                pooled_fn = fn

            # Profiler配置
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
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
                for _ in range(warmup + repeat):
                    # 升频清cache
                    if self.freq_boost:
                        self._boost_freq_and_clear_cache()

                    # 执行函数
                    pooled_fn()
                    prof.step()

            # 清理输入池
            if pool:
                pool.clear()

            # 解析trace
            trace_file = self._find_trace(prof_dir)
            if trace_file:
                total_us = self._parse_trace_kernel_only(trace_file)
                result.elapsed_us = round(total_us / repeat, 2)
            else:
                result.error = "no trace_view.json produced"

        except Exception as e:
            result.error = str(e)

        finally:
            # 清理临时目录
            if prof_dir and os.path.isdir(prof_dir):
                shutil.rmtree(prof_dir, ignore_errors=True)

        return result

    def _measure_simple(
        self,
        fn: Callable,
        inputs: List[Any],
        warmup: int,
        repeat: int
    ) -> PerfResult:
        """简单计时测量（CPU模式）"""
        result = PerfResult(case_id="perf", _repeat=repeat)

        try:
            # 创建输入池
            if inputs:
                pool = InputPool(inputs, warmup + repeat)
                pooled_fn = lambda: fn(*pool.get_next())
            else:
                pool = None
                pooled_fn = fn

            # Warmup
            for _ in range(warmup):
                pooled_fn()

            # Measurement
            times = []
            for _ in range(repeat):
                if torch.cuda.is_available() if hasattr(torch, 'cuda') else False:
                    torch.cuda.synchronize()
                elif hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.synchronize()

                start = time.perf_counter()
                pooled_fn()

                if torch.cuda.is_available() if hasattr(torch, 'cuda') else False:
                    torch.cuda.synchronize()
                elif hasattr(torch, 'npu') and torch.npu.is_available():
                    torch.npu.synchronize()

                elapsed = time.perf_counter() - start
                times.append(elapsed * 1e6)  # 转换为微秒

            if pool:
                pool.clear()

            result.elapsed_us = round(sum(times) / len(times), 2)

        except Exception as e:
            result.error = str(e)

        return result

    def _find_trace(self, prof_dir: str) -> Optional[str]:
        """查找trace_view.json文件"""
        for root, _, files in os.walk(prof_dir):
            for f in files:
                if f == "trace_view.json":
                    return os.path.join(root, f)
        return None

    def _parse_trace_kernel_only(self, trace_file: str) -> float:
        """
        解析chrome trace，提取kernel-only时间

        原理：
        - Device kernel events没有cat字段
        - Host-side events有cat字段
        - 过滤掉warmup kernels (MatMul, ReduceMax, ReduceD)
        """
        if not trace_file or not os.path.exists(trace_file):
            return 0.0

        total = 0.0
        try:
            with open(trace_file) as f:
                data = json.load(f)

            events = data if isinstance(data, list) else data.get("traceEvents", [])
            for ev in events:
                if ev.get("ph") != "X":
                    continue

                dur = ev.get("dur", 0)
                if dur <= 0:
                    continue

                name = ev.get("name", "")
                if not name or name.startswith(_SKIP_EVENT_PREFIXES):
                    continue

                # 过滤warmup kernels
                if any(kw in name for kw in _WARMUP_KERNEL_KEYWORDS):
                    continue

                # Device kernel events没有cat字段
                if "cat" in ev:
                    continue

                total += dur

        except Exception:
            pass

        return total

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

    def run_profiled(self, case_id: str, func: Callable, *args, **kwargs) -> Tuple[Any, PerfResult]:
        """使用 schedule 机制执行 warmup + repeat 次，对 repeat 次结果取平均

        执行流程：
        1. 执行 warmup 次（预热，不采集）
        2. 执行 repeat 次（采集数据），profiling 中间目录直接输出到归档位置
        3. 异步解析 trace 文件，对 repeat 次取平均

        返回 (outputs, PerfResult)，op_times 在后台解析完成后填充，需调用 wait_all()
        """
        import torch_npu

        result = PerfResult(case_id=case_id, _repeat=self.repeat, warmup_used=self.freq_boost)
        last_outputs = None

        level, op_name, caseid = self._parse_case_id(case_id)
        if self.archive_prof:
            prof_dir = os.path.join(self.prof_data_dir, level, op_name, caseid)
            os.makedirs(prof_dir, exist_ok=True)
        else:
            prof_dir = tempfile.mkdtemp(prefix="cann_prof_")

        # 准备升频tensors
        if self.freq_boost:
            self._prepare_warmup_tensors()

        try:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[torch_npu.profiler.ExportType.Text],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            )

            # 直接传递 tensorboard_trace_handler，确保数据写入 prof_dir
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(
                    wait=0, warmup=self.warmup, active=self.repeat, repeat=1
                ),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(prof_dir),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                experimental_config=experimental_config,
            ) as prof:
                # 执行 warmup + active 次循环，都调用 prof.step()
                for _ in range(self.warmup + self.repeat):
                    # 升频清cache
                    if self.freq_boost:
                        self._boost_freq_and_clear_cache()

                    last_outputs = func(*args, **kwargs)
                    prof.step()
        except Exception as e:
            result.error = str(e)

        # profiling 完成后在 prof_dir 下查找 trace_view.json
        for root, dirs, files in os.walk(prof_dir):
            for f in files:
                if f == "trace_view.json":
                    trace_file = os.path.join(root, f)
                    future = self._executor.submit(self._parse_trace_file, trace_file)
                    self._pending.append((future, result, prof_dir, None))
                    break

        return last_outputs, result

    def wait_all(self):
        """等待所有后台解析任务完成，填充 op_times（取平均，保留两位小数）并清理临时文件"""
        if not self._pending:
            return

        for future, result, prof_dir, _ in self._pending:
            try:
                op_times, kernel_us = future.result(timeout=120)
                repeat = result._repeat
                if repeat > 1:
                    for cat in op_times:
                        for name in op_times[cat]:
                            op_times[cat][name] = round(op_times[cat][name] / repeat, 2)
                    kernel_us = round(kernel_us / repeat, 2)
                else:
                    for cat in op_times:
                        for name in op_times[cat]:
                            op_times[cat][name] = round(op_times[cat][name], 2)
                    kernel_us = round(kernel_us, 2)
                result.op_times = op_times
                result.elapsed_us = kernel_us
            except Exception as e:
                result.error = str(e)

        # 只清理非归档模式的临时目录
        for _, _, prof_dir, _ in self._pending:
            if prof_dir and os.path.isdir(prof_dir) and not self.archive_prof:
                try:
                    shutil.rmtree(prof_dir, ignore_errors=True)
                except OSError:
                    pass

        self._pending.clear()

    def _parse_trace_file(self, trace_file: str) -> Tuple[Dict[str, Dict[str, float]], float]:
        """解析 chrome trace JSON，通过 cat 字段区分 Host/Device 阶段"""
        if not trace_file or not os.path.exists(trace_file):
            return {}, 0.0

        host_ops: Dict[str, float] = {}
        device_kernels: Dict[str, float] = {}
        total_kernel_us = 0.0

        try:
            with open(trace_file, 'r') as f:
                data = json.load(f)

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

                if name.startswith(_SKIP_EVENT_PREFIXES):
                    continue

                # 过滤warmup kernels
                if any(kw in name for kw in _WARMUP_KERNEL_KEYWORDS):
                    continue

                # 通过 cat 字段判断：有 cat = Host 端，无 cat = Device 端
                if 'cat' in event:
                    host_ops[name] = host_ops.get(name, 0) + dur
                else:
                    device_kernels[name] = device_kernels.get(name, 0) + dur
                    total_kernel_us += dur
        except Exception:
            pass

        op_times = {}
        if host_ops:
            op_times["host_ops"] = host_ops
        if device_kernels:
            op_times["device_kernels"] = device_kernels

        return op_times, total_kernel_us

    def shutdown(self):
        """关闭线程池，释放资源，清理warmup tensors"""
        self.wait_all()
        self._executor.shutdown(wait=True)

        # 清理warmup tensors
        if self._warmup_tensors is not None:
            del self._warmup_tensors
            self._warmup_tensors = None