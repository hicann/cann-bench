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
性能采集管理器

职责：
1. NPU 模式下使用 torch_npu.profiler 采集算子级别性能数据
2. CPU 模式下使用 SimpleTimer 简单计时
3. 异步解析 profiling 数据，解析完成后自动清理 trace 文件
4. 归档 profiling 中间目录到 test/reports/prof_data/{level}/{op_name}/{caseid}/
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


@dataclass
class ProfilerResult:
    """性能采集结果"""
    case_id: str
    elapsed_us: float = 0
    op_times: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error: Optional[str] = None
    _repeat: int = 1


class ProfilerManager:
    """NPU 性能采集管理器

    使用 torch_npu.profiler 采集 NPU 算子级别性能数据。
    通过解析 chrome trace JSON 中的 AiCore 事件获取 NPU 内核执行时间。
    仅在 NPU 模式下启用，CPU 模式下由 SimpleTimer 处理。

    使用 schedule 机制支持 warmup 和多次执行：
    - warmup: 预热步数（不采集数据）
    - repeat: 采集步数
    异步解析：解析在后台线程进行，调用 wait_all() 等待完成。
    """

    PROF_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports", "prof_data")

    # 预热提频/清缓存产生的算子内核关键词，解析 trace 时需过滤
    _WARMUP_KERNEL_KEYWORDS = ('MatMul', 'ReduceMax', 'ReduceD')

    def __init__(self, enabled: bool = False, mode: str = "basic", device_manager: DeviceManager = None,
                 warmup: int = 3, repeat: int = 5, archive_prof: bool = True, freq_boost: bool = True):
        self.enabled = enabled
        self.mode = mode
        self.device_manager = device_manager
        self.warmup = warmup
        self.repeat = repeat
        self.archive_prof = archive_prof
        self.freq_boost = freq_boost
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="profiler_parse")
        self._pending: List[Tuple[Future, ProfilerResult, str, Optional[str]]] = []

    def _parse_case_id(self, case_id: str) -> Tuple[str, str, str]:
        """解析 case_id 为 (level, op_name, caseid)

        支持格式：L1_SwiGLU_12 -> ("L1", "SwiGLU", "12")
        """
        match = re.match(r'^L(\d+)_(.+?)_(\d+)$', case_id)
        if match:
            return f"L{match.group(1)}", match.group(2), match.group(3)
        return "unknown", case_id, "0"

    def _prepare_warmup_tensors(self):
        """准备预热用的张量，用于提升NPU频率和清空L2缓存

        使用大矩阵（10240x10240）MatMul 充分利用 Cube 单元提频，
        使用大张量（96x1024x1024）ReduceMax 冲刷 L2 缓存。
        """
        mm1 = torch.rand((10240, 10240), dtype=torch.float16).npu()
        mm2 = torch.rand((10240, 10240), dtype=torch.float16).npu()
        reduce_input = torch.rand((96, 1024, 1024), dtype=torch.float16).npu()
        return mm1, mm2, reduce_input

    def _boost_freq_and_clear_cache(self, mm1, mm2, reduce_input):
        """通过 MatMul 提升 NPU 频率，通过 ReduceMax 清空 L2 缓存"""
        torch.matmul(mm1, mm2)
        torch.npu.synchronize()
        torch.max(reduce_input)
        torch.npu.synchronize()

    def run_profiled(self, case_id: str, func: Callable, *args, **kwargs) -> Tuple[Any, ProfilerResult]:
        """使用 schedule 机制执行 warmup + repeat 次，对 repeat 次结果取平均

        执行流程：
        1. 执行 warmup 次（预热，不采集）
        2. 执行 repeat 次（采集数据），profiling 中间目录直接输出到归档位置
        3. 异步解析 trace 文件，对 repeat 次取平均

        返回 (outputs, ProfilerResult)，op_times 在后台解析完成后填充，需调用 wait_all()
        """
        import torch_npu

        result = ProfilerResult(case_id=case_id, _repeat=self.repeat)
        last_outputs = None

        level, op_name, caseid = self._parse_case_id(case_id)
        if self.archive_prof:
            prof_dir = os.path.join(self.PROF_DATA_DIR, level, op_name, caseid)
            os.makedirs(prof_dir, exist_ok=True)
        else:
            prof_dir = tempfile.mkdtemp(prefix="cann_prof_")

        # 准备预热张量：MatMul 提频 + ReduceMax 清 L2 缓存
        warmup_tensors = None
        if self.freq_boost:
            warmup_tensors = self._prepare_warmup_tensors()

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
                    if warmup_tensors:
                        self._boost_freq_and_clear_cache(*warmup_tensors)
                    last_outputs = func(*args, **kwargs)
                    prof.step()
        except Exception as e:
            result.error = str(e)
        finally:
            del warmup_tensors

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

                if name.startswith(('empty_tensor', 'profiler', 'torch_to_npu', 'HostToDevice', 'Free', 'Computing')):
                    continue

                # 过滤预热提频 (MatMul) 和清缓存 (ReduceMax) 产生的算子
                if self.freq_boost and any(kw in name for kw in self._WARMUP_KERNEL_KEYWORDS):
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
        """关闭线程池，释放资源"""
        self.wait_all()
        self._executor.shutdown(wait=True)
