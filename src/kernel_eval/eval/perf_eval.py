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
3. CPU 模式下使用简单计时
4. 解析 kernel_details.csv，使用精确形状匹配过滤 warmup kernel
5. 归档 profiling 中间目录到 reports/prof_data/{rel_path}/{caseid}/

参考evaluation/core/profiler_manager.py
"""

import csv
import json
import os
import re
import shutil
import sys
import tempfile
import time
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field

import torch

from ..utils.device_manager import DeviceManager
from ..config import Config, get_config
from .input_pool import InputPool


# Warmup kernel 精确形状特征（用于过滤）
WARMUP_MATMUL_SHAPE = '"10240,10240;10240,10240"'
WARMUP_REDUCE_SHAPE = '"96,1024,1024;3"'


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
        """
        if self._warmup_tensors is not None:
            mm1, mm2, reduce_input = self._warmup_tensors
            torch.matmul(mm1, mm2)
            torch.npu.synchronize(mm1.device)
            torch.max(reduce_input)
            torch.npu.synchronize(mm1.device)

    def _clear_cache(self):
        """清空 L2 cache (在每次测量 step 前调用，保证测量间 cache 状态一致)"""
        if self._warmup_tensors is not None:
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
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)

        try:
            # Redirect stdout and stderr to /dev/null at system level
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)

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
                for i in range(warmup + repeat):
                    if self.freq_boost and i >= warmup:
                        self._clear_cache()
                    fn()
                    prof.step()

            # 等待 profiler 解析完成（在恢复 stdout/stderr 之前）
            # 解析器进程在 profiler context 退出后开始工作
            try:
                from torch_npu.profiler.analysis.prof_common_func._multi_process_pool import MultiProcessPool
                pool = MultiProcessPool()
                pool.close_pool(wait=True)  # 等待解析完成
            except Exception:
                pass

        finally:
            # Restore original stdout/stderr
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)
            os.close(devnull_fd)
            logging.basicConfig = original_basicConfig

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

        result = PerfResult(case_id=case_id, _repeat=repeat, warmup_used=self.freq_boost)

        if not self.config.enable_profiler:
            self._measure_simple(result, func, inputs, warmup, repeat,
                                 use_input_pool, args, kwargs)
            return None, result

        if self.freq_boost:
            self._prepare_warmup_tensors()

        rel_path, caseid = self._parse_case_id(case_id)
        if self.archive_prof:
            prof_dir = os.path.join(self.prof_data_dir, rel_path, caseid)
            os.makedirs(prof_dir, exist_ok=True)
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
            result.error = str(e)

        try:
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
            elif not result.error:
                # 只在 profiler 未报错时才设置此错误，避免覆盖 profiler 异常信息
                result.error = "no kernel_details.csv produced"

        finally:
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
        except Exception:
            pass

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

        except Exception:
            pass

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

        Warmup kernel 特征：
        - MatMulV3: Input Shapes = '"10240,10240;10240,10240"'
        - ReduceMax: Input Shapes = '"96,1024,1024;3"'
        """
        if not op_type or not input_shapes:
            return False

        # 精确匹配 MatMulV3 warmup
        if op_type == 'MatMulV3' and WARMUP_MATMUL_SHAPE in input_shapes:
            return True

        # 精确匹配 ReduceMax warmup
        if op_type == 'ReduceMax' and WARMUP_REDUCE_SHAPE in input_shapes:
            return True

        return False