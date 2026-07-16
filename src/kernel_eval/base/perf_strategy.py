#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
性能指标策略基类

职责：
1. 定义 PerfMetricStrategy ABC — 接收 profiler 产出文件路径，自主解析并填充 PerfResult
2. ProfFileLocations — 文件定位结果数据类
3. KernelDetailsStrategy — 默认策略（kernel_details.csv 为唯一 elapsed_us 数据源，
    trace_view 仅用于补充 tilefwk/PYPTO 指标和 sanity check）
4. TraceViewStrategy — PYPTO 口径策略（**待收编**；
    KernelDetailsStrategy 的 metadata 已包含 aicore_e2e 等指标，
    待确认下游无直接依赖后删除此策略）

设计原则：
- perf_eval 只负责 profiler 运行和文件定位（_locate_prof_files）
- 文件解析逻辑完全交给 strategy，不同策略自主选择读哪些文件、怎么解析
- 每个策略只接受自己口径的数据，缺失时明确报错，不静默 fallback 到不同口径
"""

import csv
import json
import logging
import os
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 文件定位结果
# ---------------------------------------------------------------------------

@dataclass
class ProfFileLocations:
    """profiler 产出文件定位结果（由 PerfEvaluator._locate_prof_files 返回）

    perf_eval 只填充路径，不做任何解析。strategy 按需读取。
    """
    ascend_output_dir: Optional[str] = None   # ASCEND_PROFILER_OUTPUT 目录
    csv_path: Optional[str] = None            # kernel_details.csv 完整路径
    trace_view_path: Optional[str] = None     # trace_view.json 完整路径
    prof_dir: str = ""                         # profiler 输出根目录
    msprof_summary_paths: List[str] = field(default_factory=list)  # msprof 导出的 op_summary_*.csv


# ---------------------------------------------------------------------------
# 策略 ABC 基类
# ---------------------------------------------------------------------------

class PerfMetricStrategy(ABC):
    """性能指标策略基类。

    职责：接收 profiler 产出文件路径，自主解析并填充 PerfResult。
    不同子类实现不同解析口径（如 kernel-level / PYPTO trace_view-level）。

    perf_eval 只负责文件定位，解析逻辑完全在 strategy 内。
    """

    @abstractmethod
    def parse(self, prof_files: ProfFileLocations, result: Any) -> Any:
        """从 profiler 产出文件中解析性能数据，填充 PerfResult。

        Args:
            prof_files: 文件定位结果（csv_path, trace_view_path 等）
            result: 待填充的 PerfResult（已有 case_id 等基础 metadata）

        Returns:
            填充后的 PerfResult（elapsed_us, op_times, metadata 已写入）
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """策略名称标识"""
        pass


# ---------------------------------------------------------------------------
# Warmup kernel 过滤（共享逻辑）
# ---------------------------------------------------------------------------

# V3 Anti-Cheat: Warmup kernel 专用命名，无需 Shape 匹配
# cann_bench_utils 将 kernel 函数命名为 CannBenchWarmup / CannBenchCacheClean。
# Profiler 采集到的 Type 是 C++ mangled 符号（如 _Z19CannBenchCacheCleanIDhEvPhS0_llj），
# 其中嵌入了保留名 CannBenchWarmup / CannBenchCacheClean。
#
# 安全性：这两个 token 是框架保留名，提交算子不会（也不应）用这些名字命名 kernel。
# 与旧版模糊匹配（cache_clean_kernel/warmup_kernel 这类通用词）不同，此处匹配的是
# 专属保留标识，提交方无法在不刻意冒充框架内部算子的前提下命中。
WARMUP_KERNEL_TOKENS = ('CannBenchWarmup', 'CannBenchCacheClean')

# 向后兼容：保留环境变量支持（在未使用 v3 的环境中回退）
WARMUP_MATMUL_SHAPE = os.environ.get(
    "CANN_BENCH_WARMUP_MATMUL_SHAPE", '"10240,10240;10240,10240"'
)
WARMUP_REDUCE_SHAPE = os.environ.get(
    "CANN_BENCH_WARMUP_REDUCE_SHAPE", '"96,1024,1024;3"'
)


def extract_warmup_names_from_csv(csv_path: Optional[str]) -> Set[str]:
    """从 kernel_details.csv 提取 warmup kernel 名称列表。

    V3: 优先使用专用 Type 名称匹配（CannBenchWarmup/CannBenchCacheClean）
    Fallback: 使用 Input Shapes 精确匹配 MatMulV3/ReduceMax（兼容旧环境）

    Args:
        csv_path: kernel_details.csv 文件路径。为 None 时返回空集合。

    Returns:
        warmup kernel 名称集合（用于后续过滤 trace_view kernel 事件）
    """
    if not csv_path or not os.path.isfile(csv_path):
        return set()

    warmup_names = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                op_type = row.get('Type', '')
                input_shapes = row.get('Input Shapes', '')
                name = row.get('Name', '')
                if _is_warmup_kernel(op_type, input_shapes):
                    warmup_names.add(name)
    except Exception as e:
        _logger.warning("extract_warmup_names_from_csv failed for %s: %s", csv_path, e)

    return warmup_names


def _is_warmup_kernel(op_type: str, input_shapes: str = '') -> bool:
    """判断是否为 warmup kernel。

    V3: 优先匹配专用 Type (CannBenchWarmup/CannBenchCacheClean)
    Fallback: 通过精确形状匹配 MatMulV3/ReduceMax（兼容旧环境）

    注意: 旧版 Workaround 的模糊匹配已移除（存在作弊风险）
    """
    if not op_type:
        return False

    # V3: 专用保留名匹配（无需 Shape）
    # profiler 的 Type 可能是干净名（CannBenchWarmup）或 C++ mangled 符号
    # （_Z19CannBenchCacheCleanIDhEvPhS0_llj），后者内嵌保留 token，故用子串匹配。
    for token in WARMUP_KERNEL_TOKENS:
        if token in op_type:
            return True

    # Fallback: 旧版 Shape 匹配（向后兼容）
    if not input_shapes:
        return False
    if op_type == 'MatMulV3' and WARMUP_MATMUL_SHAPE in input_shapes:
        return True
    if op_type == 'ReduceMax' and WARMUP_REDUCE_SHAPE in input_shapes:
        return True

    return False


def _infer_op_type_from_name(name: str) -> str:
    """从 kernel Name 推断 Type（op_type）。

    ACLNN kernel 命名模式：aclnn<Op>_<Type>AiCore_<Type>
    e.g., aclnnMax_ReduceMaxAiCore_ReduceMax → ReduceMax
    e.g., aclnnExp_ExpAiCore_Exp → Exp
    """
    # 提取 "<Type>AiCore_<Type>" 部分
    if 'AiCore_' in name:
        suffix = name.split('AiCore_')[-1]
        return suffix
    # Fallback: 取最后一个下划线后的部分
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        return parts[1]
    return name


# ---------------------------------------------------------------------------
# trace_view kernel 级事件解析（共享逻辑）
# ---------------------------------------------------------------------------

def parse_trace_view_kernels(trace_view_path: str,
                              warmup_names: Set[str] = None) -> Dict[str, Any]:
    """从 trace_view.json 提取 kernel 级事件，按 ProfilerStep 分组，取中位数。

    Args:
        trace_view_path: trace_view.json 文件路径
        warmup_names: warmup kernel 名称集合（用于过滤）

    Returns:
        {
            "device_kernels": {kernel_name: median_us},
            "total_kernel_us": float,
            "step_kernel_times": {step: {name: [durs]}},  # 供 strategy 进一步分析
        }
    """
    if not trace_view_path or not os.path.isfile(trace_view_path):
        return {}

    warmup_names = warmup_names or set()

    try:
        with open(trace_view_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        _logger.warning("parse_trace_view_kernels: failed to load %s: %s", trace_view_path, e)
        return {}

    # 1. 提取 ProfilerStep 时间窗口
    steps = []
    for d in data:
        name = d.get('name', '')
        if 'ProfilerStep' in name:
            steps.append({
                'name': name,
                'ts': float(d['ts']),
                'end': float(d['ts']) + d['dur'],
            })

    # 2. 提取 kernel 级事件（name/dur/ts）
    #    NPU kernel 级事件特征：
    #    - name 以 "aclnn" 开头且含 "AiCore_"（ACLNN kernel 命名）
    #    - dur > 0, cat 为空（非 cpu_op/enqueue 等）
    #    - "Computing" 事件是 wrapper，与 kernel 事件同名同 ts，
    #      需要去重（只保留含 AiCore_ 的条目）
    excluded_cats = {'cpu_op', 'enqueue', 'dequeue', 'HostToDevice',
                     'async_npu', 'async_task_queue'}
    seen_ts_name = set()  # 用于去重 Computing 与 AiCore_ 条目
    kernel_events = []
    for d in data:
        name = d.get('name', '')
        dur = d.get('dur', 0)
        cat = d.get('cat', '')
        ts = float(d.get('ts', 0))
        if not name or dur <= 0:
            continue
        if cat in excluded_cats:
            continue
        # "Computing" 是 wrapper event — 与同名 AiCore_ 条目重叠，跳过
        if name == 'Computing':
            continue
        # ACLNN kernel 命名模式（含 AiCore_）
        if name.startswith('aclnn') and 'AiCore_' in name:
            # 去重：同一 ts+name 只保留一条
            key = (round(ts, 1), name)
            if key not in seen_ts_name:
                seen_ts_name.add(key)
                kernel_events.append(d)

    # 3. 按 ProfilerStep 时间窗口分组
    step_kernel_times: Dict[str, Dict[str, List[float]]] = {}
    for ke in kernel_events:
        ke_name = ke['name']
        ke_ts = float(ke['ts'])
        ke_dur = ke['dur']

        # 过滤 warmup kernel
        if ke_name in warmup_names:
            continue

        # 也可以从 name 推断 type，再做 warmup 过滤
        inferred_type = _infer_op_type_from_name(ke_name)
        if inferred_type in ('ReduceMax', 'MatMulV3'):
            # 没有 Input Shapes，无法精确过滤。
            # 但如果 name 已在 warmup_names 中（来自 CSV），就已被过滤了。
            # 如果不在 warmup_names 中但 type 匹配，保持它（可能是被评测的算子本身）
            pass

        # 找所属 step
        step_name = "unknown"
        for s in steps:
            if ke_ts >= s['ts'] - 1 and ke_ts + ke_dur <= s['end'] + 1:
                step_name = s['name']
                break

        if step_name not in step_kernel_times:
            step_kernel_times[step_name] = {}
        if ke_name not in step_kernel_times[step_name]:
            step_kernel_times[step_name][ke_name] = []
        step_kernel_times[step_name][ke_name].append(ke_dur)

    if not step_kernel_times:
        return {}

    # 4. 计算每个 kernel 跨 step 的中位数
    all_kernel_times: Dict[str, List[float]] = {}
    for step_id, kernels in step_kernel_times.items():
        for name, times in kernels.items():
            step_total = sum(times)
            if name not in all_kernel_times:
                all_kernel_times[name] = []
            all_kernel_times[name].append(step_total)

    device_kernels: Dict[str, float] = {}
    total_kernel_us = 0.0
    for name, times in all_kernel_times.items():
        median_time = _median(times)
        device_kernels[name] = round(median_time, 2)
        total_kernel_us += median_time

    total_kernel_us = round(total_kernel_us, 2)

    return {
        "device_kernels": device_kernels,
        "total_kernel_us": total_kernel_us,
        "step_kernel_times": step_kernel_times,
    }


def parse_csv_kernels(csv_path: str,
                       warmup_names: Set[str] = None) -> Dict[str, Any]:
    """从 kernel_details.csv 解析 kernel 级数据。

    kernel_details.csv 没有 Step Id 字段，Task ID 是 per-kernel-launch 的唯一 ID，
    不能直接用作 step 分组。当算子在单次 measurement step 内多次 launch 同一 kernel
    （例如 foreach_norm 一个 step 执行 3 对 reduce+combine），用 Task ID 分组会把
    每次 launch 当成独立 step，导致 per-step total 被低估到 1/N。

    修复：用 cache-clear kernel（ReduceMax/MatMulV3 warmup kernel）作为 step 分隔符，
    分隔符之间的所有非 warmup kernel 归为同一个 measurement step。

    Args:
        csv_path: kernel_details.csv 文件路径
        warmup_names: warmup kernel 名称集合

    Returns:
        {
            "device_kernels": {kernel_name: median_us},
            "total_kernel_us": float,
        }
    """
    if not csv_path or not os.path.isfile(csv_path):
        return {}

    warmup_names = warmup_names or set()

    # 第一遍：读取所有行，识别 warmup kernel（step 分隔符）
    all_rows: List[Dict[str, str]] = []
    warmup_indices: List[int] = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                op_type = (row.get('Type') or '').strip()
                input_shapes = (row.get('Input Shapes') or '').strip()
                name = (row.get('Name') or op_type).strip()
                duration_str = (row.get('Duration(us)') or '0').strip()

                row['_name'] = name
                row['_op_type'] = op_type
                row['_input_shapes'] = input_shapes
                row['_duration_str'] = duration_str
                all_rows.append(row)

                if _is_warmup_kernel(op_type, input_shapes):
                    warmup_indices.append(i)
    except Exception as e:
        _logger.warning("parse_csv_kernels failed to read %s: %s", csv_path, e)
        return {}

    if not all_rows:
        return {}

    # 第二遍：按 warmup kernel 分隔 step
    # 每个 measurement step = warmup_kernel + N 个目标 kernel launch
    # warmup kernel 本身不纳入统计
    step_kernel_times: Dict[int, Dict[str, List[float]]] = {}

    if warmup_indices:
        for si, wi in enumerate(warmup_indices):
            step_id = si
            start = wi + 1
            end = warmup_indices[si + 1] if si + 1 < len(warmup_indices) else len(all_rows)
            for i in range(start, end):
                row = all_rows[i]
                name = row['_name']
                if not name:
                    continue
                if _is_warmup_kernel(row['_op_type'], row['_input_shapes']):
                    continue
                if name in warmup_names:
                    continue
                try:
                    duration = float(row['_duration_str'])
                except (ValueError, TypeError):
                    continue
                if duration <= 0:
                    continue
                if step_id not in step_kernel_times:
                    step_kernel_times[step_id] = {}
                if name not in step_kernel_times[step_id]:
                    step_kernel_times[step_id][name] = []
                step_kernel_times[step_id][name].append(duration)
    else:
        # Fallback: 无 warmup kernel（freq_boost=False），用 Task ID 分组
        for row in all_rows:
            step_id_str = (row.get('Step Id') or row.get('Task ID') or '').strip()
            if not step_id_str:
                continue
            try:
                step_id = int(step_id_str)
            except ValueError:
                step_id = abs(hash(step_id_str)) % (10 ** 9)
            name = row['_name']
            if not name or name in warmup_names:
                continue
            try:
                duration = float(row['_duration_str'])
            except (ValueError, TypeError):
                continue
            if duration <= 0:
                continue
            if step_id not in step_kernel_times:
                step_kernel_times[step_id] = {}
            if name not in step_kernel_times[step_id]:
                step_kernel_times[step_id][name] = []
            step_kernel_times[step_id][name].append(duration)

    if not step_kernel_times:
        return {}

    all_kernel_times: Dict[str, List[float]] = {}
    for step_id, kernels in step_kernel_times.items():
        for name, times in kernels.items():
            step_total = sum(times)
            if name not in all_kernel_times:
                all_kernel_times[name] = []
            all_kernel_times[name].append(step_total)

    device_kernels: Dict[str, float] = {}
    total_kernel_us = 0.0
    for name, times in all_kernel_times.items():
        median_time = _median(times)
        device_kernels[name] = round(median_time, 2)
        total_kernel_us += median_time

    return {
        "device_kernels": device_kernels,
        "total_kernel_us": round(total_kernel_us, 2),
    }


def _median(values: List[float]) -> float:
    """计算中位数"""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


# ---------------------------------------------------------------------------
# trace_view tilefwk/PYPTO 解析（PYPTO 口径）
# ---------------------------------------------------------------------------

def parse_tilefwk_metrics(trace_view_path: str,
                          warmup_names: Set[str] = None) -> Dict[str, Any]:
    """从 trace_view.json 提取 tilefwk/PYPTO 事件的 aicore_e2e 指标。

    与原 parse_trace_view_prof 核心逻辑一致。

    Args:
        trace_view_path: trace_view.json 文件路径
        warmup_names: warmup kernel 名称集合（辅助判断）

    Returns:
        空 dict 表示无 tilefwk/PYPTO 事件（取决于算子实现方式，不 fallback）
        或 {"aicore_e2e": float, "aicpukernel_gap": float, "aicore_e2e_jitter": float}
    """
    if not trace_view_path or not os.path.isfile(trace_view_path):
        return {}

    try:
        with open(trace_view_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        _logger.warning("parse_tilefwk_metrics: failed to load %s: %s", trace_view_path, e)
        return {}

    perf_data = {"aicore_e2e": [], "aicpu_kernel": []}
    for d in data:
        name = d.get("name", "")
        if name == "KERNEL_AICPU":
            d["end_time"] = float(d["ts"]) + d["dur"]
            perf_data["aicpu_kernel"].append(d)
        if "tilefwk" in name or "PYPTO" in name:
            d["end_time"] = float(d["ts"]) + d["dur"]
            perf_data["aicore_e2e"].append(d)

    if not perf_data["aicore_e2e"]:
        return {}  # 无 tilefwk/PYPTO → 返回空，由 strategy 报错

    # 过滤离群点
    min_aicore_dur = min(d["dur"] for d in perf_data["aicore_e2e"])
    perf_data_filter = {"aicore_e2e": []}
    aicore_e2e_time_list = []
    for sample in perf_data["aicore_e2e"]:
        if (sample["dur"] - min_aicore_dur) < max(50, 1.5 * min_aicore_dur):
            perf_data_filter["aicore_e2e"].append(sample)
            aicore_e2e_time_list.append([float(sample["ts"]), sample["end_time"]])

    aicore_e2e_list = [d["dur"] for d in perf_data_filter["aicore_e2e"]]
    aicore_e2e_list.sort()
    aicore_e2e = round(sum(aicore_e2e_list) / len(aicore_e2e_list), 2)

    # jitter: 取后 40 轮数据
    aicore_e2e_jitter_list = [d["dur"] for d in perf_data["aicore_e2e"]][-40:]
    aicore_e2e_jitter = round(
        (max(aicore_e2e_jitter_list) - min(aicore_e2e_jitter_list))
        / min(aicore_e2e_jitter_list), 2
    )

    # aicpukernel_gap
    for d in perf_data["aicpu_kernel"]:
        s = float(d["ts"])
        e = d["end_time"]
        for aicore_e2e_time in aicore_e2e_time_list:
            if s <= aicore_e2e_time[0] and e >= aicore_e2e_time[-1]:
                aicore_e2e_time.append(e)
                break

    gap_list = []
    for time_data in aicore_e2e_time_list:
        gap = 0 if len(time_data) == 2 else max(time_data[-1] - time_data[-2], 0)
        gap_list.append(gap)

    return {
        "aicore_e2e": aicore_e2e,
        "aicpukernel_gap": round(sum(gap_list) / len(gap_list), 2),
        "aicore_e2e_jitter": aicore_e2e_jitter,
    }


# ---------------------------------------------------------------------------
# KernelDetailsStrategy — 默认策略
# ---------------------------------------------------------------------------

class KernelDetailsStrategy(PerfMetricStrategy):
    """默认策略：Σ kernel Duration 中位数 作为 elapsed_us。

    数据源：kernel_details.csv（唯一权威源）。
    CSV 不可用 → 明确报错，不 fallback 到 trace_view。

    为什么只用 CSV：kernel_details.csv 列出了**每一个** kernel（既包括
    aclnn 辅助 kernel，也包括 direct-launch / 自定义 AscendC kernel，如
    ``*_custom``），并带有 Type / Duration(us) / Step Id / Input Shapes。
    而 ``parse_trace_view_kernels`` 只识别 ``aclnn*AiCore_`` 命名，会**静默
    漏掉**自定义 kernel——当一个 case 同时含小的 ACLNN 辅助 kernel 和大的
    自定义 kernel 时，trace_view 口径只统计到辅助 kernel，导致 elapsed_us
    偏小、speedup 虚高。

    trace_view.json 的用途（不参与 elapsed_us 计算）：
    - 补充 tilefwk/PYPTO 指标（aicore_e2e / aicpukernel_gap / aicore_e2e_jitter）
      写入 op_times["trace_view"] 和 metadata，供下游查询
    - sanity check：对比 trace_view kernel total vs CSV total，诊断告警
    """

    def parse(self, prof_files: ProfFileLocations, result: Any) -> Any:
        """解析 profiler 产出文件，填充 PerfResult。"""
        from .result import PerfResult

        warmup_names = extract_warmup_names_from_csv(prof_files.csv_path)

        # kernel_details.csv —— 唯一 elapsed_us 数据源
        if prof_files.csv_path:
            kernel_data = parse_csv_kernels(prof_files.csv_path, warmup_names)
            if kernel_data.get("total_kernel_us") and kernel_data["total_kernel_us"] > 0:
                result.elapsed_us = kernel_data["total_kernel_us"]
                result.op_times = {"device_kernels": kernel_data["device_kernels"]}
                result.metadata["elapsed_us_source"] = "kernel_details.total_kernel_us"
                result.metadata["data_source"] = "kernel_details_csv"

                # trace_view 补充 tilefwk/PYPTO 指标（不参与 elapsed_us 计算）
                if prof_files.trace_view_path:
                    tilefwk_metrics = parse_tilefwk_metrics(
                        prof_files.trace_view_path, warmup_names
                    )
                    if tilefwk_metrics:
                        result.op_times["trace_view"] = tilefwk_metrics
                        # 将 aicore_e2e 等关键指标也写入 metadata，便于下游直接查询
                        result.metadata["aicore_e2e"] = tilefwk_metrics.get("aicore_e2e")
                        result.metadata["aicpukernel_gap"] = tilefwk_metrics.get("aicpukernel_gap")
                        result.metadata["aicore_e2e_jitter"] = tilefwk_metrics.get("aicore_e2e_jitter")

                    # sanity check：若 trace_view kernel 口径明显小于 CSV，
                    # 说明它漏掉了 CSV 中的 kernel（典型为自定义 kernel）。
                    # 此时 elapsed 已用 CSV，不受影响，仅记一条诊断告警。
                    tv = parse_trace_view_kernels(prof_files.trace_view_path, warmup_names)
                    tv_total = tv.get("total_kernel_us") or 0.0
                    csv_total = kernel_data["total_kernel_us"]
                    if tv_total and tv_total < csv_total * 0.9:
                        missing = sorted(
                            set(kernel_data["device_kernels"]) - set(tv.get("device_kernels", {}))
                        )
                        _logger.warning(
                            "trace_view kernel total (%.2fus) < CSV total (%.2fus); "
                            "trace_view dropped %d kernel(s) %s (likely custom / "
                            "direct-launch). Using CSV total.",
                            tv_total, csv_total, len(missing), missing[:5],
                        )
                return result

        # CSV 不可用 → 明确报错（不 fallback 到 trace_view，
        # 因为 parse_trace_view_kernels 的 aclnn*AiCore_ 过滤器会漏掉自定义 kernel，
        # 且有 trace_view 时 CSV 必然存在——两者是同批次 profiler 产出）
        result.elapsed_us = 0.0
        result.error_msg = (
            f"KernelDetailsStrategy: kernel_details.csv not found or empty — "
            f"csv_path={prof_files.csv_path}"
        )
        return result

    def get_strategy_name(self) -> str:
        return "kernel_details"


# ---------------------------------------------------------------------------
# TraceViewStrategy — PYPTO 口径策略
# ---------------------------------------------------------------------------

class TraceViewStrategy(PerfMetricStrategy):
    """PYPTO 口径策略：trace_view.aicore_e2e 作为 elapsed_us。

    **待收编** — KernelDetailsStrategy 的 metadata 已包含
    aicore_e2e / aicpukernel_gap / aicore_e2e_jitter，完全覆盖了此策略
    的输出。待确认下游无直接依赖后删除此策略，届时请使用
    KernelDetailsStrategy（默认策略）替代。

    数据源要求：trace_view.json 中的 tilefwk/PYPTO 事件。
    无 fallback — tilefwk/PYPTO 缺失时明确报错，不静默切换口径。

    使用条件：被评测的算子使用了 tilefwk/PYPTO 实现。
    tilefwk/PYPTO 事件的存在与否取决于**算子的实现方式**，
    而不是 profiler level — 非 tilefwk/PYPTO 算子（如 mish、exp）
    不会产出这类事件，此时 TraceViewStrategy 不可用。
    """

    def parse(self, prof_files: ProfFileLocations, result: Any) -> Any:
        """解析 profiler 产出文件，填充 PerfResult。"""
        import warnings
        warnings.warn(
            "TraceViewStrategy 待收编。"
            "KernelDetailsStrategy 的 metadata 已包含 aicore_e2e 等指标，"
            "待确认下游无直接依赖后将删除此策略。",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        from .result import PerfResult

        if not prof_files.trace_view_path:
            result.elapsed_us = 0.0
            result.error_msg = (
                "TraceViewStrategy: trace_view.json not found — "
                "this strategy requires trace_view.json with tilefwk/PYPTO events "
                "(only available for operators that use tilefwk/PYPTO implementation)"
            )
            return result

        warmup_names = extract_warmup_names_from_csv(prof_files.csv_path)

        # 明确要求 tilefwk/PYPTO 事件
        trace_metrics = parse_tilefwk_metrics(prof_files.trace_view_path, warmup_names)

        if not trace_metrics.get("aicore_e2e") or trace_metrics["aicore_e2e"] <= 0:
            result.elapsed_us = 0.0
            result.error_msg = (
                "TraceViewStrategy: no tilefwk/PYPTO events found in trace_view.json — "
                "this strategy is only applicable to operators that use tilefwk/PYPTO "
                "implementation. For non-tilefwk/PYPTO operators, use KernelDetailsStrategy."
            )
            result.metadata["aicore_e2e_missing"] = True
            return result

        # 成功 — 仅暴露 PYPTO 3 字段
        result.elapsed_us = trace_metrics["aicore_e2e"]
        result.op_times = {"trace_view": {
            k: trace_metrics[k]
            for k in ("aicore_e2e", "aicpukernel_gap", "aicore_e2e_jitter")
            if k in trace_metrics
        }}
        result.metadata["elapsed_us_source"] = "trace_view.aicore_e2e"
        result.metadata["aicore_e2e_source"] = "tilefwk"

        return result

    def get_strategy_name(self) -> str:
        return "trace_view"


# ---------------------------------------------------------------------------
# MsProfSummaryStrategy — 基准采集专用策略
# ---------------------------------------------------------------------------

# msprof op_summary 可识别的 kernel 类型（与 TTK rts_sequence.py 一致）
KERNEL_TYPES = ("AI_CORE", "AIV_SQE", "AI_VECTOR_CORE",
                "MIX_AIC", "MIX_AIV",
                "KERNEL_AIVEC", "KERNEL_AICORE")


def parse_msprof_op_summary(csv_paths: List[str]) -> Dict[str, Any]:
    """从 msprof op_summary_*.csv 解析 kernel 级数据。

    实测验证：op_summary 包含 Input Shapes 列，可用 _is_warmup_kernel() 精确过滤。
    列名：Op Name, OP Type, Task Type, Task Duration(us), Input Shapes 等

    Returns:
        {"device_kernels": {name: median_us}, "total_kernel_us": float}
    """
    all_kernel_times: Dict[str, List[float]] = {}

    for csv_path in csv_paths:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    task_type = row.get('Task Type', '').strip()
                    op_type = row.get('OP Type', '').strip()
                    op_name = row.get('Op Name', op_type)
                    duration_str = row.get('Task Duration(us)', '0')
                    input_shapes = row.get('Input Shapes', '')
                    try:
                        duration = float(duration_str)
                    except (ValueError, TypeError):
                        continue
                    if duration <= 0:
                        continue
                    # 过滤非 AI_CORE 类型
                    if task_type not in KERNEL_TYPES:
                        continue
                    # Warmup 过滤（精确 — 用 Input Shapes，与 kernel_details.csv 一致）
                    if _is_warmup_kernel(op_type, input_shapes):
                        continue
                    if op_name not in all_kernel_times:
                        all_kernel_times[op_name] = []
                    all_kernel_times[op_name].append(duration)
        except Exception as e:
            _logger.warning("parse_msprof_op_summary failed for %s: %s", csv_path, e)

    if not all_kernel_times:
        return {}

    device_kernels: Dict[str, float] = {}
    total_kernel_us = 0.0
    for name, times in all_kernel_times.items():
        median_time = _median(times)
        device_kernels[name] = round(median_time, 2)
        total_kernel_us += median_time

    return {
        "device_kernels": device_kernels,
        "total_kernel_us": round(total_kernel_us, 2),
    }


class MsProfSummaryStrategy(PerfMetricStrategy):
    """基准采集专用策略：优先 kernel_details.csv，fallback msprof op_summary。

    适用场景：baseline 性能数据采集 — 需要最完整的 kernel 覆盖，
    包括自定义 AscendC kernel、direct-launch kernel 等。

    数据源优先级：
    1. kernel_details.csv（更精确：有 Step Id、Input Shapes 精确过滤）
    2. msprof op_summary（更完整：包含所有 kernel，不受 Level1 过滤限制）
    3. 全部不可用 → 明确报错

    与 KernelDetailsStrategy 的区别：
    - KernelDetailsStrategy 只用 kernel_details.csv（缺失时报错，不 fallback）
    - MsProfSummaryStrategy 先尝试 CSV，不可用时 fallback 到 msprof
    - 正式评测用 KernelDetailsStrategy，基准采集用 MsProfSummaryStrategy
    """

    def parse(self, prof_files: ProfFileLocations, result: Any) -> Any:
        from .result import PerfResult

        # === 第一优先级：kernel_details.csv ===
        if prof_files.csv_path:
            warmup_names = extract_warmup_names_from_csv(prof_files.csv_path)
            kernel_data = parse_csv_kernels(prof_files.csv_path, warmup_names)
            if kernel_data.get("total_kernel_us") and kernel_data["total_kernel_us"] > 0:
                result.elapsed_us = kernel_data["total_kernel_us"]
                result.op_times = {"device_kernels": kernel_data["device_kernels"]}
                result.metadata["elapsed_us_source"] = "kernel_details.total_kernel_us"
                result.metadata["data_source"] = "kernel_details_csv"

                # trace_view 补充
                self._add_trace_view_supplement(prof_files, result, warmup_names)

                # sanity check
                if prof_files.trace_view_path:
                    tv = parse_trace_view_kernels(prof_files.trace_view_path, warmup_names)
                    tv_total = tv.get("total_kernel_us") or 0.0
                    csv_total = kernel_data["total_kernel_us"]
                    if tv_total and tv_total < csv_total * 0.9:
                        missing = sorted(
                            set(kernel_data["device_kernels"]) - set(tv.get("device_kernels", {}))
                        )
                        _logger.warning(
                            "trace_view kernel total (%.2fus) < CSV total (%.2fus); "
                            "trace_view dropped %d kernel(s) %s (likely custom / "
                            "direct-launch). Using CSV total.",
                            tv_total, csv_total, len(missing), missing[:5],
                        )

                return result

        # === 第二优先级：msprof op_summary ===
        if prof_files.msprof_summary_paths:
            warmup_names = extract_warmup_names_from_csv(prof_files.csv_path)
            msprof_data = parse_msprof_op_summary(prof_files.msprof_summary_paths)
            if msprof_data.get("total_kernel_us") and msprof_data["total_kernel_us"] > 0:
                result.elapsed_us = msprof_data["total_kernel_us"]
                result.op_times = {"device_kernels": msprof_data["device_kernels"]}
                result.metadata["elapsed_us_source"] = "msprof_op_summary.total_kernel_us"
                result.metadata["data_source"] = "msprof_op_summary"
                _logger.info(
                    "MsProfSummaryStrategy: elapsed_us=%.2f from msprof op_summary, "
                    "kernels=%s",
                    msprof_data["total_kernel_us"],
                    list(msprof_data["device_kernels"].keys())[:5],
                )

                # trace_view 补充
                self._add_trace_view_supplement(prof_files, result, warmup_names)
                return result

        # === 全部不可用 → 明确报错 ===
        result.elapsed_us = 0.0
        result.error_msg = (
            f"MsProfSummaryStrategy: kernel_details.csv and msprof op_summary "
            f"both unavailable — "
            f"csv_path={prof_files.csv_path}, "
            f"msprof_paths={prof_files.msprof_summary_paths}"
        )
        return result

    def _add_trace_view_supplement(self, prof_files: ProfFileLocations,
                                    result: Any, warmup_names: Set[str]):
        """补充 trace_view tilefwk/PYPTO 指标（不参与 elapsed_us 计算）"""
        if not prof_files.trace_view_path:
            return
        tilefwk_metrics = parse_tilefwk_metrics(
            prof_files.trace_view_path, warmup_names
        )
        if tilefwk_metrics:
            result.op_times["trace_view"] = tilefwk_metrics
            result.metadata["aicore_e2e"] = tilefwk_metrics.get("aicore_e2e")
            result.metadata["aicpukernel_gap"] = tilefwk_metrics.get("aicpukernel_gap")
            result.metadata["aicore_e2e_jitter"] = tilefwk_metrics.get("aicore_e2e_jitter")

    def get_strategy_name(self) -> str:
        return "msprof_summary"