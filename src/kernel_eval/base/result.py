#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
评测结果基类（统一）

包含：
- OutputResult: 单个输出的判断结果抽象基类
- AccuracyResult: 统一的精度判断结果
- PerfResult: 统一的性能评测结果

Why: 提供统一的评测结果接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type


FAILURE_TYPE_PRECISION_MISMATCH = "precision_mismatch"
FAILURE_TYPE_COMPILE_RUNTIME_ERROR = "compile_runtime_error"

PRECISION_FAILURE_TYPES = {
    FAILURE_TYPE_PRECISION_MISMATCH,
    "precision",
}

COMPILE_RUNTIME_FAILURE_TYPES = {
    FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
    "runtime_error",
    "interface_error",
    "golden_error",
    "timeout",
    "subprocess_failure",
    "oom_killed",
    "cascade_device",
    "skipped",
}

STRUCTURAL_FAILURE_MARKERS = (
    "输出数量不匹配",
    "同精度输出数量不匹配",
    "形状不匹配",
    "shape mismatch",
    "is None",
    "AI算子执行失败",
    "TypeError:",
    "RuntimeError:",
    "ValueError:",
)


# === 输出结果注册表 ===

_OUTPUT_RESULT_REGISTRY: Dict[str, Type[OutputResult]] = {}


def register_output_result(name: str, cls: Type[OutputResult]) -> None:
    """注册 OutputResult 子类，供反序列化时按 checker_name 查找"""
    _OUTPUT_RESULT_REGISTRY[name] = cls


def get_output_result_cls(name: str) -> Optional[Type[OutputResult]]:
    """按 checker_name 查找已注册的 OutputResult 子类"""
    return _OUTPUT_RESULT_REGISTRY.get(name)


# === 输出结果基类 ===

class OutputResult(ABC):
    """单个输出的判断结果抽象基类

    各 Checker 实现不同子类，子类自行决定需要哪些字段。
    通用接口只要求 to_dict() 和 format_summary()。
    """

    index: int
    passed: bool
    dtype: str
    error_msg: str

    def get_index(self) -> int:
        """获取输出索引"""
        return self.index

    def is_passed(self) -> bool:
        """是否通过"""
        return self.passed

    def get_dtype(self) -> str:
        """获取数据类型"""
        return self.dtype

    def get_error_msg(self) -> str:
        """获取错误信息"""
        return self.error_msg

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        pass

    @abstractmethod
    def format_summary(self) -> str:
        """格式化摘要"""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> OutputResult:
        """从字典反序列化"""
        pass


# === 精度结果 ===

@dataclass
class AccuracyResult:
    """统一的精度判断结果

    包含必要字段 + metadata 扩展字段。
    """
    passed: bool
    threshold: Optional[float] = None
    error_msg: Optional[str] = None
    output_results: List[OutputResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_passed(self) -> bool:
        """是否通过"""
        return self.passed

    def get_threshold(self) -> Optional[float]:
        """获取精度阈值"""
        return self.threshold

    def get_error_msg(self) -> Optional[str]:
        """获取错误信息"""
        return self.error_msg

    def get_output_results(self) -> List[OutputResult]:
        """获取各输出独立结果"""
        return self.output_results

    def get_metadata(self) -> Dict[str, Any]:
        """获取扩展元数据"""
        return self.metadata

    def get_first_dtype(self) -> str:
        """获取第一个输出的 dtype"""
        if self.output_results:
            return self.output_results[0].get_dtype()
        return ""

    def get_failed_dtype(self) -> str:
        """获取第一个失败输出的 dtype"""
        for r in self.output_results:
            if not r.is_passed():
                return r.get_dtype()
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = {
            'passed': self.passed,
            'error_msg': self.error_msg,
            'output_results': [r.to_dict() for r in self.output_results],
            'metadata': self.metadata,
        }
        if self.threshold is not None:
            d['threshold'] = self.threshold
        return d

    def format_summary(self) -> str:
        """格式化摘要"""
        dtype = self.get_failed_dtype() or self.get_first_dtype()
        if self.passed:
            return f"[{dtype}] ✅ threshold={self.threshold:.6f}"
        else:
            if self.error_msg:
                return f"[{dtype}] ❌ {self.error_msg}"
            return f"[{dtype}] ❌ threshold={self.threshold:.6f}"

    def format_all_outputs(self) -> str:
        """格式化所有输出判定结果"""
        lines = []
        for r in self.output_results:
            lines.append(f"  - {r.format_summary()}")
        return "\n".join(lines)


def get_accuracy_failure_type(accuracy_result: Optional[AccuracyResult]) -> Optional[str]:
    """Classify a failed accuracy result for scoring.

    Only true numerical precision mismatches keep the compile/runtime component.
    Structural/interface failures during the precision stage, such as wrong
    output count or shape, are scored as compile/runtime errors.
    """
    if accuracy_result is None or accuracy_result.passed:
        return None

    metadata = accuracy_result.metadata or {}
    explicit = metadata.get("failure_type")
    if explicit:
        return str(explicit)

    messages = []
    if accuracy_result.error_msg:
        messages.append(str(accuracy_result.error_msg))
    for output in accuracy_result.output_results or []:
        if getattr(output, "error_msg", None):
            messages.append(str(output.error_msg))
        output_metadata = getattr(output, "metadata", None) or {}
        output_failure_type = output_metadata.get("failure_type")
        if output_failure_type:
            return str(output_failure_type)

    joined = "\n".join(messages)
    if any(marker in joined for marker in STRUCTURAL_FAILURE_MARKERS):
        return FAILURE_TYPE_COMPILE_RUNTIME_ERROR

    return FAILURE_TYPE_PRECISION_MISMATCH


def is_precision_failure_type(failure_type: Optional[str]) -> bool:
    return str(failure_type or "") in PRECISION_FAILURE_TYPES


def is_compile_runtime_failure_type(failure_type: Optional[str]) -> bool:
    return str(failure_type or "") in COMPILE_RUNTIME_FAILURE_TYPES


def is_compile_runtime_case_failure(case_result: Any) -> bool:
    """Return True when a failed case should deduct compile/runtime score."""
    if getattr(case_result, "success", False):
        return False

    failure_type = getattr(case_result, "failure_type", None)
    if failure_type:
        return not is_precision_failure_type(failure_type)

    accuracy_result = getattr(case_result, "accuracy_result", None)
    if accuracy_result is None and isinstance(case_result, dict):
        accuracy_result = case_result.get("accuracy")
        failure_type = case_result.get("failure_type")
        if failure_type:
            return not is_precision_failure_type(failure_type)

    if isinstance(accuracy_result, dict):
        if accuracy_result.get("passed"):
            return False
        metadata = accuracy_result.get("metadata") or {}
        explicit = metadata.get("failure_type") or accuracy_result.get("failure_type")
        if explicit:
            return not is_precision_failure_type(str(explicit))
        messages = [
            str(accuracy_result.get("error_msg") or ""),
            str(case_result.get("error_msg") or ""),
        ]
        for output in accuracy_result.get("output_results") or []:
            if isinstance(output, dict):
                if output.get("failure_type"):
                    return not is_precision_failure_type(str(output.get("failure_type")))
                metadata = output.get("metadata")
                if isinstance(metadata, dict) and metadata.get("failure_type"):
                    return not is_precision_failure_type(str(metadata.get("failure_type")))
                messages.append(str(output.get("error_msg") or ""))
        joined = "\n".join(messages)
        if any(marker in joined for marker in STRUCTURAL_FAILURE_MARKERS):
            return True
        return False

    classified = get_accuracy_failure_type(accuracy_result)
    if classified:
        return not is_precision_failure_type(classified)

    return True


# === 性能结果 ===

@dataclass
class PerfResult:
    """统一的性能评测结果

    包含必要字段 + metadata 扩展字段。
    """
    elapsed_us: float = 0.0
    op_times: Dict[str, Any] = field(default_factory=dict)
    error_msg: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_success(self) -> bool:
        """是否成功"""
        return self.error_msg is None and self.elapsed_us > 0

    def get_elapsed_us(self) -> float:
        """获取运行时间"""
        return self.elapsed_us

    def get_op_times(self) -> Dict[str, Any]:
        """获取 profiler 详细数据"""
        return self.op_times

    def get_error_msg(self) -> Optional[str]:
        """获取错误信息"""
        return self.error_msg

    def get_metadata(self) -> Dict[str, Any]:
        """获取扩展元数据"""
        return self.metadata

    def get_baseline_us(self) -> float:
        """获取基线时间（从 metadata）"""
        return self.metadata.get('baseline_us', 0.0)

    def get_t_hw_us(self) -> float:
        """获取理论硬件下界（从 metadata）"""
        return self.metadata.get('t_hw_us', 0.0)

    def get_speedup(self) -> float:
        """计算加速比"""
        baseline = self.get_baseline_us()
        if baseline > 0 and self.elapsed_us > 0:
            return baseline / self.elapsed_us
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'elapsed_us': self.elapsed_us,
            'op_times': self.op_times,
            'error_msg': self.error_msg,
            'metadata': self.metadata,
            'speedup': self.get_speedup(),
        }

    def format_summary(self) -> str:
        """格式化摘要"""
        if self.is_success():
            speedup_str = f", speedup={self.get_speedup():.2f}x" if self.get_speedup() > 0 else ""
            return f"elapsed={self.elapsed_us:.2f}us{speedup_str}"
        else:
            if self.error_msg:
                return f"❌ {self.error_msg}"
            return f"elapsed={self.elapsed_us:.2f}us (no valid time)"


def compute_speedup(elapsed_us: float, baseline_us: float) -> float:
    """计算加速比（便捷函数）"""
    if baseline_us <= 0 or elapsed_us <= 0:
        return 0.0
    return baseline_us / elapsed_us
