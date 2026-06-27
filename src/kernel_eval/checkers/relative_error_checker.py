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
相对误差精度判断器

采用 MERE/MARE 标准 + 小值域 + 相消处理:
- MERE (平均相对误差) = avg(|actual - golden| / (|golden| + 1e-7))
- MARE (最大相对误差) = max(|actual - golden| / (|golden| + 1e-7))

通过条件: MERE < threshold AND MARE < 10*threshold

整数类型: 精确匹配（内部分支，不拆出独立 Checker）

特殊场景处理:
- 小值域: 当 |golden| < small_value_threshold 时，采用 ErrorCount 比值标准
- 相消处理: 当 output ≈ 0 且 golden 在精度边界附近时，采用 CPU 同精度对照标准

Why: 提供通用的相对误差判断标准，CANN 评测集选用此 Checker
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch

from ..base.checker import CorrectnessChecker
from ..base.result import (
    FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
    FAILURE_TYPE_PRECISION_MISMATCH,
    STRUCTURAL_FAILURE_MARKERS,
    OutputResult,
    AccuracyResult,
    register_output_result,
)
from ..utils.compare import compare_tensors, SingleOutputResult


@dataclass
class RelativeErrorOutputResult(OutputResult):
    """相对误差单输出判断结果

    通用字段 + metadata 扩展（threshold/mere/mare/max_diff/small_value_*/cancel_* 等）。
    """
    index: int
    name: str = ""
    dtype: str = ""
    passed: bool = True
    error_msg: str = ""
    mismatch_count: int = 0
    total_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'index': self.index,
            'name': self.name,
            'dtype': self.dtype,
            'passed': self.passed,
            'error_msg': self.error_msg,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
        }
        d.update(self.metadata)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RelativeErrorOutputResult':
        """从字典反序列化（to_dict 将 metadata 扁平化到顶层，需回收）"""
        _KNOWN_KEYS = {'index', 'name', 'dtype', 'passed', 'error_msg', 'mismatch_count', 'total_count'}
        metadata = {k: v for k, v in d.items() if k not in _KNOWN_KEYS}
        return cls(
            index=d.get('index', 0),
            name=d.get('name', ''),
            dtype=d.get('dtype', ''),
            passed=d.get('passed', True),
            error_msg=d.get('error_msg', ''),
            mismatch_count=d.get('mismatch_count', 0),
            total_count=d.get('total_count', 0),
            metadata=metadata,
        )

    def format_summary(self) -> str:
        """格式化摘要（相对误差标准特有展示逻辑）"""
        dtype_str = f"{self.dtype}[{self.name or self.index}]"
        dtype_category = self.metadata.get('dtype_category', '')

        if dtype_category == 'int':
            if self.passed:
                return f"{dtype_str}: ✅ (exact match)"
            else:
                ratio = self.mismatch_count / max(self.total_count, 1)
                max_abs_diff = self.metadata.get('max_abs_diff', 0)
                return f"{dtype_str}: ❌ mismatch={self.mismatch_count}/{self.total_count} ({ratio:.2%}), max_diff={max_abs_diff}"
        else:
            mere = self.metadata.get('mere', 0)
            mare = self.metadata.get('mare', 0)
            threshold = self.metadata.get('threshold', 0.0)
            if self.passed:
                # 智能格式：小数值用科学计数法，大数值用定点
                mere_str = f"{mere:.6e}" if mere != 0 and mere < 0.001 else f"{mere:.6f}"
                mare_str = f"{mare:.6e}" if mare != 0 and mare < 0.001 else f"{mare:.6f}"
                base = f"{dtype_str}: ✅ MERE={mere_str}, MARE={mare_str}"
                # 小值域/相消有 NPU 错误但兜底判定通过时，追加对比信息
                # 让用户理解"为什么 MARE 很低却存在大量小值域误差"
                sv_err = self.metadata.get('small_value_error_count', 0)
                sv_cpu_err = self.metadata.get('small_value_cpu_error_count', 0)
                sv_total = self.metadata.get('small_value_total_count', 0)
                cancel_err = self.metadata.get('cancel_error_count', 0)
                cancel_cpu_err = self.metadata.get('cancel_cpu_error_count', 0)
                cancel_total = self.metadata.get('cancel_total_count', 0)
                # 有 NPU 错误才显示（全部精确匹配时不显示，避免噪声）
                if sv_err > 0 or cancel_err > 0:
                    sv_ratio = f"{sv_err}/{sv_cpu_err}" if sv_total > 0 else "0"
                    cancel_ratio = f"{cancel_err}/{cancel_cpu_err}" if cancel_total > 0 else "0"
                    base += f", 小值域NPU/CPU错误={sv_ratio}(总{sv_total}), 相消NPU/CPU错误={cancel_ratio}(总{cancel_total})"
                return base
            else:
                mare_threshold = 10 * threshold
                mere_str = f"{mere:.6e}" if mere != 0 and mere < 0.001 else f"{mere:.6f}"
                mare_str = f"{mare:.6e}" if mare != 0 and mare < 0.001 else f"{mare:.6f}"
                base_msg = f"{dtype_str}: ❌ MERE={mere_str}, MARE={mare_str} (threshold={threshold:.6e}, mare_threshold={mare_threshold:.6e})"
                # 失败时也展示小值域/相消兜底判定信息
                # 当相对误差超标且兜底判定也未通过时，只显示 MERE/MARE 会遗漏关键失败原因
                sv_passed = self.metadata.get('small_value_passed', True)
                cancel_passed = self.metadata.get('cancel_passed', True)
                sv_err = self.metadata.get('small_value_error_count', 0)
                sv_cpu_err = self.metadata.get('small_value_cpu_error_count', 0)
                sv_total = self.metadata.get('small_value_total_count', 0)
                cancel_err = self.metadata.get('cancel_error_count', 0)
                cancel_cpu_err = self.metadata.get('cancel_cpu_error_count', 0)
                cancel_total = self.metadata.get('cancel_total_count', 0)
                # 小值域兜底未通过时追加信息
                if not sv_passed and sv_total > 0:
                    base_msg += f", 小值域兜底❌(NPU/CPU错误={sv_err}/{sv_cpu_err}, 总={sv_total})"
                # 相消兜底未通过时追加信息
                if not cancel_passed and cancel_total > 0:
                    base_msg += f", 相消兜底❌(NPU/CPU错误={cancel_err}/{cancel_cpu_err}, 总={cancel_total})"
                # 如果有 error_msg（如 NaN位置不匹配），追加显示
                if self.error_msg:
                    return f"{base_msg}, {self.error_msg}"
                return base_msg


def _convert_to_output_result(sr: SingleOutputResult) -> RelativeErrorOutputResult:
    """将 compare_tensors 的 SingleOutputResult 转换为 RelativeErrorOutputResult"""
    return RelativeErrorOutputResult(
        index=sr.index,
        name=sr.name,
        dtype=sr.dtype,
        passed=sr.passed,
        error_msg=sr.error_msg,
        mismatch_count=sr.mismatch_count,
        total_count=sr.total_count,
        metadata=sr.metadata,
    )


def _classify_compare_failure(error_msg: Optional[str], output_results: List[RelativeErrorOutputResult]) -> str:
    messages = [error_msg or ""]
    for output in output_results:
        messages.append(output.error_msg or "")
        failure_type = (output.metadata or {}).get("failure_type")
        if failure_type:
            return str(failure_type)

    joined = "\n".join(messages)
    if any(marker in joined for marker in STRUCTURAL_FAILURE_MARKERS):
        return FAILURE_TYPE_COMPILE_RUNTIME_ERROR
    return FAILURE_TYPE_PRECISION_MISMATCH


class RelativeErrorChecker(CorrectnessChecker):
    """相对误差精度判断器

    封装 compare_tensors，提供完整的精度判断能力:
    - 浮点: MERE/MARE 标准 + 小值域 + 相消处理
    - 整数: 精确匹配
    - 多输出支持
    """

    def get_name(self) -> str:
        return "relative_error"

    def get_description(self) -> str:
        return "相对误差精度判断器（MERE/MARE + 小值域 + 相消处理）"

    def check(
        self,
        ai_outputs: Union[torch.Tensor, List[torch.Tensor], tuple],
        golden_outputs: Union[torch.Tensor, List[torch.Tensor], tuple],
        dtype: str,
        threshold: float,
        native_outputs: Optional[Union[torch.Tensor, List[torch.Tensor], tuple]] = None,
        ignore_indices: Optional[List[int]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> AccuracyResult:
        """精度判断（多输出）

        Args:
            ai_outputs: AI算子输出（单或多输出）
            golden_outputs: Golden参考输出（FP64精度）
            dtype: 数据类型字符串
            threshold: 精度阈值
            native_outputs: 同精度参考输出（用于小值域比较）
            ignore_indices: 需要忽略对比的输出索引列表
            custom_thresholds: 自定义精度阈值表

        Returns:
            AccuracyResult: 统一格式的精度结果
        """
        compare_result = compare_tensors(
            output=ai_outputs,
            golden=golden_outputs,
            dtype=dtype,
            threshold=threshold,
            native_output=native_outputs,
            ignore_output_indices=ignore_indices,
            custom_thresholds=custom_thresholds,
        )

        # 转换 SingleOutputResult → RelativeErrorOutputResult
        output_results = [_convert_to_output_result(sr) for sr in compare_result.output_results]

        # compare_tensors 异常路径时 output_results 为空列表，
        # 创建一个 fallback 结果，避免 eval JSON 中 output_results=[] 且丢失逐输出细节
        if not output_results and compare_result.error_msg:
            output_results.append(RelativeErrorOutputResult(
                index=0, dtype=dtype, passed=False,
                error_msg=compare_result.error_msg,
                metadata={'failure_type': FAILURE_TYPE_COMPILE_RUNTIME_ERROR},
            ))

        failure_type = None
        if not compare_result.passed:
            failure_type = _classify_compare_failure(compare_result.error_msg, output_results)

        # 聚合指标存入 metadata
        metadata = {
            'checker_name': 'relative_error',
            'mere': compare_result.mere,
            'mare': compare_result.mare,
            'max_diff': compare_result.max_diff,
            'mean_diff': compare_result.mean_diff,
            'mismatch_count': compare_result.mismatch_count,
            'total_count': compare_result.total_count,
            'mismatch_ratio': compare_result.mismatch_ratio,
            'small_value_error_count': compare_result.small_value_error_count,
            'small_value_cpu_error_count': compare_result.small_value_cpu_error_count,
            'small_value_total_count': compare_result.small_value_total_count,
            'cancel_error_count': compare_result.cancel_error_count,
            'cancel_cpu_error_count': compare_result.cancel_cpu_error_count,
            'cancel_total_count': compare_result.cancel_total_count,
            'small_value_passed': compare_result.small_value_passed,
            'cancel_passed': compare_result.cancel_passed,
            'failure_type': failure_type,
        }

        return AccuracyResult(
            passed=compare_result.passed,
            threshold=compare_result.threshold,
            error_msg=compare_result.error_msg,
            output_results=output_results,
            metadata=metadata,
        )


register_output_result('relative_error', RelativeErrorOutputResult)
register_output_result('cann_default', RelativeErrorOutputResult)  # 兼容旧名
