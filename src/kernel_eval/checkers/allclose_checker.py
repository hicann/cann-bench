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
AllClose 精度判断器

使用 torch.allclose(atol/rtol) 对比，不区分 float/int 类型，
统一用 allclose 处理所有 dtype。

Why: 提供通用的容差判定标准，StanfordBench 选用此 Checker，
     其他评测集也可选用
"""

import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from ..base.checker import CorrectnessChecker
from ..base.result import (
    FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
    FAILURE_TYPE_PRECISION_MISMATCH,
    OutputResult,
    AccuracyResult,
    register_output_result,
)


@dataclass
class AllCloseOutputResult(OutputResult):
    """AllClose 单输出判断结果"""
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
    def from_dict(cls, d: Dict[str, Any]) -> 'AllCloseOutputResult':
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
        """格式化摘要（allclose 标准特有展示逻辑）"""
        dtype_str = f"{self.dtype}[{self.name or self.index}]"
        atol = self.metadata.get('atol', 0)
        rtol = self.metadata.get('rtol', 0)
        if self.passed:
            return f"{dtype_str}: ✅ (allclose passed, atol={atol}, rtol={rtol})"
        else:
            return f"{dtype_str}: ❌ (allclose failed, atol={atol}, rtol={rtol})"


class AllCloseChecker(CorrectnessChecker):
    """AllClose 精度判断器

    使用 torch.allclose 进行对比。
    """

    def get_name(self) -> str:
        return "allclose"

    def get_description(self) -> str:
        return "AllClose 精度判断器（torch.allclose, atol/rtol）"

    def check(
        self,
        ai_outputs: Union[torch.Tensor, List[torch.Tensor]],
        golden_outputs: Union[torch.Tensor, List[torch.Tensor]],
        dtype: str,
        threshold: float = 0.01,
        native_outputs=None,
        ignore_indices: Optional[List[int]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None,
    ) -> AccuracyResult:
        """使用 allclose 对比

        Args:
            ai_outputs: AI 算子输出
            golden_outputs: Golden 参考输出
            dtype: 数据类型
            threshold: 默认精度阈值
            native_outputs: 未使用
            ignore_indices: 需要忽略对比的输出索引
            custom_thresholds: 自定义阈值 {'atol': x, 'rtol': y}

        Returns:
            AccuracyResult: 精度对比结果
        """
        ai_list = self._normalize_outputs(ai_outputs)
        golden_list = self._normalize_outputs(golden_outputs)

        # 检查输出数量是否匹配
        if len(ai_list) != len(golden_list):
            return AccuracyResult(
                passed=False,
                error_msg=f"输出数量不匹配: ai={len(ai_list)}, golden={len(golden_list)}",
                metadata={
                    'checker_name': 'allclose',
                    'failure_type': FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
                },
            )

        # 获取阈值
        atol = custom_thresholds.get('atol', threshold) if custom_thresholds else threshold
        rtol = custom_thresholds.get('rtol', threshold) if custom_thresholds else threshold

        results: List[AllCloseOutputResult] = []
        all_passed = True
        structural_failure = False

        for i, (ai, golden) in enumerate(zip(ai_list, golden_list)):
            # 检查是否需要忽略
            if ignore_indices and i in ignore_indices:
                results.append(AllCloseOutputResult(
                    index=i,
                    name=f"output_{i}",
                    dtype=dtype,
                    passed=True,
                    error_msg="(跳过对比)",
                ))
                continue

            # 确保 tensor 在 CPU 上进行对比
            ai_cpu = self._ensure_cpu(ai)
            golden_cpu = self._ensure_cpu(golden)

            # 形状不匹配直接判失败，避免 torch.allclose 抛异常
            if ai_cpu.shape != golden_cpu.shape:
                results.append(AllCloseOutputResult(
                    index=i,
                    name=f"output_{i}",
                    dtype=dtype,
                    passed=False,
                    error_msg=f"形状不匹配: ai={ai_cpu.shape}, golden={golden_cpu.shape}",
                    metadata={
                        'atol': atol,
                        'rtol': rtol,
                        'failure_type': FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
                    },
                ))
                all_passed = False
                structural_failure = True
                continue

            # 使用 allclose
            passed = torch.allclose(ai_cpu, golden_cpu, rtol=rtol, atol=atol)

            # 计算 mismatch_count
            diff = torch.abs(ai_cpu - golden_cpu)
            mismatch_count = int((diff > atol + rtol * torch.abs(golden_cpu)).sum().item()) if ai_cpu.numel() > 0 else 0
            total_count = ai_cpu.numel()

            results.append(AllCloseOutputResult(
                index=i,
                name=f"output_{i}",
                dtype=dtype,
                passed=passed,
                mismatch_count=mismatch_count,
                total_count=total_count,
                metadata={'atol': atol, 'rtol': rtol},
            ))

            if not passed:
                all_passed = False

        # 聚合统计
        agg_mismatch_count = sum(r.mismatch_count for r in results)
        agg_total_count = sum(r.total_count for r in results)
        agg_mismatch_ratio = agg_mismatch_count / agg_total_count if agg_total_count > 0 else 0.0

        return AccuracyResult(
            passed=all_passed,
            output_results=results,
            threshold=None,
            error_msg=None if all_passed else "精度不达标（allclose 失败）",
            metadata={
                'checker_name': 'allclose',
                'atol': atol,
                'rtol': rtol,
                'mismatch_count': agg_mismatch_count,
                'total_count': agg_total_count,
                'mismatch_ratio': agg_mismatch_ratio,
                'failure_type': (
                    None if all_passed else
                    FAILURE_TYPE_COMPILE_RUNTIME_ERROR if structural_failure else
                    FAILURE_TYPE_PRECISION_MISMATCH
                ),
            },
        )


register_output_result('allclose', AllCloseOutputResult)
