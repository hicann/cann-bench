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
精度评测器

职责：
1. 对比AI生成算子输出与Golden参考输出
2. 采用生态算子开源精度标准（MERE/MARE）
3. 支持CPU fp64 Golden计算（避免NPU溢出污染）
4. 支持二次验证机制（防止缓存作弊）
5. 输出精度对比结果

通过条件: MERE < threshold, MARE < 10 * threshold

注意：AccuracyResult 现在从 checkers 模块导入，为统一接口。
      trial（验证轮次）存储在 metadata 中。
"""

from typing import Any, Dict, List, Optional, Union, Tuple

import torch

from ..utils.thresholds import PRECISION_THRESHOLDS
from ..security.type_checker import check_multi_output
from ..base.result import AccuracyResult
from ..registry.checker_registry import get_correctness_checker, list_correctness_checkers


class AccuracyEvaluator:
    """精度评测器"""

    def __init__(
        self,
        custom_thresholds: Dict[str, float] = None,
        checker_name: str = "relative_error",
    ):
        """
        Args:
            custom_thresholds: 自定义精度阈值表，格式为 {dtype: threshold}
            checker_name: 精度判断器名称，默认 "relative_error"
        """
        self.thresholds = custom_thresholds or PRECISION_THRESHOLDS
        self.checker_name = checker_name
        self._checker = None

    def _get_checker(self):
        """获取精度判断器（延迟加载）"""
        if self._checker is None:
            self._checker = get_correctness_checker(self.checker_name)
            if self._checker is None:
                raise ValueError(f"未找到精度判断器: {self.checker_name}, "
                                 f"已注册: {list_correctness_checkers()}")
        return self._checker

    def evaluate(
        self,
        ai_output: Union[torch.Tensor, Tuple, List],
        golden_output: Union[torch.Tensor, Tuple, List],
        dtype: str,
        trial: int = 1,
        custom_thresholds: Dict[str, float] = None,
        native_output: Union[torch.Tensor, Tuple, List] = None,
        ignore_output_indices: List[int] = None,
        checker_name: Optional[str] = None,
    ) -> AccuracyResult:
        """
        评测AI算子输出的精度（采用MERE/MARE标准 + 小值域处理）

        Args:
            ai_output: AI生成算子的输出
            golden_output: Golden参考输出（FP64精度）
            dtype: 数据类型字符串
            trial: 验证轮次（1或2），存储在 metadata 中
            custom_thresholds: 自定义精度阈值表，优先级高于全局配置
            native_output: 同精度参考输出（用于小值域比较，可为 AI 算子的同精度 golden 执行结果）
            ignore_output_indices: 需要忽略对比的输出索引列表
            checker_name: 精度判断器名称（可选，覆盖实例配置）

        Returns:
            AccuracyResult: 精度评测结果（统一接口）
        """
        # 类型检查（安全防护）
        try:
            check_multi_output(ai_output)
            check_multi_output(golden_output)
        except RuntimeError as e:
            return AccuracyResult(
                passed=False,
                threshold=0,
                error_msg=str(e),
                metadata={'trial': trial},
            )

        # 获取阈值（优先使用自定义阈值）
        threshold = self._get_threshold(dtype, custom_thresholds)

        # 获取精度判断器
        checker = self._get_checker() if checker_name is None else get_correctness_checker(checker_name)
        if checker is None:
            return AccuracyResult(
                passed=False,
                threshold=threshold,
                error_msg=f"未找到精度判断器: {checker_name or self.checker_name}",
                metadata={'trial': trial},
            )

        # 执行精度判断
        result = checker.check(
            ai_outputs=ai_output,
            golden_outputs=golden_output,
            dtype=dtype,
            threshold=threshold,
            native_outputs=native_output,
            ignore_indices=ignore_output_indices,
            custom_thresholds=custom_thresholds,
        )

        # 在 metadata 中添加 trial
        metadata = result.get_metadata()
        metadata['trial'] = trial

        return result

    def evaluate_batch(
        self,
        ai_outputs: List,
        golden_outputs: List,
        dtypes: List[str],
    ) -> List[AccuracyResult]:
        """
        批量评测多个输出

        Args:
            ai_outputs: AI算子输出列表
            golden_outputs: Golden输出列表
            dtypes: 数据类型列表

        Returns:
            List[AccuracyResult]: 精度评测结果列表
        """
        results = []
        for i, dtype in enumerate(dtypes):
            ai_out = ai_outputs[i] if i < len(ai_outputs) else None
            gold_out = golden_outputs[i] if i < len(golden_outputs) else None

            if ai_out is None or gold_out is None:
                results.append(AccuracyResult(
                    passed=False,
                    threshold=0,
                    error_msg=f"输出索引{i}不存在",
                ))
            else:
                results.append(self.evaluate(ai_out, gold_out, dtype))

        return results

    def _get_threshold(self, dtype: str, custom_thresholds: Dict[str, float] = None) -> float:
        """获取精度阈值（优先使用自定义阈值）"""
        # 优先使用调用时传入的自定义阈值
        thresholds = custom_thresholds or self.thresholds
        dtype_lower = dtype.lower()
        if dtype_lower in thresholds:
            return thresholds[dtype_lower]
        # 默认使用 float32 阈值
        return thresholds.get('float32', 2**-13)

    def check_output_shape(self, ai_output: torch.Tensor, expected_shape: tuple) -> bool:
        """检查输出形状是否匹配"""
        if isinstance(ai_output, torch.Tensor):
            return ai_output.shape == expected_shape
        return False

    def check_output_dtype(self, ai_output: torch.Tensor, expected_dtype: str) -> bool:
        """检查输出数据类型是否匹配"""
        if isinstance(ai_output, torch.Tensor):
            from ..utils.dtype_mapper import torch_dtype_to_str
            actual_dtype = torch_dtype_to_str(ai_output.dtype)
            return actual_dtype.lower() == expected_dtype.lower()
        return False