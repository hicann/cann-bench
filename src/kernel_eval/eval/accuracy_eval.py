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
精度评测器

职责：
1. 对比AI生成算子输出与Golden参考输出
2. 采用生态算子开源精度标准（MERE/MARE）
3. 支持CPU fp64 Golden计算（避免NPU溢出污染）
4. 支持二次验证机制（防止缓存作弊）
5. 输出精度对比结果

通过条件: MERE < threshold, MARE < 10 * threshold

参考evaluation/core/precision_checker.py
"""

import torch
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass

from ..utils.precision import compare_tensors, CompareResult, PRECISION_THRESHOLDS
from ..security.type_checker import check_output_type, check_multi_output


@dataclass
class AccuracyResult:
    """精度评测结果"""
    passed: bool
    dtype: str
    threshold: float  # 精度阈值
    mere: float  # 平均相对误差
    mare: float  # 最大相对误差
    max_diff: float = 0.0
    mean_diff: float = 0.0
    mismatch_count: int = 0
    total_count: int = 0
    mismatch_ratio: float = 0.0
    trial: int = 1  # 验证轮次（1或2）
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'dtype': self.dtype,
            'threshold': self.threshold,
            'mere': self.mere,
            'mare': self.mare,
            'max_diff': self.max_diff,
            'mean_diff': self.mean_diff,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
            'mismatch_ratio': self.mismatch_ratio,
            'trial': self.trial,
            'error_msg': self.error_msg,
        }


class AccuracyEvaluator:
    """精度评测器"""

    def __init__(self, custom_thresholds: Dict[str, float] = None):
        """
        Args:
            custom_thresholds: 自定义精度阈值表，格式为 {dtype: threshold}
        """
        self.thresholds = custom_thresholds or PRECISION_THRESHOLDS

    def compute_golden_fp64(
        self,
        golden_fn: Callable,
        inputs: List[Any],
        param_builder: Any,
        case: Any
    ) -> torch.Tensor:
        """
        在CPU fp64精度下计算Golden参考值

        原理：Golden函数在CPU fp64下计算，比NPU原生dtype精度更高，
        避免溢出/下溢同时污染参考值。精度对比时双方都cast回fp32。

        Args:
            golden_fn: Golden函数
            inputs: 输入张量列表
            param_builder: 参数构建器
            case: 用例信息

        Returns:
            Golden输出Tensor
        """
        # 转换输入到CPU fp64 —— 只对浮点 tensor 做 upcast，整型/bool 保持原 dtype。
        # 否则像 GCD（int16/int32 输入）、CrossEntropyLoss（int64 target）这类
        # 算子的 golden 会在 CPU 上因为 dtype 错误报错，例如：
        #   "gcd_cpu not implemented for 'Double'"
        #   "expected scalar type Long but found Double"
        def _to_fp64_golden(t: torch.Tensor) -> torch.Tensor:
            t = t.cpu()
            return t.double() if t.is_floating_point() else t

        fp64_inputs = []
        for item in inputs:
            if isinstance(item, torch.Tensor):
                fp64_inputs.append(_to_fp64_golden(item))
            elif isinstance(item, (list, tuple)):
                fp64_inputs.append([
                    _to_fp64_golden(sub) if isinstance(sub, torch.Tensor) else sub
                    for sub in item
                ])
            else:
                fp64_inputs.append(item)

        # 构建调用参数
        params = param_builder.build_call_params(golden_fn, case, fp64_inputs)

        # 执行Golden函数
        with torch.no_grad():
            golden_out = golden_fn(**params)

        # 处理多输出情况
        if isinstance(golden_out, (tuple, list)):
            golden_out = golden_out[0] if golden_out else None

        return golden_out

    def evaluate(
        self,
        ai_output: Union[torch.Tensor, Tuple, List],
        golden_output: Union[torch.Tensor, Tuple, List],
        dtype: str,
        trial: int = 1
    ) -> AccuracyResult:
        """
        评测AI算子输出的精度（采用MERE/MARE标准）

        Args:
            ai_output: AI生成算子的输出
            golden_output: Golden参考输出
            dtype: 数据类型字符串
            trial: 验证轮次（1或2）

        Returns:
            AccuracyResult: 精度评测结果
        """
        # 类型检查（安全防护）
        try:
            check_multi_output(ai_output)
            check_multi_output(golden_output)
        except RuntimeError as e:
            return AccuracyResult(
                passed=False,
                dtype=dtype,
                threshold=0,
                mere=0,
                mare=0,
                trial=trial,
                error_msg=str(e)
            )

        # 获取阈值
        threshold = self._get_threshold(dtype)

        # 对比张量（双方都转fp32计算MERE/MARE）
        compare_result = compare_tensors(ai_output, golden_output, dtype, threshold)

        return AccuracyResult(
            passed=compare_result.passed,
            dtype=compare_result.dtype,
            threshold=compare_result.threshold,
            mere=compare_result.mere,
            mare=compare_result.mare,
            max_diff=compare_result.max_diff,
            mean_diff=compare_result.mean_diff,
            mismatch_count=compare_result.mismatch_count,
            total_count=compare_result.total_count,
            mismatch_ratio=compare_result.mismatch_ratio,
            trial=trial,
            error_msg=compare_result.error_msg,
        )

    def evaluate_with_retry(
        self,
        golden_fn: Callable,
        custom_fn: Callable,
        inputs: List[Any],
        param_builder: Any,
        case: Any,
        data_gen: Any,
        device: str,
        dtype: str,
        perturb_input: bool = True
    ) -> AccuracyResult:
        """
        执行二次验证评测

        原理：用新鲜输入重跑一次，防止缓存作弊。
        如果submission缓存第一次结果或翻转"computed-once"标志，
        第二次用不同输入会产生垃圾结果。

        Args:
            golden_fn: Golden函数
            custom_fn: AI算子函数
            inputs: 第一轮输入张量
            param_builder: 参数构建器
            case: 用例信息
            data_gen: 数据生成器
            device: 设备类型
            dtype: 数据类型
            perturb_input: 是否微扰输入（防止seed重复）

        Returns:
            AccuracyResult: 最终评测结果
        """
        # 第一轮验证
        result1 = self._single_eval(
            golden_fn, custom_fn, inputs, param_builder, case, device, dtype, trial=1
        )

        if not result1.passed:
            return result1

        # 第二轮验证：新鲜输入
        fresh_inputs = data_gen.generate_input_tensors_from_case(
            case.input_shapes, case.dtypes, case.value_ranges
        )

        # 微扰输入，防止seed重复导致相同数据
        if perturb_input:
            for item in fresh_inputs:
                if isinstance(item, torch.Tensor) and item.is_floating_point():
                    item.add_(0.01)
                    break
                elif isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, torch.Tensor) and sub.is_floating_point():
                            sub.add_(0.01)
                            break

        result2 = self._single_eval(
            golden_fn, custom_fn, fresh_inputs, param_builder, case, device, dtype, trial=2
        )

        # 如果第二轮失败，返回第二轮结果
        if not result2.passed:
            return result2

        # 两轮都通过，返回第一轮结果（使用原始输入）
        return result1

    def _single_eval(
        self,
        golden_fn: Callable,
        custom_fn: Callable,
        inputs: List[Any],
        param_builder: Any,
        case: Any,
        device: str,
        dtype: str,
        trial: int
    ) -> AccuracyResult:
        """单轮评测"""
        try:
            # Golden计算（CPU fp64）
            golden_out = self.compute_golden_fp64(golden_fn, inputs, param_builder, case)

            # Custom算子计算（NPU或CPU）
            if device.startswith('npu'):
                npu_inputs = []
                for item in inputs:
                    if isinstance(item, torch.Tensor):
                        npu_inputs.append(item.to(device))
                    elif isinstance(item, (list, tuple)):
                        npu_inputs.append([sub.to(device) if isinstance(sub, torch.Tensor) else sub for sub in item])
                    else:
                        npu_inputs.append(item)
                flat_inputs = [t if isinstance(t, torch.Tensor) else t[0] for t in npu_inputs]
            else:
                flat_inputs = [t if isinstance(t, torch.Tensor) else t[0] for t in inputs]

            with torch.no_grad():
                custom_out = custom_fn(*flat_inputs)

            # 处理多输出
            if isinstance(golden_out, (tuple, list)):
                golden_out = golden_out[0] if golden_out else None
            if isinstance(custom_out, (tuple, list)):
                custom_out = custom_out[0] if custom_out else None

            # 类型检查
            if custom_out is not None:
                check_output_type(custom_out, torch.Tensor, strict=True)

            # 精度对比（双方都转CPU）
            if golden_out is None or custom_out is None:
                return AccuracyResult(
                    passed=False,
                    dtype=dtype,
                    threshold=0,
                    mere=0,
                    mare=0,
                    trial=trial,
                    error_msg="输出为空"
                )

            return self.evaluate(custom_out.cpu(), golden_out.cpu().float(), dtype, trial)

        except Exception as e:
            return AccuracyResult(
                passed=False,
                dtype=dtype,
                threshold=0,
                mere=0,
                mare=0,
                trial=trial,
                error_msg=str(e)[:200]
            )

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
                    dtype=dtype,
                    threshold=0,
                    mere=0,
                    mare=0,
                    error_msg=f"输出索引{i}不存在"
                ))
            else:
                results.append(self.evaluate(ai_out, gold_out, dtype))

        return results

    def _get_threshold(self, dtype: str) -> float:
        """获取精度阈值"""
        dtype_lower = dtype.lower()
        if dtype_lower in self.thresholds:
            return self.thresholds[dtype_lower]
        # 默认使用 float32 阈值
        return self.thresholds.get('float32', 2**-13)

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