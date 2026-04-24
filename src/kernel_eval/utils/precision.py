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
精度验证工具

职责：
1. 提供张量对比验证功能
2. 采用生态算子开源精度标准（MERE/MARE）
3. 通过条件: MERE < threshold, MARE < 10 * threshold

误差指标：
- MERE (平均相对误差) = avg(|actual - golden| / (|golden| + 1e-7))
- MARE (最大相对误差) = max(|actual - golden| / (|golden| + 1e-7))
"""

import math
import torch
from typing import Union, Tuple, Dict, Any, Optional, List
from dataclasses import dataclass


# 精度阈值表（采用生态算子开源精度标准）
PRECISION_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-10,      # ≈ 0.000976
    'bfloat16': 2**-7,      # ≈ 0.007812
    'float32': 2**-13,      # ≈ 0.000122
    'float64': 2**-13,      # 使用float32阈值
    'hifloat32': 2**-11,    # ≈ 0.000488
    'float8_e4m3': 2**-3,   # ≈ 0.125
    'float8_e5m2': 2**-2,   # ≈ 0.25
    'int8': 0,              # 完全相等
    'int16': 0,
    'int32': 0,
    'int64': 0,
    'uint8': 0,
    'uint16': 0,
    'uint32': 0,
    'uint64': 0,
}


@dataclass
class CompareResult:
    """对比结果"""
    passed: bool
    dtype: str
    threshold: float  # 精度阈值
    mere: float = 0.0  # 平均相对误差
    mare: float = 0.0  # 最大相对误差
    max_diff: float = 0.0
    mean_diff: float = 0.0
    mismatch_count: int = 0
    total_count: int = 0
    mismatch_ratio: float = 0.0
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
            'error_msg': self.error_msg,
        }


def get_threshold(dtype_str: str) -> float:
    """获取精度阈值"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in PRECISION_THRESHOLDS:
        # 默认使用 float32 阈值
        return PRECISION_THRESHOLDS['float32']
    return PRECISION_THRESHOLDS[dtype_lower]


def compare_tensors(
    output: Union[torch.Tensor, Tuple, List],
    golden: Union[torch.Tensor, Tuple, List],
    dtype: str = 'float32',
    threshold: Optional[float] = None,
) -> CompareResult:
    """
    对比输出张量与Golden参考结果（采用MERE/MARE标准）

    Args:
        output: 算子输出（单个张量或多张量）
        golden: Golden参考输出（单个张量或多张量）
        dtype: 数据类型字符串
        threshold: 精度阈值（可选，默认根据dtype自动选择）

    Returns:
        CompareResult: 对比结果

    通过条件:
        MERE < threshold 且 MARE < 10 * threshold
    """
    # 获取阈值
    if threshold is None:
        threshold = get_threshold(dtype)

    try:
        # 处理多输出情况
        outputs = _normalize_outputs(output)
        goldens = _normalize_outputs(golden)

        if len(outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"输出数量不匹配: output={len(outputs)}, golden={len(goldens)}"
            )

        # 逐个对比
        all_passed = True
        mere_sum = 0.0
        mare_max = 0.0
        max_diff = 0.0
        mean_diff = 0.0
        mismatch_count = 0
        total_count = 0

        for out_tensor, gold_tensor in zip(outputs, goldens):
            result = _compare_single_tensor(out_tensor, gold_tensor, threshold, dtype)
            all_passed = all_passed and result.passed
            mere_sum += result.mere * result.total_count
            mare_max = max(mare_max, result.mare)
            max_diff = max(max_diff, result.max_diff)
            mean_diff += result.mean_diff * result.total_count
            mismatch_count += result.mismatch_count
            total_count += result.total_count

        if total_count > 0:
            mere = mere_sum / total_count
            mean_diff = mean_diff / total_count
        else:
            mere = 0.0

        # 最终通过条件
        mare_threshold = 10 * threshold
        passed = all_passed and mere < threshold and mare_max < mare_threshold

        return CompareResult(
            passed=passed,
            dtype=dtype,
            threshold=threshold,
            mere=mere,
            mare=mare_max,
            max_diff=max_diff,
            mean_diff=mean_diff,
            mismatch_count=mismatch_count,
            total_count=total_count,
            mismatch_ratio=mismatch_count / total_count if total_count > 0 else 0.0,
        )

    except Exception as e:
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=str(e)
        )


def _normalize_outputs(output: Any) -> List[torch.Tensor]:
    """将输出标准化为张量列表"""
    if isinstance(output, torch.Tensor):
        return [output]
    elif isinstance(output, (tuple, list)):
        result = []
        for item in output:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        result.append(sub_item)
        return result
    else:
        return []


def _compare_single_tensor(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold: float,
    dtype: str,
) -> CompareResult:
    """对比单个张量（计算MERE/MARE）"""
    # Golden runs on CPU (see OpRunner.run_golden) while the AI op runs on
    # NPU. Normalize both sides to CPU so subtract/equal don't trip on
    # mixed devices.
    if output.is_cuda or output.device.type == "npu":
        output = output.cpu()
    if golden.is_cuda or golden.device.type == "npu":
        golden = golden.cpu()

    # 检查形状
    if output.shape != golden.shape:
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=f"形状不匹配: output={output.shape}, golden={golden.shape}"
        )

    # 检查数据类型是否一致
    if output.dtype != golden.dtype:
        # 尝试转换到相同类型
        golden = golden.to(output.dtype)

    # 处理NaN和Inf
    if torch.any(torch.isnan(output)) or torch.any(torch.isnan(golden)):
        nan_out = torch.isnan(output)
        nan_gold = torch.isnan(golden)
        if not torch.all(nan_out == nan_gold):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg="NaN位置不匹配"
            )

    # 处理Inf
    if torch.any(torch.isinf(output)) or torch.any(torch.isinf(golden)):
        inf_out = torch.isinf(output)
        inf_gold = torch.isinf(golden)
        if not torch.all(inf_out == inf_gold):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg="Inf位置不匹配"
            )
        # Inf值需要符号一致
        if not torch.all(torch.sign(output[inf_out]) == torch.sign(golden[inf_gold])):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg="Inf符号不匹配"
            )

    # 对于整数类型，要求完全相等
    if threshold == 0:
        if not torch.equal(output, golden):
            diff = torch.abs(output - golden)
            mismatch_count = int((diff != 0).sum())
            # torch.Tensor.mean 不支持整数 dtype，先升到 float
            diff_float = diff.float() if not diff.is_floating_point() else diff
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
                mean_diff=float(diff_float.mean()) if diff.numel() > 0 else 0.0,
                mismatch_count=mismatch_count,
                total_count=output.numel(),
                mismatch_ratio=mismatch_count / output.numel() if output.numel() > 0 else 0.0,
                error_msg="整数类型要求完全相等"
            )
        return CompareResult(
            passed=True,
            dtype=dtype,
            threshold=threshold,
            mere=0.0,
            mare=0.0,
            max_diff=0.0,
            mean_diff=0.0,
            mismatch_count=0,
            total_count=output.numel(),
            mismatch_ratio=0.0,
        )

    # 计算相对误差: |actual - golden| / (|golden| + 1e-7)
    diff = torch.abs(output - golden)
    golden_abs = torch.abs(golden) + 1e-7  # 防止除0

    # 计算MERE和MARE
    relative_error = diff / golden_abs

    # 排除NaN和Inf位置的误差
    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error))
    valid_relative_error = relative_error[valid_mask]

    if len(valid_relative_error) == 0:
        return CompareResult(
            passed=True,
            dtype=dtype,
            threshold=threshold,
            mere=0.0,
            mare=0.0,
            max_diff=0.0,
            mean_diff=0.0,
            mismatch_count=0,
            total_count=output.numel(),
            mismatch_ratio=0.0,
        )

    mere = float(valid_relative_error.mean())  # 平均相对误差
    mare = float(valid_relative_error.max())    # 最大相对误差

    # 计算绝对差异统计
    valid_diff = diff[valid_mask]
    max_diff = float(valid_diff.max())
    mean_diff = float(valid_diff.mean())
    total_count = output.numel()

    # 通过条件: MERE < threshold 且 MARE < 10 * threshold
    mare_threshold = 10 * threshold
    passed = mere < threshold and mare < mare_threshold

    # 计算不匹配数量（相对误差超过阈值的点）
    mismatch_mask = relative_error > mare_threshold
    mismatch_mask[~valid_mask] = False  # 排除NaN/Inf
    mismatch_count = int(mismatch_mask.sum())

    return CompareResult(
        passed=passed,
        dtype=dtype,
        threshold=threshold,
        mere=mere,
        mare=mare,
        max_diff=max_diff,
        mean_diff=mean_diff,
        mismatch_count=mismatch_count,
        total_count=total_count,
        mismatch_ratio=mismatch_count / total_count if total_count > 0 else 0.0,
    )


def compare_with_custom_threshold(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold_dict: Dict[str, float] = None,
) -> CompareResult:
    """
    使用自定义阈值表对比

    Args:
        output: 算子输出张量
        golden: Golden参考输出张量
        threshold_dict: 自定义阈值表，格式为 {dtype: threshold}

    Returns:
        CompareResult: 对比结果
    """
    if threshold_dict is None:
        threshold_dict = PRECISION_THRESHOLDS

    # 从张量推断dtype
    dtype_str = str(output.dtype).replace('torch.', '')
    threshold = get_threshold(dtype_str)
    if dtype_str.lower() in threshold_dict:
        threshold = threshold_dict[dtype_str.lower()]

    return compare_tensors(output, golden, dtype_str, threshold)