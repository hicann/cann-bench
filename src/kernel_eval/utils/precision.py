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

误差指标（标准公式）：
- MERE (平均相对误差) = avg(|actual - golden| / (|golden| + 1e-7))
- MARE (最大相对误差) = max(|actual - golden| / (|golden| + 1e-7))

特殊场景处理：
- 小值域处理：当 |golden| < small_value_threshold 时，采用 ErrorCount 比值标准
- 相消处理：当 output ≈ 0 且 golden 在精度边界附近时，采用 CPU 同精度对照标准

理论依据：
- IEEE 754 浮点标准：精度位数决定有效数字范围
- Kahan 灾难性相消理论：接近大数相减导致精度丢失
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

# 小值域阈值表（来自 docs/kernel_bench_design_v1.0.md）
# 当 |golden| < small_value_threshold 时，采用小值域标准
SMALL_VALUE_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-11,      # ≈ 4.88e-4
    'bfloat16': 2**-8,      # ≈ 3.91e-3
    'float32': 2**-14,      # ≈ 6.10e-5
    'float64': 2**-14,      # 使用float32阈值
    'hifloat32': 2**-12,    # ≈ 2.44e-4
    'float8_e4m3': 2**-4,   # ≈ 0.0625
    'float8_e5m2': 2**-3,   # ≈ 0.125
}

# 小值域误差阈值表（来自 docs/kernel_bench_design_v1.0.md）
# 当 |golden| < small_value_threshold 且 |actual - golden| > small_value_error 时，计入 ErrorCount
SMALL_VALUE_ERROR_THRESHOLDS: Dict[str, float] = {
    'float16': 2**-16,      # ≈ 1.53e-5
    'bfloat16': 2**-16,     # ≈ 1.53e-5
    'float32': 2**-30,      # ≈ 9.31e-10
    'float64': 2**-30,      # 使用float32阈值
    'hifloat32': 2**-28,    # ≈ 3.73e-9
    'float8_e4m3': 2**-6,   # ≈ 1.56e-2
    'float8_e5m2': 2**-5,   # ≈ 3.12e-2
}

# ============================================================================
# 相消精度边界阈值表（基于 IEEE 754 精度位数理论）
# ============================================================================
#
# 理论依据：
# 1. IEEE 754 标准：不同 dtype 的尾数位数决定了有效数字范围
#    - FP32: 23 位尾数，相对精度 ~2^-23 ≈ 10^-7，约 7 位有效数字
#    - FP16: 10 位尾数，相对精度 ~2^-10 ≈ 10^-3，约 3 位有效数字
#    - BF16: 7 位尾数，相对精度 ~2^-7 ≈ 10^-2，约 2 位有效数字
#
# 2. Kahan 灾难性相消理论：
#    当两个接近的大数相减时，结果的有效位数急剧丢失。
#    例如：FP32 中两个 ~10^4 的数相减得到 ~10^-3，但精度只够表示 7 位，
#    结果相对于原操作数丢失精度，可能输出为 0。
#
# 3. 相消判定条件：
#    - output ≈ 0：因相消丢失精度，结果接近零
#    - golden 在精度边界附近：非零小值，但小于 dtype 能可靠表示的范围
#    - 不在小值域内（排除极小值）
#
# 阈值选择原则：
#    cancel_boundary 应覆盖因精度位数丢失可能导致相消的范围。
#    对于 FP32，当操作数规模 ~10^4 时，结果 ~10^-3 可能相消丢失。
#    设置 cancel_boundary = 2^-8 ≈ 0.004，覆盖常见相消场景。
#
CANCEL_BOUNDARY_THRESHOLDS: Dict[str, float] = {
    # FP32: 精度 ~7 位，设置 2^-8 ≈ 0.004
    # 当 golden < 0.004 且 output ≈ 0 时，可能是 FP32 相消导致
    'float32': 2**-8,       # ≈ 3.91e-3 ≈ 0.004
    'float64': 2**-8,       # 使用float32阈值

    # FP16: 精度 ~3 位，设置 2^-5 ≈ 0.031
    # 当 golden < 0.031 且 output ≈ 0 时，可能是 FP16 相消导致
    'float16': 2**-5,       # ≈ 3.12e-2 ≈ 0.031

    # BF16: 精度 ~2 位，设置 2^-3 ≈ 0.125
    # 当 golden < 0.125 且 output ≈ 0 时，可能是 BF16 相消导致
    'bfloat16': 2**-3,      # ≈ 1.25e-1 ≈ 0.125

    'hifloat32': 2**-8,     # ≈ 3.91e-3
    'float8_e4m3': 2**-1,   # ≈ 0.5
    'float8_e5m2': 2**-0,   # ≈ 1.0
}

# 相消 output 零值判定阈值
# 当 |output| < cancel_zero_threshold 时，判定 output ≈ 0（因相消丢失精度）
CANCEL_ZERO_THRESHOLDS: Dict[str, float] = {
    # 与 cancel_boundary 一致，确保 output 接近零时判定为相消
    'float32': 2**-8,       # ≈ 0.004
    'float64': 2**-8,
    'float16': 2**-5,       # ≈ 0.031
    'bfloat16': 2**-3,      # ≈ 0.125
    'hifloat32': 2**-8,
    'float8_e4m3': 2**-1,
    'float8_e5m2': 2**-0,
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
    small_value_error_count: int = 0  # 小值域 NPU 错误计数
    small_value_cpu_error_count: int = 0  # 小值域 CPU 错误计数
    small_value_total_count: int = 0  # 小值域总计数
    cancel_error_count: int = 0  # 相消位置 NPU 错误计数
    cancel_cpu_error_count: int = 0  # 相消位置 CPU 错误计数
    cancel_total_count: int = 0  # 相消位置总计数
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
            'small_value_error_count': self.small_value_error_count,
            'small_value_cpu_error_count': self.small_value_cpu_error_count,
            'small_value_total_count': self.small_value_total_count,
            'cancel_error_count': self.cancel_error_count,
            'cancel_cpu_error_count': self.cancel_cpu_error_count,
            'cancel_total_count': self.cancel_total_count,
            'error_msg': self.error_msg,
        }


def get_threshold(dtype_str: str) -> float:
    """获取精度阈值"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in PRECISION_THRESHOLDS:
        # 默认使用 float32 阈值
        return PRECISION_THRESHOLDS['float32']
    return PRECISION_THRESHOLDS[dtype_lower]


def get_small_value_threshold(dtype_str: str) -> float:
    """获取小值域阈值"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in SMALL_VALUE_THRESHOLDS:
        return SMALL_VALUE_THRESHOLDS['float32']
    return SMALL_VALUE_THRESHOLDS[dtype_lower]


def get_small_value_error(dtype_str: str) -> float:
    """获取小值域误差阈值"""
    dtype_lower = dtype_str.lower()
    if dtype_lower not in SMALL_VALUE_ERROR_THRESHOLDS:
        return SMALL_VALUE_ERROR_THRESHOLDS['float32']
    return SMALL_VALUE_ERROR_THRESHOLDS[dtype_lower]


def get_cancel_boundary(dtype_str: str) -> float:
    """
    获取相消精度边界阈值（基于 IEEE 754 精度位数理论）

    当 |golden| < cancel_boundary 且 |output| ≈ 0 时，判定为潜在相消位置。

    理论依据：
    - IEEE 754 尾数位数决定了有效数字范围
    - Kahan 灾难性相消理论：接近大数相减导致精度丢失
    """
    dtype_lower = dtype_str.lower()
    if dtype_lower not in CANCEL_BOUNDARY_THRESHOLDS:
        return CANCEL_BOUNDARY_THRESHOLDS['float32']
    return CANCEL_BOUNDARY_THRESHOLDS[dtype_lower]


def get_cancel_zero_threshold(dtype_str: str) -> float:
    """
    获取相消 output 零值判定阈值

    当 |output| < cancel_zero_threshold 时，判定 output ≈ 0（因相消丢失精度）。
    """
    dtype_lower = dtype_str.lower()
    if dtype_lower not in CANCEL_ZERO_THRESHOLDS:
        return CANCEL_ZERO_THRESHOLDS['float32']
    return CANCEL_ZERO_THRESHOLDS[dtype_lower]


def compare_tensors(
    output: Union[torch.Tensor, Tuple, List],
    golden: Union[torch.Tensor, Tuple, List],
    dtype: str = 'float32',
    threshold: Optional[float] = None,
    cpu_output: Optional[Union[torch.Tensor, Tuple, List]] = None,
    ignore_output_indices: Optional[List[int]] = None,
) -> CompareResult:
    """
    对比输出张量与Golden参考结果（采用MERE/MARE标准 + 小值域处理）

    Args:
        output: 算子输出（单个张量或多张量）
        golden: Golden参考输出（单个张量或多张量），通常为 FP64 精度
        dtype: 数据类型字符串
        threshold: 精度阈值（可选，默认根据dtype自动选择）
        cpu_output: CPU 相同精度下的输出（可选，用于小值域比较）
                    如果不提供，则使用 golden 截断到目标精度作为 CPU 输出

    Returns:
        CompareResult: 对比结果

    通过条件:
        正常值域: MERE < threshold 且 MARE < 10 * threshold
        小值域: ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2
    """
    # 获取阈值
    if threshold is None:
        threshold = get_threshold(dtype)

    try:
        # 处理多输出情况
        outputs = _normalize_outputs(output)
        goldens = _normalize_outputs(golden)
        cpu_outputs = _normalize_outputs(cpu_output) if cpu_output is not None else None

        if len(outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"输出数量不匹配: output={len(outputs)}, golden={len(goldens)}"
            )

        if cpu_outputs is not None and len(cpu_outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"CPU输出数量不匹配: cpu_output={len(cpu_outputs)}, golden={len(goldens)}"
            )

        # 逐个对比
        all_passed = True
        mere_sum = 0.0
        mare_max = 0.0
        max_diff = 0.0
        mean_diff = 0.0
        mismatch_count = 0
        total_count = 0
        small_value_error_count = 0
        small_value_cpu_error_count = 0
        small_value_total_count = 0
        cancel_error_count = 0
        cancel_cpu_error_count = 0
        cancel_total_count = 0

        for i, (out_tensor, gold_tensor) in enumerate(zip(outputs, goldens)):
            # 跳过不需要对比的输出
            if ignore_output_indices and i in ignore_output_indices:
                continue

            cpu_tensor = cpu_outputs[i] if cpu_outputs is not None else None
            result = _compare_single_tensor(out_tensor, gold_tensor, threshold, dtype, cpu_tensor)
            is_passed = result.passed
            all_passed = all_passed and is_passed
            mere_sum += result.mere * result.total_count
            mare_max = max(mare_max, result.mare)
            max_diff = max(max_diff, result.max_diff)
            mean_diff += result.mean_diff * result.total_count
            mismatch_count += result.mismatch_count
            total_count += result.total_count
            small_value_error_count += result.small_value_error_count
            small_value_cpu_error_count += result.small_value_cpu_error_count
            small_value_total_count += result.small_value_total_count
            cancel_error_count += result.cancel_error_count
            cancel_cpu_error_count += result.cancel_cpu_error_count
            cancel_total_count += result.cancel_total_count

        if total_count > 0:
            mere = mere_sum / total_count
            mean_diff = mean_diff / total_count
        else:
            mere = 0.0

        # 最终通过条件
        passed = all_passed

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
            small_value_error_count=small_value_error_count,
            small_value_cpu_error_count=small_value_cpu_error_count,
            small_value_total_count=small_value_total_count,
            cancel_error_count=cancel_error_count,
            cancel_cpu_error_count=cancel_cpu_error_count,
            cancel_total_count=cancel_total_count,
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
    cpu_output: Optional[torch.Tensor] = None,
) -> CompareResult:
    """对比单个张量（计算MERE/MARE + 小值域处理）

    生态算子精度标准：
    1. 平均相对误差（MERE）= avg(|actual - golden| / (|golden| + 1e-7))
    2. 最大相对误差（MARE）= max(|actual - golden| / (|golden| + 1e-7))
    3. 通过标准: MERE < threshold 且 MARE < 10 * threshold

    小值域处理（来自 docs/kernel_bench_design_v1.0.md）：
    当 |golden| < small_value_threshold 时，采用小值域通过标准：
    - ErrorCount = 统计满足 (|golden| < threshold 且 |actual - golden| > error) 的位置数
    - 通过标准: ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2

    Args:
        output: NPU/算子输出张量
        golden: Golden参考输出（FP64精度）
        threshold: 精度阈值
        dtype: 数据类型字符串
        cpu_output: CPU 相同精度下的输出（可选）
                   如果不提供，使用 golden 截断到目标精度作为 CPU 输出
    """
    # Golden runs on CPU while the AI op runs on NPU.
    # Normalize both sides to CPU so subtract/equal don't trip on mixed devices.
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

    # 对于整数类型，使用绝对差值容差比较
    if output.dtype in (torch.int8, torch.int16, torch.int32, torch.int64,
                        torch.uint8):
        if torch.equal(output, golden):
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
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
            )

        # 不完全相等时，检查差值是否在容差范围内
        diff = torch.abs(output.int() - golden.int())
        mismatch_mask = diff > max(threshold, 0)
        mismatch_count = int(mismatch_mask.sum())

        diff_float = diff.float()
        if mismatch_count == 0:
            return CompareResult(
                passed=True,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
                mean_diff=float(diff_float.mean()) if diff.numel() > 0 else 0.0,
                mismatch_count=0,
                total_count=output.numel(),
                mismatch_ratio=0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
            )

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
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
            error_msg=f"整数类型差值超出容差({threshold}): {mismatch_count}/{output.numel()} 个元素不匹配",
        )

    # 浮点类型对比
    # Golden 主动截断到 output.dtype，模拟算子输出的精度限制
    target_dtype = output.dtype
    golden_truncated = golden.to(target_dtype).double()
    output_fp64 = output.double()

    # 处理 NaN
    if torch.any(torch.isnan(output_fp64)) or torch.any(torch.isnan(golden_truncated)):
        nan_out = torch.isnan(output_fp64)
        nan_gold = torch.isnan(golden_truncated)
        if not torch.all(nan_out == nan_gold):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg="NaN位置不匹配"
            )

    # 处理 Inf
    # 饱和边界处理：当一方 inf 另一方有限值时，将 inf 替换为 dtype 最大有限值后继续比较。
    # 这处理 fp16 饱和边界场景（NPU fp32→fp16 截断到 inf，golden fp64→fp16 未越界）。
    # 替换后由 MRE/MARE 决定 pass/fail——只有底层数值本身足够接近才能通过。
    replaced_inf = False
    inf_match_mask = torch.zeros_like(output_fp64, dtype=torch.bool)  # 初始化 inf 匹配掩码
    if torch.any(torch.isinf(output_fp64)) or torch.any(torch.isinf(golden_truncated)):
        inf_out = torch.isinf(output_fp64)
        inf_gold = torch.isinf(golden_truncated)
        inf_mismatch = inf_out != inf_gold

        if torch.any(inf_mismatch):
            # 获取目标 dtype 的最大有限值
            if target_dtype == torch.float16:
                max_finite = float(torch.finfo(torch.float16).max)  # 65504.0
            elif target_dtype == torch.bfloat16:
                max_finite = float(torch.finfo(torch.bfloat16).max)  # ~3.389e38
            elif target_dtype == torch.float32:
                max_finite = float(torch.finfo(torch.float32).max)
            else:
                max_finite = float(torch.finfo(target_dtype).max)

            mismatch_count = int(inf_mismatch.sum())
            print(f"[inf_sat] {mismatch_count} element(s) saturated to inf on one side only, "
                  f"replacing inf with {max_finite} and continuing comparison")

            # 替换 inf 为最大有限值（保留符号）
            if torch.any(inf_out & ~inf_gold):
                mask = inf_out & ~inf_gold
                output_fp64[mask] = torch.sign(output_fp64[mask]) * max_finite
                replaced_inf = True
            if torch.any(inf_gold & ~inf_out):
                mask = inf_gold & ~inf_out
                golden_truncated[mask] = torch.sign(golden_truncated[mask]) * max_finite
                replaced_inf = True

        # 双方都 inf 且符号相同的位置，直接视为匹配并排除后续比较
        both_inf = inf_out & inf_gold
        if torch.any(both_inf):
            if not torch.all(torch.sign(output_fp64[both_inf]) == torch.sign(golden_truncated[both_inf])):
                return CompareResult(
                    passed=False,
                    dtype=dtype,
                    threshold=threshold,
                    error_msg="Inf符号不匹配"
                )
            inf_match_mask[both_inf] = True  # 更新 inf 匹配掩码

    # 计算相对误差（标准公式）
    # 公式: |actual - golden| / (|golden| + 1e-7)
    diff = torch.abs(output_fp64 - golden_truncated)
    golden_abs = torch.abs(golden_truncated)
    denominator = golden_abs + 1e-7  # 防止除0

    relative_error = diff / denominator

    # 排除 NaN、Inf 和匹配的 Inf 位置
    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error) | inf_match_mask)
    valid_relative_error = relative_error[valid_mask]

    if len(valid_relative_error) == 0:
        # 所有位置都是 NaN/Inf 且匹配，视为通过
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
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
        )

    mere = float(valid_relative_error.mean())
    mare = float(valid_relative_error.max())

    # 计算绝对差异统计
    valid_diff = diff[valid_mask]
    max_diff = float(valid_diff.max()) if len(valid_diff) > 0 else 0.0
    mean_diff = float(valid_diff.mean()) if len(valid_diff) > 0 else 0.0
    total_count = output.numel()

    # === 小值域处理 ===
    # 获取小值域阈值
    small_value_threshold = get_small_value_threshold(dtype)
    small_value_error = get_small_value_error(dtype)

    # 筛选小值域位置: |golden| < small_value_threshold
    small_value_mask = golden_abs < small_value_threshold
    small_value_mask[~valid_mask] = False  # 排除NaN/Inf
    small_value_total_count = int(small_value_mask.sum())

    # 小值域 NPU 错误计数: |golden| < threshold 且 |output - golden| > error
    small_value_npu_error_mask = small_value_mask & (diff > small_value_error)
    small_value_error_count = int(small_value_npu_error_mask.sum())

    # 小值域 CPU 错误计数
    # 如果提供了 cpu_output，使用它；否则使用 golden 截断到目标精度作为 CPU 输出
    # 注意：CPU 差异是与原始 FP64 golden 比较，而不是截断后的 golden
    if cpu_output is not None:
        cpu_output_fp64 = cpu_output.double()
        # 使用原始 golden (FP64) 进行比较，而不是 golden_truncated
        cpu_diff = torch.abs(cpu_output_fp64 - golden.double())
    else:
        # golden 截断到目标精度后再升到 FP64，这就是 CPU 相同精度下的"理想"输出
        cpu_output_fp64 = golden.to(target_dtype).double()
        cpu_diff = torch.abs(cpu_output_fp64 - golden.double())

    # CPU 小值域错误计数: |golden| < threshold 且 |cpu_output - golden| > error
    # 使用原始 golden (FP64) 的绝对值判断小值域
    cpu_golden_abs = torch.abs(golden.double())
    cpu_small_value_mask = cpu_golden_abs < small_value_threshold
    cpu_small_value_mask[~valid_mask] = False
    cpu_small_value_error_mask = cpu_small_value_mask & (cpu_diff > small_value_error)
    small_value_cpu_error_count = int(cpu_small_value_error_mask.sum())

    # === 正常值域通过条件 ===
    # 排除小值域位置后计算 MERE 和 MARE
    # 因为小值域位置有自己的判断标准，不应该影响正常值域的误差计算
    normal_value_mask = ~small_value_mask & valid_mask
    normal_relative_error = relative_error[normal_value_mask]

    if len(normal_relative_error) > 0:
        normal_mere = float(normal_relative_error.mean())
        normal_mare = float(normal_relative_error.max())
    else:
        # 所有位置都在小值域内，正常值域直接通过
        normal_mere = 0.0
        normal_mare = 0.0

    mare_threshold = 10 * threshold
    normal_passed = normal_mere < threshold and normal_mare < mare_threshold

    # === 相消位置处理（基于 IEEE 754 精度位数理论 + CPU 同精度对照）===
    #
    # 理论依据：
    # 1. IEEE 754 标准：FP32 有 23 位尾数，相对精度 ~2^-23 ≈ 10^-7
    #    当两个接近的大数相减时，结果的有效位数急剧丢失（Kahan 灾难性相消）
    # 2. 相消现象特征：
    #    - output ≈ 0（因精度位数不足，相消结果丢失）
    #    - golden 在精度边界附近（非零小值，如 FP32 的 10^-3）
    #    - 不在小值域内（排除极小值场景）
    #
    # 检测条件：
    # - |output| < cancel_zero_threshold（output 因相消接近零）
    # - |golden| < cancel_boundary（golden 在精度边界附近）
    # - |golden| >= small_value_threshold（排除小值域）
    # - 是有效值（非 NaN/Inf）

    # 获取相消阈值
    cancel_boundary = get_cancel_boundary(dtype)
    cancel_zero_threshold = get_cancel_zero_threshold(dtype)

    # 检测相消位置
    output_abs = torch.abs(output_fp64)
    output_near_zero = output_abs < cancel_zero_threshold
    golden_in_cancel_range = (golden_abs < cancel_boundary) & (golden_abs >= small_value_threshold)
    cancel_mask = output_near_zero & golden_in_cancel_range & valid_mask
    cancel_total_count = int(cancel_mask.sum())

    # 相消位置"错误"判断：相对误差超过 mare_threshold
    cancel_npu_error_mask = cancel_mask & (relative_error > mare_threshold)
    cancel_error_count = int(cancel_npu_error_mask.sum())

    # CPU 相对误差超标计数：CPU 在相消位置是否也有相同的精度限制
    cpu_relative_error = cpu_diff / (cpu_golden_abs + 1e-7)
    cancel_cpu_error_mask = cancel_mask & (cpu_relative_error > mare_threshold)
    cancel_cpu_error_count = int(cancel_cpu_error_mask.sum())

    # 相消位置通过标准：ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2
    # 如果 NPU 和 CPU 都有相同数量的"错误"位置，说明是 dtype 精度限制而非算子 bug
    if cancel_total_count > 0:
        max_cpu_cancel_error = max(cancel_cpu_error_count, 1)
        cancel_ratio = cancel_error_count / max_cpu_cancel_error
        cancel_passed = cancel_ratio <= 2
    else:
        cancel_passed = True

    # 计算不匹配数量（相对误差超过阈值的点，排除小值域和相消位置）
    # 如果相消位置通过了（NPU 和 CPU 一致），则不计入 mismatch
    mismatch_mask = relative_error > mare_threshold
    mismatch_mask[~valid_mask] = False
    mismatch_mask[small_value_mask] = False  # 排除小值域位置
    # 排除相消位置中 NPU 和 CPU 都"出错"的位置（这些是真正的精度限制）
    # 但保留 NPU "出错"而 CPU 不"出错"的位置（这些可能是算子 bug）
    if cancel_total_count > 0 and cancel_passed:
        # 相消通过：排除所有相消位置
        mismatch_mask[cancel_mask] = False
    mismatch_count = int(mismatch_mask.sum())

    # 如果所有不匹配位置都因相消通过了，更新 normal_passed
    if mismatch_count == 0 and cancel_total_count > 0 and cancel_passed:
        normal_passed = True

    # === 小值域通过标准 ===
    # ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2
    # 来自 docs/kernel_bench_design_v1.0.md
    if small_value_total_count > 0:
        # NPU ErrorCount 不应超过 CPU ErrorCount 的 2 倍
        # 如果 CPU ErrorCount = 0，则 NPU ErrorCount 必须 ≤ 2
        max_cpu_error = max(small_value_cpu_error_count, 1)
        small_value_ratio = small_value_error_count / max_cpu_error
        small_value_passed = small_value_ratio <= 2
    else:
        # 无小值域位置，直接通过
        small_value_passed = True

    # 最终通过条件: 正常值域通过 且 小值域通过 且 相消位置通过
    passed = normal_passed and small_value_passed and cancel_passed

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
        small_value_error_count=small_value_error_count,
        small_value_cpu_error_count=small_value_cpu_error_count,
        small_value_total_count=small_value_total_count,
        cancel_error_count=cancel_error_count,
        cancel_cpu_error_count=cancel_cpu_error_count,
        cancel_total_count=cancel_total_count,
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