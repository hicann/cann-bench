# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .compare import TensorCompareResult


@dataclass
class PrecisionSummary:
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    mean_rel_diff: float
    threshold_abs: float
    threshold_rel: float
    message: str
    # Compatibility fields for existing JSON consumers. Values are raw metrics.
    mare_ratio: float = 0.0
    mere_ratio: float = 0.0
    rmse_ratio: float = 0.0
    threshold_mare_ratio: float = 0.0
    threshold_mere_ratio: float = 0.0
    threshold_rmse_ratio: float = 0.0
    small_value_count: int = 0
    small_value_error_count: int = 0
    threshold_small_value_error_count: int = 0
    small_value_native_error_count: int = 0
    cancel_count: int = 0
    cancel_error_count: int = 0
    cancel_native_error_count: int = 0


def judge_precision(
    result: TensorCompareResult,
    *,
    max_abs_error: float,
    max_rel_error: float,
    tolerance: dict[str, Any] | None = None,
) -> PrecisionSummary:
    """Return the decision already made by the typed ecosystem comparator.

    Legacy per-case Huawei L0/L1/L2 values are deliberately ignored: those
    values describe commercial double-benchmark ratios, while Ascend Arena has
    no third-party benchmark output. Quantized integer AE is applied before
    this function by compare_outputs_for_case().
    """
    del tolerance
    thresholds = [output.threshold for output in result.output_results if output.threshold > 0]
    threshold = max(thresholds, default=0.0)
    return PrecisionSummary(
        passed=result.passed,
        max_abs_diff=result.max_abs_diff,
        max_rel_diff=result.max_rel_diff,
        mean_abs_diff=result.mean_abs_diff,
        mean_rel_diff=result.mean_rel_diff,
        threshold_abs=max_abs_error,
        threshold_rel=max_rel_error,
        message=result.message if result.passed else f"precision_failed:{result.message}",
        mare_ratio=result.mare_ratio,
        mere_ratio=result.mere_ratio,
        rmse_ratio=result.rmse_ratio,
        threshold_mare_ratio=10.0 * threshold,
        threshold_mere_ratio=threshold,
        small_value_count=result.small_value_count,
        small_value_error_count=result.small_value_error_count,
        small_value_native_error_count=result.small_value_native_error_count,
        cancel_count=result.cancel_count,
        cancel_error_count=result.cancel_error_count,
        cancel_native_error_count=result.cancel_native_error_count,
    )
