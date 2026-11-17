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

from typing import Any


from kernel_eval.utils.thresholds import (
    PRECISION_THRESHOLDS, SMALL_VALUE_THRESHOLDS, SMALL_VALUE_ERROR_THRESHOLDS,
    CANCEL_BOUNDARY_THRESHOLDS, CANCEL_ZERO_THRESHOLDS,
)

# Upper bounds observed in the pinned cann-bench task set. Operator-specific
# overrides may loosen the defaults, but values beyond these bounds make the
# precision gate ineffective and must be reviewed before this table changes.
MAX_PRECISION_THRESHOLD_OVERRIDES = {
    "float16": 0.1,
    "bfloat16": 0.05,
    "float32": 0.05,
    "float64": 0.05,
    "hifloat32": 0.05,
    "float8_e4m3fn": 2.0**-3,
    "float8_e5m2": 2.0**-2,
    "int8": 1.0,
    "uint8": 1.0,
    "int16": 0.0,
    "uint16": 0.0,
    "int32": 0.0,
    "uint32": 0.0,
    "int64": 0.0,
    "uint64": 0.0,
    "bool": 0.0,
}


def normalize_dtype(dtype: Any) -> str:
    value = str(dtype).split(".")[-1].lower().replace("-", "_")
    aliases = {
        "half": "float16",
        "float": "float32",
        "double": "float64",
        "float8_e4m3": "float8_e4m3fn",
    }
    return aliases.get(value, value)


def get_precision_threshold(dtype: Any) -> float | None:
    return PRECISION_THRESHOLDS.get(normalize_dtype(dtype))


def get_small_value_threshold(dtype: Any) -> float | None:
    return SMALL_VALUE_THRESHOLDS.get(normalize_dtype(dtype))


def get_small_value_error(dtype: Any) -> float | None:
    return SMALL_VALUE_ERROR_THRESHOLDS.get(normalize_dtype(dtype))


def get_cancel_boundary(dtype: Any) -> float | None:
    return CANCEL_BOUNDARY_THRESHOLDS.get(normalize_dtype(dtype))


def get_cancel_zero_threshold(dtype: Any) -> float | None:
    return CANCEL_ZERO_THRESHOLDS.get(normalize_dtype(dtype))
