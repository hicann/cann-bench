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

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .thresholds import MAX_PRECISION_THRESHOLD_OVERRIDES, normalize_dtype


ORACLE_INPUT_MODES = frozenset({"fp64", "native"})
OUTPUT_PRECISION_STANDARDS = frozenset(
    {
        "auto",
        "commercial_float",
        "exact",
        "packed_int4_ae1",
        "quantized_integer_ae1",
    }
)


@dataclass(frozen=True)
class PrecisionPolicy:
    thresholds: dict[str, float] = field(default_factory=dict)
    ignore_output_indices: frozenset[int] = frozenset()
    optional_output_indices: frozenset[int] = frozenset()
    output_dtypes: tuple[frozenset[str], ...] = ()
    output_precision_standards: tuple[str, ...] = ()
    oracle_input_mode: str = "fp64"


def load_precision_policy(path: Path) -> PrecisionPolicy:
    if not path.is_file():
        return PrecisionPolicy()
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"invalid precision policy: {path}")
    operator = payload.get("operator", {})
    if not isinstance(operator, dict):
        raise ValueError(f"invalid operator definition in precision policy: {path}")

    raw_precision = operator.get("precision", {}) or {}
    if not isinstance(raw_precision, dict):
        raise ValueError(f"operator precision must be a mapping: {path}")
    oracle_input_mode = str(raw_precision.get("oracle_input_mode", "fp64"))
    if oracle_input_mode not in ORACLE_INPUT_MODES:
        raise ValueError(f"unsupported oracle_input_mode: {oracle_input_mode}")
    if oracle_input_mode != "fp64":
        standard = raw_precision.get("standard")
        evidence = raw_precision.get("evidence")
        if standard != "cann_bench" or not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(
                f"non-default oracle_input_mode requires standard=cann_bench and evidence: {path}"
            )

    thresholds: dict[str, float] = {}
    raw_thresholds = operator.get("precision_thresholds", {}) or {}
    if not isinstance(raw_thresholds, dict):
        raise ValueError(f"precision_thresholds must be a mapping: {path}")
    for dtype, raw_value in raw_thresholds.items():
        normalized_dtype = normalize_dtype(dtype)
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"precision threshold must be finite and non-negative: {dtype}={raw_value}")
        maximum = MAX_PRECISION_THRESHOLD_OVERRIDES.get(normalized_dtype)
        if maximum is None:
            raise ValueError(f"unsupported precision threshold dtype: {dtype}")
        if value > maximum:
            raise ValueError(
                f"precision threshold exceeds reviewed maximum: {dtype}={raw_value} > {maximum}"
            )
        thresholds[normalized_dtype] = value

    ignored: set[int] = set()
    optional: set[int] = set()
    outputs = operator.get("outputs", []) or []
    if not isinstance(outputs, list):
        raise ValueError(f"operator outputs must be a list: {path}")
    output_dtypes: list[frozenset[str]] = []
    output_precision_standards: list[str] = []
    for index, output in enumerate(outputs):
        if not isinstance(output, dict):
            raise ValueError(f"operator output must be a mapping: {path} output[{index}]")
        if output.get("compare", True) is False:
            ignored.add(index)
        if output.get("optional", False) is True:
            optional.add(index)
        raw_dtypes = output.get("dtype", []) or []
        if isinstance(raw_dtypes, str):
            raw_dtypes = [raw_dtypes]
        if not isinstance(raw_dtypes, list):
            raise ValueError(f"operator output dtype must be a list: {path} output[{index}]")
        output_dtypes.append(frozenset(normalize_dtype(dtype) for dtype in raw_dtypes))
        precision_standard = str(output.get("precision_standard", "auto"))
        if precision_standard not in OUTPUT_PRECISION_STANDARDS:
            raise ValueError(
                f"unsupported output precision_standard: {precision_standard} in {path} output[{index}]"
            )
        output_precision_standards.append(precision_standard)

    return PrecisionPolicy(
        thresholds=thresholds,
        ignore_output_indices=frozenset(ignored),
        optional_output_indices=frozenset(optional),
        output_dtypes=tuple(output_dtypes),
        output_precision_standards=tuple(output_precision_standards),
        oracle_input_mode=oracle_input_mode,
    )
