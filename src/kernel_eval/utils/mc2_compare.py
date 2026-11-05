# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MC2 three-way metrics, following mc2_test/script/new_compare.py.

The candidate and the device cascade are compared to the SAME CPU FP32
reference. These are absolute-error metrics, not MERE/MARE. A separate finite
element tolerance handles exact cascade references without altering the metrics.
NumPy keeps this numeric core testable off NPU.
"""

import numpy as np


MULTIPLIERS = {"diff_big_max": 10, "diff_big_avg": 2, "diff_big_sum": 2, "diff_rmse": 2}
SMALL_COUNT_GATE = 100000
FINITE_TOLERANCES = {
    "float16": (1e-5, 2**-10),
    "bfloat16": (1e-5, 2**-7),
    "float32": (1e-9, 2**-23),
}


def error_metrics(value, golden, dtype):
    """Return FP32 error statistics without modifying either input."""
    value = np.array(value, dtype=np.float32, copy=True)
    golden = np.array(golden, dtype=np.float32, copy=True)
    if value.shape != golden.shape:
        raise ValueError(f"shape mismatch: {value.shape} != {golden.shape}")
    if dtype in ("float16", "bfloat16"):
        small, atol = 1e-3, 1e-5
        limit = 65504.0 if dtype == "float16" else float.fromhex("0x1.fep127")
        # Round-to-nearest-even overflows at max finite + half its final ULP,
        # not immediately above max finite. Preserve FP32 G below that point.
        overflow = 65520.0 if dtype == "float16" else float.fromhex("0x1.ffp127")
        golden[golden >= overflow] = np.inf
        golden[golden <= -overflow] = -np.inf
    elif dtype == "float32":
        small, atol, limit = 1e-6, 1e-9, np.finfo(np.float32).max
    else:
        raise ValueError(f"Unsupported MC2 comparison dtype: {dtype}")
    nonfinite = {}
    for name, mask in (("nan", np.isnan), ("inf", np.isposinf), ("ninf", np.isneginf)):
        nonfinite[f"num_total_{name}"] = int(mask(golden).sum())
        nonfinite[f"err_total_{name}"] = int(
            np.logical_xor(mask(value), mask(golden)).sum()
        )
    np.nan_to_num(value, copy=False, nan=1, posinf=limit, neginf=-limit)
    np.nan_to_num(golden, copy=False, nan=1, posinf=limit, neginf=-limit)
    big = np.abs(golden) >= small
    small_count = int((np.abs(golden) < small).sum())
    big_count = int(big.sum())
    with np.errstate(over="ignore", invalid="ignore"):
        delta = value - golden
        diff = np.abs(delta)
        big_diff = np.where(big, diff, np.float32(0))
        total = float(big_diff.sum(dtype=np.float32))
        metrics = {
            "total_big_num": big_count,
            "total_small_num": small_count,
            "total_count": int(value.size),
            "diff_big_max": float(big_diff.max()) if value.size else 0.0,
            "diff_big_sum": total,
            "diff_big_avg": float(np.float32(total) / np.float32(big_count))
            if big_count
            else 0.0,
            # Keep the reference's <= boundary for errors, but < for count.
            "err_small_num": int(((np.abs(golden) <= small) & (diff > atol)).sum()),
            "diff_rmse": float(np.sqrt(np.mean(np.square(diff), dtype=np.float32)))
            if value.size
            else 0.0,
            "rst_eb": abs(int((delta > 0).sum()) - int((delta < 0).sum())),
            "diff_eb": float(delta.sum(dtype=np.float32)),
            "small_value": small,
            "small_value_atol": atol,
            **nonfinite,
        }
    return metrics


def compare_three_way(candidate, cascade, golden, dtype):
    """Return serializable per-output evidence and the actual acceptance limits."""
    actual = error_metrics(candidate, golden, dtype)
    baseline = error_metrics(cascade, golden, dtype)
    multipliers = dict(MULTIPLIERS)
    if actual["total_small_num"] >= SMALL_COUNT_GATE:
        multipliers["err_small_num"] = 2
    limits = {name: baseline[name] * factor for name, factor in multipliers.items()}
    ratio_failed = [name for name, limit in limits.items() if actual[name] > limit]
    failed = [
        name
        for name, limit in limits.items()
        if not np.isfinite(actual[name]) or not np.isfinite(limit)
    ]
    # Counts and exceptional values are common gates, never rescued by tolerance.
    if "err_small_num" in ratio_failed:
        failed.append("err_small_num")
    # The upstream diagnostic records these mismatches but can silently pass
    # them. Fail closed here: NaN/Inf must agree with the CPU reference.
    for side, metrics in (("candidate", actual), ("cascade", baseline)):
        for name in ("err_total_nan", "err_total_inf", "err_total_ninf"):
            if metrics[name]:
                failed.append(f"{side}.{name}")
    atol, rtol = FINITE_TOLERANCES[dtype]
    a, g = np.asarray(candidate, dtype=np.float64), np.asarray(golden, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(g)
    tolerance_passed = bool(
        np.all(np.abs(a[finite] - g[finite]) <= atol + rtol * np.abs(g[finite]))
    )
    accepted_by = None
    if not failed:
        if not ratio_failed:
            accepted_by = "cascade_ratio"
        elif tolerance_passed:
            accepted_by = "finite_tolerance"
        else:
            failed.extend(ratio_failed)

    # JSON must remain standards compliant even when arithmetic overflows.
    def finite_dict(data):
        return {
            k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in data.items()
        }

    return {
        "passed": not failed,
        "accepted_by": accepted_by,
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "candidate_metrics": finite_dict(actual),
        "cascade_metrics": finite_dict(baseline),
        "limits": finite_dict(limits),
        "failed_metrics": failed,
        "comparison_dtype": dtype,
        "small_count_gate": SMALL_COUNT_GATE,
    }
