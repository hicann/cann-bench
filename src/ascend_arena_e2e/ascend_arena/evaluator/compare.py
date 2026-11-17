# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Tensor comparison using the cann-bench ecosystem precision standard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .thresholds import (
    get_cancel_boundary,
    get_cancel_zero_threshold,
    get_precision_threshold,
    get_small_value_error,
    get_small_value_threshold,
    normalize_dtype,
)


@dataclass
class OutputCompareResult:
    index: int
    passed: bool
    message: str
    checker: str
    expected_dtype: str
    actual_dtype: str
    shape: tuple[int, ...]
    threshold: float = 0.0
    max_abs_diff: float = 0.0
    mean_abs_diff: float = 0.0
    mare: float = 0.0
    mere: float = 0.0
    rmse: float = 0.0
    small_value_count: int = 0
    small_value_error_count: int = 0
    small_value_native_error_count: int = 0
    small_value_ratio: float = 0.0
    cancel_count: int = 0
    cancel_error_count: int = 0
    cancel_native_error_count: int = 0
    cancel_ratio: float = 0.0


@dataclass
class TensorCompareResult:
    passed: bool
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    mean_rel_diff: float
    message: str
    # Retained for payload compatibility. These now contain raw MARE/MERE/RMSE,
    # not commercial double-benchmark ratios.
    mare_ratio: float = 0.0
    mere_ratio: float = 0.0
    rmse_ratio: float = 0.0
    small_value_count: int = 0
    small_value_error_count: int = 0
    small_value_native_error_count: int = 0
    cancel_count: int = 0
    cancel_error_count: int = 0
    cancel_native_error_count: int = 0
    output_results: list[OutputCompareResult] = field(default_factory=list)


def _output_structure(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return "tensor"
    if output is None:
        return "none"
    if isinstance(output, list):
        return ("list", tuple(_output_structure(item) for item in output))
    if isinstance(output, tuple):
        return ("tuple", tuple(_output_structure(item) for item in output))
    return ("unsupported", type(output).__name__)


def _flatten_outputs(output: Any) -> list[torch.Tensor | None]:
    if isinstance(output, torch.Tensor) or output is None:
        return [output]
    if isinstance(output, (list, tuple)):
        leaves: list[torch.Tensor | None] = []
        for item in output:
            leaves.extend(_flatten_outputs(item))
        return leaves
    raise TypeError(f"unsupported output type: {type(output)!r}")


def _prepare_comparison_tensors(
    exp_tensor: torch.Tensor,
    act_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return exp_tensor.detach().to("cpu"), act_tensor.detach().to("cpu")


def compare_tensors(
    expected: Any,
    actual: Any,
    *,
    max_abs_error: float,
    max_rel_error: float,
    native: Any | None = None,
    integer_atol: float = 0.0,
    bitwise: bool = False,
    ignore_output_indices: set[int] | None = None,
    optional_output_indices: set[int] | None = None,
    precision_thresholds: dict[str, float] | None = None,
    allowed_output_dtypes: tuple[frozenset[str], ...] | None = None,
    output_precision_standards: tuple[str, ...] | None = None,
) -> TensorCompareResult:
    del max_abs_error, max_rel_error  # Legacy CLI tolerances do not weaken ecosystem checks.
    native = _ideal_native(expected, actual) if native is None else native
    ignored = ignore_output_indices or set()

    expected_structure = _output_structure(expected)
    native_structure = _output_structure(native)
    actual_structure = _output_structure(actual)
    if expected_structure != native_structure:
        return _structural_failure("golden_native_structure_mismatch")
    if actual_structure != native_structure:
        return _structural_failure("output_structure_mismatch")

    try:
        expected_tensors = _flatten_outputs(expected)
        native_tensors = _flatten_outputs(native)
        actual_tensors = _flatten_outputs(actual)
    except TypeError:
        return _structural_failure("unsupported_output_type")
    if len(expected_tensors) != len(actual_tensors) or len(native_tensors) != len(actual_tensors):
        return _structural_failure("output_count_mismatch")
    contract_indices = _output_contract_indices(
        expected,
        len(allowed_output_dtypes or ()),
        optional_output_indices or set(),
    )
    if len(contract_indices) != len(actual_tensors):
        return _structural_failure("output_contract_mismatch")

    output_results: list[OutputCompareResult] = []
    for index, (contract_index, (golden_tensor, native_tensor, actual_tensor)) in enumerate(
        zip(contract_indices, zip(expected_tensors, native_tensors, actual_tensors))
    ):
        if contract_index in ignored:
            continue
        if golden_tensor is None or native_tensor is None or actual_tensor is None:
            both_none = golden_tensor is None and native_tensor is None and actual_tensor is None
            output_results.append(
                OutputCompareResult(
                    index=index,
                    passed=both_none,
                    message="ok" if both_none else "optional_output_mismatch",
                    checker="none",
                    expected_dtype="none",
                    actual_dtype="none" if actual_tensor is None else str(actual_tensor.dtype),
                    shape=(),
                )
            )
            continue
        output_results.append(
            _compare_single_output(
                index,
                golden_tensor,
                native_tensor,
                actual_tensor,
                integer_atol=integer_atol,
                bitwise=bitwise,
                precision_thresholds=precision_thresholds,
                precision_standard=(output_precision_standards or ())[contract_index]
                if output_precision_standards and contract_index < len(output_precision_standards)
                else "auto",
                allowed_dtypes=(allowed_output_dtypes or ())[contract_index]
                if allowed_output_dtypes and contract_index < len(allowed_output_dtypes)
                else frozenset(),
            )
        )

    return _aggregate_results(output_results)


def _compare_single_output(
    index: int,
    golden: torch.Tensor,
    native: torch.Tensor,
    actual: torch.Tensor,
    *,
    integer_atol: float,
    bitwise: bool,
    precision_thresholds: dict[str, float] | None,
    precision_standard: str,
    allowed_dtypes: frozenset[str],
) -> OutputCompareResult:
    golden_cpu, actual_cpu = _prepare_comparison_tensors(golden, actual)
    native_cpu = native.detach().to("cpu")
    expected_dtype = str(native_cpu.dtype)
    actual_dtype = str(actual_cpu.dtype)
    shape = tuple(native_cpu.shape)

    if precision_standard == "packed_int4_ae1" and any(
        tensor.dtype != torch.int32 for tensor in (golden_cpu, native_cpu, actual_cpu)
    ):
        return _output_failure(
            index,
            "packed_int4_storage_dtype_mismatch",
            expected_dtype,
            actual_dtype,
            shape,
        )
    if tuple(golden_cpu.shape) != shape:
        return _output_failure(index, "golden_native_shape_mismatch", expected_dtype, actual_dtype, shape)
    if tuple(actual_cpu.shape) != shape:
        return _output_failure(index, "shape_mismatch", expected_dtype, actual_dtype, shape)
    if actual_cpu.dtype != native_cpu.dtype:
        return _output_failure(index, "dtype_mismatch", expected_dtype, actual_dtype, shape)
    if allowed_dtypes and normalize_dtype(actual_cpu.dtype) not in allowed_dtypes:
        return _output_failure(index, "dtype_not_allowed_by_policy", expected_dtype, actual_dtype, shape)
    if not native_cpu.is_floating_point() and golden_cpu.dtype != native_cpu.dtype:
        return _output_failure(index, "golden_native_dtype_mismatch", expected_dtype, actual_dtype, shape)

    threshold = _output_threshold(native_cpu.dtype, precision_thresholds)
    if precision_standard == "exact":
        return _compare_bitwise(index, golden_cpu.to(native_cpu.dtype), actual_cpu)
    if precision_standard == "packed_int4_ae1":
        if native_cpu.dim() == 0:
            return _output_failure(
                index,
                "packed_int4_storage_shape_mismatch",
                expected_dtype,
                actual_dtype,
                shape,
            )
        golden_int4 = _unpack_packed_int4(golden_cpu)
        native_int4 = _unpack_packed_int4(native_cpu)
        actual_int4 = _unpack_packed_int4(actual_cpu)
        return _compare_integer(index, golden_int4, native_int4, actual_int4, 1.0)
    if precision_standard == "commercial_float" and not native_cpu.is_floating_point():
        return _output_failure(index, "precision_standard_dtype_mismatch", expected_dtype, actual_dtype, shape)
    if precision_standard == "quantized_integer_ae1":
        if native_cpu.dtype in {torch.int8, torch.uint8}:
            return _compare_integer(index, golden_cpu, native_cpu, actual_cpu, 1.0)
        if not native_cpu.is_floating_point():
            return _output_failure(index, "precision_standard_dtype_mismatch", expected_dtype, actual_dtype, shape)
    if bitwise:
        return _compare_bitwise(index, golden_cpu.to(native_cpu.dtype), actual_cpu)
    if native_cpu.is_floating_point() and threshold == 0.0:
        return _compare_bitwise(index, golden_cpu.to(native_cpu.dtype), actual_cpu)
    if not native_cpu.is_floating_point():
        legacy_quant_dtype = native_cpu.dtype in {torch.int8, torch.uint8}
        atol = threshold if threshold is not None else (integer_atol if legacy_quant_dtype else 0.0)
        return _compare_integer(index, golden_cpu, native_cpu, actual_cpu, atol)
    if threshold is None:
        return _output_failure(index, "unsupported_float_dtype", expected_dtype, actual_dtype, shape)
    return _compare_float(index, golden_cpu, native_cpu, actual_cpu, threshold)


def _output_threshold(dtype: Any, overrides: dict[str, float] | None) -> float | None:
    key = normalize_dtype(dtype)
    if overrides is not None and key in overrides:
        return float(overrides[key])
    return get_precision_threshold(dtype)


def _ideal_native(expected: Any, actual: Any) -> Any:
    if expected is None and actual is None:
        return None
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return expected.detach().to("cpu").to(actual.dtype)
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return [_ideal_native(exp_item, act_item) for exp_item, act_item in zip(expected, actual)]
    if isinstance(expected, tuple) and isinstance(actual, tuple) and len(expected) == len(actual):
        return tuple(_ideal_native(exp_item, act_item) for exp_item, act_item in zip(expected, actual))
    return expected


def _output_contract_indices(
    output: Any,
    contract_count: int,
    optional_output_indices: set[int],
) -> list[int]:
    if isinstance(output, torch.Tensor) or output is None:
        return [0]
    if not isinstance(output, (list, tuple)):
        return []
    if contract_count == 1:
        return [0] * len(_flatten_outputs(output))
    indices: list[int] = []
    for top_index, item in enumerate(output):
        indices.extend([top_index] * len(_flatten_outputs(item)))
    if contract_count and len(output) != contract_count:
        missing_indices = set(range(len(output), contract_count))
        if len(output) > contract_count or not missing_indices.issubset(optional_output_indices):
            return []
    return indices


def _compare_bitwise(index: int, expected: torch.Tensor, actual: torch.Tensor) -> OutputCompareResult:
    expected_c = expected.contiguous()
    actual_c = actual.contiguous()
    if expected_c.is_floating_point():
        view_dtype = {
            torch.float16: torch.int16,
            torch.bfloat16: torch.int16,
            torch.float32: torch.int32,
            torch.float64: torch.int64,
        }.get(expected_c.dtype)
        if view_dtype is None:
            return _output_failure(index, "unsupported_bitwise_dtype", str(expected.dtype), str(actual.dtype), tuple(expected.shape))
        equal = torch.equal(expected_c.view(view_dtype), actual_c.view(view_dtype))
    else:
        equal = torch.equal(expected_c, actual_c)
    return OutputCompareResult(
        index=index,
        passed=equal,
        message="ok" if equal else "bitwise_mismatch",
        checker="bitwise",
        expected_dtype=str(expected.dtype),
        actual_dtype=str(actual.dtype),
        shape=tuple(expected.shape),
    )


def _unpack_packed_int4(tensor: torch.Tensor) -> torch.Tensor:
    work = tensor.to(torch.int64)
    shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=work.device)
    lanes = (work.unsqueeze(-1) >> shifts) & 0xF
    signed_lanes = torch.where(lanes >= 8, lanes - 16, lanes).to(torch.int8)
    return signed_lanes.reshape(*work.shape[:-1], work.shape[-1] * 8)


def _compare_integer(
    index: int,
    golden: torch.Tensor,
    native: torch.Tensor,
    actual: torch.Tensor,
    atol: float,
) -> OutputCompareResult:
    if golden.numel() == 0:
        max_diff = mean_diff = 0.0
        passed = True
    elif atol > 0:
        # Legacy quantized tolerances are restricted to int8/uint8 by the
        # caller, so int64 promotion computes the exact difference.
        diff = torch.abs(golden.to(torch.int64) - actual.to(torch.int64))
        max_diff = float(diff.max().item())
        mean_diff = float(diff.double().mean().item())
        passed = max_diff <= atol
    else:
        passed = torch.equal(golden, actual)
        # FP64 is diagnostic-only here; pass/fail remains exact even for int64.
        diff = torch.abs(golden.double() - actual.double())
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
    return OutputCompareResult(
        index=index,
        passed=passed,
        message="ok" if passed else "integer_mismatch",
        checker="absolute_error" if atol > 0 else "exact",
        expected_dtype=str(native.dtype),
        actual_dtype=str(actual.dtype),
        shape=tuple(native.shape),
        threshold=float(atol),
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
    )


def _compare_float(
    index: int,
    golden: torch.Tensor,
    native: torch.Tensor,
    actual: torch.Tensor,
    threshold: float,
) -> OutputCompareResult:
    small_threshold = get_small_value_threshold(native.dtype)
    small_error = get_small_value_error(native.dtype)
    cancel_boundary = get_cancel_boundary(native.dtype)
    cancel_zero = get_cancel_zero_threshold(native.dtype)
    if None in (small_threshold, small_error, cancel_boundary, cancel_zero):
        return _output_failure(index, "unsupported_float_dtype", str(native.dtype), str(actual.dtype), tuple(native.shape))

    # cann-bench compares against the ideal value representable by the output dtype.
    golden_truncated = golden.to(native.dtype).double()
    native_fp64 = native.double()
    actual_fp64 = actual.double()

    nan_expected = torch.isnan(golden_truncated)
    nan_actual = torch.isnan(actual_fp64)
    if not torch.equal(nan_expected, nan_actual):
        return _output_failure(index, "nan_position_mismatch", str(native.dtype), str(actual.dtype), tuple(native.shape))

    inf_expected = torch.isinf(golden_truncated)
    inf_actual = torch.isinf(actual_fp64)
    both_inf = inf_expected & inf_actual
    if bool(both_inf.any().item()) and not torch.equal(
        torch.sign(golden_truncated[both_inf]), torch.sign(actual_fp64[both_inf])
    ):
        return _output_failure(index, "inf_sign_mismatch", str(native.dtype), str(actual.dtype), tuple(native.shape))

    one_sided_inf = inf_expected ^ inf_actual
    if bool(one_sided_inf.any().item()):
        max_finite = float(torch.finfo(native.dtype).max)
        golden_truncated = golden_truncated.clone()
        actual_fp64 = actual_fp64.clone()
        golden_only = inf_expected & ~inf_actual
        actual_only = inf_actual & ~inf_expected
        golden_truncated[golden_only] = torch.sign(golden_truncated[golden_only]) * max_finite
        actual_fp64[actual_only] = torch.sign(actual_fp64[actual_only]) * max_finite

    valid_mask = ~(nan_expected | both_inf)
    diff = torch.abs(actual_fp64 - golden_truncated)
    native_diff = torch.abs(native_fp64 - golden_truncated)
    relative = diff / (torch.abs(golden_truncated) + 1e-7)
    native_relative = native_diff / (torch.abs(golden_truncated) + 1e-7)
    finite_relative = relative[valid_mask]

    max_abs = float(diff[valid_mask].max().item()) if bool(valid_mask.any().item()) else 0.0
    mean_abs = float(diff[valid_mask].mean().item()) if bool(valid_mask.any().item()) else 0.0
    overall_mare = float(finite_relative.max().item()) if finite_relative.numel() else 0.0
    overall_mere = float(finite_relative.mean().item()) if finite_relative.numel() else 0.0
    overall_rmse = float(torch.sqrt((diff[valid_mask] ** 2).mean()).item()) if bool(valid_mask.any().item()) else 0.0

    small_mask = valid_mask & (torch.abs(golden_truncated) < float(small_threshold))
    small_npu_errors = int((small_mask & (diff > float(small_error))).sum().item())
    small_native_errors = int((small_mask & (native_diff > float(small_error))).sum().item())
    small_ratio = small_npu_errors / max(small_native_errors, 1)

    cancel_mask = (
        valid_mask
        & (torch.abs(actual_fp64) < float(cancel_zero))
        & (torch.abs(golden_truncated) < float(cancel_boundary))
        & (torch.abs(golden_truncated) >= float(small_threshold))
    )
    cancel_npu_errors = int((cancel_mask & (relative > 10.0 * float(threshold))).sum().item())
    cancel_native_errors = int((cancel_mask & (native_relative > 10.0 * float(threshold))).sum().item())
    cancel_ratio = cancel_npu_errors / max(cancel_native_errors, 1)

    overall_pass = overall_mere < float(threshold) and overall_mare < 10.0 * float(threshold)
    normal_mask = valid_mask & ~small_mask & ~cancel_mask
    normal_relative = relative[normal_mask]
    normal_mare = float(normal_relative.max().item()) if normal_relative.numel() else 0.0
    normal_mere = float(normal_relative.mean().item()) if normal_relative.numel() else 0.0
    normal_error_count = int((normal_mask & (relative > 10.0 * float(threshold))).sum().item())
    normal_pass = normal_error_count == 0
    small_pass = small_ratio <= 2.0
    cancel_pass = cancel_ratio <= 2.0
    passed = overall_pass or (normal_pass and small_pass and cancel_pass)

    if overall_pass:
        reported_mare = overall_mare
        reported_mere = overall_mere
    elif passed:
        reported_mare = normal_mare
        reported_mere = normal_mere
    elif not normal_pass and normal_relative.numel():
        reported_mare = normal_mare
        reported_mere = normal_mere
    else:
        reported_mare = overall_mare
        reported_mere = overall_mere

    if passed:
        message = "ok"
    elif not normal_pass:
        message = "relative_error_mismatch"
    elif not small_pass:
        message = "small_value_mismatch"
    else:
        message = "cancellation_mismatch"

    return OutputCompareResult(
        index=index,
        passed=passed,
        message=message,
        checker="relative_error",
        expected_dtype=str(native.dtype),
        actual_dtype=str(actual.dtype),
        shape=tuple(native.shape),
        threshold=float(threshold),
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        mare=reported_mare,
        mere=reported_mere,
        rmse=overall_rmse,
        small_value_count=int(small_mask.sum().item()),
        small_value_error_count=small_npu_errors,
        small_value_native_error_count=small_native_errors,
        small_value_ratio=small_ratio,
        cancel_count=int(cancel_mask.sum().item()),
        cancel_error_count=cancel_npu_errors,
        cancel_native_error_count=cancel_native_errors,
        cancel_ratio=cancel_ratio,
    )


def _aggregate_results(outputs: list[OutputCompareResult]) -> TensorCompareResult:
    passed = all(output.passed for output in outputs)
    first_failure = next((output for output in outputs if not output.passed), None)
    total_weight = sum(_output_weight(output) for output in outputs)
    mean_abs = (
        sum(output.mean_abs_diff * _output_weight(output) for output in outputs) / total_weight
        if total_weight
        else 0.0
    )
    mean_rel = (
        sum(output.mere * _output_weight(output) for output in outputs) / total_weight
        if total_weight
        else 0.0
    )
    return TensorCompareResult(
        passed=passed,
        max_abs_diff=max((output.max_abs_diff for output in outputs), default=0.0),
        max_rel_diff=max((output.mare for output in outputs), default=0.0),
        mean_abs_diff=mean_abs,
        mean_rel_diff=mean_rel,
        message="ok" if passed else f"output[{first_failure.index}]:{first_failure.message}",
        mare_ratio=max((output.mare for output in outputs), default=0.0),
        mere_ratio=mean_rel,
        rmse_ratio=max((output.rmse for output in outputs), default=0.0),
        small_value_count=sum(output.small_value_count for output in outputs),
        small_value_error_count=sum(output.small_value_error_count for output in outputs),
        small_value_native_error_count=sum(output.small_value_native_error_count for output in outputs),
        cancel_count=sum(output.cancel_count for output in outputs),
        cancel_error_count=sum(output.cancel_error_count for output in outputs),
        cancel_native_error_count=sum(output.cancel_native_error_count for output in outputs),
        output_results=outputs,
    )


def _shape_numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _output_weight(output: OutputCompareResult) -> int:
    if output.checker in {"none", "structural"}:
        return 0
    return _shape_numel(output.shape)


def _output_failure(
    index: int,
    message: str,
    expected_dtype: str,
    actual_dtype: str,
    shape: tuple[int, ...],
) -> OutputCompareResult:
    return OutputCompareResult(
        index=index,
        passed=False,
        message=message,
        checker="structural",
        expected_dtype=expected_dtype,
        actual_dtype=actual_dtype,
        shape=shape,
    )


def _structural_failure(message: str) -> TensorCompareResult:
    return TensorCompareResult(False, 0.0, 0.0, 0.0, 0.0, message)
