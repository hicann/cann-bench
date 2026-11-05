# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Output contract and report adapter for the opt-in MC2 three-way check."""

from dataclasses import dataclass, field
from typing import Any, Dict

from ..base.result import (
    FAILURE_TYPE_REFERENCE_INVALID,
    OutputResult,
    register_output_result,
)
from ..utils.mc2_compare import compare_three_way


@dataclass
class MC2ThreeWayOutputResult(OutputResult):
    index: int
    name: str = ""
    dtype: str = ""
    passed: bool = False
    error_msg: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "index": self.index,
            "name": self.name,
            "dtype": self.dtype,
            "passed": self.passed,
            "error_msg": self.error_msg,
            **self.metadata,
        }

    @classmethod
    def from_dict(cls, data):
        fields = {
            k: data[k]
            for k in ("index", "name", "dtype", "passed", "error_msg")
            if k in data
        }
        return cls(
            **fields, metadata={k: v for k, v in data.items() if k not in fields}
        )

    def format_summary(self):
        if self.metadata.get("failure_type") == FAILURE_TYPE_REFERENCE_INVALID:
            return f"{self.name}: MC2 three-way REFERENCE INVALID; candidate unscored; {self.error_msg}"
        status = "PASS" if self.passed else "FAIL"
        candidate = self.metadata.get("candidate_metrics", {})
        cascade = self.metadata.get("cascade_metrics", {})
        return (
            f"{self.name}: {status} MC2 three-way; "
            f"RMSE candidate={candidate.get('diff_rmse')}, cascade={cascade.get('diff_rmse')}"
            + (f"; {self.error_msg}" if self.error_msg else "")
        )


register_output_result("mc2_three_way", MC2ThreeWayOutputResult)


class MC2ReferenceInvalidError(RuntimeError):
    """The benchmark reference failed, so candidate accuracy cannot be scored."""


def _reference_health(native, cpu, staged_cpu=None):
    """Check B against an independent CPU reference, never against the candidate.

    With a task-provided stage-rounded CPU cascade, apply the same MC2 error
    ratio to B versus that independent cascade and the mathematical FP32 G.
    The generic pointwise checker is intended for a single operation and
    falsely rejects small/cancelling entries in million-element MC2 GEMMs.
    Keep its fixed policy for tasks without a stage-rounded reference.
    """
    from ..utils.compare import compare_tensors
    from ..utils.mc2_compare import FINITE_TOLERANCES
    from ..utils.thresholds import PRECISION_THRESHOLDS

    import numpy as np

    dtype = str(native.dtype).removeprefix("torch.")
    output = native.float() if dtype.startswith("float8") else native
    if staged_cpu is not None:
        effective_dtype = "float32" if dtype.startswith("float8") else dtype
        golden = cpu.numpy()
        evidence = compare_three_way(
            output.float().numpy(), staged_cpu.numpy(), golden, effective_dtype
        )

        # The independent stage may contain ordinary low-precision rounding,
        # but a biased/wrong CPU stage must not inflate the NPU allowance.
        # Use aggregate fixed dtype limits here; pointwise MARE is unsuitable
        # for cancellation in a multi-stage reduction.
        flat = np.asarray(golden, dtype=np.float32).ravel()
        stage_flat = np.asarray(staged_cpu.numpy(), dtype=np.float32).ravel()
        native_flat = np.asarray(output.float().numpy(), dtype=np.float32).ravel()
        exact_fp32_stage = dtype == "float32" and np.array_equal(stage_flat, flat)
        atol, rtol = FINITE_TOLERANCES[effective_dtype]
        native_within_fixed_tolerance = True
        square_sum = 0.0
        stage_square_sum = 0.0
        npu_stage_square_sum = 0.0
        npu_stage_signed_sum = 0.0
        npu_stage_max = 0.0
        finite_count = 0
        for start in range(0, flat.size, 1_000_000):
            block = flat[start : start + 1_000_000].astype(np.float64)
            stage_block = stage_flat[start : start + 1_000_000].astype(np.float64)
            native_block = native_flat[start : start + 1_000_000].astype(np.float64)
            # compare_three_way above enforces exact signed Inf/NaN masks
            # against finite G's output-dtype range, independently for B and
            # the CPU stage. Matching overflow is not an arithmetic error.
            # Exclude it from BOTH the RMSE numerator and normalization scale:
            # Inf - Inf is NaN, and large overflow G must not dilute errors in
            # the remaining finite outputs. Bad masks still fail evidence.
            finite = np.isfinite(stage_block) & np.isfinite(native_block)
            block = block[finite]
            stage_block, native_block = stage_block[finite], native_block[finite]
            finite_count += block.size
            stage_delta = stage_block - block
            npu_stage_delta = native_block - stage_block
            if exact_fp32_stage and native_within_fixed_tolerance:
                native_within_fixed_tolerance = bool(
                    np.all(np.abs(npu_stage_delta) <= atol + rtol * np.abs(block))
                )
            square_sum += float(np.dot(block, block))
            stage_square_sum += float(np.dot(stage_delta, stage_delta))
            npu_stage_square_sum += float(np.dot(npu_stage_delta, npu_stage_delta))
            npu_stage_signed_sum += float(npu_stage_delta.sum(dtype=np.float64))
            if npu_stage_delta.size:
                npu_stage_max = max(
                    npu_stage_max, float(np.max(np.abs(npu_stage_delta)))
                )
        golden_rms = float(np.sqrt(square_sum / finite_count)) if finite_count else 0.0
        # Keep the existing all-finite path's numeric statistics unchanged.
        if finite_count == flat.size:
            stage_rmse = evidence["cascade_metrics"]["diff_rmse"]
        else:
            stage_rmse = (
                float(np.sqrt(stage_square_sum / finite_count)) if finite_count else 0.0
            )
        # For FP32 output, the staged CPU reference can be bit-identical to G.
        # Then B-versus-G/B-versus-stage ratio limits are all zero, so any
        # ordinary FP32 GEMM accumulation difference fails reference health.
        # The published FP32 precision threshold is an independent bound for
        # this exact-stage case; candidate comparison still uses the original
        # three-way metrics and the measured NPU cascade.
        health_rtol = PRECISION_THRESHOLDS["float32"] if exact_fp32_stage else rtol
        # Preserve the repository's absolute floor for near-zero tensors.
        # Dividing by atol itself would make the effective allowance
        # rtol * atol, much stricter than the published atol.
        scale = max(golden_rms, atol / health_rtol)
        stage_rms_ratio = stage_rmse / scale if stage_rmse is not None else None
        npu_stage_rmse = (
            float(np.sqrt(npu_stage_square_sum / finite_count)) if finite_count else 0.0
        )
        npu_stage_rms_ratio = npu_stage_rmse / scale
        npu_stage_mean = npu_stage_signed_sum / finite_count if finite_count else 0.0
        npu_stage_bias_fraction = (
            abs(npu_stage_mean) / npu_stage_rmse if npu_stage_rmse else 0.0
        )
        stage_rms_limit = health_rtol
        stage_valid = bool(
            stage_rms_ratio is not None
            and np.isfinite(stage_rms_ratio)
            and stage_rms_ratio <= stage_rms_limit
            and np.isfinite(npu_stage_rms_ratio)
            and npu_stage_rms_ratio <= stage_rms_limit
        )
        failed = list(evidence["failed_metrics"])
        max_relative_to_rms = npu_stage_max / scale
        # A single bad value cannot be hidden by averaging over a large GEMM.
        max_limit = health_rtol
        # Ordinary FP32 accumulation noise is approximately zero-mean.
        # A coherent offset (for example, an omitted/faulty bias) must not
        # consume the independent precision allowance across every element.
        bias_fraction_limit = 0.5
        # The bias gate protects the extra FP32 allowance. Ordinary fixed
        # tolerance may include same-sign one-ULP rounding. Verify that bound
        # explicitly: evidence can also pass via ratios (e.g. RMSE underflow).
        fixed_tolerance_accepted = evidence["passed"] and native_within_fixed_tolerance
        # Only the four finite numeric ratios may use the independent FP32
        # bound. Preserve every other hard gate, especially NaN/Inf masks and
        # the small-value error count.
        finite_ratios = {"diff_big_max", "diff_big_avg", "diff_big_sum", "diff_rmse"}
        exact_stage_fallback = (
            exact_fp32_stage
            and all(name in finite_ratios for name in failed)
            and np.isfinite(max_relative_to_rms)
            and max_relative_to_rms <= max_limit
            and np.isfinite(npu_stage_bias_fraction)
            and npu_stage_bias_fraction <= bias_fraction_limit
        )
        if exact_stage_fallback:
            failed = []
        if (
            stage_rms_ratio is None
            or not np.isfinite(stage_rms_ratio)
            or stage_rms_ratio > stage_rms_limit
        ):
            failed.append("cpu_staged_cascade.relative_rmse")
        if (
            not np.isfinite(npu_stage_rms_ratio)
            or npu_stage_rms_ratio > stage_rms_limit
        ):
            failed.append("npu_vs_cpu_staged.relative_rmse")
        if exact_fp32_stage and (
            not np.isfinite(max_relative_to_rms) or max_relative_to_rms > max_limit
        ):
            failed.append("npu_vs_cpu_staged.max_relative_to_rms")
        if (
            exact_fp32_stage
            and not fixed_tolerance_accepted
            and (
                not np.isfinite(npu_stage_bias_fraction)
                or npu_stage_bias_fraction > bias_fraction_limit
            )
        ):
            failed.append("npu_vs_cpu_staged.bias_fraction")
        return {
            "policy": "independent_cpu_staged_mc2_ratio",
            "dtype": effective_dtype,
            "passed": not failed and stage_valid,
            "accepted_by": (
                "fixed_fp32_reference_precision"
                if exact_stage_fallback and not evidence["passed"]
                else evidence["accepted_by"]
                if stage_valid
                else None
            ),
            "npu_metrics": evidence["candidate_metrics"],
            "cpu_stage_metrics": evidence["cascade_metrics"],
            "limits": evidence["limits"],
            "failed_metrics": failed,
            "finite_health_count": int(finite_count),
            "expected_overflow_count": (
                evidence["cascade_metrics"]["num_total_inf"]
                + evidence["cascade_metrics"]["num_total_ninf"]
            ),
            "npu_stage_max_relative_to_rms": max_relative_to_rms
            if exact_fp32_stage
            else None,
            "npu_stage_max_relative_to_rms_limit": max_limit
            if exact_fp32_stage
            else None,
            "npu_stage_bias_fraction": npu_stage_bias_fraction
            if exact_fp32_stage
            else None,
            "npu_stage_bias_fraction_limit": bias_fraction_limit
            if exact_fp32_stage
            else None,
            "cpu_stage_relative_rmse": stage_rms_ratio,
            "cpu_stage_relative_rmse_limit": stage_rms_limit,
            "npu_stage_relative_rmse": npu_stage_rms_ratio,
            "npu_stage_relative_rmse_limit": stage_rms_limit,
            "error_msg": ", ".join(failed),
        }
    checked = compare_tensors(
        output, cpu, dtype=str(output.dtype).removeprefix("torch."), native_output=None
    )
    failed_regions = [
        name
        for name, passed in (
            ("normal", checked.normal_passed),
            ("small_value", checked.small_value_passed),
            ("cancellation", checked.cancel_passed),
        )
        if not passed
    ]
    return {
        "policy": "fixed_repository_precision",
        "dtype": str(output.dtype).removeprefix("torch."),
        **{
            name: getattr(checked, name)
            for name in (
                "passed",
                "threshold",
                "mere",
                "mare",
                "small_value_error_count",
                "cancel_error_count",
                "normal_error_count",
                "small_value_passed",
                "cancel_passed",
                "normal_passed",
            )
        },
        "failed_regions": failed_regions,
        "error_msg": checked.error_msg or ", ".join(failed_regions),
    }


def check_mc2_references(
    cascade, golden, rank, comparison_dtype=None, *, cascade_golden=None
):
    """Validate references before their errors can relax candidate thresholds.

    An optional trusted CPU cascade emulates intermediate rounding for health
    only. Error metrics and candidate thresholds still use mathematical FP32 G.
    """
    import torch

    results = []

    staged = cascade_golden is not None

    def visit(native, cpu, health_cpu, path):
        result = MC2ThreeWayOutputResult(len(results), f"rank{rank}.{path}")
        result.metadata["rank"] = rank
        if isinstance(cpu, (tuple, list)):
            if (
                not isinstance(native, (tuple, list))
                or len(native) != len(cpu)
                or not cpu
            ):
                result.error_msg = "reference output count/type mismatch"
            elif staged and (
                not isinstance(health_cpu, (tuple, list)) or len(health_cpu) != len(cpu)
            ):
                result.error_msg = "CPU staged cascade output count/type mismatch"
            else:
                for i, (b, g) in enumerate(zip(native, cpu)):
                    visit(b, g, health_cpu[i] if staged else None, f"{path}.{i}")
                return
        elif cpu is None:
            result.passed = (
                native is None
                and path != "output"
                and (not staged or health_cpu is None)
            )
            if not result.passed:
                result.error_msg = "required reference is None"
        elif not isinstance(native, torch.Tensor) or not isinstance(cpu, torch.Tensor):
            result.error_msg = "reference is not a Tensor"
        elif native.shape != cpu.shape:
            result.error_msg = (
                f"reference shape mismatch: cascade={native.shape}, CPU={cpu.shape}"
            )
        elif cpu.dtype != torch.float32:
            result.error_msg = f"CPU reference must be FP32, got {cpu.dtype}"
        elif not torch.isfinite(cpu).all().item():
            result.error_msg = "CPU FP32 reference contains nonfinite values"
        elif staged and (
            not isinstance(health_cpu, torch.Tensor)
            or health_cpu.shape != cpu.shape
            or health_cpu.dtype != torch.float32
            or health_cpu.device.type != "cpu"
        ):
            result.error_msg = (
                "CPU staged cascade must be a CPU FP32 tensor with the reference shape"
            )
        else:
            result.dtype = str(native.dtype).removeprefix("torch.")
            effective = comparison_dtype or (
                "float32" if result.dtype.startswith("float8") else result.dtype
            )
            try:
                evidence = compare_three_way(
                    native.float().numpy(),
                    native.float().numpy(),
                    cpu.numpy(),
                    effective,
                )
                # In particular, do not inherit compare_tensors' Inf saturation relaxation.
                result.passed = evidence.pop("passed")
                # This is reference-only evidence, not a measurement of the candidate.
                evidence.pop("candidate_metrics", None)
                result.metadata.update(evidence)
                if result.passed:
                    # A stage-rounded FP16/BF16 reference may legitimately
                    # overflow even though mathematical FP32 G is finite.
                    # _reference_health validates its signed exceptional-value
                    # masks, rather than blanket-rejecting every staged Inf.
                    health = _reference_health(
                        native, cpu, health_cpu if staged else None
                    )
                    health["reference"] = (
                        "cpu_staged_cascade" if staged else "cpu_fp32_golden"
                    )
                    result.metadata["reference_check"] = health
                    result.passed = health["passed"]
                    result.error_msg = health["error_msg"] or (
                        "reference exceeds fixed precision limits"
                        if not result.passed
                        else ""
                    )
                else:
                    result.error_msg = ", ".join(evidence["failed_metrics"])
            except Exception as exc:
                # Return evidence so every rank still reaches the shared invalid
                # flag; checker errors cannot become candidate runtime failures.
                result.passed = False
                result.error_msg = f"reference precision check failed: {exc}"
        if not result.passed:
            result.metadata["failure_type"] = FAILURE_TYPE_REFERENCE_INVALID
            result.metadata["reference_valid"] = False
            result.metadata["accepted_by"] = None
        else:
            result.metadata["reference_valid"] = True
        results.append(result)

    visit(cascade, golden, cascade_golden, "output")
    return results


def compare_mc2_outputs(
    candidate, cascade, golden, rank, comparison_dtype=None, *, cascade_golden=None
):
    """Check every leaf, including optional outputs; missing references are errors."""
    import torch

    references = check_mc2_references(
        cascade, golden, rank, comparison_dtype, cascade_golden=cascade_golden
    )
    if any(not result.passed for result in references):
        return references
    reference_metadata = {result.name: result.metadata for result in references}
    results = []

    def visit(actual, native, cpu, path):
        result = MC2ThreeWayOutputResult(len(results), f"rank{rank}.{path}")
        result.metadata["rank"] = rank
        reference = reference_metadata.get(result.name, {})
        result.metadata.update(
            {
                key: reference[key]
                for key in ("reference_check", "reference_valid")
                if key in reference
            }
        )
        if isinstance(cpu, (tuple, list)):
            if not (
                isinstance(actual, (tuple, list))
                and isinstance(native, (tuple, list))
                and len(actual) == len(native) == len(cpu)
            ):
                result.error_msg = "output count/type mismatch"
            else:
                for i, (a, b, g) in enumerate(zip(actual, native, cpu)):
                    visit(a, b, g, f"{path}.{i}")
                return
        elif cpu is None:
            result.passed = actual is None and native is None and path != "output"
            if not result.passed:
                result.error_msg = (
                    "CPU primary reference is None"
                    if path == "output"
                    else "optional output must be None"
                )
        elif not all(isinstance(x, torch.Tensor) for x in (actual, native, cpu)):
            result.error_msg = "required output is None or not a Tensor"
        elif actual.shape != cpu.shape or native.shape != cpu.shape:
            result.error_msg = f"shape mismatch: candidate={actual.shape}, cascade={native.shape}, CPU={cpu.shape}"
        elif actual.dtype != native.dtype:
            result.error_msg = (
                f"dtype mismatch: candidate={actual.dtype}, cascade={native.dtype}"
            )
        elif cpu.dtype != torch.float32:
            result.error_msg = f"CPU reference must be FP32, got {cpu.dtype}"
        else:
            result.dtype = str(actual.dtype).removeprefix("torch.")
            effective = comparison_dtype or (
                "float32" if result.dtype.startswith("float8") else result.dtype
            )
            evidence = compare_three_way(
                actual.float().numpy(), native.float().numpy(), cpu.numpy(), effective
            )
            result.passed = evidence.pop("passed")
            result.metadata.update(evidence)
            result.error_msg = ", ".join(evidence["failed_metrics"])
        if result.error_msg and "candidate_metrics" not in result.metadata:
            result.metadata["failure_type"] = "compile_runtime_error"
        results.append(result)

    visit(candidate, cascade, golden, "output")
    if not results:
        results.append(
            MC2ThreeWayOutputResult(
                0, f"rank{rank}.output", error_msg="empty output contract"
            )
        )
    return results
