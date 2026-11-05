#!/usr/bin/env python3

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Reproducible, optimistic A5 MC2 roofline bounds. See mc2_a5.md.

Only documented resource bounds are used; no measured candidate latency enters
T_HW. --write updates only A5 MC2 entries, preserving other published metadata.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

try:
    from docs.hap_thw_model import hap_lib as H
except ModuleNotFoundError:
    import hap_lib as H  # Direct execution from docs/hap_thw_model/.

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "bench_lab/kernel_bench"
PLATFORMS = tuple(H.PUBLIC_PLAT)
OPS = {
    "matmul_all_reduce",
    "all_gather_matmul",
    "all_gather_matmul_v2",
    "allto_all_matmul",
    "matmul_allto_all",
    "matmul_reduce_scatter_v2",
    "allto_allv_grouped_mat_mul",
    "grouped_mat_mul_allto_allv",
    "quant_all_reduce",
    "quant_reduce_scatter",
    "quant_grouped_mat_mul_allto_allv",
    "allto_allv_quant_grouped_mat_mul",
}
BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "int8": 1,
    "uint8": 1,
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "int32": 4,
    "int64": 8,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}


def a5_cases(bench=BENCH):
    """Discover all explicitly A5 MC2 cases; fail on unsupported new operators."""
    for path in sorted((bench / "level4").glob("*/cases.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        for case in raw.get("cases", []) if isinstance(raw, dict) else raw:
            attrs = case.get("attrs", {})
            if attrs.get("mc2_distributed") and "A5" in str(attrs.get("platform", "")):
                if path.parent.name not in OPS:
                    raise ValueError(f"Missing MC2 resource model: {path.parent.name}")
                yield path.parent.name, case


def resource_counts(op, case):
    shapes, dtypes, attrs = case["input_shape"], case["dtype"], case["attrs"]
    p = int(attrs["world_size"])
    if p != 2:
        raise ValueError("This published model is validated for world_size=2 only")
    if op in ("allto_all_matmul", "matmul_allto_all") and attrs.get("is_quant", False):
        return mx_alltoall_resource_counts(op, case)
    if op in ("quant_all_reduce", "quant_reduce_scatter"):
        return quant_collective_resource_counts(op, case)
    if op in (
        "allto_allv_grouped_mat_mul",
        "grouped_mat_mul_allto_allv",
        "quant_grouped_mat_mul_allto_allv",
        "allto_allv_quant_grouped_mat_mul",
    ):
        return grouped_resource_counts(op, case)
    m, k = math.prod(shapes[0][:-1]), shapes[0][-1]
    trans = attrs.get("is_trans_x2", attrs.get("is_trans_b", False))
    n = shapes[1][-2] if trans else shapes[1][-1]
    weight_k = shapes[1][-1] if trans else shapes[1][-2]
    xbytes = math.prod(shapes[0]) * BYTES[dtypes[0]]
    out_bytes = BYTES[attrs.get("out_dtype", dtypes[0])]
    if op == "allto_all_matmul":
        if m % p:
            raise ValueError("AlltoAllMatmul M must divide world_size")
        gemm_m, gemm_k = m // p, k * p
        network_bytes = xbytes // 2
    elif op in ("all_gather_matmul", "all_gather_matmul_v2"):
        gemm_m, gemm_k = m * p, k
        network_bytes = xbytes
    else:
        gemm_m, gemm_k = m, k
        # Two ranks: all-reduce sends one full partial output per rank;
        # reduce-scatter/all-to-all sends the remote half.
        network_bytes = m * n * out_bytes
        if op != "matmul_all_reduce":
            network_bytes //= 2
    if gemm_k != weight_k:
        raise ValueError(f"{op}: K mismatch {gemm_k} != {weight_k}")
    read_bytes = sum(
        math.prod(s) * BYTES[d] for s, d in zip(shapes, dtypes) if s is not None
    )
    return {
        "cube_ops": 2 * gemm_m * n * gemm_k,
        "read_bytes": read_bytes,
        "network_bytes": network_bytes,
        "cube_type": "cube8" if BYTES[dtypes[0]] == 1 else "cube16",
    }


def mx_alltoall_resource_counts(op, case):
    """MX input/scale traffic for AlltoAll-first versus output traffic for GEMM-first."""
    shapes, dtypes, attrs = case["input_shape"], case["dtype"], case["attrs"]
    if len(shapes) != 5 or len(dtypes) != 5:
        raise ValueError("MX AlltoAll requires x1, x2, optional bias and two scales")
    fp8 = ("float8_e4m3fn", "float8_e5m2")
    if dtypes[0] not in fp8 or dtypes[1] not in fp8 or dtypes[3:] != ["uint8", "uint8"]:
        raise ValueError("MX AlltoAll requires FP8 matrices and raw E8M0 scales")
    if (attrs.get("x1_quant_mode"), attrs.get("x2_quant_mode")) != (6, 6):
        raise ValueError("MX AlltoAll requires quantization modes (6, 6)")
    if not attrs.get("is_trans_x2") or any(len(shapes[i]) != 2 for i in (0, 1)):
        raise ValueError("MX AlltoAll requires two matrices and transposed x2")
    m, k = shapes[0]
    n, weight_k = shapes[1]
    p = int(attrs["world_size"])
    gather_first = op == "allto_all_matmul"
    gemm_k = k * p if gather_first else k
    gemm_m = m // p if gather_first else m
    if min(m, k, n) <= 0 or weight_k != gemm_k:
        raise ValueError("MX AlltoAll has invalid matrix dimensions or K mismatch")
    if (gather_first and (m % p or k % 64)) or (not gather_first and n % p):
        raise ValueError("MX AlltoAll dimensions violate communication/scale alignment")
    if shapes[3:] != [[m, (k + 63) // 64, 2], [n, (gemm_k + 63) // 64, 2]]:
        raise ValueError("MX AlltoAll scale shapes do not match the matrices")
    if shapes[2] is not None and (shapes[2] != [n] or dtypes[2] != "float32"):
        raise ValueError("MX AlltoAll bias must be FP32 with shape [N]")
    if attrs.get("out_dtype") not in ("float16", "bfloat16"):
        raise ValueError("MX AlltoAll output must be FP16 or BF16")
    read_bytes = sum(
        math.prod(s) * BYTES[d] for s, d in zip(shapes, dtypes) if s is not None
    )
    if gather_first:
        # The raw FP8 tokens AND their E8M0 scale groups cross the network.
        network_bytes = (m * k + math.prod(shapes[3])) // 2
    else:
        # Quantized operands stay local; the rounded FP16/BF16 output is sent.
        network_bytes = m * n * BYTES[attrs["out_dtype"]] // 2
    return {
        "cube_ops": 2 * gemm_m * n * gemm_k,
        "read_bytes": read_bytes,
        "network_bytes": network_bytes,
        "cube_type": "cube8",
    }


def quant_collective_resource_counts(op, case):
    """INT8 + FP32 scales are sent before dequantization, not FP32 output data."""
    shapes, dtypes, attrs = case["input_shape"], case["dtype"], case["attrs"]
    x, scales = shapes
    if len(x) not in (2, 3) or any(size <= 0 for size in x):
        raise ValueError("Quant collective x must be a nonempty 2D or 3D tensor")
    h = x[-1]
    if not 1024 <= h <= 8192 or h % 128 or math.prod(x[:-1]) % attrs["world_size"]:
        raise ValueError(
            "Quant collective H or row count does not satisfy A5 constraints"
        )
    if scales != x[:-1] + [h // 128] or dtypes != ["int8", "float32"]:
        raise ValueError(
            "Quant collective model requires INT8 with FP32 group-128 scales"
        )
    read_bytes = sum(math.prod(s) * BYTES[d] for s, d in zip(shapes, dtypes))
    # Two ranks: QAR gathers the entire peer input; QRS exchanges half per rank.
    network_bytes = read_bytes if op == "quant_all_reduce" else read_bytes // 2
    return {
        "cube_ops": 0,
        "read_bytes": read_bytes,
        "network_bytes": network_bytes,
        "cube_type": "none",
    }


def grouped_resource_counts(op, case):
    """Two-rank GMM; counts are C[source][destination][expert]."""
    shapes, dtypes, attrs = case["input_shape"], case["dtype"], case["attrs"]
    p, e = attrs["world_size"], attrs["e"]
    gather_first = op in (
        "allto_allv_grouped_mat_mul",
        "allto_allv_quant_grouped_mat_mul",
    )
    counts = attrs["token_counts"]
    if len(counts) != p or any(len(row) != p for row in counts):
        raise ValueError("token_counts must have shape [world_size, world_size, e]")
    if any(
        len(edge) != e or any(type(n) is not int or n < 0 for n in edge)
        for row in counts
        for edge in row
    ):
        raise ValueError("token_counts must contain nonnegative integer expert counts")
    sends = [sum(sum(edge) for edge in row) for row in counts]
    receives = [sum(sum(counts[s][r]) for s in range(p)) for r in range(p)]
    if sends != [shapes[0][0]] * p or receives != sends:
        raise ValueError(
            "Published GMM cases require equal per-rank input/output row totals"
        )
    k = shapes[0][1]
    trans = attrs.get("trans_gmm_weight", False)
    weight_k, n = shapes[1][2 if trans else 1], shapes[1][1 if trans else 2]
    if shapes[1][0] != e or weight_k != k:
        raise ValueError("GMM expert count or K mismatch")
    compute_rows = receives if gather_first else sends
    cube_ops = max(2 * rows * k * n for rows in compute_rows)
    # send_counts/recv_counts are host lists, not device tensor reads.
    read_bytes = sum(
        math.prod(shapes[i]) * BYTES[dtypes[i]]
        for i in (0, 4, 5)
        if shapes[i] is not None
    )
    if gather_first:
        active = [
            sum(any(counts[s][r][expert] for s in range(p)) for expert in range(e))
            for r in range(p)
        ]
    else:
        active = [
            sum(any(counts[r][d][expert] for d in range(p)) for expert in range(e))
            for r in range(p)
        ]
    # Empty experts do not require their weight matrices to be fetched.
    read_bytes += max(active) * k * n * BYTES[dtypes[1]]
    if op in ("quant_grouped_mat_mul_allto_allv", "allto_allv_quant_grouped_mat_mul"):
        # MX scales: one raw E8M0 byte for each group of 32 values.
        # The first six input slots match nonquantized GMM; scales follow.
        read_bytes += sum(
            math.prod(shapes[i]) * BYTES[dtypes[i]]
            for i in (6, 8, 9)
            if shapes[i] is not None
        )
        read_bytes += max(active) * math.prod(shapes[7][1:]) * BYTES[dtypes[7]]
    if shapes[4] is not None:
        m2, k2 = shapes[4]
        trans_mm = attrs.get("trans_mm_weight", False)
        weight_k2, n2 = shapes[5][1 if trans_mm else 0], shapes[5][0 if trans_mm else 1]
        if weight_k2 != k2:
            raise ValueError("Shared MatMul K mismatch")
        cube_ops += 2 * m2 * k2 * n2
    width = k if gather_first else n
    remote_rows = max(
        sum(sum(counts[r][d]) for d in range(p) if d != r) for r in range(p)
    )
    network_bytes = remote_rows * width * BYTES[attrs.get("out_dtype", dtypes[0])]
    if op == "allto_allv_quant_grouped_mat_mul":
        scale_pairs = (k + 63) // 64
        if shapes[6] != [shapes[0][0], scale_pairs, 2]:
            raise ValueError(
                "AlltoAllvQuantGMM requires MX FP8 with padded E8M0 scales"
            )
        # Both FP8 input values and E8M0 input scales cross the network;
        # output FP16/BF16 tensors are computed locally after the exchange.
        network_bytes = remote_rows * (k * BYTES[dtypes[0]] + scale_pairs * 2)
    return {
        "cube_ops": cube_ops,
        "read_bytes": read_bytes,
        "network_bytes": network_bytes,
        "cube_type": "cube8" if BYTES[dtypes[0]] == 1 else "cube16",
    }


def supported_platforms(case):
    platforms = case["attrs"].get("supported_devices", PLATFORMS)
    if not platforms or any(platform not in PLATFORMS for platform in platforms):
        raise ValueError(f"Unsupported A5 platform selection: {platforms}")
    return platforms


def compute(op, case, platform):
    if platform not in supported_platforms(case):
        raise ValueError(f"{op} does not support {platform}")
    counts = resource_counts(op, case)
    dtype = {"cube8": "int8", "cube16": "float16", "none": "float16"}[
        counts["cube_type"]
    ]
    parts = {
        "cube_us": H.t_cube_us(counts["cube_ops"], dtype, platform)
        if counts["cube_ops"]
        else 0.0,
        "read_us": H.t_read_us(counts["read_bytes"], platform),
        "network_us": H.t_network_us(counts["network_bytes"], platform),
    }
    # Round down: published rounding must not tighten an optimistic bound.
    return {
        "t_hw_us": max(1.0, math.floor(max(parts.values()) * 1e6) / 1e6),
        "components_us": parts,
        "counts": counts,
    }


def fingerprint(value):
    """Hash canonical JSON, independent of YAML formatting and dictionary ordering."""
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_metadata(cases, platform):
    selected = sorted(
        ((op, case) for op, case in cases if platform in supported_platforms(case)),
        key=lambda item: (item[0], item[1]["case_id"]),
    )
    ranges = {}
    for op, case in selected:
        ranges.setdefault(op, []).append(case["case_id"])
    return {
        "model": "mc2-a5-resource-lower-bound-v1",
        "source": "docs/hap_thw_model/mc2_a5.md",
        "hardware_constants": H.PUBLIC_PLAT[platform],
        "model_sha256": fingerprint(
            {
                "mc2": Path(__file__).read_text(encoding="utf-8"),
                "hap": Path(H.__file__).read_text(encoding="utf-8"),
            }
        ),
        "case_sha256": fingerprint(selected),
        "case_count": len(selected),
        "case_ids": {op: f"{min(ids)}-{max(ids)}" for op, ids in ranges.items()},
        "world_size": 2,
        "confidence": "optimistic theoretical resource bound, not calibrated latency",
    }


def generate(write=False, bench=BENCH):
    cases = list(a5_cases(bench))
    for platform in PLATFORMS:
        path = bench / "metadata" / f"{platform}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        meta = data.setdefault("_metadata", {})
        meta.setdefault("hardware", platform)
        meta["mc2_hap"] = model_metadata(cases, platform)
        values = []
        for op, case in cases:
            if platform not in supported_platforms(case):
                continue
            result = compute(op, case, platform)
            entry = (
                data.setdefault("level4", {})
                .setdefault(op, {})
                .setdefault(str(case["case_id"]), {})
            )
            entry["t_hw_us"] = result["t_hw_us"]
            values.append(result["t_hw_us"])
        if write:
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        else:
            current = json.loads(path.read_text(encoding="utf-8"))
            if current != data:
                raise ValueError(f"Stale MC2 metadata: {path}; regenerate with --write")
        print(
            f"{platform}: {len(values)} A5 cases; T_HW {min(values):.6f}..{max(values):.6f} us"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    generate(parser.parse_args().write)
