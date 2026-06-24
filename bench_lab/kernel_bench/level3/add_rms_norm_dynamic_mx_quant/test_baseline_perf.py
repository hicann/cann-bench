#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See License in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""AddRmsNormDynamicMxQuant 算子 NPU 性能基准测试

基于 cases.csv 中的 shape 配置，支持 warmup + trials 模式运行。
运行方式:
    # 运行单个 case (例如 case 1)
    python3 test_baseline_perf.py 1

    # 运行所有 cases
    python3 test_baseline_perf.py all

    # 用 msprof 采集性能
    msprof --application="python3 test_baseline_perf.py 1" --output=./prof_out

环境变量:
    DEVICE_ID  — NPU 卡号 (默认 0)
    NUM_WARMUP — 预热次数 (默认 10)
    NUM_TRIALS — 计时轮数 (默认 50)
"""

import csv
import json
import os
import sys
import time

import torch
import torch_npu

# =============================================================================
# 参数配置
# =============================================================================
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
NUM_WARMUP = int(os.environ.get("NUM_WARMUP", "10"))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "50"))
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_CSV = os.path.join(SCRIPT_DIR, "cases.csv")

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

# cases.yaml / proto.yaml 中 dst_type 使用 aclnn 文档枚举值（35/36/40/41）。
# 当前 torch_npu.npu_add_rms_norm_dynamic_mx_quant 需要传入 torch dtype 常量。
_DST_TYPE_MAP = {
    35: torch_npu.float8_e5m2,
    36: torch_npu.float8_e4m3fn,
    40: torch_npu.float4_e2m1fn_x2,
    41: torch_npu.float4_e1m2fn_x2,
}


def load_cases(csv_path):
    """从 cases.csv 加载测试用例。"""
    cases = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = int(row["case_id"])
            shapes = json.loads(row["input_shape"])
            dtypes = json.loads(row["dtype"])
            value_ranges = json.loads(row["value_range"])
            attrs = json.loads(row["attrs"])
            cases[case_id] = {
                "input_shape": shapes,
                "dtype": dtypes,
                "value_range": value_ranges,
                "attrs": attrs,
                "note": row.get("note", ""),
            }
    return cases


def build_input(shape, dtype_str, value_range, seed_offset):
    """根据 case 配置构造单个输入张量。"""
    if shape is None or dtype_str is None:
        return None
    dtype = DTYPE_MAP[dtype_str]
    lo, hi = value_range
    torch.manual_seed(SEED + seed_offset)
    x = torch.rand(shape, dtype=torch.float64) * (hi - lo) + lo
    return x.to(dtype)


def run_single_case(case_id, case_cfg, device):
    """运行单个 case 的 warmup + trials。"""
    shapes = case_cfg["input_shape"]
    dtypes = case_cfg["dtype"]
    ranges = case_cfg["value_range"]
    attrs = case_cfg["attrs"]

    x1 = build_input(shapes[0], dtypes[0], ranges[0], seed_offset=case_id * 10 + 0)
    x2 = build_input(shapes[1], dtypes[1], ranges[1], seed_offset=case_id * 10 + 1)
    gamma = build_input(shapes[2], dtypes[2], ranges[2], seed_offset=case_id * 10 + 2)
    beta = build_input(shapes[3], dtypes[3], ranges[3], seed_offset=case_id * 10 + 3)

    epsilon = attrs["epsilon"]
    scale_alg = attrs["scale_alg"]
    round_mode = attrs["round_mode"]
    dst_type = _DST_TYPE_MAP.get(attrs["dst_type"], attrs["dst_type"])
    output_rstd = attrs.get("output_rstd", True)

    x1 = x1.to(device).requires_grad_(output_rstd)
    x2 = x2.to(device).requires_grad_(output_rstd)
    gamma = gamma.to(device)
    beta = beta.to(device) if beta is not None else None

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] input_shape={shapes}, dtype={dtypes}")
    print(f"           attrs={attrs}")
    print(f"           note: {case_cfg['note']}")
    print(f"{'='*60}")

    def _run_op():
        return torch_npu.npu_add_rms_norm_dynamic_mx_quant(
            x1, x2, gamma,
            beta=beta,
            epsilon=epsilon,
            scale_alg=scale_alg,
            round_mode=round_mode,
            dst_type=dst_type,
        )

    # 预执行一次，确认算子可运行
    try:
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
    except Exception as e:
        print(f"[ERROR] Case {case_id} 首次执行失败: {e}")
        return False

    # Warmup
    print(f"[INFO] Warmup {NUM_WARMUP} iterations...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
    print("[INFO] Warmup finished.")

    # Trials
    print(f"[INFO] Running {NUM_TRIALS} trials...")
    timings = []
    for _ in range(NUM_TRIALS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e6)  # us

    timings.sort()
    avg_us = sum(timings) / len(timings)
    min_us = timings[0]
    max_us = timings[-1]
    p50_us = timings[len(timings) // 2]

    print(f"[RESULT] Case {case_id}: avg={avg_us:.2f} us, "
          f"min={min_us:.2f} us, max={max_us:.2f} us, p50={p50_us:.2f} us")

    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_id | all>")
        print("       case_id: 1~20 的整数")
        print("       all: 运行全部 cases")
        sys.exit(1)

    arg = sys.argv[1]

    if not os.path.exists(CASES_CSV):
        print(f"[ERROR] cases.csv 不存在: {CASES_CSV}")
        sys.exit(1)
    cases = load_cases(CASES_CSV)

    torch.npu.set_device(DEVICE_ID)
    device = f"npu:{DEVICE_ID}"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Warmup: {NUM_WARMUP}, Trials: {NUM_TRIALS}")

    if arg.lower() == "all":
        case_ids = sorted(cases.keys())
    else:
        try:
            case_ids = [int(arg)]
        except ValueError:
            print(f"[ERROR] 无效的 case_id: {arg}")
            sys.exit(1)

    success_count = 0
    for cid in case_ids:
        if cid not in cases:
            print(f"[WARN] Case {cid} 不存在，跳过。")
            continue
        ok = run_single_case(cid, cases[cid], device)
        if ok:
            success_count += 1

    print(f"\n{'='*60}")
    print(f"[SUMMARY] 成功: {success_count} / {len(case_ids)}")
    print("[INFO] 如需 msprof 采集，请执行:")
    print(f'    msprof --application="python3 {__file__} {arg}" --output=./prof_out')
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
