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

"""NLLLoss 算子 NPU 性能基准测试

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

# cases.csv 路径（相对于脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_CSV = os.path.join(SCRIPT_DIR, "cases.csv")

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
}

REDUCTION_MAP = {"none": 0, "mean": 1, "sum": 2}


def _flatten_nll_loss_inputs(input, target):
    """将 1D/高维输入 flatten 为 torch.ops.aten.nll_loss_forward 支持的 2D 形式。

    torch.ops.aten.nll_loss_forward 原生仅支持 1D/2D 输入，对于 3D/4D/5D 输入
    需要 flatten 为 2D 后再调用。

    Returns:
        (input_2d, target_1d, orig_target_shape, is_1d)
    """
    is_1d = input.dim() == 1
    if is_1d:
        input_2d = input.unsqueeze(0)
        target_1d = target.unsqueeze(0)
        return input_2d, target_1d, None, True

    if input.dim() == 2:
        return input, target, None, False

    # 高维: (N, C, d1, d2, ...) -> (N*D, C), target -> (N*D,)
    N, C = input.shape[0], input.shape[1]
    input_2d = input.reshape(N, C, -1).transpose(1, 2).reshape(-1, C)
    target_1d = target.reshape(-1)
    return input_2d, target_1d, target.shape, False


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
                "dtype": [DTYPE_MAP[d] for d in dtypes],
                "value_range": value_ranges,
                "attrs": attrs,
                "note": row.get("note", ""),
            }
    return cases


def build_input(shape, dtype, value_range):
    """根据 case 配置构造单个输入张量。"""
    lo, hi = value_range
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        torch.manual_seed(SEED)
        x = torch.rand(shape, dtype=torch.float64) * (hi - lo) + lo
        return x.to(dtype)
    else:
        return torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)


def run_single_case(case_id, case_cfg, device):
    """运行单个 case 的 warmup + trials。"""
    shapes = case_cfg["input_shape"]
    dtypes = case_cfg["dtype"]
    ranges = case_cfg["value_range"]
    attrs = case_cfg["attrs"]

    x = build_input(shapes[0], dtypes[0], ranges[0]).to(device)
    target = build_input(shapes[1], dtypes[1], ranges[1]).to(torch.int64).to(device)
    weight = build_input(shapes[2], dtypes[2], ranges[2]).to(device)

    x_flat, target_flat, _, _ = _flatten_nll_loss_inputs(x, target)
    reduction_int = REDUCTION_MAP[attrs["reduction"]]
    ignore_index = attrs["ignore_index"]

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] input_shape={shapes}, dtype={dtypes}")
    print(f"           reduction={attrs['reduction']}, ignore_index={ignore_index}")
    print(f"           note: {case_cfg['note']}")
    print(f"{'='*60}")

    def _run_op():
        return torch.ops.aten.nll_loss_forward(
            x_flat, target_flat, weight, reduction_int, ignore_index
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

    # 加载 cases
    if not os.path.exists(CASES_CSV):
        print(f"[ERROR] cases.csv 不存在: {CASES_CSV}")
        sys.exit(1)
    cases = load_cases(CASES_CSV)

    # 设置设备
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
