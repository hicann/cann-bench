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

"""GroupedDynamicBlockQuant 算子 NPU 性能基准测试

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
    "int64": torch.int64,
    "int32": torch.int32,
    "uint8": torch.uint8,
}

# cases.yaml 中 dst_type 使用 aclnn 文档枚举值（34/35/36）。
# 实测传入整型 290/291/292 在部分 shape 上会触发 507899 拷贝错误，因此映射到
# torch dtype 常量。
DST_TYPE_MAP = {
    34: torch_npu.hifloat8,
    35: torch.float8_e5m2,
    36: torch.float8_e4m3fn,
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


def build_group_list(m, attrs):
    """根据 attrs 构造 group_list 张量。"""
    group_list_values = attrs.get("group_list")
    if group_list_values is None:
        group_list_values = [m]
    else:
        group_list_values = list(group_list_values)
    if group_list_values[-1] != m:
        raise ValueError(
            f"group_list 最后一个元素 {group_list_values[-1]} 必须与 x 的 M 轴 {m} 相等"
        )
    return torch.tensor(group_list_values, dtype=torch.int32)


def run_single_case(case_id, case_cfg, device):
    """运行单个 case 的 warmup + trials。"""
    shapes = case_cfg["input_shape"]
    dtypes = case_cfg["dtype"]
    ranges = case_cfg["value_range"]
    attrs = case_cfg["attrs"]

    x = build_input(shapes[0], dtypes[0], ranges[0]).to(device)
    m = x.shape[-2]
    group_list = build_group_list(m, attrs).to(device)

    min_scale = attrs["min_scale"]
    round_mode = attrs["round_mode"]
    dst_type = DST_TYPE_MAP.get(attrs["dst_type"], attrs["dst_type"])
    row_block_size = attrs["row_block_size"]
    col_block_size = attrs["col_block_size"]
    group_list_type = attrs["group_list_type"]

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] input_shape={shapes}, dtype={dtypes}")
    print(f"           attrs={attrs}")
    print(f"           note: {case_cfg['note']}")
    print(f"{'='*60}")

    def _run_op():
        return torch_npu.npu_grouped_dynamic_block_quant(
            x=x,
            group_list=group_list,
            min_scale=min_scale,
            round_mode=round_mode,
            dst_type=dst_type,
            row_block_size=row_block_size,
            col_block_size=col_block_size,
            group_list_type=group_list_type,
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
