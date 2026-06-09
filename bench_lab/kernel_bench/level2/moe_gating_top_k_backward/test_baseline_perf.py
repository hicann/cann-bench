#!/usr/bin/env python3
"""
npu_moe_gating_top_k_backward 算子性能测试脚本

基于 cases.csv 中的 shape 配置，支持 warmup + trials 模式运行。
运行方式:
    # 运行单个 case (例如 case 1)
    python3 run_moe_gating_top_k_backward_perf.py 1

    # 运行所有 cases
    python3 run_moe_gating_top_k_backward_perf.py all

    # 用 msprof 采集性能
    msprof --application="python3 run_moe_gating_top_k_backward_perf.py 1" --output=./prof_out
"""

import os
import sys
import csv
import json
import time
import random
import numpy as np
import torch
import torch_npu

# =============================================================================
# 参数配置
# =============================================================================
DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
NUM_WARMUP = int(os.environ.get("NUM_WARMUP", "10"))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "50"))
SEED = 42

# cases.csv 路径
CASES_CSV = "./cases.csv"


# =============================================================================
# dtype 映射
# =============================================================================
DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
}


def load_cases(csv_path):
    """从 cases.csv 加载测试用例。"""
    cases = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = int(row["case_id"])
            shapes = json.loads(row["input_shape"])  # [[M,N], [M,K], [M,K]]
            dtypes = json.loads(row["dtype"])        # ["float32", "bfloat16", "int32"]
            attrs = json.loads(row["attrs"])
            value_range = json.loads(row["value_range"])
            cases[case_id] = {
                "shape_x_norm": shapes[0],      # [M, N]
                "shape_grad_y": shapes[1],      # [M, K]
                "shape_expert_idx": shapes[2],  # [M, K]
                "dtype_x_norm": DTYPE_MAP[dtypes[0]],
                "dtype_grad_y": DTYPE_MAP[dtypes[1]],
                "dtype_expert_idx": DTYPE_MAP[dtypes[2]],
                "renorm": attrs["renorm"],
                "norm_type": attrs["norm_type"],
                "routed_scaling_factor": float(attrs["routed_scaling_factor"]),
                "eps": float(attrs["eps"]),
                "value_range": value_range,
                "note": row.get("note", ""),
            }
    return cases


def build_inputs(case_cfg, device):
    """根据 case 配置构造输入张量。"""
    M, N = case_cfg["shape_x_norm"]
    _, K = case_cfg["shape_grad_y"]

    # 固定随机种子保证可复现
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # x_norm: float32
    x_range = case_cfg["value_range"][0]
    x_norm = torch.rand(M, N, dtype=torch.float32) * (x_range[1] - x_range[0]) + x_range[0]

    # grad_y: 对应 dtype
    y_range = case_cfg["value_range"][1]
    grad_y = torch.rand(M, K, dtype=torch.float32) * (y_range[1] - y_range[0]) + y_range[0]
    grad_y = grad_y.to(case_cfg["dtype_grad_y"])

    # expert_idx: int32, 范围 [0, N-1]
    idx_range = case_cfg["value_range"][2]
    expert_idx = torch.randint(idx_range[0], idx_range[1] + 1, (M, K), dtype=torch.int32)

    return x_norm.to(device), grad_y.to(device), expert_idx.to(device)


def run_single_case(case_id, case_cfg, device):
    """运行单个 case 的 warmup + trials。"""
    x_norm, grad_y, expert_idx = build_inputs(case_cfg, device)

    M, N = case_cfg["shape_x_norm"]
    _, K = case_cfg["shape_grad_y"]
    grad_y_dtype = case_cfg["dtype_grad_y"]

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] M={M}, N={N}, K={K}, grad_y_dtype={grad_y_dtype}")
    print(f"           renorm={case_cfg['renorm']}, norm_type={case_cfg['norm_type']}, "
          f"routed_scaling_factor={case_cfg['routed_scaling_factor']}, eps={case_cfg['eps']}")
    print(f"           note: {case_cfg['note']}")
    print(f"{'='*60}")
    

    def _run_op():
        return torch_npu.npu_moe_gating_top_k_backward(
            x_norm,
            grad_y,
            expert_idx,
            renorm=case_cfg["renorm"],
            norm_type=case_cfg["norm_type"],
            routed_scaling_factor=case_cfg["routed_scaling_factor"],
            eps=case_cfg["eps"],
        )

    # 预执行一次，确认算子可运行
    try:
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
    except Exception as e:
        print(f"[ERROR] Case {case_id} 首次执行失败: {e}")
        return False

    # Case boundary marker (visible in profiling)
    marker = torch.add(torch.zeros(1, device=device), float(case_id))
    torch.npu.synchronize()
    print(f"[MARKER] Begin case {case_id} marker_value={marker.item():.1f}")

    # Warmup
    print(f"[INFO] Warmup {NUM_WARMUP} iterations...")
    for i in range(NUM_WARMUP):
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
    print(f"[INFO] Warmup finished.")

    # Trials
    print(f"[INFO] Running {NUM_TRIALS} trials...")
    timings = []
    for i in range(NUM_TRIALS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e6)  # us

    avg_us = sum(timings) / len(timings)
    min_us = min(timings)
    max_us = max(timings)
    p50_us = sorted(timings)[len(timings) // 2]

    print(f"[RESULT] Case {case_id}: avg={avg_us:.2f} us, min={min_us:.2f} us, max={max_us:.2f} us, p50={p50_us:.2f} us")

    # Case end marker (visible in profiling)
    marker_end = torch.add(torch.zeros(1, device=device), float(case_id) + 0.5)
    torch.npu.synchronize()
    print(f"[MARKER] End case {case_id} marker_value={marker_end.item():.1f}")

    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_id | all>")
        print(f"       case_id: 1~20 的整数")
        print(f"       all: 运行全部 cases")
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
    print(f"[INFO] 如需 msprof 采集，请执行:")
    print(f'    msprof --application="python3 {__file__} {arg}" --output=./prof_out')
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
