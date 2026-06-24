#!/usr/bin/env python3
"""
MaxPool3dGradWithArgmax 算子 NPU 性能基准测试。

用法:
    python3 test_baseline_perf.py <case_id|all>
    msprof --application="python3 test_baseline_perf.py <case_id>" --output=./prof_out

环境变量:
    DEVICE_ID  — NPU 卡号 (默认 0)
    NUM_WARMUP — 预热次数 (默认 10)
    NUM_TRIALS — 计时轮数 (默认 50)
"""

import os
import sys
import csv
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
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
    "int32": torch.int32,
    "int64": torch.int64,
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
            attrs = json.loads(row["attrs"])
            value_range = json.loads(row["value_range"])
            cases[case_id] = {
                "shapes": shapes,
                "dtypes": [DTYPE_MAP[d] for d in dtypes],
                "attrs": attrs,
                "value_range": value_range,
                "note": row.get("note", ""),
            }
    return cases


def _gen_tensor(shape, dtype, value_range, device):
    """生成一个输入张量。"""
    vmin, vmax = value_range
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if dtype in (torch.int32, torch.int64):
        # 整数占位张量，后续会被 get_input 替换
        return torch.zeros(shape, dtype=dtype, device=device)

    if vmin == 0 and vmax == 0:
        t = torch.zeros(shape, dtype=dtype)
    else:
        t = torch.rand(shape, dtype=torch.float32) * (vmax - vmin) + vmin
        t = torch.clamp(t, torch.finfo(dtype).min, torch.finfo(dtype).max)
        t = t.to(dtype)
    return t.to(device)


def _gen_valid_inputs(x, ksize, strides, pads, dilation, ceil_mode):
    """运行正向 maxpool 生成合法的 grad 与 argmax。"""
    with torch.no_grad():
        _, indices = F.max_pool3d(
            x.float(),
            kernel_size=ksize,
            stride=strides,
            padding=pads,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=True,
        )
    grad = torch.randn_like(indices, dtype=x.dtype)
    return grad, indices


def build_inputs(case_cfg, device):
    """根据 case 配置构造输入张量。"""
    shapes = case_cfg["shapes"]
    dtypes = case_cfg["dtypes"]
    ranges = case_cfg["value_range"]
    attrs = case_cfg["attrs"]

    x = _gen_tensor(shapes[0], dtypes[0], ranges[0], device)
    grad = _gen_tensor(shapes[1], dtypes[1], ranges[1], device)
    argmax = _gen_tensor(shapes[2], dtypes[2], ranges[2], device)

    grad, argmax = _gen_valid_inputs(
        x,
        attrs["ksize"],
        attrs["strides"],
        attrs["pads"],
        attrs.get("dilation", [1, 1, 1]),
        attrs.get("ceil_mode", False),
    )
    return x, grad, argmax


def run_single_case(case_id, case_cfg, device):
    """运行单个 case 的 warmup + trials。"""
    x, grad, argmax = build_inputs(case_cfg, device)
    attrs = case_cfg["attrs"]

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] x_shape={case_cfg['shapes'][0]}, dtype={case_cfg['dtypes'][0]}")
    print(f"           ksize={attrs['ksize']}, strides={attrs['strides']}, pads={attrs['pads']}, ceil_mode={attrs['ceil_mode']}")
    print(f"           note: {case_cfg['note']}")
    print(f"{'='*60}")

    def _run_op():
        return torch.ops.aten.max_pool3d_with_indices_backward(
            grad,
            x,
            kernel_size=attrs["ksize"],
            stride=attrs["strides"],
            padding=attrs["pads"],
            dilation=attrs.get("dilation", [1, 1, 1]),
            ceil_mode=attrs.get("ceil_mode", False),
            indices=argmax,
        )

    try:
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
    except Exception as e:
        print(f"[ERROR] Case {case_id} 首次执行失败: {e}")
        return False

    marker = torch.add(torch.zeros(1, device=device), float(case_id))
    torch.npu.synchronize()
    print(f"[MARKER] Begin case {case_id} marker_value={marker.item():.1f}")

    print(f"[INFO] Warmup {NUM_WARMUP} iterations...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
    print("[INFO] Warmup finished.")

    print(f"[INFO] Running {NUM_TRIALS} trials...")
    timings = []
    for _ in range(NUM_TRIALS):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = _run_op()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1e6)

    avg_us = sum(timings) / len(timings)
    min_us = min(timings)
    max_us = max(timings)
    p50_us = sorted(timings)[len(timings) // 2]

    print(f"[RESULT] Case {case_id}: avg={avg_us:.2f} us, min={min_us:.2f} us, max={max_us:.2f} us, p50={p50_us:.2f} us")

    marker_end = torch.add(torch.zeros(1, device=device), float(case_id) + 0.5)
    torch.npu.synchronize()
    print(f"[MARKER] End case {case_id} marker_value={marker_end.item():.1f}")

    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <case_id | all>")
        print("       case_id: 1~N 的整数")
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
