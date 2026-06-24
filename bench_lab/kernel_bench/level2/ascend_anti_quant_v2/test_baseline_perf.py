#!/usr/bin/env python3
"""
AscendAntiQuantV2 NPU 性能测试脚本

用法:
    python3 test_baseline_perf.py [case_id|all]
"""

import os
import sys
import time
import random
import yaml
import numpy as np
import torch
import torch_npu

DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
NUM_WARMUP = int(os.environ.get("NUM_WARMUP", "10"))
NUM_TRIALS = int(os.environ.get("NUM_TRIALS", "50"))
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_YAML = os.path.join(SCRIPT_DIR, "cases.yaml")

DTYPE_MAP = {
    "int8": torch.int8,
    "int32": torch.int32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def load_cases(yaml_path):
    cases = {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for raw in data.get("cases", []):
        case_id = int(raw["case_id"])
        input_shapes = raw.get("input_shape", [])
        dtypes = raw.get("dtype", [])
        value_ranges = raw.get("value_range", [])

        x_shape = list(input_shapes[0])
        x_dtype = DTYPE_MAP[dtypes[0]]
        if x_dtype == torch.int32:
            x_shape[-1] = x_shape[-1] // 8

        scale_shape = list(input_shapes[1]) if input_shapes[1] is not None else []
        scale_dtype = DTYPE_MAP[dtypes[1]] if len(dtypes) > 1 else torch.float32

        offset_shape = list(input_shapes[2]) if len(input_shapes) > 2 and input_shapes[2] is not None else []
        offset_dtype = DTYPE_MAP[dtypes[2]] if len(dtypes) > 2 and input_shapes[2] is not None else None

        cases[case_id] = {
            "input_shape": x_shape,
            "dtype": x_dtype,
            "scale_dtype": scale_dtype,
            "scale_shape": scale_shape,
            "offset_dtype": offset_dtype,
            "offset_shape": offset_shape,
            "attrs": raw.get("attrs", {}),
            "value_range": value_ranges[0] if value_ranges else [-128, 127],
            "note": raw.get("note", ""),
        }
    return cases


def build_input(shape, dtype, value_range):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    vmin, vmax = value_range
    if dtype == torch.int32:
        num_int4 = shape[-1] * 8
        int4_shape = tuple(shape[:-1]) + (num_int4,)
        if vmin == 0 and vmax == 0:
            int4_vals = np.zeros(int4_shape, dtype=np.int8)
        else:
            int4_vals = np.random.randint(vmin, vmax + 1, size=int4_shape, dtype=np.int8)
        int4_u8 = (int4_vals.astype(np.uint8) & 0x0F).reshape(-1, 8)
        packed = np.zeros((int4_u8.shape[0],), dtype=np.uint32)
        for i in range(8):
            packed |= (int4_u8[:, i].astype(np.uint32) << (4 * i))
        return torch.from_numpy(packed.reshape(shape)).to(torch.int32)
    if dtype == torch.int8:
        if vmin == 0 and vmax == 0:
            return torch.zeros(shape, dtype=torch.int8)
        return torch.randint(vmin, vmax + 1, shape, dtype=torch.int8)
    if vmin == 0 and vmax == 0:
        return torch.zeros(shape, dtype=dtype)
    x = torch.rand(shape, dtype=torch.float32) * (vmax - vmin) + vmin
    return x.to(dtype)


def build_scale_offset(scale_shape, offset_shape, scale_dtype, offset_dtype=None):
    torch.manual_seed(SEED + 1)
    np.random.seed(SEED + 1)
    random.seed(SEED + 1)
    scale = torch.rand(scale_shape, dtype=torch.float32) * 0.1 + 0.001
    scale = scale.to(scale_dtype)
    offset = None
    odtype = offset_dtype if offset_dtype is not None else scale_dtype
    if offset_shape:
        offset = (torch.rand(offset_shape, dtype=torch.float32) - 0.5) * 0.2
        offset = offset.to(odtype)
    return scale, offset


def run_single_case(case_id, case_cfg, device):
    x = build_input(case_cfg["input_shape"], case_cfg["dtype"], case_cfg["value_range"]).clone().contiguous().to(device)
    scale, offset = build_scale_offset(
        case_cfg["scale_shape"], case_cfg["offset_shape"],
        case_cfg["scale_dtype"], case_cfg.get("offset_dtype"))
    scale = scale.to(device)
    if offset is not None:
        offset = offset.to(device)

    dst_type = case_cfg["attrs"]["dst_type"]
    npu_dtype = torch.bfloat16 if dst_type == 27 else torch.float16

    kwargs = {"dst_dtype": npu_dtype}
    if offset is not None:
        kwargs["offset"] = offset

    def _run_op():
        return torch_npu.npu_anti_quant(x, scale, **kwargs)

    print(f"\n{'='*60}")
    print(f"[CASE {case_id}] shape={case_cfg['input_shape']}, dtype={case_cfg['dtype']}, dst_type={dst_type}")
    print(f"           scale_shape={case_cfg['scale_shape']}, offset_shape={case_cfg['offset_shape']}")
    print(f"           note: {case_cfg['note']}")
    print(f"{'='*60}")

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
        sys.exit(1)

    arg = sys.argv[1]
    cases = load_cases(CASES_YAML)

    torch.npu.set_device(DEVICE_ID)
    device = f"npu:{DEVICE_ID}"
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Warmup: {NUM_WARMUP}, Trials: {NUM_TRIALS}")

    if arg.lower() == "all":
        case_ids = sorted(cases.keys())
    else:
        case_ids = [int(arg)]

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
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
