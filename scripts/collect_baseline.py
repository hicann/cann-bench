#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Baseline 性能采集脚本

复用评测体系的性能采集逻辑（PerfEvaluator + KernelDetailsStrategy），
复用 inner 的 NPU 参考算子代码（refs/level{1-4}.py + inputs.py），
产出 metadata/<hardware>.json（BaselineStore 可加载）。

使用方式:
    python scripts/collect_baseline.py --op level1/exp
    python scripts/collect_baseline.py --level 1
    python scripts/collect_baseline.py --all
    python scripts/collect_baseline.py --op level1/exp --warmup 5 --repeat 20
"""

import argparse
import gc
import json
import logging
import os
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# ---------------------------------------------------------------------------
# ACL 模式设置（必须在 import torch/torch_npu 之前设置）
# 单算子 benchmark 场景下 ACL 是正确模式：
# - Ge 图优化不适用（只有一个算子，无融合对象）
# - ACL 确保 profiler 产出完整 kernel_details.csv
# ---------------------------------------------------------------------------
os.environ["ASCEND_LAUNCH_MODE"] = "ACL"

# ---------------------------------------------------------------------------
# torchair alias — torch_npu 内嵌了 torchair 子模块，
# refs 中使用 `import torchair`，但顶层 torchair 包在 pip 上不可独立安装。
# 将 torch_npu.dynamo.torchair 注册到 sys.modules 使得 `import torchair` 能找到。
# ---------------------------------------------------------------------------
try:
    import torch_npu
    from torch_npu.dynamo import torchair as _torchair_inner
    sys.modules.setdefault("torchair", _torchair_inner)
except ImportError:
    pass  # 没有 torch_npu 就没有 torchair，refs 中会自然 skip

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
BASELINE_DIR = PROJECT_ROOT / "scripts" / "baseline"

sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(BASELINE_DIR))

# 导入 refs 体系（迁移后的 inner 代码）
import inputs as bench_inputs
import ref_registry

# 导入评测体系的公开类（不修改 src 下任何文件）
from kernel_eval.config import Config, get_config, get_project_root
from kernel_eval.eval.perf_eval import PerfEvaluator
from kernel_eval.eval.op_runner import OpRunner, OpRunResult
from kernel_eval.utils.device_manager import DeviceManager, DeviceConfig
from kernel_eval.utils.baseline_store import BaselineStore
from kernel_eval.utils.baseline_resolver import DEFAULT_HARDWARE, resolve_hardware
from kernel_eval.registry.loader_registry import get_case_loader, get_task_loader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _normalize_attrs(attrs: dict) -> dict:
    """PyYAML-string normalization（与 bench_baseline.py 一致）"""
    if not attrs:
        return {}
    out = {}
    for k, v in attrs.items():
        if isinstance(v, str):
            sv = v.strip()
            sv_l = sv.lower()
            if sv_l in ("inf", "+inf"):
                out[k] = float("inf"); continue
            if sv_l == "-inf":
                out[k] = float("-inf"); continue
            if sv_l == "nan":
                out[k] = float("nan"); continue
            if sv_l in ("null", "~", ""):
                out[k] = None; continue
            if sv_l == "true":
                out[k] = True; continue
            if sv_l == "false":
                out[k] = False; continue
            try:
                if any(c in sv for c in ".eE"):
                    out[k] = float(sv); continue
                out[k] = int(sv); continue
            except ValueError:
                pass
        out[k] = v
    return out


def _flatten_with_structure(inputs):
    """Flatten possibly-nested input list. Returns (flat_tensors, structure).

    structure is per-input: 'T' for single tensor, n>=1 for list-of-n.
    （与 bench_baseline.py 的 _flatten_with_structure 一致）
    """
    flat = []
    structure = []
    for entry in inputs:
        if isinstance(entry, list):
            structure.append(len(entry))
            flat.extend(entry)
        else:
            structure.append("T")
            flat.append(entry)
    return flat, structure


class GenericRefModule:
    """Wrapper so OpRunner.run() can call ref_fn(*flat_args).

    ref_fn signature: (inputs, attrs) -> outputs
    OpRunner.run() signature: func(**params) where params is a dict

    We flatten inputs into positional args, then reconstruct inside forward().
    """

    def __init__(self, ref_fn, attrs, structure):
        self.ref_fn = ref_fn
        self.attrs = attrs
        self.structure = structure

    def __call__(self, **kwargs):
        """OpRunner.run() calls func(**params). We receive flat tensors as
        positional-like kwargs. Since OpRunner._update_params maps device_tensors
        to params by position, we need to reconstruct the original input structure.

        Alternative: bypass OpRunner and call ref_fn directly.
        """
        # OpRunner.run() calls func(**updated_params), where updated_params
        # maps tensor indices to param names. But for baseline collection,
        # we don't use ParamBuilder — we use flat tensor args from inputs.py.
        #
        # The simplest approach: skip OpRunner's params mechanism entirely.
        # Instead, we pass flat tensors directly to PerfEvaluator.run_profiled().
        # This requires a different execution path than OpRunner.run().
        raise NotImplementedError(
            "GenericRefModule should not be called via OpRunner params. "
            "Use BaselineCollector._run_with_profiler() instead."
        )


# ---------------------------------------------------------------------------
# BaselineCollector — 主采集逻辑
# ---------------------------------------------------------------------------

class BaselineCollector:
    """Baseline 性能采集器

    复用评测体系的 PerfEvaluator + KernelDetailsStrategy，
    复用 refs 体系的 ref_fn + inputs.py，
    产出 metadata/<hardware>.json。
    """

    def __init__(self, config: Config, bench_root: Path,
                 warmup: int = 5, repeat: int = 20):
        self.config = config
        self.bench_root = Path(bench_root)
        self.warmup = warmup
        self.repeat = repeat

        # 初始化设备管理器
        self.device_manager = DeviceManager(DeviceConfig(
            type=config.device_type,
            device_id=config.device_id,
        ))

        # 自动检测当前硬件
        device_name = self.device_manager.get_device_name()
        self.hardware = resolve_hardware(device_name) if device_name != "unknown" else DEFAULT_HARDWARE

        # 初始化性能评测器
        self.perf_evaluator = PerfEvaluator(
            config=config,
            device_manager=self.device_manager,
            warmup=warmup,
            repeat=repeat,
            archive_prof=True,
        )

        # 初始化数据加载器
        self.case_loader = get_case_loader("cann", tasks_root=str(bench_root))
        self.task_loader = get_task_loader("cann", tasks_root=str(bench_root))

        # 初始化 BaselineStore（读取已有 t_hw_us）
        project_root = get_project_root()
        self.baseline_store = BaselineStore(
            bench_root=self.bench_root,
            project_root=project_root,
            hardware=self.hardware,
        )
        self.baseline_store.load()

        self._results: Dict[str, Dict] = {}  # op_path -> case_id -> result

    # === Case 发现 ===

    def discover_ops(self, op_filter=None, level_filter=None) -> List[str]:
        """发现需要采集的算子列表（返回 op_path 列表）"""
        # 用 CannTaskLoader 扫描所有算子
        all_tasks = self.task_loader.list_tasks()

        # 获取有 ref 注册的 op_path
        registered = set(ref_registry.all_keys())

        # 构建 op_path 列表
        ops = []
        for task in all_tasks:
            op_path = task.rel_path
            if op_path not in registered:
                continue  # 无 ref，跳过
            if level_filter and not op_path.startswith(f"level{level_filter}/"):
                continue
            if op_filter and op_path != op_filter and task.name != op_filter:
                continue
            ops.append(op_path)

        return sorted(ops)

    # === 单 case 采集 ===

    def collect_one_case(self, op_path: str, case_id: int,
                         case_raw: Dict) -> Optional[Dict]:
        """采集单个 case 的 baseline 性能数据

        Args:
            op_path: 算子路径，如 "level2/cummin"
            case_id: case ID
            case_raw: cases.yaml 中的单个 case dict
        """
        case_id_str = f"{op_path}_{case_id}"
        attrs = _normalize_attrs(case_raw.get("attrs") or {})

        # 1. 获取 ref 函数
        ref_fn = ref_registry.get_ref(op_path)
        if ref_fn is None:
            logger.warning("SKIP: no ref registered for %s", op_path)
            return {"op_path": op_path, "case_id": case_id,
                    "skipped": True, "error": "no_ref"}

        # 2. 构建输入（使用 inputs.py）
        try:
            inputs_cpu = bench_inputs.build_inputs(
                case_raw["input_shape"],
                case_raw["dtype"],
                case_raw.get("value_range"),
                case_id,
                op_key=op_path,
            )
        except Exception as e:
            logger.warning("input build failed for %s case %d: %s", op_path, case_id, e)
            return {"op_path": op_path, "case_id": case_id,
                    "elapsed_us": None, "error_msg": f"input_build_FAIL: {e}"}

        # 3. To device + NPU 别名处理
        try:
            device = self.device_manager.get_device()
            inputs_npu = bench_inputs.to_device(inputs_cpu, device)
            inputs_npu = bench_inputs.apply_npu_op_aliases(inputs_npu, op_path, attrs)
        except Exception as e:
            logger.warning("to_device/aliases failed for %s case %d: %s", op_path, case_id, e)
            return {"op_path": op_path, "case_id": case_id,
                    "elapsed_us": None, "error_msg": f"device_FAIL: {e}"}

        # 4. Flatten inputs for profiling
        flat, structure = _flatten_with_structure(inputs_npu)

        # 5. 构建 profiling wrapper: ref_fn(inputs, attrs)
        #    PerfEvaluator.run_profiled(case_id, func, *flat_args)
        #    func is called as func(*flat_args) during profiling.
        #    We need to reconstruct inputs from flat_args before calling ref_fn.

        def _ref_wrapper(*flat_args):
            """Reconstruct inputs from flat_args and call ref_fn(inputs, attrs)"""
            reconstructed = []
            i = 0
            for s in structure:
                if s == "T":
                    reconstructed.append(flat_args[i])
                    i += 1
                else:
                    reconstructed.append(list(flat_args[i:i + s]))
                    i += s
            result = ref_fn(reconstructed, attrs)
            # ref_fn may return None (unsupported dtype/shape)
            return result

        # 6. Run profiler
        #    PerfEvaluator.run_profiled(self, case_id, func, *args) calls func(*args).
        #    Our _ref_wrapper(*flat_args) reconstructs inputs from flat_args.
        #    Must pass case_id and func as positional args (not keyword),
        #    because *flat expands into positional args after them.
        try:
            _, perf_result = self.perf_evaluator.run_profiled(
                case_id_str, _ref_wrapper, *flat
            )
        except Exception as e:
            logger.warning("profiler failed for %s case %d: %s", op_path, case_id, e)
            return {"op_path": op_path, "case_id": case_id,
                    "elapsed_us": None, "error_msg": f"profiler_FAIL: {e}"}

        # 7. Check result
        if perf_result.error_msg:
            logger.warning("perf_result has error for %s case %d: %s",
                           op_path, case_id, perf_result.error_msg)
            return {"op_path": op_path, "case_id": case_id,
                    "elapsed_us": None, "error_msg": perf_result.error_msg}

        if perf_result.elapsed_us <= 0:
            logger.warning("elapsed_us=0 for %s case %d", op_path, case_id)
            return {"op_path": op_path, "case_id": case_id,
                    "elapsed_us": None, "error_msg": "elapsed_us=0"}

        # 8. 读取 t_hw_us from BaselineStore
        t_hw_us = self.baseline_store.get_t_hw(op_path, case_id)

        # 9. 汇总结果
        result = {
            "op_path": op_path,
            "case_id": case_id,
            "elapsed_us": perf_result.elapsed_us,
            "baseline_perf_us": perf_result.elapsed_us,
            "t_hw_us": t_hw_us,
            "aicore_e2e": perf_result.metadata.get("aicore_e2e"),
            "aicpukernel_gap": perf_result.metadata.get("aicpukernel_gap"),
            "aicore_e2e_jitter": perf_result.metadata.get("aicore_e2e_jitter"),
            "device_kernels": perf_result.op_times.get("device_kernels", {}),
            "data_source": perf_result.metadata.get("data_source"),
            "kernels_per_call": self._count_kernels(perf_result),
            "error_msg": None,
        }

        self._results.setdefault(op_path, {})[case_id] = result
        return result

    def _count_kernels(self, perf_result) -> Optional[int]:
        """从 PerfResult 中提取每 call 的 kernel 数"""
        device_kernels = perf_result.op_times.get("device_kernels", {})
        if device_kernels:
            return len(device_kernels)
        return None

    # === 批量采集 ===

    def collect_op(self, op_path: str, cases_filter: Set[int] = None) -> List[Dict]:
        """采集单个算子的所有 case"""
        # 加载 cases.yaml
        cases_yaml = self.bench_root / op_path / "cases.yaml"
        if not cases_yaml.exists():
            logger.warning("cases.yaml not found: %s", cases_yaml)
            return []

        with open(cases_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "cases" not in data:
            logger.warning("empty cases.yaml: %s", cases_yaml)
            return []

        cases = data["cases"]
        if cases_filter:
            cases = [c for c in cases if int(c["case_id"]) in cases_filter]

        # 检查是否有 ref
        ref_fn = ref_registry.get_ref(op_path)
        if ref_fn is None:
            print(f"[SKIP] {op_path}: no ref registered")
            return []

        print(f"\n=== {op_path} ({len(cases)} cases, device {self.device_manager.get_device()}) ===")

        results = []
        for i, case in enumerate(cases, 1):
            cid = int(case["case_id"])
            print(f"[{i}/{len(cases)}] {op_path}_{cid} ...", end=" ", flush=True)

            result = self.collect_one_case(op_path, cid, case)
            results.append(result)

            if result and result.get("elapsed_us") is not None:
                elapsed = result["elapsed_us"]
                k_per_call = result.get("kernels_per_call")
                kernels_str = ""
                dk = result.get("device_kernels", {})
                if dk:
                    kernels_str = " + ".join(
                        f"{k}×{Counter([k]).get(k,1)}"
                        for k in dk.keys()
                    )
                print(f"✅ {elapsed:.2f}μs (k={k_per_call}, {kernels_str})")
            elif result and result.get("skipped"):
                print(f"⏭️ no ref")
            else:
                err = result.get("error_msg", "unknown") if result else "unknown"
                print(f"❌ {err}")

            # 清理内存
            try:
                import torch_npu
                torch_npu.npu.empty_cache()
            except Exception:
                pass
            gc.collect()

        return results

    # === 输出 ===

    def write_metadata_json(self, output_path: Path) -> Path:
        """将采集结果写入 metadata JSON（与 BaselineStore 兼容）"""
        # 构建嵌套结构: level -> op -> case_id -> {baseline_perf_us, t_hw_us}
        metadata = {
            "_metadata": {
                "description": "CANN baseline 性能数据（collect_baseline.py 采集）",
                "hardware": self.hardware,
                "generated_at": datetime.now().isoformat(),
                "source": "collect_baseline.py (PerfEvaluator + KernelDetailsStrategy + refs)",
                "warmup": self.warmup,
                "repeat": self.repeat,
                "profiler_level": self.config.profiler_level,
                "input_builder": "inputs.py (seed=0xC0FFEE+case_id*31337)",
                "collection_method": "ref_func_profiling",
            }
        }

        total_collected = 0
        for op_path, cases_dict in self._results.items():
            for case_id, result in cases_dict.items():
                if result.get("elapsed_us") is None:
                    continue

                # Parse op_path: "level2/cummin" -> level="level2", op="cummin"
                parts = op_path.split("/")
                if len(parts) == 2 and parts[0].startswith("level"):
                    level_key, op_key = parts
                else:
                    # No level prefix (bench_lab ops etc.)
                    level_key, op_key = "", op_path

                # Navigate/create nested structure
                if level_key:
                    level_data = metadata.setdefault(level_key, {})
                    op_data = level_data.setdefault(op_key, {})
                else:
                    op_data = metadata.setdefault(op_key, {})

                # Build case entry
                case_entry = {
                    "baseline_perf_us": result["baseline_perf_us"],
                    "t_hw_us": result.get("t_hw_us", 0.0),
                }

                # If t_hw_us is 0 and we have it from existing metadata, use that
                if case_entry["t_hw_us"] == 0.0:
                    existing_t_hw = self.baseline_store.get_t_hw(op_path, case_id)
                    if existing_t_hw > 0:
                        case_entry["t_hw_us"] = existing_t_hw

                op_data[str(case_id)] = case_entry
                total_collected += 1

        # Merge with existing metadata JSON (preserve existing entries)
        existing_data = {}
        if output_path.exists():
            try:
                with open(output_path, encoding="utf-8") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load existing metadata: %s", e)

        # Deep merge: new data overrides existing, but preserves entries not in new data
        merged = _deep_merge(existing_data, metadata)

        # Write
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

        print(f"\n[INFO] 写入 metadata: {output_path}")
        print(f"[INFO] 本次采集: {total_collected} case(s)")
        print(f"[INFO] 文件大小: {output_path.stat().st_size / 1024:.1f} KB")

        return output_path

    def print_summary(self):
        """打印采集摘要"""
        print("\n" + "=" * 80)
        print("COLLECTION SUMMARY")
        print("=" * 80)

        for op_path, cases_dict in sorted(self._results.items()):
            collected = [r for r in cases_dict.values() if r.get("elapsed_us") is not None]
            skipped = [r for r in cases_dict.values() if r.get("skipped") or r.get("elapsed_us") is None]
            if collected:
                elapsed_list = [r["elapsed_us"] for r in collected]
                mean_us = sum(elapsed_list) / len(elapsed_list)
                print(f"  {op_path}: collected={len(collected)}, skipped={len(skipped)}, "
                      f"mean={mean_us:.2f}μs")
            else:
                print(f"  {op_path}: all skipped/failed ({len(skipped)} cases)")


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    """Deep merge overlay into base. overlay values take precedence."""
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            if key == "_metadata":
                # metadata: overlay replaces entirely
                result[key] = value
            else:
                result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _check_npu_available():
    """检查 NPU 环境是否可用"""
    import torch
    try:
        import torch_npu
        if torch.npu.is_available():
            # ACL 模式：单算子 benchmark 不需要 Ge 图优化，
            # ACL 逐算子下发确保 profiler 产出完整 kernel_details.csv
            os.environ["ASCEND_LAUNCH_MODE"] = "ACL"
            return True
        print("[ERROR] torch_npu 已导入但 NPU 不可用")
        return False
    except ImportError:
        print("[ERROR] torch_npu 未安装 — baseline 性能采集需要 NPU 环境")
        return False


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Baseline 性能采集脚本\n\n复用评测体系的 PerfEvaluator + refs 的 NPU 参考算子",
    )
    parser.add_argument("--op", dest="op_path", default=None,
                        help="算子路径 (如 level1/exp) 或算子名 (如 Sigmoid)")
    parser.add_argument("--level", type=int, default=None,
                        help="采集指定级别的所有算子 (1/2/3/4)")
    parser.add_argument("--all", action="store_true",
                        help="采集所有级别")
    parser.add_argument("--cases", default="",
                        help="逗号分隔的 case_id（默认全部）")
    parser.add_argument("--device-id", type=int, default=0,
                        help="NPU 设备 ID (默认 0)")
    parser.add_argument("--bench-root", default=None,
                        help="评测集根目录 (默认 tasks/)")
    parser.add_argument("--output", default=None,
                        help="输出 JSON 文件路径 (默认 scripts/baseline/output/<hardware>.json)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="预热次数 (默认 5)")
    parser.add_argument("--repeat", type=int, default=20,
                        help="采集次数 (默认 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印计划，不执行")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有 metadata 的 case")
    parser.add_argument("--force-recollect", action="store_true",
                        help="强制重新采集所有 case")
    args = parser.parse_args()

    if not args.op_path and not args.level and not args.all:
        parser.error("指定 op_path (如 level1/exp) 或 --level N 或 --all")

    # 检查 NPU 环境
    if not args.dry_run and not _check_npu_available():
        sys.exit(1)

    # 设置 bench_root
    bench_root = Path(args.bench_root) if args.bench_root else PROJECT_ROOT / "tasks"

    # 解析 cases filter
    cases_filter = None
    if args.cases:
        cases_filter = set(int(x.strip()) for x in args.cases.split(",") if x.strip())

    # 设置 Config（基准采集使用 MsProfSummaryStrategy + ACL 模式）
    config = Config(
        device_type="npu",
        device_id=args.device_id,
        enable_profiler=True,
        profiler_level="Level1",
        warmup=args.warmup,
        repeat=args.repeat,
        tasks_root=str(bench_root),
        reports_dir=str(PROJECT_ROOT / "reports"),
        torch_op_guard_mode="off",
        enable_accuracy_retry=False,
        eval_seed=0,
        # 基准采集专用配置
        enable_acl_launch_mode=True,
        enable_msprof_export=True,
        perf_metric_strategy_override="msprof_summary",
    )

    # 创建采集器
    collector = BaselineCollector(
        config=config,
        bench_root=bench_root,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    # 发现算子
    ops = collector.discover_ops(
        op_filter=args.op_path,
        level_filter=args.level,
    )

    if not ops:
        print("[WARN] 没有找到可采集的算子（检查 op_path 或 ref 注册）")
        sys.exit(0)

    print(f"[INFO] 计划采集 {len(ops)} 个算子")

    # dry-run: 只打印计划
    if args.dry_run:
        print("\nDry-run 计划:")
        for op_path in ops:
            cases_yaml = bench_root / op_path / "cases.yaml"
            if cases_yaml.exists():
                with open(cases_yaml, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                n_cases = len(data.get("cases", [])) if data else 0
                print(f"  {op_path}: {n_cases} cases")
            else:
                print(f"  {op_path}: cases.yaml 不存在")
        print(f"\n[INFO] warmup={args.warmup}, repeat={args.repeat}, device_id={args.device_id}")
        sys.exit(0)

    # 执行采集
    for op_path in ops:
        collector.collect_op(op_path, cases_filter=cases_filter)

    # 输出 metadata JSON（写入独立目录，不污染 tasks/metadata/ 的 BaselineStore 数据）
    hardware = collector.hardware
    if args.output:
        output_path = Path(args.output)
    else:
        # 默认输出到 scripts/baseline/output/<hardware>.json
        baseline_output_dir = PROJECT_ROOT / "scripts" / "baseline" / "output"
        output_path = baseline_output_dir / f"{hardware}.json"
    collector.write_metadata_json(output_path)

    # 打印摘要
    collector.print_summary()


if __name__ == "__main__":
    main()