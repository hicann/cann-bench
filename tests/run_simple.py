#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
测试执行脚本

职责：
1. --cpu 模式：在 CPU 上执行 golden，验证算子定义正确性
2. --npu 模式：在 NPU 上执行模拟评测（golden 伪装成 AI 算子），验证 NPU 实现正确性并采集性能

使用方式:
    python run_simple.py --cpu --operator Sigmoid          # CPU 简单验证
    python run_simple.py --npu --operator Scatter          # NPU 模拟评测
    python run_simple.py --npu --task-dir kernel_bench/level2/scatter  # 指定算子目录
"""

import os
import sys
import argparse
import warnings

# 禁用 torch backend 自动加载，避免 torch_npu 导入失败
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')

# 默认静默模式，抑制第三方库日志和警告
import logging
logging.getLogger().setLevel(logging.WARNING)
for name in ['torch', 'torch_npu', 'ascend']:
    logging.getLogger(name).setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)

# 解析参数以提前检测 --verbose
args = argparse.ArgumentParser(add_help=False)
args.add_argument('-v', '--verbose', action='store_true')
known_args, _ = args.parse_known_args()

# 详细模式下恢复日志级别
if known_args.verbose:
    logging.getLogger().setLevel(logging.INFO)
    for name in ['torch', 'torch_npu', 'ascend']:
        logging.getLogger(name).setLevel(logging.INFO)
    warnings.filterwarnings('default', category=UserWarning)

import json
from pathlib import Path
from datetime import datetime

import torch

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.kernel_eval.data.case_loader import CaseLoader
from src.kernel_eval.eval.evaluator import EvalOperatorResult
from src.kernel_eval.eval.process_pool import ProcessPoolCoordinator, ProcessConfig
from src.kernel_eval.config import get_config, set_config, Config
from src.kernel_eval.data.golden_packager import GoldenPackager
from src.kernel_eval.utils.path_resolver import resolve_task_dir


def ensure_golden_package_installed(bench_root: str, verbose: bool = False) -> bool:
    """确保 cann_bench_golden 已安装

    Args:
        bench_root: bench 目录路径
        verbose: 是否显示详细输出

    Returns:
        是否成功安装/已存在
    """
    try:
        import cann_bench_golden
        if verbose:
            print(f"[INFO] cann_bench_golden 已安装")
        return True
    except ImportError:
        pass

    # 未安装，自动打包
    print("[INFO] cann_bench_golden 未安装，正在自动打包...")
    import subprocess
    import tempfile

    output_dir = Path(tempfile.mkdtemp(prefix="golden_whl_"))
    try:
        packager = GoldenPackager(bench_root, str(output_dir))
        whl_path = packager.package(clean_up=True)

        # 使用 --no-deps 安装，避免触发 torch 依赖安装
        result = subprocess.run(
            ["pip", "install", str(whl_path), "--no-deps", "--force-reinstall"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"[ERROR] 安装失败: {result.stderr}")
            return False

        print(f"[INFO] cann_bench_golden 安装成功")

        # 清理 whl 文件
        whl_path.unlink(missing_ok=True)
        output_dir.rmdir()

        return True
    except Exception as e:
        print(f"[ERROR] 打包安装失败: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='测试执行脚本')

    # 设备选项（互斥）
    parser.add_argument('--cpu', action='store_true', help='CPU 简单验证模式')
    parser.add_argument('--npu', action='store_true', help='NPU 模拟评测模式')
    parser.add_argument('--device-id', type=int, default=None,
                        help='指定 NPU 设备 ID（单卡模式），不指定则自动使用全部可用卡（多卡并行模式）')

    # 多进程并行配置
    parser.add_argument('--processes-per-card', type=int, default=2,
                        help='每卡进程数（默认: 2）')
    parser.add_argument('--timeout-per-operator', type=int, default=300,
                        help='单算子超时时间（秒，默认: 300）。进程总超时 = 算子数 × timeout_per_operator')

    # 目录配置（替代 --level）
    parser.add_argument('--task-dir', type=str, default=None,
                        help='指定评测目录（bench根目录或算子目录，默认: kernel_bench）')

    # 用例筛选
    parser.add_argument('--operator', type=str, default=None,
                        help='按算子名称筛选')
    parser.add_argument('--case-id', type=int, default=None,
                        help='按用例编号筛选')

    # 性能配置（仅 NPU 模式）
    parser.add_argument('--warmup', type=int, default=3, help='预热次数')
    parser.add_argument('--repeat', type=int, default=5, help='采集次数')
    parser.add_argument('--timeout', type=int, default=240,
                        help='算子级子进程超时时间（秒）')

    # 输出选项
    parser.add_argument('--output', type=str, default='reports/test_results.json')
    parser.add_argument('--export-baseline', type=str, default=None,
                        help='导出性能基线到 JSON（仅 NPU 模式有效）')
    parser.add_argument('--no-perf', action='store_true', help='关闭性能采集，仅做精度验证')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出模式')

    return parser.parse_args()


def filter_cases(cases: list, operator=None, case_id=None) -> list:
    """筛选用例"""
    result = cases
    if operator:
        result = [c for c in result if c.operator.lower() == operator.lower()]
    if case_id:
        result = [c for c in result if c.case_id == case_id]
    return result


def run_cpu_mode(args) -> dict:
    """CPU 模式：验证 golden 可执行

    组件一次初始化，复用 GoldenLoader（含模块导入缓存）、DataGenerator、
    ParamBuilder 和 OpRunner，避免每个 case 重复创建。
    """
    print("\n" + "=" * 60)
    print("CPU 验证模式")
    print("=" * 60)

    # 解析目录参数
    bench_root, filter_prefix = resolve_task_dir(args.task_dir, project_root)
    print(f"[INFO] Bench目录: {bench_root}")
    if filter_prefix:
        print(f"[INFO] 筛选路径: {filter_prefix}")

    config = Config()
    config.device_type = "cpu"
    config.enable_profiler = False
    config.kernel_bench_root = bench_root
    set_config(config)

    # 一次初始化，所有 case 复用
    from src.kernel_eval.data.golden_loader import GoldenLoader
    from src.kernel_eval.data.data_generator import DataGenerator
    from src.kernel_eval.utils.param_builder import ParamBuilder
    from src.kernel_eval.utils.device_manager import DeviceManager, DeviceConfig
    from src.kernel_eval.eval.op_runner import OpRunner

    golden_loader = GoldenLoader(bench_root)
    data_generator = DataGenerator()
    param_builder = ParamBuilder(golden_loader)
    device_mgr = DeviceManager(DeviceConfig(type="cpu", device_id=0, auto_fallback=True))
    runner = OpRunner(device_mgr, None)

    loader = CaseLoader(bench_root)
    all_cases = loader.scan_all_cases()

    # 如果指定了筛选前缀，筛选该前缀下的用例
    if filter_prefix:
        all_cases = [c for c in all_cases if c.rel_path.startswith(filter_prefix + '/') or c.rel_path == filter_prefix]

    cases = filter_cases(all_cases, args.operator, args.case_id)

    if not cases:
        print("[WARN] 无匹配用例")
        return {"total": 0, "passed": 0, "failed": 0}

    print(f"[INFO] 用例数: {len(cases)}")

    passed = failed = skipped = 0

    for i, case in enumerate(cases, 1):
        case_id_str = case.get_case_id_str()
        print(f"\n[{i}/{len(cases)}] {case_id_str}")

        try:
            golden_func = golden_loader.get_golden_function(case.rel_path)

            input_tensors = data_generator.generate_input_tensors_from_case(
                input_shapes=case.input_shapes,
                dtypes=case.dtypes,
                value_ranges=case.value_ranges,
            )

            get_input_func = golden_loader.get_input_function(case.rel_path)
            if get_input_func is not None:
                # proto.yaml inputs 顺序已与 schema 一致，直接用 param_builder 构建
                params_for_get_input = param_builder.build_call_params(
                    get_input_func, case, input_tensors)
                case_attrs = getattr(case, 'attrs', None) or {}
                for attr_key, attr_val in case_attrs.items():
                    if attr_key not in params_for_get_input:
                        params_for_get_input[attr_key] = attr_val
                if 'skip2_exist' not in params_for_get_input:
                    params_for_get_input['skip2_exist'] = case_attrs.get('skip2_exist', True)
                input_tensors = get_input_func(**params_for_get_input)
                if isinstance(input_tensors, tuple):
                    input_tensors = list(input_tensors)

            # 构建 golden 函数参数
            params = param_builder.build_call_params(golden_func, case, input_tensors)
            result = runner.run_golden(golden_func, params, case_id_str, input_tensors)

            if result.success:
                print(f"[PASS] {case_id_str} - {result.elapsed_us / 1000:.2f}ms")
                passed += 1
            else:
                print(f"[FAIL] {case_id_str}: {result.error}")
                failed += 1

        except Exception as e:
            print(f"[SKIP] {case_id_str}: {e}")
            skipped += 1

    print(f"\n[汇总] 通过: {passed}, 失败: {failed}, 跳过: {skipped}")
    return {"total": len(cases), "passed": passed, "failed": failed, "skipped": skipped}


def run_npu_mode(args) -> EvalOperatorResult:
    """NPU 模式：统一使用进程池架构

    支持单卡和多卡：
    - 指定 --device-id 时：单卡模式，所有进程绑定到该卡
    - 未指定 --device-id 时：多卡模式，自动检测并分配
    """
    import time

    # 解析目录参数
    bench_root, target_rel_path = resolve_task_dir(args.task_dir, project_root)

    # 配置
    config = get_config()
    config.device_type = "npu"
    config.auto_fallback = False
    config.enable_profiler = not args.no_perf
    config.warmup = args.warmup
    config.repeat = args.repeat
    config.reports_dir = str(Path(args.output).parent)
    config.processes_per_card = args.processes_per_card
    config.kernel_bench_root = bench_root
    set_config(config)

    # 确保 cann_bench_golden 已安装
    if not ensure_golden_package_installed(bench_root, args.verbose):
        print("[ERROR] 无法安装 cann_bench_golden，退出")
        return None

    # 加载用例
    loader = CaseLoader(bench_root)
    all_cases = loader.scan_all_cases()

    # 如果指定了筛选前缀，筛选该前缀下的用例
    if target_rel_path:
        # 前缀匹配：target_rel_path="level1" 匹配 "level1/exp", "level1/sigmoid" 等
        all_cases = [
            c for c in all_cases
            if c.rel_path.startswith(target_rel_path + '/') or c.rel_path == target_rel_path
        ]

    cases = filter_cases(all_cases, args.operator, args.case_id)

    if not cases:
        print("[WARN] 无匹配用例")
        return None

    # 获取唯一的 rel_paths（算子目录）
    rel_paths = set(c.rel_path for c in cases)
    print(f"[INFO] Bench目录: {bench_root}")
    if target_rel_path:
        print(f"[INFO] 筛选路径: {target_rel_path}")
    print(f"[INFO] 算子数: {len(rel_paths)}, 用例数: {len(cases)}")

    # 打印模式信息
    print("\n" + "=" * 60)
    if args.device_id is not None:
        print(f"NPU 单卡进程池模式 (NPU:{args.device_id})")
    else:
        print("NPU 多卡进程池模式")
    print("=" * 60)
    print(f"[CONFIG] 每卡进程数: {args.processes_per_card}")
    print(f"[CONFIG] 单算子超时: {args.timeout_per_operator}s")
    print(f"[CONFIG] Warmup/Repeat: {args.warmup}/{args.repeat}")
    if args.no_perf:
        print("[CONFIG] 性能采集: 关闭（仅精度验证）")

    # 创建进程池配置
    process_config = ProcessConfig(
        processes_per_card=args.processes_per_card,
        timeout_per_operator=args.timeout_per_operator,
        enable_profiler=not args.no_perf,
    )

    # 创建进程池协调器
    coordinator = ProcessPoolCoordinator(
        base_config=config,
        process_config=process_config,
        device_id=args.device_id,
    )

    if coordinator.card_count == 0:
        print("[ERROR] 无可用 NPU 卡")
        return None

    print(f"[INFO] 使用 {coordinator.total_processes} 个进程池并行")

    start_time = time.time()

    # 执行评测
    all_results = coordinator.evaluate_operators(
        rel_paths=list(rel_paths),
    )

    total_time = time.time() - start_time

    # 汇总
    _print_summary(all_results)
    print(f"\n[效率] 总耗时: {total_time:.2f}s, 平均: {total_time / len(cases):.2f}s/case")

    # 导出基线
    perf_data = {}
    for op_result in all_results:
        for result in op_result.results:
            if result.success and result.perf_result:
                perf_data[result.case_id] = {
                    "rel_path": result.rel_path,
                    "operator": result.operator,
                    "case_id": result.case_num,
                    "elapsed_us": result.perf_result.elapsed_us,
                    "timestamp": datetime.now().isoformat()
                }

    total_cases = sum(r.total_cases for r in all_results)
    total_passed = sum(r.passed_cases for r in all_results)
    if args.export_baseline and perf_data:
        export_baseline(args.export_baseline, perf_data, total_cases, args.verbose)

    _save_results(args.output, all_results, total_cases, total_passed, args.verbose)

    coordinator.shutdown()

    return all_results[0] if all_results else None


def _print_summary(all_results):
    """打印汇总信息"""
    total_passed = sum(r.passed_cases for r in all_results)
    total_cases = sum(r.total_cases for r in all_results)
    overall_rate = total_passed / total_cases if total_cases > 0 else 0

    print(f"\n{'=' * 60}")
    print("汇总")
    print(f"{'=' * 60}")
    print(f"总用例: {total_cases}")
    print(f"通过: {total_passed}")
    print(f"失败: {total_cases - total_passed}")
    print(f"整体通过率: {overall_rate * 100:.1f}%")

    if overall_rate == 1.0:
        print("\n[✓] Golden NPU 实现与 CPU 参考一致，验证通过！")
    else:
        print("\n[✗] 存在失败的用例，Golden NPU 实现可能有问题。")


def export_baseline(output_path: str, perf_data: dict, total_cases: int, verbose: bool = False):
    """导出性能基线"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_cases": total_cases,
                "collected": len(perf_data),
                "timestamp": datetime.now().isoformat()
            },
            "baselines": perf_data
        }, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n[INFO] 性能基线已导出到: {output}")


def _save_results(output_path: str, results: list, total: int, passed: int, verbose: bool = False):
    """保存详细结果"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total_cases": total,
                "passed_cases": passed,
                "failed_cases": total - passed,
                "pass_rate": passed / total if total > 0 else 0,
                "timestamp": datetime.now().isoformat()
            },
            "operators": [r.to_dict() for r in results]
        }, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"[INFO] 详细结果已保存到: {output}")


def main():
    args = parse_args()

    try:
        if args.cpu:
            result = run_cpu_mode(args)
            if result.get('failed', 0) > 0:
                sys.exit(1)
        elif args.npu:
            result = run_npu_mode(args)
            if result is None or (hasattr(result, 'pass_rate') and result.pass_rate < 1.0):
                sys.exit(1)
        else:
            print("[ERROR] 请指定 --cpu 或 --npu 模式")
            print("  --cpu: CPU 简单验证")
            print("  --npu: NPU 模拟评测")
            sys.exit(1)

        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断执行")
        sys.exit(130)  # 128 + SIGINT(2) = 130


if __name__ == "__main__":
    main()