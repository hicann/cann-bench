#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
简化测试执行脚本

直接遍历执行测试用例，避免 pytest 的参数化收集开销
"""

import sys
import time
import argparse
import gc
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.core.case_loader import CaseLoader, CaseInfo
from tests.core.device_runner import DeviceRunner
from tests.core.result_recorder import ResultRecorder
from tests.utils.device_manager import DeviceManager, DeviceConfig
from tests.utils.golden_importer import GoldenImporter
from tests.utils.param_builder import ParamBuilder
from tests.core.profiler_manager import ProfilerManager


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='简化测试执行脚本')

    # 设备选项
    parser.add_argument('--cpu', action='store_true', help='使用 CPU 设备')
    parser.add_argument('--npu', action='store_true', help='使用 NPU 设备')

    # 筛选选项
    parser.add_argument('--level', type=int, default=None, choices=[1, 2, 3, 4],
                        help='按难度级别筛选')
    parser.add_argument('--operator', type=str, default=None,
                        help='按算子名称筛选（支持模糊匹配）')
    parser.add_argument('--case-id', type=int, default=None,
                        help='按用例编号筛选')

    # 性能采集
    parser.add_argument('--prof', action='store_true', help='启用性能采集')
    parser.add_argument('--warmup', type=int, default=3, help='Profiling 预热次数')
    parser.add_argument('--repeat', type=int, default=5, help='Profiling 采集次数')

    # 输出选项
    parser.add_argument('--output', type=str, default='reports/test_results.json',
                        help='结果输出文件路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')

    return parser.parse_args()


def create_device_manager(device_type: str) -> DeviceManager:
    """创建设备管理器"""
    config = DeviceConfig(
        type=device_type,
        device_id=0,
        auto_fallback=True
    )
    return DeviceManager(config)


def filter_cases(cases: list, level: int = None, operator: str = None,
                 case_id: int = None) -> list:
    """筛选用例"""
    result = cases

    if level:
        result = [c for c in result if c.level == level]

    if operator:
        # 通过 CaseInfo.operator 字段直接匹配，避免目录 snake_case 与 CamelCase 不一致
        result = [c for c in result if operator.lower() == c.operator.lower()]

    if case_id:
        result = [c for c in result if c.case_id == case_id]

    return result


def run_single_case(case: CaseInfo, runner: DeviceRunner, importer: GoldenImporter,
                    result_recorder: ResultRecorder, verbose: bool = False) -> str:
    """执行单个测试用例，返回 status: "passed" / "failed" / "skipped" """
    case_id_str = case.get_case_id_str()

    if verbose:
        print(f"\n执行: {case_id_str}")
        print(f"  算子: {case.operator}")
        print(f"  Shape: {case.input_shapes}")
        print(f"  Dtype: {case.dtypes}")

    # 延迟导入 torch（只在执行时导入）
    from tests.core.data_generator import DataGenerator

    # 1. 获取 golden 函数
    try:
        golden_func = importer.get_golden_function(case.level, case.operator)
    except ImportError as e:
        result_recorder.record_skip(case, f"Golden模块不存在: {e}")
        print(f"[SKIP] {case_id_str}: Golden模块不存在")
        return "skipped"
    except AttributeError as e:
        result_recorder.record_skip(case, f"Golden函数不存在: {e}")
        print(f"[SKIP] {case_id_str}: Golden函数不存在")
        return "skipped"

    # 2. 生成输入数据
    generator = DataGenerator()
    try:
        input_tensors = generator.generate_input_tensors_from_case(
            input_shapes=case.input_shapes,
            dtypes=case.dtypes,
            value_ranges=case.value_ranges
        )
    except Exception as e:
        result_recorder.record_skip(case, f"生成输入数据失败: {e}")
        print(f"[SKIP] {case_id_str}: 生成输入数据失败 - {e}")
        return "skipped"

    # 2.5 调用 get_input 预处理（如果存在）
    try:
        get_input_func = importer.get_input_function(case.level, case.operator)
        if get_input_func is not None:
            # 构建 get_input 的调用参数（与 golden 函数参数一致）
            builder_temp = ParamBuilder(importer)
            get_input_params = builder_temp.build_call_params(golden_func, case, input_tensors)
            input_tensors = get_input_func(**get_input_params)
            # get_input 返回可能是 tuple 或 list，转换为 list
            if isinstance(input_tensors, tuple):
                input_tensors = list(input_tensors)
    except Exception as e:
        result_recorder.record_skip(case, f"get_input预处理失败: {e}")
        print(f"[SKIP] {case_id_str}: get_input预处理失败 - {e}")
        return "skipped"

    # 3. 构建调用参数
    builder = ParamBuilder(importer)
    try:
        params = builder.build_call_params(golden_func, case, input_tensors)
    except Exception as e:
        result_recorder.record_skip(case, f"构建参数失败: {e}")
        print(f"[SKIP] {case_id_str}: 构建参数失败 - {e}")
        return "skipped"

    # 4. 执行测试
    try:
        run_result = runner.run(golden_func, params, case_id_str, input_tensors)
        result_recorder.record(case, run_result, profiler_result=run_result.profiler_result)

        if run_result.success:
            elapsed_ms = run_result.elapsed_us / 1000
            print(f"[PASS] {case_id_str} - {elapsed_ms:.2f}ms")
            return "passed"
        else:
            print(f"[FAIL] {case_id_str}: {run_result.error}")
            if verbose and run_result.traceback:
                print(f"  Traceback:\n{run_result.traceback}")
            return "failed"

    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        result_recorder.record_skip(case, f"执行异常: {e}")
        print(f"[ERROR] {case_id_str}: {e}")
        if verbose:
            print(f"  Traceback:\n{tb_str}")
        return "failed"

    finally:
        # 清理内存
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # NPU 设备清理（torch_npu）
            import torch_npu
            if hasattr(torch_npu, 'npu'):
                torch_npu.npu.empty_cache()
        except Exception:
            pass


def main():
    """主函数"""
    args = parse_args()

    # 确定设备类型
    device_type = "npu" if args.npu else "cpu"

    # 1. 加载用例
    bench_root = project_root / "kernel_bench"
    loader = CaseLoader(str(bench_root))

    start_time = time.time()
    all_cases = loader.scan_all_cases()
    load_time = time.time() - start_time

    # 筛选用例
    cases = filter_cases(all_cases, args.level, args.operator, args.case_id)

    if len(cases) == 0:
        print("[WARN] 无匹配用例，退出")
        return

    # 2. 初始化组件
    device_manager = create_device_manager(device_type)

    # 创建 ProfilerManager
    profiler_manager = ProfilerManager(enabled=args.prof, device_manager=device_manager,
                                       warmup=args.warmup, repeat=args.repeat)

    # 创建执行器
    runner = DeviceRunner(device_manager, profiler_manager)

    # Golden 导入器
    importer = GoldenImporter(str(bench_root))

    # 结果记录器
    result_recorder = ResultRecorder(args.output)

    # 3. 执行测试
    print(f"\n开始执行测试...")
    print(f"[INFO] 加载 {len(all_cases)} 个用例，耗时 {load_time:.2f}s")
    print(f"[INFO] 筛选后 {len(cases)} 个用例")
    print(f"[INFO] 使用{'CPU' if device_type == 'cpu' else 'NPU'}设备")
    run_start_time = time.time()

    passed = 0
    failed = 0
    skipped = 0

    for i, case in enumerate(cases):
        case_id_str = case.get_case_id_str()
        print(f"\n[{i+1}/{len(cases)}] {case_id_str}")

        status = run_single_case(
            case, runner, importer, result_recorder, args.verbose
        )

        if status == "passed":
            passed += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1

    run_elapsed = time.time() - run_start_time

    # 等待所有 profiling 解析完成并清理中间文件
    profiler_manager.wait_all()

    # 4. 输出结果
    result_recorder.save()
    result_recorder.print_summary()

    print(f"\n执行耗时: {run_elapsed:.2f}s")
    print(f"平均每用例: {run_elapsed/len(cases):.3f}s")

    # 返回退出码
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()