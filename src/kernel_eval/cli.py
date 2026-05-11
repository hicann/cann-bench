#!/usr/bin/python3
# coding=utf-8

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
命令行入口

职责：
1. 提供命令行接口
2. 支持eval、list、info等命令
3. 处理命令参数解析

模块名: kernel_eval
命令名: kernel-bench (保持不变，作为CLI入口名称)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from .config import Config, get_config, get_project_root, set_config
from .data.operator_loader import OperatorLoader
from .data.case_loader import CaseLoader
from .eval.evaluator import Evaluator
from .report.report_generator import ReportGenerator
from .utils.path_resolver import resolve_task_dir

# 尝试导入 cann_bench_golden，触发 torch.ops.cann_bench 注册
try:
    import cann_bench_golden
except ImportError:
    pass


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog='kernel-bench',
        description='算子评测工程命令行工具',
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # eval 命令
    eval_parser = subparsers.add_parser('eval', help='执行评测')
    eval_parser.add_argument('--source-dir', type=str, default=None,
                             help='AI生成的算子源码目录（不指定则使用已安装的cann_bench）')
    eval_parser.add_argument('--task-dir', type=str, default=None,
                             help='评测目录（bench根目录或算子目录），替代 --level。'
                                  '支持: kernel_bench, kernel_bench/level1, kernel_bench/level1/exp 等')
    eval_parser.add_argument('--operator', type=str, default=None,
                             help='算子名称（如 Exp, Softmax）')
    eval_parser.add_argument('--level', type=int, default=None, choices=[1, 2, 3, 4],
                             help='难度级别筛选（已废弃，建议使用 --task-dir）')
    eval_parser.add_argument('--case-id', type=int, default=None,
                             help='用例编号筛选')
    eval_parser.add_argument('--device-id', type=int, default=None,
                             help='NPU 设备 ID（单卡模式）。不指定则自动使用全部可用卡（多卡并行）')
    eval_parser.add_argument('--device', type=str, default='npu',
                             choices=['cpu', 'npu'],
                             help='设备类型（默认: npu）')
    eval_parser.add_argument('--processes-per-card', type=int, default=2,
                             help='每卡进程数（多卡并行模式，默认: 2）')
    eval_parser.add_argument('--timeout-per-operator', type=int, default=300,
                             help='单算子超时（秒，默认: 300）。进程总超时 = 算子数 × timeout_per_operator')
    eval_parser.add_argument('--warmup', type=int, default=3,
                             help='预热次数（默认: 3）')
    eval_parser.add_argument('--repeat', type=int, default=5,
                             help='采集次数（默认: 5）')
    eval_parser.add_argument('--reports-dir', type=str, default='reports',
                             help='报告输出目录（默认: reports）')
    eval_parser.add_argument('--output', type=str, default=None,
                             help='报告输出目录')
    eval_parser.add_argument('--eval-code', type=str, default=None,
                             help='评测代号')
    eval_parser.add_argument('-v', '--verbose', action='store_true', help='详细输出')
    eval_parser.add_argument('--no-subprocess-isolation', action='store_true',
                             help='关闭子进程隔离（默认开启）。开启后每个算子在独立 '
                                  '子进程评测，一个 kernel 挂死/崩溃不会污染后面的 '
                                  '算子。关闭可少 ~5s/op 的 fork + import 开销。')
    eval_parser.add_argument('--op-timeout-sec', type=int, default=240,
                             help='子进程隔离下 per-op 超时。超时先 SIGTERM，10s 宽限后 SIGKILL。'
                                  '默认 240 秒。')
    eval_parser.add_argument('--no-iterative-compile', action='store_true',
                             help='关闭迭代隔离编译（默认开启）。开启时 build.sh 失败会自动'
                                  '识别并隔离编译不过的算子到 _quarantine/，剩下的算子继续'
                                  '编译和评测；此开关关闭该逻辑，任何一个算子编译失败整个'
                                  '提交直接判定为失败，用于想要严格"全过或全挂"的场景。')
    eval_parser.add_argument('--no-perf', action='store_true',
                             help='关闭性能采集，仅做精度验证')
    eval_parser.add_argument('--profiler-level', type=str, default='Level1',
                             choices=['Level1', 'Level2'],
                             help='Profiler 级别（默认: Level1）。Level1 产出 47 列 CSV，'
                                  'Level2 增加更详细的 AICPU 采集。')
    # 内部开关：子进程模式下由父进程传入，不要手工设置
    eval_parser.add_argument('--skip-install', action='store_true',
                             help=argparse.SUPPRESS)
    eval_parser.add_argument('--child-json-output', type=str, default=None,
                             help=argparse.SUPPRESS)

    # list 命令
    list_parser = subparsers.add_parser('list', help='列出算子/用例')
    list_parser.add_argument('--level', type=int, default=None, choices=[1, 2, 3, 4],
                             help='按级别筛选')
    list_parser.add_argument('--operator', type=str, default=None,
                             help='按算子筛选')
    list_parser.add_argument('--cases', action='store_true', help='列出用例而非算子')

    # info 命令
    info_parser = subparsers.add_parser('info', help='显示算子详细信息')
    info_parser.add_argument('--operator', type=str, required=True,
                             help='算子名称')
    info_parser.add_argument('--level', type=int, default=None, choices=[1, 2, 3, 4],
                             help='难度级别')

    # config 命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_parser.add_argument('--show', action='store_true', help='显示当前配置')
    config_parser.add_argument('--kernel-bench-root', type=str, default=None,
                               help='设置kernel_bench数据目录')
    config_parser.add_argument('--reports-dir', type=str, default=None,
                               help='设置报告输出目录')

    # eval-process 命令（进程池子进程专用）
    eval_process_parser = subparsers.add_parser('eval-process', help='进程池子进程执行')
    eval_process_parser.add_argument('--process-id', type=int, required=True,
                                      help='进程 ID')
    eval_process_parser.add_argument('--card-id', type=int, required=True,
                                      help='NPU 卡 ID')
    eval_process_parser.add_argument('--output', type=str, required=True,
                                      help='结果输出文件路径')
    eval_process_parser.add_argument('--rel-paths', type=str, default=None,
                                      help='算子相对路径列表（逗号分隔，rel_path_parallel 模式）')
    eval_process_parser.add_argument('--cases-file', type=str, default=None,
                                      help='用例数据文件（case_parallel 模式）')
    eval_process_parser.add_argument('--warmup', type=int, default=3,
                                      help='预热次数')
    eval_process_parser.add_argument('--repeat', type=int, default=5,
                                      help='采集次数')
    eval_process_parser.add_argument('--enable-profiler', action='store_true',
                                      help='启用 profiler 性能采集')

    return parser


def _resolve_operator_info(operator_name: str, config):
    """按算子名称查找 OperatorInfo，返回 None 若未找到"""
    try:
        from .data.operator_loader import OperatorLoader
        loader = OperatorLoader(config.kernel_bench_root)
        return loader.get_operator_by_name(operator_name)
    except Exception:
        return None


def _create_config_from_args(args, bench_root: str) -> Config:
    """从命令行参数创建配置"""
    config = Config()
    config.kernel_bench_root = bench_root

    if getattr(args, 'reports_dir', None):
        config.reports_dir = args.reports_dir
    if args.output:
        config.reports_dir = args.output
    if args.source_dir:
        config.source_dir = args.source_dir

    # 设备配置
    if args.device == 'cpu':
        config.device_type = 'cpu'
        config.device_id = 0
    else:
        config.device_type = 'npu'
        config.device_id = getattr(args, 'device_id', None) or 0

    if hasattr(args, 'warmup'):
        config.warmup = args.warmup
    if hasattr(args, 'repeat'):
        config.repeat = args.repeat
    if getattr(args, 'no_perf', False):
        config.enable_profiler = False
    if hasattr(args, 'profiler_level'):
        config.profiler_level = args.profiler_level

    set_config(config)
    return config


def _cmd_eval_multi_card(args, bench_root: str, filter_prefix: str, config: Config,
                         report_generator: ReportGenerator) -> int:
    """多卡并行模式评测"""
    from .eval.process_pool import ProcessPoolCoordinator, ProcessConfig
    from .data.case_loader import CaseLoader
    import time

    processes_per_card = getattr(args, 'processes_per_card', 2)
    timeout_per_process = getattr(args, 'timeout_per_process', 300)

    loader = CaseLoader(bench_root)
    all_cases = loader.scan_all_cases()

    if filter_prefix:
        all_cases = [
            c for c in all_cases
            if c.rel_path.startswith(filter_prefix + '/') or c.rel_path == filter_prefix
        ]

    if args.operator:
        all_cases = [c for c in all_cases if c.operator.lower() == args.operator.lower()]

    if args.case_id:
        all_cases = [c for c in all_cases if c.case_id == args.case_id]

    if not all_cases:
        print("[WARN] 无匹配用例")
        return 0

    rel_paths = list(set(c.rel_path for c in all_cases))

    print(f"\n[INFO] 多卡并行模式")
    print(f"[INFO] Bench目录: {bench_root}")
    if filter_prefix:
        print(f"[INFO] 筛选路径: {filter_prefix}")
    print(f"[INFO] 算子数: {len(rel_paths)}, 用例数: {len(all_cases)}")
    print(f"[INFO] 每卡进程数: {processes_per_card}")
    print(f"[INFO] 进程超时: {timeout_per_process}s")
    print(f"[INFO] Warmup/Repeat: {args.warmup}/{args.repeat}")
    if args.no_perf:
        print("[INFO] 性能采集: 关闭")

    process_config = ProcessConfig(
        processes_per_card=processes_per_card,
        timeout_per_process=timeout_per_process,
        enable_profiler=not args.no_perf,
    )

    coordinator = ProcessPoolCoordinator(
        base_config=config,
        process_config=process_config,
        device_id=None,
    )

    if coordinator.card_count == 0:
        print("[ERROR] 无可用 NPU 卡")
        return 1

    print(f"[INFO] 使用 {coordinator.total_processes} 个进程并行")

    start_time = time.time()
    all_results = coordinator.evaluate_operators(rel_paths=rel_paths)
    total_time = time.time() - start_time

    for op_result in all_results:
        report_generator.add_operator_result(op_result)

    print(f"\n[效率] 总耗时: {total_time:.2f}s")
    coordinator.shutdown()
    return 0


def _cmd_eval_source(args, config: Config, report_generator: ReportGenerator,
                     operator_filter: list, case_filter: dict,
                     subprocess_isolation: bool, op_timeout_sec: int,
                     case_timeout_sec: int, case_subprocess_isolation: bool,
                     iterative_compile: bool) -> int:
    """从源码目录评测"""
    evaluator = Evaluator(config)

    session_result = evaluator.evaluate_from_source(
        source_dir=args.source_dir,
        operator_filter=operator_filter,
        case_filter=case_filter,
        verbose=args.verbose,
        subprocess_isolation=subprocess_isolation,
        op_timeout_sec=op_timeout_sec,
        case_timeout_sec=case_timeout_sec,
        case_subprocess_isolation=case_subprocess_isolation,
        iterative_compile=iterative_compile,
    )
    for op_result in session_result.operators:
        report_generator.add_operator_result(op_result)
    evaluator.shutdown()
    return 0


def _cmd_eval_skip_install(args, config: Config, report_generator: ReportGenerator,
                           operator_filter: list, case_filter: dict,
                           subprocess_isolation: bool) -> int:
    """跳过安装评测"""
    evaluator = Evaluator(config)

    if args.operator:
        op_info = _resolve_operator_info(args.operator, config)
        result = evaluator.evaluate_operator(
            operator=args.operator,
            rel_path=op_info.rel_path if op_info else args.operator,
            case_filter=case_filter,
        )
        report_generator.add_operator_result(result)
    else:
        session_result = evaluator.evaluate_skip_build(
            operator_filter=operator_filter,
            case_filter=case_filter,
            operator_subprocess_isolation=subprocess_isolation,
        )
        for op_result in session_result.operators:
            report_generator.add_operator_result(op_result)
    evaluator.shutdown()
    return 0


def _cmd_eval_golden(args, config: Config, report_generator: ReportGenerator,
                     operator_filter: list, case_filter: dict) -> int:
    """单卡 Golden 模式评测"""
    evaluator = Evaluator(config)

    if args.operator:
        op_info = _resolve_operator_info(args.operator, config)
        result = evaluator.evaluate_golden_only(
            operator=args.operator,
            rel_path=op_info.rel_path if op_info else args.operator,
            case_filter=case_filter,
        )
        report_generator.add_operator_result(result)
    else:
        session_result = evaluator.evaluate_skip_build(
            operator_filter=operator_filter,
            case_filter=case_filter,
        )
        for op_result in session_result.operators:
            report_generator.add_operator_result(op_result)
    evaluator.shutdown()
    return 0


def cmd_eval(args):
    """执行评测命令"""
    project_root = get_project_root()

    # 解析 --task-dir 参数
    bench_root, filter_prefix = resolve_task_dir(args.task_dir, project_root)

    # 创建配置
    config = _create_config_from_args(args, bench_root)

    # 初始化报告生成器
    report_generator = ReportGenerator(
        output_dir=config.reports_dir,
        eval_code=args.eval_code,
        config=config,
    )

    # 构建筛选条件
    operator_filter = [args.operator] if args.operator else None
    case_filter = {'case_id': args.case_id} if args.case_id else None

    # 提取运行参数
    subprocess_isolation = not getattr(args, 'no_subprocess_isolation', False)
    op_timeout_sec = getattr(args, 'op_timeout_sec', 240)
    case_timeout_sec = getattr(args, 'case_timeout_sec', None)
    case_subprocess_isolation = True
    skip_install = getattr(args, 'skip_install', False)
    child_json_output = getattr(args, 'child_json_output', None)
    iterative_compile = not getattr(args, 'no_iterative_compile', False)

    # 判断是否使用多卡并行模式
    use_multi_card = (
        args.device == 'npu'
        and getattr(args, 'device_id', None) is None
        and not args.source_dir
    )

    # 执行对应模式的评测
    if use_multi_card:
        ret = _cmd_eval_multi_card(args, bench_root, filter_prefix, config, report_generator)
    elif args.source_dir and not skip_install:
        ret = _cmd_eval_source(args, config, report_generator, operator_filter, case_filter,
                               subprocess_isolation, op_timeout_sec, case_timeout_sec,
                               case_subprocess_isolation, iterative_compile)
    elif args.source_dir and skip_install:
        ret = _cmd_eval_skip_install(args, config, report_generator, operator_filter,
                                     case_filter, subprocess_isolation)
    else:
        ret = _cmd_eval_golden(args, config, report_generator, operator_filter, case_filter)

    # 生成报告
    report = report_generator.generate()

    # 子进程模式：写入 JSON 文件，不走常规 save_all
    if child_json_output:
        payload = {"operators": [op.to_dict() for op in report_generator.operator_results]}
        Path(child_json_output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    report_generator.save_all(report)
    report_generator.print_summary(report)
    return 0 if report.passed_cases > 0 else 1


def cmd_list(args):
    """列出算子/用例"""
    config = get_config()

    if args.cases:
        # 列出用例
        case_loader = CaseLoader(config.kernel_bench_root)

        if args.operator:
            cases = case_loader.scan_by_operator(args.operator)
        else:
            cases = case_loader.scan_all_cases()

        print(f"\n共 {len(cases)} 个用例:")
        for case in cases[:50]:  # 限制显示数量
            print(f"  {case.rel_path}_{case.operator}_{case.case_id}: {case.dtypes}")
        if len(cases) > 50:
            print(f"  ... 还有 {len(cases) - 50} 个用例")

    else:
        # 列出算子
        operator_loader = OperatorLoader(config.kernel_bench_root)
        operators = operator_loader.list_operators()

        print(f"\n共 {len(operators)} 个算子:")
        for op in operators:
            diff_info = f" ({op.difficulty})" if op.difficulty else ""
            print(f"  {op.name}{diff_info}: {op.category} - {op.description[:50]}")

    return 0


def cmd_info(args):
    """显示算子详细信息"""
    config = get_config()
    operator_loader = OperatorLoader(config.kernel_bench_root)

    op_info = operator_loader.get_operator_by_name(args.operator)
    if op_info is None:
        print(f"[ERROR] 算子 {args.operator} 不存在")
        return 1

    print(f"\n算子信息:")
    print(f"  名称: {op_info.name}")
    print(f"  路径: {op_info.rel_path}")
    print(f"  类别: {op_info.category}")
    print(f"  难度: {op_info.difficulty}")
    print(f"  公式: {op_info.formula}")
    print(f"  描述: {op_info.description}")
    print(f"  目录: {op_info.dir_name}")
    print(f"\n接口签名:")
    print(f"  {op_info.schema}")
    print(f"\n输入:")
    for inp in op_info.inputs:
        print(f"  - {inp.name}: {inp.dtype}")
    print(f"\n输出:")
    for out in op_info.outputs:
        print(f"  - {out.name}: {out.dtype}")
    print(f"\n属性:")
    for attr in op_info.attrs:
        default_str = f" (默认: {attr.default})" if attr.default is not None else ""
        print(f"  - {attr.name}: {attr.type}{default_str}")

    # 显示用例统计
    case_loader = CaseLoader(config.kernel_bench_root)
    cases = case_loader.scan_by_operator(args.operator)
    print(f"\n用例数: {len(cases)}")

    return 0


def cmd_config(args):
    """配置管理"""
    config = get_config()

    if args.kernel_bench_root:
        config.kernel_bench_root = args.kernel_bench_root
        set_config(config)
        print(f"[INFO] kernel_bench_root 设置为: {config.kernel_bench_root}")

    if args.reports_dir:
        config.reports_dir = args.reports_dir
        set_config(config)
        print(f"[INFO] reports_dir 设置为: {config.reports_dir}")

    if args.show or not (args.kernel_bench_root or args.reports_dir):
        print("\n当前配置:")
        print(f"  kernel_bench_root: {config.kernel_bench_root}")
        print(f"  reports_dir: {config.reports_dir}")
        print(f"  source_dir: {config.source_dir}")
        print(f"  warmup: {config.warmup}")
        print(f"  repeat: {config.repeat}")
        print(f"\n精度阈值:")
        for dtype, threshold in config.precision_thresholds.items():
            # 显示阈值和10倍阈值（MARE阈值）
            mare_threshold = 10 * threshold if threshold > 0 else 0
            print(f"  {dtype}: threshold={threshold:.6f}, mare_threshold={mare_threshold:.6f}")

    return 0


def cmd_eval_process(args):
    """进程池子进程执行评测

    由 ProcessPoolCoordinator 通过 subprocess 调用，
    执行分配到的 operators 或 cases，结果写入 JSON 文件。
    """
    import os
    import multiprocessing

    # 强制使用 fork 方式启动子进程
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    # 抑制 CANN/Ascend C++ 层日志（不影响 Python logging）
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'

    import torch
    import torch_npu

    # 初始化 NPU 设备
    # 注意：必须先初始化 NPU，再设置编译模式
    torch.npu.set_device(args.card_id)

    # 关闭 JIT 编译模式
    # 某些算子（如 ForeachNorm）在脚本模式下首次调用时
    # JIT 编译会因内部状态未初始化而失败
    try:
        torch.npu.set_compile_mode(jit_compile=False)
    except Exception as e:
        print(f"[Process {args.process_id}] set_compile_mode failed: {e}")

    # 设置环境变量
    kernel_bench_root = os.environ.get('KERNEL_BENCH_ROOT', '')
    if kernel_bench_root:
        config = get_config()
        config.kernel_bench_root = kernel_bench_root
        set_config(config)

    # 创建配置
    config = Config()
    config.device_type = "npu"
    config.device_id = args.card_id
    config.auto_fallback = False
    config.enable_profiler = args.enable_profiler
    config.warmup = args.warmup
    config.repeat = args.repeat
    set_config(config)

    # 打印进程信息
    print(f"[Process {args.process_id}] Card {args.card_id} 开始执行")
    print(f"[Process {args.process_id}] Profiler: {args.enable_profiler}")

    from .eval.evaluator import Evaluator
    from .eval.results import EvalOperatorResult, EvalCaseResult, summarize_case_results
    from .data.case_loader import CaseLoader, CaseInfo
    from .data.golden_loader import GoldenLoader

    evaluator = Evaluator(config)
    results = []

    try:
        if getattr(args, 'rel_paths', None):
            # rel_path_parallel 模式
            rel_paths = args.rel_paths.split(',')

            print(f"[Process {args.process_id}] 评测算子: {rel_paths}")

            for rel_path in rel_paths:
                print(f"[Process {args.process_id}] 开始评测算子 {rel_path}")
                # 获取算子信息
                from .data.operator_loader import OperatorLoader
                op_loader = OperatorLoader(config.kernel_bench_root)
                try:
                    op_info = op_loader.get_operator(rel_path)
                    operator_name = op_info.name
                except Exception:
                    operator_name = Path(rel_path).name

                # 加载该算子的用例并评测
                case_loader = CaseLoader(config.kernel_bench_root)
                cases = case_loader.scan_by_rel_path(rel_path)

                if not cases:
                    print(f"[Process {args.process_id}] {rel_path}: 无用例")
                    continue

                # 评测每个用例
                golden_loader = GoldenLoader(config.kernel_bench_root)
                case_results = []
                for i, case in enumerate(cases, 1):
                    case_id_str = case.get_case_id_str()
                    print(f"[Process {args.process_id}] [{i}/{len(cases)}] {case_id_str}")

                    golden_func = golden_loader.get_golden_function(case.rel_path)
                    result = evaluator.evaluate_case(case, golden_func)
                    case_results.append(result)

                    status = "✅" if result.success else "❌"
                    elapsed = result.perf_result.elapsed_us if result.perf_result else 0
                    speedup = result.get_speedup()
                    # 精度信息
                    acc_info = ""
                    if result.accuracy_result:
                        mare = result.accuracy_result.mare
                        max_diff = result.accuracy_result.max_diff
                        acc_info = f"MARE={mare:.6f}, max_diff={max_diff:.6f}"
                    # speedup 信息
                    speedup_info = f", speedup={speedup:.2f}x" if speedup > 0 else ""
                    error_hint = result.error_msg[:80] if result.error_msg else ""
                    print(
                        f"[Process {args.process_id}] [{i}/{len(cases)}] "
                        f"{case_id_str}: {status} ({elapsed:.2f}μs{speedup_info}) {acc_info} {error_hint}"
                    )

                # 合并为算子结果
                summary = summarize_case_results(case_results)

                op_result = EvalOperatorResult(
                    rel_path=rel_path,
                    operator=operator_name,
                    total_cases=len(cases),
                    passed_cases=summary.passed,
                    failed_cases=summary.failed,
                    skipped_cases=summary.skipped,
                    results=case_results,
                    pass_rate=summary.pass_rate,
                    avg_speedup=summary.avg_speedup,
                )
                results.append(op_result.to_dict())
                print(f"[Process {args.process_id}] {rel_path}: 通过 {summary.passed}/{len(cases)}")

        elif args.cases_file:
            # case_parallel 模式
            with open(args.cases_file, 'r') as f:
                cases_data = json.load(f)

            # 重建 CaseInfo 对象
            cases = []
            for c in cases_data:
                case = CaseInfo(
                    rel_path=c['rel_path'],
                    operator=c['operator'],
                    case_id=c['case_id'],
                    input_shapes=c['input_shapes'],
                    dtypes=c['dtypes'],
                    value_ranges=c['value_ranges'],
                    attrs=c.get('attrs', {}),
                    note=c.get('note', ''),
                    yaml_path=c.get('yaml_path', ''),
                )
                if 'baseline_perf_us' in c:
                    case.baseline_perf_us = c['baseline_perf_us']
                if 't_hw_us' in c:
                    case.t_hw_us = c['t_hw_us']
                cases.append(case)

            operator = cases[0].operator if cases else 'unknown'
            rel_path = cases[0].rel_path if cases else 'unknown'

            print(f"[Process {args.process_id}] 评测用例: {len(cases)} 个 ({operator})")

            # 获取 golden 函数
            golden_loader = GoldenLoader(config.kernel_bench_root)

            case_results = []
            for i, case in enumerate(cases, 1):
                case_id_str = case.get_case_id_str()
                print(f"[Process {args.process_id}] [{i}/{len(cases)}] {case_id_str}")

                # 使用 golden 作为 AI 算子（模拟评测）
                golden_func = golden_loader.get_golden_function(case.rel_path)

                result = evaluator.evaluate_case(case, golden_func)
                case_results.append(result)

                status = "✅" if result.success else "❌"
                elapsed = result.perf_result.elapsed_us if result.perf_result else 0
                speedup = result.get_speedup()
                # 精度信息
                acc_info = ""
                if result.accuracy_result:
                    mare = result.accuracy_result.mare
                    max_diff = result.accuracy_result.max_diff
                    acc_info = f"MARE={mare:.6f}, max_diff={max_diff:.6f}"
                # speedup 信息
                speedup_info = f", speedup={speedup:.2f}x" if speedup > 0 else ""
                error_hint = result.error_msg[:80] if result.error_msg else ""
                print(
                    f"[Process {args.process_id}] [{i}/{len(cases)}] "
                    f"{case_id_str}: {status} ({elapsed:.2f}μs{speedup_info}) {acc_info} {error_hint}"
                )

            # 合并为算子结果
            summary = summarize_case_results(case_results)

            op_result = EvalOperatorResult(
                rel_path=rel_path,
                operator=operator,
                total_cases=len(cases),
                passed_cases=summary.passed,
                failed_cases=summary.failed,
                skipped_cases=summary.skipped,
                results=case_results,
                pass_rate=summary.pass_rate,
                avg_speedup=summary.avg_speedup,
            )
            results.append(op_result.to_dict())

        else:
            print(f"[Process {args.process_id}] ERROR: 未指定任务")

    except Exception as e:
        print(f"[Process {args.process_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        evaluator.shutdown()
        # 强制杀死 profiler fork 子进程
        # 使用进程组杀死所有子进程（包括 profiler fork 的解析进程）
        try:
            import os
            import signal
            # 杀死当前进程组中的所有进程（不包括当前进程）
            pgid = os.getpgid(0)  # 获取当前进程组 ID
            if pgid != os.getpid():  # 如果进程组 ID 不等于当前进程 ID
                os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass

    # 写入结果文件
    output_data = {"results": results, "process_id": args.process_id}
    Path(args.output).write_text(json.dumps(output_data, ensure_ascii=False, indent=2))

    print(f"[Process {args.process_id}] 完成，结果写入 {args.output}")

    return 0


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    project_root = get_project_root()

    # 设置默认kernel_bench路径
    config = get_config()
    if not config.kernel_bench_root:
        config.kernel_bench_root = str(project_root / "kernel_bench")
        set_config(config)

    # 执行命令
    start_time = time.time()
    try:
        if args.command == 'eval':
            ret = cmd_eval(args)
        elif args.command == 'list':
            ret = cmd_list(args)
        elif args.command == 'info':
            ret = cmd_info(args)
        elif args.command == 'config':
            ret = cmd_config(args)
        elif args.command == 'eval-process':
            ret = cmd_eval_process(args)
        else:
            parser.print_help()
            ret = 0
    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        import traceback
        traceback.print_exc()
        ret = 1

    elapsed = time.time() - start_time
    print(f"\n执行耗时: {elapsed:.2f}s")

    return ret


if __name__ == '__main__':
    sys.exit(main())