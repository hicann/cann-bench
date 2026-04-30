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

from .config import Config, get_config, set_config
from .data.operator_loader import OperatorLoader
from .data.case_loader import CaseLoader
from .eval.evaluator import Evaluator
from .report.report_generator import ReportGenerator


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
    eval_parser.add_argument('--operator', type=str, default=None,
                             help='算子名称（如 Exp, Softmax）')
    eval_parser.add_argument('--level', type=int, default=None, choices=[1, 2, 3, 4],
                             help='难度级别筛选')
    eval_parser.add_argument('--case-id', type=int, default=None,
                             help='用例编号筛选')
    eval_parser.add_argument('--device-id', type=int, default=0,
                             help='NPU 设备 ID（默认: 0）')
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

    return parser


def cmd_eval(args):
    """执行评测命令"""
    # 创建配置
    config = Config()

    if getattr(args, 'kernel_bench_root', None):
        config.kernel_bench_root = args.kernel_bench_root
    if args.output:
        config.reports_dir = args.output
    if args.source_dir:
        config.source_dir = args.source_dir
    if hasattr(args, 'device_id'):
        config.device_id = args.device_id
    if getattr(args, 'no_perf', False):
        config.enable_profiler = False

    set_config(config)

    # 初始化评测器
    evaluator = Evaluator(config)
    report_generator = ReportGenerator(
        output_dir=args.output,
        eval_code=args.eval_code,
    )

    # 构建筛选条件
    operator_filter = None
    if args.operator:
        operator_filter = [args.operator]

    case_filter = {'case_id': args.case_id} if args.case_id else None

    subprocess_isolation = not getattr(args, 'no_subprocess_isolation', False)
    op_timeout_sec = getattr(args, 'op_timeout_sec', 240)
    skip_install = getattr(args, 'skip_install', False)
    child_json_output = getattr(args, 'child_json_output', None)
    iterative_compile = not getattr(args, 'no_iterative_compile', False)

    # 确定评测方式
    if args.source_dir and not skip_install:
        # 从源码目录评测（自动扫描、迭代编译、安装、并行评测）
        session_result = evaluator.evaluate_from_source(
            source_dir=args.source_dir,
            operator_filter=operator_filter,
            case_filter=case_filter,
            verbose=args.verbose,
            subprocess_isolation=subprocess_isolation,
            op_timeout_sec=op_timeout_sec,
            iterative_compile=iterative_compile,
        )
        for op_result in session_result.operators:
            report_generator.add_operator_result(op_result)

    elif args.source_dir and skip_install:
        # --skip-install：子进程模式下父进程已经装好 wheel，这里只要扫描
        # 已安装的 cann_bench 跑指定算子即可（内部不再 fork 子进程）
        session_result = evaluator.evaluate_skip_build(
            operator_filter=operator_filter,
            case_filter=case_filter,
        )
        for op_result in session_result.operators:
            report_generator.add_operator_result(op_result)

    else:
        # 不指定source-dir，使用已安装的cann_bench评测
        if operator_filter and args.level:
            # 仅执行Golden验证（不安装whl包）
            result = evaluator.evaluate_golden_only(
                operator=args.operator,
                level=args.level,
                case_filter=case_filter,
            )
            report_generator.add_operator_result(result)
        else:
            # 评测已安装的cann_bench所有匹配的算子
            session_result = evaluator.evaluate_skip_build(
                operator_filter=operator_filter,
                case_filter=case_filter,
            )
            for op_result in session_result.operators:
                report_generator.add_operator_result(op_result)

    # 生成报告
    report = report_generator.generate()

    # 子进程模式：只把每个算子的 to_dict() 写到 --child-json-output，父进程
    # lift 出来合入最终会话结果；不走常规 save_all / print_summary（那会在
    # 父进程里再做一次）。
    if child_json_output:
        payload = {"operators": [op.to_dict() for op in report_generator.operator_results]}
        Path(child_json_output).write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        report_generator.save_all(report)
        report_generator.print_summary(report)

    # 关闭评测器
    evaluator.shutdown()

    # Child subprocesses signal success/failure through the JSON frag, not
    # the exit code — the parent only uses rc to detect crashes (segfault,
    # OOM kill, timeout). A run where every case legitimately FAILed is
    # still a successful run from the child's perspective.
    if child_json_output:
        return 0
    return 0 if report.passed_cases > 0 else 1


def cmd_list(args):
    """列出算子/用例"""
    config = get_config()

    if args.cases:
        # 列出用例
        case_loader = CaseLoader(config.kernel_bench_root)

        if args.level and args.operator:
            cases = case_loader.scan_by_operator(args.level, args.operator)
        elif args.level:
            cases = case_loader.scan_by_level(args.level)
        else:
            cases = case_loader.scan_all_cases()

        print(f"\n共 {len(cases)} 个用例:")
        for case in cases[:50]:  # 限制显示数量
            print(f"  L{case.level}_{case.operator}_{case.case_id}: {case.dtypes}")
        if len(cases) > 50:
            print(f"  ... 还有 {len(cases) - 50} 个用例")

    else:
        # 列出算子
        operator_loader = OperatorLoader(config.kernel_bench_root)
        operators = operator_loader.list_operators(args.level)

        print(f"\n共 {len(operators)} 个算子:")
        for op in operators:
            print(f"  {op.name} (L{op.level}): {op.category} - {op.description[:50]}")

    return 0


def cmd_info(args):
    """显示算子详细信息"""
    config = get_config()
    operator_loader = OperatorLoader(config.kernel_bench_root)

    # 查找算子
    level = args.level
    if level is None:
        # 在所有level中查找
        for lv in [1, 2, 3, 4]:
            try:
                op_info = operator_loader.get_operator(args.operator, lv)
                level = lv
                break
            except FileNotFoundError:
                continue

    if level is None:
        print(f"[ERROR] 算子 {args.operator} 不存在")
        return 1

    op_info = operator_loader.get_operator(args.operator, level)

    print(f"\n算子信息:")
    print(f"  名称: {op_info.name}")
    print(f"  级别: L{op_info.level}")
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
    cases = case_loader.scan_by_operator(level, args.operator)
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


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent

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