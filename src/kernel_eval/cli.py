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
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import Config, get_config, get_project_root, set_config
from .data.data_generator import INPUT_DIST_CHOICES
from .data.tensor_cache import add_cache_arguments, configure_cache
from .base.enums import DifficultyLevel
from .benches.cann import CannTaskLoader, CannCaseLoader
from .eval.evaluator import Evaluator, _write_json_atomic
from .eval.results import EvalOperatorResult
from .base.score_weights import score_weights_context
from .base.perf_metric import resolve_perf_metric
from .report.report_generator import ReportGenerator
from .utils.path_resolver import resolve_task_dir

# 导入 benches 模块，触发所有评测集组件注册（使用相对导入）
from .benches import cann as _cann_bench


LEVEL_CHOICES = tuple(int(level.value[1:]) for level in DifficultyLevel)


def _matches_level(rel_path: str, level: int) -> bool:
    """Return whether a task/case path belongs to the requested level."""
    return rel_path.split("/", 1)[0] == f"level{level}"


def _operator_names_for_level(bench_name: str, tasks_root: str, level: int) -> List[str]:
    """Return operator names in one difficulty level for CPU evaluation."""
    from .registry import get_task_loader

    loader = get_task_loader(bench_name, tasks_root=tasks_root)
    return [op.name for op in loader.list_tasks() if op.get_level_id() == level]


# 尝试导入 cann_bench（用户提交的算子包），触发 torch.ops.cann_bench 注册
try:
    import cann_bench
except ImportError:
    pass


def _get_available_memory_mb() -> float:
    """获取系统可用内存（MB），用于内存监控和诊断。"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog='kernel-bench',
        description='算子评测工程命令行工具',
    )

    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # eval 命令
    eval_parser = subparsers.add_parser('eval', help='执行评测')

    # === 评测集参数 ===
    eval_parser.add_argument('--bench-name', type=str, default='cann',
                             help='评测集名称（默认: cann）。指定后自动加载对应配置：'
                                  'Loader、评分方案、精度判断器等。可通过 BenchRegistry 注册自定义评测集。')

    # === 原有参数 ===
    eval_parser.add_argument('--source-dir', type=str, default=None,
                             help='AI生成的算子源码目录（不指定则使用已安装的cann_bench）')
    eval_parser.add_argument('--task-dir', type=str, default=None,
                             help='评测目录（bench根目录或算子目录），替代 --level。'
                                  '支持: tasks, tasks/level1, tasks/level1/exp 等')
    eval_parser.add_argument('--operator', type=str, default=None,
                             help='算子名称（如 Exp, Softmax）')
    eval_parser.add_argument('--level', type=int, default=None, choices=LEVEL_CHOICES,
                             help='难度级别筛选（可选 1-5）')
    eval_parser.add_argument('--case-id', type=int, default=None,
                             help='用例编号筛选')
    eval_parser.add_argument('--device-id', type=int, default=None,
                             help='NPU 设备 ID（单卡模式）。不指定则自动使用全部可用卡（多卡并行）')
    eval_parser.add_argument('--device', type=str, default='npu',
                             choices=['cpu', 'npu'],
                             help='设备类型（默认: npu）')
    eval_parser.add_argument('--processes-per-card', type=int, default=2,
                             help='每卡进程数（多卡并行模式，默认: 2）')
    eval_parser.add_argument('--max-cases-per-task-unit', type=int, default=64,
                             help='单个 eval-child 最多处理的 case 数（默认: 64）。'
                                  '通过轮换子进程限制 host 内存累积，不降低并发度')
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
    eval_parser.add_argument('--op-timeout-sec', type=int, default=240,
                             help='子进程隔离下 per-op 超时。超时先 SIGTERM，10s 宽限后 SIGKILL。'
                                  '默认 240 秒。')
    eval_parser.add_argument('--no-iterative-compile', action='store_true',
                             help='已废弃（保留以兼容旧脚本，无实际作用）。编译统一为整体编译：'
                                  'build.sh 编译失败即本次提交相关算子全部按编译失败计 0 分，'
                                  '不再隔离/补救失败算子，也不修改用户源码（为支持并行化评测）。')
    eval_parser.add_argument('--no-perf', action='store_true',
                             help='关闭性能采集，仅做精度验证')
    eval_parser.add_argument('--perf-batch-cases', dest='perf_batch_cases',
                             action='store_true',
                             help='将同一 TaskUnit 的多个 case 合并到一次 NPU profiler 会话（默认开启）')
    eval_parser.add_argument('--no-perf-batch-cases', dest='perf_batch_cases',
                             action='store_false',
                             help='关闭批量 case 性能采集，恢复逐 case profiler 会话')
    eval_parser.set_defaults(perf_batch_cases=True)
    eval_parser.add_argument('--perf-batch-case-memory-limit-mb', type=float,
                             default=32 * 1024,
                             help='允许 case 进入批量 profiler 的正确性阶段 VmRSS 峰值上限（MB，'
                                  '默认: 32768；<=0 表示不按内存分流）')
    eval_parser.add_argument('--profiler-level', type=str, default='Level1',
                             choices=['Level1', 'Level2'],
                             help='Profiler 级别（默认: Level1）。Level1 产出 47 列 CSV，'
                                  'Level2 增加更详细的 AICPU 采集。')
    eval_parser.add_argument('--no-freq-boost', action='store_true',
                             help='关闭 NPU 升频清 cache（10240x10240 大矩阵 warmup）。'
                                  '部分 NPU 卡在大矩阵运算时可能 hang，可临时关闭。')
    eval_parser.add_argument('--perf-metric-strategy', type=str, default=None,
                             help='Override perf metric strategy (kernel_details | trace_view). '
                                  'When set, overrides the strategy from BenchConfig. '
                                  'trace_view uses tilefwk/PYPTO aicore_e2e (PYPTO 口径).')
    eval_parser.add_argument('--torch-op-guard-mode', type=str, default=None,
                             choices=['off', 'warn', 'block'],
                             help='AI 算子调用禁用 PyTorch 内置计算 API 时的处理方式。'
                                  '默认使用 Config.torch_op_guard_mode（block）。'
                                  '生产评测应使用 block；调试可临时使用 warn/off。')

    eval_parser.add_argument('--eval-seed', type=int, default=0,
                             help='输入生成确定性种子（默认: 0 = 基于case_id自动确定）。'
                                  '改变 seed 可获得不同但可复现的输入。'
                                  '设为 -1 表示纯随机模式（不推荐，会导致 flaky 测试）。')

    eval_parser.add_argument('--input-dist', type=str, default='uniform',
                             choices=INPUT_DIST_CHOICES,
                             metavar='DIST',
                             help='输入数据分布（默认: uniform）。uniform = 在 value_range 内'
                                  '均匀采样（与历史行为逐位一致）；normal = 正态分布，'
                                  'mu=(min+max)/2、sigma=(max-min)/6，采样后 clamp 回 value_range。'
                                  '别名: u / norm / gaussian / n。仅作用于浮点输入，'
                                  '整数与 bool 始终均匀（索引类输入需覆盖整个维度）。'
                                  '本项对整场评测生效，评测范围仍由 --task-dir/--operator/'
                                  '--case-id 控制。')

    # 内部开关：跳过编译安装，使用 PYTHONPATH 上的 cann_bench（ST harness 使用）
    eval_parser.add_argument('--skip-install', action='store_true',
                             help=argparse.SUPPRESS)

    # list 命令
    list_parser = subparsers.add_parser('list', help='列出算子/用例')
    list_parser.add_argument('--bench-name', type=str, default='cann',
                             help='评测集名称（默认: cann）')
    list_parser.add_argument('--level', type=int, default=None, choices=LEVEL_CHOICES,
                             help='按级别筛选（可选 1-5）')
    list_parser.add_argument('--operator', type=str, default=None,
                             help='按算子筛选')
    list_parser.add_argument('--cases', action='store_true', help='列出用例而非算子')
    list_parser.add_argument('--task-dir', type=str, default=None,
                             help='评测目录（默认使用配置的 tasks_root）')

    # info 命令
    info_parser = subparsers.add_parser('info', help='显示算子详细信息')
    info_parser.add_argument('--bench-name', type=str, default='cann',
                             help='评测集名称（默认: cann）')
    info_parser.add_argument('--operator', type=str, required=True,
                             help='算子名称')
    info_parser.add_argument('--level', type=int, default=None, choices=LEVEL_CHOICES,
                             help='难度级别（可选 1-5）')
    info_parser.add_argument('--task-dir', type=str, default=None,
                             help='评测目录（默认使用配置的 tasks_root）')

    # config 命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_parser.add_argument('--show', action='store_true', help='显示当前配置')
    config_parser.add_argument('--tasks-root', '--kernel-bench-root', dest='tasks_root',
                               type=str, default=None,
                               help='设置 tasks 数据目录（--kernel-bench-root 为向后兼容别名）')
    config_parser.add_argument('--reports-dir', type=str, default=None,
                               help='设置报告输出目录')
    config_parser.add_argument('--list-benches', action='store_true',
                               help='列出已注册的评测集')
    config_parser.add_argument('--list-scoring-schemes', action='store_true',
                               help='列出已注册的评分方案')
    config_parser.add_argument('--list-checkers', action='store_true',
                               help='列出已注册的精度判断器')

    # eval-child 命令（子进程专用入口，纯执行者）
    # 不做调度、不做编译安装、不 fork，只从 JSON 加载 cases 逐个评测
    child_parser = subparsers.add_parser('eval-child', help=argparse.SUPPRESS)
    child_parser.add_argument('--bench-name', type=str, default='cann')
    child_parser.add_argument('--device-id', type=int, required=True,
                              help='NPU 卡 ID')
    child_parser.add_argument('--cases-file', type=str, required=True,
                              help='TaskUnit JSON 文件路径')
    child_parser.add_argument('--output', type=str, required=True,
                              help='结果 JSON 输出路径')
    child_parser.add_argument('--reports-dir', type=str, default=None,
                              help=argparse.SUPPRESS)
    child_parser.add_argument('--task-dir', type=str, default=None)
    child_parser.add_argument('--source-dir', type=str, default=None,
                              help=argparse.SUPPRESS)  # Stanford bench 透传
    child_parser.add_argument('--warmup', type=int, default=3)
    child_parser.add_argument('--repeat', type=int, default=5)
    child_parser.add_argument('--no-perf', action='store_true',
                              help='关闭性能采集')
    child_parser.add_argument('--perf-batch-cases', dest='perf_batch_cases',
                              action='store_true', help=argparse.SUPPRESS)
    child_parser.add_argument('--no-perf-batch-cases', dest='perf_batch_cases',
                              action='store_false', help=argparse.SUPPRESS)
    child_parser.set_defaults(perf_batch_cases=True)
    child_parser.add_argument('--monitor-case-memory', action='store_true',
                              help=argparse.SUPPRESS)
    child_parser.add_argument('--perf-batch-case-memory-limit-mb', type=float,
                              default=32 * 1024, help=argparse.SUPPRESS)
    child_parser.add_argument('--profiler-level', type=str, default='Level1',
                              choices=['Level1', 'Level2'])
    child_parser.add_argument('--no-freq-boost', action='store_true',
                              help=argparse.SUPPRESS)
    child_parser.add_argument('--torch-op-guard-mode', type=str, default=None,
                              choices=['off', 'warn', 'block'])
    child_parser.add_argument('--eval-seed', type=int, default=0)
    child_parser.add_argument('--input-dist', type=str, default='uniform')
    child_parser.add_argument('--perf-metric-strategy', type=str, default=None,
                              choices=['kernel_details', 'trace_view', 'msprof_summary'],
                              help=argparse.SUPPRESS)
    child_parser.add_argument('--pypto-pro-outer-case-isolation', action='store_true',
                              help=argparse.SUPPRESS)
    child_parser.add_argument('--timeout-per-operator', type=int, default=300,
                              help=argparse.SUPPRESS)

    
    add_cache_arguments(eval_parser)
    add_cache_arguments(child_parser)
    return parser


def _infer_semantic_prefix(args) -> str:
    """根据 CLI 参数推导报告名的语义前缀。

    --eval-code 显式指定时不干预（返回空串，由 _generate_eval_code 保留旧行为）。

    推导规则:
    - --operator  指定时: 算子名小写（如 Softmax → softmax）
    - --task-dir  指定时: 路径最后一段小写（如 level1/exp → exp）
    - 无筛选时:      bench_name（如 cann）
    """
    if getattr(args, 'eval_code', None):
        return ""
    if getattr(args, 'operator', None):
        return args.operator.lower()
    task_dir = getattr(args, 'task_dir', None)
    if task_dir:
        # 取路径最后一段，如 "level1/exp" → "exp", "level1" → "level1"
        return Path(task_dir).name.lower()
    # 全量评测：用 bench_name
    return getattr(args, 'bench_name', 'cann').lower()


def _resolve_operator_info(operator_name: str, config, bench_name: str = "cann"):
    """按算子名称查找 CannTaskSpec，返回 None 若未找到"""
    from .registry import get_task_loader
    try:
        loader = get_task_loader(bench_name, tasks_root=config.tasks_root)
        return loader.get_task_by_name(operator_name)
    except Exception:
        return None


def _create_config_from_args(args, bench_root: str) -> Config:
    """从命令行参数创建配置

    核心逻辑：从 BenchRegistry 获取评测集完整配置
    """
    from .registry import get_bench_config

    # 获取评测集配置（默认 cann）
    bench_name = getattr(args, 'bench_name', 'cann')
    bench_config = get_bench_config(bench_name)

    config = Config()
    config.tasks_root = bench_root

    # 从 BenchConfig 设置配置
    config.checker_name = bench_config.checker
    config.precision_thresholds = bench_config.get_precision_thresholds()

    # CLI 参数
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
    if hasattr(args, 'timeout_per_operator'):
        config.timeout_per_operator = args.timeout_per_operator
    if hasattr(args, 'max_cases_per_task_unit'):
        config.max_cases_per_task_unit = args.max_cases_per_task_unit
    if getattr(args, 'no_perf', False):
        config.enable_profiler = False
    config.perf_batch_cases = bool(
        getattr(args, 'perf_batch_cases', config.perf_batch_cases)
    )
    config.perf_batch_case_memory_limit_mb = float(getattr(
        args, 'perf_batch_case_memory_limit_mb',
        config.perf_batch_case_memory_limit_mb,
    ))
    if hasattr(args, 'profiler_level'):
        config.profiler_level = args.profiler_level
    if getattr(args, 'no_freq_boost', False):
        config.perf_freq_boost = False
    torch_op_guard_mode = getattr(args, 'torch_op_guard_mode', None)
    if torch_op_guard_mode:
        config.torch_op_guard_mode = torch_op_guard_mode

    # 评测种子：-1 表示纯随机（转换为 None），其他值为确定性种子
    eval_seed_raw = getattr(args, 'eval_seed', 0)
    config.eval_seed = None if eval_seed_raw == -1 else eval_seed_raw
    config.input_dist = getattr(args, 'input_dist', 'uniform')
    configure_cache(config, args)

# 性能指标策略覆盖：CLI --perf-metric-strategy 设置时覆盖 BenchConfig 默认值
    perf_metric_strategy = getattr(args, 'perf_metric_strategy', None)
    if perf_metric_strategy:
        config.perf_metric_strategy_override = perf_metric_strategy
    set_config(config)
    return config


def _cmd_eval_npu(args, bench_root: str, filter_prefix: str, config: Config,
                  report_generator: ReportGenerator) -> int:
    """NPU 评测统一入口

    有 source-dir → 主进程先编译安装
    无 source-dir → 直接从已安装包评测
    --device-id → 单卡退化（card_count=1）
    无 --device-id → 多卡并行（自动检测）
    """
    from .eval.process_pool import ProcessPoolCoordinator, ProcessConfig, TaskUnit, build_task_units, aggregate_by_operator
    from .registry import get_case_loader
    from .eval.evaluator import Evaluator

    bench_name = getattr(args, 'bench_name', 'cann')
    device_id = getattr(args, 'device_id', None)
    processes_per_card = getattr(args, 'processes_per_card', 2)
    max_cases_per_task_unit = getattr(args, 'max_cases_per_task_unit', 64)
    timeout_per_operator = getattr(args, 'timeout_per_operator', 300)
    skip_install = getattr(args, 'skip_install', False)

    # ---- 编译安装前置（有 source-dir 时） ----
    compile_failed_results: List[EvalOperatorResult] = []
    matched_operators = None

    if args.source_dir and skip_install:
        # --skip-install 模式：跳过编译安装，使用 PYTHONPATH 上的 cann_bench
        # ST harness 使用此模式：先构建 golden candidate，通过 PYTHONPATH 暴露
        print(f"[INFO] --skip-install 模式：跳过编译安装，使用 PYTHONPATH 上的 cann_bench")
        evaluator = Evaluator(config, bench_name=bench_name)

        # APIGuard snapshot（安全防护）
        from .security.api_guard import APIGuard
        guard = APIGuard()
        guard.snapshot()

        # 扫描已安装的 cann_bench 接口
        matched_operators = evaluator.package_manager.prepare_skip_build()

        # 设置 source_dir 供 operator_matcher 使用（加载 ai_op.py 等）
        if args.source_dir:
            config.source_dir = args.source_dir

        evaluator.shutdown()
        del evaluator
        import gc
        gc.collect()

    elif args.source_dir and bench_name != "stanford":
        iterative_compile = not getattr(args, 'no_iterative_compile', False)
        evaluator = Evaluator(config, bench_name=bench_name)

        # 1. 编译安装
        matched_operators, package_info = evaluator.package_manager.prepare_from_source(
            args.source_dir, iterative_compile=iterative_compile,
        )

        # 1.5 整体编译失败：不隔离/不补救，本次提交相关算子全部按编译失败计 0 分
        # （与 evaluator 共用 synthesize_all_compile_failures，逻辑单一真源）。
        if getattr(package_info, "build_failed", False):
            print(f"[ERROR] 编译失败，本次提交相关算子按编译失败计 0 分（错误汇总见 _compile.log）")
            op_filter = [args.operator] if getattr(args, "operator", None) else None
            compile_failed_results = evaluator.failure_synthesizer.synthesize_all_compile_failures(
                evaluator.operator_matcher, package_info, operator_filter=op_filter,
            )
            for result in compile_failed_results:
                report_generator.add_operator_result(result)
            evaluator.shutdown()
            # 控制流短路；本次评测的退出码由 cmd_eval 据报告统一推导（_compute_exit_code），
            # 此处返回值不被使用。
            return

        # 2. APIGuard 验证
        from .security.api_guard import APIGuard
        guard = APIGuard()
        try:
            guard.verify()
        except RuntimeError as e:
            # 安全篡改 → 合成全部算子的失败结果并返回
            print(f"[ERROR] APIGuard 检测到 Timing API 篡改: {e}")
            for operator_name in (matched_operators or []):
                op_info = evaluator.operator_matcher.find_operator_info(operator_name)
                if op_info:
                    compile_failed_results.append(
                        evaluator.failure_synthesizer.synthesize_security_failure(
                            op_info, str(e)))
            for snake_op_name, err in (package_info.compile_errors or {}).items():
                op_info = evaluator.operator_matcher.find_operator_info_by_snake(snake_op_name)
                if op_info:
                    compile_failed_results.append(
                        evaluator.failure_synthesizer.synthesize_compile_failure(
                            op_info, err))
            for result in compile_failed_results:
                report_generator.add_operator_result(result)
            evaluator.shutdown()
            return 1

        # 3. 合成编译失败结果
        for snake_op_name, err in (package_info.compile_errors or {}).items():
            op_info = evaluator.operator_matcher.find_operator_info_by_snake(snake_op_name)
            if op_info:
                compile_failed_results.append(
                    evaluator.failure_synthesizer.synthesize_compile_failure(op_info, err))

        evaluator.shutdown()

        # 编译后强制清理内存：编译阶段加载了大量源码、编译产物、临时对象，
        # 必须在创建子进程前释放，避免主进程内存过高导致 OOM 时被杀。
        import gc
        del evaluator
        gc.collect()
        gc.collect()  # 二次 GC 清理循环引用

        avail_mb = _get_available_memory_mb()
        print(f"[INFO] 编译完成后可用内存: {avail_mb:.0f} MB", flush=True)

    elif args.source_dir and bench_name == "stanford":
        # StanfordBench: 不编译安装，只设置 source_dir 到 matcher
        config.source_dir = args.source_dir

    # ---- 构建任务列表 ----
    loader = get_case_loader(bench_name, tasks_root=bench_root)
    all_cases = loader.scan_all()

    if filter_prefix:
        all_cases = [
            c for c in all_cases
            if c.rel_path.startswith(filter_prefix + '/') or c.rel_path == filter_prefix
        ]

    if args.level is not None:
        all_cases = [c for c in all_cases if _matches_level(c.rel_path, args.level)]

    if args.operator:
        all_cases = [c for c in all_cases if c.operator.lower() == args.operator.lower()]

    if args.case_id is not None:
        all_cases = [c for c in all_cases if c.case_num == args.case_id]

    if not all_cases:
        print("[WARN] 无匹配用例")
        return 0

    # 编译失败的算子排除（已在 compile_failed_results 中）
    if matched_operators is not None:
        compiled_ops = set(op.lower() for op in matched_operators)
        all_cases = [c for c in all_cases if c.operator.lower() in compiled_ops]

    # 按算子分组用例
    cases_by_operator: Dict[str, List] = defaultdict(list)
    for c in all_cases:
        cases_by_operator[c.operator].append(c)

    # ---- 多卡并行调度 ----
    # 每卡并发度继续遵循 --processes-per-card（默认 2）；独立工作目录
    # 避免并发 worker 共享运行时产物。
    from .eval.subprocess_utils import detect_pypto_pro_submission
    pypto_outer_isolation = detect_pypto_pro_submission()
    config.pypto_pro_outer_case_isolation = pypto_outer_isolation
    if pypto_outer_isolation:
        print("[INFO] PyPTO Pro 提交: 启用独立 worker 隔离"
              f"（{processes_per_card} 并发/卡）", flush=True)

    process_config = ProcessConfig(
        processes_per_card=processes_per_card,
        timeout_per_operator=timeout_per_operator,
        enable_profiler=not args.no_perf,
    )

    config.bench_name = bench_name
    config.timeout_per_operator = timeout_per_operator
    if getattr(args, 'reports_dir', None):
        config.reports_dir = args.reports_dir

    coordinator = ProcessPoolCoordinator(
        base_config=config,
        process_config=process_config,
        device_id=device_id,
    )

    if coordinator.card_count == 0:
        print("[ERROR] 无可用 NPU 卡")
        return 1

    # 构建 TaskUnit 并分配到各卡
    task_units = build_task_units(
        cases_by_operator,
        coordinator.card_count,
        isolate_each_case=pypto_outer_isolation,
        max_cases_per_task_unit=max_cases_per_task_unit,
    )

    rel_paths = list(set(c.rel_path for c in all_cases))
    print(f"\n[INFO] NPU 评测 [{bench_name}]")
    print(f"[INFO] Bench目录: {bench_root}")
    if filter_prefix:
        print(f"[INFO] 筛选路径: {filter_prefix}")
    print(f"[INFO] 卡数: {coordinator.card_count}, 并发: {coordinator.total_processes}")
    print(f"[INFO] 算子数: {len(rel_paths)}, 用例数: {len(all_cases)}")
    print(f"[INFO] TaskUnit数: {len(task_units)}")
    print(f"[INFO] 单 TaskUnit case 上限: {max_cases_per_task_unit}")
    if args.source_dir:
        print(f"[INFO] 源码目录: {args.source_dir}（已编译安装）")
    print(f"[INFO] Warmup/Repeat: {args.warmup}/{args.repeat}")
    if args.no_perf:
        print("[INFO] 性能采集: 关闭")

    start_time = time.time()
    all_case_results = coordinator.evaluate_task_units(task_units)
    total_time = time.time() - start_time

    # 合入编译失败 + 评测结果
    for result in compile_failed_results:
        report_generator.add_operator_result(result)

    operator_results = aggregate_by_operator(all_case_results)
    for op_result in operator_results:
        report_generator.add_operator_result(op_result)

    print(f"\n[效率] 总耗时: {total_time:.2f}s")
    coordinator.shutdown()
    return 0




def cmd_eval_child(args):
    """子进程专用入口：从 JSON 文件加载 TaskUnit，逐个评测用例

    纯执行者——不做调度、不做编译安装、不 fork。
    """
    import os
    import multiprocessing

    # OOM 保护：子进程自设 oom_score_adj=1000，确保 OOM Killer 优先杀子进程
    # 而不是杀主进程。自设比父进程外部写入更可靠——消除 Popen 到
    # 父进程写 /proc/<pid>/oom_score_adj 之间的窗口期，
    # 也覆盖父进程写入权限不足的情况。
    from .eval.subprocess_utils import _write_oom_score_adj
    _write_oom_score_adj(os.getpid(), 1000)

    # 强制使用 fork 方式启动子进程
    try:
        multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    # 抑制 CANN/Ascend C++ 层日志
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = '0'
    os.environ['ASCEND_GLOBAL_LOG_LEVEL'] = '3'

    # APIGuard 安全防护
    from .security.api_guard import APIGuard
    guard = APIGuard()
    guard.snapshot()

    # 初始化 NPU 设备
    import torch
    import torch_npu
    # F745: 当设置了 ASCEND_RT_VISIBLE_DEVICES 后，子进程只能看到一张卡，
    # 这张卡在子进程内部被重新编号为 npu:0，无论物理卡号是多少。
    # 例如：ASCEND_RT_VISIBLE_DEVICES=1 时，物理卡1 在子进程中变成 npu:0
    #
    # 因此，我们需要根据可见设备数量来决定使用哪个设备 ID：
    # - 如果只能看到 1 张卡（隔离模式）：使用 npu:0
    # - 如果能看到多张卡（非隔离模式）：使用传入的 device_id
    visible_device_count = torch.npu.device_count()
    if visible_device_count == 1:
        # 隔离模式：只有一张卡可见，使用 npu:0
        actual_device_id = 0
    else:
        # 非隔离模式：使用传入的 device_id
        actual_device_id = args.device_id

    torch.npu.set_device(actual_device_id)
    try:
        torch.npu.set_compile_mode(jit_compile=False)
    except Exception as e:
        print(f"[eval-child] set_compile_mode failed: {e}")

    # 创建配置
    project_root = get_project_root()
    bench_root, _ = resolve_task_dir(args.task_dir, project_root)

    config = _create_config_from_args_for_child(args, bench_root)
    config.device_type = "npu"
    config.device_id = actual_device_id  # 使用调整后的 device_id

    from .eval.evaluator import Evaluator
    evaluator = Evaluator(config, bench_name=args.bench_name,
                          incremental_output_path=args.output)

    # Stanford bench: 透传 source_dir 到 matcher，使其能加载 ai_op.py
    source_dir = getattr(args, 'source_dir', None)
    if source_dir and args.bench_name == "stanford":
        evaluator.operator_matcher.set_source_dir(source_dir)

    # 从 JSON 文件加载 cases
    from .registry import get_bench_config
    bench_config = get_bench_config(args.bench_name)
    case_spec_cls = bench_config.get_case_spec_cls()
    cases = [case_spec_cls.from_dict(c) for c in json.loads(Path(args.cases_file).read_text())]

    # 评测（复用 evaluator 的评测循环，包含设备恢复、详细输出等）
    operator_name = cases[0].operator if cases else "Unknown"
    rel_path = cases[0].rel_path if cases else ""
    print(f"[eval-child] Card {args.device_id}: {len(cases)} 用例开始评测")

    op_result = evaluator.run_cases(cases, operator_name, rel_path)
    case_results = op_result.results

    # F747: 移除显式等待，依赖 shutdown() 中的 close_pool(wait=True) 同步机制
    # shutdown() 中会调用 MultiProcessPool.close_pool(wait=True)，
    # 这应该足够等待所有 profiler 解析完成。
    #
    # 测试：如果移除显式等待后仍然成功，说明同步机制有效；
    # 如果失败，说明 close_pool(wait=True) 不可靠，需要显式等待。

    evaluator.shutdown()
    guard.verify()

    # 写出 case 结果 JSON
    payload = {"case_results": [r.to_dict() for r in case_results]}
    _write_json_atomic(args.output, payload)

    passed = sum(1 for r in case_results if r.success)
    print(f"[eval-child] Card {args.device_id}: 完成 {passed}/{len(cases)} 通过")
    return 0


def _create_config_from_args_for_child(args, bench_root: str) -> Config:
    """从 eval-child 子命令参数创建配置"""
    from .registry import get_bench_config

    bench_name = args.bench_name
    bench_config = get_bench_config(bench_name)

    config = Config()
    config.tasks_root = bench_root
    config.checker_name = bench_config.checker
    config.precision_thresholds = bench_config.get_precision_thresholds()
    config.warmup = args.warmup
    config.repeat = args.repeat
    config.enable_profiler = not args.no_perf
    config.perf_batch_cases = bool(
        getattr(args, 'perf_batch_cases', config.perf_batch_cases)
    )
    config.monitor_case_memory = bool(getattr(args, 'monitor_case_memory', False))
    config.perf_batch_case_memory_limit_mb = float(getattr(
        args, 'perf_batch_case_memory_limit_mb',
        config.perf_batch_case_memory_limit_mb,
    ))
    config.profiler_level = args.profiler_level
    config.perf_freq_boost = not getattr(args, 'no_freq_boost', False)
    config.bench_name = bench_name
    if getattr(args, 'reports_dir', None):
        config.reports_dir = args.reports_dir

    # source-dir 透传（Stanford bench 等需要在子进程中加载 ai_op.py）
    source_dir = getattr(args, 'source_dir', None)
    if source_dir:
        config.source_dir = source_dir

    eval_seed_raw = getattr(args, 'eval_seed', 0)
    config.eval_seed = None if eval_seed_raw == -1 else eval_seed_raw
    config.pypto_pro_outer_case_isolation = getattr(
        args, 'pypto_pro_outer_case_isolation', False)
    config.timeout_per_operator = getattr(args, 'timeout_per_operator', 300)
    config.input_dist = getattr(args, 'input_dist', 'uniform')
    configure_cache(config, args)

    torch_op_guard_mode = getattr(args, 'torch_op_guard_mode', None)
    if torch_op_guard_mode:
        config.torch_op_guard_mode = torch_op_guard_mode

    perf_metric_strategy = getattr(args, 'perf_metric_strategy', None)
    if perf_metric_strategy:
        config.perf_metric_strategy_override = perf_metric_strategy

    set_config(config)
    return config




def _compute_exit_code(report) -> int:
    """根据评测报告计算进程退出码（0=成功，非零=不达标/出错）。

    退出码语义 = "这次提交是否达标"，供 CI/CD 与 shell `&&` 判定：
    - F042: 旧版以 `passed_cases>0` 为成功 → 52/53 失败仍退出 0 误导 CI；
      改为只要有**真实失败用例**（failed_cases）即非零。
    - 编译失败的用例按 cascade_device 统计、不计入 failed_cases，若只看
      failed_cases 会得到退出码 0（"编译都没过"被误判为通过）。这里显式把
      "带 compilation_error 的算子数"计入退出码，使编译失败 → 整批 0 分 →
      退出码非零。不改报告统计口径（cascade_cases 等保持原样）。

    返回值封顶 255（POSIX 退出码上限）。
    """
    compile_failed_ops = sum(1 for op in report.operators if op.compilation_error)
    return min(report.failed_cases + compile_failed_ops, 255)


def cmd_eval(args):
    """执行评测命令"""
    # OOM 保护策略：
    # eval-child 子进程 oom_score_adj=1000（父进程外部写 + 子进程自设双保险）
    # 确保子进程优先被 OOM Kill，主进程存活以恢复部分结果、合成失败结果、生成报告。

    project_root = get_project_root()

    # 解析 --task-dir 参数
    bench_root, filter_prefix = resolve_task_dir(args.task_dir, project_root)

    with score_weights_context(bench_root):
        resolve_perf_metric()
        return _eval_with_score_profile(args, bench_root, filter_prefix)


def _eval_with_score_profile(args, bench_root, filter_prefix):
    bench_name = getattr(args, 'bench_name', 'cann')
    # 创建配置
    config = _create_config_from_args(args, bench_root)

    # 初始化报告生成器（语义前缀自动推导）
    report_generator = ReportGenerator(
        output_dir=config.reports_dir,
        eval_code=args.eval_code,
        semantic_prefix=_infer_semantic_prefix(args),
        config=config,
    )

    # 构建筛选条件
    operator_filter = [args.operator] if args.operator else None
    case_filter = {'case_id': args.case_id} if args.case_id is not None else None
    level_selected_nothing = False
    backend_exit_code = 0
    if args.level is not None and args.device == 'cpu':
        level_operators = set(_operator_names_for_level(bench_name, bench_root, args.level))
        if operator_filter:
            operator_filter = [name for name in operator_filter if name in level_operators]
        else:
            operator_filter = sorted(level_operators)
        if not operator_filter:
            # 不评测, 但照常往下走到报告生成 -- 与 NPU 路径的 "无匹配用例" 对齐
            # (见 _cmd_eval_npu): 两条路径都退出码 0, 且下游总能拿到 reports/*.md
            print(f"[WARN] level{args.level} 下无匹配算子")
            level_selected_nothing = True

    # 执行评测：CPU 调用仿真模块，NPU 统一走多卡并行
    # 注：simulate / _cmd_eval_npu 的返回值不作为退出码——真正的退出码由下方
    # _compute_exit_code(report) 依据报告统一推导（含编译失败维度），故此处不接收返回值。
    if args.device == 'cpu':
        if not level_selected_nothing:
            from .simulation import simulate
            backend_exit_code = simulate(
                config,
                bench_name=bench_name,
                operator_filter=operator_filter,
                case_filter=case_filter,
                report_generator=report_generator,
            ) or 0
    else:
        backend_exit_code = _cmd_eval_npu(
            args, bench_root, filter_prefix, config, report_generator
        ) or 0

    # 生成报告
    report = report_generator.generate()
    report_generator.save_all(report)
    report_generator.print_summary(report)
    compile_failed_ops = sum(1 for op in report.operators if op.compilation_error)
    if compile_failed_ops:
        print(f"[ERROR] 检测到 {compile_failed_ops} 个算子编译失败，本次提交相关算子整批计 0 分，"
              f"退出码非零。请在提交前本地确保编译通过。", file=sys.stderr, flush=True)
    report_exit_code = _compute_exit_code(report)
    return min(max(report_exit_code, backend_exit_code), 255)




def cmd_list(args):
    """列出算子/用例"""
    from .registry import get_task_loader, get_case_loader

    project_root = get_project_root()
    bench_name = getattr(args, 'bench_name', 'cann')

    bench_root, _ = resolve_task_dir(args.task_dir, project_root)

    if args.cases:
        # 列出用例
        case_loader = get_case_loader(bench_name, tasks_root=bench_root)

        if args.operator:
            cases = case_loader.scan_by_operator(args.operator)
        else:
            cases = case_loader.scan_all()
        if args.level is not None:
            cases = [case for case in cases if _matches_level(case.rel_path, args.level)]

        print(f"\n[{bench_name}] 共 {len(cases)} 个用例:")
        for case in cases[:50]:  # 限制显示数量
            print(f"  {case.rel_path}_{case.operator}_{case.case_num}: {case.dtypes}")
        if len(cases) > 50:
            print(f"  ... 还有 {len(cases) - 50} 个用例")

    else:
        # 列出算子
        task_loader = get_task_loader(bench_name, tasks_root=bench_root)
        operators = task_loader.list_tasks()
        if args.level is not None:
            operators = [op for op in operators if op.get_level_id() == args.level]
        if args.operator:
            operators = [
                op for op in operators if op.name.lower() == args.operator.lower()
            ]

        print(f"\n[{bench_name}] 共 {len(operators)} 个算子:")
        for op in operators[:50]:
            diff_info = f" ({op.difficulty})" if op.difficulty else ""
            desc = op.description[:50] if op.description else ""
            print(f"  {op.name}{diff_info}: {op.category} - {desc}")
        if len(operators) > 50:
            print(f"  ... 还有 {len(operators) - 50} 个算子")

    return 0


def cmd_info(args):
    """显示算子详细信息"""
    from .registry import get_task_loader, get_case_loader

    project_root = get_project_root()
    bench_name = getattr(args, 'bench_name', 'cann')

    bench_root, _ = resolve_task_dir(args.task_dir, project_root)

    task_loader = get_task_loader(bench_name, tasks_root=bench_root)

    op_info = task_loader.get_task_by_name(args.operator)
    if op_info is None:
        print(f"[ERROR] 算子 {args.operator} 不存在")
        return 1
    if args.level is not None and op_info.get_level_id() != args.level:
        print(f"[ERROR] 算子 {args.operator} 不属于 level{args.level}")
        return 1

    print(f"\n[{bench_name}] 算子信息:")
    print(f"  名称: {op_info.name}")
    print(f"  路径: {op_info.rel_path}")
    print(f"  类别: {op_info.category}")
    print(f"  难度: {op_info.difficulty}")
    print(f"  公式: {op_info.formula}")
    print(f"  描述: {op_info.description}")
    if hasattr(op_info, 'dir_name'):
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
    case_loader = get_case_loader(bench_name, tasks_root=bench_root)
    cases = case_loader.scan_by_operator(args.operator)
    print(f"\n用例数: {len(cases)}")

    return 0


def cmd_config(args):
    """配置管理"""
    config = get_config()

    if args.tasks_root:
        config.tasks_root = args.tasks_root
        set_config(config)
        print(f"[INFO] tasks_root 设置为: {config.tasks_root}")

    if args.reports_dir:
        config.reports_dir = args.reports_dir
        set_config(config)
        print(f"[INFO] reports_dir 设置为: {config.reports_dir}")

    # 列出已注册的评测集
    if args.list_benches:
        from .registry import BenchRegistry
        benches = BenchRegistry.list_benches()
        print("\n已注册的评测集:")
        for bench in benches:
            config = BenchRegistry.get(bench)
            desc = config.description if config else ""
            print(f"  - {bench}: {desc}")
        return

    # 列出已注册的评分方案
    if args.list_scoring_schemes:
        from .report.scoring_scheme import ScoringSchemeRegistry
        schemes = ScoringSchemeRegistry.list_schemes()
        print("\n已注册的评分方案:")
        for scheme_name in schemes:
            scheme = ScoringSchemeRegistry.get(scheme_name)
            desc = scheme.get_scheme_description() if scheme else ""
            print(f"  - {scheme_name}: {desc}")
        return

    # 列出已注册的精度判断器
    if args.list_checkers:
        from .eval.checkers import list_correctness_checkers, get_checker_info
        checkers = list_correctness_checkers()
        print("\n已注册的精度判断器:")
        for checker_name in checkers:
            info = get_checker_info(checker_name)
            desc = info.get('description', '') if info else ''
            print(f"  - {checker_name}: {desc}")
        return

    if args.show or not (args.tasks_root or args.reports_dir or args.list_benches or args.list_scoring_schemes or args.list_checkers):
        print("\n当前配置:")
        print(f"  tasks_root: {config.tasks_root}")
        print(f"  reports_dir: {config.reports_dir}")
        print(f"  source_dir: {config.source_dir}")
        print(f"  warmup: {config.warmup}")
        print(f"  repeat: {config.repeat}")
        print(f"  checker: {config.checker_name}")
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

    project_root = get_project_root()

    # 设置默认tasks 路径
    config = get_config()
    if not config.tasks_root:
        config.tasks_root = str(project_root / "tasks")
        set_config(config)

    # 执行命令
    start_time = time.time()
    try:
        if args.command == 'eval':
            ret = cmd_eval(args)
        elif args.command == 'eval-child':
            ret = cmd_eval_child(args)
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
