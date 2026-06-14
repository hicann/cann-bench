#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
CPU 仿真评测模块

职责：在无 NPU 硬件的环境下做功能验证/调试，不涉及多卡并行和编译安装。

主流程（_cmd_eval_npu）不包含 CPU 仿真路径，两者完全分离。
仿真模块复用 evaluator 的 evaluate_operator / evaluate_skip_build 等方法，
底层均为 in-process 执行（不再 fork 子进程）。

调用方式：
1. CLI: kernel-bench eval --device cpu --operator Exp
2. CLI: kernel-bench simulate --operator Exp  （独立入口，等价于上面）
3. Python API: from kernel_eval.simulation import simulate
"""

import sys
from typing import Dict, List, Optional

from kernel_eval.config import Config, get_config, set_config
from kernel_eval.eval.evaluator import Evaluator
from kernel_eval.report.report_generator import ReportGenerator
from kernel_eval.security.api_guard import APIGuard
from kernel_eval.registry import get_task_loader


def simulate(
    config: Config,
    bench_name: str = "cann",
    operator_filter: List[str] = None,
    case_filter: Dict = None,
    report_generator: ReportGenerator = None,
) -> int:
    """CPU 仿真评测入口

    在 CPU 上串行评测算子，用于无 NPU 环境下的功能验证。
    只做评测，不生成报告 — 由调用方（cmd_eval / cmd_simulate）统一生成。

    Args:
        config: 评测配置（device_type 应为 'cpu'）
        bench_name: 评测集名称
        operator_filter: 算子筛选列表
        case_filter: 用例筛选条件
        report_generator: 报告生成器（必须由调用方传入）

    Returns:
        退出码（0=全部通过，非零=有失败）
    """
    guard = APIGuard()
    guard.snapshot()

    if report_generator is None:
        # 独立入口（cmd_simulate）时自行创建报告生成器
        report_generator = ReportGenerator(
            output_dir=config.reports_dir,
            eval_code=f"cpu_sim_{bench_name}",
            semantic_prefix=f"cpu_sim_{bench_name}",
            config=config,
        )
        standalone = True
    else:
        # 被 cmd_eval 调用时，报告由 cmd_eval 统一生成
        standalone = False

    evaluator = Evaluator(config, bench_name=bench_name)

    if operator_filter:
        for operator_name in operator_filter:
            task_loader = get_task_loader(bench_name, tasks_root=config.tasks_root)
            op_info = task_loader.get_task_by_name(operator_name)
            result = evaluator.evaluate_operator(
                operator=operator_name,
                rel_path=op_info.rel_path if op_info else operator_name,
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

    try:
        guard.verify()
    except RuntimeError as e:
        print(f"[SECURITY] {e}", file=sys.stderr, flush=True)
        if standalone:
            report = report_generator.generate()
            report_generator.save_all(report)
            report_generator.print_summary(report)
        return 1

    if standalone:
        report = report_generator.generate()
        report_generator.save_all(report)
        report_generator.print_summary(report)
        return 0 if report.failed_cases == 0 else min(report.failed_cases, 255)
    return 0