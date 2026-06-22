#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
子进程公共工具

职责：
1. OOM Killer 保护（oom_score_adj 设置 + 检测）
2. CANN/Ascend 环境变量继承列表
3. 子进程失败结果合成

供 ProcessPoolCoordinator 和 eval-child 共用。
"""

import subprocess
from typing import Dict, List

from .results import EvalCaseResult


# ---------------------------------------------------------------------------
# OOM Killer 保护
# ---------------------------------------------------------------------------

def _write_oom_score_adj(pid: int, value: int) -> bool:
    """写入 /proc/<pid>/oom_score_adj，返回是否成功。

    value 范围 [-1000, 1000]：
      - -1000: 该进程几乎不会被 OOM Killer 选为牺牲者
      - 0:      默认值
      + 1000:  该进程最优先被 OOM Killer 杀死
    """
    path = f"/proc/{pid}/oom_score_adj"
    try:
        with open(path, "w") as f:
            f.write(str(value))
        print(f"[INFO] oom_score_adj={value} 设置成功: PID={pid}", flush=True)
        return True
    except PermissionError:
        print(f"[WARN] oom_score_adj 写入失败: {path} — 权限不足"
              f"（需要 root 或 CAP_SYS_ADMIN，OOM 保护未生效）", flush=True)
        return False
    except FileNotFoundError:
        print(f"[WARN] oom_score_adj 写入失败: {path} — 进程已退出"
              f"（子进程可能瞬间崩溃）", flush=True)
        return False
    except OSError as e:
        print(f"[WARN] oom_score_adj 写入失败: {path} — {e}"
              f"（OOM 保护未生效）", flush=True)
        return False


def get_available_memory_mb() -> float:
    """获取系统可用内存（MB）。

    读取 /proc/meminfo 中的 MemAvailable（包括可回收的 page cache）。
    若无法读取，回退到 psutil 或返回 0。
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    # 格式: "MemAvailable:    12345678 kB"
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (OSError, ValueError, IndexError):
        pass

    # 回退：尝试 psutil
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        return 0.0


def _is_oom_killed(proc: subprocess.Popen, rc: int) -> bool:
    """判断子进程是否疑似被 OOM Killer 杀死。

    检测条件：退出码为 -9 (Python Popen) 或 137 (bash)，即 SIGKILL。

    注意：任何 SIGKILL（OOM Killer、手动 kill -9、cgroup 杀进程等）都产生
    相同退出码，此函数无法区分来源。超时路径的 SIGKILL 由调用方通过
    try/except TimeoutExpired 分支排除，不会进入此函数。
    """
    if rc not in (-9, 137):
        return False
    return True


# ---------------------------------------------------------------------------
# CANN/Ascend 环境变量列表（子进程需要继承才能正确访问 NPU）
# ---------------------------------------------------------------------------

_CANN_ENV_VARS = [
    "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "ASCEND_AICPU_PATH", "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH", "PATH", "TBE_IMPL_PATH",
    "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH",
]


# ---------------------------------------------------------------------------
# 子进程失败结果合成
# ---------------------------------------------------------------------------

def _synthesize_failure_cases(
    task_cases: list,
    failure_type: str,
    error_msg: str,
) -> List[EvalCaseResult]:
    """为子进程失败的 TaskUnit 合成 all-FAIL 的 EvalCaseResult 列表。

    当 eval-child 子进程因 OOM/超时/崩溃等原因无法正常返回结果时，
    使用 TaskUnit 中已有的 CaseSpec 列表合成失败结果，
    确保失败算子仍然出现在报告中（而非完全失踪）。

    Args:
        task_cases: TaskUnit.cases 列表（CaseSpec 对象）
        failure_type: 失败类型标记（"oom_killed" / "timeout" / "subprocess_failure"）
        error_msg: 失败原因描述
    """
    results = []
    for c in task_cases:
        case_id_str = c.get_case_id_str()
        results.append(EvalCaseResult(
            case_id=case_id_str,
            rel_path=c.rel_path,
            operator=c.operator,
            case_num=c.case_num,
            success=False,
            error_msg=error_msg,
            failure_type=failure_type,
            baseline_perf_us=getattr(c, 'baseline_perf_us', 0.0) or 0.0,
            t_hw_us=getattr(c, 't_hw_us', 0.0) or 0.0,
        ))
    return results


def _try_recover_partial_results(output_file: str) -> List[EvalCaseResult]:
    """尝试从 output_file 读取 eval-child 增量写入的部分结果。

    eval-child 子进程通过 incremental_output 增量写入已完成用例结果。
    OOM Kill 时父进程可从 output_file 恢复已完成的部分结果。

    Args:
        output_file: eval-child 的 --output 文件路径

    Returns:
        已完成的 EvalCaseResult 列表。解析失败时返回空列表。
    """
    import json
    from pathlib import Path

    if not Path(output_file).exists() or Path(output_file).stat().st_size == 0:
        return []

    try:
        data = json.loads(Path(output_file).read_text())
    except (json.JSONDecodeError, OSError):
        return []

    raw = data.get("case_results", [])
    if not raw:
        # 兼容旧格式 {"operators": [...]}
        ops = data.get("operators", [])
        if ops:
            raw = ops[0].get("results", [])

    return [EvalCaseResult.from_dict(r) for r in raw]
