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
子进程公共工具

职责：
1. OOM Killer 保护（oom_score_adj 设置 + 检测）
2. CANN/Ascend 环境变量继承列表
3. 子进程失败结果合成

供 ProcessPoolCoordinator 和 eval-child 共用。
"""

import ast
import os
import signal
import subprocess
from importlib import metadata
from pathlib import Path
from typing import List

from .results import EvalCaseResult


# ---------------------------------------------------------------------------
# OOM Killer 保护
# ---------------------------------------------------------------------------


def _signal_process_group(proc: subprocess.Popen, sig: int) -> None:
    """向由 ``start_new_session=True`` 启动的子进程组发送信号。

    评测子进程可能继续启动 profiler 或 runtime 子进程。只终止
    Popen 直接对象会把这些后代留在 NPU 上，因此 POSIX 环境下必须
    按进程组清理。非 POSIX 或进程组不存在时回退到 Popen API。
    """
    try:
        # start_new_session=True 令 pgid 等于子进程 pid。即便组长已经被
        # OOM killer 回收，msprof/parser 后代仍可能留在这个进程组中，
        # 因此不能以 proc.poll() 为提前返回条件。
        os.killpg(proc.pid, sig)
        return
    except (AttributeError, ProcessLookupError, PermissionError, OSError):
        pass

    if proc.poll() is not None:
        return
    try:
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.terminate()
    except (ProcessLookupError, OSError):
        pass


def _terminate_process_group(proc: subprocess.Popen, grace_sec: float = 10.0) -> None:
    """终止子进程及其同进程组后代，并回收直接子进程。"""
    _signal_process_group(proc, signal.SIGTERM)
    if proc.poll() is None:
        try:
            proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            pass

    # 无论直接子进程是否已退出，都再向原进程组发 SIGKILL。这样能清理
    # 被 OOM 杀掉组长之后仍存活的 profiler/parser 后代。
    _signal_process_group(proc, signal.SIGKILL)
    if proc.poll() is None:
        # 极端情况下再尝试回收直接子进程，不让清理无限阻塞。
        try:
            proc.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (ProcessLookupError, OSError):
                pass


def _installed_package_sources(pkg_dir: str) -> List[Path]:
    """返回当前 wheel 实际拥有的 Python 源文件。

    不直接遍历整个包目录：不同提交都使用 ``cann_bench`` 分发名时，
    ``pip --force-reinstall`` 可能留下旧 wheel 不再拥有的文件。把这些
    残留文件纳入检测会将普通 PyPTO 提交误判为 PyPTO Pro。
    """
    package_root = Path(pkg_dir).resolve()
    try:
        dist = metadata.distribution("cann-bench")
    except metadata.PackageNotFoundError:
        return []

    sources = []
    for installed_file in dist.files or ():
        if Path(installed_file).suffix != ".py":
            continue
        try:
            source = Path(dist.locate_file(installed_file)).resolve()
            source.relative_to(package_root)
        except (OSError, RuntimeError, ValueError):
            continue
        sources.append(source)
    return sources


def detect_pypto_pro_submission() -> bool:
    """通过当前 ``cann_bench`` wheel 拥有的源码检测 PyPTO Pro。"""
    try:
        import cann_bench

        pkg_dir = os.path.dirname(cann_bench.__file__)
    except (ImportError, OSError, AttributeError, TypeError):
        return False

    source_files = _installed_package_sources(pkg_dir)
    if not source_files:
        # 源码运行或 editable install 可能没有可用的 wheel RECORD。
        source_files = [
            Path(root) / name
            for root, _dirs, files in os.walk(pkg_dir)
            for name in files
            if name.endswith(".py") and "__pycache__" not in Path(root).parts
        ]

    for source_file in source_files:
        try:
            tree = ast.parse(source_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(
                    alias.name == "pypto_pro" or alias.name.startswith("pypto_pro.")
                    for alias in node.names
                ):
                    return True
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "pypto_pro" or module.startswith("pypto_pro."):
                    return True
    return False


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
        print(
            f"[WARN] oom_score_adj 写入失败: {path} — 权限不足"
            f"（需要 root 或 CAP_SYS_ADMIN，OOM 保护未生效）",
            flush=True,
        )
        return False
    except FileNotFoundError:
        print(
            f"[WARN] oom_score_adj 写入失败: {path} — 进程已退出（子进程可能瞬间崩溃）",
            flush=True,
        )
        return False
    except OSError as e:
        print(
            f"[WARN] oom_score_adj 写入失败: {path} — {e}（OOM 保护未生效）", flush=True
        )
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
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "ASCEND_AICPU_PATH",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "PATH",
    "TBE_IMPL_PATH",
    "ASCEND_CACHE_PATH",
    "ASCEND_WORK_PATH",
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
        results.append(
            EvalCaseResult(
                case_id=case_id_str,
                rel_path=c.rel_path,
                operator=c.operator,
                case_num=c.case_num,
                success=False,
                error_msg=error_msg,
                failure_type=failure_type,
                baseline_perf_us=getattr(c, "baseline_perf_us", 0.0) or 0.0,
                baseline_source=(getattr(c, "metadata", None) or {}).get(
                    "baseline_source", {}
                ),
                t_hw_us=getattr(c, "t_hw_us", 0.0) or 0.0,
            )
        )
    return results


def _try_recover_partial_results(output_file: str) -> List[EvalCaseResult]:
    """尝试从 output_file 读取 eval-child 增量写入的部分结果。

    eval-child 子进程通过 incremental_output 增量写入已完成用例结果。
    OOM Kill、超时或普通异常退出（例如 SIGSEGV）时，父进程可从
    output_file 恢复已完成的部分结果。

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
