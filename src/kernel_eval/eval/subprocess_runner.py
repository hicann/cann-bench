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
子进程执行器

职责：
1. Fork 子进程运行单个算子评测
2. 处理超时（SIGTERM → SIGKILL）
3. 解析子进程输出 JSON

从 evaluator.py 拆分出来，简化职责。
"""

import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

from ..config import get_project_root
from .results import EvalCaseResult, EvalOperatorResult
from .failure_synthesizer import FailureSynthesizer


class SubprocessRunner:
    """子进程执行器"""

    def __init__(
        self,
        failure_synthesizer: FailureSynthesizer,
        device_id: int = 0,
        kernel_eval_root: str = None,
        config=None,
    ):
        self.failure_synthesizer = failure_synthesizer
        self.device_id = device_id
        self.kernel_eval_root = kernel_eval_root or str(get_project_root() / "src")
        # 父进程的评测配置，用于把性能/profiler flag 透传给隔离子进程
        self.config = config

    def _build_child_cmd(
        self,
        operator_name: str,
        frag_path: str,
        source_dir: str,
        case_filter: Optional[Dict] = None,
        task_dir: Optional[str] = None,
        unbuffered: bool = False,
    ) -> List[str]:
        """构造隔离子进程的 kernel_eval.cli eval 命令。

        子进程默认 enable_profiler=True、warmup/repeat/profiler_level 取 argparse
        默认值。若不把父进程的配置透传过去，--no-perf / --warmup / --repeat /
        --profiler-level 在默认隔离路径下会被静默忽略（gitcode issue #11）。

        --no-subprocess-isolation + --skip-install 用于避免无限递归与重复安装。
        """
        cmd = [sys.executable]
        if unbuffered:
            # 强制无缓冲输出，确保所有 case 日志都被实时捕获
            cmd.append("-u")
        cmd += [
            "-m", "kernel_eval.cli", "eval",
            "--source-dir", str(source_dir),
            "--operator", operator_name,
            "--child-json-output", frag_path,
            "--no-subprocess-isolation",
            "--skip-install",
        ]
        if case_filter and "case_id" in case_filter:
            cmd += ["--case-id", str(case_filter["case_id"])]

        effective_task_dir = task_dir if task_dir is not None else self._task_dir_arg()
        if effective_task_dir:
            cmd += ["--task-dir", effective_task_dir]

        cmd += self._forward_config_flags()
        return cmd

    def _task_dir_arg(self, rel_path: str = "") -> Optional[str]:
        """Return the task directory that must be forwarded to child evals."""
        cfg = self.config
        if cfg is None:
            return None

        tasks_root = getattr(cfg, "tasks_root", "") or ""
        if not tasks_root:
            return None

        root = Path(tasks_root)
        if rel_path:
            candidate = root / rel_path
            if candidate.exists():
                return str(candidate)
        return str(root)

    def _forward_config_flags(self) -> List[str]:
        """把父进程的性能/profiler 相关配置透传给隔离子进程。

        config 缺省时返回空列表（子进程沿用默认配置）。
        """
        cfg = self.config
        if cfg is None:
            return []
        flags: List[str] = []
        if not getattr(cfg, "enable_profiler", True):
            flags.append("--no-perf")
        if getattr(cfg, "warmup", None) is not None:
            flags += ["--warmup", str(cfg.warmup)]
        if getattr(cfg, "repeat", None) is not None:
            flags += ["--repeat", str(cfg.repeat)]
        profiler_level = getattr(cfg, "profiler_level", None)
        if profiler_level:
            flags += ["--profiler-level", str(profiler_level)]
        device_id = getattr(cfg, "device_id", None)
        if device_id is not None:
            flags += ["--device-id", str(device_id)]
        return flags

    def run_operator_subprocess(
        self,
        operator_name: str,
        rel_path: str = "",
        source_dir: str = "",
        case_filter: Optional[Dict] = None,
        timeout_sec: int = 240,
        filter_func: Optional[callable] = None,
    ) -> EvalOperatorResult:
        """Fork 一个子进程运行单个算子的评测

        子进程用 --skip-install + --no-subprocess-isolation 避免重复安装和无限递归。
        超时先 SIGTERM 给 finally 块做 NPU 清理的机会，10s 宽限后 SIGKILL。

        Args:
            operator_name: 算子名称
            rel_path: 相对路径
            source_dir: 源码目录
            case_filter: 用例筛选条件
            timeout_sec: 超时时间（秒）
            filter_func: 筛选函数（可选）

        Returns:
            EvalOperatorResult: 算子评测结果
        """
        fd, frag_path = tempfile.mkstemp(suffix=".json", prefix="cannbench_child_")
        os.close(fd)
        try:
            # 子进程命令（透传父进程的性能/profiler 配置）
            cmd = self._build_child_cmd(
                operator_name, frag_path, source_dir=source_dir,
                case_filter=case_filter,
                task_dir=self._task_dir_arg(rel_path),
            )

            print(f"[INFO] {operator_name}: subprocess (timeout {timeout_sec}s)")

            # 设置 PYTHONPATH
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                paths = existing_pythonpath.split(":")
                if self.kernel_eval_root not in paths:
                    env["PYTHONPATH"] = f"{self.kernel_eval_root}:{existing_pythonpath}"
            else:
                env["PYTHONPATH"] = self.kernel_eval_root

            # 继承 CANN/Ascend 环境变量，确保子进程能正确访问 NPU
            cann_env_vars = [
                "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
                "ASCEND_AICPU_PATH", "ASCEND_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH", "PATH", "TBE_IMPL_PATH",
                "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH",
            ]
            for var in cann_env_vars:
                if var in os.environ:
                    env[var] = os.environ[var]

            proc = subprocess.Popen(cmd, start_new_session=True, env=env)

            try:
                rc = proc.wait(timeout=timeout_sec)
                if rc != 0:
                    return self.failure_synthesizer.synthesize_subprocess_failure(
                        operator_name, rel_path=rel_path,
                        reason=f"subprocess exited rc={rc}",
                        case_filter=case_filter, filter_func=filter_func,
                    )
            except subprocess.TimeoutExpired:
                print(f"[WARN] {operator_name} 超过 {timeout_sec}s — SIGTERM")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"[WARN] {operator_name} 宽限后仍未退出 — SIGKILL")
                    proc.kill()
                    proc.wait()
                return self.failure_synthesizer.synthesize_subprocess_failure(
                    operator_name, rel_path=rel_path,
                    reason=f"exceeded {timeout_sec}s timeout — killed",
                    case_filter=case_filter, filter_func=filter_func,
                )

            if not os.path.exists(frag_path) or os.path.getsize(frag_path) == 0:
                return self.failure_synthesizer.synthesize_subprocess_failure(
                    operator_name, rel_path=rel_path,
                    reason="subprocess produced no output",
                    case_filter=case_filter, filter_func=filter_func,
                )

            try:
                data = json.loads(Path(frag_path).read_text())
            except Exception as e:
                return self.failure_synthesizer.synthesize_subprocess_failure(
                    operator_name, rel_path=rel_path,
                    reason=f"parse child JSON: {e}",
                    case_filter=case_filter, filter_func=filter_func,
                )

            ops = data.get("operators", [])
            if not ops:
                return self.failure_synthesizer.synthesize_subprocess_failure(
                    operator_name, rel_path=rel_path,
                    reason="subprocess output had no operators",
                    case_filter=case_filter, filter_func=filter_func,
                )

            # 解析结果
            op_d = ops[0]
            case_results: List[EvalCaseResult] = [
                EvalCaseResult.from_dict(r) for r in op_d.get("results", [])
            ]

            return EvalOperatorResult(
                rel_path=op_d.get("rel_path", rel_path),
                operator=op_d.get("operator", operator_name),
                total_cases=op_d.get("total_cases", len(case_results)),
                passed_cases=op_d.get("passed_cases", 0),
                failed_cases=op_d.get("failed_cases", len(case_results)),
                skipped_cases=op_d.get("skipped_cases", 0),
                results=case_results,
                pass_rate=op_d.get("pass_rate", 0.0),
                avg_speedup=op_d.get("avg_speedup", 0.0),
            )
        finally:
            try:
                os.unlink(frag_path)
            except OSError:
                pass

    def run_operator_subprocess_simple(
        self,
        operator_name: str,
        rel_path: str = "",
        case_filter: Optional[Dict] = None,
        timeout_sec: int = 240,
        filter_func: Optional[callable] = None,
    ) -> EvalOperatorResult:
        """Fork 一个子进程运行单个算子的评测（用于 run_simple.py 场景）

        使用 kernel_eval.cli 进行评测，Golden-as-AI 模式。
        每个算子在独立子进程执行，一个挂起不影响其他算子。

        Args:
            operator_name: 算子名称
            rel_path: 相对路径
            case_filter: 用例筛选条件
            timeout_sec: 子进程超时（秒）
            filter_func: 筛选函数（可选）

        Returns:
            EvalOperatorResult: 子评测结果
        """
        fd, frag_path = tempfile.mkstemp(suffix=".json", prefix="cannbench_op_")
        os.close(fd)
        try:
            # 使用 kernel_eval.cli 代替 run_simple.py
            # 使用 -u 参数强制无缓冲输出，确保所有case日志都被捕获
            # source-dir 优先用父进程配置，回退到 "tasks"（透传性能/profiler 配置）
            source_dir = getattr(self.config, "source_dir", "") or "tasks"
            cmd = self._build_child_cmd(
                operator_name, frag_path, source_dir=source_dir,
                case_filter=case_filter,
                task_dir=self._task_dir_arg(rel_path),
                unbuffered=True,
            )

            # 设置 PYTHONPATH
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            if self.kernel_eval_root not in existing_pythonpath:
                env["PYTHONPATH"] = (
                    f"{self.kernel_eval_root}:{existing_pythonpath}"
                    if existing_pythonpath
                    else self.kernel_eval_root
                )

            # 强制子进程使用无缓冲stdout（解决日志丢失问题）
            env["PYTHONUNBUFFERED"] = "1"

            # 继承 CANN/Ascend 环境变量
            cann_env_vars = [
                "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
                "ASCEND_AICPU_PATH", "ASCEND_VISIBLE_DEVICES",
                "LD_LIBRARY_PATH", "PATH", "TBE_IMPL_PATH",
                "ASCEND_CACHE_PATH", "ASCEND_WORK_PATH",
            ]
            for var in cann_env_vars:
                if var in os.environ:
                    env[var] = os.environ[var]

            print(f"[INFO] {operator_name}: subprocess (timeout {timeout_sec}s)")
            proc = subprocess.Popen(cmd, start_new_session=True, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # 合并stderr到stdout
                                    bufsize=1, text=True)  # 行缓冲，文本模式

            # 实时输出子进程日志
            # F020: was an unbounded list — verbose / profiler-Level2 children
            # can dump tens of thousands of lines, and only the last 10 are
            # ever consumed. Use a bounded deque so peak memory stays small.
            stdout_lines: deque = deque(maxlen=256)
            try:
                for line in proc.stdout:
                    stdout_lines.append(line)
                    print(line, end='')  # 实时输出到终端
                proc.wait(timeout=timeout_sec)
                rc = proc.returncode
                if rc != 0:
                    return self.failure_synthesizer.synthesize_subprocess_failure(
                        operator_name, rel_path=rel_path,
                        reason=f"subprocess exited rc={rc}, output: {''.join(list(stdout_lines)[-10:])}",
                        case_filter=case_filter, filter_func=filter_func,
                    )
            except subprocess.TimeoutExpired:
                print(f"[WARN] {operator_name} 超过 {timeout_sec}s — SIGTERM")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"[WARN] {operator_name} 宽限后仍未退出 — SIGKILL")
                    proc.kill()
                    proc.wait()
                return self.failure_synthesizer.synthesize_subprocess_failure(
                    operator_name, rel_path=rel_path,
                    reason=f"exceeded {timeout_sec}s timeout",
                    case_filter=case_filter, filter_func=filter_func,
                )

            if not os.path.exists(frag_path) or os.path.getsize(frag_path) == 0:
                return self.failure_synthesizer.synthesize_subprocess_failure(
                    operator_name, rel_path=rel_path,
                    reason="subprocess produced no output",
                    case_filter=case_filter, filter_func=filter_func,
                )

            try:
                data = json.loads(Path(frag_path).read_text())
            except Exception as e:
                return self.failure_synthesizer.synthesize_subprocess_failure(
                    operator_name, rel_path=rel_path,
                    reason=f"parse output: {e}",
                    case_filter=case_filter, filter_func=filter_func,
                )

            # 解析结果
            ops = data.get("operators", [])
            if not ops:
                return self.failure_synthesizer.synthesize_subprocess_failure(
                    operator_name, rel_path=rel_path,
                    reason="output had no operators",
                    case_filter=case_filter, filter_func=filter_func,
                )

            op_d = ops[0]
            case_results: List[EvalCaseResult] = [
                EvalCaseResult.from_dict(r) for r in op_d.get("results", [])
            ]

            passed = sum(1 for r in case_results if r.success)
            speedups = [r.get_speedup() for r in case_results if r.success and r.get_speedup() > 0]
            avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

            return EvalOperatorResult(
                rel_path=op_d.get("rel_path", rel_path),
                operator=op_d.get("operator", operator_name),
                total_cases=op_d.get("total_cases", len(case_results)),
                passed_cases=op_d.get("passed_cases", passed),
                failed_cases=op_d.get("failed_cases", len(case_results) - passed),
                skipped_cases=op_d.get("skipped_cases", 0),
                results=case_results,
                pass_rate=op_d.get("pass_rate", passed / len(case_results) if case_results else 0.0),
                avg_speedup=avg_speedup,
            )
        finally:
            try:
                os.unlink(frag_path)
            except OSError:
                pass
