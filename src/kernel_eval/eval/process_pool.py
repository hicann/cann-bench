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
进程池协调器 — 统一 TaskUnit 调度

核心设计：
1. 任务单元 = (算子, 用例组, device_id)，统一调度粒度
2. 不管单算子还是多算子，按 (算子×用例组) 均分到各卡
3. 子进程通过 eval-child 独立子命令执行（纯执行者，不做调度/编译/fork）
4. 主进程按算子维度聚合 case 结果 → EvalOperatorResult

配置示例：
    processes_per_card = 1  # 每卡并发进程数（profiler 开启时强制为 1）
    card_count = 8          # 8 张 NPU 卡
    timeout_per_operator = 300  # 单算子超时（秒）
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .results import EvalOperatorResult, EvalCaseResult, summarize_case_results
from .subprocess_utils import (
    _CANN_ENV_VARS,
    _write_oom_score_adj,
    _is_oom_killed,
    _synthesize_failure_cases,
    _try_recover_partial_results,
)
from ..config import Config, get_config, get_project_root
from ..base.models import CaseSpec


_DEVICE_VISIBILITY_ENV_VARS = (
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NPU_VISIBLE_DEVICES",
)



# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class ProcessConfig:
    """进程池配置"""
    processes_per_card: int = 2      # 每卡最大并发进程数
    timeout_per_operator: int = 300  # 单算子超时（秒）
    enable_profiler: bool = True     # 是否启用 profiler


@dataclass
class TaskUnit:
    """统一任务单元 = (算子, 用例组, device_id)"""
    operator: str               # 算子名称
    rel_path: str               # 算子相对路径
    cases: List[CaseSpec]       # 该进程需要跑的用例列表
    device_id: int              # 分配的 NPU 卡 ID


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def split_into_chunks(items: list, n: int) -> list:
    """将列表分成 n 个尽量均匀的块"""
    if n <= 0:
        return [items]
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def build_task_units(
    cases_by_operator: Dict[str, List[CaseSpec]],
    card_count: int,
) -> List[TaskUnit]:
    """将算子×用例拆分为 TaskUnit，均分到各卡。

    每个算子的用例按卡数分组，形成 TaskUnit 列表。
    多算子场景：算子A的用例分到卡0-7，算子B的用例也分到卡0-7 → 自然负载均衡。
    单算子场景：用例分到卡0-7 → 单算子多卡并行。
    单卡场景：只有一个 chunk → 退化串行。
    """
    task_units: List[TaskUnit] = []
    card_ids = list(range(card_count))

    for operator_name, cases in cases_by_operator.items():
        chunks = split_into_chunks(cases, card_count)
        for i, chunk in enumerate(chunks):
            if chunk:
                task_units.append(TaskUnit(
                    operator=operator_name,
                    rel_path=chunk[0].rel_path,
                    cases=chunk,
                    device_id=card_ids[i % len(card_ids)],
                ))

    return task_units


def aggregate_by_operator(
    all_case_results: List[EvalCaseResult],
) -> List[EvalOperatorResult]:
    """按算子名聚合 case 结果 → EvalOperatorResult"""
    grouped: Dict[str, List[EvalCaseResult]] = defaultdict(list)
    for cr in all_case_results:
        grouped[cr.operator].append(cr)

    results: List[EvalOperatorResult] = []
    for op_name, case_results in grouped.items():
        summary = summarize_case_results(case_results)
        rel_path = case_results[0].rel_path if case_results else ""
        results.append(EvalOperatorResult(
            rel_path=rel_path,
            operator=op_name,
            total_cases=len(case_results),
            passed_cases=summary.passed,
            failed_cases=summary.failed,
            skipped_cases=summary.skipped,
            results=case_results,
            pass_rate=summary.pass_rate,
            avg_speedup=summary.avg_speedup,
        ))

    return results


# ---------------------------------------------------------------------------
# ProcessPoolCoordinator
# ---------------------------------------------------------------------------

class ProcessPoolCoordinator:
    """进程池协调器

    管理 card_count × processes_per_card 个并发槽位，
    按 TaskUnit 分配任务并汇总结果。

    支持单卡和多卡模式：
    - 单卡模式：指定 device_id，card_count=1
    - 多卡模式：不指定 device_id，自动检测
    """

    def __init__(
        self,
        base_config: Config = None,
        process_config: ProcessConfig = None,
        device_id: int = None,  # 指定单卡时，所有进程绑定到该卡
    ):
        self.base_config = base_config or get_config()
        self.process_config = process_config or ProcessConfig()
        self.device_id = device_id

        # 单卡模式：所有进程绑定到指定卡
        if device_id is not None:
            self.card_count = 1
        else:
            # 多卡模式：自动检测
            self.card_count = self._detect_cards()

        # torch_npu.profiler 使用 ACL 设备级 profiling 硬件资源，
        # 同一 NPU 卡上多进程并发 profile 会竞争该资源。
        # profiling 开启时每卡仅 1 进程，保证 profiler 独占硬件资源。
        if self.process_config.enable_profiler:
            self.process_config.processes_per_card = 1

        self.total_processes = self.card_count * self.process_config.processes_per_card
        # 记录活跃子进程，用于 shutdown 时清理
        self._active_processes: List[subprocess.Popen] = []

    def _detect_cards(self) -> int:
        """检测可用 NPU 卡数，并提供详细的诊断信息"""
        if self.base_config.device_type != "npu":
            return 0

        try:
            import torch_npu
        except ImportError as e:
            print("[ERROR] 无法导入 torch_npu 模块")
            print(f"  原因: {e}")
            self._check_npu_smi()
            return 0

        if not torch.npu.is_available():
            print("[ERROR] torch.npu.is_available() 返回 False")
            self._check_npu_smi()
            return 0

        card_count = torch.npu.device_count()
        if card_count == 0:
            print("[ERROR] torch.npu.device_count() 返回 0")
            self._check_npu_smi()
            return 0

        print(f"[INFO] 检测到 {card_count} 张 NPU 卡")
        for i in range(card_count):
            try:
                name = torch.npu.get_device_name(i) if hasattr(torch.npu, 'get_device_name') else 'unknown'
                print(f"  NPU:{i} - {name}")
            except Exception:
                print(f"  NPU:{i}")

        return card_count

    def _check_npu_smi(self):
        """调用 npu-smi info 检查硬件状态"""
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                print("\n[诊断] npu-smi info 输出:")
                lines = result.stdout.strip().split('\n')
                for line in lines[:20]:
                    print(f"  {line}")
                if len(lines) > 20:
                    print(f"  ... (共 {len(lines)} 行，已截断)")
        except FileNotFoundError:
            print("\n[诊断] 未找到 npu-smi 命令")
        except subprocess.TimeoutExpired:
            print("\n[诊断] npu-smi info 执行超时")
        except Exception as e:
            print(f"\n[诊断] npu-smi info 执行异常: {e}")

    def _build_env(self) -> Dict[str, str]:
        """构建子进程环境变量"""
        kernel_eval_root = str(get_project_root() / "src")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        if kernel_eval_root not in existing_pythonpath:
            env["PYTHONPATH"] = f"{kernel_eval_root}:{existing_pythonpath}" if existing_pythonpath else kernel_eval_root
        env["PYTHONUNBUFFERED"] = "1"
        env["TASKS_ROOT"] = self.base_config.tasks_root
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

        for var in _CANN_ENV_VARS:
            if var in os.environ:
                env[var] = os.environ[var]

        return env

    @staticmethod
    def _visible_device_tokens_from_env(env: Dict[str, str]) -> List[str]:
        """Return parent-visible physical device tokens in logical order."""
        for var in _DEVICE_VISIBILITY_ENV_VARS:
            raw = env.get(var, "")
            if not raw:
                continue
            tokens = [token.strip() for token in raw.split(",") if token.strip()]
            if tokens and not any(token.lower() in {"all", "none"} for token in tokens):
                return tokens
        return []

    def _physical_device_token_for_task(self, task: TaskUnit, env: Dict[str, str]) -> str:
        visible_tokens = self._visible_device_tokens_from_env(env)
        if len(visible_tokens) >= self.card_count and 0 <= task.device_id < len(visible_tokens):
            return visible_tokens[task.device_id]
        return str(task.device_id)

    def _should_narrow_child_visibility(self) -> bool:
        return (
            self.base_config.device_type == "npu"
            and self.device_id is None
            and self.card_count > 1
        )

    def _child_device_id(self, task: TaskUnit) -> int:
        # Once the child process sees exactly one physical NPU, torch_npu
        # remaps that card to logical device 0.
        return 0 if self._should_narrow_child_visibility() else task.device_id

    def _build_env_for_task(self, base_env: Dict[str, str], task: TaskUnit) -> Dict[str, str]:
        env = base_env.copy()
        if not self._should_narrow_child_visibility():
            return env

        physical_device = self._physical_device_token_for_task(task, base_env)
        for var in _DEVICE_VISIBILITY_ENV_VARS:
            env[var] = physical_device
        env["KERNEL_EVAL_PHYSICAL_DEVICE_ID"] = physical_device
        env["KERNEL_EVAL_LOGICAL_DEVICE_ID"] = "0"
        return env

    def _build_child_cmd(self, task: TaskUnit, cases_file: str, output_file: str) -> List[str]:
        """构建 eval-child 子进程命令"""
        cmd = [sys.executable, "-u", "-m", "kernel_eval.cli", "eval-child",
               "--bench-name", self.base_config.bench_name,
               "--device-id", str(self._child_device_id(task)),
               "--cases-file", cases_file,
               "--output", output_file,
               "--warmup", str(self.base_config.warmup),
               "--repeat", str(self.base_config.repeat),
               ]

        reports_dir = getattr(self.base_config, "reports_dir", "") or ""
        if reports_dir:
            cmd += ["--reports-dir", str(reports_dir)]

        # task-dir 透传
        tasks_root = getattr(self.base_config, "tasks_root", "")
        if tasks_root:
            cmd += ["--task-dir", str(tasks_root)]

        # source-dir 透传（Stanford bench 等需要在子进程中加载 ai_op.py）
        source_dir = getattr(self.base_config, "source_dir", "") or ""
        if source_dir:
            cmd += ["--source-dir", str(source_dir)]

        # profiler 配置
        if not self.process_config.enable_profiler:
            cmd.append("--no-perf")
        profiler_level = getattr(self.base_config, "profiler_level", None)
        if profiler_level:
            cmd += ["--profiler-level", str(profiler_level)]

        # torch op guard 模式
        torch_op_guard_mode = getattr(self.base_config, "torch_op_guard_mode", None)
        if torch_op_guard_mode:
            cmd += ["--torch-op-guard-mode", str(torch_op_guard_mode)]

        # eval seed
        eval_seed = getattr(self.base_config, "eval_seed", None)
        if eval_seed is not None:
            cmd += ["--eval-seed", str(eval_seed)]

        return cmd

    def evaluate_task_units(self, task_units: List[TaskUnit]) -> List[EvalCaseResult]:
        """按 TaskUnit 并行评测

        每个 TaskUnit 启动一个 eval-child 子进程，
        通过 ThreadPoolExecutor 实现多卡并行和动态负载均衡。
        """
        if self.card_count == 0:
            if os.environ.get("ALLOW_NO_NPU_CARDS") == "1":
                print("[WARN] 无可用 NPU 卡 (ALLOW_NO_NPU_CARDS=1)", flush=True)
                return []
            raise RuntimeError(
                "[ERROR] 无可用 NPU 卡 (card_count=0)。多卡评测需要至少 1 张 NPU。"
                "如确需在无 NPU 环境跑空评测做 dry-run，设置 ALLOW_NO_NPU_CARDS=1。"
            )

        if not task_units:
            return []

        max_workers = self.card_count * self.process_config.processes_per_card
        total_cases = sum(len(t.cases) for t in task_units)

        print(f"[INFO] 配置: {self.card_count} 卡 × {self.process_config.processes_per_card} 并发/卡")
        print(f"[INFO] TaskUnit 数: {len(task_units)}, 用例数: {total_cases}")
        print(f"[INFO] 单算子超时: {self.process_config.timeout_per_operator}s")
        print(f"[INFO] 最大并发: {max_workers}")

        base_env = self._build_env()
        all_case_results: List[EvalCaseResult] = []
        completed_count = 0

        def _run_task(idx_and_task):
            """在线程中运行一个 TaskUnit"""
            idx, task = idx_and_task
            # 写 cases JSON 文件
            fd, cases_file = tempfile.mkstemp(suffix=".json", prefix="cases_")
            os.close(fd)
            try:
                Path(cases_file).write_text(json.dumps([c.to_dict() for c in task.cases], ensure_ascii=False))

                # 写 output 文件占位
                fd, output_file = tempfile.mkstemp(suffix=".json", prefix="cannbench_")
                os.close(fd)

                cmd = self._build_child_cmd(task, cases_file, output_file)
                env = self._build_env_for_task(base_env, task)
                timeout = len(task.cases) * self.process_config.timeout_per_operator

                proc = subprocess.Popen(cmd, start_new_session=True, env=env)
                self._active_processes.append(proc)
                oom_ok = _write_oom_score_adj(proc.pid, 1000)
                # 父进程外部写是双保险，子进程自设（cmd_eval_child）才是主路径
                if not oom_ok:
                    print(f"[WARN] 子进程 PID={proc.pid} oom_score_adj 设置失败"
                          f"（OOM Kill 时主进程也可能被杀）", flush=True)

                try:
                    rc = proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # 从活跃列表移除已完成的子进程
                    try:
                        self._active_processes.remove(proc)
                    except ValueError:
                        pass

                    print(f"[WARN] TaskUnit {task.operator}@Card{task.device_id} 超时 ({timeout}s)")
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    # 超时：尝试恢复部分结果，失败则合成全量 timeout 失败
                    partial = _try_recover_partial_results(output_file)
                    if partial:
                        completed_ids = {r.case_id for r in partial}
                        remaining = [c for c in task.cases if c.get_case_id_str() not in completed_ids]
                        oom_rest = _synthesize_failure_cases(remaining, "timeout",
                            f"子进程超时 ({timeout}s) 被 SIGTERM/SIGKILL")
                        print(f"[INFO] {task.operator}: 超时后恢复 {len(partial)} 个已完成用例，"
                              f"合成 {len(oom_rest)} 个超时失败用例")
                        return (task, partial + oom_rest)
                    return (task, _synthesize_failure_cases(task.cases, "timeout",
                        f"子进程超时 ({timeout}s) 被 SIGTERM/SIGKILL"))

                # 从活跃列表移除已完成的子进程，避免内存累积
                try:
                    self._active_processes.remove(proc)
                except ValueError:
                    pass

                if rc != 0:
                    if _is_oom_killed(proc, rc):
                        # OOM Kill：尝试恢复部分结果 + 合成剩余用例的 oom_killed 失败
                        partial = _try_recover_partial_results(output_file)
                        if partial:
                            completed_ids = {r.case_id for r in partial}
                            remaining = [c for c in task.cases if c.get_case_id_str() not in completed_ids]
                            oom_rest = _synthesize_failure_cases(remaining, "oom_killed",
                                "子进程被 OOM Killer 杀死 (SIGKILL/-9)，内存不足")
                            print(f"[WARN] {task.operator}@Card{task.device_id}: OOM Kill (rc={rc})")
                            print(f"[INFO] {task.operator}: OOM Kill 后恢复 {len(partial)} 个已完成用例，"
                                  f"合成 {len(oom_rest)} 个 OOM 失败用例")
                            return (task, partial + oom_rest)
                        print(f"[WARN] {task.operator}@Card{task.device_id}: OOM Kill (rc={rc})，无部分结果可恢复")
                        return (task, _synthesize_failure_cases(task.cases, "oom_killed",
                            "子进程被 OOM Killer 杀死 (SIGKILL/-9)，内存不足"))
                    print(f"[WARN] {task.operator}@Card{task.device_id}: 子进程异常退出 rc={rc}")
                    return (task, _synthesize_failure_cases(task.cases, "subprocess_failure",
                        f"子进程异常退出 rc={rc}"))

                # 正常退出：读取完整结果
                # 从活跃列表移除已完成的子进程
                try:
                    self._active_processes.remove(proc)
                except ValueError:
                    pass

                try:
                    data = json.loads(Path(output_file).read_text())
                except (json.JSONDecodeError, OSError) as e:
                    print(f"[WARN] TaskUnit {task.operator}@Card{task.device_id} 结果解析失败: {e}")
                    return (task, _synthesize_failure_cases(task.cases, "subprocess_failure",
                        f"子进程结果 JSON 解析失败: {e}"))

                case_results = [EvalCaseResult.from_dict(r) for r in data.get("case_results", [])]
                return (task, case_results)

            finally:
                try:
                    os.unlink(cases_file)
                except OSError:
                    pass
                try:
                    os.unlink(output_file)
                except OSError:
                    pass

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            indexed_tasks = list(enumerate(task_units))
            futures = {
                executor.submit(_run_task, item): item
                for item in indexed_tasks
            }

            for future in as_completed(futures):
                item = futures[future]
                idx, task = item
                try:
                    task_info, case_results = future.result()
                    completed_count += 1
                    all_case_results.extend(case_results)
                    passed = sum(1 for r in case_results if r.success)
                    status = "✅" if passed > 0 else "❌"
                    print(f"[INFO] [{completed_count}/{len(task_units)}] Card {task.device_id}: "
                          f"{task.operator} {status} ({passed}/{len(task.cases)})")

                    # 定期清理已退出的子进程引用，避免内存累积
                    self._cleanup_completed_processes()

                    # 每完成 3 个任务执行一次 GC，回收主进程临时对象
                    if completed_count % 3 == 0:
                        import gc
                        gc.collect()
                        avail_mb = self._get_available_memory_mb()
                        if avail_mb > 0 and avail_mb < 2048:
                            print(f"[WARN] 可用内存低: {avail_mb:.0f} MB，"
                                  f"活跃子进程: {len(self._active_processes)}", flush=True)

                except Exception as e:
                    completed_count += 1
                    print(f"[WARN] [{completed_count}/{len(task_units)}] Card {task.device_id}: "
                          f"{task.operator} 异常: {e}")

        print(f"[INFO] 调度完成: {completed_count}/{len(task_units)} 个 TaskUnit")
        return all_case_results

    def _cleanup_completed_processes(self):
        """清理已完成的子进程引用，避免内存累积。

        `_active_processes` 列表持有所有子进程的 Popen 对象引用。
        已退出的进程对象虽然轻量但仍占用内存，定期清理可防止长时间运行时内存泄漏。
        """
        completed = [p for p in self._active_processes if p.poll() is not None]
        for p in completed:
            try:
                self._active_processes.remove(p)
            except ValueError:
                pass
        if completed:
            import gc
            gc.collect()

    @staticmethod
    def _get_available_memory_mb() -> float:
        """获取系统可用内存（MB）。"""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return kb / 1024.0
        except (OSError, ValueError, IndexError):
            pass
        return 0.0

    def shutdown(self):
        """关闭所有活跃子进程

        SIGTERM 先，10s 宽限后 SIGKILL。
        """
        grace_sec = 5
        for proc in self._active_processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

        if self._active_processes:
            deadline = time.time() + grace_sec
            for proc in self._active_processes:
                remaining = max(deadline - time.time(), 0)
                if remaining <= 0 or proc.poll() is not None:
                    continue
                try:
                    proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass

        for proc in self._active_processes:
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._active_processes = []

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'device_id': self.device_id,
            'card_count': self.card_count,
            'processes_per_card': self.process_config.processes_per_card,
            'total_processes': self.total_processes,
        }
