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
进程池协调器

基于多卡 × 多进程并行方案实现：
1. 每个进程独立运行，拥有独立的 torch_npu.profiler 单例
2. 进程内单线程执行，避免 profiler 竞争
3. 任务分配策略：动态调度（每算子一个进程，每卡并发上限）
4. 无进程间通信，通过文件传递结果

配置示例：
    processes_per_card = 2  # 每卡最大并发进程数
    card_count = 2          # 2 张卡
    timeout_per_operator = 60  # 单算子超时（秒）
"""

import json
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue, Empty as QueueEmpty
from typing import List, Dict, Optional, Any

import torch

from .results import EvalOperatorResult, EvalCaseResult, summarize_case_results
from ..config import Config, get_config, get_project_root
from ..base.models import CaseSpec


@dataclass
class ProcessConfig:
    """进程池配置"""
    processes_per_card: int = 2      # 每卡最大并发进程数
    timeout_per_operator: int = 300  # 单算子超时（秒）
    enable_profiler: bool = True     # 是否启用 profiler


@dataclass
class OperatorTask:
    """单个算子评测任务"""
    rel_path: str
    card_id: int
    process: Optional[subprocess.Popen] = None
    output_file: Optional[str] = None
    started_at: float = 0.0
    timeout: int = 60
    completed: bool = False
    result: Optional[Dict] = None


class OperatorScheduler:
    """算子调度器 - 动态管理算子进程

    每个算子一个独立进程，同一时刻每卡最多 N 个进程并发执行。
    实现动态负载均衡，避免静态分配导致的任务不均。
    """

    def __init__(
        self,
        process_config: ProcessConfig,
        base_config: Config,
        card_count: int,
    ):
        self.process_config = process_config
        self.base_config = base_config
        self.card_count = card_count

        self.max_concurrent_per_card = process_config.processes_per_card
        self.timeout_per_operator = process_config.timeout_per_operator

        # 任务队列和状态
        self.pending_queue: Queue[str] = Queue()
        self.running_tasks: Dict[int, List[OperatorTask]] = defaultdict(list)
        self.completed_results: List[Dict] = []
        self.lock = threading.Lock()

    def submit_operators(self, rel_paths: List[str]):
        """提交算子列表到队列"""
        for rel_path in rel_paths:
            self.pending_queue.put(rel_path)
        print(f"[INFO] 提交 {len(rel_paths)} 个算子到调度队列")

    def run(self) -> List[EvalOperatorResult]:
        """执行调度循环，返回所有结果"""
        total_operators = self.pending_queue.qsize()
        started_count = 0
        completed_count = 0

        print(f"[INFO] 开始动态调度，每卡最大并发: {self.max_concurrent_per_card}")

        while not self.pending_queue.empty() or self._has_running_tasks():
            # 1. 尝试启动新任务
            new_tasks = self._start_pending_tasks()
            for task in new_tasks:
                started_count += 1
                print(f"[INFO] [{started_count}/{total_operators}] Card {task.card_id}: 启动算子 {task.rel_path}")

            # 2. 检查完成的任务
            completed = self._collect_completed_tasks()
            for task in completed:
                completed_count += 1
                status = "✅" if task.result and task.result.get('passed_cases', 0) > 0 else "❌"
                passed = task.result.get('passed_cases', 0) if task.result else 0
                total = task.result.get('total_cases', 0) if task.result else 0
                print(f"[INFO] [{completed_count}/{total_operators}] Card {task.card_id}: 算子 {task.rel_path} 完成 {status} ({passed}/{total})")

            # 3. 短暂等待，避免空转
            if self._has_running_tasks() and self.pending_queue.empty():
                time.sleep(0.2)  # 等待运行中的任务
            else:
                time.sleep(0.05)

        print(f"[INFO] 调度完成: {completed_count}/{total_operators} 个算子")
        return [EvalOperatorResult.from_dict(d) for d in self.completed_results]

    def _has_running_tasks(self) -> bool:
        """检查是否有运行中的任务"""
        with self.lock:
            return any(len(tasks) > 0 for tasks in self.running_tasks.values())

    def _get_available_card(self) -> Optional[int]:
        """找到有空闲槽位的卡"""
        with self.lock:
            for card_id in range(self.card_count):
                if len(self.running_tasks[card_id]) < self.max_concurrent_per_card:
                    return card_id
            return None

    def _start_pending_tasks(self) -> List[OperatorTask]:
        """启动待执行的算子"""
        started_tasks = []

        while not self.pending_queue.empty():
            card_id = self._get_available_card()
            if card_id is None:
                break  # 所有卡都满了

            try:
                rel_path = self.pending_queue.get_nowait()
            except QueueEmpty:
                # F043: 旧版 bare `except:` 会吞 SystemExit / KeyboardInterrupt，
                # 长跑评测中 Ctrl+C 无反应、进程池僵尸化。精确捕获 Empty 即可。
                break

            task = self._create_task(rel_path, card_id)

            with self.lock:
                self.running_tasks[card_id].append(task)

            started_tasks.append(task)

        return started_tasks

    def _create_task(self, rel_path: str, card_id: int) -> OperatorTask:
        """创建并启动算子进程"""
        # 创建输出文件
        fd, output_file = tempfile.mkstemp(
            suffix=".json",
            prefix=f"op_{rel_path.replace('/', '_')}_",
        )
        os.close(fd)

        # 构建命令
        kernel_eval_root = str(get_project_root() / "src")
        # 使用 card_id 作为 process-id（整数）
        cmd = [
            sys.executable, "-m", "kernel_eval.cli", "eval-process",
            "--process-id", str(card_id),
            "--card-id", str(card_id),
            "--output", output_file,
            "--warmup", str(self.base_config.warmup),
            "--repeat", str(self.base_config.repeat),
            "--bench-name", self.base_config.bench_name,
            "--reports-dir", self.base_config.reports_dir,
            "--rel-paths", rel_path,  # 单个算子
        ]

        if self.process_config.enable_profiler:
            cmd.append("--enable-profiler")

        # 设置环境变量
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{kernel_eval_root}:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = kernel_eval_root
        env["TASKS_ROOT"] = self.base_config.tasks_root
        env["PYTHONUNBUFFERED"] = "1"

        # 继承 CANN 环境变量（日志抑制）
        cann_env_vars = [
            "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
            "ASCEND_AICPU_PATH", "ASCEND_VISIBLE_DEVICES",
            "TBE_IMPL_PATH",
        ]
        for var in cann_env_vars:
            if var in os.environ:
                env[var] = os.environ[var]

        # 强制设置日志抑制环境变量（确保子进程继承）
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # ERROR level

        # 启动进程
        # start_new_session=True 创建新的进程组，便于清理 profiler fork 子进程
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=None,  # 直接输出到终端
            stderr=None,
            text=True,
            start_new_session=True,  # 创建新进程组
        )

        return OperatorTask(
            rel_path=rel_path,
            card_id=card_id,
            process=process,
            output_file=output_file,
            started_at=time.time(),
            timeout=self.timeout_per_operator,
        )

    def _collect_completed_tasks(self) -> List[OperatorTask]:
        """检查并收集已完成的任务

        F033: kill-on-timeout used to call ``time.sleep(2)`` inside the
        ``with self.lock`` critical section, blocking other cards from
        scheduling new work for 2s on every timeout. Split the work:
        identify timed-out tasks under the lock, then release it and do
        the SIGTERM / sleep / SIGKILL / wait outside.
        """
        completed_tasks = []
        timed_out_tasks: List[OperatorTask] = []

        with self.lock:
            for card_id, tasks in list(self.running_tasks.items()):
                remaining_tasks = []

                for task in tasks:
                    if task.completed:
                        continue

                    elapsed = time.time() - task.started_at
                    if elapsed > task.timeout:
                        print(f"[WARN] Card {task.card_id}: 算子 {task.rel_path} 超时 ({task.timeout}s)")
                        task.completed = True
                        timed_out_tasks.append(task)
                        completed_tasks.append(task)
                        continue

                    # 检查进程状态
                    poll_result = task.process.poll()
                    if poll_result is not None:
                        # 进程已退出，读取结果
                        task.completed = True
                        task.result = self._read_result(task)
                        self.completed_results.append(task.result)
                        completed_tasks.append(task)
                        continue

                    # 进程仍在运行，保留
                    remaining_tasks.append(task)

                self.running_tasks[card_id] = remaining_tasks

        # Kill the timed-out tasks outside the lock so other cards can
        # keep scheduling. SIGTERM → 2s grace → SIGKILL → wait.
        for task in timed_out_tasks:
            self._kill_timed_out_task(task)

        return completed_tasks

    def _kill_timed_out_task(self, task: OperatorTask) -> None:
        """SIGTERM grace → SIGKILL — process group cleanup for timed-out task."""
        try:
            import signal
            pgid = os.getpgid(task.process.pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(2)
            try:
                os.killpg(pgid, 0)
                os.killpg(pgid, signal.SIGKILL)
            except OSError:
                pass
        except OSError:
            task.process.kill()
        try:
            task.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            task.process.kill()

    def _read_result(self, task: OperatorTask) -> Dict:
        """读取任务结果文件

        F031: tempfile cleanup used to live inside the success branch only.
        On JSONDecodeError / generic exceptions the file stayed in /tmp,
        and a batch run with many failures would fill the disk with
        `cannbench_op_*.json` orphans. Use try/finally so cleanup runs
        on every exit path.
        """
        if not (task.output_file and os.path.exists(task.output_file)):
            return {}
        try:
            with open(task.output_file, 'r') as f:
                data = json.load(f)
            return data.get("results", [{}])[0] if data.get("results") else {}
        except json.JSONDecodeError:
            print(f"[WARN] Card {task.card_id}: 算子 {task.rel_path} 结果文件解析失败")
            return {}
        except Exception as e:
            print(f"[WARN] Card {task.card_id}: 算子 {task.rel_path} 读取结果失败: {e}")
            return {}
        finally:
            try:
                if task.output_file and os.path.exists(task.output_file):
                    os.unlink(task.output_file)
            except OSError:
                pass


class ProcessWorker:
    """单进程工作单元

    独立子进程，拥有独立的：
    - MultiProcessPool 单例（torch_npu.profiler）
    - device_id
    - 单线程执行器

    通过文件传递结果，无进程间通信。
    """

    def __init__(
        self,
        process_id: int,
        card_id: int,
        base_config: Config,
        process_config: ProcessConfig,
    ):
        self.process_id = process_id
        self.card_id = card_id
        self.base_config = base_config
        self.process_config = process_config

        self._process: Optional[subprocess.Popen] = None
        self._output_file: Optional[str] = None
        self._cases_file: Optional[str] = None
        self._started = False
        self._operator_count: int = 1  # 算子数量，用于动态计算超时

    def start(
        self,
        rel_paths: List[str] = None,
        cases: List[CaseSpec] = None,
    ):
        """启动子进程

        Args:
            rel_paths: 算子相对路径列表（rel_path_parallel 模式）
            cases: 用例列表（case_parallel 模式）
        """
        # 记录算子数量（用于动态计算超时）
        if rel_paths:
            self._operator_count = len(rel_paths)
        elif cases:
            self._operator_count = 1  # 单算子模式
        # 创建输出文件
        fd, self._output_file = tempfile.mkstemp(
            suffix=".json",
            prefix=f"proc{self.process_id}_",
        )
        os.close(fd)

        # 构建命令
        kernel_eval_root = str(get_project_root() / "src")
        cmd = [
            sys.executable, "-m", "kernel_eval.cli", "eval-process",
            "--process-id", str(self.process_id),
            "--card-id", str(self.card_id),
            "--output", self._output_file,
            "--warmup", str(self.base_config.warmup),
            "--repeat", str(self.base_config.repeat),
            "--bench-name", self.base_config.bench_name,
            "--reports-dir", self.base_config.reports_dir,
        ]

        # 添加 profiler 配置
        if self.process_config.enable_profiler:
            cmd.append("--enable-profiler")

        # 添加任务数据
        if rel_paths:
            # rel_path_parallel 模式：传递 rel_path 列表
            cmd.extend(["--rel-paths", ",".join(rel_paths)])

        elif cases:
            # case_parallel 模式：传递 case 数据文件
            fd, self._cases_file = tempfile.mkstemp(suffix=".json", prefix="cases_")
            os.close(fd)
            case_data = self._serialize_cases(cases)
            with open(self._cases_file, 'w') as f:
                json.dump(case_data, f)
            cmd.extend(["--cases-file", self._cases_file])

        # 设置环境变量
        env = os.environ.copy()
        # PYTHONPATH 需要追加，不能覆盖（保留父进程的 TBE 等模块路径）
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{kernel_eval_root}:{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = kernel_eval_root
        env["TASKS_ROOT"] = self.base_config.tasks_root

        # 强制无缓冲输出（stdout 是管道而非 TTY 时 Python 默认块缓冲）
        env["PYTHONUNBUFFERED"] = "1"

        # 继承关键的 CANN/Ascend 环境变量（确保子进程能正确访问 NPU）
        cann_env_vars = [
            "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASCEND_OPP_PATH",
            "ASCEND_AICPU_PATH", "ASCEND_VISIBLE_DEVICES",
            "TBE_IMPL_PATH",
        ]
        for var in cann_env_vars:
            if var in os.environ:
                env[var] = os.environ[var]

        # 强制设置日志抑制环境变量（确保子进程继承）
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"  # ERROR level

        # 启动子进程
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # 行缓冲，实时输出
        )
        self._started = True

    def _serialize_cases(self, cases: List[CaseSpec]) -> List[Dict]:
        """序列化用例数据"""
        return [c.to_dict() for c in cases]

    def wait(self, timeout: int = None) -> List[Dict]:
        """等待子进程完成并返回结果

        Args:
            timeout: 超时时间（秒），默认为 case_count × timeout_per_case

        Returns:
            结果数据列表（字典格式）
        """
        if not self._started or self._process is None:
            return []

        # 动态计算超时：算子数量 × 单算子超时时间
        if timeout is None:
            timeout = self._operator_count * self.process_config.timeout_per_operator

        start_time = time.time()
        deadline = start_time + timeout

        try:
            # 使用 select 实现非阻塞读取，确保能及时检测超时
            import select

            # F035: was an unbounded list — verbose / profiler-Level2
            # children can dump tens of thousands of lines, and the buffer
            # is only used (if at all) for tail context. Use a bounded
            # deque so peak memory stays small.
            stdout_lines: deque = deque(maxlen=10000)
            while True:
                # 计算剩余时间
                remaining = deadline - time.time()
                if remaining <= 0:
                    print(f"[WARN] Process {self.process_id} 超时 ({timeout}s)，已终止")
                    self._process.kill()
                    self._process.wait(timeout=5)
                    return []

                # 使用 select 检查 stdout 是否有数据可读
                # 设置超时为剩余时间，确保不会无限等待
                try:
                    readable, _, _ = select.select([self._process.stdout], [], [], min(remaining, 1.0))
                except (select.error, OSError):
                    # select 出错（如 stdout 已关闭），检查进程状态
                    poll_result = self._process.poll()
                    if poll_result is not None:
                        break
                    continue

                if readable:
                    line = self._process.stdout.readline()
                    if not line:
                        # EOF，进程已关闭 stdout
                        break
                    stdout_lines.append(line)
                    print(line.rstrip())
                else:
                    # 没有数据可读，检查进程是否已退出
                    poll_result = self._process.poll()
                    if poll_result is not None:
                        break

            # 进程已正常退出，等待并读取结果
            self._process.wait(timeout=5)

            # 读取结果文件
            if self._output_file and os.path.exists(self._output_file):
                with open(self._output_file, 'r') as f:
                    data = json.load(f)
                return data.get("results", [])

        except subprocess.TimeoutExpired:
            self._process.kill()
            print(f"[WARN] Process {self.process_id} 超时，已终止")
            return []
        except json.JSONDecodeError as e:
            print(f"[WARN] Process {self.process_id} 结果文件解析失败: {e}")
            return []
        finally:
            self._cleanup()

        return []

    def _cleanup(self):
        """清理临时文件"""
        if self._output_file and os.path.exists(self._output_file):
            try:
                os.unlink(self._output_file)
            except OSError:
                pass
        if self._cases_file and os.path.exists(self._cases_file):
            try:
                os.unlink(self._cases_file)
            except OSError:
                pass
        self._output_file = None
        self._cases_file = None

    def is_alive(self) -> bool:
        """检查进程是否仍在运行"""
        return self._process is not None and self._process.poll() is None

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'process_id': self.process_id,
            'card_id': self.card_id,
            'started': self._started,
            'alive': self.is_alive(),
        }


class ProcessPoolCoordinator:
    """进程池协调器

    管理 card_count × processes_per_card 个独立进程，
    分配任务并汇总结果。

    支持单卡和多卡模式：
    - 单卡模式：指定 device_id，所有进程绑定到该卡
    - 多卡模式：不指定 device_id，自动检测并轮询分配
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
        # 同一 NPU 卡上多进程并发 profile 会竞争该资源导致 "Failed to get acl to
        # npu flow events"，无法产出 kernel_details.csv（elapsed_us=0）。
        # profiling 开启时每卡仅 1 进程，保证 profiler 独占硬件资源。
        if self.process_config.enable_profiler:
            self.process_config.processes_per_card = 1

        self.total_processes = self.card_count * self.process_config.processes_per_card
        self.workers: List[ProcessWorker] = []

    def _detect_cards(self) -> int:
        """检测可用 NPU 卡数，并提供详细的诊断信息"""
        if self.base_config.device_type != "npu":
            return 0

        # 诊断步骤 1: 检查 torch_npu 是否可导入
        try:
            import torch_npu
        except ImportError as e:
            print("[ERROR] 无法导入 torch_npu 模块")
            print(f"  原因: {e}")
            print("  建议:")
            print("    1. 检查是否安装了 torch_npu: pip list | grep torch_npu")
            print("    2. 确认 CANN 环境已正确配置")
            print("    3. 检查 ASCEND_HOME_PATH 等环境变量是否设置")
            # 尝试调用 npu-smi info 检查硬件状态
            self._check_npu_smi()
            return 0

        # 诊断步骤 2: 检查 NPU 是否可用
        if not torch.npu.is_available():
            print("[ERROR] torch.npu.is_available() 返回 False")
            print("  建议:")
            print("    1. 检查 NPU 驱动是否已安装")
            print("    2. 检查 ASCEND_VISIBLE_DEVICES 环境变量是否正确设置")
            print("    3. 确认当前用户有 NPU 设备访问权限")
            # 尝试调用 npu-smi info 检查硬件状态
            self._check_npu_smi()
            return 0

        # 诊断步骤 3: 检查 NPU 卡数
        card_count = torch.npu.device_count()
        if card_count == 0:
            print("[ERROR] torch.npu.device_count() 返回 0")
            print("  建议:")
            print("    1. 检查是否有设备被 ASCEND_VISIBLE_DEVICES 过滤")
            # 尝试调用 npu-smi info 检查硬件状态
            self._check_npu_smi()
            return 0

        # 成功检测到 NPU
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
        import subprocess
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                print("\n[诊断] npu-smi info 输出:")
                # 只打印关键信息，避免输出过长
                lines = result.stdout.strip().split('\n')
                # 打印前 20 行（通常包含设备列表）
                for line in lines[:20]:
                    print(f"  {line}")
                if len(lines) > 20:
                    print(f"  ... (共 {len(lines)} 行，已截断)")
            elif result.returncode != 0:
                print("\n[诊断] npu-smi info 执行失败:")
                print(f"  错误: {result.stderr.strip() if result.stderr else '未知错误'}")
                print("  可能原因:")
                print("    1. npu-smi 命令未安装或不在 PATH 中")
                print("    2. NPU 驱动未正确安装")
                print("    3. 当前用户无权限执行该命令")
        except FileNotFoundError:
            print("\n[诊断] 未找到 npu-smi 命令")
            print("  建议: 检查 NPU 驱动是否已安装")
        except subprocess.TimeoutExpired:
            print("\n[诊断] npu-smi info 执行超时")
        except Exception as e:
            print(f"\n[诊断] npu-smi info 执行异常: {e}")

    def _create_workers(self) -> List[ProcessWorker]:
        """创建进程工作单元"""
        workers = []
        for i in range(self.total_processes):
            if self.device_id is not None:
                # 单卡模式：所有进程绑定到指定卡
                card_id = self.device_id
            else:
                # 多卡模式：轮询分配到各卡
                card_id = i // self.process_config.processes_per_card
            worker = ProcessWorker(
                process_id=i,
                card_id=card_id,
                base_config=self.base_config,
                process_config=self.process_config,
            )
            workers.append(worker)
        return workers

    def distribute_rel_paths(
        self,
        rel_paths: List[str],
    ) -> Dict[int, List[str]]:
        """分配 rel_paths 到进程池

        Args:
            rel_paths: 算子相对路径列表

        Returns:
            {process_id: [rel_paths]} 分配映射
        """
        distribution = defaultdict(list)
        for i, rel_path in enumerate(rel_paths):
            process_id = i % self.total_processes
            distribution[process_id].append(rel_path)
        return dict(distribution)

    def distribute_cases(
        self,
        cases: List[CaseSpec],
    ) -> Dict[int, List[CaseSpec]]:
        """分配 cases 到进程池

        Args:
            cases: 用例列表

        Returns:
            {process_id: [cases]} 分配映射
        """
        distribution = defaultdict(list)
        for i, case in enumerate(cases):
            process_id = i % self.total_processes
            distribution[process_id].append(case)
        return dict(distribution)

    def evaluate_operators(
        self,
        rel_paths: List[str],
        progress_callback: callable = None,
    ) -> List[EvalOperatorResult]:
        """并行评测多个算子

        自动选择并行策略：
        - 多算子：使用 OperatorScheduler 动态调度（每算子一个进程）
        - 单算子：按用例分配到进程（case_parallel）

        Args:
            rel_paths: 算子相对路径列表
            progress_callback: 进度回调（未实现）

        Returns:
            算子评测结果列表
        """
        if self.card_count == 0:
            # F120: 旧版仅 WARN + 返回空列表 → 多卡评测静默降级为不评测，
            # 前端看到 "0 算子通过" 无法分辨是设备故障还是算子集为空。
            # 改为抛 RuntimeError 阻断 — 调用方可显式 catch 走 CPU/单卡 fallback。
            # ALLOW_NO_NPU_CARDS=1 提供 escape hatch 给只想 dry-run 的场景。
            if os.environ.get("ALLOW_NO_NPU_CARDS") == "1":
                print(
                    "[WARN] 无可用 NPU 卡 (ALLOW_NO_NPU_CARDS=1 — 评测返回空结果)",
                    flush=True,
                )
                return []
            raise RuntimeError(
                "[ERROR] 无可用 NPU 卡 (card_count=0)。多卡评测需要至少 1 张 NPU。"
                "如确需在无 NPU 环境跑空评测做 dry-run，设置 ALLOW_NO_NPU_CARDS=1。"
            )

        print(f"[INFO] 配置: {self.card_count} 卡 × {self.process_config.processes_per_card} 并发/卡")
        print(f"[INFO] 单算子超时: {self.process_config.timeout_per_operator}s")

        # 根据 rel_path 数量选择并行策略
        if len(rel_paths) == 1:
            # 单算子：使用 case_parallel 模式
            rel_path = rel_paths[0]
            print(f"[INFO] 单算子模式，按用例分配到 {self.total_processes} 个进程")
            result = self.evaluate_cases_parallel(rel_path)
            return [result] if result else []
        else:
            # 多算子：使用 OperatorScheduler 动态调度
            return self._evaluate_with_scheduler(rel_paths)

    def _evaluate_with_scheduler(
        self,
        rel_paths: List[str],
    ) -> List[EvalOperatorResult]:
        """使用 OperatorScheduler 动态调度多算子"""
        scheduler = OperatorScheduler(
            process_config=self.process_config,
            base_config=self.base_config,
            card_count=self.card_count,
        )
        scheduler.submit_operators(rel_paths)
        return scheduler.run()

    def _collect_worker_results(self) -> List[EvalCaseResult]:
        """从所有已启动 worker 收集并解析结果"""
        all_case_results = []
        started = [w for w in self.workers if w._started]
        with ThreadPoolExecutor(max_workers=max(len(started), 1)) as executor:
            futures = {executor.submit(worker.wait): worker for worker in started}
            for future in as_completed(futures):
                for data in future.result():
                    if 'results' in data:
                        for case_data in data['results']:
                            all_case_results.append(EvalCaseResult.from_dict(case_data))
                    else:
                        all_case_results.append(EvalCaseResult.from_dict(data))
        return all_case_results

    def _build_operator_result(
        self, operator_name: str, rel_path: str,
        total_cases: int, all_case_results: List[EvalCaseResult],
    ) -> EvalOperatorResult:
        """汇总 case 结果并构造 EvalOperatorResult"""
        summary = summarize_case_results(all_case_results)
        return EvalOperatorResult(
            rel_path=rel_path,
            operator=operator_name,
            total_cases=total_cases,
            passed_cases=summary.passed,
            failed_cases=summary.failed,
            skipped_cases=summary.skipped,
            results=all_case_results,
            pass_rate=summary.pass_rate,
            avg_speedup=summary.avg_speedup,
        )

    def evaluate_cases_parallel(
        self,
        rel_path: str,
    ) -> Optional[EvalOperatorResult]:
        """单算子多进程并行评测（case_parallel 模式）

        将单个算子的用例分配到多个进程并行执行。

        Args:
            rel_path: 算子相对路径

        Returns:
            算子评测结果（合并所有进程结果）
        """
        # 加载用例
        from ..registry.loader_registry import get_case_loader
        loader = get_case_loader(self.base_config.bench_name, tasks_root=self.base_config.tasks_root)
        cases = loader.scan_by_rel_path(rel_path)

        if not cases:
            print(f"[WARN] 算子 {rel_path} 无用例")
            return EvalOperatorResult(
                rel_path=rel_path,
                operator=Path(rel_path).name,  # fallback to dir name
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                skipped_cases=0,
                results=[],
                pass_rate=0.0,
                avg_speedup=0.0,
            )

        operator_name = cases[0].operator
        # 当 rel_path == "." 时，使用 op_dir_name 显示
        op_dir_name = cases[0].metadata.get('op_dir_name', '')
        display_path = op_dir_name if op_dir_name and rel_path == "." else rel_path
        print(f"[INFO] 算子 {display_path} ({operator_name}), 用例数: {len(cases)}")

        # 分配用例到进程
        distribution = self.distribute_cases(cases)

        print(f"[INFO] 任务分配 (case_parallel):")
        for proc_id, proc_cases in distribution.items():
            card_id = proc_id // self.process_config.processes_per_card
            print(f"  Process {proc_id} (Card {card_id}): {len(proc_cases)} 用例")

        # 创建并启动进程
        self.workers = self._create_workers()
        for worker in self.workers:
            proc_id = worker.process_id
            if proc_id in distribution and distribution[proc_id]:
                worker.start(cases=distribution[proc_id])

        # 等待并收集结果
        all_case_results = self._collect_worker_results()
        return self._build_operator_result(operator_name, rel_path, len(cases), all_case_results)

    def evaluate_cases(
        self,
        cases: List[CaseSpec],
        rel_path: str,
        progress_callback: callable = None,
    ) -> EvalOperatorResult:
        """并行评测单个算子的多个用例

        Args:
            cases: 用例列表
            rel_path: 算子相对路径
            progress_callback: 进度回调（未实现）

        Returns:
            算子评测结果（合并所有进程结果）
        """
        operator_name = cases[0].operator if cases else Path(rel_path).name

        if self.card_count == 0:
            print("[WARN] 无可用 NPU 卡")
            return EvalOperatorResult(
                rel_path=rel_path,
                operator=operator_name,
                total_cases=len(cases),
                passed_cases=0,
                failed_cases=len(cases),
                skipped_cases=0,
                results=[],
                pass_rate=0.0,
                avg_speedup=0.0,
            )

        print(f"[INFO] 使用 {self.total_processes} 个进程池并行评测")
        print(f"[INFO] 配置: {self.card_count} 卡 × {self.process_config.processes_per_card} 进程/卡")

        # 分配任务
        distribution = self.distribute_cases(cases)

        print(f"[INFO] 任务分配 (case_parallel):")
        for proc_id, proc_cases in distribution.items():
            card_id = proc_id // self.process_config.processes_per_card
            print(f"  Process {proc_id} (Card {card_id}): {len(proc_cases)} 用例")

        # 创建并启动进程
        self.workers = self._create_workers()
        for worker in self.workers:
            proc_id = worker.process_id
            if proc_id in distribution and distribution[proc_id]:
                worker.start(cases=distribution[proc_id])

        # 等待并收集结果
        all_case_results = self._collect_worker_results()
        return self._build_operator_result(operator_name, rel_path, len(cases), all_case_results)

    def shutdown(self):
        """关闭所有进程

        F034: was sending SIGKILL straight away. Children never got a
        chance to run their `finally` blocks, so torch_npu profiler fork
        children were orphaned and held NPU device context — the next
        evaluation could fail with "device is in use" / Bus error. Send
        SIGTERM first, give a grace window, then SIGKILL anything that
        is still alive.
        """
        grace_sec = 5
        for worker in self.workers:
            if worker.is_alive():
                try:
                    worker._process.terminate()  # SIGTERM
                except Exception:
                    pass

        if self.workers:
            deadline = time.time() + grace_sec
            for worker in self.workers:
                remaining = max(deadline - time.time(), 0)
                if remaining <= 0 or not worker.is_alive():
                    continue
                try:
                    worker._process.wait(timeout=remaining)
                except subprocess.TimeoutExpired:
                    pass

        for worker in self.workers:
            if worker.is_alive():
                try:
                    worker._process.kill()
                except Exception:
                    pass
        self.workers = []

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'device_id': self.device_id,
            'card_count': self.card_count,
            'processes_per_card': self.process_config.processes_per_card,
            'total_processes': self.total_processes,
            'workers': [w.get_stats() for w in self.workers],
        }