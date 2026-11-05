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
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .results import (
    EvalOperatorResult,
    EvalCaseResult,
    summarize_case_results,
    dedup_case_results,
)
from .subprocess_utils import (
    _CANN_ENV_VARS,
    _write_oom_score_adj,
    _is_oom_killed,
    _synthesize_failure_cases,
    _try_recover_partial_results,
    _signal_process_group,
    _terminate_process_group,
)
from ..config import Config, get_config, get_project_root
from ..base.models import CaseSpec
from ..base.result import is_precision_failure_type


# ---------------------------------------------------------------------------
# 子进程崩溃诊断
# ---------------------------------------------------------------------------

_SIGNAL_NAMES = {
    -11: "SIGSEGV",
    -6: "SIGABRT",
    -7: "SIGBUS",
    -9: "SIGKILL",
    -15: "SIGTERM",
    -4: "SIGILL",
    -8: "SIGFPE",
}


def _extract_crash_diag(rc: int, stdout: str, stderr: str) -> str:
    """构建子进程崩溃诊断字符串，附加到 error_msg 尾部。

    rc < 0 时为信号死亡（如 SIGSEGV=-11），提取 stderr 末尾行帮助定位。
    返回的字符串以 " | " 前导，可直接拼接到 error_msg 后；无诊断时返回空串。
    """
    parts: List[str] = []
    if rc < 0:
        sig_name = _SIGNAL_NAMES.get(rc, f"signal {-rc}")
        parts.append(f"signal={sig_name}({rc})")
    tail_lines = (stderr or "").strip().splitlines()
    if not tail_lines:
        tail_lines = (stdout or "").strip().splitlines()
    # 取最后 5 行非空行作为诊断摘要
    tail = [ln.strip() for ln in tail_lines if ln.strip()][-5:]
    if tail:
        parts.append("stderr_tail=" + " | ".join(tail))
    if not parts:
        return ""
    return " | " + " | ".join(parts)


def _forward_child_output(stdout: str, stderr: str) -> None:
    """把捕获的子进程输出透传到主进程日志。

    子进程 stdout/stderr 经 PIPE 捕获后若只用于崩溃诊断，staged.log 会丢失
    全部用例级细节（如 [i/n] 用例进度），与历史上管道直连的行为不一致。
    子进程退出后按原内容整体透传，恢复可观测性；不同 TaskUnit 的输出因此
    按任务整块呈现，不再逐行交叠。
    """
    if stdout and stdout.strip():
        print(stdout.rstrip(), flush=True)
    if stderr and stderr.strip():
        print(stderr.rstrip(), file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass
class ProcessConfig:
    """进程池配置"""

    processes_per_card: int = 2  # 每卡最大并发进程数
    timeout_per_operator: int = 300  # 单算子超时（秒）
    enable_profiler: bool = True  # 是否启用 profiler
    max_retries: int = 1  # 最大重试次数
    retry_on_timeout: bool = True  # 超时是否重试
    retry_on_oom: bool = True  # OOM 是否重试
    retry_on_failure: bool = True  # 子进程异常是否重试
    exclude_repeatedly_failed_cases: bool = True  # 排除多次失败的case


@dataclass
class TaskUnit:
    """统一任务单元 = (算子, 用例组, device_id)"""

    operator: str  # 算子名称
    rel_path: str  # 算子相对路径
    cases: List[CaseSpec]  # 该进程需要跑的用例列表
    device_id: int  # 分配的 NPU 卡 ID
    retry_count: int = 0  # 重试次数
    excluded_devices: set = None  # 排除的设备ID集合
    parent_task_id: str = None  # 父任务ID（用于追踪）
    # None 跟随 base_config；False 用于批量 profiler OOM 后强制逐 case。
    perf_batch_cases: Optional[bool] = None

    def __post_init__(self):
        if self.excluded_devices is None:
            self.excluded_devices = set()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def split_into_chunks(items: list, n: int) -> list:
    """将列表分成 n 个尽量均匀的块"""
    if n <= 0:
        return [items]
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def build_task_units(
    cases_by_operator: Dict[str, List[CaseSpec]],
    card_count: int,
    isolate_each_case: bool = False,
    max_cases_per_task_unit: int = 64,
) -> List[TaskUnit]:
    """将算子×用例拆分为 TaskUnit，均分到各卡。

    每个算子的用例按卡数分组，形成 TaskUnit 列表。
    多算子场景：算子A的用例分到卡0-7，算子B的用例也分到卡0-7 → 自然负载均衡。
    单算子场景：用例分到卡0-7 → 单算子多卡并行。
    单卡场景：按 max_cases_per_task_unit 继续拆分，以限制单个子进程
    持续执行大量 case 时 native host 内存池的累积。
    """
    if card_count <= 0:
        return []
    if max_cases_per_task_unit <= 0:
        raise ValueError("max_cases_per_task_unit must be positive")

    task_units: List[TaskUnit] = []
    card_ids = list(range(card_count))

    for operator_name, cases in cases_by_operator.items():
        if isolate_each_case:
            for i, case in enumerate(cases):
                task_units.append(
                    TaskUnit(
                        operator=operator_name,
                        rel_path=case.rel_path,
                        cases=[case],
                        device_id=card_ids[i % len(card_ids)],
                    )
                )
            continue

        card_chunks = split_into_chunks(cases, card_count)
        for card_index, card_chunk in enumerate(card_chunks):
            for offset in range(0, len(card_chunk), max_cases_per_task_unit):
                chunk = card_chunk[offset : offset + max_cases_per_task_unit]
                task_units.append(
                    TaskUnit(
                        operator=operator_name,
                        rel_path=chunk[0].rel_path,
                        cases=chunk,
                        device_id=card_ids[card_index],
                    )
                )

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
        # 去重：子进程崩溃重试时，同一 case_id 会同时存在合成桩与真实结果，
        # 保留有 accuracy_result 的真实记录，丢弃桩，避免 total_cases 虚高。
        case_results = dedup_case_results(case_results)
        summary = summarize_case_results(case_results)
        rel_path = case_results[0].rel_path if case_results else ""
        results.append(
            EvalOperatorResult(
                rel_path=rel_path,
                operator=op_name,
                total_cases=len(case_results),
                passed_cases=summary.passed,
                failed_cases=summary.failed,
                skipped_cases=summary.skipped,
                results=case_results,
                pass_rate=summary.pass_rate,
                avg_speedup=summary.avg_speedup,
            )
        )

    return results


# ---------------------------------------------------------------------------
# _DevicePool — 设备池，动态卡分配
# ---------------------------------------------------------------------------


class _DevicePool:
    """设备池 — 动态卡分配，每卡含 slots_per_card 个槽位。

    仅在主调度线程中调用 acquire/release（worker 线程不接触池），
    因此无需加锁。
    """

    def __init__(self, cards: List[int], slots_per_card: int):
        self._pool: deque = deque()
        for card in cards:
            for _ in range(slots_per_card):
                self._pool.append(card)
        self._total = len(self._pool)

    def acquire(self, exclude: Optional[set] = None) -> Optional[int]:
        """取一张可用卡。可指定排除集（排除集中的卡不被选中）。

        全部被排除时返回 None。
        """
        if exclude is None:
            exclude = set()
        for _ in range(len(self._pool)):
            card = self._pool.popleft()
            if card not in exclude:
                return card
            self._pool.append(card)
        return None

    def release(self, card_id: int):
        """归还卡到池中"""
        self._pool.append(card_id)

    def has_available(self) -> bool:
        return len(self._pool) > 0

    def total_slots(self) -> int:
        return self._total


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
            self._available_cards = [device_id]
        else:
            # 多卡模式：自动检测，过滤掉不健康和忙碌的卡
            self.card_count = self._detect_cards()
            # _detect_cards 内部设置 self._available_cards；
            # 若被 mock（单测）未设置，则回退到全卡
            if not getattr(self, "_available_cards", None):
                self._available_cards = list(range(self.card_count))

        # torch_npu.profiler 使用 ACL 设备级 profiling 硬件资源，
        # 同一 NPU 卡上多进程并发 profile 会竞争该资源。
        # profiling 开启时每卡仅 1 进程，保证 profiler 独占硬件资源。
        if self.process_config.enable_profiler:
            self.process_config.processes_per_card = 1

        self.total_processes = self.card_count * self.process_config.processes_per_card
        # 记录活跃子进程，用于 shutdown 时清理
        self._active_processes: List[subprocess.Popen] = []

    def _detect_cards(self) -> int:
        """检测可用 NPU 卡数，并提供详细的诊断信息

        通过 npu-smi info 检测每张卡的健康状态，过滤掉 Alarm/异常卡。
        """
        if self.base_config.device_type != "npu":
            return 0

        try:
            import torch_npu  # noqa: F401 -- register torch.npu before checking device availability
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

        # 检测卡健康状态，过滤掉 Alarm 卡
        healthy_cards = self._filter_healthy_cards(card_count)

        print(f"[INFO] 检测到 {card_count} 张 NPU 卡, {len(healthy_cards)} 张健康")
        for i in healthy_cards:
            try:
                name = (
                    torch.npu.get_device_name(i)
                    if hasattr(torch.npu, "get_device_name")
                    else "unknown"
                )
                print(f"  NPU:{i} - {name}")
            except Exception:
                print(f"  NPU:{i}")

        if not healthy_cards:
            print("[ERROR] 没有健康的 NPU 卡可用")
            self._available_cards = []
            return 0

        self._available_cards = healthy_cards
        return len(healthy_cards)

    # npu-smi 的 health 列位置随驱动/CANN 版本变化（有时 id+name 合并、有时 health
    # 不在 parts[3]）。fail-open：仅当某行**明确**出现坏状态词且能解析出 npu_id 时才跳过；
    # 解析不出或列错位时一律保留，避免把健康卡误杀导致整 eval 无可用卡（见 ST 0-case 故障）。
    _NPU_BAD_HEALTH = ("ALARM", "CRITICAL", "WARNING", "ABNORMAL", "FAULT", "ERROR")

    def _filter_healthy_cards(self, card_count: int) -> List[int]:
        """通过 npu-smi info 检测卡健康状态，返回健康卡的 ID 列表（fail-open）。

        列布局不固定，故扫描整行的 `|` 分隔 cell：任一 cell 命中坏状态词才视为异常行，
        再从行内取首个可解析为 int 的 cell 作为 npu_id。无法对齐/解析时保留该卡。
        """
        healthy = list(range(card_count))
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return healthy

            for line in result.stdout.strip().split("\n"):
                if "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                # 行内任一 cell 是坏状态词才考虑跳过；否则保留（fail-open）
                if not any(p.upper() in self._NPU_BAD_HEALTH for p in parts):
                    continue
                # npu-smi 数据行：首个 cell 的首 token 是 NPU ID
                # （可能单独 "0" 或与 name 合并 "1 Ascend910"）
                first_cell = parts[1] if len(parts) > 1 else ""
                id_tokens = first_cell.split()
                if not id_tokens:
                    continue
                try:
                    npu_id = int(id_tokens[0])
                except ValueError:
                    continue
                if npu_id < card_count and npu_id in healthy:
                    print(f"[WARN] NPU:{npu_id} 状态异常，跳过该卡")
                    healthy.remove(npu_id)
        except FileNotFoundError:
            pass
        except Exception:
            pass
        return healthy

    def _check_npu_smi(self):
        """调用 npu-smi info 检查硬件状态"""
        try:
            result = subprocess.run(
                ["npu-smi", "info"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                print("\n[诊断] npu-smi info 输出:")
                lines = result.stdout.strip().split("\n")
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

    def _build_device_mapping(self) -> Dict[int, tuple]:
        """解析 npu-smi info -m，构建 {logic_id: (npu_id, chip_id)} 映射。

        fail-open：解析失败时返回空 dict，调用方按逻辑ID直查（退化为不检测）。
        """
        mapping = {}
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-m"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return mapping
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                # 格式: NPU_ID  Chip_ID  Chip_Logic_ID  Phy_ID  Name
                if len(parts) >= 4 and parts[2] != "-":
                    try:
                        logic_id = int(parts[2])
                        npu_id = int(parts[0])
                        chip_id = int(parts[1])
                        mapping[logic_id] = (npu_id, chip_id)
                    except ValueError:
                        continue
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return mapping

    def _build_env(self) -> Dict[str, str]:
        """构建子进程环境变量"""
        kernel_eval_root = str(get_project_root() / "src")
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        if kernel_eval_root not in existing_pythonpath:
            env["PYTHONPATH"] = (
                f"{kernel_eval_root}:{existing_pythonpath}"
                if existing_pythonpath
                else kernel_eval_root
            )
        env["PYTHONUNBUFFERED"] = "1"
        env["TASKS_ROOT"] = self.base_config.tasks_root
        env["ASCEND_SLOG_PRINT_TO_STDOUT"] = "0"
        env["ASCEND_GLOBAL_LOG_LEVEL"] = "3"

        for var in _CANN_ENV_VARS:
            if var in os.environ:
                env[var] = os.environ[var]

        return env

    def _build_env_for_task(
        self, base_env: Dict[str, str], task: TaskUnit
    ) -> Dict[str, str]:
        """构建子进程的环境变量

        F745: 为每个子进程设置独立的 ASCEND_RT_VISIBLE_DEVICES，
        让每个进程只能看到分配给它的那张卡，从而彻底隔离多卡之间的资源竞争。

        关键语义：task.device_id 是**相对父进程可见集的逻辑索引**（0..card_count-1），
        因为 card_count 来自 torch.npu.device_count()，已受父进程可见性约束。
        ASCEND_RT_VISIBLE_DEVICES 接受的也是父进程可见空间内的相对索引，而非全局
        物理卡号——因此这里直接写 task.device_id 即可，无需做物理号映射。

        （曾误改为"逻辑→物理号映射"，在父进程用 ASCEND_VISIBLE_DEVICES=12,13 收窄时
        会写出 ASCEND_RT_VISIBLE_DEVICES=12 导致子进程 device_count=0、set_device 失败。）
        """
        env = base_env.copy()
        if any((case.attrs or {}).get("mc2_distributed") for case in task.cases):
            # MC2 tasks run exclusively below and need the entire parent visible
            # set. Never apply the ordinary one-device worker isolation here.
            if env.get("BENCH_DEVICE_VISIBILITY"):
                env["ASCEND_RT_VISIBLE_DEVICES"] = env["BENCH_DEVICE_VISIBILITY"]
            return env

        # 让每个子进程只看到分配给它的那张卡。
        # task.device_id 是相对父进程可见集的逻辑索引，本应直接写即可。
        # 但 benchsite-runner 的父进程 (eval-child) 中 torch_npu 初始化会
        # 原地改写 os.environ['ASCEND_RT_VISIBLE_DEVICES']，导致 base_env
        # 丢失了原始的物理可见集。此时 task.device_id=0 会被 CANN 解释为
        # "物理 chip 0" 而非 "可见集内第 0 个设备"。
        #
        # 通过 BENCH_DEVICE_VISIBILITY（benchsite-runner 设置的非 ASCEND_
        # 前缀安全备份）还原物理可见集，再把 task.device_id 映射为正确的物理 chip。
        #
        # 判据是**谁做的收窄**，而不是"有没有可见集变量"：
        # - ASCEND_RT_VISIBLE_DEVICES 逐进程生效、取值是物理号，子进程重设它时
        #   必须同样给物理号 —— 故这里要映射。BENCH_DEVICE_VISIBILITY 是它被
        #   torch_npu 原地覆盖前的备份，优先用。
        # - ASCEND_VISIBLE_DEVICES 是容器/k8s 设备插件施加的，进容器时设备空间
        #   **已经**被收窄并重编号，不能再映射一次。k8s 只分物理 1 号卡时父进程
        #   device_count()==1、可见集是 {0}，把逻辑 0 映射回 "1" 等于在 {0} 里挑
        #   1 号 —— 挑空，子进程 device_count()==0：
        #     ChgUserDevIdToDeviceId failed ... userDevId is invalid. Expected value: [0, 0)
        #     RuntimeError: ... aclInit, error code is 107001
        #   分到 0 号卡时映射恰好等于相对索引，故此坑只在非 0 号卡上现形。
        vis = env.get("BENCH_DEVICE_VISIBILITY", "") or env.get(
            "ASCEND_RT_VISIBLE_DEVICES", ""
        )
        if vis:
            chips = [c.strip() for c in vis.split(",") if c.strip()]
            if task.device_id < len(chips):
                env["ASCEND_RT_VISIBLE_DEVICES"] = chips[task.device_id]
            else:
                env["ASCEND_RT_VISIBLE_DEVICES"] = str(task.device_id)
        else:
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(task.device_id)

        return env

    def _select_device_for_retry(
        self, original_device: int, excluded_devices: set, healthy_cards: List[int]
    ) -> int:
        """为重试任务选择设备，优先选择未出问题的卡

        Args:
            original_device: 原始失败的设备ID
            excluded_devices: 已排除的设备ID集合
            healthy_cards: 健康的卡ID列表

        Returns:
            选择的设备ID
        """
        # 优先选择从未失败过的卡
        available = [d for d in healthy_cards if d not in excluded_devices]
        if available:
            # 轮询选择（简单的负载均衡）
            return available[0]

        # 所有卡都有问题，重用原设备（可能是偶发问题）
        return original_device

    def _build_child_cmd(
        self, task: TaskUnit, cases_file: str, output_file: str
    ) -> List[str]:
        """构建 eval-child 子进程命令"""
        outer_isolation = bool(
            getattr(self.base_config, "pypto_pro_outer_case_isolation", False)
        )

        def child_path(path: str) -> str:
            return str(Path(path).resolve()) if outer_isolation else str(path)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "kernel_eval.cli",
            "eval-child",
            "--bench-name",
            self.base_config.bench_name,
            "--device-id",
            str(task.device_id),
            "--cases-file",
            cases_file,
            "--output",
            output_file,
            "--warmup",
            str(self.base_config.warmup),
            "--repeat",
            str(self.base_config.repeat),
        ]

        reports_dir = getattr(self.base_config, "reports_dir", "") or ""
        if reports_dir:
            cmd += ["--reports-dir", child_path(reports_dir)]

        # task-dir 透传
        tasks_root = getattr(self.base_config, "tasks_root", "")
        if tasks_root:
            cmd += ["--task-dir", child_path(tasks_root)]

        # source-dir 透传（Stanford bench 等需要在子进程中加载 ai_op.py）
        source_dir = getattr(self.base_config, "source_dir", "") or ""
        if source_dir:
            cmd += ["--source-dir", child_path(source_dir)]

        # profiler 配置
        if not self.process_config.enable_profiler:
            cmd.append("--no-perf")
        batch_cases = task.perf_batch_cases
        if batch_cases is None:
            batch_cases = getattr(self.base_config, "perf_batch_cases", True)
        if outer_isolation:
            batch_cases = False
        if batch_cases:
            cmd.append("--perf-batch-cases")
        else:
            cmd.append("--no-perf-batch-cases")
        if getattr(self.base_config, "monitor_case_memory", False):
            cmd.append("--monitor-case-memory")
        memory_limit_mb = getattr(
            self.base_config, "perf_batch_case_memory_limit_mb", None
        )
        if memory_limit_mb is not None:
            cmd += ["--perf-batch-case-memory-limit-mb", str(memory_limit_mb)]
        profiler_level = getattr(self.base_config, "profiler_level", None)
        if profiler_level:
            cmd += ["--profiler-level", str(profiler_level)]
        if not getattr(self.base_config, "perf_freq_boost", True):
            cmd.append("--no-freq-boost")

        # torch op guard 模式
        torch_op_guard_mode = getattr(self.base_config, "torch_op_guard_mode", None)
        if torch_op_guard_mode:
            cmd += ["--torch-op-guard-mode", str(torch_op_guard_mode)]

        # eval seed
        eval_seed = getattr(self.base_config, "eval_seed", None)
        if eval_seed is not None:
            cmd += ["--eval-seed", str(eval_seed)]

        cache_mode = getattr(self.base_config, "tensor_cache_mode", "off")
        if cache_mode != "off":
            cmd += [
                "--tensor-cache-mode",
                cache_mode,
                "--tensor-cache-dir",
                str(Path(self.base_config.tensor_cache_dir).expanduser().resolve()),
            ]
            for task_path in self.base_config.tensor_cache_tasks:
                cmd += ["--tensor-cache-task", task_path]

        # 输入分布：非默认值才透传，保持默认命令行不变
        input_dist = getattr(self.base_config, "input_dist", "uniform") or "uniform"
        if input_dist != "uniform":
            cmd += ["--input-dist", str(input_dist)]

        # perf-metric-strategy (trace_view / kernel_details / msprof_summary)
        perf_strategy = getattr(self.base_config, "perf_metric_strategy_override", None)
        if perf_strategy:
            cmd += ["--perf-metric-strategy", str(perf_strategy)]

        if outer_isolation:
            cmd.append("--pypto-pro-outer-case-isolation")
            cmd += [
                "--timeout-per-operator",
                str(self.process_config.timeout_per_operator),
            ]
        elif any((case.attrs or {}).get("mc2_distributed") for case in task.cases):
            cmd += [
                "--timeout-per-operator",
                str(self.process_config.timeout_per_operator),
            ]

        return cmd

    def _task_uses_perf_batch(self, task: TaskUnit) -> bool:
        batch_cases = task.perf_batch_cases
        if batch_cases is None:
            batch_cases = getattr(self.base_config, "perf_batch_cases", True)
        return bool(
            batch_cases
            and self.process_config.enable_profiler
            and len(task.cases) > 1
            and not getattr(self.base_config, "pypto_pro_outer_case_isolation", False)
        )

    def _task_uses_pypto_batch(self, task: TaskUnit) -> bool:
        return bool(
            getattr(self.base_config, "pypto_pro_outer_case_isolation", False)
            and len(task.cases) > 1
        )

    def _needs_process_group_cleanup(self) -> bool:
        return bool(
            self.process_config.enable_profiler
            or getattr(self.base_config, "pypto_pro_outer_case_isolation", False)
        )

    @staticmethod
    def _is_unexecuted_pro_case(result: EvalCaseResult) -> bool:
        return bool(
            not result.success
            and result.failure_type == "cascade_device"
            and result.skipped_before_execution is True
            and result.accuracy_result is None
            and result.perf_result is None
            and result.ai_run_result is None
            and result.golden_run_result is None
        )

    @classmethod
    def _is_terminal_pro_verdict(cls, result: EvalCaseResult) -> bool:
        """Never discard a recorded failure to rerun until a candidate passes.

        Even perf_collection_failed can include candidate execution errors;
        it is not sufficient evidence of a retryable infrastructure failure.
        """
        if cls._is_unexecuted_pro_case(result):
            return False
        if not result.success or (
            result.accuracy_result is not None and not result.accuracy_result.passed
        ):
            return True
        perf = result.perf_result
        if perf is None:
            return False
        if perf.metadata.get("cpu_fallback_detected"):
            return True
        return not perf.is_success()

    @staticmethod
    def _pro_failure_priority(result: EvalCaseResult) -> int:
        """Duplicate rows must not downgrade a definitive rejection."""
        if is_precision_failure_type(result.failure_type):
            return 4
        perf = result.perf_result
        if perf is not None:
            if perf.metadata.get("cpu_fallback_detected"):
                return 3
            if (
                not perf.is_success()
                and perf.metadata.get("perf_collection_failed") is False
            ):
                return 2
        return 1

    @staticmethod
    def _read_pro_results(
        output_file: str, strict: bool = False
    ) -> List[EvalCaseResult]:
        """Recover valid rows independently so a bad row cannot erase a verdict."""
        try:
            rows = json.loads(Path(output_file).read_text()).get("case_results", [])
        except (OSError, ValueError, AttributeError):
            return []
        if not isinstance(rows, list):
            return []
        results = []
        for row in rows:
            try:
                if not isinstance(row.get("case_id"), str) or not isinstance(
                    row.get("success"), bool
                ):
                    raise ValueError("invalid Pro result identity or success flag")
                result = EvalCaseResult.from_dict(row)
                perf = result.perf_result
                if perf is not None and perf.metadata.get("cpu_fallback_detected"):
                    perf.metadata["perf_collection_failed"] = False
                if perf is not None and (
                    not perf.is_success() or perf.metadata.get("cpu_fallback_detected")
                ):
                    perf.elapsed_us = 0.0
                results.append(result)
            except (ValueError, TypeError, KeyError, AttributeError):
                if strict:
                    return []
                continue
        return results

    def _communicate_task(self, proc, task: TaskUnit, output_file: str):
        """Bound a reused worker by both progress and total elapsed time.

        Completed IDs are a liveness signal, not a security boundary. Only
        strict monotonic progress within this task renews the per-case budget.
        """
        budget = self.process_config.timeout_per_operator
        if not self._task_uses_pypto_batch(task):
            return proc.communicate(timeout=len(task.cases) * budget)
        expected = {case.get_case_id_str() for case in task.cases}
        completed = set()
        now = time.monotonic()
        absolute_deadline = now + len(task.cases) * budget
        progress_deadline = now + budget
        while True:
            remaining = min(absolute_deadline, progress_deadline) - time.monotonic()
            if remaining <= 0:
                reason = (
                    "total deadline"
                    if time.monotonic() >= absolute_deadline
                    else "no case progress"
                )
                raise subprocess.TimeoutExpired([*proc.args, f"({reason})"], budget)
            try:
                return proc.communicate(timeout=min(5.0, remaining))
            except subprocess.TimeoutExpired:
                rows = self._read_pro_results(output_file, strict=True)
                ids = {row.case_id for row in rows}
                if len(ids) == len(rows) and completed < ids <= expected:
                    completed = ids
                    progress_deadline = time.monotonic() + budget

    def _pypto_batch_failure(
        self, task: TaskUnit, reason: str, retry_allowed: bool, observed_results=None
    ):
        """Retry infrastructure failures without erasing already observed failures."""
        expected = {case.get_case_id_str() for case in task.cases}
        terminal = {}
        for result in observed_results or []:
            if result.case_id not in expected or not self._is_terminal_pro_verdict(
                result
            ):
                continue
            previous = terminal.get(result.case_id)
            if previous is None or self._pro_failure_priority(
                result
            ) > self._pro_failure_priority(previous):
                terminal[result.case_id] = result
        retry_cases = [
            case for case in task.cases if case.get_case_id_str() not in terminal
        ]
        should_retry = bool(
            retry_cases
            and retry_allowed
            and task.retry_count < self.process_config.max_retries
        )
        action = (
            f"{len(retry_cases)} 个 case 将在独立进程重试，保留 {len(terminal)} 个失败结论"
            if should_retry
            else "批次已丢弃，重试已关闭或耗尽"
        )
        print(
            f"[WARN] {task.operator}@Card{task.device_id}: PyPTO Pro worker 批量失败"
            f" ({reason})，{action}",
            flush=True,
        )
        return (
            task,
            list(terminal.values())
            + _synthesize_failure_cases(
                retry_cases,
                "subprocess_failure",
                f"PyPTO Pro batch discarded; isolated retry required: {reason}",
            ),
            should_retry,
            "pypto_pro_batch_failure",
            retry_cases,
        )

    def _terminate_timed_out_process(
        self,
        proc: subprocess.Popen,
        task: Optional[TaskUnit] = None,
    ) -> None:
        """按评测模式终止超时 worker。

        Profiler 或 PyPTO Pro worker 可能派生 msprof/parser/Runtime 后代，
        必须清理整个进程组；纯 correctness worker 保持只终止直接子进程。
        """
        if self._needs_process_group_cleanup():
            _terminate_process_group(proc)
            return

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _group_pro_tasks(self, task_units: List[TaskUnit]) -> List[TaskUnit]:
        """Reuse the common per-card grouping policy for initial Pro tasks."""
        if not getattr(self.base_config, "pypto_pro_outer_case_isolation", False):
            return task_units
        if any(
            len(t.cases) != 1 or t.retry_count or t.perf_batch_cases is not None
            for t in task_units
        ):
            return task_units
        cap = self.base_config.max_cases_per_task_unit
        if cap == 1:
            return task_units
        grouped = {}
        for task in task_units:
            key = (task.operator, task.rel_path)
            grouped.setdefault(key, []).extend(task.cases)
        return [
            unit
            for (operator, _), cases in grouped.items()
            for unit in build_task_units(
                {operator: cases},
                self.card_count,
                max_cases_per_task_unit=cap,
            )
        ]

    def evaluate_task_units(self, task_units: List[TaskUnit]) -> List[EvalCaseResult]:
        """按 TaskUnit 并行评测

        每个 TaskUnit 启动一个 eval-child 子进程，
        通过 ThreadPoolExecutor 实现多卡并行和动态负载均衡。
        支持失败任务自动重试机制。
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

        mc2_units = [
            t
            for t in task_units
            if any((c.attrs or {}).get("mc2_distributed") for c in t.cases)
        ]
        if mc2_units and len(task_units) > 1:
            # No ordinary worker or other MC2 task may contend for these ranks.
            mc2_ids = {id(task) for task in mc2_units}
            regular = [task for task in task_units if id(task) not in mc2_ids]
            results = self.evaluate_task_units(regular) if regular else []
            for task in mc2_units:
                results.extend(self.evaluate_task_units([task]))
            return results
        if mc2_units:
            needed = max(
                int(
                    (c.attrs or {}).get(
                        "world_size", (c.attrs or {}).get("ep_world_size", 1)
                    )
                )
                for c in mc2_units[0].cases
            )
            if needed > self.card_count:
                raise RuntimeError(
                    f"MC2 needs {needed} visible NPUs; only {self.card_count} available"
                )
        else:
            task_units = self._group_pro_tasks(task_units)

        max_workers = self.card_count * self.process_config.processes_per_card
        if mc2_units:
            max_workers = 1
        total_cases = sum(len(t.cases) for t in task_units)

        print(
            f"[INFO] 配置: {self.card_count} 卡 × {self.process_config.processes_per_card} 并发/卡"
        )
        print(f"[INFO] TaskUnit 数: {len(task_units)}, 用例数: {total_cases}")
        print(f"[INFO] 单算子超时: {self.process_config.timeout_per_operator}s")
        print(f"[INFO] 最大并发: {max_workers}")
        if self.process_config.max_retries > 0:
            print(
                f"[INFO] 重试策略: 最大重试 {self.process_config.max_retries} 次 "
                f"(timeout={self.process_config.retry_on_timeout}, "
                f"oom={self.process_config.retry_on_oom}, "
                f"failure={self.process_config.retry_on_failure})"
            )

        base_env = self._build_env()
        all_case_results: List[EvalCaseResult] = []
        completed_count = 0
        retry_queue: List[TaskUnit] = []  # 重试队列
        case_failure_count: Dict[str, int] = defaultdict(int)  # 记录每个case的失败次数

        def _run_task(idx_and_task):
            """在线程中运行一个 TaskUnit，返回 (task, case_results, should_retry, failure_type)"""
            idx, task = idx_and_task
            # 写 cases JSON 文件
            fd, cases_file = tempfile.mkstemp(suffix=".json", prefix="cases_")
            os.close(fd)
            child_cwd = None
            output_file = None
            try:
                Path(cases_file).write_text(
                    json.dumps([c.to_dict() for c in task.cases], ensure_ascii=False)
                )

                # 写 output 文件占位
                fd, output_file = tempfile.mkstemp(suffix=".json", prefix="cannbench_")
                os.close(fd)

                cmd = self._build_child_cmd(task, cases_file, output_file)
                env = self._build_env_for_task(base_env, task)
                timeout = len(task.cases) * self.process_config.timeout_per_operator

                if getattr(self.base_config, "pypto_pro_outer_case_isolation", False):
                    child_cwd = tempfile.mkdtemp(prefix="pypto_eval_child_")

                popen_kwargs = {
                    "start_new_session": True,
                    "env": env,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                }
                if child_cwd is not None:
                    popen_kwargs["cwd"] = child_cwd
                proc = subprocess.Popen(cmd, **popen_kwargs)
                self._active_processes.append(proc)
                oom_ok = _write_oom_score_adj(proc.pid, 1000)
                # 父进程外部写是双保险，子进程自设（cmd_eval_child）才是主路径
                if not oom_ok:
                    print(
                        f"[WARN] 子进程 PID={proc.pid} oom_score_adj 设置失败"
                        f"（OOM Kill 时主进程也可能被杀）",
                        flush=True,
                    )

                try:
                    child_stdout, child_stderr = self._communicate_task(
                        proc, task, output_file
                    )
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    # 从活跃列表移除已完成的子进程
                    try:
                        self._active_processes.remove(proc)
                    except ValueError:
                        pass

                    deadline_info = (
                        f"无 case 进展上限 {self.process_config.timeout_per_operator}s，整组上限 {timeout}s"
                        if self._task_uses_pypto_batch(task)
                        else f"{timeout}s"
                    )
                    print(
                        f"[WARN] TaskUnit {task.operator}@Card{task.device_id} 超时 ({deadline_info})"
                    )
                    pro_worker = getattr(
                        self.base_config, "pypto_pro_outer_case_isolation", False
                    )
                    if pro_worker:
                        self._terminate_timed_out_process(proc, task)
                    try:
                        child_stdout, child_stderr = proc.communicate(
                            timeout=1 if pro_worker else 5
                        )
                    except Exception:
                        child_stdout, child_stderr = "", ""
                    if not pro_worker:
                        self._terminate_timed_out_process(proc, task)
                    _forward_child_output(child_stdout, child_stderr)
                    if self._task_uses_pypto_batch(task):
                        return self._pypto_batch_failure(
                            task,
                            f"timeout ({deadline_info})",
                            self.process_config.retry_on_timeout,
                            self._read_pro_results(output_file),
                        )
                    # 超时：尝试恢复部分结果，失败则合成全量 timeout 失败
                    partial = _try_recover_partial_results(output_file)
                    if partial:
                        completed_ids = {r.case_id for r in partial}
                        remaining = [
                            c
                            for c in task.cases
                            if c.get_case_id_str() not in completed_ids
                        ]
                        oom_rest = _synthesize_failure_cases(
                            remaining,
                            "timeout",
                            f"子进程超时 ({timeout}s) 被 SIGTERM/SIGKILL",
                        )
                        print(
                            f"[INFO] {task.operator}: 超时后恢复 {len(partial)} 个已完成用例，"
                            f"合成 {len(oom_rest)} 个超时失败用例"
                        )
                        # 判断是否应该重试
                        should_retry = (
                            self.process_config.retry_on_timeout
                            and task.retry_count < self.process_config.max_retries
                            and len(remaining) > 0
                        )
                        return (
                            task,
                            partial + oom_rest,
                            should_retry,
                            "timeout",
                            remaining,
                        )
                    should_retry = (
                        self.process_config.retry_on_timeout
                        and task.retry_count < self.process_config.max_retries
                    )
                    return (
                        task,
                        _synthesize_failure_cases(
                            task.cases,
                            "timeout",
                            f"子进程超时 ({timeout}s) 被 SIGTERM/SIGKILL",
                        ),
                        should_retry,
                        "timeout",
                        task.cases,
                    )

                # 从活跃列表移除已完成的子进程，避免内存累积
                try:
                    self._active_processes.remove(proc)
                except ValueError:
                    pass

                # 透传子进程输出（含成功路径），恢复 staged.log 的用例级细节
                _forward_child_output(child_stdout, child_stderr)

                if rc != 0:
                    batch_task = self._task_uses_perf_batch(task)
                    if self._needs_process_group_cleanup():
                        _terminate_process_group(proc, grace_sec=1.0)
                    crash_diag = _extract_crash_diag(rc, child_stdout, child_stderr)
                    if crash_diag:
                        print(
                            f"[WARN] {task.operator}@Card{task.device_id}:{crash_diag}",
                            flush=True,
                        )
                    if self._task_uses_pypto_batch(task):
                        oom = _is_oom_killed(proc, rc)
                        return self._pypto_batch_failure(
                            task,
                            f"worker exit {rc}{crash_diag}",
                            self.process_config.retry_on_oom
                            if oom
                            else self.process_config.retry_on_failure,
                            self._read_pro_results(output_file),
                        )
                    if _is_oom_killed(proc, rc):
                        if batch_task:
                            should_retry = (
                                self.process_config.retry_on_oom
                                and task.retry_count < self.process_config.max_retries
                            )
                            print(
                                f"[WARN] {task.operator}@Card{task.device_id}: "
                                f"批量 profiler OOM Kill，已停止 profiler 进程组；"
                                f"{len(task.cases)} 个 case 将逐 case 回退"
                            )
                            return (
                                task,
                                _synthesize_failure_cases(
                                    task.cases,
                                    "oom_killed",
                                    "批量 profiler 子进程被 OOM Killer 杀死；"
                                    "已清理 profiler 会话并切换逐 case 性能重试",
                                ),
                                should_retry,
                                "perf_batch_oom",
                                task.cases,
                            )
                        # OOM Kill：尝试恢复部分结果 + 合成剩余用例的 oom_killed 失败
                        partial = _try_recover_partial_results(output_file)
                        if partial:
                            completed_ids = {r.case_id for r in partial}
                            remaining = [
                                c
                                for c in task.cases
                                if c.get_case_id_str() not in completed_ids
                            ]
                            oom_rest = _synthesize_failure_cases(
                                remaining,
                                "oom_killed",
                                "子进程被 OOM Killer 杀死 (SIGKILL/-9)，内存不足",
                            )
                            print(
                                f"[WARN] {task.operator}@Card{task.device_id}: OOM Kill (rc={rc})"
                            )
                            print(
                                f"[INFO] {task.operator}: OOM Kill 后恢复 {len(partial)} 个已完成用例，"
                                f"合成 {len(oom_rest)} 个 OOM 失败用例"
                            )
                            # 判断是否应该重试
                            should_retry = (
                                self.process_config.retry_on_oom
                                and task.retry_count < self.process_config.max_retries
                                and len(remaining) > 0
                            )
                            return (
                                task,
                                partial + oom_rest,
                                should_retry,
                                "oom_killed",
                                remaining,
                            )
                        print(
                            f"[WARN] {task.operator}@Card{task.device_id}: OOM Kill (rc={rc})，无部分结果可恢复"
                        )
                        should_retry = (
                            self.process_config.retry_on_oom
                            and task.retry_count < self.process_config.max_retries
                        )
                        return (
                            task,
                            _synthesize_failure_cases(
                                task.cases,
                                "oom_killed",
                                "子进程被 OOM Killer 杀死 (SIGKILL/-9)，内存不足",
                            ),
                            should_retry,
                            "oom_killed",
                            task.cases,
                        )
                    print(
                        f"[WARN] {task.operator}@Card{task.device_id}: 子进程异常退出 rc={rc}"
                    )
                    partial = _try_recover_partial_results(output_file)
                    if partial:
                        completed_ids = {r.case_id for r in partial}
                        remaining = [
                            c
                            for c in task.cases
                            if c.get_case_id_str() not in completed_ids
                        ]
                        failed = _synthesize_failure_cases(
                            remaining,
                            "subprocess_failure",
                            f"子进程异常退出 rc={rc}{crash_diag}",
                        )
                        print(
                            f"[INFO] {task.operator}: 异常退出后恢复 {len(partial)} 个已完成用例，"
                            f"合成 {len(failed)} 个子进程失败用例"
                        )
                        should_retry = (
                            self.process_config.retry_on_failure
                            and task.retry_count < self.process_config.max_retries
                            and len(remaining) > 0
                        )
                        return (
                            task,
                            partial + failed,
                            should_retry,
                            "subprocess_failure",
                            remaining,
                        )
                    should_retry = (
                        self.process_config.retry_on_failure
                        and task.retry_count < self.process_config.max_retries
                    )
                    return (
                        task,
                        _synthesize_failure_cases(
                            task.cases,
                            "subprocess_failure",
                            f"子进程异常退出 rc={rc}{crash_diag}",
                        ),
                        should_retry,
                        "subprocess_failure",
                        task.cases,
                    )

                # 正常退出：读取完整结果
                # 从活跃列表移除已完成的子进程
                try:
                    self._active_processes.remove(proc)
                except ValueError:
                    pass

                try:
                    data = json.loads(Path(output_file).read_text())
                except (json.JSONDecodeError, OSError) as e:
                    print(
                        f"[WARN] TaskUnit {task.operator}@Card{task.device_id} 结果解析失败: {e}"
                    )
                    if self._task_uses_pypto_batch(task):
                        _terminate_process_group(proc, grace_sec=1.0)
                        return self._pypto_batch_failure(
                            task,
                            f"invalid result JSON: {e}",
                            self.process_config.retry_on_failure,
                            self._read_pro_results(output_file),
                        )
                    should_retry = (
                        self.process_config.retry_on_failure
                        and task.retry_count < self.process_config.max_retries
                    )
                    return (
                        task,
                        _synthesize_failure_cases(
                            task.cases,
                            "subprocess_failure",
                            f"子进程结果 JSON 解析失败: {e}",
                        ),
                        should_retry,
                        "subprocess_failure",
                        task.cases,
                    )

                if self._task_uses_pypto_batch(task):
                    case_results = self._read_pro_results(output_file)
                    expected = {case.get_case_id_str() for case in task.cases}
                    actual = [result.case_id for result in case_results]
                    raw_results = (
                        data.get("case_results") if isinstance(data, dict) else None
                    )
                    valid = (
                        isinstance(raw_results, list)
                        and len(raw_results) == len(actual) == len(expected)
                        and set(actual) == expected
                    )
                    if self.process_config.enable_profiler:
                        valid = valid and all(
                            self._is_unexecuted_pro_case(result)
                            or self._is_terminal_pro_verdict(result)
                            or (
                                result.perf_result is not None
                                and result.perf_result.is_success()
                            )
                            for result in case_results
                        )
                    if not valid:
                        _terminate_process_group(proc, grace_sec=1.0)
                        return self._pypto_batch_failure(
                            task,
                            "incomplete or collection-failed case results",
                            self.process_config.retry_on_failure,
                            case_results,
                        )
                    unexecuted = {
                        result.case_id
                        for result in case_results
                        if self._is_unexecuted_pro_case(result)
                    }
                    if unexecuted:
                        # A clean exit completed the child API guard and result
                        # validation. Keep its completed cases and retry only
                        # cases skipped after the worker context became unhealthy.
                        _terminate_process_group(proc, grace_sec=1.0)
                        retry_cases = [
                            case
                            for case in task.cases
                            if case.get_case_id_str() in unexecuted
                        ]
                        should_retry = (
                            self.process_config.retry_on_failure
                            and task.retry_count < self.process_config.max_retries
                        )
                        print(
                            f"[WARN] {task.operator}@Card{task.device_id}: Pro worker "
                            f"设备异常，保留已执行结果，{len(retry_cases)} 个未执行 case "
                            f"需要独立进程恢复 (retry={should_retry})",
                            flush=True,
                        )
                        return (
                            task,
                            case_results,
                            should_retry,
                            "pypto_pro_batch_failure",
                            retry_cases,
                        )
                else:
                    case_results = [
                        EvalCaseResult.from_dict(row)
                        for row in data.get("case_results", [])
                    ]
                return (task, case_results, False, None, [])  # 成功，不需要重试

            finally:
                try:
                    os.unlink(cases_file)
                except OSError:
                    pass
                if output_file:
                    try:
                        os.unlink(output_file)
                    except OSError:
                        pass
                if child_cwd:
                    shutil.rmtree(child_cwd, ignore_errors=True)

        pool = _DevicePool(
            self._available_cards, self.process_config.processes_per_card
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = deque(enumerate(task_units))
            active: Dict = {}  # future -> task

            while pending or active:
                # 有空闲卡时派发任务
                while pending and pool.has_available():
                    idx, task = pending.popleft()
                    task.device_id = pool.acquire(exclude=task.excluded_devices or None)
                    if task.device_id is None:
                        task.device_id = pool.acquire()
                    future = executor.submit(_run_task, (idx, task))
                    active[future] = task

                if not active:
                    break

                done, _ = wait(list(active.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    pool.release(task.device_id)
                    try:
                        (
                            task_info,
                            case_results,
                            should_retry,
                            failure_type,
                            failed_cases,
                        ) = future.result()
                        completed_count += 1
                        all_case_results.extend(case_results)
                        passed = sum(1 for r in case_results if r.success)
                        status = "✅" if passed > 0 else "❌"
                        print(
                            f"[INFO] [{completed_count}/{len(task_units)}] Card {task.device_id}: "
                            f"{task.operator} {status} ({passed}/{len(task.cases)})"
                        )

                        # 处理重试逻辑
                        if should_retry and failed_cases:
                            # 更新失败计数
                            for case in failed_cases:
                                case_id = case.get_case_id_str()
                                case_failure_count[case_id] += 1

                            # 过滤掉多次失败的case（如果启用了排除策略）
                            if self.process_config.exclude_repeatedly_failed_cases:
                                repeatedly_failed = []
                                cases_to_retry = []
                                for case in failed_cases:
                                    case_id = case.get_case_id_str()
                                    if case_failure_count[case_id] >= 2:
                                        # 已经失败2次，不再重试
                                        repeatedly_failed.append(case)
                                    else:
                                        cases_to_retry.append(case)

                                # 将多次失败的case记录到结果中（标记为最终失败）
                                if repeatedly_failed:
                                    for case in repeatedly_failed:
                                        case_id = case.get_case_id_str()
                                        print(
                                            f"[WARN] 用例 {case_id} 已连续失败 {case_failure_count[case_id]} 次，"
                                            f"不再重试，标记为最终失败"
                                        )
                            else:
                                cases_to_retry = failed_cases

                            # 只有还有需要重试的case时才创建重试任务
                            if cases_to_retry:
                                # 创建重试任务（device_id 占位，_process_retry_queue 中由设备池动态分配）
                                isolated_retry = failure_type in (
                                    "perf_batch_oom",
                                    "pypto_pro_batch_failure",
                                )
                                retry_groups = (
                                    [[case] for case in cases_to_retry]
                                    if isolated_retry
                                    else [cases_to_retry]
                                )
                                for retry_cases in retry_groups:
                                    retry_task = TaskUnit(
                                        operator=task.operator,
                                        rel_path=task.rel_path,
                                        cases=retry_cases,
                                        device_id=task.device_id,
                                        retry_count=task.retry_count + 1,
                                        excluded_devices=task.excluded_devices
                                        | {task.device_id},
                                        parent_task_id=f"{task.operator}@Card{task.device_id}",
                                        perf_batch_cases=(
                                            False
                                            if isolated_retry
                                            else task.perf_batch_cases
                                        ),
                                    )
                                    retry_queue.append(retry_task)
                                excluded_count = len(failed_cases) - len(cases_to_retry)
                                excluded_msg = (
                                    f", 排除 {excluded_count} 个多次失败的用例"
                                    if excluded_count > 0
                                    else ""
                                )
                                if failure_type == "pypto_pro_batch_failure":
                                    retry_mode = "独立单 case worker"
                                elif isolated_retry:
                                    retry_mode = "逐 case profiler"
                                else:
                                    retry_mode = "普通重试"
                                print(
                                    f"[INFO] 重试任务已加入队列: {task.operator} "
                                    f"(retry {task.retry_count + 1}/{self.process_config.max_retries}, "
                                    f"{len(cases_to_retry)} 个用例{excluded_msg}, {retry_mode}, "
                                    f"原卡 Card{task.device_id} 故障 ({failure_type}), 将动态分配新卡)"
                                )

                        # 定期清理已退出的子进程引用，避免内存累积
                        self._cleanup_completed_processes()

                        # 每完成 3 个任务执行一次 GC，回收主进程临时对象
                        if completed_count % 3 == 0:
                            import gc

                            gc.collect()
                            avail_mb = self._get_available_memory_mb()
                            if avail_mb > 0 and avail_mb < 2048:
                                print(
                                    f"[WARN] 可用内存低: {avail_mb:.0f} MB，"
                                    f"活跃子进程: {len(self._active_processes)}",
                                    flush=True,
                                )

                    except Exception as e:
                        completed_count += 1
                        print(
                            f"[WARN] [{completed_count}/{len(task_units)}] Card {task.device_id}: "
                            f"{task.operator} 异常: {e}"
                        )

        print(f"[INFO] 初次调度完成: {completed_count}/{len(task_units)} 个 TaskUnit")

        # 处理重试队列
        if retry_queue:
            print(f"[INFO] 开始处理重试队列: {len(retry_queue)} 个任务")
            retry_results = self._process_retry_queue(
                retry_queue, base_env, max_workers
            )
            all_case_results.extend(retry_results)

        print(
            f"[INFO] 调度完成: 总共 {len(task_units)} + {len(retry_queue)} 个 TaskUnit"
        )

        # F744: 移除父进程的统一等待（无效）
        # 原因：父进程等待时，子进程已经退出，profiler 解析进程已被杀死
        # 正确的方案：在子进程退出前等待（cli.py 中的 20 秒等待）

        return all_case_results

    def _process_retry_queue(
        self, retry_queue: List[TaskUnit], base_env: Dict[str, str], max_workers: int
    ) -> List[EvalCaseResult]:
        """处理重试队列中的任务

        Args:
            retry_queue: 需要重试的TaskUnit列表
            base_env: 基础环境变量
            max_workers: 最大并发数

        Returns:
            重试任务的用例结果列表
        """
        retry_results: List[EvalCaseResult] = []
        completed_count = 0

        def _run_retry_task(idx_and_task):
            """运行重试任务（不再生成新的重试）"""
            idx, task = idx_and_task
            fd, cases_file = tempfile.mkstemp(suffix=".json", prefix="retry_cases_")
            os.close(fd)
            child_cwd = None
            output_file = None
            try:
                Path(cases_file).write_text(
                    json.dumps([c.to_dict() for c in task.cases], ensure_ascii=False)
                )

                fd, output_file = tempfile.mkstemp(
                    suffix=".json", prefix="retry_cannbench_"
                )
                os.close(fd)

                cmd = self._build_child_cmd(task, cases_file, output_file)
                env = self._build_env_for_task(base_env, task)
                timeout = len(task.cases) * self.process_config.timeout_per_operator

                if getattr(self.base_config, "pypto_pro_outer_case_isolation", False):
                    child_cwd = tempfile.mkdtemp(prefix="pypto_eval_child_")

                popen_kwargs = {
                    "start_new_session": True,
                    "env": env,
                    "stdout": subprocess.PIPE,
                    "stderr": subprocess.PIPE,
                    "text": True,
                }
                if child_cwd is not None:
                    popen_kwargs["cwd"] = child_cwd
                proc = subprocess.Popen(cmd, **popen_kwargs)
                self._active_processes.append(proc)
                _write_oom_score_adj(proc.pid, 1000)

                try:
                    child_stdout, child_stderr = proc.communicate(timeout=timeout)
                    rc = proc.returncode
                except subprocess.TimeoutExpired:
                    try:
                        self._active_processes.remove(proc)
                    except ValueError:
                        pass
                    print(
                        f"[WARN] 重试任务 {task.operator}@Card{task.device_id} 仍然超时"
                    )
                    pro_worker = getattr(
                        self.base_config, "pypto_pro_outer_case_isolation", False
                    )
                    if pro_worker:
                        self._terminate_timed_out_process(proc, task)
                    try:
                        child_stdout, child_stderr = proc.communicate(
                            timeout=1 if pro_worker else 5
                        )
                    except Exception:
                        child_stdout, child_stderr = "", ""
                    if not pro_worker:
                        self._terminate_timed_out_process(proc, task)
                    _forward_child_output(child_stdout, child_stderr)
                    partial = _try_recover_partial_results(output_file)
                    if partial:
                        completed_ids = {r.case_id for r in partial}
                        remaining = [
                            c
                            for c in task.cases
                            if c.get_case_id_str() not in completed_ids
                        ]
                        failed = _synthesize_failure_cases(
                            remaining, "timeout", f"重试后仍超时 ({timeout}s)"
                        )
                        return (task, partial + failed)
                    return (
                        task,
                        _synthesize_failure_cases(
                            task.cases, "timeout", f"重试后仍超时 ({timeout}s)"
                        ),
                    )

                try:
                    self._active_processes.remove(proc)
                except ValueError:
                    pass

                _forward_child_output(child_stdout, child_stderr)

                if rc != 0:
                    if self._needs_process_group_cleanup():
                        _terminate_process_group(proc, grace_sec=1.0)
                    crash_diag = _extract_crash_diag(rc, child_stdout, child_stderr)
                    if crash_diag:
                        print(
                            f"[WARN] 重试 {task.operator}@Card{task.device_id}:{crash_diag}",
                            flush=True,
                        )
                    if _is_oom_killed(proc, rc):
                        partial = _try_recover_partial_results(output_file)
                        if partial:
                            completed_ids = {r.case_id for r in partial}
                            remaining = [
                                c
                                for c in task.cases
                                if c.get_case_id_str() not in completed_ids
                            ]
                            failed = _synthesize_failure_cases(
                                remaining, "oom_killed", "重试后仍 OOM Kill"
                            )
                            return (task, partial + failed)
                        return (
                            task,
                            _synthesize_failure_cases(
                                task.cases, "oom_killed", "重试后仍 OOM Kill"
                            ),
                        )
                    partial = _try_recover_partial_results(output_file)
                    if partial:
                        completed_ids = {r.case_id for r in partial}
                        remaining = [
                            c
                            for c in task.cases
                            if c.get_case_id_str() not in completed_ids
                        ]
                        failed = _synthesize_failure_cases(
                            remaining,
                            "subprocess_failure",
                            f"重试后仍异常退出 rc={rc}{crash_diag}",
                        )
                        print(
                            f"[INFO] {task.operator}: 重试异常退出后恢复 {len(partial)} 个已完成用例，"
                            f"合成 {len(failed)} 个子进程失败用例"
                        )
                        return (task, partial + failed)
                    return (
                        task,
                        _synthesize_failure_cases(
                            task.cases,
                            "subprocess_failure",
                            f"重试后仍异常退出 rc={rc}{crash_diag}",
                        ),
                    )

                try:
                    data = json.loads(Path(output_file).read_text())
                    case_results = [
                        EvalCaseResult.from_dict(r)
                        for r in data.get("case_results", [])
                    ]
                    if getattr(
                        self.base_config, "pypto_pro_outer_case_isolation", False
                    ):
                        expected = {case.get_case_id_str() for case in task.cases}
                        actual = [result.case_id for result in case_results]
                        if len(actual) != len(expected) or set(actual) != expected:
                            raise ValueError(
                                "Pro retry returned incomplete or mismatched case IDs"
                            )
                    if (
                        task.perf_batch_cases is False
                        and task.parent_task_id
                        and not getattr(
                            self.base_config, "pypto_pro_outer_case_isolation", False
                        )
                    ):
                        for result in case_results:
                            if result.perf_result is None:
                                continue
                            result.perf_result.metadata.update(
                                {
                                    "perf_batch_attempted": True,
                                    "perf_batch_fallback": True,
                                    "perf_batch_external_retry": True,
                                    "perf_batch_fallback_reason": (
                                        "batch eval-child was OOM-killed; "
                                        "reran in an isolated per-case profiler"
                                    ),
                                }
                            )
                    return (task, case_results)
                except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
                    if getattr(
                        self.base_config, "pypto_pro_outer_case_isolation", False
                    ):
                        _terminate_process_group(proc, grace_sec=1.0)
                    elif not isinstance(e, (json.JSONDecodeError, OSError)):
                        raise
                    return (
                        task,
                        _synthesize_failure_cases(
                            task.cases, "subprocess_failure", f"重试后结果解析失败: {e}"
                        ),
                    )

            finally:
                try:
                    os.unlink(cases_file)
                except OSError:
                    pass
                if output_file:
                    try:
                        os.unlink(output_file)
                    except OSError:
                        pass
                if child_cwd:
                    shutil.rmtree(child_cwd, ignore_errors=True)

        pool = _DevicePool(
            self._available_cards, self.process_config.processes_per_card
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = deque(enumerate(retry_queue))
            active: Dict = {}

            while pending or active:
                while pending and pool.has_available():
                    idx, task = pending.popleft()
                    task.device_id = pool.acquire(exclude=task.excluded_devices or None)
                    if task.device_id is None:
                        task.device_id = pool.acquire()
                    future = executor.submit(_run_retry_task, (idx, task))
                    active[future] = task

                if not active:
                    break

                done, _ = wait(list(active.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    task = active.pop(future)
                    pool.release(task.device_id)
                    try:
                        task_info, case_results = future.result()
                        completed_count += 1
                        retry_results.extend(case_results)
                        passed = sum(1 for r in case_results if r.success)
                        status = "✅" if passed > 0 else "❌"
                        print(
                            f"[INFO] [重试 {completed_count}/{len(retry_queue)}] Card {task.device_id}: "
                            f"{task.operator} {status} ({passed}/{len(task.cases)}) "
                            f"(retry {task.retry_count})"
                        )

                        self._cleanup_completed_processes()

                    except Exception as e:
                        completed_count += 1
                        print(
                            f"[WARN] [重试 {completed_count}/{len(retry_queue)}] Card {task.device_id}: "
                            f"{task.operator} 异常: {e}"
                        )

        return retry_results

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

        Profiler/PyPTO Pro 清理整个进程组；纯 correctness 评测保持原有
        Popen 终止行为。
        """
        grace_sec = 5
        terminate_groups = self._needs_process_group_cleanup()
        for proc in self._active_processes:
            if terminate_groups:
                # 组长可能已经被 OOM killer 回收；仍需按原 pgid 清理后代。
                _signal_process_group(proc, signal.SIGTERM)
            elif proc.poll() is None:
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
            if terminate_groups:
                _signal_process_group(proc, signal.SIGKILL)
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        pass
            elif proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._active_processes = []

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "device_id": self.device_id,
            "card_count": self.card_count,
            "processes_per_card": self.process_config.processes_per_card,
            "total_processes": self.total_processes,
        }
