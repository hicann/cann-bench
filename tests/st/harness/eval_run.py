"""Invoke in-repo kernel_eval against a runtime candidate (the "调用 kernel_eval" part).

In-repo (cann-bench/tests/st): kernel_eval lives at <repo>/src, tasks at <repo>/tasks —
no vendored submodule, no baseline_mock. The candidate (golden_mock today, baseline_mock
later) is passed to the cli as --source-dir + exposed via PYTHONPATH.

**集成口径(single-run)**:不按 op 拆成 N 次 cli 调用,而是给 cli 一个**已按 -k/-m 修剪到
选中算子**的 --task-dir,**一次** `kernel_eval.cli eval`(不带 --operator)跑完整个子集 →
**单一报告**。让 cli 自己 discover+schedule,与真实 benchmark(scripts/run_evaluation.sh)
同一条编排路径。所有算子在独立子进程评测(OOM 保护、超时保护、进程隔离)。
--skip-install 进程内候选(PYTHONPATH 暴露 cann_bench),--task-dir/--reports-dir 被 respect。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ST_DIR = Path(__file__).resolve().parent.parent     # cann-bench/tests/st
REPO_ROOT = ST_DIR.parent.parent                     # cann-bench (holds src/, tasks/, scripts/)
KERNEL_EVAL_SRC = REPO_ROOT / "src"
TASKS = REPO_ROOT / "tasks"
RUN_EVALUATION_SH = REPO_ROOT / "scripts" / "run_evaluation.sh"
LEVELS = ("level1", "level2", "level3", "level4")


def has_npu() -> bool:
    """True iff torch + torch_npu importable and an NPU is visible."""
    try:
        import torch  # noqa: F401
        import torch_npu  # noqa: F401
        return bool(torch.npu.is_available())
    except Exception:
        return False


def kernel_eval_env(candidate_dir) -> dict:
    """Env for the cli subprocess: put in-repo kernel_eval (src) + the candidate package
    dir (exposing `cann_bench`) on PYTHONPATH.

    候选用 PYTHONPATH 暴露而非 `pip install -e` —— NPU 服务器的 immutable 容器里 uv ephemeral
    环境无需 setuptools/wheel/pip。cli 以 --skip-install 进程内评测,`import cann_bench` 经此解析。
    warn-mode guard 维持默认(golden 调 torch builtin 会触发 [WARN],非致命)。
    """
    env = dict(os.environ)
    extra = os.pathsep.join([str(KERNEL_EVAL_SRC), str(candidate_dir)])
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = extra + (os.pathsep + prev if prev else "")
    return env


def build_eval_cmd(*, source_dir, task_dir, reports_dir, operator=None, case_id=None) -> list[str]:
    """kernel_eval.cli eval 命令。operator=None → 不过滤,跑遍 --task-dir 里所有算子(集成口径)。
    所有算子在独立子进程评测(OOM 保护、超时保护、进程隔离)。"""
    cmd = [
        sys.executable, "-m", "kernel_eval.cli", "eval",
        "--bench-name", "cann", "--device", "npu",
        "--source-dir", str(source_dir), "--skip-install",
        "--task-dir", str(task_dir), "--reports-dir", str(reports_dir),
    ]
    if operator is not None:
        cmd += ["--operator", str(operator)]
    if case_id is not None:
        cmd += ["--case-id", str(case_id)]
    return cmd


def run_eval_cli(*, source_dir, task_dir, reports_dir, operator=None, case_id=None,
                 timeout=14400):
    """Run ONE kernel_eval.cli eval over --task-dir; returns CompletedProcess (capture_output).
    operator=None 跑遍整棵(已修剪的)task 树 → 单一报告。候选包经 PYTHONPATH 暴露(=source_dir)。
    timeout 默认放宽到 4h:single-run 覆盖整个选中子集(逐 op 子进程串行)。"""
    return subprocess.run(
        build_eval_cmd(source_dir=source_dir, task_dir=task_dir, reports_dir=reports_dir,
                       operator=operator, case_id=case_id),
        env=kernel_eval_env(source_dir), capture_output=True, text=True, timeout=timeout,
    )
