#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Generate a portable Markdown acceptance report for an Ascend Arena run."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
STAGES = ("smoke", "complete", "profile")

# The acceptance report is a repository-level checklist.  Operator results are
# retained as evidence, but are deliberately summarized instead of rendered
# as one section per operator.
FEATURE_DEFINITIONS = (
    {
        "id": "project_setup",
        "name": "获取项目与源码路径",
        "readme": "环境准备 / 获取项目",
        "command": "git clone --branch <branch> <repository>; cd cann-bench-optimize",
        "expected": "仓库可检出，源码路径和 PYTHONPATH 可用",
    },
    {
        "id": "environment",
        "name": "环境准备与 NPU 检查",
        "readme": "环境准备",
        "command": "source scripts/activate_ascend_arena_env.sh; npu-smi info",
        "expected": "CANN、Python、torch_npu 可加载，并能识别 NPU",
    },
    {
        "id": "python_environment",
        "name": "Python/venv/Conda 环境安装",
        "readme": "环境准备 / Python 环境",
        "command": "bash scripts/setup_ascend_arena_env.sh 或使用现有 venv/Conda",
        "expected": "项目依赖安装完成；可选择系统 venv（需系统包）或用户级 Conda",
    },
    {
        "id": "cann_installation",
        "name": "CANN Toolkit 与算子包安装",
        "readme": "环境准备 / 安装并加载 CANN",
        "command": "安装与芯片匹配的 CANN Toolkit、ops 包；加载 set_env.sh",
        "expected": "CANN 编译器、运行库和对应芯片的算子包可用",
    },
    {
        "id": "cann_version_discovery",
        "name": "CANN 版本发现与选择",
        "readme": "环境准备 / 安装并加载 CANN",
        "command": "source scripts/activate_ascend_arena_env.sh 或设置 CANN_ENV_SCRIPT=<path>/set_env.sh",
        "expected": "可在多个 CANN 版本中选择明确的 set_env.sh，且路径写入复现信息",
    },
    {
        "id": "npu_smoke",
        "name": "NPU 真实计算 Smoke 检查",
        "readme": "环境准备 / Python 环境",
        "command": "python -c 'import torch, torch_npu; ... torch.npu.device_count()'",
        "expected": "torch_npu 可导入且至少一次真实 NPU 计算通过",
    },
    {
        "id": "chip_support",
        "name": "A2/A3 芯片支持",
        "readme": "环境准备 / 芯片选择",
        "command": "export ASCEND_ARENA_CHIP=<a2-or-a3>; ./scripts/...",
        "expected": "目标芯片与 CANN 配置匹配，评测可运行",
    },
    {
        "id": "skeleton_archive",
        "name": "固定版本骨架依赖与校验",
        "readme": "环境准备 / 初始化固定版本的源码依赖",
        "command": "检查 ops-skeleton 子模块提交、工作区状态和 ops-skeleton.json",
        "expected": "骨架提交与清单一致且工作区干净；离线压缩包单独校验 SHA256",
    },
    {
        "id": "evaluate_existing",
        "name": "评测已有算子实现",
        "readme": "功能一",
        "command": "./scripts/run_evaluation.sh --run-optimize --source-dir <optimized-source> --task-dir <task-dir>",
        "expected": "编译、smoke/complete/profile 正确性和性能均完成",
    },
    {
        "id": "staged_evaluation",
        "name": "三阶段正确性评测",
        "readme": "功能一 / 运行评测",
        "command": "运行 smoke、complete、profile 三个 correctness 阶段并检查各阶段结果",
        "expected": "每个阶段均有 success、case_results、通过/失败数量，阶段失败可定位",
    },
    {
        "id": "performance_collection",
        "name": "性能采集与基线对比",
        "readme": "功能一 / 查找结果和理解字段",
        "command": "读取 result.json 与 comparison.json 的 task_duration_us、speedup",
        "expected": "性能 case 产生正的微秒时延和候选/基线对比结果",
    },
    {
        "id": "result_contract",
        "name": "评测结果契约",
        "readme": "功能一 / 结果字段",
        "command": "检查 status.json、result.json、comparison.json",
        "expected": "结果字段齐全，时延单位和加速比含义明确",
    },
    {
        "id": "baseline_reuse",
        "name": "基线一次采集与候选复用",
        "readme": "功能二 / 采集原始实现性能",
        "command": "检查 kernel_eval/init_results/ 并运行后续 validate.sh",
        "expected": "初始化基线只采集一次，候选评测复用固定基线",
    },
    {
        "id": "agent_framework",
        "name": "Agent 基类与注册",
        "readme": "功能二 / 添加自己的 Agent",
        "command": "注册 OperatorAgent 子类并运行 ./scripts/batch_optimize.py --agent <name> ...",
        "expected": "Agent 可设置环境、生成 AGENTS.md 并执行优化命令",
    },
    {
        "id": "agent_setup",
        "name": "示例 Agent CLI 与隐私配置",
        "readme": "功能二 / 配置 Agent",
        "command": "安装 codex/claude CLI；复制 agents/.env.example 为 .env 并填写 profile",
        "expected": "所选 CLI、模型和认证配置可用，真实密钥不进入 Git",
    },
    {
        "id": "single_agent_optimize",
        "name": "单算子 Agent 优化",
        "readme": "功能二 / 启动优化",
        "command": "./scripts/run_auto_pipeline.sh prepare --operator <op> --agent <name> --device-id <id>",
        "expected": "单算子 workspace 完成初始化后启动所选 Agent 优化",
    },
    {
        "id": "agent_feedback_loop",
        "name": "基线到 Agent 的反馈闭环",
        "readme": "功能二 / 采集原始实现性能、Agent 优化与每轮反馈",
        "command": "启动后先写入 kernel_eval/init_results，再由 Agent 修改并调用 ./validate.sh",
        "expected": "每个算子基线完成后即可优化；每轮结果反馈到轨迹和日志",
    },
    {
        "id": "agent_logging",
        "name": "Agent prompt 与实时日志",
        "readme": "功能二 / Agent 优化与每轮反馈",
        "command": "查看 work/artifact/<agent>.log",
        "expected": "真实 prompt、工具输出和失败信息实时可追溯",
    },
    {
        "id": "batch_optimize",
        "name": "批量多卡优化与调度",
        "readme": "功能二",
        "command": "./scripts/batch_optimize.py --agent <name> --devices 0,1,...",
        "expected": "按显式设备列表并发，每卡一个 worker，结果集中保存",
    },
    {
        "id": "checkpoint_resume",
        "name": "断点续传与重启",
        "readme": "功能二 / 中断后继续",
        "command": "重复批量命令；需要全量重跑时加 --restart",
        "expected": "progress.json 控制续传，config.json 可复现参数",
    },
    {
        "id": "docker_mode",
        "name": "Docker 隔离模式",
        "readme": "功能二 / Docker",
        "command": "./scripts/batch_optimize.py --docker --docker-image <image>",
        "expected": "基线、Agent 和候选评测在同一容器中完成",
    },
    {
        "id": "docker_image_build",
        "name": "Agent Docker 镜像构建与版本检查",
        "readme": "功能二 / Docker",
        "command": "docker build --network=host --tag <image> docker/dev；检查 codex/claude --version",
        "expected": "与 A2/A3 匹配的镜像可构建，容器内工具链可用",
    },
    {
        "id": "report_visualization",
        "name": "自包含优化报告",
        "readme": "功能二 / 查看最终优化结果",
        "command": "./scripts/generate_batch_report.py <run-root>",
        "expected": "生成自包含 report.html，含最优曲线、浮窗和点击轨迹",
    },
    {
        "id": "trajectory_data",
        "name": "优化轨迹与轮次结果",
        "readme": "功能二 / 查看最终优化结果",
        "command": "检查 kernel_eval/optimization_trajectory.html",
        "expected": "记录每轮正确性、时延、加速比和源码变化",
    },
    {
        "id": "workspace_only",
        "name": "只构建 workspace",
        "readme": "功能三",
        "command": "./scripts/run_auto_pipeline.sh prepare ... --workspace-only",
        "expected": "只生成 workspace，不调用 Agent 或采集基线",
    },
    {
        "id": "external_agent",
        "name": "外部自定义 Agent 接入",
        "readme": "功能三 / 使用自己的 Agent",
        "command": "cd <workspace>/work && ./validate.sh init",
        "expected": "外部 Agent 可修改源码并复用 validate.sh 获取结果",
    },
    {
        "id": "report_regeneration",
        "name": "手动结果重新生成总报告",
        "readme": "功能三 / 获取结果",
        "command": "./validate.sh only-html; ./scripts/generate_batch_report.py <run-root>",
        "expected": "已有 workspace 的结果可重新生成单一 report.html",
    },
    {
        "id": "workspace_assets",
        "name": "Workspace 资产与可移动布局",
        "readme": "功能三 / 只生成文件",
        "command": "检查 workspace/<operator>/work 下的源码、binding、cases、workspace.json、validate.sh",
        "expected": "保留原仓评测资产时，workspace 目录可移动并继续验证",
    },
    {
        "id": "acceptance_report",
        "name": "功能完整性验收报告",
        "readme": "生成仓库功能完整性报告",
        "command": "python scripts/generate_acceptance_report.py <run-root>",
        "expected": "生成可移交 Markdown、report_data.json、evidence 和截图目录",
    },
    {
        "id": "acceptance_evidence_input",
        "name": "验收证据与截图录入",
        "readme": "生成仓库功能完整性报告",
        "command": "填写 feature_results.json，并用 --screenshots-dir / --screenshot 重新生成",
        "expected": "每项功能的 actual、状态、证据和备注显示在同一份 Markdown 中",
    },
    {
        "id": "portable_acceptance_artifacts",
        "name": "验收报告跨机器归档",
        "readme": "生成仓库功能完整性报告",
        "command": "复制 acceptance_report/ 整目录到另一台机器并打开 acceptance_report.md",
        "expected": "JSON 和 Markdown 只使用相对证据链接，不泄露生成主机绝对路径",
    },
    {
        "id": "batch_acceptance",
        "name": "批量功能验收脚本",
        "readme": "当前算子与批量环境验收",
        "command": "python tests/run_ascend_arena_batch_e2e.py --output <run-root>",
        "expected": "自动发现空闲卡并按卡设置并发，生成 e2e_summary.json",
    },
    {
        "id": "diagnostics",
        "name": "日志、退出码与故障诊断",
        "readme": "排查评测异常",
        "command": "检查 status.json、失败日志和脚本退出码",
        "expected": "成功日志按策略清理，失败日志保留且路径可追溯",
    },
    {
        "id": "root_benchmark",
        "name": "通用 CANN Bench 评测入口",
        "readme": "根 README / Quick Start",
        "command": "准备 CANN、torch_npu 和 requirements.txt",
        "expected": "通用评测前置依赖和任务目录可用",
    },
    {
        "id": "root_task_levels",
        "name": "评测集等级与任务组织",
        "readme": "根 README / Task Description、Directory Structure",
        "command": "按 tasks/level1..level4 选择评测集或使用 bench_lab 实验集",
        "expected": "任务可按等级和评测集根目录发现、筛选和加载",
    },
    {
        "id": "root_docker_evaluation",
        "name": "通用评测 Docker 入口",
        "readme": "根 README / Quick Start / Docker",
        "command": "bash docker/eval/build.sh; bash docker/eval/run.sh <source>",
        "expected": "Docker 评测镜像构建成功并生成报告",
    },
    {
        "id": "root_direct_evaluation",
        "name": "通用评测直接脚本入口",
        "readme": "根 README / Quick Start / 直接跑脚本",
        "command": "./scripts/run_evaluation.sh --source-dir <source>",
        "expected": "扫描、编译、安装、正确性和性能流程完成",
    },
    {
        "id": "root_eval_options",
        "name": "通用评测筛选与阶段参数",
        "readme": "根 README / Quick Start、scripts/run_evaluation.sh",
        "command": "使用 --bench-name、--task-dir、--operator、--case-id、--device-id、--no-perf、--warmup/--repeat 等参数",
        "expected": "可选择评测集、算子、用例、设备、并发、Profiler 和确定性输入",
    },
    {
        "id": "root_report_generation",
        "name": "通用评测报告生成",
        "readme": "根 README / 生成评测报告",
        "command": "./scripts/utils/gen_report.sh reports/<result>.json",
        "expected": "从 JSON 生成 Markdown/HTML 报告",
    },
    {
        "id": "task_asset_contract",
        "name": "任务资产格式",
        "readme": "根 README / Test Case Structure",
        "command": "检查 proto.yaml、golden.py、cases.yaml/csv、desc.md",
        "expected": "任务资产完整且可被评测器读取",
    },
    {
        "id": "task_metadata_contract",
        "name": "硬件基线元数据",
        "readme": "根 README / Evaluation",
        "command": "检查 <bench>/metadata/<hardware>.json 中的 baseline_perf_us、t_hw_us",
        "expected": "硬件理论上界和基线时延按芯片版本可追溯，未硬编码到用例文件",
    },
    {
        "id": "operator_examples",
        "name": "算子工程与评测样例",
        "readme": "根 README / Examples",
        "command": "运行 ACLNN、direct、TileLang、Triton-Ascend、StanfordBench 或 fixture 示例",
        "expected": "示例工程可构建，fixture 可完成轻量评测",
    },
    {
        "id": "custom_operator_api",
        "name": "自定义算子 Python API",
        "readme": "根 README / Examples",
        "command": "import torch_npu, cann_bench；调用自定义算子 API",
        "expected": "用户代码可加载安装后的算子并在 NPU 上执行",
    },
    {
        "id": "auto_pipeline",
        "name": "auto_pipeline 算子生成",
        "readme": "Ascend Arena README / 功能二、功能三",
        "command": "./scripts/run_auto_pipeline.sh run --config <config> --workspace <workspace>",
        "expected": "根据规格生成可评测 Ascend C 工程",
    },
    {
        "id": "scoring_policy",
        "name": "HAP 与综合评分策略",
        "readme": "根 README / Evaluation",
        "command": "核对 HAP、正确性权重和 Level 总分",
        "expected": "评分字段、边界值和权重与规范一致",
    },
    {
        "id": "compile_failure_policy",
        "name": "整批编译失败策略",
        "readme": "根 README / Evaluation",
        "command": "提交一个编译失败的多算子任务并检查退出码",
        "expected": "相关算子计 0，汇总错误，进程返回非零",
    },
    {
        "id": "golden_wheel",
        "name": "Golden wheel 构建与安装",
        "readme": "scripts/README.md / build_golden_wheel.sh",
        "command": "./scripts/utils/build_golden_wheel.sh [--operator <op>] [--install]",
        "expected": "tasks 下 Golden 实现可打包为 wheel 并供评测器加载",
    },
    {
        "id": "test_runner",
        "name": "单元测试与端到端测试入口",
        "readme": "scripts/README.md / run_test.sh",
        "command": "./scripts/run_test.sh [ut|e2e] [选项]",
        "expected": "可运行全部、单元或端到端测试，并支持筛选、详细输出和失败即停",
    },
    {
        "id": "stanford_mode",
        "name": "Stanford Bench 评测集切换",
        "readme": "scripts/run_evaluation.sh / --bench-name",
        "command": "./scripts/run_evaluation.sh --bench-name stanford --source-dir <source>",
        "expected": "自动使用 bench_lab/stanford_bench/KernelBench 任务目录并生成对应报告",
    },
    {
        "id": "yaml_tools",
        "name": "评测资产格式转换工具",
        "readme": "scripts/README.md / YAML 工具",
        "command": "python scripts/utils/yaml_to_csv.py ...；python scripts/utils/yaml_block_to_flow.py ...",
        "expected": "YAML 用例可转换为 CSV 或 flow 格式并保持内容可读",
    },
    {
        "id": "cli_inspection",
        "name": "评测 CLI 列表、详情与配置查询",
        "readme": "scripts/README.md / run_evaluation.sh",
        "command": "./scripts/run_evaluation.sh --action list|info|config",
        "expected": "用户可在执行评测前查看算子、用例和当前评测配置",
    },
    {
        "id": "task_contribution",
        "name": "评测任务贡献规范",
        "readme": "根 README / 社区贡献",
        "command": "按 proto.yaml、golden.py、cases.yaml/csv、desc.md 规范提交新任务",
        "expected": "新任务资产完整、参考实现可验证、文档和测试说明齐全",
    },
    {
        "id": "submission_rules",
        "name": "提交行为与反作弊约束",
        "readme": "根 README / 评测前必读",
        "command": "核对 docs/spec/submission_spec.md 与 docs/guide/submission_rules.md",
        "expected": "提交实现遵守真实 NPU kernel、输入泛化、输出布局和构建接口约束",
    },
)
FEATURE_STATUSES = ("PASS", "FAIL", "INCOMPLETE", "N/A")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _safe_component(value: Any) -> str:
    if value in (None, ""):
        return "unknown"
    name = Path(str(value)).name
    return name if name not in {"", ".", ".."} else "unknown"


def _run_text(command: list[str], timeout: float = 10, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, cwd=cwd)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or result.stderr or "").strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def _skeleton_manifest_path(run_root: Path) -> Path | None:
    candidates = [
        run_root / "ops-skeleton.json",
        run_root / "dependency" / "ops-skeleton.json",
        run_root / "work" / "dependency" / "ops-skeleton.json",
    ]
    for depth in range(1, 4):
        candidates.extend(sorted(run_root.glob("/".join(["*"] * depth + ["dependency", "ops-skeleton.json"]))))
    return next((path for path in candidates if path.is_file()), None)


def _skeleton_metadata(run_root: Path) -> dict[str, Any]:
    dependency = Path(__file__).resolve().parents[1] / "ascend_arena_e2e" / "dependency"
    source = _skeleton_manifest_path(run_root) or dependency / "ops-skeleton.json"
    manifest = _load_json(source)
    archive = source.with_name("ops-skeleton.tar.gz")
    archive_sha256 = _sha256(archive)
    declared_sha256 = manifest.get("sha256", "")
    checkout = source.with_name("ops-skeleton")
    revision = _run_text(["git", "rev-parse", "HEAD"], cwd=checkout) if (checkout / ".git").exists() else ""
    clean = not _run_text(["git", "status", "--porcelain"], cwd=checkout) if revision else None
    return {
        "repository": manifest.get("repository", ""),
        "commit": manifest.get("commit", ""),
        "distribution": manifest.get("distribution", "archive"),
        "checkout_commit": revision,
        "checkout_clean": clean,
        "revision_match": revision == manifest.get("commit") if revision else None,
        "sha256": declared_sha256 or archive_sha256,
        "declared_sha256": declared_sha256,
        "archive_sha256": archive_sha256,
        "sha256_match": declared_sha256 == archive_sha256 if declared_sha256 and archive_sha256 else None,
        "_manifest_source": source,
        "archive_source": str(archive),
    }


def _relative(path: Path, report_dir: Path, run_root: Path) -> str:
    """Return a portable report link, preferring paths inside the run root."""
    try:
        path = path.resolve()
    except OSError:
        pass
    try:
        return Path(os.path.relpath(path, report_dir)).as_posix()
    except ValueError:
        return str(path)


def _operator_root(run_root: Path, operator: str) -> Path:
    # Accept both a batch root and a directly supplied operator/workspace root.
    if run_root.name == "kernel_eval" and run_root.is_dir():
        return run_root.parent
    for root in (run_root, run_root / "work", run_root / operator, run_root / operator / "work"):
        if (root / "kernel_eval").is_dir():
            return root
    if (run_root / "results" / "status.json").is_file():
        for root in (run_root.parent.parent, run_root.parent, run_root):
            if (root / "kernel_eval").is_dir() or (root / "evaluation.log").is_file():
                return root
    return run_root / operator


def _find_status(run_root: Path, operator: str, raw: Any, candidate: bool) -> Path | None:
    if raw:
        path = Path(str(raw)).expanduser()
        if not path.is_absolute():
            path = run_root / path
        try:
            if path.is_file() and path.resolve().is_relative_to(run_root.resolve()):
                return path
        except (OSError, ValueError):
            pass
    if candidate:
        direct = run_root / "results" / "status.json"
        if direct.is_file():
            return direct
    else:
        direct = run_root / "baseline_results" / "status.json"
        if direct.is_file():
            return direct
    reports = _operator_root(run_root, operator) / "kernel_eval"
    if candidate:
        direct = reports / "results" / "status.json"
        if direct.is_file():
            return direct
        paths = sorted(reports.glob(f"ascend_arena_{operator}_*/results/status.json"))
        return paths[-1] if paths else None
    path = reports / "init_results" / "status.json"
    if path.is_file():
        return path
    # Direct evaluation stores the baseline beside the candidate results.
    paths = sorted(reports.glob(f"ascend_arena_{operator}_*/baseline_results/status.json"))
    if paths:
        return paths[-1]
    direct = reports / "baseline_results" / "status.json"
    return direct if direct.is_file() else None


def _infer_operator_name(run_root: Path) -> str:
    """Infer an operator from a direct workspace/evaluation/candidate root."""
    for path in sorted(run_root.glob("*.json")):
        payload = _load_json(path)
        if payload.get("operator"):
            return str(payload["operator"])
    candidate_roots = [run_root, run_root / "kernel_eval", run_root / "work" / "kernel_eval"]
    for base in candidate_roots[:]:
        for path in sorted(base.glob("ascend_arena_*/")):
            if path not in candidate_roots:
                candidate_roots.append(path)
    for path in candidate_roots:
        for report in sorted(path.glob("*.json")):
            payload = _load_json(report)
            if payload.get("operator"):
                return str(payload["operator"])
        if path.name.startswith("ascend_arena_"):
            name = path.name.removeprefix("ascend_arena_")
            name = re.sub(r"_\d{8}_\d{6}_[0-9a-f]+$", "", name)
            if name.endswith("_run"):
                name = name[:-4]
            if name:
                return name
    metadata = _load_json(run_root / "workspace.json")
    if metadata.get("operator"):
        return str(metadata["operator"])
    if run_root.name == "kernel_eval":
        return run_root.parent.name
    if run_root.name == "work":
        return run_root.parent.name
    return run_root.name


def _find_candidate_dir(status_path: Path | None) -> Path | None:
    if status_path is None:
        return None
    candidate_dir = status_path.parent.parent
    return candidate_dir if candidate_dir.is_dir() else None


def _load_candidate_report(candidate_dir: Path | None) -> dict[str, Any]:
    if candidate_dir is None:
        return {}
    preferred = candidate_dir / f"{candidate_dir.name}.json"
    paths = [preferred] if preferred.is_file() else []
    paths.extend(path for path in sorted(candidate_dir.glob("*.json")) if path not in paths and path.name != "comparison.json")
    for path in paths:
        payload = _load_json(path)
        if payload and any(key in payload for key in ("operator", "summary", "setup_info", "candidate_status")):
            return payload
    return {}


def _correctness(results_dir: Path | None, report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    report_stages = {
        str(item.get("stage")): item
        for item in report.get("correctness_stages", [])
        if isinstance(item, dict) and item.get("stage")
    }
    for stage in STAGES:
        payload = _load_json(results_dir / f"{stage}-correctness_result.json") if results_dir else {}
        cases = payload.get("case_results")
        if not isinstance(cases, list):
            summary = report_stages.get(stage, {})
            values[stage] = {
                "success": summary.get("success"),
                "total": summary.get("total_cases"),
                "passed": summary.get("passed_cases"),
                "failed": summary.get("failed_cases"),
            }
            continue
        passed = sum(isinstance(case, dict) and case.get("passed") is True for case in cases)
        values[stage] = {
            "success": payload.get("success") is True and bool(cases) and passed == len(cases),
            "total": len(cases),
            "passed": passed,
            "failed": len(cases) - passed,
        }
    return values


def _comparison(candidate_dir: Path | None, report: dict[str, Any]) -> dict[str, Any]:
    comparison_path = candidate_dir / "comparison.json" if candidate_dir else None
    comparison = _load_json(comparison_path) if comparison_path else {}
    cases = comparison.get("cases")
    available = bool(comparison_path and comparison_path.is_file())
    if not isinstance(cases, list):
        cases = report.get("cases") if isinstance(report.get("cases"), list) else []
        available = available or bool(cases)
    valid = [
        case for case in cases
        if isinstance(case, dict)
        and case.get("candidate_passed") is True
        and isinstance(case.get("speedup"), (int, float))
        and not isinstance(case.get("speedup"), bool)
        and math.isfinite(float(case["speedup"]))
        and float(case["speedup"]) > 0
        and case.get("timing_comparable") is not False
        and all(
            isinstance(case.get(key), (int, float))
            and not isinstance(case.get(key), bool)
            and math.isfinite(float(case[key]))
            and float(case[key]) > 0
            for key in ("baseline_us", "candidate_us")
        )
    ]
    speedups = [float(case["speedup"]) for case in valid]
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    reported_average = summary.get("average_speedup")
    average = mean(speedups) if speedups else None
    return {
        "cases": cases,
        "available": available,
        "valid_cases": len(valid),
        "average_speedup": float(average) if isinstance(average, (int, float)) else None,
        "reported_average_speedup": (
            float(reported_average)
            if isinstance(reported_average, (int, float)) and math.isfinite(float(reported_average))
            else None
        ),
    }


def _operator_record(run_root: Path, item: dict[str, Any]) -> dict[str, Any]:
    operator = str(item.get("operator") or "unknown")
    baseline_status = _find_status(run_root, operator, item.get("baseline_status"), False)
    candidate_status = _find_status(run_root, operator, item.get("candidate_status"), True)
    candidate_dir = _find_candidate_dir(candidate_status)
    candidate_report = _load_candidate_report(candidate_dir)
    candidate_results = candidate_status.parent if candidate_status else None
    correctness = _correctness(candidate_results, candidate_report)
    comparison = _comparison(candidate_dir, candidate_report)
    baseline = _load_json(baseline_status) if baseline_status else {}
    candidate = _load_json(candidate_status) if candidate_status else {}
    device_id = item.get("device_id")
    if device_id is None:
        device_value = candidate_report.get("device") or candidate.get("device")
        match = re.search(r"(?:npu|device)\s*[:#]?\s*(\d+)", str(device_value), re.IGNORECASE)
        if match:
            device_id = int(match.group(1))
    profile_cases = candidate.get("profile_cases")
    if not isinstance(profile_cases, int):
        profile_cases = len(comparison["cases"]) or None

    # Prefer evidence files.  Summary booleans are only a fallback for older
    # or still-being-collected runs where the detailed files do not exist.
    if baseline_status and candidate_status:
        build_states = [baseline.get("build_success"), candidate.get("build_success")]
        compile_ok = False if any(state is False for state in build_states) else True if all(state is True for state in build_states) else None
    elif "compile" in item and not baseline_status and not candidate_status:
        compile_ok = item.get("compile") if isinstance(item.get("compile"), bool) else None
    else:
        compile_ok = None

    stage_states = [correctness[stage].get("success") for stage in STAGES]
    if any(state is not None for state in stage_states):
        correctness_ok = all(state is True for state in stage_states)
    elif "correctness" in item and not candidate_status:
        correctness_ok = item.get("correctness") if isinstance(item.get("correctness"), bool) else None
    else:
        correctness_ok = None

    if baseline_status and candidate_status:
        profile_states = [baseline.get("profile_success"), candidate.get("profile_success")]
        if any(state is False for state in profile_states):
            performance_ok = False
        elif not all(state is True for state in profile_states) or not comparison["available"]:
            performance_ok = None
        else:
            performance_ok = bool(profile_cases) and comparison["valid_cases"] > 0
    elif "performance" in item and not baseline_status and not candidate_status:
        performance_ok = item.get("performance") if isinstance(item.get("performance"), bool) else None
    else:
        performance_ok = None

    evidence_complete = bool(
        baseline_status
        and candidate_status
        and all(state is not None for state in stage_states)
        and comparison["available"]
        and comparison["valid_cases"] > 0
    )
    complete = evidence_complete and compile_ok is True and correctness_ok is True and performance_ok is True
    evidence_failure = (
        compile_ok is False or correctness_ok is False or performance_ok is False
        or any(state is False for state in stage_states)
        or baseline.get("success") is False or candidate.get("success") is False
        or item.get("success") is False
        or any(item.get(key) not in (None, 0) for key in ("baseline_returncode", "candidate_returncode"))
    )
    status = "FAIL" if evidence_failure else "PASS" if complete else "INCOMPLETE"
    success = True if status == "PASS" else False if status == "FAIL" else None
    operator_root = _operator_root(run_root, operator)
    return {
        "operator": operator,
        "device_id": device_id,
        "success": success,
        "status": status,
        "evidence_complete": evidence_complete,
        "compile": compile_ok,
        "correctness": correctness_ok,
        "performance": performance_ok,
        "baseline_returncode": item.get("baseline_returncode"),
        "candidate_returncode": item.get("candidate_returncode"),
        "profile_cases": profile_cases,
        "correctness_stages": correctness,
        "comparison": comparison,
        "baseline_status": baseline_status,
        "candidate_status": candidate_status,
        "candidate_dir": candidate_dir,
        "evaluation_log": operator_root / "evaluation.log",
        "report_json": candidate_dir / f"{candidate_dir.name}.json" if candidate_dir else None,
        "report_html": candidate_dir / f"{candidate_dir.name}.html" if candidate_dir else None,
        "trajectory_html": operator_root / "kernel_eval" / "optimization_trajectory.html",
    }


def _recorded_environment(operators: list[dict[str, Any]]) -> dict[str, Any]:
    for item in operators:
        report_path = item.get("report_json")
        report = _load_json(report_path) if isinstance(report_path, Path) else {}
        setup = report.get("setup_info") if isinstance(report.get("setup_info"), dict) else {}
        values = setup.get("environment") if isinstance(setup.get("environment"), dict) else {}
        if values:
            return values
    return {}


def _collect_environment(*, probe_framework: bool = True) -> dict[str, Any]:
    toolkit_home = os.environ.get("ASCEND_TOOLKIT_HOME", "")
    env_script = os.environ.get("CANN_ENV_SCRIPT", "")
    if not env_script and toolkit_home and (Path(toolkit_home) / "set_env.sh").is_file():
        env_script = str(Path(toolkit_home) / "set_env.sh")
    environment: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cann_env_script": env_script,
        "ascend_toolkit_home": toolkit_home,
        "ascend_opp_path": os.environ.get("ASCEND_OPP_PATH", ""),
        "chip": os.environ.get("ASCEND_ARENA_CHIP", ""),
        "docker_image": os.environ.get("CANN_BENCH_IMAGE", ""),
    }
    try:
        if not probe_framework:
            raise RuntimeError("use recorded evaluator environment")
        import torch

        environment["pytorch"] = getattr(torch, "__version__", "")
        environment["npu_count"] = int(torch.npu.device_count())
        environment["npu_name"] = str(torch.npu.get_device_name(0)) if environment["npu_count"] else ""
    except Exception:
        environment["pytorch"] = ""
        environment["npu_count"] = ""
        environment["npu_name"] = ""
    try:
        if not probe_framework:
            raise RuntimeError("use recorded evaluator environment")
        import torch_npu

        environment["torch_npu"] = str(getattr(torch_npu, "__version__", ""))
    except Exception:
        environment["torch_npu"] = ""
    environment["npu_smi"] = _run_text(["npu-smi", "info"], timeout=3)
    return environment


def _merge_recorded_environment(environment: dict[str, Any], operators: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer the environment captured by the evaluator over the report generator's shell."""
    recorded = _recorded_environment(operators)
    mapping = {
        "npu": "npu_name", "cpu": "machine", "cann": "cann", "driver": "driver",
        "pytorch": "pytorch", "pytorch_npu": "torch_npu", "python": "python",
        "os": "platform", "docker": "docker_image",
    }
    for source, target in mapping.items():
        if recorded.get(source) not in (None, ""):
            environment[target] = recorded[source]
    npu_name = str(environment.get("npu_name") or "")
    if "×" in npu_name and not environment.get("npu_count"):
        try:
            environment["npu_count"] = int(npu_name.rsplit("×", 1)[1].strip())
        except ValueError:
            pass
    return environment


def _feature_status(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    normalized = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "PASS": "PASS",
        "OK": "PASS",
        "SUCCESS": "PASS",
        "FAIL": "FAIL",
        "FAILED": "FAIL",
        "ERROR": "FAIL",
        "INCOMPLETE": "INCOMPLETE",
        "PENDING": "INCOMPLETE",
        "N_A": "N/A",
        "N/A": "N/A",
        "NA": "N/A",
        "NOT_APPLICABLE": "N/A",
        "SKIP": "N/A",
    }
    return aliases.get(normalized)


def _infer_feature_status(
    feature_id: str,
    *,
    run_root: Path,
    environment: dict[str, Any],
    skeleton: dict[str, Any],
    batch: dict[str, Any],
    operators: list[dict[str, Any]],
) -> tuple[str, str]:
    """Infer only what the run artifacts can prove; everything else is N/A."""
    if feature_id == "environment":
        if environment.get("source") == "recorded evaluator environment" and environment.get("npu_count") not in (None, "", 0) and (
            environment.get("cann") or environment.get("cann_env_script")
        ):
            return "PASS", "记录到 CANN 环境和 NPU 数量。"
    elif feature_id == "chip_support":
        target = f"{environment.get('chip', '')} {environment.get('npu_name', '')}".lower()
        if environment.get("source") == "recorded evaluator environment" and any(token in target for token in ("910", "a2", "a3")):
            return "PASS", f"记录到目标设备：{environment.get('npu_name') or environment.get('chip')}."
    elif feature_id == "skeleton_archive":
        if skeleton.get("distribution") == "submodule":
            if skeleton.get("revision_match") is False or skeleton.get("checkout_clean") is False:
                return "FAIL", "骨架子模块提交与清单不一致或存在本地修改。"
            if skeleton.get("revision_match") is True and skeleton.get("checkout_clean") is True:
                return "PASS", "骨架子模块提交与清单一致且工作区干净。"
            return "N/A", "未能验证骨架子模块检出状态。"
        if skeleton.get("sha256_match") is True:
            return "PASS", "压缩包实际 SHA256 与骨架清单一致。"
        if skeleton.get("declared_sha256") and skeleton.get("archive_sha256"):
            return "FAIL", "压缩包实际 SHA256 与骨架清单不一致。"
    elif feature_id == "evaluate_existing":
        if operators and all(item.get("status") == "PASS" for item in operators):
            return "PASS", "本次运行的算子评测证据完整且通过。"
        if any(item.get("status") == "FAIL" for item in operators):
            return "FAIL", "至少一个算子评测存在失败证据。"
        if operators:
            return "INCOMPLETE", "已开始评测，但仍缺少完整证据。"
    elif feature_id == "result_contract":
        if (run_root / "e2e_summary.json").is_file() and operators:
            return "PASS", "存在汇总和阶段结果证据。"
    elif feature_id == "staged_evaluation":
        if operators and all(
            all(
                item.get("correctness_stages", {}).get(stage, {}).get("success") is not None
                for stage in STAGES
            )
            for item in operators
        ):
            return "PASS", "每个算子均记录 smoke、complete、profile 三阶段结果。"
    elif feature_id == "performance_collection":
        if operators and all(item.get("performance") is True for item in operators):
            return "PASS", "每个算子均有通过正确性约束的性能对比结果。"
    elif feature_id == "batch_acceptance":
        if (run_root / "e2e_summary.json").is_file() and batch.get("num_processes") not in (None, ""):
            return "PASS", "存在批量验收汇总和并发配置。"
    elif feature_id == "diagnostics":
        if operators and all(item.get("status") in {"PASS", "FAIL", "INCOMPLETE"} for item in operators):
            return "PASS", "每个运行项均有机器可读状态。"
    # Agent, Docker, workspace and skeleton behavior need a separately captured
    # command/result; inferring them from file existence would overclaim.
    return "N/A", "本次运行未提供该功能的独立证据。"


def _load_feature_input(run_root: Path, feature_results: Path | None) -> tuple[Path | None, dict[str, Any]]:
    source = feature_results.expanduser() if feature_results else run_root / "feature_results.json"
    payload = _load_json(source)
    raw = payload.get("features") if isinstance(payload.get("features"), dict) else {}
    return (source if source.is_file() else None), raw


def _feature_evidence_exists(raw: Any, run_root: Path, report_dir: Path) -> bool:
    if not raw:
        return False
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        report_path = report_dir / path
        path = report_path if report_path.is_file() else run_root / path
    return path.is_file()


def _build_features(
    run_root: Path,
    *,
    report_dir: Path,
    environment: dict[str, Any],
    skeleton: dict[str, Any],
    batch: dict[str, Any],
    operators: list[dict[str, Any]],
    feature_results: Path | None,
) -> tuple[list[dict[str, Any]], Path | None]:
    source, supplied = _load_feature_input(run_root, feature_results)
    definitions = {item["id"]: item for item in FEATURE_DEFINITIONS}
    records: list[dict[str, Any]] = []
    for feature_id, definition in definitions.items():
        entry = supplied.get(feature_id) if isinstance(supplied.get(feature_id), dict) else {}
        supplied_status = _feature_status(entry.get("status"))
        inferred_status, inferred_actual = _infer_feature_status(
            feature_id,
            run_root=run_root,
            environment=environment,
            skeleton=skeleton,
            batch=batch,
            operators=operators,
        )
        evidence = list(entry.get("evidence") or []) if isinstance(entry.get("evidence"), list) else []
        if not evidence and supplied_status is None and inferred_status == "PASS":
            if feature_id == "skeleton_archive":
                evidence = [
                    path for path in (skeleton.get("_manifest_source"), skeleton.get("archive_source")) if path
                ]
            elif feature_id == "environment" and operators and operators[0].get("report_json"):
                evidence = [operators[0]["report_json"]]
            elif (run_root / "e2e_summary.json").is_file():
                evidence = ["e2e_summary.json"]
        status = supplied_status or ("INCOMPLETE" if entry.get("status") not in (None, "") else inferred_status)
        note = entry.get("note") or ""
        if supplied_status == "PASS" and (not entry.get("actual") or not evidence or not any(
            _feature_evidence_exists(raw, run_root, report_dir) for raw in evidence
        )):
            status = "INCOMPLETE"
            note = note or "PASS 需要 actual 和可读 evidence；请补充后重新生成。"
        elif supplied_status == "FAIL" and not (
            entry.get("actual")
            or entry.get("returncode") is not None
            or any(_feature_evidence_exists(raw, run_root, report_dir) for raw in evidence)
        ):
            status = "INCOMPLETE"
            note = note or "FAIL 需要实际错误结果或可读 evidence。"
        records.append(
            {
                "id": feature_id,
                "name": entry.get("name") or definition["name"],
                "readme": entry.get("readme") or definition["readme"],
                "command": entry.get("command") or definition["command"],
                "expected": entry.get("expected") or definition["expected"],
                "actual": entry.get("actual") or inferred_actual,
                "returncode": entry.get("returncode"),
                "status": status,
                "evidence": evidence,
                "note": note,
            }
        )
    # Keep user-defined checks visible without requiring a code change for each
    # repository extension.
    for feature_id, entry in supplied.items():
        if feature_id in definitions or not isinstance(entry, dict):
            continue
        supplied_status = _feature_status(entry.get("status"))
        evidence = list(entry.get("evidence") or []) if isinstance(entry.get("evidence"), list) else []
        status = supplied_status or "INCOMPLETE"
        note = entry.get("note") or ""
        if supplied_status == "PASS" and (not entry.get("actual") or not evidence or not any(
            _feature_evidence_exists(raw, run_root, report_dir) for raw in evidence
        )):
            status = "INCOMPLETE"
            note = note or "PASS 需要 actual 和可读 evidence；请补充后重新生成。"
        elif supplied_status == "FAIL" and not (
            entry.get("actual")
            or entry.get("returncode") is not None
            or any(_feature_evidence_exists(raw, run_root, report_dir) for raw in evidence)
        ):
            status = "INCOMPLETE"
            note = note or "FAIL 需要实际错误结果或可读 evidence。"
        records.append(
            {
                "id": str(feature_id),
                "name": entry.get("name") or str(feature_id),
                "readme": entry.get("readme") or "自定义",
                "command": entry.get("command") or "N/A",
                "expected": entry.get("expected") or "N/A",
                "actual": entry.get("actual") or "N/A",
                "returncode": entry.get("returncode"),
                "status": status,
                "evidence": evidence,
                "note": note,
            }
        )
    return records, source


def _feature_summary(features: list[dict[str, Any]]) -> dict[str, int | str]:
    counts = {status: sum(item.get("status") == status for item in features) for status in FEATURE_STATUSES}
    if counts["FAIL"]:
        result = "FAIL"
    elif counts["INCOMPLETE"] or counts["N/A"] or not features:
        result = "INCOMPLETE"
    else:
        result = "PASS"
    return {**counts, "total": len(features), "result": result}


def _copy_evidence(data: dict[str, Any], report_dir: Path) -> None:
    """Copy small, reviewable artifacts without duplicating profiler output trees."""
    evidence_dir = report_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(str(data["run_root"]))

    def copy(source: Path | None, destination: Path) -> Path | None:
        if source is None or not source.is_file():
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            if source.resolve() == destination.resolve():
                return destination
        except OSError:
            pass
        shutil.copy2(source, destination)
        return destination

    summary = copy(run_root / "e2e_summary.json", evidence_dir / "e2e_summary.json")
    config = copy(run_root / "config.json", evidence_dir / "config.json")
    feature_results_source = data.get("feature_results_source")
    feature_results_path = Path(str(feature_results_source)) if feature_results_source else None
    feature_results_copy = copy(feature_results_path, evidence_dir / "feature_results.json")
    data["evidence"]["summary"] = os.path.relpath(summary, report_dir) if summary else ""
    data["evidence"]["config"] = os.path.relpath(config, report_dir) if config else ""
    data["evidence"]["feature_results"] = os.path.relpath(feature_results_copy, report_dir) if feature_results_copy else ""
    manifest_source = data["skeleton"].get("_manifest_source")
    if not isinstance(manifest_source, Path):
        manifest_source = Path(__file__).resolve().parents[1] / "ascend_arena_e2e" / "dependency" / "ops-skeleton.json"
    manifest = copy(manifest_source, evidence_dir / "ops-skeleton.json")
    data["skeleton"]["manifest_path"] = os.path.relpath(manifest, report_dir) if manifest else ""
    archive_source = data["skeleton"].get("archive_source")
    archive = copy(Path(str(archive_source)) if archive_source else None, evidence_dir / "ops-skeleton.tar.gz")
    data["skeleton"]["archive_path"] = os.path.relpath(archive, report_dir) if archive else ""

    for item in data["operators"]:
        operator_dir = evidence_dir / _safe_component(item["operator"])
        candidate_dir = item.get("candidate_dir")
        sources = {
            "baseline_status.json": item.get("baseline_status"),
            "candidate_status.json": item.get("candidate_status"),
            "report.json": item.get("report_json"),
            "report.html": item.get("report_html"),
            "optimization_trajectory.html": item.get("trajectory_html"),
            "evaluation.log": item.get("evaluation_log"),
        }
        if isinstance(candidate_dir, Path):
            sources["comparison.json"] = candidate_dir / "comparison.json"
            results_dir = candidate_dir / "results"
            for stage in STAGES:
                sources[f"{stage}-correctness_result.json"] = results_dir / f"{stage}-correctness_result.json"
        baseline_status = item.get("baseline_status")
        candidate_status = item.get("candidate_status")
        if isinstance(baseline_status, Path):
            for stage in STAGES:
                sources[f"baseline_{stage}-correctness_result.json"] = baseline_status.parent / f"{stage}-correctness_result.json"
            for name in ("compile.log", "profile.log", "stage-process.log",
                         *(f"{stage}-correctness.log" for stage in STAGES)):
                sources[f"baseline_{name}"] = baseline_status.parent / name
        if isinstance(candidate_status, Path):
            for name in ("compile.log", "profile.log", "stage-process.log",
                         *(f"{stage}-correctness.log" for stage in STAGES)):
                sources[f"candidate_{name}"] = candidate_status.parent / name
        for kind, status_path in (("baseline", baseline_status), ("candidate", candidate_status)):
            if isinstance(status_path, Path):
                for name in ("compile_result.json", "result.json"):
                    sources[("baseline_" if kind == "baseline" else "") + name] = status_path.parent / name
        copied: dict[str, str] = {}
        for name, source in sources.items():
            copied_path = copy(source if isinstance(source, Path) else None, operator_dir / name)
            if copied_path:
                copied[name] = os.path.relpath(copied_path, report_dir)
        item["evidence"] = copied
        for kind in ("baseline", "candidate"):
            archived_path = copied.get(f"{kind}_status.json")
            if not archived_path:
                continue
            path = report_dir / archived_path
            status = _load_json(path)
            # Status paths stay relative to the archived status file.
            for phase in status.get("stages", {}).values():
                for key, prefix in (("result_json", "baseline_" if kind == "baseline" else ""),
                                    ("log", f"{kind}_")):
                    name = prefix + Path(phase[key]).name if phase.get(key) else ""
                    phase[key] = name if name in copied else None
            for key in ("compile_log", "profile_log"):
                if key in status:
                    name = f"{kind}_" + Path(status[key]).name if status[key] else ""
                    status[key] = name if name in copied else None
            path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for feature in data.get("features", []):
        if not isinstance(feature, dict):
            continue
        copied: list[str] = []
        feature_dir = evidence_dir / "features" / _safe_component(feature.get("id"))
        for raw in feature.get("evidence", []):
            if not raw:
                continue
            source = Path(str(raw)).expanduser()
            if not source.is_absolute():
                # Evidence may already be a screenshot copied into the report,
                # or a path relative to the run root.
                report_source = report_dir / source
                source = report_source if report_source.is_file() else run_root / source
            if not source.is_file():
                continue
            destination = feature_dir / _safe_component(source.name)
            copied_path = copy(source, destination)
            if copied_path:
                copied.append(os.path.relpath(copied_path, report_dir))
        feature["evidence"] = copied


def _collect_screenshots(
    report_dir: Path,
    screenshots_dir: Path | None,
    screenshot_files: Iterable[Path],
) -> list[dict[str, str]]:
    target = report_dir / "screenshots"
    candidates: list[Path] = []
    source_dir = screenshots_dir if screenshots_dir and screenshots_dir.is_dir() else target
    if source_dir.is_dir():
        candidates.extend(sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES))
    candidates.extend(path for path in screenshot_files if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    target.mkdir(parents=True, exist_ok=True)
    output: list[dict[str, str]] = []
    used: set[str] = set()
    for source in candidates:
        name = source.name
        stem, suffix = source.stem, source.suffix
        index = 2
        while name in used or (target / name).exists() and source.resolve() != (target / name).resolve():
            name = f"{stem}_{index}{suffix}"
            index += 1
        destination = target / name
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        used.add(name)
        output.append({"name": name, "path": f"screenshots/{name}"})
    return output


def build_acceptance_data(
    run_root: Path,
    report_dir: Path,
    *,
    command: str = "",
    screenshots: list[dict[str, str]] | None = None,
    feature_results: Path | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root).expanduser().resolve()
    summary = _load_json(run_root / "e2e_summary.json")
    config = _load_json(run_root / "config.json")
    parameters = config.get("parameters") if isinstance(config.get("parameters"), dict) else {}
    progress = _load_json(run_root / "progress.json").get("operators", {})
    raw_results = summary.get("results") if isinstance(summary.get("results"), list) else []
    if not raw_results:
        names = parameters.get("operators", [])
        if not isinstance(names, list):
            names = []
        progress = _load_json(run_root / "progress.json").get("operators", {})
        if not names and isinstance(progress, dict):
            names = list(progress)
        if not names:
            summary_names = summary.get("operators", [])
            if isinstance(summary_names, list):
                names = summary_names
        if not names and (
            (run_root / "kernel_eval").is_dir()
            or run_root.name == "kernel_eval"
            or (run_root / "results" / "status.json").is_file()
        ):
            operator = _infer_operator_name(run_root)
            names = [operator]
        raw_results = [{"operator": name} for name in names if name]
    expected_names = summary.get("operators") or parameters.get("operators") or []
    if isinstance(expected_names, list):
        reported_names = {item.get("operator") for item in raw_results if isinstance(item, dict)}
        raw_results = [*raw_results, *({"operator": name} for name in expected_names
                                     if name and name not in reported_names)]
    normalized_results = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        item = dict(item)
        state = progress.get(item.get("operator"), {}) if isinstance(progress, dict) else {}
        if isinstance(state, dict) and item.get("device_id") is None:
            item["device_id"] = state.get("device_id")
        normalized_results.append(item)
    operators = [_operator_record(run_root, item) for item in normalized_results]
    skeleton = _skeleton_metadata(run_root)
    statuses = [item.get("status") for item in operators]
    operator_result = (
        "PASS" if operators and all(status == "PASS" for status in statuses)
        else "FAIL" if any(status == "FAIL" for status in statuses)
        else "INCOMPLETE"
    )
    recorded_environment = _recorded_environment(operators)
    environment = _collect_environment(probe_framework=not bool(recorded_environment))
    environment = _merge_recorded_environment(environment, operators)
    environment["source"] = "recorded evaluator environment" if recorded_environment else "report generator host"
    batch = {
        "created_at": summary.get("created_at", ""),
        "operators": summary.get("operators", [item["operator"] for item in operators]),
        "idle_devices": summary.get("idle_devices", []),
        "busy_devices": summary.get("busy_devices", []),
        "num_processes": summary.get("num_processes", ""),
        "success": summary.get("success"),
        "agent": parameters.get("agent") or parameters.get("harness") or "",
        "docker": parameters.get("docker", False),
        "docker_image": parameters.get("docker_image", ""),
    }
    features, feature_results_source = _build_features(
        run_root,
        report_dir=report_dir,
        environment=environment,
        skeleton=skeleton,
        batch=batch,
        operators=operators,
        feature_results=feature_results,
    )
    repository_root = Path(__file__).resolve().parents[2]
    return {
        "report_id": run_root.name,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_root": str(run_root),
        # `result` remains the operator-run conclusion for callers that used
        # the original report schema.  `feature_result` is the repository-level
        # completeness conclusion shown first in the new report.
        "result": operator_result,
        "operator_result": operator_result,
        "feature_result": _feature_summary(features)["result"],
        "feature_summary": _feature_summary(features),
        "feature_profile": "full-repository matrix" if feature_results_source else "runtime-only",
        "command": command or str(config.get("command") or parameters.get("command") or ""),
        "environment": environment,
        "repository": {
            "commit": _run_text(["git", "rev-parse", "HEAD"], cwd=repository_root),
            "branch": _run_text(["git", "branch", "--show-current"], cwd=repository_root),
            "dirty": bool(_run_text(["git", "status", "--porcelain"], cwd=repository_root)),
        },
        "skeleton": skeleton,
        "batch": batch,
        "operators": operators,
        "features": features,
        "feature_results_source": feature_results_source,
        "screenshots": screenshots or [],
        "evidence": {
            "summary": _relative(run_root / "e2e_summary.json", report_dir, run_root),
            "config": _relative(run_root / "config.json", report_dir, run_root) if (run_root / "config.json").is_file() else "",
            "feature_results": _relative(feature_results_source, report_dir, run_root) if feature_results_source else "",
        },
    }


def _status(value: Any) -> str:
    return "PASS" if value is True else "FAIL" if value is False else "N/A"


def _evidence_status(item: dict[str, Any]) -> str:
    if item.get("evidence_complete") is True:
        return "PASS"
    return "FAIL" if item.get("status") == "FAIL" else "INCOMPLETE"


def _link(path: Path | None, report_dir: Path, run_root: Path, label: str) -> str:
    if path is None or not path.exists():
        return "N/A"
    try:
        path.resolve().relative_to(report_dir.resolve())
    except (OSError, ValueError):
        # Acceptance reports must not contain links that escape the portable
        # report directory or silently point back to the producer's workspace.
        return "N/A"
    return f"[{label}]({_relative(path, report_dir, run_root)})"


def _evidence_link(item: dict[str, Any], name: str, report_dir: Path, run_root: Path, label: str) -> str:
    evidence = item.get("evidence", {})
    path = Path(str(evidence[name])) if isinstance(evidence, dict) and evidence.get(name) else None
    if path is not None and not path.is_absolute():
        path = report_dir / path
    return _link(path, report_dir, run_root, label)


def _md_cell(value: Any) -> str:
    text = "N/A" if value in (None, "") else str(value)
    return html.escape(text.replace("|", "\\|"), quote=False).replace("\n", "<br>")


def _feature_evidence_links(feature: dict[str, Any], report_dir: Path) -> str:
    links = []
    for raw in feature.get("evidence", []):
        path = Path(str(raw))
        if not path.is_absolute():
            path = report_dir / path
        if path.is_file():
            links.append(_link(path, report_dir, Path("."), path.name))
    return " ".join(links) or "N/A"


def render_acceptance_report(data: dict[str, Any], report_dir: Path) -> str:
    run_root = Path(str(data["run_root"]))
    env = data["environment"]
    batch = data["batch"]
    operators = data["operators"]
    features = data.get("features", [])
    feature_summary = data.get("feature_summary", {})
    summary_evidence = data.get("evidence", {}).get("summary") if isinstance(data.get("evidence"), dict) else ""
    summary_path = Path(str(summary_evidence)) if summary_evidence else run_root / "e2e_summary.json"
    if not summary_path.is_absolute():
        summary_path = report_dir / summary_path
    config_evidence = data.get("evidence", {}).get("config") if isinstance(data.get("evidence"), dict) else ""
    config_path = Path(str(config_evidence)) if config_evidence else None
    if config_path is not None and not config_path.is_absolute():
        config_path = report_dir / config_path
    feature_evidence = data.get("evidence", {}).get("feature_results") if isinstance(data.get("evidence"), dict) else ""
    feature_path = Path(str(feature_evidence)) if feature_evidence else None
    if feature_path is not None and not feature_path.is_absolute():
        feature_path = report_dir / feature_path

    compile_pass = sum(item.get("compile") is True for item in operators)
    correctness_pass = sum(item.get("correctness") is True for item in operators)
    performance_pass = sum(item.get("performance") is True for item in operators)
    profile_cases = sum(item.get("profile_cases") or 0 for item in operators)
    speedups = [item["comparison"].get("average_speedup") for item in operators]
    speedups = [float(value) for value in speedups if isinstance(value, (int, float)) and math.isfinite(float(value))]
    average_speedup = mean(speedups) if speedups else None
    operator_names = ", ".join(f"`{item['operator']}`" for item in operators) or "N/A"
    executed = sum(feature.get("status") != "N/A" for feature in features)

    lines = [
        "# Ascend Arena 仓库功能完整性测试报告",
        "",
        "本报告以 README 中定义的仓库功能为验收对象；算子结果只作为总体摘要和证据，不在正文逐个展开。",
        "",
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 报告 ID | `{data['report_id']}` |",
        f"| 功能完整性结论 | **{data.get('feature_result', 'INCOMPLETE')}** |",
        f"| 本次算子评测结论 | **{data.get('operator_result', data.get('result', 'INCOMPLETE'))}** |",
        f"| 测试 profile | `{data.get('feature_profile', 'runtime-only')}`（未提供独立证据的功能标记 N/A） |",
        f"| 已执行功能 | `{executed}/{len(features)}`（PASS/FAIL/INCOMPLETE；N/A 表示本次未执行） |",
        f"| 生成时间 | `{data['created_at']}` |",
        f"| 测试汇总时间 | `{batch.get('created_at') or 'N/A'}` |",
        f"| 测试机器 | `{env.get('hostname') or 'N/A'}` |",
        f"| 运行目录 | `{run_root.name}` |",
        "",
        "## 1. 测试范围与复现信息",
        "",
        f"- 仓库分支 / commit：`{data['repository'].get('branch') or 'N/A'}` / `{data['repository'].get('commit') or 'N/A'}`",
        f"- 工作区状态：`{'dirty' if data['repository'].get('dirty') else 'clean'}`",
        f"- 本次算子样本数量：`{len(operators)}`；名称仅用于追溯：{operator_names}",
        f"- 完整命令：`{data['command'] or '请补充实际执行命令'}`",
        f"- 原始汇总：{_link(summary_path, report_dir, run_root, 'e2e_summary.json')}",
        f"- 运行配置：{_link(config_path, report_dir, run_root, 'config.json')}",
        f"- 功能结果输入：{_link(feature_path, report_dir, run_root, 'feature_results.json') if feature_path else '未提供（请按功能矩阵补充）'}",
        "",
        "## 2. 环境与第三方设施",
        "",
        "| 字段 | 值 |",
        "|---|---|",
    ]
    env_fields = (
        ("主机平台", env.get("platform")), ("CPU 架构", env.get("machine")),
        ("环境来源", env.get("source")), ("NPU 型号", env.get("npu_name")),
        ("NPU 数量", env.get("npu_count")), ("Python", env.get("python")),
        ("PyTorch", env.get("pytorch")), ("torch_npu", env.get("torch_npu")),
        ("芯片目标", env.get("chip")), ("驱动", env.get("driver")),
        ("CANN 版本", env.get("cann")), ("CANN 环境脚本", env.get("cann_env_script")),
        ("CANN Toolkit", env.get("ascend_toolkit_home")),
        ("Docker 镜像", env.get("docker_image") or batch.get("docker_image") or "裸机/未设置"),
    )
    lines.extend(f"| {label} | `{_md_cell(value)}` |" for label, value in env_fields)
    lines.extend([
        "",
        f"- CANN Bench 骨架仓：`{data['skeleton'].get('repository') or 'N/A'}`",
        f"- 骨架 commit：`{data['skeleton'].get('commit') or 'N/A'}`",
        f"- 骨架分发方式：`{data['skeleton'].get('distribution') or 'N/A'}`；检出 commit：`{data['skeleton'].get('checkout_commit') or 'N/A'}`；提交一致性：`{_status(data['skeleton'].get('revision_match'))}`；工作区干净：`{_status(data['skeleton'].get('checkout_clean'))}`",
        f"- 骨架清单 SHA256：`{data['skeleton'].get('sha256') or 'N/A'}`",
        f"- 骨架压缩包实际 SHA256：`{data['skeleton'].get('archive_sha256') or 'N/A'}`；一致性：`{_status(data['skeleton'].get('sha256_match') if data['skeleton'].get('archive_sha256') else None)}`",
        f"- 骨架清单：{_link((report_dir / Path(str(data['skeleton'].get('manifest_path')))) if data['skeleton'].get('manifest_path') else None, report_dir, run_root, 'ops-skeleton.json') if data['skeleton'].get('manifest_path') else 'N/A'}",
        f"- 骨架压缩包：{_link((report_dir / Path(str(data['skeleton'].get('archive_path')))) if data['skeleton'].get('archive_path') else None, report_dir, run_root, 'ops-skeleton.tar.gz') if data['skeleton'].get('archive_path') else 'N/A'}",
        "",
        "### npu-smi 信息",
        "",
        "```text",
        env.get("npu_smi") or "未能执行 npu-smi info，请手工补充。",
        "```",
        "",
        "## 3. README 功能验收矩阵",
        "",
        "状态规则：`PASS` 有可复核证据且结果符合预期；`FAIL` 有明确失败证据；`INCOMPLETE` 已执行但证据/结果不完整；`N/A` 本次未执行，不能计入完整通过。",
        "",
        "| 功能 ID | README 功能 | 执行命令 | 预期产物/结果 | 实际结果 | 证据 | 状态 | 备注 |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for feature in features:
        lines.append(
            f"| `{_md_cell(feature.get('id'))}` | {_md_cell(feature.get('name'))}<br><small>{_md_cell(feature.get('readme'))}</small> | "
            f"`{_md_cell(feature.get('command'))}` | {_md_cell(feature.get('expected'))} | {_md_cell(feature.get('actual'))} | "
            f"{_feature_evidence_links(feature, report_dir)} | **{_md_cell(feature.get('status'))}** | {_md_cell(feature.get('note'))} |"
        )
    lines.extend([
        "",
        f"功能状态汇总：PASS `{feature_summary.get('PASS', 0)}`，FAIL `{feature_summary.get('FAIL', 0)}`，INCOMPLETE `{feature_summary.get('INCOMPLETE', 0)}`，N/A `{feature_summary.get('N/A', 0)}`。",
        "",
        "## 4. 算子评测总体摘要",
        "",
        "下表只说明本次仓库功能测试使用的算子样本规模，不展开单个算子或 case。详细机器可读结果仍保存在 `evidence/`。",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 算子样本数 | `{len(operators)}` |",
        f"| 编译通过 | `{compile_pass}/{len(operators)}` |",
        f"| 正确性通过 | `{correctness_pass}/{len(operators)}` |",
        f"| 性能采集通过 | `{performance_pass}/{len(operators)}` |",
        f"| 有效 profile case 数 | `{profile_cases}` |",
        f"| 平均加速比 | `{average_speedup:.4f}x` |" if average_speedup is not None else "| 平均加速比 | `N/A` |",
        "",
        "> 平均加速比仅用于概览；性能是否提升应以归档的 `comparison.json` 和 correctness 结果共同判断。",
        "",
        "## 5. 结果、证据与截图",
        "",
        "证据目录保留汇总 JSON、状态文件、失败日志和必要的评测 HTML，便于审核人员按需追溯；正文不复制逐 case 表格。",
        f"- 汇总证据：{_link(summary_path, report_dir, run_root, 'e2e_summary.json')}",
        f"- 配置证据：{_link(config_path, report_dir, run_root, 'config.json')}",
        f"- 功能结果：{_link(feature_path, report_dir, run_root, 'feature_results.json') if feature_path else 'N/A'}",
        "",
    ])
    screenshots = data.get("screenshots") or []
    if screenshots:
        lines.extend(["| 文件 | 用途 |", "|---|---|"])
        lines.extend(f"| ![{image['name']}]({image['path']}) | `{image['name']}` |" for image in screenshots)
    else:
        lines.extend([
            "未提供截图。将审核所需图片放入 `screenshots/` 后重新运行生成脚本：",
            "",
            "- `01_environment.png`：CANN、设备和空闲卡信息",
            "- `02_command.png`：实际命令与开始时间",
            "- `03_batch_summary.png`：终端摘要或 e2e_summary.json",
            "- `04_report.png`：总 HTML 报告或功能结果页面",
        ])
    lines.extend([
        "",
        "## 6. 审核清单",
        "",
        "- [ ] README 功能矩阵中的每一项都已执行，或明确标注 N/A 的适用范围",
        "- [ ] CANN、驱动、PyTorch、torch_npu、Python、芯片和 Docker 信息已核对",
        "- [ ] 第三方骨架 commit 与 SHA256 已核对",
        "- [ ] 已有算子评测的编译、smoke、complete、profile 和性能摘要可追溯",
        "- [ ] Agent、批量调度、断点续传、Docker、workspace-only 和外部 Agent 均有命令/截图/产物证据",
        "- [ ] 失败项附有可读日志，成功项的日志清理行为已理解",
        "- [ ] 报告和截图中没有 API Key、Token、密码等敏感信息",
        "",
        "| 审核人 | 日期 | 结论 | 备注 |",
        "|---|---|---|---|",
        "|  |  | PASS / FAIL / INCOMPLETE |  |",
        "",
    ])
    return "\n".join(lines)


def _portable_data(data: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """Remove host-local source paths from the machine-readable report artifact."""
    # The Markdown report already links copied evidence.  Keep the JSON equally
    # movable so a reviewer can unpack the whole acceptance_report directory on
    # another host without seeing stale source paths from the producer machine.
    portable = json.loads(json.dumps(data, ensure_ascii=False, default=str))
    portable["run_root"] = run_root.name
    portable["portable"] = True
    environment = portable.get("environment")
    if isinstance(environment, dict):
        for field in ("cann_env_script", "ascend_toolkit_home", "ascend_opp_path"):
            value = environment.get(field)
            if isinstance(value, str) and value:
                environment[field] = Path(value).name
        environment["paths_are_host_local"] = True
    portable.get("skeleton", {}).pop("archive_source", None)
    portable.get("skeleton", {}).pop("_manifest_source", None)
    portable.pop("feature_results_source", None)
    fields = {
        "baseline_status": "baseline_status.json",
        "candidate_status": "candidate_status.json",
        "report_json": "report.json",
        "report_html": "report.html",
        "trajectory_html": "optimization_trajectory.html",
        "evaluation_log": "evaluation.log",
    }
    for item in portable.get("operators", []):
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        for field, evidence_name in fields.items():
            item[field] = evidence.get(evidence_name, "")
        item["candidate_dir"] = f"evidence/{_safe_component(item.get('operator'))}" if evidence else ""
    portable["evidence"] = portable.get("evidence", {})
    portable["evidence"]["feature_results"] = portable["evidence"].get("feature_results", "")
    return portable


def write_acceptance_report(
    run_root: Path,
    output: Path | None = None,
    *,
    command: str = "",
    screenshots_dir: Path | None = None,
    screenshot_files: Iterable[Path] = (),
    feature_results: Path | None = None,
) -> Path:
    run_root = Path(run_root).expanduser().resolve()
    output = (Path(output).expanduser().resolve() if output else run_root / "acceptance_report" / "acceptance_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    screenshots = _collect_screenshots(output.parent, screenshots_dir, screenshot_files)
    data = build_acceptance_data(
        run_root,
        output.parent,
        command=command,
        screenshots=screenshots,
        feature_results=feature_results,
    )
    _copy_evidence(data, output.parent)
    output.write_text(render_acceptance_report(data, output.parent), encoding="utf-8")
    portable = _portable_data(data, run_root)
    (output.parent / "report_data.json").write_text(
        json.dumps(portable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output
