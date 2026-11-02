#!/usr/bin/python3
# coding=utf-8

"""Default staged evaluator for cann-bench local runs.

Stages:
1. compile: build the submitted cann_bench package.
2. correctness: run precision-only evaluation and keep pass/fail as authority.
3. performance: profile only correctness-passed cases, then merge timings back.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .config import Config, get_project_root, set_config
from .data.tensor_cache import add_cache_arguments, configure_cache
from .data.data_generator import INPUT_DIST_CHOICES


CaseKey = Tuple[str, int]


# ---------------------------------------------------------------------------
# CANN 构建/运行版本一致性护栏
# ---------------------------------------------------------------------------
#
# 背景（2026-08 incident）：950PR runner 容器内并存多套 CANN（如 cann-9.1.0 与
# cann-9.1.0-beta.1，ascend-toolkit/latest 指向 beta.1）。提交方 build.sh 若重新
# source 默认 toolkit，会用与评测运行不同的 CANN 构建 kernel。版本错配的 kernel
# 在 profiler 下经 __asc_LaunchAndProfiling → AscendCGetProfkTypeImpl 查询 kType
# 元数据时，rtFunctionGetMetaInfo 返回成功但元数据指针为 NULL，CANN 自带 stub
# AscendCFunctionGetMetaInfoKtype 不做空指针检查直接解引用，评测子进程 SIGSEGV。
# 正确性阶段不开 profiler，因此该类错配表现为"精度全过、性能采集必崩"。
#
# 本护栏在编译成功后、正确性/性能阶段前比对构建日志与运行环境使用的 CANN
# 版本目录名；可判定的不一致直接按编译失败终止，给出可操作的错误信息，
# 避免评测进程崩溃且分数被静默按 0 计入。

_ASCEND_CANN_DIR_RE = re.compile(r"/usr/local/Ascend/(cann-[\w.\-]+)")


def _cann_home_dirname(path: str) -> Optional[str]:
    """把 CANN 安装路径规范化为版本目录名（如 cann-9.1.0）。

    解析 ascend-toolkit/latest 等符号链接；无法识别为 cann-* 目录时返回 None。
    """
    if not path:
        return None
    resolved = os.path.realpath(path)
    name = os.path.basename(resolved.rstrip("/"))
    return name if name.startswith("cann-") else None


def _detect_build_cann_dirname(source_dir: Path) -> Optional[str]:
    """从编译产物识别构建使用的 CANN 版本目录名。

    优先 compile_commands.json（cmake 精确记录编译命令），回退统计 _compile.log
    中出现最多的 /usr/local/Ascend/cann-* 路径。识别不到返回 None（判不了时
    不误伤——纯 Python 提交等场景没有 CANN 编译痕迹）。
    """
    compile_commands = source_dir / "build" / "compile_commands.json"
    if compile_commands.is_file():
        try:
            commands = json.loads(compile_commands.read_text(errors="replace"))
        except (json.JSONDecodeError, OSError):
            commands = []
        kernel_versions: Counter = Counter()
        fallback_versions: Counter = Counter()
        for entry in commands:
            cmd = entry.get("command", "") if isinstance(entry, dict) else ""
            found = _ASCEND_CANN_DIR_RE.findall(cmd)
            if not found:
                continue
            if "bisheng" in cmd or "ccec_compiler" in cmd:
                kernel_versions.update(found)
            else:
                fallback_versions.update(found)
        for versions in (kernel_versions, fallback_versions):
            if versions:
                return versions.most_common(1)[0][0]

    compile_log = source_dir / "_compile.log"
    if compile_log.is_file():
        try:
            text = compile_log.read_text(errors="replace")
        except OSError:
            return None
        versions = Counter(_ASCEND_CANN_DIR_RE.findall(text))
        if versions:
            return versions.most_common(1)[0][0]
    return None


def _check_build_cann_consistency(args: argparse.Namespace) -> Optional[str]:
    """构建与运行 CANN 版本一致性检查。

    返回 None 表示一致或无法判定（不误伤）；否则返回可操作的 mismatch 描述。
    """
    if not args.source_dir:
        return None
    build_dirname = _detect_build_cann_dirname(Path(args.source_dir))
    run_dirname = _cann_home_dirname(os.environ.get("ASCEND_HOME_PATH", ""))
    if not build_dirname or not run_dirname or build_dirname == run_dirname:
        return None
    return (
        f"提交代码构建使用的 CANN 版本 ({build_dirname}) 与评测运行环境 "
        f"({run_dirname}) 不一致。版本错配的 kernel 在 profiler 性能采集阶段会触发 "
        f"CANN 运行时元数据查询空指针解引用（SIGSEGV），导致性能项被按 0 计入。"
        f"请修正 build.sh 使其使用评测方提供的 CANN 环境（ASCEND_HOME_PATH），"
        f"不要重新 source 其他版本的 set_env.sh。"
    )


def _write_cann_mismatch_report(args: argparse.Namespace, bench_root: str, message: str) -> None:
    """把 CANN 版本错配合成为提交级编译失败报告（result_stage=compile_failed）。"""
    cfg = _make_config(args, bench_root, enable_profiler=False)
    from .eval.evaluator import Evaluator

    evaluator = Evaluator(cfg, bench_name=args.bench_name)
    failures = [evaluator.failure_synthesizer.synthesize_submission_compile_failure(message)]
    _save_report(args, cfg, failures, stage="compile_failed", contains_performance=False)
    evaluator.shutdown()


def _case_num_from_value(value) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        pass
    match = re.search(r"(?:^|_)(\d+)$", str(value))
    return int(match.group(1)) if match else 0


def _case_num(case) -> int:
    case_num = _case_num_from_value(getattr(case, "case_num", None))
    if case_num:
        return case_num
    return _case_num_from_value(getattr(case, "case_id", None))


def _case_result_key(case: EvalCaseResult) -> CaseKey:
    return (case.rel_path, _case_num(case))


def _case_spec_key(case) -> CaseKey:
    return (str(getattr(case, "rel_path", "") or ""), _case_num(case))


def _make_config(args: argparse.Namespace, bench_root: str, *, enable_profiler: bool) -> Config:
    cfg = Config()
    cfg.tasks_root = bench_root
    cfg.bench_name = args.bench_name
    cfg.device_type = args.device
    cfg.device_id = int(args.device_id or 0)
    cfg.warmup = args.warmup
    cfg.repeat = args.repeat
    cfg.enable_profiler = enable_profiler
    cfg.perf_batch_cases = bool(
        getattr(args, 'perf_batch_cases', cfg.perf_batch_cases)
    )
    cfg.perf_batch_case_memory_limit_mb = float(getattr(
        args, 'perf_batch_case_memory_limit_mb',
        cfg.perf_batch_case_memory_limit_mb,
    ))
    cfg.monitor_case_memory = bool(
        not enable_profiler
        and cfg.perf_batch_cases
        and cfg.perf_batch_case_memory_limit_mb > 0
    )
    cfg.profiler_level = args.profiler_level
    cfg.timeout_per_operator = args.timeout_per_operator
    cfg.reports_dir = args.reports_dir
    cfg.processes_per_card = args.processes_per_card
    cfg.max_cases_per_task_unit = getattr(args, "max_cases_per_task_unit", 64)
    cfg.eval_seed = None if args.eval_seed == -1 else args.eval_seed
    cfg.input_dist = getattr(args, "input_dist", "uniform")
    configure_cache(cfg, args)
    if args.source_dir:
        cfg.source_dir = args.source_dir
    cfg.agent_skill = getattr(args, 'agent_skill', '') or ''
    cfg.base_model = getattr(args, 'base_model', '') or ''
    cfg.harness = getattr(args, 'harness', '') or ''
    if args.torch_op_guard_mode:
        cfg.torch_op_guard_mode = args.torch_op_guard_mode
    if args.perf_metric_strategy:
        cfg.perf_metric_strategy_override = args.perf_metric_strategy
    set_config(cfg)
    return cfg


def _operator_rel_paths(
    matched_operators: Iterable[str],
    bench_root: str,
    selected: Optional[Iterable[str]] = None,
) -> List[str]:
    from .benches import cann as _cann_bench  # noqa: F401
    from .registry.loader_registry import get_task_loader

    matched = {str(op).lower() for op in matched_operators}
    selected_set = {str(op).lower() for op in selected} if selected else None
    loader = get_task_loader("cann", tasks_root=bench_root)
    rel_paths = []
    for spec in loader.list_tasks():
        names = {
            str(getattr(spec, "name", "")).lower(),
            str(spec.get_function_name()).lower(),
        }
        if not (names & matched):
            continue
        if selected_set is not None:
            all_names = names | {
                str(spec.rel_path).lower(),
                Path(spec.rel_path).name.lower(),
            }
            if not (all_names & selected_set):
                continue
        rel_paths.append(spec.rel_path)
    return sorted(set(rel_paths))


def _install_or_scan(args: argparse.Namespace, cfg: Config) -> List[str]:
    from .benches import cann as _cann_bench  # noqa: F401
    from .data.package_manager import PackageManager

    pm = PackageManager(config=cfg)
    if not args.source_dir:
        return pm.prepare_skip_build()

    from .security.api_guard import APIGuard

    package_info = pm.scan_source_dir(args.source_dir)
    if not package_info.whl_path:
        raise RuntimeError("compile stage did not produce a cann_bench wheel")

    guard = APIGuard()
    guard.snapshot()
    if not pm.install_packages(package_info):
        raise RuntimeError("package install failed")
    matched = pm.prepare_skip_build()
    guard.verify()
    return matched


def _load_cases(
    args: argparse.Namespace,
    bench_root: str,
    rel_paths: List[str],
    *,
    filter_prefix: str,
    allowlist: Optional[Set[CaseKey]] = None,
) -> List:
    from .benches import cann as _cann_bench  # noqa: F401
    from .registry.loader_registry import get_case_loader

    loader = get_case_loader(args.bench_name, tasks_root=bench_root)
    all_cases = []
    rel_path_set = set(rel_paths)
    for case in loader.scan_all():
        if rel_path_set and case.rel_path not in rel_path_set:
            continue
        if filter_prefix and not (case.rel_path == filter_prefix or case.rel_path.startswith(filter_prefix + "/")):
            continue
        if args.operator and str(case.operator).lower() != args.operator.lower():
            continue
        if args.case_id is not None and _case_num(case) != int(args.case_id):
            continue
        if allowlist is not None and _case_spec_key(case) not in allowlist:
            continue
        all_cases.append(case)
    return all_cases


def _evaluate_cases(
    args: argparse.Namespace,
    cfg: Config,
    cases: List,
    *,
    enable_profiler: bool,
) -> List[EvalOperatorResult]:
    from .eval.process_pool import (
        ProcessConfig,
        ProcessPoolCoordinator,
        aggregate_by_operator,
        build_task_units,
    )

    if not cases:
        return []

    cases_by_operator: Dict[str, List] = defaultdict(list)
    for case in cases:
        cases_by_operator[str(case.operator)].append(case)

    from .eval.subprocess_utils import detect_pypto_pro_submission
    pypto_outer_isolation = detect_pypto_pro_submission()
    cfg.pypto_pro_outer_case_isolation = pypto_outer_isolation

    coordinator = ProcessPoolCoordinator(
        base_config=cfg,
        process_config=ProcessConfig(
            processes_per_card=args.processes_per_card,
            timeout_per_operator=args.timeout_per_operator,
            enable_profiler=enable_profiler,
        ),
        device_id=args.device_id,
    )
    try:
        task_units = build_task_units(
            cases_by_operator,
            coordinator.card_count,
            isolate_each_case=pypto_outer_isolation,
            max_cases_per_task_unit=getattr(args, "max_cases_per_task_unit", 64),
        )
        return aggregate_by_operator(coordinator.evaluate_task_units(task_units))
    finally:
        coordinator.shutdown()


def _save_report(
    args: argparse.Namespace,
    cfg: Config,
    operator_results: List[EvalOperatorResult],
    *,
    stage: str,
    contains_performance: bool,
) -> Tuple[dict, object]:
    from .report.report_generator import ReportGenerator

    generator = ReportGenerator(
        output_dir=args.reports_dir,
        eval_code=f"{args.eval_code}_{stage}" if args.eval_code else None,
        semantic_prefix=f"{args.bench_name}_{stage}",
        config=cfg,
    )
    for op_result in operator_results:
        generator.add_operator_result(op_result)
    report = generator.generate()
    payload = report.to_dict()
    payload["result_stage"] = stage
    payload["contains_performance"] = contains_performance
    if not contains_performance:
        payload["overall_score"] = None
        payload.setdefault("summary", {})["overall_score"] = None
        payload["summary"]["score_unavailable_reason"] = "performance stage has not completed"
    paths = generator.save_all(report)
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload, report


def _passed_case_keys(operator_results: List[EvalOperatorResult]) -> Set[CaseKey]:
    return {
        _case_result_key(case)
        for op_result in operator_results
        for case in op_result.results
        if case.success
    }


def _partition_performance_case_keys(
    operator_results: List[EvalOperatorResult],
    *,
    batch_enabled: bool,
    memory_limit_mb: float,
) -> Tuple[Set[CaseKey], Set[CaseKey]]:
    """按正确性阶段 VmRSS 峰值把通过 case 分为批量/逐 case 两组。"""
    batched: Set[CaseKey] = set()
    individual: Set[CaseKey] = set()
    for op_result in operator_results:
        for case in op_result.results:
            if not case.success:
                continue
            key = _case_result_key(case)
            peak_mb = getattr(case, "memory_peak_mb", None)
            if not batch_enabled:
                individual.add(key)
            elif memory_limit_mb <= 0:
                batched.add(key)
            elif peak_mb is None or peak_mb > memory_limit_mb:
                # 采样缺失时 fail closed，性能仍会测，只是不进入共享会话。
                individual.add(key)
            else:
                batched.add(key)
    return batched, individual


def _combine_operator_results(
    groups: Iterable[List[EvalOperatorResult]],
) -> List[EvalOperatorResult]:
    from .eval.process_pool import aggregate_by_operator

    return aggregate_by_operator([
        case
        for operator_results in groups
        for op_result in operator_results
        for case in op_result.results
    ])


def _merge_results(
    correctness_ops: List[EvalOperatorResult],
    performance_ops: List[EvalOperatorResult],
) -> List[EvalOperatorResult]:
    from .eval.results import EvalOperatorResult
    from .base.result import (
        FAILURE_TYPE_PRECISION_MISMATCH,
        is_precision_failure_type,
    )

    # 全量索引性能阶段所有 case（含失败），用途有二：
    #  1) 成功 → 回填时延；
    #  2) 精度失败 → 识别"correctness 过、performance 精度翻车"的 case。
    #     两次跑输入与 golden 完全一致（已定种子），唯一变量是 NPU kernel 自身输出，
    #     故这类翻车基本等价于"算子非确定"。按策略视为该 case 的精度错误。
    perf_all = {
        _case_result_key(case): case
        for op_result in performance_ops
        for case in op_result.results
    }

    merged: List[EvalOperatorResult] = []
    for op_result in correctness_ops:
        cases = []
        for case in op_result.results:
            if not case.success:
                # correctness 阶段本就失败：保持原判定，无性能分。
                case.perf_result = None
                cases.append(case)
                continue

            perf_case = perf_all.get(_case_result_key(case))
            if perf_case is not None and perf_case.success:
                # 性能阶段精度复检同样通过 → 回填时延。
                case.perf_result = perf_case.perf_result
                case.ai_run_result = perf_case.ai_run_result
            elif perf_case is not None and is_precision_failure_type(perf_case.failure_type):
                # correctness 过、performance 精度翻车 → 视为该 case 精度错误：
                # success=False + precision_mismatch，按原公式扣精度分（不扣编译分），
                # 且无对应性能分；同时打标签便于在 results.json 中定位疑似非确定算子。
                case.success = False
                case.failure_type = FAILURE_TYPE_PRECISION_MISMATCH
                if perf_case.accuracy_result is not None:
                    case.accuracy_result = perf_case.accuracy_result
                case.error_msg = (
                    "性能阶段精度复检失败（correctness 阶段已通过）——疑似 NPU 非确定性算子: "
                    + (perf_case.error_msg or "")
                )
                case.perf_result = None
                case.perf_recheck = {
                    "status": "precision_unstable",
                    "correctness_passed": True,
                    "perf_failure_type": perf_case.failure_type,
                    "note": (
                        "passed precision in the correctness stage but failed the "
                        "precision re-check in the performance stage; likely a "
                        "non-deterministic NPU kernel."
                    ),
                }
            else:
                # 性能阶段无法测量（timeout / runtime / 未重跑）：非精度问题，
                # 沿用 correctness 通过判定，仅标注缺失原因，无性能分。
                case.perf_result = None
                if perf_case is not None and not perf_case.success:
                    failure_reason = perf_case.error_msg or (
                        "performance stage did not produce a valid timing"
                    )
                    case.perf_recheck = {
                        "status": "perf_unmeasured",
                        "correctness_passed": True,
                        "perf_failure_type": perf_case.failure_type,
                        "error_msg": failure_reason,
                        "note": (
                            "passed correctness but the performance stage could not "
                            "produce a valid timing (e.g. timeout / runtime error). "
                            f"Reason: {failure_reason}"
                        ),
                    }
            cases.append(case)

        speedups = [case.get_speedup() for case in cases if case.success and case.get_speedup() > 0]
        total_cases = max(op_result.total_cases, len(cases))
        passed_cases = sum(1 for case in cases if case.success)
        skipped_cases = sum(
            1 for case in cases
            if not case.success and case.failure_type in ("cascade_device", "skipped")
        )
        failed_cases = sum(
            1 for case in cases
            if not case.success and case.failure_type not in ("cascade_device", "skipped")
        )
        merged.append(EvalOperatorResult(
            rel_path=op_result.rel_path,
            operator=op_result.operator,
            total_cases=total_cases,
            passed_cases=passed_cases,
            failed_cases=failed_cases,
            skipped_cases=max(op_result.skipped_cases, skipped_cases),
            results=cases,
            pass_rate=passed_cases / total_cases if total_cases else 0.0,
            avg_speedup=sum(speedups) / len(speedups) if speedups else 0.0,
            compilation_error=op_result.compilation_error,
            subprocess_failure_reason=op_result.subprocess_failure_reason,
        ))
    return merged


def _write_compile_failure_report(
    args: argparse.Namespace,
    bench_root: str,
    package_info,
) -> None:
    cfg = _make_config(args, bench_root, enable_profiler=False)
    from .eval.evaluator import Evaluator

    evaluator = Evaluator(cfg, bench_name=args.bench_name)
    op_filter = _compile_failure_operator_filter(args)
    failures = evaluator.failure_synthesizer.synthesize_all_compile_failures(
        evaluator.operator_matcher,
        package_info,
        operator_filter=op_filter,
    )
    _save_report(args, cfg, failures, stage="compile_failed", contains_performance=False)
    evaluator.shutdown()


def _compile_failure_operator_filter(args: argparse.Namespace) -> Optional[List[str]]:
    """Return the requested operators that a submission-level build error belongs to."""
    if args.operator:
        return [args.operator]
    selected = getattr(args, "selected_operators", None)
    return list(selected) if selected else None


def _compile(args: argparse.Namespace, bench_root: str) -> int:
    from .data.package_manager import PackageManager

    if not args.source_dir:
        print("[staged_eval] compile: no --source-dir, using installed cann_bench")
        return 0

    cfg = _make_config(args, bench_root, enable_profiler=False)
    pm = PackageManager(config=cfg)
    package_info = pm.build_packages(args.source_dir, iterative=not args.no_iterative_compile)
    if getattr(package_info, "build_failed", False) or not package_info.whl_path:
        _write_compile_failure_report(args, bench_root, package_info)
        return 1
    print(f"[staged_eval] compile: built {Path(package_info.whl_path).name}")
    return 0


def run(args: argparse.Namespace) -> int:
    from .utils.path_resolver import resolve_task_dir

    if args.bench_name != "cann":
        raise SystemExit("staged_eval currently supports --bench-name cann only")
    if args.device != "npu":
        raise SystemExit("staged_eval currently supports NPU evaluation only")

    project_root = get_project_root()
    bench_root, filter_prefix = resolve_task_dir(args.task_dir, project_root)
    from .base.perf_metric import resolve_perf_metric
    from .base.score_weights import score_weights_context

    with score_weights_context(bench_root):
        resolve_perf_metric()
        return _run_with_score_profile(args, bench_root, filter_prefix)


def _run_with_score_profile(args, bench_root, filter_prefix):
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)

    print("[staged_eval] stage 1/3: compile")
    compile_rc = _compile(args, bench_root)
    if compile_rc != 0:
        return compile_rc

    # CANN 构建/运行版本一致性护栏：错配的 kernel 会在 profiler 下使评测子进程
    # SIGSEGV（详见函数注释），此处按编译失败显式终止，而不是让性能项静默按 0 计入。
    mismatch = _check_build_cann_consistency(args)
    if mismatch is not None:
        print(f"[staged_eval] compile rejected: {mismatch}")
        _write_cann_mismatch_report(args, bench_root, mismatch)
        return 1

    print("[staged_eval] stage 2/3: correctness")
    correctness_cfg = _make_config(args, bench_root, enable_profiler=False)
    matched = _install_or_scan(args, correctness_cfg)
    rel_paths = _operator_rel_paths(matched, bench_root, selected=args.selected_operators)
    correctness_cases = _load_cases(args, bench_root, rel_paths, filter_prefix=filter_prefix)
    correctness_ops = _evaluate_cases(args, correctness_cfg, correctness_cases, enable_profiler=False)
    correctness_payload, _ = _save_report(
        args, correctness_cfg, correctness_ops, stage="correctness", contains_performance=False,
    )

    if args.no_perf:
        failed = int(correctness_payload.get("failed_cases") or 0)
        return min(failed, 255)

    allowlist = _passed_case_keys(correctness_ops)
    memory_limit_mb = float(getattr(
        args, "perf_batch_case_memory_limit_mb", 32 * 1024))
    batch_allowlist, individual_allowlist = _partition_performance_case_keys(
        correctness_ops,
        batch_enabled=bool(getattr(args, "perf_batch_cases", True)),
        memory_limit_mb=memory_limit_mb,
    )
    print(
        f"[staged_eval] stage 3/3: performance ({len(allowlist)} correctness-passed cases; "
        f"{len(batch_allowlist)} batch eligible, "
        f"{len(individual_allowlist)} per-case profiler)"
    )
    performance_cfg = _make_config(args, bench_root, enable_profiler=True)
    matched = _install_or_scan(args, performance_cfg)
    rel_paths = _operator_rel_paths(matched, bench_root, selected=args.selected_operators)
    performance_cases = _load_cases(
        args,
        bench_root,
        rel_paths,
        filter_prefix=filter_prefix,
        allowlist=allowlist,
    )
    batch_cases = [
        case for case in performance_cases
        if _case_spec_key(case) in batch_allowlist
    ]
    individual_cases = [
        case for case in performance_cases
        if _case_spec_key(case) in individual_allowlist
    ]

    batch_ops = _evaluate_cases(
        args, performance_cfg, batch_cases, enable_profiler=True)
    individual_cfg = _make_config(args, bench_root, enable_profiler=True)
    individual_cfg.perf_batch_cases = False
    individual_cfg.monitor_case_memory = False
    individual_ops = _evaluate_cases(
        args, individual_cfg, individual_cases, enable_profiler=True)
    performance_ops = _combine_operator_results([batch_ops, individual_ops])
    _save_report(args, performance_cfg, performance_ops, stage="performance", contains_performance=True)

    merged_ops = _merge_results(correctness_ops, performance_ops)
    final_payload, _ = _save_report(
        args, performance_cfg, merged_ops, stage="final", contains_performance=True,
    )
    failed = int(final_payload.get("failed_cases") or 0)
    return min(failed, 255)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cann-bench in compile/correctness/performance stages")
    parser.add_argument("--bench-name", default="cann")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--task-dir", default="tasks")
    parser.add_argument("--operator", default=None)
    parser.add_argument("--case-id", type=int, default=None)
    parser.add_argument("--selected-operators", nargs="*", default=None,
                        help="仅评测指定算子（匹配 name / function_name / rel_path / 目录名，大小写不敏感）")
    parser.add_argument("--device", choices=["npu"], default="npu")
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--processes-per-card", type=int, default=2)
    parser.add_argument("--max-cases-per-task-unit", type=int, default=64,
                        help="单个 eval-child 最多处理的 case 数（默认: 64）")
    parser.add_argument("--timeout-per-operator", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--eval-code", default=None)
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--perf-batch-cases", dest="perf_batch_cases",
                        action="store_true",
                        help="同一 TaskUnit 的 case 共用一次 profiler（默认开启）")
    parser.add_argument("--no-perf-batch-cases", dest="perf_batch_cases",
                        action="store_false",
                        help="关闭批量 case 性能采集，恢复逐 case profiler")
    parser.set_defaults(perf_batch_cases=True)
    parser.add_argument("--perf-batch-case-memory-limit-mb", type=float,
                        default=32 * 1024,
                        help="正确性阶段单 case VmRSS 峰值上限（MB，默认: 32768）；"
                             "超过或采样失败的 case 改为逐 case profiler，<=0 禁用分流")
    parser.add_argument("--profiler-level", choices=["Level1", "Level2"], default="Level1")
    parser.add_argument("--perf-metric-strategy", default=None)
    parser.add_argument("--torch-op-guard-mode", choices=["off", "warn", "block"], default=None)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--input-dist", type=str, default="uniform",
                        choices=INPUT_DIST_CHOICES,
                        metavar="DIST",
                        help="输入数据分布（默认: uniform）。normal 走正态分布，"
                             "参数由 value_range 推出。仅作用于浮点输入。")
    parser.add_argument("--no-iterative-compile", action="store_true")
    parser.add_argument("--agent-skill", default="", help="评测对象 Agent/Skill 标签，写入报告元信息")
    parser.add_argument("--base-model", default="", help="评测对象 BaseModel 标签，写入报告元信息")
    parser.add_argument("--harness", default="", help="评测对象 Harness 标签，写入报告元信息")
    add_cache_arguments(parser)
    return parser


def main() -> int:
    return run(create_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
