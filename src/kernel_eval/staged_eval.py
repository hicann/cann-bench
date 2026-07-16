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
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .config import Config, get_project_root, set_config


CaseKey = Tuple[str, int]


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
    cfg.profiler_level = args.profiler_level
    cfg.timeout_per_operator = args.timeout_per_operator
    cfg.reports_dir = args.reports_dir
    cfg.processes_per_card = args.processes_per_card
    cfg.eval_seed = None if args.eval_seed == -1 else args.eval_seed
    if args.source_dir:
        cfg.source_dir = args.source_dir
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
        task_units = build_task_units(cases_by_operator, coordinator.card_count)
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
                    case.perf_recheck = {
                        "status": "perf_unmeasured",
                        "correctness_passed": True,
                        "perf_failure_type": perf_case.failure_type,
                        "note": (
                            "passed correctness but the performance stage could not "
                            "produce a valid timing (e.g. timeout / runtime error)."
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
    op_filter = [args.operator] if args.operator else None
    failures = evaluator.failure_synthesizer.synthesize_all_compile_failures(
        evaluator.operator_matcher,
        package_info,
        operator_filter=op_filter,
    )
    _save_report(args, cfg, failures, stage="compile_failed", contains_performance=False)
    evaluator.shutdown()


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
    Path(args.reports_dir).mkdir(parents=True, exist_ok=True)

    print("[staged_eval] stage 1/3: compile")
    compile_rc = _compile(args, bench_root)
    if compile_rc != 0:
        return compile_rc

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
    print(f"[staged_eval] stage 3/3: performance ({len(allowlist)} correctness-passed cases)")
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
    performance_ops = _evaluate_cases(args, performance_cfg, performance_cases, enable_profiler=True)
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
    parser.add_argument("--timeout-per-operator", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--eval-code", default=None)
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--profiler-level", choices=["Level1", "Level2"], default="Level1")
    parser.add_argument("--perf-metric-strategy", default=None)
    parser.add_argument("--torch-op-guard-mode", choices=["off", "warn", "block"], default=None)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--no-iterative-compile", action="store_true")
    return parser


def main() -> int:
    return run(create_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
