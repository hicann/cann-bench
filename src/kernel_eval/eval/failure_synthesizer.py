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
失败结果合成器

职责：
1. 为编译失败的算子合成 FAIL 结果
2. 为安全检查失败的算子合成 FAIL 结果
3. 为子进程异常的算子合成 FAIL 结果

这样失败算子仍然出现在 session 结果里，报告可见原因。
"""

from typing import Dict, List, Optional, Any

from .results import EvalCaseResult, EvalOperatorResult
from ..data import CaseLoader
from ..base.models import TaskSpec


class FailureSynthesizer:
    """失败结果合成器"""

    def __init__(self, case_loader: CaseLoader):
        self.case_loader = case_loader

    # ------------------------------------------------------------------
    # 统一内部实现
    # ------------------------------------------------------------------

    def _synthesize_failure(
        self,
        error_text: str,
        error_prefix: str,
        error_field: str,
        operator_name: str,
        rel_path: str,
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> Optional[EvalOperatorResult]:
        """为无法评测的算子生成一条 all-FAIL 的 EvalOperatorResult。

        三个公有方法（compile / security / subprocess）的共享实现，
        仅 error_prefix 与 diagnostic error_field 不同。

        Args:
            error_text: 错误原文（全文存入诊断字段）
            error_prefix: 单用例 error_msg 前缀，如 ``"compile failed:"``
            error_field: 算子级诊断字段名，``"compilation_error"`` 或
                ``"subprocess_failure_reason"``
            operator_name: 算子名称
            rel_path: 相对路径
        """
        try:
            cases = self.case_loader.scan_by_operator(operator_name)
            if case_filter and filter_func:
                cases = filter_func(cases, case_filter)
        except Exception:
            cases = []

        excluded_getter = getattr(self.case_loader, "get_hardware_excluded_cases", None)
        if not cases and excluded_getter is not None:
            excluded = excluded_getter(operator_name)
            if case_filter and filter_func:
                excluded = filter_func(excluded, case_filter)
            if excluded:
                return None

        first_line = (error_text.strip().splitlines() or ["(no detail)"])[0]
        reason_short = f"{error_prefix} {first_line[:180]}"

        case_results: List[EvalCaseResult] = []
        for c in cases:
            case_results.append(
                EvalCaseResult(
                    case_id=str(getattr(c, "case_id", "")),
                    rel_path=rel_path,
                    operator=operator_name,
                    case_num=getattr(c, "case_num", 0),
                    success=False,
                    error_msg=reason_short,
                    baseline_perf_us=getattr(c, "baseline_perf_us", 0.0) or 0.0,
                    t_hw_us=getattr(c, "t_hw_us", 0.0) or 0.0,
                    baseline_source=(getattr(c, "metadata", None) or {}).get(
                        "baseline_source", {}
                    ),
                    failure_type="cascade_device",  # 子进程崩溃/编译失败等合成的结果标记为级联失败
                )
            )

        kwargs: Dict[str, Any] = {
            "rel_path": rel_path,
            "operator": operator_name,
            "total_cases": len(case_results),
            "passed_cases": 0,
            "failed_cases": len(case_results),
            "skipped_cases": 0,
            "results": case_results,
            "pass_rate": 0.0,
            "avg_speedup": 0.0,
        }
        kwargs[error_field] = error_text

        return EvalOperatorResult(**kwargs)

    # ------------------------------------------------------------------
    # 公有方法（薄封装）
    # ------------------------------------------------------------------

    def synthesize_compile_failure(
        self,
        op_info: TaskSpec,
        error_excerpt: str,
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> Optional[EvalOperatorResult]:
        """为编译失败的算子生成一条 all-FAIL 的 EvalOperatorResult"""
        return self._synthesize_failure(
            error_text=error_excerpt,
            error_prefix="compile failed:",
            error_field="compilation_error",
            operator_name=op_info.name,
            rel_path=op_info.rel_path,
            case_filter=case_filter,
            filter_func=filter_func,
        )

    def synthesize_all_compile_failures(
        self,
        operator_matcher,
        package_info,
        operator_filter: Optional[List[str]] = None,
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> List[EvalOperatorResult]:
        """编译整体失败时，把 package_info.compile_errors 合成为 0 分结果列表。

        cli 与 evaluator 两条编译失败处理路径的**单一真源**，避免两处逻辑漂移。

        - 能反查到 spec 的算子逐个合成（受 operator_filter 约束）；
        - 编译日志只有 ``<build>`` 等提交级错误时，若调用方明确指定了目标算子，
          将错误归到这些真实算子，而不是新增 ``<submission>`` 伪算子；
        - 反查不到的（`<build>` 兜底键 / 未注册 snake 名）打 WARN 跳过；
        - 仅当编译日志和显式目标都**无法反查到 spec** 时，合成一条提交级记录，
          避免空报告静默失败。注意：只在"无法映射"时兜底，**不**在"算子被
          operator_filter 过滤掉"时兜底——否则会把用户主动过滤误判为无法映射
          而虚报失败。
        """
        import sys

        requested_specs = []
        requested_names = set()
        for requested in operator_filter or []:
            op_info = operator_matcher.find_operator_info(requested)
            if op_info is None:
                op_info = operator_matcher.find_operator_info_by_snake(requested)
            if op_info is None:
                print(
                    f"[WARN] 指定算子 {requested!r} 未在 OperatorMatcher 中找到对应 spec。",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if op_info.name in requested_names:
                continue
            requested_specs.append(op_info)
            requested_names.add(op_info.name)

        results: List[EvalOperatorResult] = []
        mapped_any = False
        for snake_op_name, err in (package_info.compile_errors or {}).items():
            op_info = operator_matcher.find_operator_info_by_snake(snake_op_name)
            if op_info is None:
                print(
                    f"[WARN] 编译失败算子 {snake_op_name!r} 未在 OperatorMatcher 中找到对应 spec，"
                    f"跳过逐算子合成（可能是 <build> 兜底键或未注册的 snake 名）。",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            mapped_any = True
            if operator_filter and op_info.name not in requested_names:
                continue
            result = self.synthesize_compile_failure(
                op_info,
                err,
                case_filter,
                filter_func,
            )
            if result is not None:
                results.append(result)

        if not mapped_any and not results:
            joined = (
                "\n".join(
                    f"[{k}] {v}" for k, v in (package_info.compile_errors or {}).items()
                )
                or "build.sh 编译失败（无更多详情）"
            )
            if requested_specs:
                print(
                    "[WARN] 编译失败日志未能映射到具体算子，"
                    "按调用方指定的目标算子合成编译失败结果。",
                    file=sys.stderr,
                    flush=True,
                )
                requested_results = [
                    self.synthesize_compile_failure(
                        op_info,
                        joined,
                        case_filter,
                        filter_func,
                    )
                    for op_info in requested_specs
                ]
                return [result for result in requested_results if result is not None]
            print(
                "[WARN] 编译失败但未能映射到任何已注册算子，"
                "合成提交级编译失败记录以避免空报告。",
                file=sys.stderr,
                flush=True,
            )
            results.append(self.synthesize_submission_compile_failure(joined))
        return results

    def synthesize_submission_compile_failure(
        self,
        error_text: str,
        operator_name: str = "<submission>",
    ) -> EvalOperatorResult:
        """整体编译失败、但 compile_errors 无法映射到任何已注册算子时的兜底。

        触发场景：build.sh 在编译算子内核前就失败（cmake/依赖/链接错误等），
        或提交目录布局异常 → compile_errors 退化为 {"<build>": ...} 这类无法
        反查 spec 的键。此时若按算子逐条合成会得到空结果，上层报告将不体现任何
        编译失败（"静默失败"）。本方法合成一条**提交级** all-FAIL 记录，保证报告
        至少出现一条编译失败、退出码非零。

        rel_path 故意留空：report / html_generator 对空 rel_path 均有兜底，
        不会把这条记录错误归入某个 level，也不会触发 rel_path 解析崩溃。
        """
        first_line = (error_text.strip().splitlines() or ["(no detail)"])[0]
        reason_short = f"compile failed: {first_line[:180]}"
        case_results = [
            EvalCaseResult(
                case_id="",
                rel_path="",
                operator=operator_name,
                case_num=0,
                success=False,
                error_msg=reason_short,
                failure_type="cascade_device",
            )
        ]
        return EvalOperatorResult(
            rel_path="",
            operator=operator_name,
            total_cases=1,
            passed_cases=0,
            failed_cases=1,
            skipped_cases=0,
            results=case_results,
            pass_rate=0.0,
            avg_speedup=0.0,
            compilation_error=error_text,
        )

    def synthesize_security_failure(
        self,
        op_info: TaskSpec,
        security_error: str,
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> Optional[EvalOperatorResult]:
        """为安全检查失败的算子生成一条 all-FAIL 的 EvalOperatorResult"""
        return self._synthesize_failure(
            error_text=security_error,
            error_prefix="security check failed:",
            error_field="subprocess_failure_reason",
            operator_name=op_info.name,
            rel_path=op_info.rel_path,
            case_filter=case_filter,
            filter_func=filter_func,
        )

    def synthesize_subprocess_failure(
        self,
        operator_name: str,
        rel_path: str = "",
        reason: str = "",
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> Optional[EvalOperatorResult]:
        """子进程超时 / 崩溃时合成 all-FAIL 的 EvalOperatorResult"""
        return self._synthesize_failure(
            error_text=reason,
            error_prefix="subprocess failed:",
            error_field="subprocess_failure_reason",
            operator_name=operator_name,
            rel_path=rel_path,
            case_filter=case_filter,
            filter_func=filter_func,
        )

    def synthesize_oom_failure(
        self,
        operator_name: str,
        rel_path: str = "",
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> EvalOperatorResult:
        """子进程被 OOM Killer 杀死时合成 all-FAIL 的 EvalOperatorResult。

        与 synthesize_subprocess_failure 不同：
        - failure_type 使用 "oom_killed" 标记，便于统计和报告区分
        - error_msg 明确标注是 OOM Killer 导致的进程死亡
        """
        try:
            cases = self.case_loader.scan_by_operator(operator_name)
            if case_filter and filter_func:
                cases = filter_func(cases, case_filter)
        except Exception:
            cases = []

        error_msg = "子进程被 OOM Killer 杀死 (SIGKILL/-9)，内存不足"

        case_results: List[EvalCaseResult] = []
        for c in cases:
            case_results.append(
                EvalCaseResult(
                    case_id=str(getattr(c, "case_id", "")),
                    rel_path=rel_path,
                    operator=operator_name,
                    case_num=getattr(c, "case_num", 0),
                    success=False,
                    error_msg=error_msg,
                    baseline_perf_us=getattr(c, "baseline_perf_us", 0.0) or 0.0,
                    baseline_source=(getattr(c, "metadata", None) or {}).get(
                        "baseline_source", {}
                    ),
                    t_hw_us=getattr(c, "t_hw_us", 0.0) or 0.0,
                    failure_type="oom_killed",
                )
            )

        return EvalOperatorResult(
            rel_path=rel_path,
            operator=operator_name,
            total_cases=len(case_results),
            passed_cases=0,
            failed_cases=len(case_results),
            skipped_cases=0,
            results=case_results,
            pass_rate=0.0,
            avg_speedup=0.0,
            subprocess_failure_reason=error_msg,
        )
