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
    ) -> EvalOperatorResult:
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

        first_line = (error_text.strip().splitlines() or ["(no detail)"])[0]
        reason_short = f"{error_prefix} {first_line[:180]}"

        case_results: List[EvalCaseResult] = []
        for c in cases:
            case_results.append(EvalCaseResult(
                case_id=str(getattr(c, "case_id", "")),
                rel_path=rel_path,
                operator=operator_name,
                case_num=getattr(c, "case_num", 0),
                success=False,
                error_msg=reason_short,
                failure_type="cascade_device",  # 子进程崩溃/编译失败等合成的结果标记为级联失败
            ))

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
    ) -> EvalOperatorResult:
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

    def synthesize_security_failure(
        self,
        op_info: TaskSpec,
        security_error: str,
        case_filter: Optional[Dict] = None,
        filter_func: Optional[callable] = None,
    ) -> EvalOperatorResult:
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
    ) -> EvalOperatorResult:
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