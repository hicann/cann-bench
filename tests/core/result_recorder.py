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
测试结果记录器

职责：
1. 记录执行状态和性能数据
2. 输出JSON格式报告
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """测试结果"""
    level: int
    operator: str
    case_id: int
    status: str  # success / failed / skipped
    elapsed_us: float = 0
    op_times: Optional[Dict[str, Dict[str, float]]] = None
    error_msg: Optional[str] = None
    device: str = ""
    timestamp: str = ""
    _profiler_result: Any = None

    def resolve_profiling(self):
        if self._profiler_result is not None:
            pr = self._profiler_result
            self.elapsed_us = pr.elapsed_us
            self.op_times = pr.op_times
            if pr.error:
                self.error_msg = pr.error
            self._profiler_result = None
        if self.op_times is None:
            self.op_times = {}


class ResultRecorder:
    """结果记录器"""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.results: List[TestResult] = []
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, case: Any, run_result: Any, profiler_result: Any = None):
        """记录测试结果"""
        self.results.append(TestResult(
            level=case.level,
            operator=case.operator,
            case_id=case.case_id,
            status="success" if run_result.success else "failed",
            elapsed_us=run_result.elapsed_us,
            _profiler_result=profiler_result,
            error_msg=run_result.error,
            device=run_result.device,
            timestamp=datetime.now().isoformat()
        ))

    def record_skip(self, case: Any, reason: str):
        """记录跳过的用例"""
        self.results.append(TestResult(
            level=case.level,
            operator=case.operator,
            case_id=case.case_id,
            status="skipped",
            error_msg=reason,
            timestamp=datetime.now().isoformat()
        ))

    def save(self):
        """保存结果到JSON文件"""
        for r in self.results:
            r.resolve_profiling()
        summary = self._build_summary()
        output = {"summary": summary, "results": [asdict(r) for r in self.results]}
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 结果已保存到: {self.output_path}")

    def _build_summary(self) -> Dict:
        """构建摘要"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == "success")
        failed = sum(1 for r in self.results if r.status == "failed")
        skipped = sum(1 for r in self.results if r.status == "skipped")

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "pass_rate": f"{passed/total*100:.2f}%" if total > 0 else "N/A",
            "timestamp": datetime.now().isoformat()
        }

    def print_summary(self):
        """打印摘要"""
        summary = self._build_summary()
        print("\n" + "=" * 50)
        print("测试结果摘要")
        print("=" * 50)
        print(f"总计: {summary['total']}")
        print(f"成功: {summary['passed']}")
        print(f"失败: {summary['failed']}")
        print(f"跳过: {summary['skipped']}")
        print(f"成功率: {summary['pass_rate']}")
        print("=" * 50 + "\n")