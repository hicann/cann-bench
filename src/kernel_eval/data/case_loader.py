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
用例加载器

职责：
1. 扫描YAML测试用例文件
2. 解析YAML格式，提取测试用例
"""

import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _coerce_baseline_us(value: Any) -> float:
    """Normalize ``baseline_perf_us`` into a float.

    Some yaml files encode a missing baseline as the literal ``None`` —
    unquoted, it parses as the string ``"None"``, which is truthy and
    later breaks numeric comparisons (``'>' not supported between str and
    int``). Accept None, numeric, or numeric-like strings; anything else
    collapses to 0.0.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in ("none", "null", "nan"):
            return 0.0
        try:
            return float(s)
        except ValueError:
            return 0.0
    return 0.0


@dataclass
class CaseInfo:
    """测试用例信息"""
    level: int
    operator: str
    case_id: int
    input_shapes: List
    dtypes: List[str]
    attrs: Dict[str, Any]
    value_ranges: List
    note: str
    yaml_path: str
    baseline_perf_us: float = 0.0

    def get_case_id_str(self) -> str:
        return f"L{self.level}_{self.operator}_{self.case_id}"


class CaseLoader:
    """YAML用例加载器"""

    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
        if not self.bench_root.exists():
            raise ValueError(f"kernel_bench目录不存在: {bench_root}")

    def scan_all_cases(self) -> List[CaseInfo]:
        """扫描所有level的用例"""
        all_cases = []
        for level in [1, 2, 3, 4]:
            all_cases.extend(self.scan_by_level(level))
        return all_cases

    def scan_by_level(self, level: int) -> List[CaseInfo]:
        """扫描指定level的用例"""
        level_dir = self.bench_root / f"level{level}"
        if not level_dir.exists():
            return []

        cases = []
        for op_dir in level_dir.iterdir():
            if op_dir.is_dir() and not op_dir.name.startswith('.'):
                cases_yaml = op_dir / "cases.yaml"
                if cases_yaml.exists():
                    cases.extend(self._load_yaml(cases_yaml, level))
        return cases

    def scan_by_operator(self, level: int, operator: str) -> List[CaseInfo]:
        """扫描指定算子的用例"""
        level_dir = self.bench_root / f"level{level}"
        if not level_dir.exists():
            return []

        # 尝试匹配目录名（支持PascalCase和snake_case）
        op_dir = self._find_operator_dir(level_dir, operator)
        if not op_dir:
            return []

        cases_yaml = op_dir / "cases.yaml"
        if cases_yaml.exists():
            return self._load_yaml(cases_yaml, level)
        return []

    def _find_operator_dir(self, level_dir: Path, operator: str) -> Path:
        """查找算子目录"""
        # 尝试直接匹配
        direct_path = level_dir / operator
        if direct_path.exists():
            return direct_path

        # 尝试snake_case转换
        import re
        s0 = re.sub(r'([0-9])([A-Z])', r'\1_\2', operator)
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s0)
        snake_name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
        snake_path = level_dir / snake_name
        if snake_path.exists():
            return snake_path

        # 尝试小写匹配
        lower_path = level_dir / operator.lower()
        if lower_path.exists():
            return lower_path

        return None

    def _load_yaml(self, yaml_path: Path, level: int) -> List[CaseInfo]:
        """解析YAML文件"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'cases' not in data:
            return []

        cases = []
        for raw in data['cases']:
            case = self._parse_case(raw, level, str(yaml_path))
            if case:
                cases.append(case)
        return cases

    def _parse_case(self, raw: Dict, level: int, yaml_path: str) -> CaseInfo:
        """解析单个用例"""
        input_shapes = raw.get('input_shape', [])
        if isinstance(input_shapes, list) and input_shapes and not isinstance(input_shapes[0], list):
            input_shapes = [input_shapes]

        dtypes = raw.get('dtype', [])
        if isinstance(dtypes, str):
            dtypes = [dtypes]

        return CaseInfo(
            level=level,
            operator=raw.get('operator', ''),
            case_id=raw.get('case_id', 0),
            input_shapes=input_shapes,
            dtypes=dtypes,
            attrs=raw.get('attrs', {}) or {},
            value_ranges=raw.get('value_range', []) or [],
            note=raw.get('note', '') or '',
            yaml_path=yaml_path,
            baseline_perf_us=_coerce_baseline_us(raw.get('baseline_perf_us'))
        )

    def get_statistics(self) -> Dict[int, Dict[str, int]]:
        """获取用例统计"""
        stats = {}
        for level in [1, 2, 3, 4]:
            cases = self.scan_by_level(level)
            operator_counts = {}
            for case in cases:
                operator_counts[case.operator] = operator_counts.get(case.operator, 0) + 1
            stats[level] = {'total': len(cases), 'operators': operator_counts}
        return stats