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
3. 支持任意目录结构，递归查找proto.yaml识别算子目录
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


def _coerce_baseline_us(value: Any) -> float:
    """Normalize ``baseline_perf_us`` / ``t_hw_us`` into a float.

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
    rel_path: str       # 相对路径，如 "level2/scatter"
    operator: str       # 算子名称
    case_id: int        # 用例编号
    input_shapes: List  # 输入形状
    dtypes: List[str]   # 数据类型
    attrs: Dict[str, Any]  # 算子属性
    value_ranges: List  # 值范围
    note: str           # 备注
    yaml_path: str      # YAML文件路径
    baseline_perf_us: float = 0.0  # 性能基线
    t_hw_us: float = 0.0  # 硬件下界 T_HW

    def get_case_id_str(self) -> str:
        return f"{self.rel_path}_{self.case_id}"


class CaseLoader:
    """YAML用例加载器"""

    # 算子目录必须包含的文件
    REQUIRED_FILES = ['proto.yaml', 'cases.yaml', 'golden.py']

    def __init__(self, bench_root: str):
        self.bench_root = Path(bench_root)
        if not self.bench_root.exists():
            raise ValueError(f"bench目录不存在: {bench_root}")

    def _is_operator_dir(self, dir_path: Path) -> bool:
        """检查目录是否为有效的算子目录"""
        for required_file in self.REQUIRED_FILES:
            if not (dir_path / required_file).exists():
                return False
        return True

    def _is_bench_root(self) -> bool:
        """检查当前目录是否为bench根目录（包含多个算子目录）"""
        # 如果目录下有proto.yaml且是算子目录，则为算子目录而非bench根目录
        if (self.bench_root / 'proto.yaml').exists():
            return not self._is_operator_dir(self.bench_root)
        # 如果没有任何proto.yaml，不是有效目录
        return any(self._find_operator_dirs())

    def _find_operator_dirs(self) -> List[Path]:
        """递归查找所有算子目录"""
        operator_dirs = []
        # 递归查找所有proto.yaml文件
        for proto_path in self.bench_root.rglob("proto.yaml"):
            op_dir = proto_path.parent
            if self._is_operator_dir(op_dir):
                operator_dirs.append(op_dir)
        return sorted(operator_dirs)

    def scan_all_cases(self) -> List[CaseInfo]:
        """扫描所有用例"""
        # 判断是算子目录还是bench根目录
        if self._is_operator_dir(self.bench_root):
            # 直接是算子目录
            return self._load_operator_cases(self.bench_root)
        else:
            # bench根目录，扫描所有算子
            all_cases = []
            for op_dir in self._find_operator_dirs():
                all_cases.extend(self._load_operator_cases(op_dir))
            return all_cases

    def _load_operator_cases(self, op_dir: Path) -> List[CaseInfo]:
        """加载单个算子目录的用例"""
        cases_yaml = op_dir / "cases.yaml"
        if not cases_yaml.exists():
            return []

        # 计算相对路径
        try:
            rel_path = op_dir.relative_to(self.bench_root)
        except ValueError:
            # 如果op_dir不在bench_root下，使用目录名
            rel_path = op_dir.name

        return self._load_yaml(cases_yaml, str(rel_path))

    def scan_by_operator(self, operator: str) -> List[CaseInfo]:
        """扫描指定算子的用例"""
        all_cases = self.scan_all_cases()
        return [c for c in all_cases if c.operator.lower() == operator.lower()]

    def scan_by_rel_path(self, rel_path: str) -> List[CaseInfo]:
        """扫描指定相对路径的用例"""
        all_cases = self.scan_all_cases()
        return [c for c in all_cases if c.rel_path == rel_path]

    def _load_yaml(self, yaml_path: Path, rel_path: str) -> List[CaseInfo]:
        """解析YAML文件"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'cases' not in data:
            return []

        cases = []
        for raw in data['cases']:
            # 校验格式并输出警告
            warnings = self._validate_case(raw, str(yaml_path))
            for warning in warnings:
                print(f"[WARN] {warning}")
            case = self._parse_case(raw, rel_path, str(yaml_path))
            if case:
                cases.append(case)
        return cases

    def _validate_case(self, raw: Dict, yaml_path: str) -> List[str]:
        """校验 YAML 格式，返回警告列表

        校验规则：
        - input_shape: 必须是嵌套列表 [[shape1], [shape2], ...]
        - dtype: 与 input_shape 同维度
        - value_range: 与 input_shape 同维度（可选）
        - case_id: 必填，整数
        """
        warnings = []

        # case_id 校验
        case_id = raw.get('case_id')
        if case_id is None:
            warnings.append(f"{yaml_path}: missing 'case_id'")
        elif not isinstance(case_id, int):
            warnings.append(f"{yaml_path}: case_id should be int, got {type(case_id).__name__}")

        # input_shape 校验
        input_shapes = raw.get('input_shape', [])
        if not isinstance(input_shapes, list) or not input_shapes:
            warnings.append(f"{yaml_path}: 'input_shape' must be non-empty list")
        elif isinstance(input_shapes[0], int):
            # 扁平格式 [N, C, H, W]，需要包装为嵌套
            warnings.append(f"{yaml_path}: input_shape is not nested list [[...]], auto-fixing")
        elif not all(isinstance(s, list) or s is None for s in input_shapes):
            warnings.append(f"{yaml_path}: input_shape elements should all be lists or None (optional placeholder)")

        # dtype 校验
        dtypes = raw.get('dtype', [])
        if isinstance(dtypes, str):
            dtypes = [dtypes]
        if input_shapes and isinstance(input_shapes[0], list):  # 嵌套格式
            # 允许 dtype 单值简写（表示所有 tensor 使用相同类型）
            if len(dtypes) == 1:
                pass  # 单值简写合法，无需警告
            elif len(dtypes) != len(input_shapes):
                warnings.append(f"{yaml_path}: dtype len={len(dtypes)} != input_shape len={len(input_shapes)}")

        return warnings

    def _parse_case(self, raw: Dict, rel_path: str, yaml_path: str) -> CaseInfo:
        """解析单个用例"""
        input_shapes = raw.get('input_shape', [])
        if isinstance(input_shapes, list) and input_shapes and not isinstance(input_shapes[0], list):
            input_shapes = [input_shapes]

        dtypes = raw.get('dtype', [])
        if isinstance(dtypes, str):
            dtypes = [dtypes]
        # 单值 dtype 展开为与 input_shapes 相同长度的列表
        if len(dtypes) == 1 and len(input_shapes) > 1:
            dtypes = dtypes * len(input_shapes)

        return CaseInfo(
            rel_path=rel_path,
            operator=raw.get('operator', ''),
            case_id=raw.get('case_id', 0),
            input_shapes=input_shapes,
            dtypes=dtypes,
            attrs=raw.get('attrs', {}) or {},
            value_ranges=raw.get('value_range', []) or [],
            note=raw.get('note', '') or '',
            yaml_path=yaml_path,
            baseline_perf_us=_coerce_baseline_us(raw.get('baseline_perf_us')),
            t_hw_us=_coerce_baseline_us(raw.get('t_hw_us')),
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取用例统计"""
        cases = self.scan_all_cases()
        operator_counts = {}
        for case in cases:
            operator_counts[case.operator] = operator_counts.get(case.operator, 0) + 1
        return {
            'total': len(cases),
            'operators': operator_counts,
            'operator_dirs': [str(d.relative_to(self.bench_root)) for d in self._find_operator_dirs()]
        }