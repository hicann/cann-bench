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
算子定义加载器

职责：
1. 解析 proto.yaml 文件
2. 提供算子schema、attrs、inputs、outputs信息
3. 支持按level/operator查询
"""

import yaml
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class AttrInfo:
    """算子属性信息"""
    name: str
    type: str
    default: Any = None
    description: str = ""


@dataclass
class TensorInfo:
    """张量信息"""
    name: str
    description: str = ""
    dtype: List[str] = field(default_factory=list)
    compare: bool = True  # 是否参与精度对比，默认True


@dataclass
class OperatorInfo:
    """算子定义信息"""
    name: str
    level: int
    category: str = ""
    difficulty: str = ""
    formula: str = ""
    description: str = ""
    shape_support: str = ""
    note: str = ""
    precision_thresholds: Dict[str, float] = field(default_factory=dict)  # 自定义精度阈值
    attrs: List[AttrInfo] = field(default_factory=list)
    inputs: List[TensorInfo] = field(default_factory=list)
    outputs: List[TensorInfo] = field(default_factory=list)
    schema: str = ""
    dir_name: str = ""  # 实际目录名

    def get_function_name(self) -> str:
        """从schema解析函数名"""
        if self.schema:
            match = re.match(r'^(\w+)\s*\(', self.schema.strip())
            if match:
                return match.group(1)
        return self.name.lower()


class OperatorLoader:
    """算子定义加载器"""

    def __init__(self, bench_root: str = None):
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            # 默认使用项目根目录下的kernel_bench
            project_root = Path(__file__).parent.parent.parent.parent
            self.bench_root = project_root / "kernel_bench"

        self._cache: Dict[str, OperatorInfo] = {}
        self._dir_cache: Dict[int, Dict[str, str]] = {}

    def _camel_to_snake(self, name: str) -> str:
        """将 PascalCase 名称转换为 snake_case"""
        s0 = re.sub(r'([0-9])([A-Z])', r'\1_\2', name)
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s0)
        return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _get_dir_name(self, level: int, operator: str) -> str:
        """将算子名解析为实际目录名"""
        cache_key = f"L{level}_{operator}"
        if cache_key in self._dir_cache:
            return self._dir_cache[cache_key]

        # 尝试多种命名方式
        level_dir = self.bench_root / f"level{level}"
        candidates = [
            operator,           # 原名
            operator.lower(),   # 小写
            self._camel_to_snake(operator),  # snake_case
        ]

        for candidate in candidates:
            path = level_dir / candidate
            if path.exists() and (path / "proto.yaml").exists():
                self._dir_cache[cache_key] = candidate
                return candidate

        # 扫描目录查找匹配的proto.yaml
        if level_dir.is_dir():
            for entry in level_dir.iterdir():
                if entry.is_dir() and not entry.name.startswith('.'):
                    proto_path = entry / "proto.yaml"
                    if proto_path.exists():
                        try:
                            with open(proto_path, 'r', encoding='utf-8') as f:
                                data = yaml.safe_load(f)
                            if data and 'operator' in data:
                                op_name = data['operator'].get('name', '')
                                if op_name == operator:
                                    self._dir_cache[cache_key] = entry.name
                                    return entry.name
                        except Exception:
                            pass

        # 最终 fallback
        self._dir_cache[cache_key] = self._camel_to_snake(operator)
        return self._dir_cache[cache_key]

    def get_operator(self, operator: str, level: int) -> OperatorInfo:
        """获取算子定义信息"""
        cache_key = f"L{level}_{operator}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        dir_name = self._get_dir_name(level, operator)
        proto_path = self.bench_root / f"level{level}" / dir_name / "proto.yaml"

        if not proto_path.exists():
            raise FileNotFoundError(f"proto.yaml不存在: {proto_path}")

        with open(proto_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'operator' not in data:
            raise ValueError(f"proto.yaml格式错误: {proto_path}")

        op_data = data['operator']
        op_info = self._parse_operator(op_data, level, dir_name)
        self._cache[cache_key] = op_info
        return op_info

    def _parse_operator(self, data: Dict, level: int, dir_name: str) -> OperatorInfo:
        """解析算子定义"""
        # 解析attrs
        attrs = []
        for attr_data in data.get('attrs', []) or []:
            attrs.append(AttrInfo(
                name=attr_data.get('name', ''),
                type=attr_data.get('type', ''),
                default=attr_data.get('default'),
                description=attr_data.get('description', '')
            ))

        # 解析inputs
        inputs = []
        for input_data in data.get('inputs', []) or []:
            dtype_list = input_data.get('dtype', [])
            if isinstance(dtype_list, str):
                dtype_list = [dtype_list]
            inputs.append(TensorInfo(
                name=input_data.get('name', ''),
                description=input_data.get('description', ''),
                dtype=dtype_list
            ))

        # 解析outputs
        outputs = []
        for output_data in data.get('outputs', []) or []:
            dtype_list = output_data.get('dtype', [])
            if isinstance(dtype_list, str):
                dtype_list = [dtype_list]
            outputs.append(TensorInfo(
                name=output_data.get('name', ''),
                description=output_data.get('description', ''),
                dtype=dtype_list,
                compare=output_data.get('compare', True)
            ))

        # 解析自定义精度阈值
        precision_thresholds = data.get('precision_thresholds', {}) or {}

        return OperatorInfo(
            name=data.get('name', ''),
            level=level,
            category=data.get('category', ''),
            difficulty=data.get('difficulty', ''),
            formula=data.get('formula', ''),
            description=data.get('description', ''),
            shape_support=data.get('shape_support', ''),
            note=data.get('note', ''),
            precision_thresholds=precision_thresholds,
            attrs=attrs,
            inputs=inputs,
            outputs=outputs,
            schema=data.get('schema', ''),
            dir_name=dir_name
        )

    def list_operators(self, level: int = None) -> List[OperatorInfo]:
        """列出算子"""
        operators = []
        levels = [level] if level else [1, 2, 3, 4]

        for lv in levels:
            level_dir = self.bench_root / f"level{lv}"
            if not level_dir.is_dir():
                continue

            for entry in level_dir.iterdir():
                if entry.is_dir() and not entry.name.startswith('.'):
                    proto_path = entry / "proto.yaml"
                    if proto_path.exists():
                        try:
                            op_info = self.get_operator_from_path(proto_path, lv, entry.name)
                            operators.append(op_info)
                        except Exception:
                            pass

        return operators

    def get_operator_from_path(self, proto_path: Path, level: int, dir_name: str) -> OperatorInfo:
        """从proto.yaml路径加载算子信息"""
        with open(proto_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'operator' not in data:
            raise ValueError(f"proto.yaml格式错误: {proto_path}")

        return self._parse_operator(data['operator'], level, dir_name)

    def get_statistics(self) -> Dict[int, Dict[str, Any]]:
        """获取算子统计"""
        stats = {}
        for level in [1, 2, 3, 4]:
            operators = self.list_operators(level)
            stats[level] = {
                'total': len(operators),
                'operators': [op.name for op in operators],
                'categories': {}
            }
            for op in operators:
                cat = op.category or 'Unknown'
                stats[level]['categories'][cat] = stats[level]['categories'].get(cat, 0) + 1
        return stats