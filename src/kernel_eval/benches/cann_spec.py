#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
CANN 任务和用例规格（特化）

设计理念:
- CannTaskSpec 继承 TaskSpec，添加 cann-bench 特有字段
- CannCaseSpec 继承 CaseSpec，添加 cann-bench 特有字段（baseline、t_hw 等）
- InputSpec/OutputSpec 无需分层（通过 metadata 或 cann_outputs 扩展）

Why: 提供 cann-bench 特化的任务和用例定义，支持性能基线、Golden 验证等场景
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base.models import TaskSpec, CaseSpec, InputSpec, OutputSpec
from ..base.enums import DifficultyLevel


@dataclass
class CannInputSpec(InputSpec):
    """CANN 输入规格（特化）

    继承 InputSpec，添加多 dtype 支持。
    """
    dtypes: List[str] = field(default_factory=list)  # 支持的多种 dtype

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base = super().to_dict()
        base['dtypes'] = self.dtypes
        return base


@dataclass
class CannOutputSpec(OutputSpec):
    """CANN 输出规格（特化）

    继承 OutputSpec，添加多 dtype 支持。
    """
    dtypes: List[str] = field(default_factory=list)  # 支持的多种 dtype

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base = super().to_dict()
        base['dtypes'] = self.dtypes
        return base


@dataclass
class CannTaskSpec(TaskSpec):
    """CANN 任务规格（特化）

    继承 TaskSpec，添加 cann-bench 特有字段:
    - rel_path: 相对路径（用于定位算子目录）
    - formula: 数学公式
    - schema: 函数签名
    - shape_support: 形态支持说明
    - precision_thresholds: 自定义精度阈值
    - category: 类别

    特有字段说明:
    - rel_path: 如 "level2/scatter"，用于定位 tasks/level2/scatter 目录
    - schema: 函数签名，如 "scatter_add(tensor, index, src, dim)"
    - precision_thresholds: 覆盖默认阈值，如 {"float16": 0.002}
    """
    rel_path: str = ""                          # 相对路径
    formula: str = ""                           # 数学公式
    schema: str = ""                            # 函数签名
    shape_support: str = ""                     # 形态支持说明
    precision_thresholds: Dict[str, float] = field(default_factory=dict)  # 自定义精度阈值
    category: str = ""                          # 类别
    note: str = ""                              # 备注
    dir_name: str = ""                          # 实际目录名

    # 覆盖 inputs/outputs 类型为 CannInputSpec/CannOutputSpec
    inputs: List[CannInputSpec] = field(default_factory=list)
    outputs: List[CannOutputSpec] = field(default_factory=list)

    def get_function_name(self) -> str:
        """从 schema 解析函数名"""
        import re
        if self.schema:
            match = re.match(r'^(\w+)\s*\(', self.schema.strip())
            if match:
                return match.group(1)
        return self.name.lower()

    def get_threshold(self, dtype: str) -> float:
        """获取指定 dtype 的精度阈值"""
        # 优先使用自定义阈值
        dtype_lower = dtype.lower()
        if dtype_lower in self.precision_thresholds:
            return self.precision_thresholds[dtype_lower]
        # 使用默认阈值
        from ..utils.thresholds import get_threshold
        return get_threshold(dtype)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base = super().to_dict()
        base.update({
            'rel_path': self.rel_path,
            'formula': self.formula,
            'schema': self.schema,
            'shape_support': self.shape_support,
            'precision_thresholds': self.precision_thresholds,
            'category': self.category,
            'note': self.note,
            'dir_name': self.dir_name,
        })
        return base


@dataclass
class CannCaseSpec(CaseSpec):
    """CANN 用例规格（特化）

    继承 CaseSpec，添加 cann-bench 特有字段:
    - rel_path: 关联算子路径
    - operator: 算子名称
    - baseline_perf_us: 性能基线
    - t_hw_us: 硬件下界
    - yaml_path: YAML 文件路径
    - compare_outputs: 各输出是否参与对比

    特有字段说明:
    - baseline_perf_us: 性能基线（用于加速比计算）
    - t_hw_us: 理论硬件下界（用于 SOL-Score 计算）
    - compare_outputs: 继承 output_spec.compare 的列表形式
    """
    rel_path: str = ""                          # 关联算子路径
    operator: str = ""                          # 算子名称
    case_num: int = 0                           # 用例编号（数字）
    baseline_perf_us: float = 0.0               # 性能基线
    t_hw_us: float = 0.0                        # 理论硬件下界
    yaml_path: str = ""                         # YAML 文件路径
    note: str = ""                              # 备注
    compare_outputs: List[bool] = field(default_factory=list)  # 各输出是否对比

    def get_case_id_str(self) -> str:
        """获取完整用例标识字符串"""
        # 格式: rel_path_case_num，如 "level2/scatter_1"
        if self.rel_path and self.case_num:
            return f"{self.rel_path}_{self.case_num}"
        return self.case_id

    def get_speedup(self, elapsed_us: float) -> float:
        """计算加速比"""
        if self.baseline_perf_us > 0 and elapsed_us > 0:
            return self.baseline_perf_us / elapsed_us
        return 0.0

    def get_perf_score(self, elapsed_us: float) -> Optional[float]:
        """计算性能得分"""
        if elapsed_us <= 0 or self.baseline_perf_us <= 0:
            return None
        from ..report.scoring import per_case_sol_score
        return per_case_sol_score(
            float(self.baseline_perf_us),
            float(elapsed_us),
            float(self.t_hw_us),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        d = super().to_dict()
        d.update({
            'yaml_path': self.yaml_path,
            'note': self.note,
            'compare_outputs': self.compare_outputs,
        })
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CannCaseSpec':
        """从字典重建"""
        return cls(
            case_id=data.get('case_id', ''),
            operator=data.get('operator', ''),
            rel_path=data.get('rel_path', ''),
            case_num=int(data.get('case_num', 0)),
            baseline_perf_us=float(data.get('baseline_perf_us', 0.0)),
            t_hw_us=float(data.get('t_hw_us', 0.0)),
            input_shapes=data.get('input_shapes', []),
            dtypes=data.get('dtypes', []),
            attrs=data.get('attrs', {}),
            value_ranges=data.get('value_ranges', []),
            tolerance=data.get('tolerance', {"rtol": 1e-4, "atol": 1e-4}),
            metadata=data.get('metadata', {}),
            yaml_path=data.get('yaml_path', ''),
            note=data.get('note', ''),
            compare_outputs=data.get('compare_outputs', []),
        )