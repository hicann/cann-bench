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
CANN 解决方案规格（特化）

设计理念:
- CannSolutionSpec 继承 SolutionSpec
- 支持 Golden 自验证场景（通过 golden_reference 字段）
- 支持多 Backend（torch_npu、ascendc 等）
- 支持动态生成代码

使用场景:
1. 正常评测: source="ai_op.py", golden_reference="file"
2. Golden 自验证: source="golden.py", golden_reference="fp64_cpu"
3. 动态生成: source_type="generated", source="<code>", golden_reference="file"

Why: 提供 cann-bench 特化的解决方案定义，支持灵活的评测场景
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..base.models import SolutionSpec
from ..base.enums import BackendType, SourceType, GoldenReference


@dataclass
class CannSolutionSpec(SolutionSpec):
    """CANN 解决方案规格（特化）

    继承 SolutionSpec，添加 cann-bench 特有字段:
    - rel_path: 关联算子路径
    - golden_reference: Golden 参考来源
    - auto_load: 是否自动加载
    - compare_indices: 需要对比的输出索引
    - custom_thresholds: 自定义精度阈值

    特有字段说明:
    - golden_reference:
        * "file": 使用 golden.py 作为参考（正常评测）
        * "self": 使用自身作为参考（同精度对比）
        * "fp64_cpu": 使用 CPU fp64 计算作为参考（Golden 自验证）
    """
    rel_path: str = ""                          # 关联算子路径
    golden_reference: GoldenReference = GoldenReference.FILE  # Golden 参考来源
    auto_load: bool = True                      # 是否自动加载
    compare_indices: List[int] = field(default_factory=list)  # 需要对比的输出索引
    custom_thresholds: Dict[str, float] = field(default_factory=dict)  # 自定义精度阈值
    checker_name: str = "relative_error"          # 精度判断器名称

    def is_golden_verify(self) -> bool:
        """是否为 Golden 自验证场景"""
        return self.golden_reference == GoldenReference.FP64_CPU or \
               self.golden_reference == GoldenReference.SELF

    def is_normal_eval(self) -> bool:
        """是否为正常评测场景"""
        return self.golden_reference == GoldenReference.FILE

    def get_source_path(self, task_dir: str) -> str:
        """获取源码完整路径"""
        if self.is_file_source():
            from pathlib import Path
            return str(Path(task_dir) / self.source)
        return self.source

    def get_golden_path(self, task_dir: str) -> str:
        """获取 Golden 文件路径"""
        from pathlib import Path
        return str(Path(task_dir) / "golden.py")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base = super().to_dict()
        base.update({
            'rel_path': self.rel_path,
            'golden_reference': self.golden_reference.value,
            'auto_load': self.auto_load,
            'compare_indices': self.compare_indices,
            'custom_thresholds': self.custom_thresholds,
            'checker_name': self.checker_name,
        })
        return base

    @classmethod
    def create_ai_solution(cls, rel_path: str, task_id: str) -> 'CannSolutionSpec':
        """创建 AI 解决方案（正常评测）"""
        return cls(
            solution_id=f"{task_id}_ai",
            task_id=task_id,
            rel_path=rel_path,
            name="AI Implementation",
            backend=BackendType.TORCH_NPU,
            source_type=SourceType.FILE,
            source="ai_op.py",
            golden_reference=GoldenReference.FILE,
        )

    @classmethod
    def create_golden_verify_solution(cls, rel_path: str, task_id: str) -> 'CannSolutionSpec':
        """创建 Golden 自验证解决方案"""
        return cls(
            solution_id=f"{task_id}_golden_verify",
            task_id=task_id,
            rel_path=rel_path,
            name="Golden Verification",
            backend=BackendType.TORCH,  # CPU 运行
            source_type=SourceType.FILE,
            source="golden.py",
            golden_reference=GoldenReference.FP64_CPU,
        )

    @classmethod
    def create_generated_solution(cls, rel_path: str, task_id: str, code: str) -> 'CannSolutionSpec':
        """创建动态生成解决方案"""
        return cls(
            solution_id=f"{task_id}_generated",
            task_id=task_id,
            rel_path=rel_path,
            name="Generated Implementation",
            backend=BackendType.TORCH_NPU,
            source_type=SourceType.GENERATED,
            source=code,
            golden_reference=GoldenReference.FILE,
        )