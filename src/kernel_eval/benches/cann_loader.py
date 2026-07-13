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
CANN 加载器（统一）

包含：
- CannTaskLoader: CANN 任务加载器（解析 proto.yaml）
- CannCaseLoader: CANN 用例加载器（解析 cases.yaml）
- GoldenLoader: Golden 函数动态导入器

职责：
1. 解析 proto.yaml 文件，返回 CannTaskSpec
2. 解析 cases.yaml 文件，返回 CannCaseSpec
3. 动态导入 golden.py 函数
4. 支持任意目录结构，递归扫描 proto.yaml

继承 base/ 目录的基类，实现 CANN 评测体系的加载逻辑。
"""

import importlib
import importlib.util
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from ..base.loaders import TaskLoader, CaseLoader, OperatorDirMixin, GoldenLoaderBase
from ..config import get_project_root
from ..utils.naming import camel_to_snake
from ..base.models import AttrSpec, TaskSpec, CaseSpec
from ..base.enums import DifficultyLevel
from .cann_spec import CannTaskSpec, CannCaseSpec, CannInputSpec, CannOutputSpec

logger = logging.getLogger(__name__)


# === 辅助函数 ===

# === CANN 任务加载器 ===

class CannTaskLoader(OperatorDirMixin, TaskLoader):
    """CANN 任务加载器

    继承 TaskLoader 基类，实现 CANN 评测体系的任务加载逻辑。
    返回 CannTaskSpec（CANN 特化任务规格）。
    """

    def __init__(self, bench_root: str = None, tasks_root: str = None):
        if tasks_root is not None:
            bench_root = tasks_root
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            self.bench_root = get_project_root() / "tasks"

        self._cache: Dict[str, CannTaskSpec] = {}

    def list_tasks(self) -> List[TaskSpec]:
        """列出所有任务（返回 CannTaskSpec 列表）"""
        operators = []
        for op_dir in self._find_operator_dirs():
            proto_path = op_dir / "proto.yaml"
            try:
                rel_path = str(op_dir.relative_to(self.bench_root))
                task_spec = self.get_task(rel_path)
                if task_spec:
                    operators.append(task_spec)
            except Exception as e:
                logger.warning("Failed to load operator from %s: %s", proto_path, e)
        return operators

    def get_task(self, task_id: str) -> Optional[TaskSpec]:
        """获取指定任务（返回 CannTaskSpec）"""
        if task_id in self._cache:
            return self._cache[task_id]

        op_dir = self.bench_root / task_id
        proto_path = op_dir / "proto.yaml"

        if not proto_path.exists():
            return None

        with open(proto_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'operator' not in data:
            logger.warning("proto.yaml format error: %s", proto_path)
            return None

        op_data = data['operator']
        task_spec = self._parse_operator(op_data, task_id, op_dir.name)
        self._cache[task_id] = task_spec
        return task_spec

    def get_statistics(self) -> Dict[str, Any]:
        """获取任务统计"""
        operators = self.list_tasks()
        categories = {}
        for op in operators:
            cat = getattr(op, 'category', '') or 'Unknown'
            categories[cat] = categories.get(cat, 0) + 1
        return {
            'total': len(operators),
            'operators': [op.name for op in operators],
            'rel_paths': [op.rel_path for op in operators],
            'categories': categories
        }

    # === CANN 特有方法（兼容旧接口） ===

    def get_operator(self, rel_path: str) -> CannTaskSpec:
        """获取算子定义信息（兼容旧接口）"""
        task_spec = self.get_task(rel_path)
        if task_spec is None:
            raise FileNotFoundError(f"proto.yaml不存在或格式错误: {self.bench_root / rel_path}")
        return task_spec

    def get_operator_by_name(self, operator: str) -> Optional[CannTaskSpec]:
        """按算子名称获取算子定义（兼容旧接口）"""
        for op_dir in self._find_operator_dirs():
            proto_path = op_dir / "proto.yaml"
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    op_name = data['operator'].get('name', '')
                    if op_name.lower() == operator.lower():
                        rel_path = str(op_dir.relative_to(self.bench_root))
                        return self.get_operator(rel_path)
            except Exception as e:
                logger.warning("Failed to parse proto.yaml at %s: %s", proto_path, e)
        return None

    def list_operators(self) -> List[CannTaskSpec]:
        """列出所有算子（兼容旧接口）"""
        return [op for op in self.list_tasks() if isinstance(op, CannTaskSpec)]

    # === 内部解析方法 ===

    def _parse_operator(self, data: Dict, rel_path: str, dir_name: str) -> CannTaskSpec:
        """解析算子定义"""
        attrs = []
        for attr_data in data.get('attrs', []) or []:
            attrs.append(AttrSpec(
                name=attr_data.get('name', ''),
                type=attr_data.get('type', ''),
                default=attr_data.get('default'),
                description=attr_data.get('description', ''),
            ))

        inputs = []
        for input_data in data.get('inputs', []) or []:
            dtype_list = input_data.get('dtype', [])
            if isinstance(dtype_list, str):
                dtype_list = [dtype_list]
            inputs.append(CannInputSpec(
                name=input_data.get('name', ''),
                description=input_data.get('description', ''),
                dtypes=dtype_list,
            ))

        outputs = []
        for output_data in data.get('outputs', []) or []:
            dtype_list = output_data.get('dtype', [])
            if isinstance(dtype_list, str):
                dtype_list = [dtype_list]
            outputs.append(CannOutputSpec(
                name=output_data.get('name', ''),
                description=output_data.get('description', ''),
                dtypes=dtype_list,
                compare=output_data.get('compare', True),
                index_gather=output_data.get('index_gather'),
            ))

        difficulty_str = data.get('difficulty', 'L1')
        difficulty = DifficultyLevel.L1
        if difficulty_str in ('L2', 'l2'):
            difficulty = DifficultyLevel.L2
        elif difficulty_str in ('L3', 'l3'):
            difficulty = DifficultyLevel.L3
        elif difficulty_str in ('L4', 'l4'):
            difficulty = DifficultyLevel.L4

        return CannTaskSpec(
            task_id=rel_path,
            name=data.get('name', ''),
            difficulty=difficulty,
            description=data.get('description', ''),
            rel_path=rel_path,
            formula=data.get('formula', ''),
            schema=data.get('schema', ''),
            shape_support=data.get('shape_support', ''),
            precision_thresholds=data.get('precision_thresholds', {}) or {},
            category=data.get('category', ''),
            note=data.get('note', ''),
            dir_name=dir_name,
            attrs=attrs,
            inputs=inputs,
            outputs=outputs,
        )


# === CANN 用例加载器 ===

class CannCaseLoader(OperatorDirMixin, CaseLoader):
    """CANN 用例加载器

    继承 CaseLoader 基类，实现 CANN 评测体系的用例加载逻辑。
    返回 CannCaseSpec（CANN 特化用例规格）。

    baseline 性能数据从评测集根目录下的 metadata/<hardware>.json 加载（BaselineStore），
    不再内嵌在 cases.yaml 中。BaselineStore 从 bench_root 向上查找 metadata/ 目录，
    确保子目录场景也能正确定位。
    """

    def __init__(self, bench_root: str = None, tasks_root: str = None):
        if tasks_root is not None:
            bench_root = tasks_root
        if bench_root is None:
            from ..config import get_project_root
            bench_root = str(get_project_root() / "tasks")
        self.bench_root = Path(bench_root)
        if not self.bench_root.exists():
            raise ValueError(f"bench目录不存在: {bench_root}")

        # 初始化 BaselineStore（从 bench_root 向上查找 metadata/<hardware>.json）
        from ..config import get_project_root as _get_root
        from ..utils.baseline_store import BaselineStore
        from ..utils.baseline_resolver import DEFAULT_HARDWARE, resolve_hardware

        # 自动检测当前 NPU 硬件，检测失败（如 CPU 模式）fallback 到默认值
        detected_hw = DEFAULT_HARDWARE
        try:
            from ..utils.device_manager import DeviceManager, DeviceConfig
            dm = DeviceManager(DeviceConfig(type="npu", device_id=0))
            device_name = dm.get_device_name()
            if device_name != "unknown":
                detected_hw = resolve_hardware(device_name)
        except Exception:
            pass

        self._baseline_store = BaselineStore(
            bench_root=self.bench_root,
            project_root=_get_root(),
            hardware=detected_hw
        )
        self._baseline_store.load()

    def _is_bench_root(self) -> bool:
        """检查当前目录是否为 bench 根目录"""
        if (self.bench_root / 'proto.yaml').exists():
            return not self._is_operator_dir(self.bench_root)
        return any(self._find_operator_dirs())

    def scan_all(self) -> List[CaseSpec]:
        """扫描所有用例（返回 CannCaseSpec 列表）"""
        if self._is_operator_dir(self.bench_root):
            return self._load_operator_cases(self.bench_root)
        else:
            all_cases = []
            for op_dir in self._find_operator_dirs():
                all_cases.extend(self._load_operator_cases(op_dir))
            return all_cases

    def scan_by_task(self, task_name: str) -> List[CaseSpec]:
        """扫描指定任务的用例"""
        all_cases = self.scan_all()
        return [c for c in all_cases if c.operator.lower() == task_name.lower()]

    def get_statistics(self) -> Dict[str, Any]:
        """获取用例统计"""
        cases = self.scan_all()
        operator_counts = {}
        for case in cases:
            operator_counts[case.operator] = operator_counts.get(case.operator, 0) + 1
        return {
            'total': len(cases),
            'operators': operator_counts,
            'operator_dirs': [str(d.relative_to(self.bench_root)) for d in self._find_operator_dirs()]
        }

    # === CANN 特有方法（兼容旧接口） ===

    def scan_all_cases(self) -> List[CannCaseSpec]:
        """扫描所有用例（兼容旧接口）"""
        return [c for c in self.scan_all() if isinstance(c, CannCaseSpec)]

    def scan_by_operator(self, operator: str) -> List[CannCaseSpec]:
        """扫描指定算子的用例（兼容旧接口）"""
        return [c for c in self.scan_by_task(operator) if isinstance(c, CannCaseSpec)]

    def scan_by_rel_path(self, rel_path: str) -> List[CannCaseSpec]:
        """扫描指定相对路径的用例（兼容旧接口）"""
        all_cases = self.scan_all()
        return [c for c in all_cases if c.rel_path == rel_path]

    # === 内部加载方法 ===

    def _load_operator_cases(self, op_dir: Path) -> List[CannCaseSpec]:
        """加载单个算子目录的用例"""
        cases_yaml = op_dir / "cases.yaml"
        if not cases_yaml.exists():
            return []

        try:
            rel_path = str(op_dir.relative_to(self.bench_root))
        except ValueError:
            rel_path = op_dir.name

        return self._load_yaml(cases_yaml, rel_path, op_dir)

    def _load_yaml(self, yaml_path: Path, rel_path: str, op_dir: Path = None) -> List[CannCaseSpec]:
        """解析 YAML 文件"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if not data or 'cases' not in data:
            return []

        op_dir_name = op_dir.name if op_dir else ""

        cases = []
        for raw in data['cases']:
            warnings = self._validate_case(raw, str(yaml_path))
            for warning in warnings:
                print(f"[WARN] {warning}")
            case = self._parse_case(raw, rel_path, str(yaml_path), op_dir_name)
            if case:
                cases.append(case)
        return cases

    def _validate_case(self, raw: Dict, yaml_path: str) -> List[str]:
        """校验 YAML 格式"""
        warnings = []

        case_id = raw.get('case_id')
        if case_id is None:
            warnings.append(f"{yaml_path}: missing 'case_id'")
        elif not isinstance(case_id, int):
            warnings.append(f"{yaml_path}: case_id should be int, got {type(case_id).__name__}")

        input_shapes = raw.get('input_shape', [])
        if not isinstance(input_shapes, list) or not input_shapes:
            warnings.append(f"{yaml_path}: 'input_shape' must be non-empty list")
        elif isinstance(input_shapes[0], int):
            warnings.append(f"{yaml_path}: input_shape is not nested list [[...]], auto-fixing")
        elif not all(isinstance(s, list) or s is None for s in input_shapes):
            warnings.append(f"{yaml_path}: input_shape elements should all be lists or None")

        dtypes = raw.get('dtype', [])
        if isinstance(dtypes, str):
            dtypes = [dtypes]
        if input_shapes and isinstance(input_shapes[0], list):
            if len(dtypes) == 1:
                pass
            elif len(dtypes) != len(input_shapes):
                warnings.append(f"{yaml_path}: dtype len={len(dtypes)} != input_shape len={len(input_shapes)}")

        return warnings

    def _parse_case(self, raw: Dict, rel_path: str, yaml_path: str, op_dir_name: str = "") -> CannCaseSpec:
        """解析单个用例

        baseline 性能数据优先从 BaselineStore 查询（集中式 JSON 文件）；
        若 JSON 文件不存在或查询不到，fallback 到 raw YAML 数据（向后兼容）。
        """
        input_shapes = raw.get('input_shape', [])
        if isinstance(input_shapes, list) and input_shapes and not isinstance(input_shapes[0], list):
            input_shapes = [input_shapes]

        dtypes = raw.get('dtype', [])
        if isinstance(dtypes, str):
            dtypes = [dtypes]
        if len(dtypes) == 1 and len(input_shapes) > 1:
            dtypes = dtypes * len(input_shapes)

        case_id_int = raw.get('case_id', 0)
        display_path = op_dir_name if op_dir_name and rel_path == "." else rel_path
        case_id_str = f"{display_path}_{case_id_int}"

        # 计算 baseline 查询路径：
        # 当 rel_path="."（单算子目录模式）时，需要相对于 metadata/ 所在目录
        # 计算真正的路径，如 "level1/exp"
        baseline_rel_path = rel_path
        if rel_path == "." and self._baseline_store._baseline_dir is not None:
            # metadata/ 的父目录就是评测集根目录（如 tasks/）
            baseline_root = self._baseline_store._baseline_dir.parent
            try:
                baseline_rel_path = str(self.bench_root.relative_to(baseline_root))
            except ValueError:
                baseline_rel_path = op_dir_name

        # 从 BaselineStore 查询 baseline 数据
        baseline_perf_us = self._baseline_store.get_perf(baseline_rel_path, case_id_int)
        t_hw_us = self._baseline_store.get_t_hw(baseline_rel_path, case_id_int)

        return CannCaseSpec(
            case_id=case_id_str,
            rel_path=rel_path,
            operator=raw.get('operator', ''),
            case_num=case_id_int,
            input_shapes=input_shapes,
            dtypes=dtypes,
            attrs=raw.get('attrs', {}) or {},
            value_ranges=raw.get('value_range', []) or [],
            note=raw.get('note', '') or '',
            yaml_path=yaml_path,
            baseline_perf_us=baseline_perf_us,
            t_hw_us=t_hw_us,
            metadata={'op_dir_name': op_dir_name},
        )


# === Golden 加载器 ===

class GoldenLoader(GoldenLoaderBase):
    """Golden函数动态导入器（cann-bench 评测体系）

    从 tasks/{level}/{op}/golden.py 加载 golden 函数。
    """

    def __init__(self, bench_root: str = None):
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            self.bench_root = get_project_root() / "tasks"
        self._func_cache: Dict[str, str] = {}
        self._module_cache: Dict[str, object] = {}

    def _load_module(self, rel_path: str):
        """导入 golden 模块（带缓存）"""
        if rel_path in self._module_cache:
            return self._module_cache[rel_path]

        module_path = self.bench_root / rel_path / "golden.py"
        if not module_path.exists():
            raise ImportError(f"Golden模块不存在: {module_path}")

        module_name = f"tasks.{rel_path.replace('/', '.')}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._module_cache[rel_path] = module
        return module

    def get_golden_function(self, rel_path: str) -> Callable:
        """获取golden函数"""
        module = self._load_module(rel_path)
        func_name = self._get_function_name(rel_path)
        if not hasattr(module, func_name):
            func_name = self._get_operator_name(rel_path).lower()
            if not hasattr(module, func_name):
                raise AttributeError(
                    f"模块 tasks.{rel_path.replace('/', '.')} 中找不到函数: {func_name}")
        return getattr(module, func_name)

    def get_mc2_distributed_golden(self, rel_path: str, required: bool = False) -> Optional[Callable]:
        """获取 MC2 多卡分布式 golden 钩子（golden.py 中的 mc2_distributed_golden）。

        MC2 类算子的 golden 逻辑依赖多 rank HCCL 通信，无法走单进程 golden 路径，
        改由各 task 的 golden.py 提供名为 ``mc2_distributed_golden`` 的可选钩子。

        Args:
            rel_path: 算子相对路径
            required: 为 True 时若钩子缺失则抛 AttributeError；否则返回 None

        Returns:
            钩子函数；缺失且 required=False 时返回 None
        """
        module = self._load_module(rel_path)
        hook = getattr(module, "mc2_distributed_golden", None)
        if hook is None and required:
            raise AttributeError(
                f"模块 tasks.{rel_path.replace('/', '.')} 中找不到分布式 golden 钩子: "
                f"mc2_distributed_golden")
        return hook

    def get_oracle_function(self, rel_path: str, required: bool = False) -> Optional[Callable]:
        """获取可选的 oracle 钩子(dtype-agnostic 的 fp64 真值实现)。

        golden.py 可提供一个名为 ``<golden_func_name>_oracle`` 的顶层函数,签名与 golden 一致,
        且不在体内硬编码 .float()/.double() —— 计算精度随 evaluator 喂入的输入精度走。
        在 golden_precision=fp64_cpu 下即为真 fp64 oracle(g)。缺失时返回 None,evaluator 回退
        到用 golden 本身(与现状一致,不影响未接入的算子)。

        Args:
            rel_path: 算子相对路径,如 ``level3/weight_quant_batch_matmul``
            required: 为 True 时缺失则抛 AttributeError;否则返回 None
        """
        module = self._load_module(rel_path)
        func_name = self._get_function_name(rel_path)
        hook = getattr(module, f"{func_name}_oracle", None)
        if hook is None and required:
            raise AttributeError(
                f"模块 tasks.{rel_path.replace('/', '.')} 中找不到 oracle 钩子: "
                f"{func_name}_oracle")
        return hook

    def get_bench_function(self, rel_path: str, required: bool = False) -> Optional[Callable]:
        """获取可选的 bench 钩子(同精度参考实现,即 checker 的 b)。

        golden.py 可提供一个名为 ``<golden_func_name>_bench`` 的顶层函数,签名与 golden 一致。
        它按该算子的**同精度约定**计算(如 weight-only A16W8:反量化到输出精度 + fp32 累加,
        与 torchao / Marlin 等库一致),作为"正确实现应有的误差下限"供 evaluator 作同精度参考
        (b) —— 使 |b−oracle| 不再恒为 0。**非对某颗硬件的复刻**(硬件可更高精度实现,会
        meet-or-exceed 此下限、照常通过)。缺失时返回 None,evaluator 回退到用 golden 本身作
        参考(与现状一致,不影响未接入的算子)。

        Args:
            rel_path: 算子相对路径
            required: 为 True 时缺失则抛 AttributeError;否则返回 None
        """
        module = self._load_module(rel_path)
        func_name = self._get_function_name(rel_path)
        hook = getattr(module, f"{func_name}_bench", None)
        if hook is None and required:
            raise AttributeError(
                f"模块 tasks.{rel_path.replace('/', '.')} 中找不到 bench 钩子: "
                f"{func_name}_bench")
        return hook

    def _get_operator_name(self, rel_path: str) -> str:
        """从 proto.yaml 获取算子名称"""
        proto_path = self.bench_root / rel_path / "proto.yaml"
        if proto_path.exists():
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    return data['operator'].get('name', '')
            except Exception as e:
                logger.warning("Failed to parse proto.yaml at %s: %s", proto_path, e)
        return Path(rel_path).name

    def _get_function_name(self, rel_path: str) -> str:
        """从proto.yaml获取函数名"""
        if rel_path in self._func_cache:
            return self._func_cache[rel_path]

        proto_path = self.bench_root / rel_path / "proto.yaml"
        if proto_path.exists():
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    schema = data['operator'].get('schema', '')
                    match = re.match(r'^(\w+)\s*\(', schema.strip())
                    if match:
                        self._func_cache[rel_path] = match.group(1)
                        return match.group(1)
            except Exception as e:
                logger.warning("Failed to parse proto.yaml at %s: %s", proto_path, e)

        return self._get_operator_name(rel_path).lower()

    def get_operator_dir(self, rel_path: str) -> Path:
        """获取算子目录路径"""
        return self.bench_root / rel_path

    def get_input_function(self, rel_path: str) -> Optional[Callable]:
        """获取 get_input 函数（可选）"""
        module = self._load_module(rel_path)
        if hasattr(module, 'get_input'):
            return getattr(module, 'get_input')
        return None

    def get_golden_by_operator_name(self, operator: str) -> Callable:
        """按算子名称查找golden函数（遍历查找）"""
        for proto_path in self.bench_root.rglob("proto.yaml"):
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    op_name = data['operator'].get('name', '')
                    if op_name.lower() == operator.lower():
                        op_dir = proto_path.parent
                        rel_path = str(op_dir.relative_to(self.bench_root))
                        return self.get_golden_function(rel_path)
            except Exception as e:
                logger.warning("Failed to parse proto.yaml at %s: %s", proto_path, e)
        raise ImportError(f"未找到算子 {operator} 的golden函数")