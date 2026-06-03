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
StanfordBench 加载器

包含：
- StanfordTaskLoader: 任务加载器（扫描 .py 文件）
- StanfordCaseLoader: 用例加载器（隐式用例）
- StanfordGoldenLoader: Golden 函数加载器

设计原则：
- 极简：TaskSpec/CaseSpec 字段全空
- 复用：利用现有 Evaluator 对空参数的兼容性
"""

import importlib.util
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import torch

from ..base.loaders import TaskLoader, CaseLoader, GoldenLoaderBase
from ..base.models import TaskSpec, CaseSpec
from ..base.enums import DifficultyLevel
from ..config import get_project_root


class StanfordTaskLoader(TaskLoader):
    """StanfordBench 任务加载器

    极简实现：扫描 .py 文件，返回空 inputs/outputs。
    不解析 Model.forward 签名（StanfordBench 由 Model 自动处理）。
    """

    def __init__(self, bench_root: str = None, tasks_root: str = None):
        if tasks_root:
            bench_root = tasks_root
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            self.bench_root = get_project_root() / "thirdparty" / "KernelBench" / "KernelBench"
        self._cache: Dict[str, TaskSpec] = {}

    def list_tasks(self) -> List[TaskSpec]:
        """扫描所有包含 Model/ModelNew 的 .py 文件"""
        tasks = []
        for py_path in self._discover_py_files():
            task_id = self._py_to_task_id(py_path)
            if task_id not in self._cache:
                self._cache[task_id] = TaskSpec(
                    task_id=task_id,
                    name=self._extract_name(py_path.stem),
                    rel_path=task_id,
                    inputs=[],   # 空：不预定义 inputs
                    outputs=[],  # 空：不预定义 outputs
                    description="",
                    metadata={
                        'py_path': str(py_path),
                        'py_stem': py_path.stem,  # 原始文件名，用于匹配 baseline
                    },
                )
            tasks.append(self._cache[task_id])
        return tasks

    def get_task(self, task_id: str) -> Optional[TaskSpec]:
        """获取单个任务"""
        if task_id in self._cache:
            return self._cache[task_id]
        # 尝试查找文件
        py_path = self._task_id_to_py(task_id)
        if py_path:
            task = TaskSpec(
                task_id=task_id,
                name=self._extract_name(py_path.stem),
                rel_path=task_id,
                inputs=[], outputs=[], metadata={'py_path': str(py_path)},
            )
            self._cache[task_id] = task
            return task
        return None

    def get_statistics(self) -> Dict[str, Any]:
        return {'total': len(self.list_tasks())}

    # 兼容旧接口（CANN 风格）
    def get_operator(self, rel_path: str) -> TaskSpec:
        task = self.get_task(rel_path)
        if task is None:
            raise FileNotFoundError(f"任务不存在: {rel_path}")
        return task

    def get_operator_by_name(self, name: str) -> Optional[TaskSpec]:
        for task in self.list_tasks():
            if task.name.lower() == name.lower():
                return task
        return None

    def list_operators(self) -> List[TaskSpec]:
        return self.list_tasks()

    def _discover_py_files(self) -> List[Path]:
        """递归查找包含 Model/ModelNew 类的 .py 文件"""
        py_files = []
        if not self.bench_root.exists():
            return py_files
        for py_path in self.bench_root.rglob("*.py"):
            if py_path.name.startswith('_'):
                continue
            try:
                content = py_path.read_text(encoding='utf-8')
                if 'class Model' in content or 'class ModelNew' in content:
                    py_files.append(py_path)
            except Exception:
                pass
        return sorted(py_files)

    def _py_to_task_id(self, py_path: Path) -> str:
        """将 py_path 转换为 task_id 格式: level1/Softmax"""
        try:
            rel = py_path.relative_to(self.bench_root)
            parts = list(rel.parts)

            # 在 parts 中查找 level 目录
            for p in parts:
                m = re.match(r'level(\d+)', p)
                if m:
                    return f"level{m.group(1)}/{py_path.stem}"

            # 检查 bench_root 是否是 level 目录
            m_root = re.match(r'level(\d+)', self.bench_root.name)
            if m_root:
                return f"level{m_root.group(1)}/{py_path.stem}"

        except ValueError:
            pass

        # 无 level 目录，直接用 stem
        return py_path.stem

    def _task_id_to_py(self, task_id: str) -> Optional[Path]:
        """将 task_id 转换回 py_path: level1/Softmax → .../level1/Softmax.py"""
        if '/' in task_id:
            level, name = task_id.split('/', 1)
            # 检查 bench_root 本身是否是 level 目录
            m_root = re.match(r'level(\d+)', self.bench_root.name)
            if m_root and self.bench_root.name == level:
                # bench_root 本身就是目标 level 目录
                py_path = self.bench_root / f"{name}.py"
                if py_path.exists():
                    return py_path
            # 否则在 bench_root 下查找 level 子目录
            for level_dir in self.bench_root.rglob(level):
                py_path = level_dir / f"{name}.py"
                if py_path.exists():
                    return py_path
        else:
            for py_path in self.bench_root.rglob(f"{task_id}.py"):
                if py_path.exists():
                    return py_path
        return None

    def _extract_name(self, stem: str) -> str:
        """从文件名提取算子名: 23_Softmax_1 → Softmax"""
        # 移除数字前缀和后缀
        name = re.sub(r'^\d+_', '', stem)
        name = re.sub(r'_\d+$', '', name)
        # 拼接成 PascalCase
        return ''.join(p.capitalize() for p in name.split('_') if p)


class StanfordCaseLoader(CaseLoader):
    """StanfordBench 用例加载器

    极简实现：每个算子一个隐式用例。
    input_shapes=[], dtypes=[], attrs={} 全空。
    输入数据由 get_inputs() 动态生成。

    Baseline 性能数据从 data/stanford_baseline.json 加载。
    """

    def __init__(self, bench_root: str = None, tasks_root: str = None):
        if tasks_root:
            bench_root = tasks_root
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            self.bench_root = get_project_root() / "thirdparty" / "KernelBench" / "KernelBench"

        # 加载 baseline 性能数据
        self._baseline_data = self._load_baseline()

    def _load_baseline(self) -> Dict[str, Dict[str, Optional[float]]]:
        """加载 baseline.json: {level: {name: perf_us}}"""
        baseline_path = get_project_root() / "data" / "stanford_baseline.json"
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _get_baseline_perf(self, task_id: str, py_stem: str) -> float:
        """获取 baseline 性能值

        Args:
            task_id: 如 "level1/Softmax"
            py_stem: 原始文件名，如 "23_Softmax_1"

        Returns:
            baseline_perf_us，若未找到返回 0.0
        """
        if '/' in task_id:
            level = task_id.split('/')[0]
            level_data = self._baseline_data.get(level, {})
            # 先尝试原始文件名
            perf = level_data.get(py_stem)
            if perf is not None:
                return float(perf)
        return 0.0

    def scan_all(self) -> List[CaseSpec]:
        """每个算子一个隐式用例"""
        task_loader = StanfordTaskLoader(str(self.bench_root))
        cases = []
        for task in task_loader.list_tasks():
            py_stem = task.metadata.get('py_stem', task.task_id.split('/')[-1])
            baseline_perf = self._get_baseline_perf(task.task_id, py_stem)
            cases.append(CaseSpec(
                case_id=f"{task.task_id}_1",
                operator=task.name,           # 算子名称
                rel_path=task.task_id,        # 相对路径
                case_num=1,                   # 用例编号
                baseline_perf_us=baseline_perf,  # 基线性能（从 JSON 加载）
                t_hw_us=0.0,                  # StanfordBench 无理论硬件下界
                # StanfordBench: 其他字段空
                input_shapes=[],   # 空
                dtypes=[],         # 空
                attrs={},          # 空
                value_ranges=[],   # 空
                metadata={
                    'py_stem': py_stem,
                },
            ))
        return cases

    def scan_by_task(self, task_name: str) -> List[CaseSpec]:
        """按算子名筛选用例"""
        return [
            c for c in self.scan_all()
            if c.operator.lower() == task_name.lower()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        cases = self.scan_all()
        operator_counts = {}
        for case in cases:
            op = case.operator
            operator_counts[op] = operator_counts.get(op, 0) + 1
        return {'total': len(cases), 'operators': operator_counts}

    # 兼容旧接口（CANN 风格）
    def scan_all_cases(self) -> List[CaseSpec]:
        return self.scan_all()

    def scan_by_operator(self, operator: str) -> List[CaseSpec]:
        return self.scan_by_task(operator)

    def scan_by_rel_path(self, rel_path: str) -> List[CaseSpec]:
        return [c for c in self.scan_all() if c.rel_path == rel_path]


class StanfordGoldenLoader(GoldenLoaderBase):
    """StanfordBench Golden 加载器

    关键设计：
    - 直接返回 get_inputs，不做参数包装
    - Evaluator 调用 get_input_func(**{}) → get_inputs() 无参数调用
    - Model 需要 device wrapper 处理权重 device 切换
    """

    def __init__(self, bench_root: str = None, random_seed: int = 42):
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            self.bench_root = get_project_root() / "thirdparty" / "KernelBench" / "KernelBench"
        self._random_seed = random_seed
        self._module_cache: Dict[str, Any] = {}

    def get_golden_function(self, task_id: str) -> Callable:
        """加载 Model.forward，返回 device wrapper"""
        torch.manual_seed(self._random_seed)

        module = self._load_module(task_id)
        model_cls = getattr(module, 'Model', None) or getattr(module, 'ModelNew', None)
        if not model_cls:
            raise ImportError(f"{task_id} 缺少 Model/ModelNew 类")

        init_inputs = self.get_init_inputs(task_id)

        if isinstance(init_inputs, list) and init_inputs:
            model = model_cls(*init_inputs)
        else:
            model = model_cls()

        return self._make_device_wrapper(model)

    def get_init_inputs(self, task_id: str) -> List[Any]:
        """返回 task 定义的 Model 构造参数。"""
        torch.manual_seed(self._random_seed)

        module = self._load_module(task_id)
        get_init_inputs_func = getattr(module, 'get_init_inputs', None)
        if get_init_inputs_func is None:
            return []

        init_inputs = get_init_inputs_func()
        if isinstance(init_inputs, tuple):
            return list(init_inputs)
        if isinstance(init_inputs, list):
            return init_inputs
        return []

    def get_input_function(self, task_id: str) -> Optional[Callable]:
        """返回 get_inputs wrapper，忽略所有参数

        StanfordBench 的 get_inputs() 无参数。
        Evaluator 会传入 attrs（如 skip2_exist），需要 wrapper 忽略这些参数。
        """
        module = self._load_module(task_id)
        get_inputs_func = getattr(module, 'get_inputs', None)
        if get_inputs_func is None:
            return None

        # 返回 wrapper，忽略所有 kwargs
        def get_inputs_wrapper(**kwargs):
            return get_inputs_func()

        return get_inputs_wrapper

    def get_operator_dir(self, task_id: str) -> Path:
        py_path = self._task_id_to_py(task_id)
        return py_path.parent if py_path else Path()

    def _load_module(self, task_id: str):
        """加载 Python 模块（带缓存）"""
        if task_id in self._module_cache:
            return self._module_cache[task_id]

        py_path = self._task_id_to_py(task_id)
        if not py_path or not py_path.exists():
            raise ImportError(f"文件不存在: {task_id}")

        spec = importlib.util.spec_from_file_location(
            f"stanford.{task_id.replace('/', '.')}",
            py_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        self._module_cache[task_id] = module
        return module

    def _task_id_to_py(self, task_id: str) -> Optional[Path]:
        """将 task_id 转换为 py_path"""
        if '/' in task_id:
            level, name = task_id.split('/', 1)
            # 检查 bench_root 本身是否是 level 目录
            m_root = re.match(r'level(\d+)', self.bench_root.name)
            if m_root and self.bench_root.name == level:
                # bench_root 本身就是目标 level 目录
                py_path = self.bench_root / f"{name}.py"
                if py_path.exists():
                    return py_path
            # 否则在 bench_root 下查找 level 子目录
            for level_dir in self.bench_root.rglob(level):
                py_path = level_dir / f"{name}.py"
                if py_path.exists():
                    return py_path
        else:
            for py_path in self.bench_root.rglob(f"{task_id}.py"):
                if py_path.exists():
                    return py_path
        return None

    @staticmethod
    def _make_device_wrapper(model) -> Callable:
        """Device wrapper：根据输入 tensor 自动切换 model device

        StanfordBench 特有需求：
        1. Model 有权重（如 Conv 的 kernel），需要跟随输入 tensor 的 device
        2. Model.forward 参数可能包含非 Tensor 类型（如 float），需要注入包含 Tensor 的类型注解
           供 evaluator 按位置从 input_tensors 取值

        注入策略：所有参数都注入 torch.Tensor 类型注解（evaluator 只检查字符串匹配）
        """
        from inspect import signature, Parameter

        # 获取原始签名并为所有参数注入 torch.Tensor 类型注解
        orig_sig = signature(model.forward)
        new_params = []
        for param_name, param in orig_sig.parameters.items():
            # 所有参数都注入 torch.Tensor，让 evaluator 按位置从 input_tensors 取值
            new_param = Parameter(
                param_name,
                param.kind,
                default=param.default,
                annotation=torch.Tensor
            )
            new_params.append(new_param)

        # 创建新的签名
        injected_sig = orig_sig.replace(parameters=new_params)

        class DeviceWrapper:
            def __init__(self, model):
                self.model = model
                self._device = None
                # 使用注入了类型注解的签名
                self.__signature__ = injected_sig

            def to(self, device):
                """显式切换 device"""
                self.model.to(device)
                self._device = device

            def __call__(self, *args, **kwargs):
                # 从输入找目标 device
                target = None
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        target = arg.device
                        break
                if target is None:
                    for v in kwargs.values():
                        if isinstance(v, torch.Tensor):
                            target = v.device
                            break

                # 切换 model 到同一 device
                if target and self._device != target:
                    self.model.to(target)
                    self._device = target

                return self.model.forward(*args, **kwargs)

        return DeviceWrapper(model)
