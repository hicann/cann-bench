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
Golden函数动态导入器

职责：
1. 根据level和operator定位golden模块
2. 动态导入golden函数
"""

import importlib
import importlib.util
import re
import yaml
from pathlib import Path
from typing import Callable, Optional, Dict


def _camel_to_snake(name: str) -> str:
    """将 PascalCase 名称转换为 snake_case，用于匹配目录名"""
    # 处理数字+大写的组合: 3D -> _3_D, 2D -> _2_D
    s0 = re.sub(r'([0-9])([A-Z])', r'\1_\2', name)
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', s0)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


class GoldenLoader:
    """Golden函数动态导入器"""

    def __init__(self, bench_root: str = None):
        if bench_root:
            self.bench_root = Path(bench_root)
        else:
            # 默认使用项目根目录下的kernel_bench
            project_root = Path(__file__).parent.parent.parent.parent
            self.bench_root = project_root / "kernel_bench"
        self._func_cache: Dict[str, str] = {}
        self._dir_cache: Dict[int, Dict[str, str]] = {}

    def _get_dir_name(self, level: int, operator: str) -> str:
        """将 PascalCase 算子名解析为实际目录名，带缓存"""
        if level in self._dir_cache and operator in self._dir_cache[level]:
            return self._dir_cache[level][operator]

        # 先用 CamelCase→snake_case 猜测
        guessed = _camel_to_snake(operator)
        guessed_path = self.bench_root / f"level{level}" / guessed
        if guessed_path.exists():
            self._dir_cache.setdefault(level, {})[operator] = guessed
            return guessed

        # 猜测失败时扫描实际目录，按 golden.py 中的函数名匹配
        level_dir = self.bench_root / f"level{level}"
        if level_dir.is_dir():
            for entry in level_dir.iterdir():
                if entry.is_dir() and not entry.name.startswith('.'):
                    golden = entry / "golden.py"
                    if golden.exists():
                        func_name = operator.lower()
                        snake_name = _camel_to_snake(operator)
                        try:
                            content = golden.read_text(encoding='utf-8')
                            if f"def {func_name}(" in content or f"def {snake_name}(" in content:
                                self._dir_cache.setdefault(level, {})[operator] = entry.name
                                return entry.name
                        except Exception:
                            pass

        # 最终 fallback
        self._dir_cache.setdefault(level, {})[operator] = guessed
        return guessed

    def get_golden_function(self, level: int, operator: str) -> Callable:
        """获取golden函数"""
        dir_name = self._get_dir_name(level, operator)
        module_path = self.bench_root / f"level{level}" / dir_name / "golden.py"
        if not module_path.exists():
            raise ImportError(f"Golden模块不存在: {module_path}")

        module_name = f"kernel_bench.level{level}.{dir_name}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        func_name = self._get_function_name(level, operator)
        if not hasattr(module, func_name):
            func_name = operator.lower()
            if not hasattr(module, func_name):
                raise AttributeError(f"模块 {module_name} 中找不到函数: {func_name}")

        return getattr(module, func_name)

    def _get_function_name(self, level: int, operator: str) -> str:
        """从proto.yaml获取函数名"""
        cache_key = f"L{level}_{operator}"
        if cache_key in self._func_cache:
            return self._func_cache[cache_key]

        proto_path = self.bench_root / f"level{level}" / self._get_dir_name(level, operator) / "proto.yaml"
        if proto_path.exists():
            try:
                with open(proto_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if data and 'operator' in data:
                    schema = data['operator'].get('schema', '')
                    match = re.match(r'^(\w+)\s*\(', schema.strip())
                    if match:
                        self._func_cache[cache_key] = match.group(1)
                        return match.group(1)
            except Exception:
                pass

        return operator.lower()

    def get_operator_dir(self, level: int, operator: str) -> Path:
        """获取算子目录路径"""
        dir_name = self._get_dir_name(level, operator)
        return self.bench_root / f"level{level}" / dir_name

    def get_input_function(self, level: int, operator: str) -> Optional[Callable]:
        """获取 get_input 函数（可选）

        检查 golden.py 是否实现了 get_input() 函数。
        如果存在则返回该函数，否则返回 None。

        Args:
            level: 难度级别
            operator: 算子名称

        Returns:
            get_input 函数或 None
        """
        dir_name = self._get_dir_name(level, operator)
        module_path = self.bench_root / f"level{level}" / dir_name / "golden.py"
        if not module_path.exists():
            return None

        module_name = f"kernel_bench.level{level}.{dir_name}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'get_input'):
            return getattr(module, 'get_input')

        return None