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
目录解析模块

职责：
1. 解析 --task-dir 参数，确定 bench_root 和筛选路径
2. 检测目录是否为算子目录
3. 统一处理绝对路径和相对路径

统一 cli.py 的 _resolve_dir_arg 和 run_simple.py 的 resolve_bench_root
"""

from pathlib import Path
from typing import Optional, Tuple


def resolve_task_dir(
    dir_arg: Optional[str],
    project_root: Path,
    check_operator_dir: bool = True,
) -> Tuple[str, Optional[str]]:
    """解析 --task-dir 参数，确定 bench_root 和筛选路径

    Args:
        dir_arg: 用户指定的目录路径（None 则使用默认 tasks）
        project_root: 项目根目录
        check_operator_dir: 是否检查算子目录特征（proto.yaml, cases.yaml, golden.py）

    Returns:
        (bench_root, filter_prefix)
        bench_root: bench 目录的绝对路径
        filter_prefix: 用于筛选算子的路径前缀，如 "level1" 或 "level2/scatter"
                       None 表示不筛选或无有效前缀

    Raises:
        ValueError: 目录不存在

    Examples:
        >>> resolve_task_dir(None, project_root)
        ("/path/to/tasks", None)

        >>> resolve_task_dir("tasks/level1", project_root)
        ("/path/to/tasks", "level1")

        >>> resolve_task_dir("tasks/level2/scatter", project_root)
        ("/path/to/tasks", "level2/scatter")

        >>> resolve_task_dir("/abs/path/tasks/level2/scatter", project_root)
        ("/abs/path/tasks", "level2/scatter")

        >>> resolve_task_dir("bench_lab/pypto_cann_bench/exp", project_root)
        ("/path/to/bench_lab/pypto_cann_bench", "exp")
    """
    # 默认值
    if dir_arg is None:
        return str(project_root / "tasks"), None

    # 解析路径
    dir_path = Path(dir_arg)

    # 处理相对路径：优先尝试相对于 project_root
    if not dir_path.is_absolute():
        relative_path = project_root / dir_arg
        if relative_path.exists():
            dir_path = relative_path
        elif dir_path.exists():
            # 保持原相对路径（当前工作目录下存在）
            pass
        else:
            raise ValueError(f"目录不存在: {dir_arg}")
    elif not dir_path.exists():
        raise ValueError(f"目录不存在: {dir_arg}")

    dir_path = dir_path.resolve()
    project_root = project_root.resolve()

    # 保留参数语义：调用方可以显式跳过算子目录特征检查。
    if check_operator_dir:
        is_operator_directory(dir_path)

    bench_root = find_bench_root(dir_path, project_root)

    try:
        filter_prefix = str(dir_path.relative_to(bench_root))
    except ValueError:
        filter_prefix = None

    # "." 表示 dir_path 即 bench_root 本身
    if filter_prefix == ".":
        filter_prefix = None

    return str(bench_root), filter_prefix


def is_operator_directory(dir_path: Path) -> bool:
    """检查目录是否为算子目录

    算子目录特征：包含 proto.yaml, cases.yaml, golden.py

    Args:
        dir_path: 目录路径

    Returns:
        是否为算子目录
    """
    from ..base.loaders import OperatorDirMixin
    return all((dir_path / f).exists() for f in OperatorDirMixin.REQUIRED_FILES)


def find_bench_root(dir_path: Path, project_root: Path) -> Path:
    """向上查找 bench_root（tasks 或 bench_lab/<suite> 目录）

    Args:
        dir_path: 起始目录
        project_root: 项目根目录（查找边界）

    Returns:
        bench_root 路径，若未找到则返回 dir_path
    """
    dir_path = dir_path.resolve()
    project_root = project_root.resolve()

    tasks_root = _find_named_ancestor(dir_path, project_root, "tasks")
    if tasks_root is not None:
        return tasks_root

    bench_lab = project_root / "bench_lab"
    try:
        rel_to_bench_lab = dir_path.relative_to(bench_lab)
    except ValueError:
        return dir_path

    if not rel_to_bench_lab.parts:
        return bench_lab
    return bench_lab / rel_to_bench_lab.parts[0]


def _find_named_ancestor(dir_path: Path, project_root: Path, name: str) -> Optional[Path]:
    current = dir_path
    while current != project_root:
        if current.name == name:
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    if project_root.name == name:
        return project_root
    return None
