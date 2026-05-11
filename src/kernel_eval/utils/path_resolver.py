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
        dir_arg: 用户指定的目录路径（None 则使用默认 kernel_bench）
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
        ("/path/to/kernel_bench", None)

        >>> resolve_task_dir("kernel_bench/level1", project_root)
        ("/path/to/kernel_bench", "level1")

        >>> resolve_task_dir("kernel_bench/level2/scatter", project_root)
        ("/path/to/kernel_bench", "level2/scatter")

        >>> resolve_task_dir("/abs/path/kernel_bench/level2/scatter", project_root)
        ("/abs/path/kernel_bench", "level2/scatter")
    """
    # 默认值
    if dir_arg is None:
        return str(project_root / "kernel_bench"), None

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

    # 检查是否为算子目录
    is_operator_dir = False
    if check_operator_dir:
        required_files = ['proto.yaml', 'cases.yaml', 'golden.py']
        is_operator_dir = all((dir_path / f).exists() for f in required_files)

    # 向上查找 bench_root（kernel_bench 目录）
    bench_root = dir_path
    while bench_root.name != 'kernel_bench' and bench_root != project_root:
        bench_root = bench_root.parent

    # 计算筛选前缀
    if bench_root == project_root:
        # 未找到 kernel_bench，使用原目录作为 bench_root
        bench_root = dir_path
        filter_prefix = None
    else:
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
    required_files = ['proto.yaml', 'cases.yaml', 'golden.py']
    return all((dir_path / f).exists() for f in required_files)


def find_bench_root(dir_path: Path, project_root: Path) -> Path:
    """向上查找 bench_root（kernel_bench 目录）

    Args:
        dir_path: 起始目录
        project_root: 项目根目录（查找边界）

    Returns:
        bench_root 路径，若未找到则返回 dir_path
    """
    bench_root = dir_path
    while bench_root.name != 'kernel_bench' and bench_root != project_root:
        bench_root = bench_root.parent

    if bench_root == project_root:
        return dir_path
    return bench_root