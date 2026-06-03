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
path_resolver 模块单元测试

测试 resolve_task_dir 函数的各种场景：
- None 参数（默认值）
- 相对路径
- 绝对路径
- 算子目录
- 非算子目录
- 不存在的目录
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from kernel_eval.utils.path_resolver import (
    resolve_task_dir,
    is_operator_directory,
    find_bench_root,
)


class TestResolveTaskDir:
    """resolve_task_dir 函数测试"""

    def test_none_argument_returns_default(self):
        """测试 None 参数返回默认 tasks"""
        project_root = Path("/project")
        bench_root, filter_prefix = resolve_task_dir(None, project_root)

        assert bench_root == str(project_root / "tasks")
        assert filter_prefix is None

    def test_relative_path_tasks_root(self, tmp_path):
        """测试相对路径指向 tasks 根目录"""
        # 创建真实的 tasks 目录
        tasks = tmp_path / "tasks"
        tasks.mkdir()

        bench_root, filter_prefix = resolve_task_dir("tasks", tmp_path)

        assert bench_root == str(tasks)
        assert filter_prefix is None

    def test_relative_path_level_subdir(self, tmp_path):
        """测试相对路径指向 level 子目录"""
        # 创建真实目录结构
        tasks = tmp_path / "tasks"
        level1 = tasks / "level1"
        level1.mkdir(parents=True)

        bench_root, filter_prefix = resolve_task_dir("tasks/level1", tmp_path)

        assert bench_root == str(tasks)
        assert filter_prefix == "level1"

    def test_absolute_path_operator_dir(self, tmp_path):
        """测试绝对路径指向算子目录"""
        # 创建模拟目录结构
        tasks = tmp_path / "tasks"
        level2 = tasks / "level2"
        scatter = level2 / "scatter"
        scatter.mkdir(parents=True)

        # 创建算子目录特征文件
        for f in ['proto.yaml', 'cases.yaml', 'golden.py']:
            (scatter / f).touch()

        bench_root, filter_prefix = resolve_task_dir(str(scatter), tmp_path)

        assert bench_root == str(tasks)
        assert filter_prefix == "level2/scatter"

    def test_absolute_path_non_operator_dir(self, tmp_path):
        """测试绝对路径指向非算子目录"""
        tasks = tmp_path / "tasks"
        level2 = tasks / "level2"
        level2.mkdir(parents=True)

        bench_root, filter_prefix = resolve_task_dir(str(level2), tmp_path)

        assert bench_root == str(tasks)
        assert filter_prefix == "level2"

    def test_nonexistent_path_raises_error(self):
        """测试不存在的路径抛出 ValueError"""
        project_root = Path("/project")

        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(ValueError, match="目录不存在"):
                resolve_task_dir("nonexistent", project_root)

    def test_path_without_tasks(self, tmp_path):
        """测试路径中不包含 tasks 目录"""
        # 创建一个不在 tasks 下的目录
        other_dir = tmp_path / "other_dir"
        other_dir.mkdir()

        bench_root, filter_prefix = resolve_task_dir(str(other_dir), tmp_path)

        # 未找到 tasks，使用原目录作为 bench_root
        assert bench_root == str(other_dir)
        assert filter_prefix is None

    def test_filter_prefix_dot_returns_none(self, tmp_path):
        """测试 filter_prefix 为 "." 时返回 None"""
        tasks = tmp_path / "tasks"
        tasks.mkdir()

        bench_root, filter_prefix = resolve_task_dir(str(tasks), tmp_path)

        assert bench_root == str(tasks)
        assert filter_prefix is None

    def test_check_operator_dir_false(self, tmp_path):
        """测试 check_operator_dir=False 时跳过算子目录检查"""
        tasks = tmp_path / "tasks"
        level2 = tasks / "level2"
        scatter = level2 / "scatter"
        scatter.mkdir(parents=True)

        # 创建算子目录特征文件
        for f in ['proto.yaml', 'cases.yaml', 'golden.py']:
            (scatter / f).touch()

        # check_operator_dir=False，仍然返回正确的路径
        bench_root, filter_prefix = resolve_task_dir(
            str(scatter), tmp_path, check_operator_dir=False
        )

        assert bench_root == str(tasks)
        assert filter_prefix == "level2/scatter"

    def test_bench_lab_suite_operator_dir(self, tmp_path):
        """测试 bench_lab/<suite>/<op> 解析为 suite 根目录"""
        pypto_root = tmp_path / "bench_lab" / "pypto_cann_bench"
        exp_dir = pypto_root / "exp"
        exp_dir.mkdir(parents=True)
        for f in ['proto.yaml', 'cases.yaml', 'golden.py']:
            (exp_dir / f).touch()

        bench_root, filter_prefix = resolve_task_dir(
            "bench_lab/pypto_cann_bench/exp", tmp_path
        )

        assert bench_root == str(pypto_root)
        assert filter_prefix == "exp"

    def test_bench_lab_suite_root(self, tmp_path):
        """测试 bench_lab/<suite> 本身解析为 bench root"""
        pypto_root = tmp_path / "bench_lab" / "pypto_cann_bench"
        pypto_root.mkdir(parents=True)

        bench_root, filter_prefix = resolve_task_dir(
            "bench_lab/pypto_cann_bench", tmp_path
        )

        assert bench_root == str(pypto_root)
        assert filter_prefix is None


class TestIsOperatorDirectory:
    """is_operator_directory 函数测试"""

    def test_operator_dir_has_all_files(self, tmp_path):
        """测试包含所有特征文件的目录"""
        for f in ['proto.yaml', 'cases.yaml', 'golden.py']:
            (tmp_path / f).touch()

        assert is_operator_directory(tmp_path) is True

    def test_non_operator_dir_missing_files(self, tmp_path):
        """测试缺少特征文件的目录"""
        (tmp_path / 'proto.yaml').touch()
        # 缺少 cases.yaml 和 golden.py

        assert is_operator_directory(tmp_path) is False

    def test_empty_dir_returns_false(self, tmp_path):
        """测试空目录返回 False"""
        assert is_operator_directory(tmp_path) is False


class TestFindBenchRoot:
    """find_bench_root 函数测试"""

    def test_find_tasks(self, tmp_path):
        """测试向上查找 tasks"""
        tasks = tmp_path / "tasks"
        level2 = tasks / "level2"
        scatter = level2 / "scatter"
        scatter.mkdir(parents=True)

        result = find_bench_root(scatter, tmp_path)
        assert result == tasks

    def test_no_tasks_returns_original(self, tmp_path):
        """测试未找到 tasks 时返回原目录"""
        other_dir = tmp_path / "other_dir"
        other_dir.mkdir()

        result = find_bench_root(other_dir, tmp_path)
        assert result == other_dir

    def test_direct_tasks(self, tmp_path):
        """测试直接指向 tasks"""
        tasks = tmp_path / "tasks"
        tasks.mkdir()

        result = find_bench_root(tasks, tmp_path)
        assert result == tasks

    def test_find_bench_lab_suite(self, tmp_path):
        """测试 bench_lab 下按 suite 查找 bench root"""
        roi_pooling = tmp_path / "bench_lab" / "kernel_bench" / "level3" / "roi_pooling"
        roi_pooling.mkdir(parents=True)

        result = find_bench_root(roi_pooling, tmp_path)
        assert result == tmp_path / "bench_lab" / "kernel_bench"


class TestResolveTaskDirIntegration:
    """集成测试：使用真实目录结构"""

    def test_full_hierarchy(self, tmp_path):
        """测试完整层级结构"""
        # 创建完整目录结构
        tasks = tmp_path / "tasks"
        level1 = tasks / "level1"
        level2 = tasks / "level2"
        exp_dir = level1 / "exp"
        scatter_dir = level2 / "scatter"

        exp_dir.mkdir(parents=True)
        scatter_dir.mkdir(parents=True)

        # 创建算子目录特征文件
        for f in ['proto.yaml', 'cases.yaml', 'golden.py']:
            (exp_dir / f).touch()
            (scatter_dir / f).touch()

        # 测试 exp 算子目录
        bench_root, filter_prefix = resolve_task_dir(str(exp_dir), tmp_path)
        assert bench_root == str(tasks)
        assert filter_prefix == "level1/exp"

        # 测试 scatter 算子目录
        bench_root, filter_prefix = resolve_task_dir(str(scatter_dir), tmp_path)
        assert bench_root == str(tasks)
        assert filter_prefix == "level2/scatter"

        # 测试 level1 目录（非算子目录）
        bench_root, filter_prefix = resolve_task_dir(str(level1), tmp_path)
        assert bench_root == str(tasks)
        assert filter_prefix == "level1"

        # 测试 tasks 根目录
        bench_root, filter_prefix = resolve_task_dir(str(tasks), tmp_path)
        assert bench_root == str(tasks)
        assert filter_prefix is None
