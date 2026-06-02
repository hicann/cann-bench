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
GoldenLoader 单元测试

测试对象：kernel_eval.data.golden_loader.GoldenLoader
核心功能：
1. golden 模块动态导入（含缓存）
2. _get_function_name 从 proto.yaml 提取函数名
3. _get_operator_name 从 proto.yaml 提取算子名
4. proto.yaml 解析容错与 fallback
5. 异常路径的 warning 日志
"""

import logging
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from kernel_eval.benches.cann_loader import GoldenLoader


class TestGoldenLoaderLoadModule:
    """测试 _load_module 方法的异常容错"""

    def test_corrupt_golden_py_raises_on_load(self, caplog):
        """损坏的 golden.py 在 import 时会抛出异常"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "testop"
            level_dir.mkdir(parents=True)
            (level_dir / "proto.yaml").write_text(
                "operator:\n  name: TestOp\n  schema: test_op(Tensor a) -> Tensor\n",
                encoding="utf-8"
            )
            # 写入一个损坏的 golden.py（非 UTF-8 二进制内容）
            (level_dir / "golden.py").write_bytes(b'\xff\xfe\x00\x00\xff\xff')

            loader = GoldenLoader(bench_root=str(root))
            with pytest.raises((SyntaxError, ValueError)):
                loader._load_module("level1/testop")

    def test_corrupt_proto_yaml_logs_warning(self, caplog):
        """损坏的 proto.yaml 应记录 warning 并 fallback 到目录名"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "test_op"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def test_op(*args):\n    return args\n", encoding="utf-8")
            # 写入一个损坏的 proto.yaml（非法 YAML）
            (level_dir / "proto.yaml").write_text(":\n\t- broken: [unclosed\n", encoding="utf-8")

            with caplog.at_level(logging.WARNING, logger="kernel_eval.data.golden_loader"):
                loader = GoldenLoader(bench_root=str(root))
                result = loader._get_function_name("level1/test_op")

            # 应 fallback 到目录名小写（_get_operator_name 的 fallback）
            assert result == "test_op"
            assert len(caplog.records) >= 1
            assert any("proto.yaml" in r.message for r in caplog.records)

    def test_missing_operator_dir_raises_on_module_load(self):
        """不存在的算子目录在加载模块时应抛出 ImportError"""
        with tempfile.TemporaryDirectory() as td:
            loader = GoldenLoader(bench_root=str(td))
            with pytest.raises(ImportError):
                loader._load_module("level1/NonExistentOp")

    def test_normal_golden_py_loads_without_warning(self, caplog):
        """正常的 golden.py 应加载成功且不产生 warning"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "add"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def add(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text(
                "operator:\n  name: Add\n  schema: add(Tensor a, Tensor b) -> Tensor\n",
                encoding="utf-8"
            )

            with caplog.at_level(logging.WARNING, logger="kernel_eval.data.golden_loader"):
                loader = GoldenLoader(bench_root=str(root))
                module = loader._load_module("level1/add")

            assert hasattr(module, 'add')
            assert len(caplog.records) == 0


class TestGoldenLoaderGetFunctionName:
    """测试 _get_function_name 方法的异常容错"""

    def test_corrupt_proto_yaml_falls_back_to_dir_name(self, caplog):
        """_get_function_name 中 proto.yaml 异常应记录 warning 并 fallback 到目录名"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "exp"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def exp(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text("\tbad: yaml: [\n", encoding="utf-8")

            with caplog.at_level(logging.WARNING, logger="kernel_eval.data.golden_loader"):
                loader = GoldenLoader(bench_root=str(root))
                result = loader._get_function_name("level1/exp")

            # fallback 到 _get_operator_name 的小写结果
            assert result == "exp"
            assert len(caplog.records) >= 1

    def test_valid_proto_yaml_extracts_function_name(self):
        """正常 proto.yaml 应正确提取函数名"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "exp"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def exp_kernel(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text(
                "operator:\n  name: Exp\n  schema: exp_kernel(Tensor a) -> Tensor\n",
                encoding="utf-8"
            )

            loader = GoldenLoader(bench_root=str(root))
            result = loader._get_function_name("level1/exp")

            assert result == "exp_kernel"

    def test_get_function_name_uses_cache(self):
        """_get_function_name 应使用缓存"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level1" / "sqr"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def square(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text(
                "operator:\n  name: Sqr\n  schema: square(Tensor a) -> Tensor\n",
                encoding="utf-8"
            )

            loader = GoldenLoader(bench_root=str(root))
            result1 = loader._get_function_name("level1/sqr")
            assert result1 == "square"
            # 第二次调用应使用缓存
            assert "level1/sqr" in loader._func_cache
            result2 = loader._get_function_name("level1/sqr")
            assert result1 == result2


class TestGoldenLoaderGetByOperatorName:
    """测试 get_golden_by_operator_name 方法"""

    def test_find_by_operator_name(self):
        """按算子名称查找 golden 函数"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            level_dir = root / "level2" / "my_op"
            level_dir.mkdir(parents=True)
            (level_dir / "golden.py").write_text("def my_op(*args):\n    return args\n", encoding="utf-8")
            (level_dir / "proto.yaml").write_text(
                "operator:\n  name: MyOp\n  schema: my_op(Tensor a) -> Tensor\n",
                encoding="utf-8"
            )
            # 需要 cases.yaml 才能被扫描到
            (level_dir / "cases.yaml").write_text("cases: []\n", encoding="utf-8")

            loader = GoldenLoader(bench_root=str(root))
            func = loader.get_golden_by_operator_name("MyOp")
            assert callable(func)

    def test_not_found_raises_import_error(self):
        """不存在的算子应抛出 ImportError"""
        with tempfile.TemporaryDirectory() as td:
            loader = GoldenLoader(bench_root=str(td))
            with pytest.raises(ImportError):
                loader.get_golden_by_operator_name("NonExistent")
