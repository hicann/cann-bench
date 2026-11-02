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
验证 P1-6: BaseRegistry 泛型基类 — 5 套注册表共享 register/get/list/clear

目标：BenchRegistry / ScoringSchemeRegistry / GoldenLoaderRegistry 继承 BaseRegistry，
     LoaderRegistry（2 dict）和 CheckerRegistry（装饰器模式）保持独立。
"""

import pytest

from kernel_eval.registry.base import BaseRegistry


class _TestRegistry(BaseRegistry[str]):
    """测试用注册表子类"""
    _items: dict = {}  # 子类提供自己的存储


class TestBaseRegistry:
    """BaseRegistry 基类功能测试"""

    def setup_method(self):
        _TestRegistry.clear()

    def test_register_and_get(self):
        _TestRegistry.register("a", "value_a")
        assert _TestRegistry.get("a") == "value_a"

    def test_get_nonexistent_returns_none(self):
        assert _TestRegistry.get("nonexistent") is None

    def test_register_duplicate_raises(self):
        _TestRegistry.register("a", "value_a")
        with pytest.raises(ValueError, match="已注册"):
            _TestRegistry.register("a", "value_b")

    def test_list_all(self):
        _TestRegistry.register("a", "va")
        _TestRegistry.register("b", "vb")
        assert sorted(_TestRegistry.list_all()) == ["a", "b"]

    def test_list_all_empty(self):
        assert _TestRegistry.list_all() == []

    def test_is_registered(self):
        assert not _TestRegistry.is_registered("a")
        _TestRegistry.register("a", "va")
        assert _TestRegistry.is_registered("a")

    def test_clear(self):
        _TestRegistry.register("a", "va")
        _TestRegistry.register("b", "vb")
        _TestRegistry.clear()
        assert _TestRegistry.list_all() == []
        assert _TestRegistry.get("a") is None

    def test_subclass_has_independent_storage(self):
        """不同子类的 _items 互相独立"""
        class _Other(BaseRegistry[int]):
            _items: dict = {}

        _TestRegistry.register("x", "vx")
        _Other.register("y", 42)

        assert _TestRegistry.get("x") == "vx"
        assert _Other.get("y") == 42
        assert _TestRegistry.get("y") is None
        assert _Other.get("x") is None
        _Other.clear()


class TestExistingRegistriesInheritBase:
    """现有 registry 应继承 BaseRegistry 并获得一致接口"""

    def test_bench_registry_inherits_base(self):
        from kernel_eval.registry.bench_registry import BenchRegistry
        assert issubclass(BenchRegistry, BaseRegistry)
        # 使用 _items 而非 _configs
        assert hasattr(BenchRegistry, '_items')
        assert not hasattr(BenchRegistry, '_configs')

    def test_bench_registry_base_methods(self, monkeypatch):
        from kernel_eval.registry.bench_registry import BenchRegistry, BenchConfig
        # Exercise clear() without removing plugins needed by later test modules.
        monkeypatch.setattr(BenchRegistry, "_items", {})
        BenchRegistry.clear()
        cfg = BenchConfig(name="__test", description="test")
        BenchRegistry.register("__test", cfg)
        assert BenchRegistry.is_registered("__test")
        assert BenchRegistry.get("__test") is cfg
        assert "__test" in BenchRegistry.list_all()
        BenchRegistry.clear()

    def test_scoring_scheme_registry_inherits_base(self):
        from kernel_eval.registry.scoring_registry import ScoringSchemeRegistry
        assert issubclass(ScoringSchemeRegistry, BaseRegistry)
        assert hasattr(ScoringSchemeRegistry, '_items')
        assert not hasattr(ScoringSchemeRegistry, '_schemes')

    def test_scoring_scheme_registry_base_methods(self, monkeypatch):
        from kernel_eval.registry.scoring_registry import ScoringSchemeRegistry
        monkeypatch.setattr(ScoringSchemeRegistry, "_items", {})
        ScoringSchemeRegistry.clear()
        assert not ScoringSchemeRegistry.is_registered("__test_nonexistent__")
        ScoringSchemeRegistry.register("__test", None)
        assert ScoringSchemeRegistry.is_registered("__test")
        ScoringSchemeRegistry.clear()

    def test_golden_loader_registry_inherits_base(self):
        from kernel_eval.registry.golden_registry import GoldenLoaderRegistry
        assert issubclass(GoldenLoaderRegistry, BaseRegistry)
        assert hasattr(GoldenLoaderRegistry, '_items')
        assert not hasattr(GoldenLoaderRegistry, '_loaders')

    def test_golden_loader_registry_base_methods(self, monkeypatch):
        from kernel_eval.registry.golden_registry import GoldenLoaderRegistry
        from kernel_eval.base.loaders import GoldenLoaderBase
        monkeypatch.setattr(GoldenLoaderRegistry, "_items", {})
        GoldenLoaderRegistry.clear()
        assert not GoldenLoaderRegistry.is_registered("__test")
        GoldenLoaderRegistry.register("__test", GoldenLoaderBase)
        assert GoldenLoaderRegistry.is_registered("__test")
        GoldenLoaderRegistry.clear()


class TestGetBenchComponents:
    def test_reuses_task_loader_for_operator_matcher(self, monkeypatch):
        from types import SimpleNamespace

        from kernel_eval.registry import bench_registry

        config = bench_registry.BenchConfig(
            name="test",
            task_loader="test-task-loader",
            case_loader="test-case-loader",
            scoring_scheme="test-scoring-scheme",
            precision_thresholds={"float32": 1e-5},
        )
        monkeypatch.setattr(bench_registry, "get_bench_config", lambda _name: config)

        task_loaders = []

        def make_task_loader(name, **kwargs):
            loader = SimpleNamespace(name=name, kwargs=kwargs)
            task_loaders.append(loader)
            return loader

        monkeypatch.setattr(bench_registry, "get_task_loader", make_task_loader)
        monkeypatch.setattr(bench_registry, "get_case_loader", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(bench_registry, "get_golden_loader", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(
            bench_registry,
            "get_operator_matcher",
            lambda _name, operator_loader: SimpleNamespace(operator_loader=operator_loader),
        )
        monkeypatch.setattr(bench_registry, "get_scoring_scheme", lambda *_args, **_kwargs: object())
        monkeypatch.setattr(bench_registry, "get_correctness_checker", lambda *_args, **_kwargs: object())

        components = bench_registry.get_bench_components("test", "/tmp/tasks with space")

        assert len(task_loaders) == 1
        assert task_loaders[0].kwargs == {"tasks_root": "/tmp/tasks with space"}
        assert components["task_loader"] is task_loaders[0]
        assert components["operator_matcher"].operator_loader is components["task_loader"]
