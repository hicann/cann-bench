#!/usr/bin/python3
# coding=utf-8

"""
精度判断器注册机制测试
"""

import pytest
import torch

# 导入 benches 模块，触发 CANN / Stanford 特化组件注册
import kernel_eval.benches

from kernel_eval.eval import (
    get_correctness_checker,
    list_correctness_checkers,
    AccuracyResult,
)
from kernel_eval.checkers.relative_error_checker import (
    RelativeErrorChecker,
    RelativeErrorOutputResult,
)
from kernel_eval.checkers.allclose_checker import (
    AllCloseChecker,
    AllCloseOutputResult,
)
from kernel_eval.registry.checker_registry import (
    register_correctness_checker,
    is_checker_registered,
    clear_checker_registry,
    CheckerRegistry,
)
from kernel_eval.base.checker import CorrectnessChecker


class TestRegistry:
    """注册机制测试"""

    def test_default_checkers_registered(self):
        """测试默认判断器已注册"""
        assert is_checker_registered("relative_error")
        assert is_checker_registered("allclose")

    def test_list_checkers(self):
        """测试列出所有判断器"""
        names = list_correctness_checkers()
        assert "relative_error" in names
        assert "allclose" in names

    def test_get_checker(self):
        """测试获取判断器"""
        checker = get_correctness_checker("relative_error")
        assert checker is not None
        assert checker.get_name() == "relative_error"

        checker2 = get_correctness_checker("allclose")
        assert checker2 is not None
        assert checker2.get_name() == "allclose"

    def test_get_nonexistent_checker(self):
        """测试获取不存在的判断器"""
        checker = get_correctness_checker("nonexistent")
        assert checker is None

    def test_register_duplicate_raises(self):
        """测试重复注册抛出异常"""
        with pytest.raises(ValueError, match="already registered"):
            @register_correctness_checker("relative_error")
            class DummyChecker(CorrectnessChecker):
                def get_name(self):
                    return "dummy"
                def check(self, *args):
                    pass

    def test_manual_register_and_unregister(self):
        """测试手动注册和注销"""
        class CustomChecker(CorrectnessChecker):
            def get_name(self):
                return "custom_test"
            def check(self, *args):
                pass

        instance = CustomChecker()
        CheckerRegistry.register("custom_test", instance)
        assert is_checker_registered("custom_test")

        CheckerRegistry.unregister("custom_test")
        assert not is_checker_registered("custom_test")

    def test_clear_registry(self):
        """测试清空注册表"""
        # 记录原始注册数量
        original_count = len(list_correctness_checkers())
        assert original_count >= 2  # 至少有 relative_error 和 allclose

        # 清空
        clear_checker_registry()
        assert len(list_correctness_checkers()) == 0

        # 验证清空后确实为空
        assert not is_checker_registered("relative_error")
        assert not is_checker_registered("allclose")


class TestRelativeErrorChecker:
    """相对误差判断器测试"""

    def test_single_output_pass(self):
        """测试单输出通过"""
        checker = RelativeErrorChecker()
        ai = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert isinstance(result, AccuracyResult)
        assert result.is_passed()
        assert len(result.get_output_results()) == 1
        assert isinstance(result.get_output_results()[0], RelativeErrorOutputResult)

    def test_single_output_fail(self):
        """测试单输出失败"""
        checker = RelativeErrorChecker()
        ai = torch.tensor([1.0, 100.0, 3.0], dtype=torch.float32)
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert isinstance(result, AccuracyResult)
        assert not result.is_passed()

    def test_multi_output(self):
        """测试多输出"""
        checker = RelativeErrorChecker()
        ai = [torch.tensor([1.0, 2.0], dtype=torch.float32),
              torch.tensor([3.0, 4.0], dtype=torch.float32)]
        golden = [torch.tensor([1.0, 2.0], dtype=torch.float64),
                  torch.tensor([3.0, 4.0], dtype=torch.float64)]
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert result.is_passed()
        assert len(result.get_output_results()) == 2

    def test_ignore_indices(self):
        """测试忽略索引"""
        checker = RelativeErrorChecker()
        ai = [torch.tensor([1.0, 2.0], dtype=torch.float32),
              torch.tensor([100.0, 200.0], dtype=torch.float32)]  # 应该被忽略
        golden = [torch.tensor([1.0, 2.0], dtype=torch.float64),
                  torch.tensor([3.0, 4.0], dtype=torch.float64)]
        result = checker.check(ai, golden, dtype="float32", threshold=0.001,
                               ignore_indices=[1])
        assert result.is_passed()
        assert len(result.get_output_results()) == 2
        assert result.get_output_results()[1].get_error_msg() == "(跳过对比)"

    def test_output_count_mismatch(self):
        """测试输出数量不匹配"""
        checker = RelativeErrorChecker()
        ai = torch.tensor([1.0, 2.0], dtype=torch.float32)
        golden = [torch.tensor([1.0, 2.0], dtype=torch.float64),
                  torch.tensor([3.0, 4.0], dtype=torch.float64)]
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert not result.is_passed()
        assert "输出数量不匹配" in result.get_error_msg()

    def test_metadata_contains_mere_mare(self):
        """测试 metadata 包含 mere/mare"""
        checker = RelativeErrorChecker()
        ai = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        metadata = result.get_metadata()
        assert 'mere' in metadata
        assert 'mare' in metadata


class TestAllCloseChecker:
    """AllClose判断器测试"""

    def test_single_output_pass(self):
        """测试单输出通过"""
        checker = AllCloseChecker()
        ai = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert isinstance(result, AccuracyResult)
        assert result.is_passed()
        assert isinstance(result.get_output_results()[0], AllCloseOutputResult)

    def test_single_output_fail(self):
        """测试单输出失败"""
        checker = AllCloseChecker()
        ai = torch.tensor([1.0, 100.0, 3.0], dtype=torch.float32)
        golden = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert not result.is_passed()

    def test_multi_output(self):
        """测试多输出"""
        checker = AllCloseChecker()
        ai = [torch.tensor([1.0, 2.0], dtype=torch.float32),
              torch.tensor([3.0, 4.0], dtype=torch.float32)]
        golden = [torch.tensor([1.0, 2.0], dtype=torch.float32),
                  torch.tensor([3.0, 4.0], dtype=torch.float32)]
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert result.is_passed()
        assert len(result.get_output_results()) == 2

    def test_shape_mismatch(self):
        """测试形状不匹配"""
        checker = AllCloseChecker()
        ai = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        golden = torch.tensor([1.0, 2.0], dtype=torch.float32)
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert not result.is_passed()


class TestAccuracyResult:
    """AccuracyResult 测试"""

    def test_get_first_dtype(self):
        """测试获取第一个 dtype"""
        checker = RelativeErrorChecker()
        ai = torch.tensor([1.0, 2.0], dtype=torch.float16)
        golden = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = checker.check(ai, golden, dtype="float16", threshold=0.001)
        assert result.get_first_dtype() == "float16"

    def test_get_failed_dtype(self):
        """测试获取失败 dtype"""
        checker = RelativeErrorChecker()
        ai = [torch.tensor([1.0, 2.0], dtype=torch.float32),
              torch.tensor([100.0, 200.0], dtype=torch.float16)]  # 失败
        golden = [torch.tensor([1.0, 2.0], dtype=torch.float64),
                  torch.tensor([3.0, 4.0], dtype=torch.float64)]
        result = checker.check(ai, golden, dtype="float16", threshold=0.001)
        if not result.is_passed():
            assert result.get_failed_dtype() == "float16"

    def test_format_summary(self):
        """测试格式化摘要"""
        checker = RelativeErrorChecker()
        ai = torch.tensor([1.0, 2.0], dtype=torch.float32)
        golden = torch.tensor([1.0, 2.0], dtype=torch.float64)
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        summary = result.format_summary()
        assert "✅" in summary


class TestOutputResult:
    """OutputResult 测试"""

    def test_relative_error_output_result_to_dict(self):
        """测试 RelativeErrorOutputResult to_dict（指标走 metadata 扁平化）"""
        output = RelativeErrorOutputResult(
            index=0,
            passed=True,
            dtype="float32",
            metadata={'threshold': 0.001, 'mere': 0.0, 'mare': 0.0},
        )
        d = output.to_dict()
        assert d['index'] == 0
        assert d['passed'] == True
        assert d['mere'] == 0.0

    def test_allclose_output_result_to_dict(self):
        """测试 AllCloseOutputResult to_dict"""
        output = AllCloseOutputResult(
            index=0,
            passed=True,
            dtype="float32",
            metadata={'threshold': 0.001},
        )
        d = output.to_dict()
        assert d['index'] == 0
        assert d['passed'] == True
