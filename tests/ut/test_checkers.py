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
from kernel_eval.base.result import FAILURE_TYPE_COMPILE_RUNTIME_ERROR


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

    def test_clear_registry(self, monkeypatch):
        """测试清空注册表"""
        from kernel_eval.registry import checker_registry
        # Keep the registered checkers available to subsequent evaluator tests.
        monkeypatch.setattr(checker_registry, "_CHECKER_REGISTRY", CheckerRegistry.get_all())
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

    def test_non_contiguous_output_is_structural_failure(self):
        """非连续输出判失败, 且归类为结构性失败而非精度不达标(issue #146)

        数值完全正确(对称矩阵转置), 只有 stride 变了。
        """
        checker = RelativeErrorChecker()
        golden = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64)
        ai = golden.to(torch.float32).t()
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert not result.is_passed()
        assert result.get_metadata()['failure_type'] == FAILURE_TYPE_COMPILE_RUNTIME_ERROR
        assert "内存布局非连续" in result.get_output_results()[0].get_error_msg()


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

    def test_non_contiguous_output_fail(self):
        """非连续输出判失败(issue #146); allclose 路径与相对误差路径同一标准"""
        checker = AllCloseChecker()
        golden = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float32)
        ai = golden.clone().t()
        assert torch.allclose(ai, golden)  # allclose 本身是过的

        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert not result.is_passed()
        assert result.get_metadata()['failure_type'] == FAILURE_TYPE_COMPILE_RUNTIME_ERROR
        assert "内存布局非连续" in result.get_output_results()[0].get_error_msg()

    def test_layout_failure_surfaces_in_standard_summary(self):
        """布局失败必须出现在**摘要**路径上, 不只是存进 error_msg 字段。

        Why: evaluator 用 format_all_outputs() 拼 case 错误串, 只断言 error_msg
        字段的话, 摘要分支把它吞掉也照样绿 -- 提交者看到的仍是没有 stride 的空话。
        且这里绝不能说 "allclose failed": 该张量数值与 golden 完全一致, allclose
        本身是过的, 根本没跑到。
        """
        checker = AllCloseChecker()
        golden = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float32)
        ai = golden.clone().t()
        assert torch.allclose(ai, golden)

        summary = checker.check(ai, golden, dtype="float32", threshold=0.001).format_all_outputs()
        assert "内存布局非连续" in summary
        assert "stride=(1, 2)" in summary
        assert "allclose failed" not in summary

    def test_shape_mismatch_surfaces_in_standard_summary(self):
        """形状不匹配同样要出现在摘要里(该分支早于 issue #146 就被摘要吞掉了)"""
        checker = AllCloseChecker()
        ai = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        golden = torch.tensor([1.0, 2.0], dtype=torch.float32)

        summary = checker.check(ai, golden, dtype="float32", threshold=0.001).format_all_outputs()
        assert "形状不匹配" in summary
        assert "allclose failed" not in summary

    def test_shape_mismatch_takes_precedence_over_layout(self):
        """形状与布局同时不对时报形状; 与相对误差路径的优先级保持一致"""
        golden = torch.zeros(2, 3, dtype=torch.float32)
        ai = torch.arange(12, dtype=torch.float32).reshape(4, 3).t()  # 3x4 且非连续
        assert ai.shape != golden.shape and not ai.is_contiguous()

        allclose_msg = AllCloseChecker().check(
            ai, golden, dtype="float32", threshold=0.001
        ).get_output_results()[0].get_error_msg()
        relative_msg = RelativeErrorChecker().check(
            [ai], [golden], "float32", 0.001
        ).get_output_results()[0].get_error_msg()

        for msg in (allclose_msg, relative_msg):
            assert "形状不匹配" in msg
            assert "内存布局非连续" not in msg

    def test_contiguous_output_still_passes(self):
        """连续输出不受影响"""
        checker = AllCloseChecker()
        golden = torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float32)
        ai = golden.clone().t().contiguous()
        result = checker.check(ai, golden, dtype="float32", threshold=0.001)
        assert result.is_passed()


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
