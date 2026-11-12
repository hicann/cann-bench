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
host 内存优化单元测试（evaluator 侧）

覆盖：
1. RSS 读取 / 阈值解析（KERNEL_EVAL_RSS_TRIM_MB 覆盖、<=0 禁用、非法回退）
2. _cleanup_memory 的 RSS 触发 / 不触发 / 降级路径
3. evaluate_case 精度对比后即时释放大对象的端到端回归
   （stub 掉 loader/runner 等协作者，验证结果正确且 run result 输出已释放）
4. 用 weakref 观测"提前释放"本身：进入 _cleanup_memory 时大对象必须已经无人引用。
   仅断言 run_result.outputs is None 是不够的——那在释放点后移到函数返回之后时同样成立。
"""

import os
import platform
import sys
import weakref
from types import SimpleNamespace

import pytest
import torch

import kernel_eval.eval.evaluator as evaluator_mod
from kernel_eval.base.result import FAILURE_TYPE_COMPILE_RUNTIME_ERROR
from kernel_eval.config import Config
from kernel_eval.eval.accuracy_eval import AccuracyEvaluator
from kernel_eval.eval.evaluator import (
    Evaluator,
    _CaseMemoryMonitor,
    _malloc_trim,
    _read_rss_mb,
    _rss_trim_threshold_mb,
)
from kernel_eval.eval.op_runner import OpRunResult

_RSS_ENV = "KERNEL_EVAL_RSS_TRIM_MB"
_DEFAULT_MB = float(48 * 1024)

# 两个"真调用"用例依赖 Linux/glibc：/proc/self/status 与 libc.so.6 的 malloc_trim。
# 被测函数在其它平台上按设计静默降级，所以这里 skip 而不是断言降级值——降级路径
# 已由同文件的 monkeypatch 用例覆盖。评测目标平台是 linux/aarch64 + Ascend NPU。
requires_proc = pytest.mark.skipif(
    not os.path.exists("/proc/self/status"), reason="needs procfs (Linux)"
)
requires_glibc = pytest.mark.skipif(
    sys.platform != "linux" or platform.libc_ver()[0] != "glibc",
    reason="needs glibc (malloc_trim)",
)


class TestReadRss:
    """_read_rss_mb：读取 /proc/self/status 的 VmRSS"""

    @requires_proc
    def test_real_proc_returns_positive_mb(self):
        rss = _read_rss_mb()
        assert rss is not None
        assert rss > 0

    def test_open_failure_returns_none(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise OSError("no /proc")

        monkeypatch.setattr("builtins.open", _boom)
        assert _read_rss_mb() is None

    def test_malformed_content_returns_none(self, monkeypatch):
        import io

        monkeypatch.setattr("builtins.open",
                            lambda *a, **k: io.StringIO("Name:\tproc\nBadLine\n"))
        assert _read_rss_mb() is None


class TestCaseMemoryMonitor:
    def test_tracks_baseline_and_peak(self, monkeypatch):
        samples = iter([100.0, 150.0, 125.0])
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: next(samples))

        with _CaseMemoryMonitor(True, interval_sec=10) as monitor:
            monitor._sample()

        assert monitor.baseline_mb == 100.0
        assert monitor.peak_mb == 150.0

    def test_unavailable_rss_is_preserved_as_none(self, monkeypatch):
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: None)

        with _CaseMemoryMonitor(True, interval_sec=10) as monitor:
            pass

        assert monitor.baseline_mb is None
        assert monitor.peak_mb is None

class TestRssTrimThreshold:
    """_rss_trim_threshold_mb：阈值解析与禁用语义"""

    def test_default_48_gib_when_unset(self, monkeypatch):
        monkeypatch.delenv(_RSS_ENV, raising=False)
        assert _rss_trim_threshold_mb() == _DEFAULT_MB

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(_RSS_ENV, "65536")
        assert _rss_trim_threshold_mb() == 65536.0

    @pytest.mark.parametrize("value", ["0", "-1", "-0.5"])
    def test_non_positive_disables(self, monkeypatch, value):
        monkeypatch.setenv(_RSS_ENV, value)
        assert _rss_trim_threshold_mb() is None

    @pytest.mark.parametrize("value", ["not-a-number", "nan", "NaN", "-nan"])
    def test_invalid_value_falls_back_to_default(self, monkeypatch, value):
        # nan 能被 float() 接受，但它与任何数的比较都为 False，不显式拦就会走成
        # "静默禁用"，与 docstring 承诺的"非法值回退默认"相反。
        monkeypatch.setenv(_RSS_ENV, value)
        assert _rss_trim_threshold_mb() == _DEFAULT_MB

    @pytest.mark.parametrize("value", ["inf", "1e12"])
    def test_huge_value_is_kept_as_disabled_in_practice(self, monkeypatch, value):
        # inf / 极大值是合法的"实际上永不触发"，不当作非法值处理
        monkeypatch.setenv(_RSS_ENV, value)
        assert _rss_trim_threshold_mb() == float(value)


class TestCleanupMemoryRssTrim:
    """_cleanup_memory：RSS 超阈值时追加 malloc_trim，失败静默降级"""

    @staticmethod
    def _call_cleanup():
        # _cleanup_memory 不读取任何实例状态，绕过重量级 __init__ 直接调用
        Evaluator._cleanup_memory(Evaluator.__new__(Evaluator))

    def test_trim_triggered_above_threshold(self, monkeypatch):
        monkeypatch.delenv(_RSS_ENV, raising=False)
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: 60 * 1024.0)
        trim_calls = []
        monkeypatch.setattr(evaluator_mod, "_malloc_trim",
                            lambda: trim_calls.append(1) or True)
        self._call_cleanup()
        assert trim_calls == [1]

    def test_trim_skipped_below_threshold(self, monkeypatch):
        monkeypatch.delenv(_RSS_ENV, raising=False)
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: 1024.0)
        trim_calls = []
        monkeypatch.setattr(evaluator_mod, "_malloc_trim",
                            lambda: trim_calls.append(1) or True)
        self._call_cleanup()
        assert trim_calls == []

    def test_trim_skipped_when_disabled_by_env(self, monkeypatch):
        monkeypatch.setenv(_RSS_ENV, "0")
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: 1024 * 1024.0)
        trim_calls = []
        monkeypatch.setattr(evaluator_mod, "_malloc_trim",
                            lambda: trim_calls.append(1) or True)
        self._call_cleanup()
        assert trim_calls == []

    def test_rss_read_failure_degrades_silently(self, monkeypatch):
        monkeypatch.delenv(_RSS_ENV, raising=False)
        monkeypatch.setattr(evaluator_mod, "_read_rss_mb", lambda: None)
        trim_calls = []
        monkeypatch.setattr(evaluator_mod, "_malloc_trim",
                            lambda: trim_calls.append(1) or True)
        self._call_cleanup()  # 不抛异常
        assert trim_calls == []

    @requires_glibc
    def test_malloc_trim_real_call_does_not_raise(self):
        # 真实调用不抛异常即可。返回值不能断言 True：malloc_trim 返回 0（当前无空闲
        # 堆页可归还）是完全正常的结果，取决于进程此刻的堆状态。
        assert isinstance(_malloc_trim(), bool)

    def test_malloc_trim_returns_false_when_nothing_released(self, monkeypatch):
        # rc=0 表示什么都没归还，对调用方而言与"没执行"等价，不应让日志谎报归还
        monkeypatch.setattr(evaluator_mod, "_libc",
                            lambda: SimpleNamespace(malloc_trim=lambda _pad: 0))
        assert _malloc_trim() is False

    def test_malloc_trim_returns_true_when_pages_released(self, monkeypatch):
        monkeypatch.setattr(evaluator_mod, "_libc",
                            lambda: SimpleNamespace(malloc_trim=lambda _pad: 1))
        assert _malloc_trim() is True

    def test_malloc_trim_without_glibc_degrades_silently(self, monkeypatch):
        monkeypatch.setattr(evaluator_mod, "_libc", lambda: None)
        assert _malloc_trim() is False

    def test_malloc_trim_call_failure_degrades_silently(self, monkeypatch):
        def _boom(_pad):
            raise OSError("call failed")

        monkeypatch.setattr(evaluator_mod, "_libc",
                            lambda: SimpleNamespace(malloc_trim=_boom))
        assert _malloc_trim() is False

    def test_libc_handle_is_resolved_once(self):
        # 解析 libc 是每次 _cleanup_memory 都可能走到的路径，必须缓存
        evaluator_mod._libc.cache_clear()
        first = evaluator_mod._libc()
        assert evaluator_mod._libc() is first
        assert evaluator_mod._libc.cache_info().hits >= 1


class _StubCase:
    rel_path = "level1/stub_op"
    operator = "StubOp"
    case_id = 1
    case_num = 1
    attrs = {}
    input_shapes = [[8]]
    dtypes = ["float32"]
    value_ranges = None
    baseline_perf_us = 100.0
    t_hw_us = 10.0

    def get_case_id_str(self):
        return "level1/stub_op_1"


class _StubGoldenLoader:
    @staticmethod
    def _golden(x):
        return x * 2

    def get_golden_function(self, rel_path):
        return self._golden

    def get_input_function(self, rel_path):
        return None

    def get_oracle_function(self, rel_path):
        return None

    def get_bench_function(self, rel_path):
        return None

    def get_output_function(self, rel_path):
        return None


class _StubDataGenerator:
    def generate_input_tensors_from_case(self, input_shapes, dtypes, value_ranges, seed):
        return [torch.ones(*input_shapes[0], dtype=torch.float32)]


class _StubParamBuilder:
    def build_call_params(self, func, case, tensors):
        return {"x": tensors[0]}


class _StubOpRunner:
    """run() 真实调用传入函数产生输出；run_ai_op 可注入输出或失败

    golden_success / ai_raises 用于驱动 evaluate_case 的另外两条出口
    （golden 执行失败、评测异常），它们与成功路径共用同一套释放逻辑。
    """

    def __init__(self, ai_output=None, ai_success=True, golden_success=True, ai_raises=None):
        self._ai_output = ai_output
        self._ai_success = ai_success
        self._golden_success = golden_success
        self._ai_raises = ai_raises
        self._run_calls = 0

    def run(self, func, params, case_id_str, inputs, to_device=False, enable_profiler=False):
        outputs = [func(**params)]
        self._run_calls += 1
        # 第一次 run() 是 golden；同精度参考（native）走的是同一个入口，不该被打成失败
        if not self._golden_success and self._run_calls == 1:
            return OpRunResult(success=False, outputs=outputs,
                               error="stub golden crash", elapsed_us=0)
        return OpRunResult(success=True, outputs=outputs, elapsed_us=1.0)

    def run_ai_op(self, ai_op_func, params, case_id_str, input_tensors, enable_perf=False):
        if self._ai_raises is not None:
            raise self._ai_raises
        if not self._ai_success:
            return OpRunResult(success=False, error="stub ai crash", elapsed_us=0)
        out = self._ai_output if self._ai_output is not None else ai_op_func(**params)
        return OpRunResult(success=True, outputs=[out], elapsed_us=2.0)


class _StubOperatorMatcher:
    def load_ai_operator(self, operator):
        return lambda x: x * 2


class _StubOperatorLoader:
    def get_operator(self, rel_path):
        return SimpleNamespace(
            inputs=[SimpleNamespace(name="x")],
            outputs=[SimpleNamespace(
                name="y", dtype=["float32"], compare=True, index_gather=None)],
            precision_thresholds=None,
        )


def _ensure_correctness_checker():
    """按需补注册 relative_error 判断器。

    全量运行 tests/ut 时 test_checkers.py 会 clear_checker_registry() 且不恢复；
    此处对齐 benches/cann.py 的幂等注册模式，保证本文件用例与执行顺序无关。
    """
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker
    from kernel_eval.registry.checker_registry import CheckerRegistry
    if 'relative_error' not in CheckerRegistry.get_all():
        CheckerRegistry.register('relative_error', RelativeErrorChecker())


def _build_stub_evaluator(op_runner):
    """绕过重量级 __init__，仅装配 evaluate_case 实际使用的协作者"""
    _ensure_correctness_checker()
    evaluator = Evaluator.__new__(Evaluator)
    # CPU-only lifetime checks use real config defaults without NPU profiling.
    evaluator.config = Config(
        eval_seed=0,
        precision_thresholds={},
        enable_accuracy_retry=False,
        enable_profiler=False,
    )
    evaluator.bench_config = SimpleNamespace(
        golden_precision="fp64_cpu",
        dtype_tolerance_map=None,
    )
    evaluator.incremental_output_path = None
    evaluator._pypto_pro_jit_isolation = False
    evaluator.perf_evaluator = None
    evaluator.golden_loader = _StubGoldenLoader()
    evaluator.data_generator = _StubDataGenerator()
    evaluator.operator_loader = _StubOperatorLoader()
    evaluator.param_builder = _StubParamBuilder()
    evaluator.op_runner = op_runner
    evaluator.operator_matcher = _StubOperatorMatcher()
    evaluator.accuracy_evaluator = AccuracyEvaluator()
    return evaluator


class _RecordingOpRunner(_StubOpRunner):
    """在 _StubOpRunner 之上，对每个产出的张量登记一个 weakref。

    用于观测"提前释放"本身：只断言 run_result.outputs is None 是不够的，
    因为把释放点挪回函数返回之后，那个断言依然成立。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.refs = []

    def _track(self, run_result):
        for tensor in run_result.outputs or []:
            if isinstance(tensor, torch.Tensor):
                self.refs.append(weakref.ref(tensor))
        return run_result

    def run(self, *args, **kwargs):
        return self._track(super().run(*args, **kwargs))

    def run_ai_op(self, *args, **kwargs):
        return self._track(super().run_ai_op(*args, **kwargs))


def _alive_when_cleanup_runs(monkeypatch, runner, evaluator):
    """跑一次 evaluate_case，返回 (结果, 进入 _cleanup_memory 时仍存活的张量数)"""
    alive = []
    original = Evaluator._cleanup_memory

    def _spy(self):
        alive.append(sum(1 for ref in runner.refs if ref() is not None))
        return original(self)

    monkeypatch.setattr(Evaluator, "_cleanup_memory", _spy)
    result = evaluator.evaluate_case(_StubCase())
    assert runner.refs, "stub runner 没有产出任何张量，用例失去意义"
    assert alive, "_cleanup_memory 没有被调用"
    return result, alive[-1]


class TestEarlyRelease:
    """大对象必须在 _cleanup_memory 之前就已经无人引用，否则其中的 gc.collect() 空转。

    这是 _release_outputs + 局部名置 None 的实际意图；把释放挪到函数返回之后，
    TestEvaluateCaseRelease 里的断言仍然全绿，只有这里会红。
    """

    def test_pass_path_releases_before_cleanup(self, monkeypatch):
        runner = _RecordingOpRunner()
        result, alive = _alive_when_cleanup_runs(
            monkeypatch, runner, _build_stub_evaluator(runner))

        assert result.success is True
        assert alive == 0, f"{alive}/{len(runner.refs)} 个输出张量在 _cleanup_memory 时仍存活"

    def test_ai_run_failure_releases_before_cleanup(self, monkeypatch):
        # 巨型 case 最可能走的就是这条路径，释放尤其不能等到函数返回
        runner = _RecordingOpRunner(ai_success=False)
        result, alive = _alive_when_cleanup_runs(
            monkeypatch, runner, _build_stub_evaluator(runner))

        assert result.success is False
        assert "AI算子执行失败" in result.error_msg
        assert alive == 0, f"{alive}/{len(runner.refs)} 个输出张量在 _cleanup_memory 时仍存活"

    def test_golden_failure_path_cleans_up(self, monkeypatch):
        # golden 执行失败此前既不释放也不 _cleanup_memory，fp64 输入副本一路活到返回
        runner = _RecordingOpRunner(golden_success=False)
        result, alive = _alive_when_cleanup_runs(
            monkeypatch, runner, _build_stub_evaluator(runner))

        assert result.success is False
        assert "Golden执行失败" in result.error_msg
        assert alive == 0

    def test_evaluation_exception_path_cleans_up(self, monkeypatch):
        # 异常路径的 traceback 会一直引用本帧，不显式置 None 就释放不掉
        runner = _RecordingOpRunner(ai_raises=RuntimeError("boom"))
        result, alive = _alive_when_cleanup_runs(
            monkeypatch, runner, _build_stub_evaluator(runner))

        assert result.success is False
        assert "评测异常" in result.error_msg
        assert alive == 0


class TestEvaluateCaseRelease:
    """精度对比完成后立即释放 fp64 golden 输入副本 / golden/native/AI 输出的回归

    这一组只保证端到端行为不被破坏（无 NameError / use-after-release）且两个
    run result 的 outputs 均已清空；释放时机本身由 TestEarlyRelease 覆盖。
    """

    def test_pass_path_releases_outputs(self):
        evaluator = _build_stub_evaluator(_StubOpRunner())
        result = evaluator.evaluate_case(_StubCase())

        assert result.success is True
        assert result.accuracy_result is not None
        assert result.golden_run_result is not None
        assert result.golden_run_result.outputs is None
        assert result.ai_run_result is not None
        assert result.ai_run_result.outputs is None

    def test_accuracy_fail_path_releases_outputs(self):
        wrong_output = torch.full((8,), 100.0, dtype=torch.float32)
        evaluator = _build_stub_evaluator(_StubOpRunner(ai_output=wrong_output))
        result = evaluator.evaluate_case(_StubCase())

        assert result.success is False
        assert result.error_msg.startswith("精度不达标")
        assert result.golden_run_result.outputs is None
        assert result.ai_run_result.outputs is None

    def test_ai_run_failure_releases_golden_outputs(self):
        evaluator = _build_stub_evaluator(_StubOpRunner(ai_success=False))
        result = evaluator.evaluate_case(_StubCase())

        assert result.success is False
        assert result.failure_type == FAILURE_TYPE_COMPILE_RUNTIME_ERROR
        assert "AI算子执行失败" in result.error_msg
        assert result.golden_run_result.outputs is None
        assert result.ai_run_result.outputs is None
