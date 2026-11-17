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

"""小值域兜底的同精度参考校准（issue #311）。

flash_attention_score_grad_enhance 的 fp32 相消用例此前对任何诚实 f32 实现
不可通过：golden 强制 f32 计算 → native 同精度参考与 golden 逐位恒等 →
兜底分母恒 0。本文件钉死修复后的三层行为：

1. compare 层：参考自身带真实 f32 噪声时，兜底比值 ≤2 营救生效；
   参考缺位/恒等时维持严格判定（防作弊语义不放宽）；
2. golden 层：fasge 的 golden 以 f64 真值计算、`<op>_bench` 钩子保持 f32
   口径，两者不逐位相同但数值相近；bench 的进程内标志用后复位；
3. evaluate 层：native 参考端到端透传后，小值域主导的失败被正确营救。
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from kernel_eval.eval.accuracy_eval import AccuracyEvaluator
from kernel_eval.utils.compare import _compare_single_tensor

FP32_THRESHOLD = 2 ** -13

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = REPO_ROOT / "tasks" / "level4" / "flash_attention_score_grad_enhance" / "golden.py"


@pytest.fixture(autouse=True)
def _ensure_default_checker():
    """幂等补注册默认精度判断器（个别测试文件清空全局注册表且不恢复）。"""
    from kernel_eval.registry import CheckerRegistry
    from kernel_eval.checkers.relative_error_checker import RelativeErrorChecker

    if "relative_error" not in CheckerRegistry.get_all():
        CheckerRegistry.register("relative_error", RelativeErrorChecker())


def _cancellation_profile(seed_candidate: int, seed_native: int):
    """构造相消特征张量：正常域干净、小值域带 f32 量级绝对噪声。

    golden 一半元素 ~1.0（正常域），一半 ~1e-6（小值域，|g| < 2^-14）；
    候选与参考在小值域各带独立的 ~3e-8 噪声（> 2^-30 的小值域错误阈，
    相对误差 ~0.03 > mare 阈值），正常域噪声 1e-9（干净）。
    """
    g = torch.Generator().manual_seed(0)
    n = 4096
    golden = torch.empty(n, dtype=torch.float64)
    golden[: n // 2] = 1.0 + torch.rand(n // 2, generator=g, dtype=torch.float64)
    golden[n // 2:] = 1e-6 * (torch.rand(n // 2, generator=g, dtype=torch.float64) + 0.5)

    def noisy(seed):
        gg = torch.Generator().manual_seed(seed)
        out = golden.clone()
        out[: n // 2] += 1e-9 * torch.randn(n // 2, generator=gg, dtype=torch.float64)
        out[n // 2:] += 3e-8 * torch.randn(n // 2, generator=gg, dtype=torch.float64)
        return out.to(torch.float32)

    return golden, noisy(seed_candidate), noisy(seed_native)


class TestCompareLevel:
    def test_dirty_native_rescues_small_value_failure(self):
        golden, candidate, native = _cancellation_profile(11, 22)
        res = _compare_single_tensor(candidate, golden, FP32_THRESHOLD, "float32",
                                     native_output=native)
        assert res.small_value_total_count > 0
        assert res.small_value_error_count > 0
        assert res.small_value_cpu_error_count > 0, "参考必须计入真实 f32 噪声"
        assert res.passed, (res.small_value_error_count, res.small_value_cpu_error_count)

    def test_missing_native_stays_strict(self):
        golden, candidate, _ = _cancellation_profile(11, 22)
        res = _compare_single_tensor(candidate, golden, FP32_THRESHOLD, "float32",
                                     native_output=None)
        assert not res.passed
        assert res.small_value_cpu_error_count == 0  # 分母恒 0 的旧形态

    def test_normal_band_errors_not_rescued(self):
        """正常值域超标不被小值域营救波及——区分度保留。

        小值域兜底是计数比值制（与现行 spec/fp16 算子语义一致，幅度不敏感，
        幅度感知标准属 issue #311 缺陷 1/2 的后续议题）；正常值域单独使用
        MARE/MERE 比值阈值判定，候选的大幅偏差不能被小值域兜底营救。
        """
        golden, _, native = _cancellation_profile(11, 22)
        bad = golden.clone()
        bad[:8] += 1e-2  # 正常域（|g|~1.0）注入超 mare 阈的误差；参考此处干净
        res = _compare_single_tensor(bad.to(torch.float32), golden, FP32_THRESHOLD,
                                     "float32", native_output=native)
        assert not res.passed
        assert res.small_value_passed and res.cancel_passed
        assert not res.normal_passed
        assert res.normal_mare_ratio > 5.0


class TestGoldenDualPrecision:
    @pytest.fixture()
    def golden_mod(self):
        spec = importlib.util.spec_from_file_location("fasge_golden", GOLDEN_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _inputs(self):
        g = torch.Generator().manual_seed(7)
        t, n, d = 48, 2, 16
        mk = lambda: torch.rand((t, n, d), generator=g, dtype=torch.float32) * 2 - 1
        return dict(query=mk(), key=mk(), value=mk(), dy=mk(), head_num=n,
                    input_layout="TND", scale_value=0.25,
                    actual_seq_qlen=[t], actual_seq_kvlen=[t])

    def test_golden_is_f64_truth_and_bench_is_f32(self, golden_mod):
        kw = self._inputs()
        dq_g, dk_g, dv_g = golden_mod.flash_attention_score_grad_enhance(**kw)
        dq_b, dk_b, dv_b = golden_mod.flash_attention_score_grad_enhance_bench(**kw)
        assert dq_g.dtype == torch.float64 and dq_b.dtype == torch.float32
        # 数值相近（同一数学），但不逐位相同（f64 真值 vs f32 舍入）
        assert torch.allclose(dq_g.float(), dq_b, rtol=1e-3, atol=1e-5)
        assert not torch.equal(dq_g.float(), dq_b)
        assert not torch.equal(dk_g.float(), dk_b)
        assert not torch.equal(dv_g.float(), dv_b)

    def test_bench_flag_resets_after_call(self, golden_mod):
        kw = self._inputs()
        golden_mod.flash_attention_score_grad_enhance_bench(**kw)
        dq_g, _, _ = golden_mod.flash_attention_score_grad_enhance(**kw)
        assert dq_g.dtype == torch.float64  # 标志复位，真值路径不被污染

    def test_native_npu_branch_untouched(self, golden_mod):
        """fp16/bf16 的 NPU native 分支判定条件保持原状（仅静态断言分支谓词）。"""
        src = GOLDEN_PATH.read_text(encoding="utf-8")
        assert 'query.device.type == "npu" and query.dtype in (torch.float16, torch.bfloat16)' in src


class TestEvaluateLevel:
    def test_native_passthrough_rescues(self):
        golden, candidate, native = _cancellation_profile(11, 22)
        ev = AccuracyEvaluator()
        res = ev.evaluate(ai_output=[candidate], golden_output=[golden],
                          dtype="float32", native_output=[native])
        assert res.passed, res.error_msg

    def test_without_native_fails(self):
        golden, candidate, _ = _cancellation_profile(11, 22)
        ev = AccuracyEvaluator()
        res = ev.evaluate(ai_output=[candidate], golden_output=[golden],
                          dtype="float32", native_output=None)
        assert not res.passed
