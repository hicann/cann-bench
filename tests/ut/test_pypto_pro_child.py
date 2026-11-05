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


"""Unit tests for PyPTO-Pro child profiler conditioning."""

from unittest.mock import patch

import pytest

pytest.importorskip(
    "torch_npu",
    reason="PyPTO-Pro child profiler tests require the torch_npu platform dependency",
)

from kernel_eval.eval import pypto_pro_child


class _FakeProfiler:
    def __init__(self, calls):
        self.calls = calls

    def step(self):
        self.calls.append("step")


class _FakeConditioner:
    def __init__(self, calls):
        self.calls = calls

    def stabilize(self):
        self.calls.append("stabilize")

    def clear_cache(self):
        self.calls.append("clear_cache")


def test_preflight_stabilizes_before_candidate():
    calls = []
    conditioner = _FakeConditioner(calls)

    with patch.object(
        pypto_pro_child.torch.npu,
        "synchronize",
        side_effect=lambda *args: calls.append("sync"),
    ):
        pypto_pro_child._run_preflight(
            lambda: calls.append("fn"), conditioner=conditioner
        )

    assert calls == ["stabilize", "fn", "sync"]


def test_profile_steps_clear_cache_only_for_active_repeats():
    calls = []
    conditioner = _FakeConditioner(calls)
    profiler = _FakeProfiler(calls)

    with patch.object(
        pypto_pro_child.torch.npu,
        "synchronize",
        side_effect=lambda *args: calls.append("sync"),
    ):
        exc = pypto_pro_child._run_profile_steps(
            lambda: calls.append("fn"),
            profiler,
            warmup=2,
            repeat=3,
            conditioner=conditioner,
        )

    assert exc is None
    assert calls == [
        "fn",
        "sync",
        "step",
        "fn",
        "sync",
        "step",
        "clear_cache",
        "fn",
        "sync",
        "step",
        "clear_cache",
        "fn",
        "sync",
        "step",
        "clear_cache",
        "fn",
        "sync",
        "step",
    ]


def test_profile_steps_advance_profiler_after_candidate_failure():
    calls = []
    profiler = _FakeProfiler(calls)
    expected = RuntimeError("candidate failed")

    def fail():
        calls.append("fn")
        raise expected

    exc = pypto_pro_child._run_profile_steps(fail, profiler, warmup=0, repeat=1)

    assert exc is expected
    assert calls == ["fn", "step"]


def test_profiler_level_and_activities_match_configuration():
    assert pypto_pro_child._resolve_profiler_level("Level1") == (
        pypto_pro_child.torch_npu.profiler.ProfilerLevel.Level1
    )
    assert pypto_pro_child._resolve_profiler_level("Level2") == (
        pypto_pro_child.torch_npu.profiler.ProfilerLevel.Level2
    )
    assert pypto_pro_child._resolve_profiler_level("invalid") == (
        pypto_pro_child.torch_npu.profiler.ProfilerLevel.Level1
    )
    assert pypto_pro_child._profiler_activities() == [
        pypto_pro_child.torch_npu.profiler.ProfilerActivity.NPU
    ]
