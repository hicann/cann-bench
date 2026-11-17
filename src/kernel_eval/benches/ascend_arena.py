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


"""Independent registry entry for the native Ascend Arena benchmark."""

from .ascend_arena_loader import AscendArenaCaseLoader, AscendArenaCaseSpec, AscendArenaGoldenLoader, AscendArenaTaskLoader
from ..checkers.relative_error_checker import RelativeErrorChecker
from ..base.matcher import OperatorMatcherBase
from .cann_scoring import RecordingOnlyScheme

__all__ = ["AscendArenaTaskLoader", "AscendArenaCaseLoader", "AscendArenaCaseSpec", "AscendArenaGoldenLoader", "AscendArenaMatcher"]


class AscendArenaMatcher(OperatorMatcherBase):
    def __init__(self, operator_loader):
        self.operator_loader = operator_loader

    def find_operator_info(self, operator_name):
        return self.operator_loader.get_task_by_name(operator_name)

    def find_operator_info_by_snake(self, snake_name):
        return self.find_operator_info(snake_name)

    def load_ai_operator(self, operator_name):
        raise RuntimeError("ascend_arena uses compiled bindings; Python AI operator loading is unsupported")

    def clear_cache(self):
        return None


_REGISTERED = False


def _register_ascend_arena_components():
    global _REGISTERED
    if _REGISTERED:
        return
    from ..registry.loader_registry import LoaderRegistry
    from ..registry.golden_registry import GoldenLoaderRegistry
    from ..registry.matcher_registry import OperatorMatcherRegistry
    from ..registry.checker_registry import CheckerRegistry
    from ..registry.scoring_registry import ScoringSchemeRegistry
    from ..registry.case_spec_registry import CaseSpecRegistry
    from ..registry.bench_registry import BenchConfig, BenchRegistry

    if "ascend_arena" not in LoaderRegistry._task_loaders:
        LoaderRegistry.register_task_loader("ascend_arena", AscendArenaTaskLoader)
    if "ascend_arena" not in LoaderRegistry._case_loaders:
        LoaderRegistry.register_case_loader("ascend_arena", AscendArenaCaseLoader)
    if "ascend_arena" not in GoldenLoaderRegistry._items:
        GoldenLoaderRegistry.register("ascend_arena", AscendArenaGoldenLoader)
    if "ascend_arena" not in OperatorMatcherRegistry._matchers:
        OperatorMatcherRegistry.register("ascend_arena", AscendArenaMatcher)
    if "ascend_arena" not in CheckerRegistry.get_all():
        CheckerRegistry.register("ascend_arena", RelativeErrorChecker())
    if "ascend_arena" not in ScoringSchemeRegistry._items:
        # Arena reports compare measured candidate/baseline latency directly;
        # they do not have CannBench's T_HW anchor for SOL scoring.
        ScoringSchemeRegistry.register("ascend_arena", RecordingOnlyScheme())
    if "ascend_arena" not in CaseSpecRegistry._items:
        CaseSpecRegistry.register("ascend_arena", AscendArenaCaseSpec)
    if "ascend_arena" not in BenchRegistry._items:
        BenchRegistry.register("ascend_arena", BenchConfig(
            task_loader="ascend_arena", case_loader="ascend_arena", golden_loader="ascend_arena",
            operator_matcher="ascend_arena", scoring_scheme="ascend_arena", checker="ascend_arena",
            case_spec_cls="ascend_arena", default_tasks_root="bench_lab/ascend_arena_bench",
            golden_precision="native_npu", description="Native Ascend Arena benchmark",
            metadata={"backend": "ascendc", "runner": "ascend_arena_e2e.ascend_arena.runners.benchmark.BenchmarkRunner"},
        ))
    _REGISTERED = True


_register_ascend_arena_components()
