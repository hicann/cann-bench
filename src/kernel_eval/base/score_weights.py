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

"""Eq.4 权重档：由实际评测集 metadata/VERSION 的大版本号驱动。

优先级：显式 profile > CANN_BENCH_SCORE_WEIGHTS > 评测入口的 bench_root。
无可用来源或未知大版本直接报错。权重与 Eq.3 的 eq3 / eq3_log 选择正交。
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .._version import read_tasks_version

SCORE_WEIGHTS_ENV = "CANN_BENCH_SCORE_WEIGHTS"


@dataclass(frozen=True)
class ScoreWeights:
    """一套 Eq.4 权重。wc + wf + wp + wb = 1，单算子名义满分 100。"""

    wc: float  # 编译/运行正确性
    wf: float  # 精度正确性
    wp: float  # 性能优化性
    wb: float  # 全用例精度通过 bonus（漏跑视为未通过，反作弊路径同步清零）


WEIGHT_PROFILES = {
    "1": ScoreWeights(wc=0.20, wf=0.30, wp=0.50, wb=0.0),
    "2": ScoreWeights(wc=0.05, wf=0.20, wp=0.70, wb=0.05),
}
VALID_SCORE_WEIGHT_PROFILES = tuple(WEIGHT_PROFILES)


def resolve_score_weight_profile(
    profile: Optional[str] = None,
    *,
    bench_root: Optional[Path] = None,
) -> str:
    """解析档位标识；只有更高优先级来源未设置时才读取题集 VERSION。"""
    if profile is None:
        profile = os.environ.get(SCORE_WEIGHTS_ENV)
    if profile is None and bench_root is not None:
        profile = read_tasks_version(bench_root).split(".")[0]
    if profile is None:
        raise ValueError(
            f"score weight profile is required: pass weights, set {SCORE_WEIGHTS_ENV}, "
            "or evaluate a bench_root with metadata/VERSION"
        )
    profile = str(profile).strip()
    if profile not in WEIGHT_PROFILES:
        raise ValueError(
            f"unknown score weight profile {profile!r}; "
            f"expected one of {VALID_SCORE_WEIGHT_PROFILES}. "
            f"Set {SCORE_WEIGHTS_ENV} to a supported profile to override "
            "the metadata/VERSION major version"
        )
    return profile


def resolve_score_weights(profile: Optional[str] = None) -> ScoreWeights:
    """每次调用读取显式参数或环境变量；不提供隐式固定默认档。"""
    return WEIGHT_PROFILES[resolve_score_weight_profile(profile)]


@contextmanager
def score_weights_context(bench_root):
    """为本次评测及其子进程提供内容默认档，结束后恢复调用方环境。

    维护者设置的环境变量优先。避免进程内连续评测不同版本时，前一次的
    内容默认档被误当成后一次的环境覆盖值。
    """
    profile = resolve_score_weight_profile(bench_root=bench_root)
    previous = os.environ.get(SCORE_WEIGHTS_ENV)
    os.environ.setdefault(SCORE_WEIGHTS_ENV, profile)
    try:
        yield profile
    finally:
        if previous is None:
            os.environ.pop(SCORE_WEIGHTS_ENV, None)
        else:
            os.environ[SCORE_WEIGHTS_ENV] = previous
