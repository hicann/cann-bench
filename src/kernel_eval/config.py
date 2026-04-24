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
评测工程配置管理

职责：
1. 定义评测工程全局配置
2. 提供配置获取接口
3. 管理路径配置（kernel_bench数据目录、报告输出目录等）
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """评测工程配置"""

    # 路径配置
    kernel_bench_root: str = ""  # kernel_bench数据目录路径
    reports_dir: str = ""        # 报告输出目录

    # 源码目录（AI生成的算子源码，通过参数传入）
    source_dir: str = ""        # AI生成的算子源码目录

    # 设备配置
    device_type: str = "npu"       # cpu / npu
    device_id: int = 0
    auto_fallback: bool = True

    # 性能配置
    enable_profiler: bool = False  # 是否启用性能分析

    # 评测配置
    warmup: int = 3              # 性能评测预热次数
    repeat: int = 5              # 性能评测采集次数

    # 精度配置（采用生态算子开源精度标准）
    # 通过条件: MERE < threshold, MARE < 10 * threshold
    # MERE = avg(|actual - golden| / (|golden| + 1e-7))
    # MARE = max(|actual - golden| / (|golden| + 1e-7))
    precision_thresholds: dict = field(default_factory=lambda: {
        'float16': 2**-10,      # ≈ 0.000976
        'bfloat16': 2**-7,      # ≈ 0.007812
        'float32': 2**-13,      # ≈ 0.000122
        'hifloat32': 2**-11,    # ≈ 0.000488
        'float8_e4m3': 2**-3,   # ≈ 0.125
        'float8_e5m2': 2**-2,   # ≈ 0.25
        'int8': 0,              # 完全相等
        'int16': 0,
        'int32': 0,
        'int64': 0,
        'uint8': 0,
        'uint16': 0,
        'uint32': 0,
        'uint64': 0,
    })

    def __post_init__(self):
        """初始化后自动设置默认路径"""
        if not self.kernel_bench_root:
            # 默认使用项目根目录下的kernel_bench目录
            project_root = Path(__file__).parent.parent.parent
            self.kernel_bench_root = str(project_root / "kernel_bench")

        if not self.reports_dir:
            # 默认报告目录
            project_root = Path(__file__).parent.parent.parent
            self.reports_dir = str(project_root / "reports")

    def get_kernel_bench_path(self) -> Path:
        """获取kernel_bench数据目录路径"""
        return Path(self.kernel_bench_root)

    def get_reports_path(self) -> Path:
        """获取报告输出目录路径"""
        return Path(self.reports_dir)

    def get_source_path(self) -> Path:
        """获取源码目录路径"""
        return Path(self.source_dir) if self.source_dir else None


# 全局配置实例
_global_config: Optional[Config] = None


def get_config() -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """设置全局配置"""
    global _global_config
    _global_config = config