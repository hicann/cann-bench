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
3. 管理路径配置（tasks 数据目录、报告输出目录等）
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


def _default_precision_thresholds() -> dict:
    """返回 utils.precision.PRECISION_THRESHOLDS 的浅拷贝。

    在函数体内延迟 import 是为了打破 utils <-> config 的潜在循环依赖。
    """
    from .utils.thresholds import PRECISION_THRESHOLDS
    return dict(PRECISION_THRESHOLDS)


@dataclass
class Config:
    """评测工程配置"""

    # 路径配置
    tasks_root: str = ""  # tasks 数据目录路径
    reports_dir: str = ""        # 报告输出目录
    bench_name: str = "cann"    # 评测集名称

    # AI 模型信息（由用户配置，用于报告摘要；为空时摘要中不体现）
    agent_skill: str = ""
    base_model: str = ""

    # 源码目录（AI生成的算子源码，通过参数传入）
    source_dir: str = ""        # AI生成的算子源码目录

    # 设备配置
    device_type: str = "npu"       # cpu / npu
    device_id: int = 0
    auto_fallback: bool = True

    # 性能配置
    # NPU 模式下默认启用 profiler 以获取 kernel-only 时间
    enable_profiler: bool = True
    # Profiler 级别：Level1（默认，47列CSV）或 Level2（更详细AICPU采集）
    profiler_level: str = "Level1"

    # 评测配置
    warmup: int = 3              # 性能评测预热次数
    repeat: int = 5              # 性能评测采集次数

    # 多进程并行配置（统一架构）
    processes_per_card: int = 2  # 每卡进程数

    # 防作弊：用新鲜输入二次验证（Evaluator._retry_with_fresh_inputs，
    # 见 eval/evaluator.py；accuracy_eval.evaluate_with_retry 是早期独立 API，
    # 已 DeprecationWarning）。
    # 防止 submission 缓存第一次输出 / 翻转 "computed-once" 标志骗过单次评测。
    # 启用后每个 case 用新鲜输入再跑一次 golden + AI，两次都通过才记 pass。
    enable_accuracy_retry: bool = False

    # 防作弊：监听 AI 算子在执行时是否直接调用了 torch.matmul / conv / softmax
    # 等内置数学 API（=把计算甩给 PyTorch，跳过自己写的 AscendC kernel）。
    # off=不监听；warn=记日志不阻断（默认，便于排查）；block=直接抛错。
    torch_op_guard_mode: str = "warn"

    # 精度配置（采用生态算子开源精度标准）
    # 通过条件: MERE < threshold, MARE < 10 * threshold
    # MERE = avg(|actual - golden| / (|golden| + 1e-7))
    # MARE = max(|actual - golden| / (|golden| + 1e-7))
    # 单一事实来源：utils.precision.PRECISION_THRESHOLDS。本字段持有它的拷贝，
    # 允许 caller 在不污染模块级常量的前提下自定义阈值（例如 op_info.precision_thresholds
    # 通过 dict.update 覆盖单 dtype）。
    precision_thresholds: dict = field(default_factory=lambda: _default_precision_thresholds())

    # 精度判断器名称
    # 支持选择不同的精度判断标准：
    # - "relative_error": MERE/MARE + 小值域 + 相消处理（默认，完整精度标准）
    # - "allclose": torch.allclose 简化对比（快速验证/调试）
    # 可通过注册机制添加自定义判断器
    checker_name: str = "relative_error"

    def __post_init__(self):
        """初始化后自动设置默认路径"""
        if not self.tasks_root:
            self.tasks_root = str(get_project_root() / "tasks")

        if not self.reports_dir:
            self.reports_dir = str(get_project_root() / "reports")

    def get_tasks_path(self) -> Path:
        """获取tasks 数据目录路径"""
        return Path(self.tasks_root)

    def get_reports_path(self) -> Path:
        """获取报告输出目录路径"""
        return Path(self.reports_dir)

    def get_source_path(self) -> Path:
        """获取源码目录路径"""
        return Path(self.source_dir) if self.source_dir else None


# 全局配置实例
_global_config: Optional[Config] = None

# 项目根目录缓存
_project_root: Optional[Path] = None


def get_project_root() -> Path:
    """返回项目根目录（向上查找 tasks 或 .git 标记）"""
    global _project_root
    if _project_root is not None:
        return _project_root
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "tasks").is_dir() or (current / ".git").is_dir():
            _project_root = current
            return current
        current = current.parent
    raise RuntimeError("Cannot determine project root")


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