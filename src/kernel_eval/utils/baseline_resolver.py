#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You can not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Baseline 硬件解析模块

职责：
1. 提供硬件名称 → baseline 逻辑名的映射（resolve_hardware）
2. 维护平台别名映射（PLATFORM_ALIAS）
3. 维护默认硬件常量（DEFAULT_HARDWARE）

baseline 性能数据已迁移至 metadata/<hardware>.json（BaselineStore 管理），
cases.yaml 中不再内嵌 baseline_perf_us / t_hw_us 字段。
本模块仅保留硬件名称解析能力，供 BaselineStore、cann_loader 等模块使用。

硬件配置：
    DEFAULT_HARDWARE 从环境变量 CANN_BENCH_HARDWARE 读取，未设置时 fallback 到 "910b2"。
"""

import logging
import os
from typing import Dict


# 默认硬件——支持环境变量覆盖以避免硬编码 (F170)。
# 未设置时 fallback 到 "910b2"，与历史行为一致。
DEFAULT_HARDWARE: str = os.environ.get("CANN_BENCH_HARDWARE", "910b2")


# ---------------------------------------------------------------------------
# 平台别名映射
# ---------------------------------------------------------------------------
# torch.npu.get_device_name() 返回的是产品型号名（如 "Ascend910_9362"），
# 而 baseline 文件使用的是简短的逻辑名（如 "910b2"）。
# 此映射将产品名 → 逻辑名，确保自动检测的硬件名能找到对应的 baseline 数据。
#
# 映射关系：
#   Ascend 910B 系列 (A2 / Atlas A2 / 910B2 / 910_9362 等) → "910b2"
#   Ascend 910B1 系列 → "910b1"
#   Ascend 310P 系列 → "310p"
#   未来新增平台只需在此处加一行映射 + 在 metadata/ 下加对应 JSON 文件。
#
# 产品型号对照（华为官方命名）：
#   Ascend910_9362  = Ascend 910B2 (Atlas A2 训练卡)
#   Ascend910_9362B = Ascend 910B2 变体
#   Ascend910_9361  = Ascend 910B1 (Atlas A2 推理卡)
#   Ascend310P_???  = Ascend 310P (Atlas 推理卡)
# ---------------------------------------------------------------------------
PLATFORM_ALIAS: Dict[str, str] = {
    # 910B2 (Atlas A2 训练卡)
    "Ascend910_9362": "910b2",
    "Ascend910_9362B": "910b2",
    "Ascend910B2": "910b2",
    "910b2": "910b2",
    "Atlas-A2": "910b2",
    # 910B1 (Atlas A2 推理卡)
    "Ascend910_9361": "910b1",
    "Ascend910B1": "910b1",
    "910b1": "910b1",
    # 310P (Atlas 推理卡) — key 是前缀，子型号由 resolve_hardware 前缀匹配
    "Ascend310P": "310p",
    "310p": "310p",
}


_logger = logging.getLogger(__name__)


def resolve_hardware(hardware: str) -> str:
    """将硬件名称（含产品型号别名）解析为 baseline 逻辑名。

    查找顺序：
    1. PLATFORM_ALIAS 中有精确映射 → 返回逻辑名（如 "910b2")
    2. PLATFORM_ALIAS 中有前缀匹配 → 返回对应的逻辑名
       （如 "Ascend310P3" 前缀匹配 key "Ascend310P" → "310p")
    3. 无匹配 → 返回原值（用户可能用了自定义名称）

    Args:
        hardware: 环境变量、torch.npu.get_device_name() 或用户指定的硬件名

    Returns:
        对应 metadata/ 下的文件名前缀（如 "910b2")
    """
    # 1. 精确匹配
    if hardware in PLATFORM_ALIAS:
        return PLATFORM_ALIAS[hardware]

    # 2. 前缀匹配（最长前缀优先，避免 "Ascend910" 误匹配 "Ascend910B2")
    best_prefix = ""
    best_value = None
    for alias_key, alias_value in PLATFORM_ALIAS.items():
        if hardware.startswith(alias_key) and len(alias_key) > len(best_prefix):
            best_prefix = alias_key
            best_value = alias_value

    if best_value is not None:
        _logger.debug("resolve_hardware: %r 前缀匹配 → %r (prefix=%r)",
                      hardware, best_value, best_prefix)
        return best_value

    # 3. 无匹配 → 返回原值
    return hardware