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
Timing API防护模块

职责：
1. 在submission代码运行前快照关键Timing API的身份
2. 安装wheel后验证API未被篡改
3. 程序退出前恢复原始API

防护原理：
- 攻击者可能通过monkey-patch修改torch.npu.Event.elapsed_time等API
- 在运行submission代码前快照API的原始callable，安装后验证是否一致
- 如果发现篡改，先恢复原始API再报错，避免atexit崩溃

参考evaluation/evaluate.py中的防篡改机制
"""

import os
from typing import Any, Dict, List, Optional, Tuple


# 关键Timing API列表
_CRITICAL_API_ENTRIES: List[Tuple[str, Any, str]] = []

# 快照存储
_API_SNAPSHOT: Dict[str, Tuple[Any, str, Any]] = {}


def _init_critical_apis():
    """初始化关键API列表（延迟加载，避免import时torch_npu不可用）"""
    global _CRITICAL_API_ENTRIES

    if _CRITICAL_API_ENTRIES:
        return

    try:
        import torch
        import torch_npu

        _CRITICAL_API_ENTRIES = [
            # Legacy event-based timing
            ("torch.npu.Event.elapsed_time", torch.npu.Event, "elapsed_time"),
            ("torch.npu.Event.record", torch.npu.Event, "record"),
            ("torch.npu.synchronize", torch.npu, "synchronize"),
            # Current profiler-based timing
            ("torch_npu.profiler.profile", torch_npu.profiler, "profile"),
            ("torch_npu.profiler.schedule", torch_npu.profiler, "schedule"),
            ("torch_npu.profiler.tensorboard_trace_handler", torch_npu.profiler, "tensorboard_trace_handler"),
            ("torch_npu.profiler._ExperimentalConfig", torch_npu.profiler, "_ExperimentalConfig"),
        ]
    except ImportError:
        # torch_npu不可用时，跳过NPU相关API
        pass


def snapshot_timing_apis() -> None:
    """
    快照Timing API身份

    在submission代码运行前调用，保存原始callable
    """
    global _API_SNAPSHOT

    _init_critical_apis()

    _API_SNAPSHOT = {}
    for name, parent, attr in _CRITICAL_API_ENTRIES:
        try:
            original = getattr(parent, attr)
            _API_SNAPSHOT[name] = (parent, attr, original)
        except AttributeError:
            # API不存在时跳过
            pass


def verify_timing_apis() -> List[str]:
    """
    验证Timing API完整性

    Returns:
        被篡改的API名称列表（空列表表示通过）
    """
    changed = []

    for name, (parent, attr, original) in _API_SNAPSHOT.items():
        try:
            current = getattr(parent, attr)
            if current is not original:
                changed.append(name)
        except AttributeError:
            # API被删除也算篡改
            changed.append(name)

    if changed:
        # 先恢复原始API，避免atexit崩溃
        restore_timing_apis()

    return changed


def restore_timing_apis() -> None:
    """
    恢复原始Timing API

    在程序退出前调用，确保torch_npu的atexit钩子使用原始API
    """
    for name, (parent, attr, original) in _API_SNAPSHOT.items():
        try:
            setattr(parent, attr, original)
        except Exception:
            pass


class APIGuard:
    """
    Timing API防护器

    使用方法：
        guard = APIGuard()
        guard.snapshot()           # 安装wheel前
        install_wheel(path)        # 安装submission
        guard.verify()             # 检查完整性
        # ... 执行评测 ...
        guard.restore()            # 程序退出前
    """

    def __init__(self):
        self._allow_tampering = os.environ.get("ALLOW_TIMING_TAMPERING") == "1"

    def snapshot(self) -> None:
        """快照API身份"""
        snapshot_timing_apis()

    def verify(self) -> bool:
        """
        验证API完整性

        Returns:
            True表示通过，False表示被篡改

        Raises:
            RuntimeError: 如果检测到篡改且未设置ALLOW_TIMING_TAMPERING
        """
        if self._allow_tampering:
            return True

        changed = verify_timing_apis()
        if changed:
            raise RuntimeError(
                f"[SECURITY] Timing API被篡改: {changed}\n"
                "评测结果不可信，已终止执行。\n"
                "如需调试，可设置环境变量 ALLOW_TIMING_TAMPERING=1"
            )
        return True

    def restore(self) -> None:
        """恢复原始API"""
        restore_timing_apis()

    def __enter__(self):
        """上下文管理器入口"""
        self.snapshot()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.restore()
        return False