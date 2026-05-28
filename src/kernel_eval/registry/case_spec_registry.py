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
CaseSpec 子类注册表

CaseSpecRegistry: 管理各评测集的 CaseSpec 子类

Why: 多卡并行子进程重建 CaseSpec 时需要知道使用哪个子类（CannCaseSpec vs CaseSpec 基类），
     通过注册表动态获取，避免硬编码
"""

from typing import Dict, Optional, Type

from ..base.models import CaseSpec


class CaseSpecRegistry:
    """CaseSpec 子类注册表"""

    _items: Dict[str, Type[CaseSpec]] = {}

    @classmethod
    def register(cls, name: str, spec_cls: Type[CaseSpec]) -> None:
        """注册 CaseSpec 子类"""
        if name in cls._items:
            raise ValueError(f"CaseSpec '{name}' 已注册")
        cls._items[name] = spec_cls

    @classmethod
    def get(cls, name: str) -> Type[CaseSpec]:
        """获取 CaseSpec 子类，未注册时返回基类"""
        return cls._items.get(name, CaseSpec)

    @classmethod
    def list_all(cls) -> list:
        """列出所有已注册名称"""
        return list(cls._items.keys())