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
数据类型映射

职责：
1. 提供字符串到torch.dtype的映射
2. 提供torch.dtype到字符串的反向映射
"""

from typing import Dict


# 基础映射表（无需缓存，导入时直接构建）
DTYPE_MAP: Dict = {}
DTYPE_STR_MAP: Dict = {}


def _build_dtype_map() -> Dict:
    """构建数据类型映射表"""
    import torch

    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'float64': torch.float64,
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
    }

    # 可选类型
    optional_types = [
        ('bfloat16', 'bfloat16'),
        ('uint16', 'uint16'),
        ('uint32', 'uint32'),
        ('uint64', 'uint64'),
        ('bool', 'bool'),
    ]

    for attr_name, key_name in optional_types:
        try:
            dtype_map[key_name] = getattr(torch, attr_name)
        except AttributeError:
            pass

    return dtype_map


def str_to_torch_dtype(dtype_str: str):
    """
    将字符串类型转换为torch.dtype

    Args:
        dtype_str: 数据类型字符串，如 'float16', 'int32'

    Returns:
        对应的torch.dtype

    Raises:
        ValueError: 不支持的数据类型
    """
    global DTYPE_MAP
    if not DTYPE_MAP:
        DTYPE_MAP = _build_dtype_map()

    dtype_str_lower = dtype_str.lower()
    if dtype_str_lower not in DTYPE_MAP:
        raise ValueError(f"不支持的数据类型: {dtype_str}. 支持的类型: {list(DTYPE_MAP.keys())}")

    return DTYPE_MAP[dtype_str_lower]


def torch_dtype_to_str(dtype) -> str:
    """
    将torch.dtype转换为字符串

    Args:
        dtype: torch数据类型

    Returns:
        对应的字符串表示

    Raises:
        ValueError: 不支持的数据类型
    """
    global DTYPE_MAP, DTYPE_STR_MAP
    if not DTYPE_MAP:
        DTYPE_MAP = _build_dtype_map()
    if not DTYPE_STR_MAP:
        DTYPE_STR_MAP = {dtype: name for name, dtype in DTYPE_MAP.items()}

    if dtype not in DTYPE_STR_MAP:
        raise ValueError(f"不支持的数据类型: {dtype}")

    return DTYPE_STR_MAP[dtype]


def is_float_dtype(dtype_str: str) -> bool:
    """判断是否为浮点类型"""
    float_types = ['float16', 'float32', 'float64', 'bfloat16']
    return dtype_str.lower() in float_types


def is_int_dtype(dtype_str: str) -> bool:
    """判断是否为整数类型"""
    int_types = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
    return dtype_str.lower() in int_types


def get_dtype_size(dtype_str: str) -> int:
    """获取数据类型的字节大小"""
    dtype = str_to_torch_dtype(dtype_str)
    return dtype.itemsize


def get_supported_dtypes() -> list:
    """获取当前torch版本支持的所有数据类型"""
    global DTYPE_MAP
    if not DTYPE_MAP:
        DTYPE_MAP = _build_dtype_map()
    return list(DTYPE_MAP.keys())