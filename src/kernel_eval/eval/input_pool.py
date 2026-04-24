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
输入池管理模块

职责：
1. 预分配一组clone输入，轮换使用
2. 防止按data_ptr缓存输出的攻击
3. 限制内存占用，避免OOM

原理：
- 攻击者可能缓存output按data_ptr缓存结果
- 预分配pool轮换使用，每次调用data_ptr不同
- pool填充时间窗口后，每次都是cache miss

参考evaluation/evaluate.py中的clone pool机制
"""

import torch
from typing import List, Any, Optional
from dataclasses import dataclass


@dataclass
class InputPoolConfig:
    """输入池配置"""
    max_pool_size: int = 8       # 最大池大小
    max_memory_mb: int = 512     # 最大内存占用（MB）


class InputPool:
    """
    输入池管理器

    用于性能测量时防止data_ptr缓存攻击

    使用方法：
        pool = InputPool(inputs, pool_size=warmup+repeat)
        for _ in range(warmup + repeat):
            inputs = pool.get_next()
            output = fn(*inputs)
    """

    def __init__(
        self,
        inputs: List[Any],
        pool_size: int,
        config: Optional[InputPoolConfig] = None
    ):
        """
        Args:
            inputs: 原始输入列表
            pool_size: 期望池大小（通常为warmup + repeat）
            config: 配置参数
        """
        self.config = config or InputPoolConfig()
        self.pool: List[List[Any]] = []
        self.idx = 0

        # 计算每个输入集的内存占用
        per_set_bytes = self._estimate_memory(inputs)

        # 根据内存限制计算实际池大小
        max_sets_by_memory = max(1, (self.config.max_memory_mb * 1024 * 1024) // per_set_bytes)
        actual_size = min(pool_size, max_sets_by_memory, self.config.max_pool_size)
        actual_size = max(1, actual_size)

        # 预分配clone池
        self._allocate_pool(inputs, actual_size)

    def _estimate_memory(self, inputs: List[Any]) -> int:
        """估算输入集的内存占用"""
        total = 0
        for item in inputs:
            if isinstance(item, torch.Tensor):
                total += item.element_size() * item.numel()
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, torch.Tensor):
                        total += sub.element_size() * sub.numel()
        return max(total, 1)  # 至少1字节

    def _allocate_pool(self, inputs: List[Any], size: int) -> None:
        """预分配clone池"""
        for _ in range(size):
            cloned = self._clone_inputs(inputs)
            self.pool.append(cloned)

    def _clone_inputs(self, inputs: List[Any]) -> List[Any]:
        """深度clone输入"""
        cloned = []
        for item in inputs:
            if isinstance(item, torch.Tensor):
                cloned.append(item.clone())
            elif isinstance(item, (list, tuple)):
                cloned.append([sub.clone() if isinstance(sub, torch.Tensor) else sub for sub in item])
            else:
                cloned.append(item)
        return cloned

    def get_next(self) -> List[Any]:
        """
        获取下一个输入集

        Returns:
            clone后的输入列表
        """
        if not self.pool:
            raise RuntimeError("输入池为空")

        inputs = self.pool[self.idx % len(self.pool)]
        self.idx += 1
        return inputs

    def size(self) -> int:
        """获取池大小"""
        return len(self.pool)

    def clear(self) -> None:
        """清空池"""
        self.pool.clear()
        self.idx = 0

    def __len__(self) -> int:
        return self.size()


def create_input_pool(
    inputs: List[Any],
    warmup: int,
    repeat: int,
    max_memory_mb: int = 512
) -> InputPool:
    """
    创建输入池的便捷函数

    Args:
        inputs: 原始输入
        warmup: 预热次数
        repeat: 采集次数
        max_memory_mb: 最大内存占用

    Returns:
        InputPool实例
    """
    config = InputPoolConfig(
        max_pool_size=warmup + repeat,
        max_memory_mb=max_memory_mb
    )
    return InputPool(inputs, warmup + repeat, config)