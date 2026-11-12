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
2. 在有限池内轮换地址（不保证阻止结果缓存）
3. 限制内存占用，避免OOM

原理：
- 攻击者可能缓存output按data_ptr缓存结果
- 预分配 pool 轮换使用；池大小为 1 或绕回后地址会重复
- 本池不恢复被修改的输入；严格恢复模式见 restored_inputs.py

参考evaluation/evaluate.py中的clone pool机制
"""

from typing import List, Any, Optional
from dataclasses import dataclass

import torch

# Try to use the framework copy kernel to avoid depending on
# aclnnInplaceCopy which may be unavailable due to anti-cheat
# TBE kernel tree deletion.
try:
    from cann_bench_utils import cann_bench_clone as _safe_tensor_clone
except ImportError:
    def _safe_tensor_clone(x):
        return x.clone()



@dataclass
class InputPoolConfig:
    """输入池配置"""
    max_pool_size: int = 8       # 最大池大小
    max_memory_mb: int = 512     # 最大内存占用（MB）


class InputPool:
    """
    输入池管理器

    用于性能测量时有限轮换输入地址

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
                cloned.append(_safe_tensor_clone(item))
            elif isinstance(item, (list, tuple)):
                cloned.append([_safe_tensor_clone(sub) if isinstance(sub, torch.Tensor) else sub for sub in item])
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


class CallInputPool:
    """调用级输入池：轮换一次完整调用 (args, kwargs) 的 clone 副本。

    与 InputPool（按位置列表轮换）不同，本类同时覆盖位置参数与关键字参数，
    用于性能测量阶段对实际算子调用防 data_ptr 缓存攻击：

    - 只 clone 张量（含嵌套于 list / tuple 中的张量）；非张量值（如 attr）按引用共享。
    - 每次 get_next 返回 (args, kwargs)，池大小 >= 2 时相邻两次的张量地址不同，
      但池绕回后仍可能命中缓存，且不恢复已修改的值。
    - clone 值与原输入逐位相等，故性能阶段的精度复检不受影响。
    - 池大小按 InputPoolConfig 的内存 / 数量上限裁剪，避免大张量 OOM。

    使用方法：
        pool = CallInputPool(args, kwargs, warmup + repeat)
        for _ in range(warmup + repeat):
            a, kw = pool.get_next()
            output = func(*a, **kw)
    """

    def __init__(
        self,
        args: Any,
        kwargs: Any,
        pool_size: int,
        config: Optional[InputPoolConfig] = None,
    ):
        self.config = config or InputPoolConfig()
        self.idx = 0
        args = tuple(args or ())
        kwargs = dict(kwargs or {})

        per_set_bytes = self._estimate_memory(args, kwargs)
        max_sets_by_memory = max(1, (self.config.max_memory_mb * 1024 * 1024) // per_set_bytes)
        actual_size = min(pool_size, max_sets_by_memory, self.config.max_pool_size)
        actual_size = max(1, actual_size)

        self.pool: List[Any] = [
            (self._clone_seq(args), self._clone_map(kwargs))
            for _ in range(actual_size)
        ]

    @staticmethod
    def _clone_one(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return _safe_tensor_clone(value)
        if isinstance(value, tuple):
            return tuple(_safe_tensor_clone(v) if isinstance(v, torch.Tensor) else v for v in value)
        if isinstance(value, list):
            return [_safe_tensor_clone(v) if isinstance(v, torch.Tensor) else v for v in value]
        return value

    def _clone_seq(self, args):
        return tuple(self._clone_one(a) for a in args)

    def _clone_map(self, kwargs):
        return {k: self._clone_one(v) for k, v in kwargs.items()}

    def _estimate_memory(self, args, kwargs) -> int:
        total = 0

        def _add(v):
            nonlocal total
            if isinstance(v, torch.Tensor):
                total += v.element_size() * v.numel()
            elif isinstance(v, (list, tuple)):
                for sub in v:
                    if isinstance(sub, torch.Tensor):
                        total += sub.element_size() * sub.numel()

        for a in args:
            _add(a)
        for v in kwargs.values():
            _add(v)
        return max(total, 1)  # 至少1字节

    def get_next(self):
        """返回下一组 (args, kwargs)（其中张量为 clone）。"""
        if not self.pool:
            raise RuntimeError("调用输入池为空")
        args, kwargs = self.pool[self.idx % len(self.pool)]
        self.idx += 1
        return args, kwargs

    def size(self) -> int:
        return len(self.pool)

    def clear(self) -> None:
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