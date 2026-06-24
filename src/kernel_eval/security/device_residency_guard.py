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
"""Dispatch 层反作弊守卫：**设备驻留** + **内置计算算子** 双重拦截。

本守卫用 ``TorchDispatchMode``（ATen 派发层，能看到每个 op 的 device，且 **C++ 与
Python 发起的调用同样可见**）做两件事：

1. **设备驻留**（防"在 CPU 上完成算子、把结果拷回 NPU"作弊）：
   ``TorchOpGuard`` 按名字拦 ``torch.matmul`` 这类内置算子，但拦不住把整段输入/权重
   ``.cpu()`` 搬到 host、用 PyTorch（甚至 ``torch.nn.GRU`` 这种融合算子）在 CPU 上算完、
   再 ``.to(npu)`` 拷回的作弊路径——计算根本没在 NPU 上发生，名字黑名单也追不过来
   （可拆成 sigmoid/tanh/add/mul 等原语）。这里只盯一个 **无法绕过的物理信号**：候选
   算子执行期间是否把"设备上的数据"成块搬回 host（NPU→CPU，即 D2H）。

2. **内置计算算子**（防 C++ 插件直接调 ``torch::matmul`` / ``torch::topk`` 绕过 guard，
   见 issue #47/#48 的「漏洞 B」）：``TorchOpGuard`` 是 Python ``TorchFunctionMode``，
   只拦 Python 侧调用，C++ 插件在 ATen 层直调内置算子不经它。本守卫在 dispatch 层按
   ``ATen_COMPUTE_LEAVES`` 黑名单兜底（只对 ``aten::`` 命名空间生效，提交的自定义
   AscendC kernel 在自有命名空间，不受影响）。可用 ``enforce_compute=False`` 关闭。

两项检查共用 ``TorchOpGuard`` 的 pause 旗标——harness 的 freq-boost / L2-flush（调
``torch.matmul`` / ReduceMax）被 ``TorchOpGuard.pause()`` 豁免，**不会**被这里的 matmul
黑名单误伤。

为什么不会误伤正常的权重/输入搬运
---------------------------------
- harness 把输入/权重搬上卡是 **H2D（CPU→NPU）**，方向相反，本守卫只看 D2H，天然忽略。
- 正常 NPU kernel 收到 NPU 张量后全程在卡上算（``.float()`` / ``.t()`` / ``cat`` / ``flip``
  都在 NPU 上），一次 D2H 都没有。
- 读动态 shape / 标量 / 小索引张量（``.item()`` / 小 ``group_list``）是合法的小额 host 读取，
  用字节阈值放行；只有把"成块的算子数据"搬回 host 才判定。
"""
from __future__ import annotations

import contextlib
from typing import List, Tuple

import torch

from .torch_op_guard import aten_compute_leaf
# 与 TorchOpGuard 共用同一个 pause 旗标：harness 的 freq-boost / L2-flush 只调
# ``TorchOpGuard.pause()``（perf_eval._boost_freq_and_clear_cache），共用后该 pause
# 同时豁免本守卫的 **设备驻留** 与 **内置计算算子** 两项检查，warmup 的 torch.matmul /
# ReduceMax 不会触发误判（否则 dispatch 层的 matmul 黑名单会把 L2-flush 打挂）。
from .torch_op_guard import _pause_state

# 小额 host 读取（动态 shape / 标量 / 小索引张量）放行；只有成块数据外流才判定。
DEFAULT_EGRESS_THRESHOLD_BYTES = 4096

# 非 NPU 的 device 类型（这些上面的张量不算"在卡上"）。
_NON_DEVICE_TYPES = frozenset({"cpu", "meta", "lazy"})


def _iter_tensors(obj):
    """递归展开 args/kwargs/返回值里的所有 Tensor。"""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from _iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from _iter_tensors(x)


def _is_on_device(t) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type not in _NON_DEVICE_TYPES


def _is_on_host(t) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == "cpu"


def _nbytes(t: torch.Tensor) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


class DeviceEgressError(RuntimeError):
    """候选算子把成块数据搬回 host（block 模式下抛出，使该 case 判失败）。"""


class BuiltinComputeError(RuntimeError):
    """候选算子在 ATen 派发层直接调用了内置计算算子（如 C++ ``torch::matmul`` /
    ``torch::topk``）。``TorchOpGuard`` 是 Python ``TorchFunctionMode`` 拦不住 C++ 调用，
    本守卫在 dispatch 层兜底（block 模式下抛出，使该 case 判失败）。见 issue #47/#48。"""


class DeviceResidencyGuard:
    """ContextManager：候选算子执行期间在 ATen 派发层做两项反作弊检查——
    (1) 成块 NPU→CPU 数据外流（设备驻留）；(2) 直接调用内置计算算子（matmul/topk/
    softmax... 含 C++ ``torch::*``）。

    参数
    ----
    mode: ``"block"`` 命中即抛错（:class:`DeviceEgressError` 或 :class:`BuiltinComputeError`，
          该 case 判失败）；``"warn"`` 只记录并打日志；``"off"`` 不启用。
    threshold_bytes: 单次 D2H 拷贝超过该字节数才计为外流（放行小额 host 读取）。
    enforce_compute: 是否启用内置计算算子黑名单（默认 True）。
    """

    def __init__(self, mode: str = "block",
                 threshold_bytes: int = DEFAULT_EGRESS_THRESHOLD_BYTES,
                 enforce_compute: bool = True):
        if mode not in ("block", "warn", "off"):
            raise ValueError(f"mode must be one of block/warn/off, got {mode!r}")
        self.mode = mode
        self.threshold_bytes = threshold_bytes
        # enforce_compute：是否在 dispatch 层同时拦内置计算算子（matmul/topk/softmax...）。
        # 默认开启——这是 TorchOpGuard 对 C++ 调用的兜底（issue #47/#48）。
        self.enforce_compute = enforce_compute
        self.egress_events: List[Tuple[str, int]] = []   # (op_name, nbytes)
        self.total_egress_bytes = 0
        self.compute_calls: List[str] = []               # 命中的 aten 计算叶子（去重前）
        self._inner = None
        self._available = mode != "off"

    @staticmethod
    @contextlib.contextmanager
    def pause():
        """临时关闭外流监听。candidate 代码拿不到本函数（security 子模块隔离）。"""
        prev = getattr(_pause_state, "paused", False)
        _pause_state.paused = True
        try:
            yield
        finally:
            _pause_state.paused = prev

    def _check_compute(self, func):
        """在算子执行 **之前** 检查是否命中 ATen 内置计算算子黑名单。

        命中即记录；block 模式下抛 :class:`BuiltinComputeError`（在执行前拦截，作弊的
        ``torch::matmul`` / ``torch::topk`` 根本不会跑出结果）。只看 ``aten::`` 命名空间，
        提交的自定义 AscendC kernel（自有命名空间）不受影响。
        """
        leaf = aten_compute_leaf(func)
        if leaf is None:
            return
        self.compute_calls.append(leaf)
        if self.mode == "block":
            raise BuiltinComputeError(
                f"[SECURITY] 内置计算算子违规：候选算子在 ATen 派发层直接调用了 "
                f"aten::{leaf}（含 C++ torch::{leaf}）。AscendC kernel 必须自己完成核心"
                f"计算，不允许把 matmul/topk/softmax 等甩给内置算子（issue #47/#48）。"
            )

    def _inspect(self, func, args, kwargs, out):
        # 只有当本次 op 的输入里有"在卡上"的张量时，输出里的 host 张量才可能是 D2H 外流
        if not any(_is_on_device(t) for t in _iter_tensors((args, kwargs))):
            return
        for t in _iter_tensors(out):
            if _is_on_host(t):
                nbytes = _nbytes(t)
                if nbytes >= self.threshold_bytes:
                    op_name = getattr(func, "__name__", None) or str(func)
                    self.egress_events.append((op_name, nbytes))
                    self.total_egress_bytes += nbytes
                    if self.mode == "block":
                        raise DeviceEgressError(
                            "[SECURITY] 设备驻留违规：检测到算子执行期间把 "
                            f"{nbytes} 字节数据从 NPU 搬回 CPU（op={op_name}）。"
                            "算子必须在 NPU 上完成计算，不允许把数据搬到 host 上算完再拷回。"
                        )

    def __enter__(self):
        if not self._available:
            return self
        try:
            from torch.utils._python_dispatch import TorchDispatchMode
        except Exception:
            print("[WARN] DeviceResidencyGuard: TorchDispatchMode 不可用，跳过设备驻留守卫。",
                  flush=True)
            self._available = False
            return self

        outer = self

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                paused = getattr(_pause_state, "paused", False)
                # 内置计算算子检查在 **执行前**：命中且 block 时直接拦下，作弊算子不跑。
                if not paused and outer.enforce_compute:
                    outer._check_compute(func)   # 命中 block 抛 BuiltinComputeError
                out = func(*args, **kwargs)
                if not paused:
                    try:
                        outer._inspect(func, args, kwargs, out)
                    except DeviceEgressError:
                        raise
                    except Exception:
                        # 守卫自身的内省错误绝不能影响算子执行
                        pass
                return out

        self._inner = _Mode()
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._inner is not None:
            self._inner.__exit__(exc_type, exc_val, exc_tb)
            if self.mode == "warn" and self.egress_events:
                ops = ", ".join(sorted({e[0] for e in self.egress_events})[:5])
                print(
                    f"[WARN] DeviceResidencyGuard: 检测到 {len(self.egress_events)} 次 "
                    f"NPU→CPU 成块外流，合计 {self.total_egress_bytes} 字节（op: {ops}）。"
                    "疑似在 host 上完成算子计算。",
                    flush=True,
                )
            if self.mode == "warn" and self.compute_calls:
                uniq = sorted(set(self.compute_calls))
                print(
                    f"[WARN] DeviceResidencyGuard: 候选算子在 ATen 派发层调用了 "
                    f"{len(self.compute_calls)} 次内置计算算子（去重 {len(uniq)} 种）: "
                    f"{', '.join('aten::' + n for n in uniq[:5])}"
                    f"{' ...' if len(uniq) > 5 else ''}。疑似把核心计算甩给内置算子。",
                    flush=True,
                )
        return False
