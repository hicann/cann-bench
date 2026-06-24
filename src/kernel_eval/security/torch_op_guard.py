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

"""Torch op call guard for AI operator execution.

防作弊：检测 AI 算子在执行时是否直接调用了 PyTorch 的内置数学 API
（torch.matmul / torch.nn.functional.conv* / torch.nn.functional.linear /
softmax / attention 等）。AI 算子应当通过编译好的 AscendC kernel 完成计算，
直接调用 torch 内置算子相当于把工作甩给 PyTorch / torch_npu 内核——评测无效。

使用方式：

    from .torch_op_guard import TorchOpGuard

    with TorchOpGuard(forbidden=BUILTIN_COMPUTE_OPS, mode="warn") as g:
        outputs = ai_op_func(**params)
    if g.forbidden_calls:
        # g.forbidden_calls: List[str]
        ...

mode='warn'：检测到禁止 API 时打印 [WARN]，不阻断执行（默认，方便排查）
mode='block'：检测到禁止 API 时抛 RuntimeError（生产 / 防作弊评测使用）
"""
from __future__ import annotations

import contextlib
import threading
from typing import Iterator, List, Optional, Set


# 计算密集型 builtin ops 的默认禁用集合——AI kernel 不应当直接调它们。
# 元数据操作（reshape / view / transpose / contiguous / to / 张量创建）
# 不在此列，AI kernel 的 Python wrapper 经常需要它们做参数预处理。
BUILTIN_COMPUTE_OPS: Set[str] = {
    # 函数形式 torch.xxx(...)
    "torch.matmul",
    "torch.mm",
    "torch.bmm",
    "torch.einsum",
    "torch.nn.functional.linear",
    "torch.nn.functional.conv1d",
    "torch.nn.functional.conv2d",
    "torch.nn.functional.conv3d",
    "torch.nn.functional.conv_transpose1d",
    "torch.nn.functional.conv_transpose2d",
    "torch.nn.functional.conv_transpose3d",
    "torch.nn.functional.softmax",
    "torch.nn.functional.log_softmax",
    "torch.nn.functional.scaled_dot_product_attention",
    "torch.nn.functional.silu",
    "torch.nn.functional.gelu",
    "torch.nn.functional.relu",
    "torch.nn.functional.layer_norm",
    "torch.nn.functional.rms_norm",
    "torch.nn.functional.batch_norm",
    # F029: Tensor 方法形式 output.matmul(weight) 等 —— _qualified_name 对
    # torch._C._TensorBase 返回 "torch.Tensor.<name>"，旧 forbidden 列表缺这一
    # 大类导致方法式调用绕过 guard。补全主要的计算类方法。
    "torch.Tensor.matmul",
    "torch.Tensor.mm",
    "torch.Tensor.bmm",
    "torch.Tensor.einsum",
    "torch.Tensor.__matmul__",  # @ 运算符
    "torch.Tensor.__rmatmul__",
    "torch.Tensor.softmax",
    "torch.Tensor.log_softmax",
    "torch.Tensor.layer_norm",
}


# Operation-leaf → canonical name mapping (anti-bypass normalization).
#
# 同一个数学运算在 PyTorch 内部以多种调度路径暴露：
#   torch.matmul(a, b)              → name='matmul', mod='torch'
#   torch.ops.aten.matmul(a, b)     → name='matmul', mod='torch._ops.aten'
#   torch.ops.aten.mm.default(a, b) → name='mm.default', mod='torch._ops.aten'
#   a @ b                            → name='__matmul__', via Tensor method
#   a.matmul(b)                      → name='matmul', via Tensor method
#   F.linear(x, w)                   → name='linear', mod='torch._C._nn'
#   F.conv2d(x, k)                   → name='conv2d', mod='torch'
# 如果只按完整 qualified name 比对，攻击者随便换一条路径就能绕过。
# 这里把"叶子名"映射到一个 canonical name，无论从哪条路径调用、overload 是
# 哪个，最终都规约到同一个 forbidden 条目上。
_LEAF_TO_CANONICAL: dict = {
    # matmul 家族（@ overload 走 __matmul__ / __rmatmul__）
    "matmul": "torch.matmul",
    "__matmul__": "torch.matmul",
    "__rmatmul__": "torch.matmul",
    "mm": "torch.mm",
    "bmm": "torch.bmm",
    "einsum": "torch.einsum",
    # linear / conv
    "linear": "torch.nn.functional.linear",
    "conv1d": "torch.nn.functional.conv1d",
    "conv2d": "torch.nn.functional.conv2d",
    "conv3d": "torch.nn.functional.conv3d",
    "conv_transpose1d": "torch.nn.functional.conv_transpose1d",
    "conv_transpose2d": "torch.nn.functional.conv_transpose2d",
    "conv_transpose3d": "torch.nn.functional.conv_transpose3d",
    # softmax / attention
    "softmax": "torch.nn.functional.softmax",
    "log_softmax": "torch.nn.functional.log_softmax",
    "scaled_dot_product_attention": "torch.nn.functional.scaled_dot_product_attention",
    # activation
    "silu": "torch.nn.functional.silu",
    "gelu": "torch.nn.functional.gelu",
    "relu": "torch.nn.functional.relu",
    # norm
    "layer_norm": "torch.nn.functional.layer_norm",
    "rms_norm": "torch.nn.functional.rms_norm",
    "batch_norm": "torch.nn.functional.batch_norm",
}


# ── Dispatch 层（ATen）计算算子黑名单 ────────────────────────────────────────
# 上面的 ``BUILTIN_COMPUTE_OPS`` 经 ``TorchFunctionMode`` 拦截，**只能看到 Python 侧**
# 发起的调用。被测算子的 C++ 插件直接调 ``torch::matmul`` / ``torch::topk`` /
# ``torch::softmax`` 走 ATen C++ dispatcher，**不经过 __torch_function__**，因此 Python
# 黑名单对其完全无效（见 issue #47 / #48 的「漏洞 B」）。
#
# ``DeviceResidencyGuard`` 用 ``TorchDispatchMode`` 在 **ATen 派发层** 拦截——无论调用
# 从 Python 还是 C++ 发起，都会经过这里。下表收录的是各被测算子「核心计算」对应的
# aten 叶子；候选算子若直接命中其一，等价于把计算甩给内置算子。
#
# 设计取舍（重要）：只收「明显承担核心计算」的算子——matmul/卷积/softmax/attention/
# norm/激活/topk-sort。**刻意不收** gather/where/max/min/abs/pow/mean/sqrt/arange 等
# 通用原语：它们是合法 AscendC kernel 的 host 侧 glue、golden 参考实现、以及 harness
# 自身（如 ReduceMax 清 L2 cache）都会大量使用，在派发层无差别拉黑会造成严重误伤。
ATEN_COMPUTE_LEAVES: Set[str] = {
    # matmul 家族（torch::matmul 视输入维度分解为 mm/bmm/addmm/baddbmm）
    "mm", "bmm", "addmm", "baddbmm", "addbmm", "matmul", "mv", "addmv", "dot",
    "einsum", "tensordot", "linear",
    # 卷积
    "convolution", "_convolution", "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d",
    "_conv_depthwise2d", "slow_conv_transpose2d", "slow_conv_transpose3d",
    # softmax / attention
    "softmax", "_softmax", "log_softmax", "_log_softmax",
    "scaled_dot_product_attention",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_attention_math",
    # norm
    "layer_norm", "native_layer_norm", "group_norm", "native_group_norm",
    "batch_norm", "native_batch_norm", "rms_norm", "_fused_rms_norm",
    # 激活（与上面 Python 黑名单的 silu/gelu/relu 对齐）
    "silu", "gelu", "glu", "relu",
    # 排序/选择（#47/#48 实证：TopK split 模式用 torch::topk + torch::gather 做合并；
    # topk/sort 是 TopK/Sort 算子的核心计算，候选不应直接调内置 aten 版本）
    "topk", "sort",
}


def aten_compute_leaf(func) -> Optional[str]:
    """若 ``func`` 是 **ATen 命名空间** 下的内置计算算子，返回其规约叶子名；否则 None。

    用于 ``TorchDispatchMode`` 派发层：``func`` 是 ``OpOverload``，``func._schema.name``
    形如 ``"aten::mm"`` / ``"aten::topk"``（overload 后缀在 ``__name__`` 上，如
    ``"mm.default"``）。C++ 发起的 ``torch::matmul`` 同样在此可见。

    **只对 ``aten::`` 命名空间生效**——提交的自定义 AscendC kernel 注册在自有命名空间
    （如 ``cann_bench::`` / PrivateUse1），其 ``_schema.name`` 不以 ``aten::`` 开头，
    绝不会被误判为作弊。
    """
    schema = getattr(func, "_schema", None)
    name = getattr(schema, "name", "") if schema is not None else ""
    if name:
        if "::" in name:
            ns, leaf = name.split("::", 1)
        else:
            ns, leaf = "aten", name
    else:
        # 无 schema 的兜底：用 __name__（形如 "mm.default"），保守按 aten 处理。
        ns, leaf = "aten", (getattr(func, "__name__", "") or "")
    leaf = leaf.split(".", 1)[0]
    if ns != "aten" or not leaf:
        return None
    return leaf if leaf in ATEN_COMPUTE_LEAVES else None


def _should_normalize_leaf(mod: str, qualname: str) -> bool:
    """Return whether a leaf op name belongs to PyTorch builtin dispatch.

    Custom operators registered under ``torch.ops.<namespace>`` may deliberately
    reuse public operator names such as ``gelu`` or ``softmax``. Normalizing only
    by the leaf name would classify ``torch.ops.cann_bench.gelu`` as
    ``torch.nn.functional.gelu`` before PyTorch dispatch reaches the submitted
    PrivateUse1 implementation. Only normalize known PyTorch builtin entry
    points and Tensor methods.
    """
    if "TensorBase" in qualname or "Tensor." in qualname:
        return True
    if mod.startswith("torch._C._TensorBase"):
        return True

    builtin_prefixes = (
        "torch._C",
        "torch._ops.aten",
        "torch.nn.functional",
    )
    return mod == "torch" or any(mod.startswith(prefix) for prefix in builtin_prefixes)


def _qualified_name(func) -> str:
    """计算操作的 canonical name，规约所有调度路径（aten / @ / method / F.*）到
    BUILTIN_COMPUTE_OPS 中的同一条目，防止单条规则被多路径绕过。

    优先级：
      1. 内置 PyTorch API / aten / Tensor method 且 ``__name__`` 的"叶子部分"
         （去掉 ``.default`` 等 overload 后缀）在 ``_LEAF_TO_CANONICAL`` 中
         → 返回 canonical name。
      2. 否则按旧规则（Tensor method 走 ``torch.Tensor.<name>``，其余 ``mod.name``）。
    """
    name = getattr(func, "__name__", "") or repr(func)
    mod = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", "") or ""
    # Strip overload suffix: "mm.default" → "mm"
    leaf = name.split(".", 1)[0]
    if leaf in _LEAF_TO_CANONICAL and _should_normalize_leaf(mod, qualname):
        return _LEAF_TO_CANONICAL[leaf]
    # Old behavior (fallback for ops not in canonical map)
    if "TensorBase" in qualname or "Tensor." in qualname:
        return f"torch.Tensor.{name}"
    if mod.startswith("torch._C._TensorBase"):
        return f"torch.Tensor.{name}"
    return f"{mod}.{name}" if mod else name


# Thread-local pause flag — set by TorchOpGuard.pause() context manager.
# Used by harness-internal code (perf_eval freq-boost warmup) to suppress
# guard during routine that is not the candidate kernel's computation.
_pause_state = threading.local()


class TorchOpGuard:
    """ContextManager：在 with 块内监听 torch.* 调用。

    用 ``torch.overrides.TorchFunctionMode``（PyTorch ≥1.11）。如果当前
    PyTorch 不支持，构造时打印 [WARN] 并降级为无操作 context。
    """

    def __init__(self, forbidden: Optional[Set[str]] = None, mode: str = "warn"):
        self.forbidden = forbidden if forbidden is not None else BUILTIN_COMPUTE_OPS
        if mode not in ("warn", "block", "off"):
            raise ValueError(f"mode must be one of warn/block/off, got {mode!r}")
        self.mode = mode
        self.forbidden_calls: List[str] = []
        self._inner = None
        self._available = mode != "off"

    @staticmethod
    @contextlib.contextmanager
    def pause() -> Iterator[None]:
        """临时关闭当前线程内的 TorchOpGuard 监听。

        harness 自身在 candidate 前后做的 freq-boost / L2-flush 会调
        ``torch.matmul`` / ``torch.max``，这些不是 candidate 的真实计算，
        必须显式排除以免产生 false positive。

        用法（仅 harness 内部）::

            with TorchOpGuard.pause():
                torch.matmul(mm1, mm2)   # warmup, 不计数
                torch.max(reduce_input)  # cache flush, 不计数

        - 嵌套安全：内层退出后外层仍处于 paused（自动保存/恢复）。
        - 不在 ``TorchOpGuard`` ``with`` 块内时为 no-op。
        - candidate kernel 拿不到该 API（``security`` 子模块对外部代码隔离），
          不应被滥用作绕过手段。
        """
        prev = getattr(_pause_state, "paused", False)
        _pause_state.paused = True
        try:
            yield
        finally:
            _pause_state.paused = prev

    def __enter__(self):
        if not self._available:
            return self
        try:
            import torch.overrides as _ov
        except ImportError:
            print("[WARN] TorchOpGuard: torch.overrides 不可用（PyTorch <1.11），跳过守卫。", flush=True)
            self._available = False
            return self

        outer = self

        class _Mode(_ov.TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                # Fast-path: harness 内部预热阶段调用，整段跳过守卫
                if getattr(_pause_state, "paused", False):
                    return func(*args, **kwargs)
                fq = _qualified_name(func)
                if fq in outer.forbidden:
                    outer.forbidden_calls.append(fq)
                    if outer.mode == "block":
                        raise RuntimeError(
                            f"[SECURITY] AI 算子调用了被禁用的 PyTorch 内置 API: {fq}。"
                            f"AscendC kernel 应通过编译好的算子完成计算，不应直接调用 torch.matmul 等。"
                        )
                return func(*args, **kwargs)

        self._inner = _Mode()
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._inner is not None:
            self._inner.__exit__(exc_type, exc_val, exc_tb)
            if self.mode == "warn" and self.forbidden_calls:
                uniq = sorted(set(self.forbidden_calls))
                print(
                    f"[WARN] TorchOpGuard: AI 算子调用了 {len(self.forbidden_calls)} 次"
                    f" 禁用 API（去重 {len(uniq)} 种）: {', '.join(uniq[:5])}"
                    f"{' ...' if len(uniq) > 5 else ''}",
                    flush=True,
                )
        return False
