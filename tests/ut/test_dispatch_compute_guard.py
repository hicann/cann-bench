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

"""DeviceResidencyGuard 的 dispatch 层「内置计算算子」黑名单单测（issue #47/#48 漏洞 B）。

TorchDispatchMode 在 CPU 上同样拦得到 ``aten::mm`` / ``aten::topk``，故这些断言无需 NPU。
真实的 C++ ``torch::topk`` on NPU 走同一个 ``aten::topk`` 派发点，行为一致（由 PreSmoke 覆盖）。
"""
import sys
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch  # noqa: E402

from src.kernel_eval.security.device_residency_guard import (  # noqa: E402
    DeviceResidencyGuard, BuiltinComputeError,
)
from src.kernel_eval.security.torch_op_guard import (  # noqa: E402
    TorchOpGuard, aten_compute_leaf, ATEN_COMPUTE_LEAVES,
)


class TestAtenComputeLeaf(unittest.TestCase):
    """aten_compute_leaf：把 OpOverload 规约为黑名单叶子名，只认 aten 命名空间。"""

    def _leaf_of(self, call):
        """跑一次 call 并捕获其 dispatch 到的 func，返回 aten_compute_leaf。"""
        captured = {}
        from torch.utils._python_dispatch import TorchDispatchMode

        class _Probe(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                leaf = aten_compute_leaf(func)
                if leaf is not None:
                    captured["leaf"] = leaf
                return func(*args, **(kwargs or {}))

        with _Probe():
            call()
        return captured.get("leaf")

    def test_matmul_leaf(self):
        a, b = torch.rand(4, 4), torch.rand(4, 4)
        self.assertEqual(self._leaf_of(lambda: torch.matmul(a, b)), "mm")

    def test_topk_leaf(self):
        self.assertEqual(self._leaf_of(lambda: torch.topk(torch.rand(20), 3)), "topk")

    def test_benign_add_is_none(self):
        a, b = torch.rand(4), torch.rand(4)
        self.assertIsNone(self._leaf_of(lambda: a + b))

    def test_custom_namespace_is_none(self):
        """提交的自定义 kernel 在自有命名空间，即便叫 matmul 也绝不命中。"""
        lib = torch.library.Library("cann_bench_ut_a", "DEF")
        lib.define("matmul(Tensor a, Tensor b) -> Tensor")
        lib.impl("matmul", lambda a, b: a * 0 + 1, "CompositeExplicitAutograd")
        op = torch.ops.cann_bench_ut_a.matmul.default
        self.assertEqual(op._schema.name, "cann_bench_ut_a::matmul")
        self.assertIsNone(aten_compute_leaf(op))

    def test_blacklist_excludes_primitives(self):
        """通用原语刻意不入黑名单（避免派发层误伤 glue/golden/harness）。"""
        for prim in ("gather", "where", "max", "min", "abs", "pow", "mean", "sqrt", "arange"):
            self.assertNotIn(prim, ATEN_COMPUTE_LEAVES, f"{prim} 不应在计算黑名单里")

    def test_blacklist_includes_core_compute(self):
        for core in ("mm", "bmm", "addmm", "matmul", "einsum", "topk",
                     "convolution", "_softmax", "linear"):
            self.assertIn(core, ATEN_COMPUTE_LEAVES)


class TestDispatchComputeGuardBlock(unittest.TestCase):
    """block 模式：候选直接调内置计算算子即抛 BuiltinComputeError。"""

    def setUp(self):
        self.a = torch.rand(8, 8)
        self.b = torch.rand(8, 8)

    def _run_block(self, body):
        with DeviceResidencyGuard(mode="block"):
            return body()

    def test_matmul_blocked(self):
        with self.assertRaises(BuiltinComputeError):
            self._run_block(lambda: torch.matmul(self.a, self.b))

    def test_topk_blocked(self):
        with self.assertRaises(BuiltinComputeError):
            self._run_block(lambda: torch.topk(torch.rand(50), 5))

    def test_softmax_blocked(self):
        with self.assertRaises(BuiltinComputeError):
            self._run_block(lambda: torch.nn.functional.softmax(self.a, dim=-1))

    def test_einsum_blocked(self):
        with self.assertRaises(BuiltinComputeError):
            self._run_block(lambda: torch.einsum("ij,jk->ik", self.a, self.b))

    def test_conv2d_blocked(self):
        with self.assertRaises(BuiltinComputeError):
            self._run_block(lambda: torch.nn.functional.conv2d(
                torch.rand(1, 1, 8, 8), torch.rand(1, 1, 3, 3)))

    def test_benign_elementwise_not_blocked(self):
        # 不抛即通过
        out = self._run_block(lambda: self.a + self.b * 2.0)
        self.assertEqual(out.shape, self.a.shape)

    def test_primitives_not_blocked(self):
        self._run_block(lambda: (self.a.sqrt(), self.a.mean(), self.a.max(),
                                 torch.gather(self.a, 1, torch.zeros(8, 1, dtype=torch.long))))

    def test_custom_namespace_op_not_blocked(self):
        lib = torch.library.Library("cann_bench_ut_b", "DEF")
        lib.define("topk(Tensor x) -> Tensor")
        lib.impl("topk", lambda x: x * 0 + 1, "CompositeExplicitAutograd")
        # 命名为 topk 的自定义 kernel 不应被拦（这是合法 AscendC 路径）
        out = self._run_block(lambda: torch.ops.cann_bench_ut_b.topk(self.a))
        self.assertEqual(out.shape, self.a.shape)


class TestPauseExemption(unittest.TestCase):
    """harness 的 freq-boost / L2-flush 经 TorchOpGuard.pause() 豁免，dispatch 层也不误伤。"""

    def test_matmul_under_pause_is_exempt(self):
        a, b = torch.rand(8, 8), torch.rand(8, 8)
        # 不抛 —— 证明 L2-flush(matmul) 不会被新黑名单打挂
        with DeviceResidencyGuard(mode="block"):
            with TorchOpGuard.pause():
                out = torch.matmul(a, b)
        self.assertEqual(out.shape, (8, 8))

    def test_shared_pause_flag_with_torch_op_guard(self):
        """两个守卫共用同一个 _pause_state（TorchOpGuard.pause 即可豁免本守卫）。"""
        from src.kernel_eval.security import torch_op_guard, device_residency_guard
        self.assertIs(torch_op_guard._pause_state, device_residency_guard._pause_state)


class TestWarnAndOffModes(unittest.TestCase):
    def test_warn_records_without_raising(self):
        a, b = torch.rand(4, 4), torch.rand(4, 4)
        g = DeviceResidencyGuard(mode="warn")
        with g:
            torch.matmul(a, b)
        self.assertIn("mm", g.compute_calls)

    def test_off_mode_disables_everything(self):
        a, b = torch.rand(4, 4), torch.rand(4, 4)
        g = DeviceResidencyGuard(mode="off")
        with g:
            torch.matmul(a, b)  # 不抛
        self.assertEqual(g.compute_calls, [])

    def test_enforce_compute_false_skips_compute_check(self):
        a, b = torch.rand(4, 4), torch.rand(4, 4)
        # 只做设备驻留、不做计算黑名单时，matmul 不应被拦
        with DeviceResidencyGuard(mode="block", enforce_compute=False):
            out = torch.matmul(a, b)
        self.assertEqual(out.shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
