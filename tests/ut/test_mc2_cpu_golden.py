# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Real PyTorch CPU reference tests. No NPU needed; skip if torch is absent."""

import importlib.util
from pathlib import Path
import types
import unittest

try:
    import torch
except ImportError:
    torch = None


LEVEL = Path(__file__).resolve().parents[2] / "bench_lab/kernel_bench/level4"


def golden(op):
    spec = importlib.util.spec_from_file_location(
        "test_golden_" + op, LEVEL / op / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, "PyTorch CPU is not installed")
class TestMC2CPUReferences(unittest.TestCase):
    def test_quant_allreduce_bias_scales_x3_before_collective(self):
        module = golden("matmul_all_reduce")
        inputs = {
            "x1": torch.tensor([[[1, 2], [3, -1]]], dtype=torch.int8),
            "x2": torch.tensor([[2, 1], [1, -2]], dtype=torch.int8),
            "bias": torch.tensor([3, -2], dtype=torch.int32),
            "pertoken_scale": torch.tensor([0.5, 0.25]),
            "dequant_scale": torch.tensor([0.25, 0.5]),
            "x3": torch.tensor([[[0.5, -0.5], [1, 2]]]),
        }
        expected_local = torch.tensor([[[1.375, -1.75], [1.5, 2.375]]])
        seen = []

        def reduce(tensor, op):
            seen.append(tensor.clone())
            tensor.add_(
                10
            )  # A distinct remote-rank contribution, not world_size multiplication.

        dist = types.SimpleNamespace(
            all_reduce=reduce, ReduceOp=types.SimpleNamespace(SUM="sum")
        )
        result = module.mc2_cpu_golden({"dist": dist}, inputs, {"quantized": True})
        torch.testing.assert_close(seen[0], expected_local)
        torch.testing.assert_close(result, expected_local + 10)
        self.assertEqual(result.dtype, torch.float32)

    def test_reduce_scatter_fp32_without_output_rounding(self):
        module = golden("matmul_reduce_scatter_v2")
        x = torch.tensor([[1.25], [2.5], [3.75], [4.5]], dtype=torch.float16)
        w = torch.tensor([[0.3]], dtype=torch.float16)

        def reduce(tensor, op):
            tensor.add_(1.123456)

        dist = types.SimpleNamespace(
            all_reduce=reduce, ReduceOp=types.SimpleNamespace(SUM="sum")
        )
        result = module.mc2_cpu_golden(
            {"dist": dist, "rank": 1, "world_size": 2},
            {"x1": x, "x2": w, "bias": None},
            {"x1_dtype": "fp16", "x2_dtype": "fp16"},
        )
        torch.testing.assert_close(result, (x.float() @ w.float() + 1.123456)[2:])
        self.assertEqual(result.dtype, torch.float32)

    def test_alltoall_matmul_distinct_rank_layout_and_optional_output(self):
        module = golden("allto_all_matmul")
        inputs = {
            "x1": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "x2": torch.arange(8, dtype=torch.float32).reshape(4, 2),
            "bias": None,
        }

        def exchange(output, sent):
            output.copy_(torch.tensor([[[3.0, 4.0]], [[30.0, 40.0]]]))

        ctx = {
            "world_size": 2,
            "dist": types.SimpleNamespace(all_to_all_single=exchange),
        }
        out, gathered = module.mc2_cpu_golden(ctx, inputs, {"alltoall_out_flag": True})
        expected_gather = torch.tensor([[3.0, 4.0, 30.0, 40.0]])
        torch.testing.assert_close(gathered, expected_gather)
        torch.testing.assert_close(out, expected_gather @ inputs["x2"])
        self.assertIsNone(module.mc2_cpu_golden(ctx, inputs, {})[1])

    def test_matmul_alltoall_splits_columns_and_concatenates_rows(self):
        module = golden("matmul_allto_all")
        inputs = {
            "x1": torch.eye(2),
            "x2": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "bias": torch.ones(4),
        }
        expected = (
            (inputs["x1"] @ inputs["x2"] + inputs["bias"])
            .reshape(2, 2, 2)
            .permute(1, 0, 2)
        )

        def exchange(output, sent):
            torch.testing.assert_close(sent, expected)
            output.copy_(sent)

        result = module.mc2_cpu_golden(
            {
                "world_size": 2,
                "dist": types.SimpleNamespace(all_to_all_single=exchange),
            },
            inputs,
            {},
        )
        torch.testing.assert_close(result, expected.contiguous().reshape(4, 2))

    def test_fp8_per_tensor_cpu_scale_and_gathered_output(self):
        module = golden("all_gather_matmul_v2")
        inputs = {
            "x1": torch.tensor([[1.0, 2.0]]).to(torch.float8_e4m3fn),
            "x2": torch.tensor([[1.0], [2.0]]).to(torch.float8_e5m2),
            "bias": torch.tensor([0.25]),
            "x1_scale": torch.tensor([0.3333]),
            "x2_scale": torch.tensor([0.2222]),
        }

        def gather(output, local):
            output.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

        ctx = {"world_size": 2, "dist": types.SimpleNamespace(_all_gather_base=gather)}
        out, gathered = module.mc2_cpu_golden(ctx, inputs, {"gather_output": True})
        scale = inputs["x1_scale"] * inputs["x2_scale"]
        scale = (scale.view(torch.int32) & -8192).view(torch.float32)
        torch.testing.assert_close(
            out, (gathered @ inputs["x2"].float() + 0.25) * scale
        )
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(gathered.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
