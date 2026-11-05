# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""A5 GMM/AlltoAllv contracts and two-rank numerical routing, without an NPU."""

import csv
import importlib.util
import json
from pathlib import Path
import types
import unittest

import yaml

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "bench_lab/kernel_bench/level4/grouped_mat_mul_allto_allv"
CASES = yaml.safe_load((TASK / "cases.yaml").read_text(encoding="utf-8"))["cases"]
A5 = [case for case in CASES if case["attrs"].get("mc2_accuracy_mode") == "three_way"]


def load_golden():
    spec = importlib.util.spec_from_file_location(
        "gmm_alltoallv_test_golden", TASK / "golden.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(case):
    return {
        "input_shapes": case["input_shape"],
        "dtypes": case["dtype"],
        "value_ranges": case["value_range"],
        "attrs": case["attrs"],
    }


class TestA5GmmMetadata(unittest.TestCase):
    def test_legacy_and_a5_cases_are_selected_only_for_their_supported_hardware(self):
        from kernel_eval.utils.case_eligibility import filter_cases_for_hardware

        for hardware in ("910b2", "Ascend910B2", "950pr"):
            self.assertEqual(filter_cases_for_hardware(CASES, hardware), [])
        for hardware in ("910c", "Ascend910_9391", "Ascend910_9362", "Atlas-A3"):
            selected = filter_cases_for_hardware(CASES, hardware)
            self.assertEqual([case["case_id"] for case in selected], list(range(1, 21)))
        for hardware in ("950dt", "Ascend950DT_9582"):
            selected = filter_cases_for_hardware(CASES, hardware)
            self.assertEqual(
                [case["case_id"] for case in selected], list(range(21, 41))
            )

    def test_cases_are_a5_nonquantized_and_counts_are_consistent(self):
        self.assertEqual(len(CASES), 40)
        self.assertEqual([case["case_id"] for case in A5], list(range(21, 41)))
        for case in A5:
            attrs, shapes = case["attrs"], case["input_shape"]
            self.assertEqual(attrs["supported_devices"], ["950dt"])
            self.assertEqual(attrs["world_size"], 2)
            self.assertEqual(attrs["mc2_compare_dtype"], "float32")
            self.assertIn(case["dtype"][0], ("float16", "bfloat16"))
            counts, experts, rows = attrs["token_counts"], attrs["e"], shapes[0][0]
            self.assertIn(experts, (1, 2, 4))
            for rank in range(2):
                self.assertEqual(sum(sum(edge) for edge in counts[rank]), rows)
                self.assertEqual(sum(sum(counts[src][rank]) for src in range(2)), rows)
                for edge in counts[rank]:
                    self.assertEqual(len(edge), experts)
                    self.assertTrue(all(count >= 0 for count in edge))
            if attrs["mm_out"]:
                self.assertEqual(rows // shapes[4][0], 2)
            else:
                self.assertEqual(shapes[4:], [None, None])
                self.assertFalse(attrs["trans_mm_weight"])
        self.assertTrue(
            any(
                0 in edge
                for case in A5
                for row in case["attrs"]["token_counts"]
                for edge in row
            )
        )
        self.assertTrue(
            any(
                sum(c[0][0]) != sum(c[0][1])
                for c in (case["attrs"]["token_counts"] for case in A5)
            )
        )

    def test_proto_and_csv_match_including_preserved_legacy_cases(self):
        proto = yaml.safe_load((TASK / "proto.yaml").read_text(encoding="utf-8"))[
            "operator"
        ]
        attrs = {item["name"] for item in proto["attrs"]}
        self.assertIn(
            "str hcomm_info, int world_size, int[] send_counts, int[] recv_counts",
            proto["schema"],
        )
        with (TASK / "cases.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(CASES))
        for case, row in zip(CASES, rows):
            self.assertTrue(set(case["attrs"]) <= attrs)
            self.assertEqual(int(row["case_id"]), case["case_id"])
            for field in ("input_shape", "dtype", "attrs", "value_range"):
                self.assertEqual(json.loads(row[field]), case[field])
            self.assertEqual(row["note"], case["note"])
        for case in CASES[:20]:
            self.assertEqual(case["attrs"]["world_size"], 8)
            self.assertNotIn("mc2_accuracy_mode", case["attrs"])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TestA5GmmNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_golden()

    def rank_inputs(self, case):
        return [
            self.module.mc2_make_rank_inputs(
                {"rank": rank, "world_size": 2, "device": "cpu"}, payload(case)
            )
            for rank in range(2)
        ]

    def oracle_messages(self, all_inputs, attrs, cascade):
        """Construct each message directly from its source expert and destination rows."""
        counts, experts = attrs["token_counts"], attrs["e"]
        messages = []
        for source in range(2):
            tensors = all_inputs[source]
            weight = tensors["gmm_weight"]
            if attrs["trans_gmm_weight"]:
                weight = weight.transpose(1, 2)
            blocks = [[], []]
            offset = 0
            for expert in range(experts):
                for destination in range(2):
                    rows = counts[source][destination][expert]
                    value = (
                        tensors["gmm_x"][offset : offset + rows].float()
                        @ weight[expert].float()
                    )
                    if cascade:
                        value = value.to(tensors["gmm_x"].dtype)
                    blocks[destination].append(value)
                    offset += rows
            messages.append([torch.cat(parts, dim=0) for parts in blocks])
        return messages

    def context(self, rank, messages):
        def exchange(output, input, output_split_sizes, input_split_sizes):
            self.assertEqual(
                input_split_sizes, [len(message) for message in messages[rank]]
            )
            self.assertEqual(
                output_split_sizes, [len(messages[source][rank]) for source in range(2)]
            )
            # Different GEMM row blocking may change FP32 accumulation or the final half ULP.
            rtol = (
                1e-5
                if input.dtype == torch.float32
                else 2 * torch.finfo(input.dtype).eps
            )
            torch.testing.assert_close(
                input, torch.cat(messages[rank], dim=0), rtol=rtol, atol=5e-5
            )
            output.copy_(
                torch.cat([messages[source][rank] for source in range(2)], dim=0)
            )

        return {
            "rank": rank,
            "world_size": 2,
            "dist": types.SimpleNamespace(all_to_all_single=exchange),
        }

    def test_all_twenty_cpu_cases_route_distinct_experts_and_ranks(self):
        for case in A5:
            with self.subTest(case_id=case["case_id"]):
                tensors, attrs = self.rank_inputs(case), case["attrs"]
                messages = self.oracle_messages(tensors, attrs, cascade=False)
                self.assertFalse(torch.equal(tensors[0]["gmm_x"], tensors[1]["gmm_x"]))
                self.assertFalse(
                    torch.equal(tensors[0]["gmm_weight"], tensors[1]["gmm_weight"])
                )
                for rank in range(2):
                    y, shared = self.module.mc2_cpu_golden(
                        self.context(rank, messages), tensors[rank], attrs
                    )
                    self.assertEqual(y.dtype, torch.float32)
                    torch.testing.assert_close(
                        y,
                        torch.cat([messages[s][rank] for s in range(2)], dim=0),
                        rtol=0,
                        atol=0,
                    )
                    if attrs["mm_out"]:
                        weight = tensors[rank]["mm_weight"]
                        weight = weight.t() if attrs["trans_mm_weight"] else weight
                        torch.testing.assert_close(
                            shared, tensors[rank]["mm_x"].float() @ weight.float()
                        )
                        self.assertEqual(shared.dtype, torch.float32)
                    else:
                        self.assertIsNone(shared)

    def test_npu_cascade_preserves_output_dtype_and_cpu_does_not_round(self):
        for case in A5[:4]:
            with self.subTest(case_id=case["case_id"]):
                tensors, attrs = self.rank_inputs(case), case["attrs"]
                messages = self.oracle_messages(tensors, attrs, cascade=True)
                fp32_messages = self.oracle_messages(tensors, attrs, cascade=False)
                for rank in range(2):
                    y, shared = self.module.mc2_distributed_golden(
                        self.context(rank, messages), tensors[rank], attrs
                    )
                    cpu, _ = self.module.mc2_cpu_golden(
                        self.context(rank, fp32_messages), tensors[rank], attrs
                    )
                    self.assertEqual(y.dtype, tensors[rank]["gmm_x"].dtype)
                    self.assertFalse(torch.equal(cpu, cpu.to(y.dtype).float()))
                    if shared is not None:
                        self.assertEqual(shared.dtype, tensors[rank]["mm_x"].dtype)

    def test_candidate_parameter_order_counts_and_optional_outputs(self):
        case = A5[3]
        tensors = self.rank_inputs(case)[1]
        sentinel = object()

        def candidate(
            x, weight, hcomm_info, world_size, send_counts, recv_counts, **kwargs
        ):
            self.assertIs(x, tensors["gmm_x"])
            self.assertIs(weight, tensors["gmm_weight"])
            self.assertEqual((hcomm_info, world_size), ("a5_comm", 2))
            self.assertEqual(send_counts, tensors["send_counts"])
            self.assertEqual(recv_counts, tensors["recv_counts"])
            self.assertIs(kwargs["mm_x"], tensors["mm_x"])
            self.assertEqual(
                kwargs["trans_gmm_weight"], case["attrs"]["trans_gmm_weight"]
            )
            return sentinel

        result = self.module.mc2_call_candidate(
            candidate,
            {"hcomm_info": "a5_comm", "world_size": 2},
            tensors,
            case["attrs"],
        )
        self.assertIs(result, sentinel)

    def test_known_asymmetric_routing_and_zero_count(self):
        # C[0] sends three rows to rank 0 and one to rank 1; C[1] reverses this.
        attrs = {"mc2_accuracy_mode": "three_way", "e": 2, "trans_gmm_weight": False}
        inputs = [
            {
                "gmm_x": torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
                "gmm_weight": torch.tensor([[[2.0]], [[10.0]]]),
                "send_counts": [1, 2, 0, 1],
                "recv_counts": [1, 2, 1, 0],
                "group_list": [1, 3],
                "input_splits": [3, 1],
                "output_splits": [3, 1],
            },
            {
                "gmm_x": torch.tensor([[5.0], [6.0], [7.0], [8.0]]),
                "gmm_weight": torch.tensor([[[3.0]], [[20.0]]]),
                "send_counts": [1, 0, 2, 1],
                "recv_counts": [0, 1, 2, 1],
                "group_list": [3, 1],
                "input_splits": [1, 3],
                "output_splits": [1, 3],
            },
        ]
        messages = [
            [torch.tensor([[2.0], [20.0], [30.0]]), torch.tensor([[40.0]])],
            [torch.tensor([[15.0]]), torch.tensor([[18.0], [21.0], [160.0]])],
        ]
        for rank, expected in enumerate(
            ([2.0, 20.0, 30.0, 15.0], [40.0, 18.0, 21.0, 160.0])
        ):
            y, shared = self.module.mc2_cpu_golden(
                self.context(rank, messages), inputs[rank], attrs
            )
            torch.testing.assert_close(y[:, 0], torch.tensor(expected))
            self.assertIsNone(shared)


if __name__ == "__main__":
    unittest.main()
