# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Contract checks for the MC2 tasks in kernel_bench (no NPU required)."""

import ast
import csv
import importlib.util
import json
import math
from pathlib import Path
import re
import runpy
import sys
import types
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[2]
LEVEL4 = ROOT / "bench_lab" / "kernel_bench" / "level4"
OPS = (
    "allto_all_matmul",
    "matmul_allto_all",
    "matmul_reduce_scatter_v2",
    "matmul_all_reduce",
    "all_gather_matmul_v2",
    "all_gather_matmul",
    "grouped_mat_mul_allto_allv",
    "allto_allv_grouped_mat_mul",
    "quant_all_reduce",
    "quant_reduce_scatter",
    "quant_grouped_mat_mul_allto_allv",
    "allto_allv_quant_grouped_mat_mul",
)
GMM_OPS = {
    "grouped_mat_mul_allto_allv",
    "allto_allv_grouped_mat_mul",
    "quant_grouped_mat_mul_allto_allv",
    "allto_allv_quant_grouped_mat_mul",
}
QUANT_OPS = {
    "quant_all_reduce",
    "quant_reduce_scatter",
    "quant_grouped_mat_mul_allto_allv",
}


def read_cases(op):
    return yaml.safe_load((LEVEL4 / op / "cases.yaml").read_text(encoding="utf-8"))[
        "cases"
    ]


class TestMC2TaskContracts(unittest.TestCase):
    def test_all_a5_cases_enable_three_way(self):
        count = 0
        for op in OPS:
            for case in read_cases(op):
                attrs = case["attrs"]
                if "A5" in attrs.get("platform", ""):
                    count += 1
                    self.assertEqual(attrs.get("world_size"), 2, (op, case["case_id"]))
                    self.assertEqual(
                        attrs.get("mc2_accuracy_mode"),
                        "three_way",
                        (op, case["case_id"]),
                    )
                    if op in OPS[:2]:
                        self.assertEqual(attrs.get("mc2_compare_dtype"), "float32")
                else:
                    self.assertNotIn("mc2_accuracy_mode", attrs)
            tree = ast.parse((LEVEL4 / op / "golden.py").read_text(encoding="utf-8"))
            self.assertIn(
                "mc2_cpu_golden",
                [node.name for node in tree.body if isinstance(node, ast.FunctionDef)],
            )
        self.assertEqual(count, 300)

    def test_definition_and_csv_consistency(self):
        for op in OPS:
            with self.subTest(operator=op):
                directory = LEVEL4 / op
                for name in (
                    "proto.yaml",
                    "cases.yaml",
                    "cases.csv",
                    "golden.py",
                    "desc.md",
                ):
                    self.assertTrue((directory / name).is_file(), name)
                proto = yaml.safe_load(
                    (directory / "proto.yaml").read_text(encoding="utf-8")
                )["operator"]
                cases = read_cases(op)
                expected_count = (
                    20
                    if op in QUANT_OPS
                    or op
                    in (
                        "allto_allv_grouped_mat_mul",
                        "allto_allv_quant_grouped_mat_mul",
                    )
                    else 40
                )
                if op == "matmul_all_reduce":
                    expected_count = 60
                self.assertEqual(len(cases), expected_count)
                self.assertEqual(
                    [c["case_id"] for c in cases], list(range(1, expected_count + 1))
                )
                self.assertEqual(proto["difficulty"], "L4")
                self.assertIn("950DT", proto["platform"])
                declared_attrs = {a["name"] for a in proto["attrs"]}
                with (directory / "cases.csv").open(
                    encoding="utf-8", newline=""
                ) as stream:
                    csv_rows = list(csv.DictReader(stream))
                self.assertEqual(len(csv_rows), len(cases))
                for case, row in zip(cases, csv_rows):
                    self.assertEqual(case["operator"], proto["name"])
                    self.assertTrue(set(case["attrs"]) <= declared_attrs)
                    self.assertNotIn("baseline_perf_us", case)
                    self.assertNotIn("t_hw_us", case)
                    for field in ("input_shape", "dtype", "value_range"):
                        self.assertLessEqual(len(case[field]), len(proto["inputs"]))
                        for missing in proto["inputs"][len(case[field]) :]:
                            self.assertTrue(missing.get("optional", False))
                    for shape, dtype, field in zip(
                        case["input_shape"], case["dtype"], proto["inputs"]
                    ):
                        if shape is None:
                            self.assertTrue(field.get("optional", False))
                        else:
                            self.assertIn(dtype, field["dtype"])
                    self.assertEqual(set(row), set(case))
                    for field, value in case.items():
                        if isinstance(value, (dict, list)):
                            self.assertEqual(json.loads(row[field]), value)
                        elif field == "case_id":
                            self.assertEqual(int(row[field]), value)
                        else:
                            self.assertEqual(row[field], str(value))

    def test_communication_shape_constraints(self):
        for op in OPS:
            for case in read_cases(op):
                with self.subTest(operator=op, case_id=case["case_id"]):
                    shapes, attrs = case["input_shape"], case["attrs"]
                    world_size = attrs["world_size"]
                    self.assertIn(world_size, (2, 4, 8))
                    if op in GMM_OPS:
                        if "A5" not in attrs.get("platform", ""):
                            continue
                        self.assertEqual(attrs["supported_devices"], ["950dt"])
                        counts = attrs["token_counts"]
                        e = attrs["e"]
                        self.assertEqual(len(counts), world_size)
                        for row in counts:
                            self.assertEqual(len(row), world_size)
                            for edge in row:
                                self.assertEqual(len(edge), e)
                                self.assertTrue(
                                    all(type(n) is int and n >= 0 for n in edge)
                                )
                        sends = [sum(sum(edge) for edge in row) for row in counts]
                        receives = [
                            sum(sum(counts[s][r]) for s in range(world_size))
                            for r in range(world_size)
                        ]
                        self.assertEqual(sends, [shapes[0][0]] * world_size)
                        self.assertEqual(receives, sends)
                        self.assertEqual(shapes[1][0], e)
                        self.assertEqual(
                            shapes[1][2 if attrs["trans_gmm_weight"] else 1],
                            shapes[0][1],
                        )
                        self.assertEqual(shapes[2:4], [[world_size * e]] * 2)
                    elif op in ("quant_all_reduce", "quant_reduce_scatter"):
                        x, scales = shapes
                        h = x[-1]
                        if attrs["scales_dtype"] == "fp8_e8m0":
                            self.assertEqual(h % 64, 0)
                            self.assertEqual(scales, x[:-1] + [h // 64, 2])
                        else:
                            self.assertEqual(h % 128, 0)
                            self.assertEqual(scales, x[:-1] + [h // 128])
                        self.assertEqual(math.prod(x[:-1]) % world_size, 0)
                        self.assertTrue(attrs["mc2_isolated_candidate_group"])
                        self.assertEqual(attrs["supported_devices"], ["950dt"])
                    else:
                        m, k = math.prod(shapes[0][:-1]), shapes[0][-1]
                        transposed = attrs.get(
                            "is_trans_b", attrs.get("is_trans_x2", False)
                        )
                        weight_k = shapes[1][1 if transposed else 0]
                        n = shapes[1][0 if transposed else 1]
                        self.assertEqual(
                            weight_k, k * world_size if op == "allto_all_matmul" else k
                        )
                        if op in OPS[:3]:
                            self.assertEqual(
                                (n if op == "matmul_allto_all" else m) % world_size, 0
                            )

    def test_doc_describes_public_operator_contract(self):
        for op in OPS:
            with self.subTest(operator=op):
                directory = LEVEL4 / op
                golden = (directory / "golden.py").read_text(encoding="utf-8").strip()
                doc = (directory / "desc.md").read_text(encoding="utf-8")
                operator = yaml.safe_load(
                    (directory / "proto.yaml").read_text(encoding="utf-8")
                )["operator"]
                self.assertTrue(doc.startswith(f"# {operator['name']} 算子 API 描述"))
                self.assertEqual(
                    re.findall(r"^## .+$", doc, re.MULTILINE),
                    [
                        "## 1. 算子简介",
                        "## 2. 算子定义",
                        "## 3. 接口规范",
                        "## 4. 精度要求",
                        "## 5. 标准 Golden 代码",
                        "## 6. 算子调用示例",
                    ],
                )
                interface = doc.split("## 3. 接口规范", 1)[1].split(
                    "## 4. 精度要求", 1
                )[0]
                self.assertIn("### 规则与约束", interface)
                blocks = re.findall(r"```python\n(.*?)\n```", doc, re.DOTALL)
                self.assertEqual(len(blocks), 3)
                # Check argument names, order and defaults against the public
                # schema without requiring a copy of the test implementation.
                signature = operator["schema"].split(" -> ")[0]
                self.assertIn(
                    re.sub(r"\s+", "", "cann_bench." + signature),
                    re.sub(r"\s+", "", blocks[0]),
                )
                arguments = signature.split("(", 1)[1].rsplit(")", 1)[0]
                for argument in arguments.split(","):
                    name = argument.strip().split()[1].split("=", 1)[0]
                    self.assertIn(f"| `{name}` |", doc)
                # Publish the entire implementation, including input builders,
                # the cascade, and the independent CPU reference hooks.
                reference_section = doc.split("## 5. 标准 Golden 代码", 1)[1].split(
                    "## 6. 算子调用示例", 1
                )[0]
                reference_blocks = re.findall(
                    r"```python\n(.*?)\n```", reference_section, re.DOTALL
                )
                self.assertEqual(len(reference_blocks), 1)
                reference = reference_blocks[0]
                source_functions = {
                    node.name: node
                    for node in ast.parse(golden).body
                    if isinstance(node, ast.FunctionDef)
                }
                reference_functions = {
                    node.name: node
                    for node in ast.parse(reference).body
                    if isinstance(node, ast.FunctionDef)
                }
                self.assertEqual(reference.strip(), golden)
                self.assertEqual(set(reference_functions), set(source_functions))
                for name, node in reference_functions.items():
                    self.assertEqual(
                        ast.get_source_segment(reference, node),
                        ast.get_source_segment(golden, source_functions[name]),
                        name,
                    )
                torch_stub = types.ModuleType("torch")
                torch_stub.Tensor = object
                torch_stub.device = object
                with patch.dict(sys.modules, {"torch": torch_stub}):
                    exec(compile(reference, str(directory / "desc.md"), "exec"), {})

                example_section = doc.split("## 6. 算子调用示例", 1)[1]
                example_blocks = re.findall(
                    r"```python\n(.*?)\n```", example_section, re.DOTALL
                )
                self.assertEqual(len(example_blocks), 1)
                example = ast.parse(example_blocks[0])
                public_name = signature.split("(", 1)[0]
                self.assertTrue(
                    any(
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "cann_bench"
                        and node.func.attr == public_name
                        for node in ast.walk(example)
                    )
                )

    def test_doc_excludes_evaluation_workflow(self):
        forbidden = (
            "当前测评集",
            "新增 A5",
            "用例 1",
            "用例 21",
            "case_id",
            "--case-id-range",
            "run_evaluation.sh",
            "T_HW",
            "```bash",
        )
        for op in OPS:
            with self.subTest(operator=op):
                doc = (LEVEL4 / op / "desc.md").read_text(encoding="utf-8")
                # Verbatim reference code may retain source comments and
                # attribute names; the API prose must not describe case sets.
                prose = re.sub(r"```.*?```", "", doc, flags=re.DOTALL)
                for text in forbidden:
                    self.assertNotIn(text, prose)
                self.assertNotRegex(prose, r"\d+\s*[-–]\s*\d+\s*号(?:为|用例)")
                self.assertNotRegex(prose, r"\d+\s*(?:条|个|组)\s*(?:测试)?用例")

    def test_a3_a5_case_selection_is_disjoint(self):
        cases = read_cases("matmul_reduce_scatter_v2")
        a3 = [c for c in cases if c["case_id"] in range(1, 21)]
        a5 = [c for c in cases if c["case_id"] in range(21, 41)]
        self.assertEqual(len(a3), 20)
        self.assertEqual(len(a5), 20)
        self.assertTrue(all(c["attrs"]["platform"] == "Atlas A3" for c in a3))
        self.assertTrue(all("950PR" in c["attrs"]["platform"] for c in a5))
        self.assertTrue(all(c["dtype"][:2] == ["int8", "int8"] for c in a3))
        self.assertTrue(all(c["dtype"][0] in ("float16", "bfloat16") for c in a5))
        self.assertTrue(all(c["input_shape"][2:] == [None, None] for c in a5))

    def test_task_paths_resolve_to_existing_suite(self):
        path = ROOT / "src/kernel_eval/utils/path_resolver.py"
        spec = importlib.util.spec_from_file_location("mc2_path_resolver", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for op in OPS:
            bench, prefix = module.resolve_task_dir(
                "bench_lab/kernel_bench/level4/" + op,
                ROOT,
                check_operator_dir=False,
            )
            self.assertEqual(Path(bench), ROOT / "bench_lab/kernel_bench")
            self.assertEqual(Path(prefix).as_posix(), "level4/" + op)


class TestA5QuantizedAdapters(unittest.TestCase):
    def check_adapter(self, op):
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object
        torch_stub.device = object
        with patch.dict(sys.modules, {"torch": torch_stub}):
            module = runpy.run_path(str(LEVEL4 / op / "golden.py"))
        calls = []

        def tensor(shape, dtype, value_range, seed, device):
            if shape is None:
                return None
            record = {"shape": shape, "dtype": dtype, "seed": seed}
            calls.append(record)
            return record

        fn = module["mc2_make_rank_inputs"]
        fn.__globals__["_make_tensor"] = tensor
        for case in read_cases(op):
            if case["attrs"].get("mc2_accuracy_mode") != "three_way":
                continue
            if op == "matmul_all_reduce" and not case["attrs"].get("quantized", False):
                continue
            with self.subTest(op=op, case_id=case["case_id"]):
                payload = dict(
                    input_shapes=case["input_shape"],
                    dtypes=case["dtype"],
                    value_ranges=case["value_range"],
                    attrs=case["attrs"],
                )
                ctx = dict(
                    rank=1,
                    device="npu:1",
                    world_size=case["attrs"]["world_size"],
                    hcomm_info="test",
                )
                inputs = fn(ctx, payload)
                args, kw = module["mc2_call_candidate"](
                    lambda *a, **kw: (a, kw), ctx, inputs, case["attrs"]
                )
                self.assertIs(args[0], inputs["x1"])
                self.assertEqual(args[2:4], ("test", ctx["world_size"]))
                self.assertEqual(kw["out_dtype"], case["attrs"]["out_dtype"])
                if op == "matmul_all_reduce":
                    for index, key in (
                        (3, "dequant_scale"),
                        (4, "pertoken_scale"),
                        (5, "x3"),
                    ):
                        self.assertIs(kw[key], inputs[key])
                        if inputs[key] is not None:
                            self.assertEqual(
                                inputs[key]["shape"], case["input_shape"][index]
                            )
                    self.assertEqual(inputs["x1"]["seed"], case["attrs"]["seed"] + 1)
                else:
                    for index, key in ((2, "x1_scale"), (3, "x2_scale"), (4, "bias")):
                        self.assertIs(kw[key], inputs[key])
                        if inputs[key] is not None:
                            self.assertEqual(
                                inputs[key]["shape"], case["input_shape"][index]
                            )
                    self.assertEqual(kw["group_sizes"], [])
                    self.assertEqual(kw["block_size"], 0)

    def test_int8_allreduce_candidate_optional_inputs(self):
        self.check_adapter("matmul_all_reduce")

    def test_fp8_allgather_scale_and_bias_positions(self):
        self.check_adapter("all_gather_matmul_v2")


class TestMatmulReduceScatterInputAdapter(unittest.TestCase):
    def setUp(self):
        # Only test adapter control flow; this stub does not simulate tensor math or HCCL.
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object
        torch_stub.device = object
        with patch.dict(sys.modules, {"torch": torch_stub}):
            self.golden = runpy.run_path(
                str(LEVEL4 / "matmul_reduce_scatter_v2/golden.py")
            )
        self.calls = []

        def make_tensor(shape, dtype, value_range, seed, device):
            record = {"shape": shape, "dtype": dtype, "seed": seed}
            self.calls.append(record)
            return record

        self.golden["mc2_make_rank_inputs"].__globals__["_make_tensor"] = make_tensor

    def make_inputs(self, case, rank):
        payload = {
            "input_shapes": case["input_shape"],
            "dtypes": case["dtype"],
            "value_ranges": case["value_range"],
            "attrs": case["attrs"],
        }
        ctx = {
            "rank": rank,
            "device": "cpu",
            "world_size": case["attrs"]["world_size"],
            "hcomm_info": "test",
        }
        inputs = self.golden["mc2_make_rank_inputs"](ctx, payload)
        args = self.golden["mc2_call_candidate"](
            lambda *a, **kw: (a, kw), ctx, inputs, case["attrs"]
        )
        return inputs, args

    def test_original_a3_scale_and_seed_path(self):
        case = read_cases("matmul_reduce_scatter_v2")[0]
        inputs, (_, kwargs) = self.make_inputs(case, rank=1)
        seed = case["attrs"]["seed"]
        self.assertEqual(
            [c["seed"] for c in self.calls], [seed, seed, seed * 11, seed * 13]
        )
        self.assertIsNotNone(inputs["x1_scale"])
        self.assertIsNotNone(inputs["x2_scale"])
        self.assertEqual(kwargs["block_size"], 128)
        self.assertEqual(kwargs["x1_dtype"], "int8")

    def test_a5_omits_scales_and_uses_rank_seed(self):
        for case in read_cases("matmul_reduce_scatter_v2")[20:]:
            with self.subTest(case_id=case["case_id"]):
                self.calls.clear()
                inputs, (_, kwargs) = self.make_inputs(case, rank=1)
                seed = case["attrs"]["seed"] + 1
                self.assertEqual([c["seed"] for c in self.calls], [seed, seed])
                self.assertIsNone(inputs["x1_scale"])
                self.assertIsNone(inputs["x2_scale"])
                self.assertIsNone(kwargs["x1_scale"])
                self.assertIsNone(kwargs["x2_scale"])
                self.assertEqual(kwargs["block_size"], 0)


if __name__ == "__main__":
    unittest.main()
