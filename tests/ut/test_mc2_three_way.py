# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MC2 numeric and adapter regression tests; runnable without torch/NPU."""

import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/kernel_eval"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Isolate dependencies so these tests exercise the actual numeric/report code,
# not a mocked numeric checker, on developer hosts without Ascend/PyTorch.
for package in ("_mc2_unit", "_mc2_unit.base", "_mc2_unit.utils", "_mc2_unit.checkers"):
    module = types.ModuleType(package)
    module.__path__ = []
    sys.modules[package] = module
base = load("_mc2_unit.base.result", SOURCE / "base/result.py")
metrics = load("_mc2_unit.utils.mc2_compare", SOURCE / "utils/mc2_compare.py")
checker = load(
    "_mc2_unit.checkers.mc2_three_way_checker",
    SOURCE / "checkers/mc2_three_way_checker.py",
)


class TestThreeWayMetrics(unittest.TestCase):
    def test_same_error_passes_and_inputs_are_immutable(self):
        g = np.array([1, 2, 3, 0], dtype=np.float32)
        b = g + np.float32(0.125)
        snapshot = g.copy(), b.copy()
        result = metrics.compare_three_way(b, b, g, "float32")
        self.assertTrue(result["passed"])
        np.testing.assert_array_equal(g, snapshot[0])
        np.testing.assert_array_equal(b, snapshot[1])
        self.assertEqual(result["limits"]["diff_big_max"], 1.25)

    def test_reference_zero_accepts_one_ulp_but_not_material_error(self):
        g = np.ones(10, np.float32)
        self.assertTrue(metrics.compare_three_way(g, g, g, "float32")["passed"])
        a = g.copy()
        a[0] = np.nextafter(a[0], np.float32(2))
        result = metrics.compare_three_way(a, g, g, "float32")
        self.assertTrue(result["passed"])
        self.assertEqual(result["accepted_by"], "finite_tolerance")
        self.assertEqual(result["limits"]["diff_big_max"], 0)
        self.assertFalse(metrics.compare_three_way(g + 1e-5, g, g, "float32")["passed"])

    def test_tolerance_is_elementwise_and_dtype_aware(self):
        for dtype, epsilon in (
            ("float16", 2**-10),
            ("bfloat16", 2**-7),
            ("float32", 2**-23),
        ):
            g = np.ones(10000, np.float32)
            a = g.copy()
            a[::100] += epsilon
            self.assertTrue(metrics.compare_three_way(a, g, g, dtype)["passed"])
            a[0] += 10 * epsilon
            self.assertFalse(metrics.compare_three_way(a, g, g, dtype)["passed"])
        g = np.array([1e10, 1], np.float32)
        self.assertFalse(
            metrics.compare_three_way(g + [0, 1e-5], g, g, "float32")["passed"]
        )

    def test_near_zero_allowance_and_common_small_count_gate(self):
        g = np.zeros(10, np.float32)
        self.assertTrue(metrics.compare_three_way(g + 1e-9, g, g, "float32")["passed"])
        self.assertFalse(metrics.compare_three_way(g + 2e-9, g, g, "float32")["passed"])
        g = np.full(100000, 0.0009, np.float32)
        result = metrics.compare_three_way(g + 1.01e-5, g, g, "float16")
        self.assertFalse(result["passed"])
        self.assertIn("err_small_num", result["failed_metrics"])

    def test_two_times_boundary_is_inclusive(self):
        g = np.ones(16, np.float32)
        b = g + 0.125
        self.assertTrue(metrics.compare_three_way(g + 0.25, b, g, "float32")["passed"])
        result = metrics.compare_three_way(g + 0.5, b, g, "float32")
        self.assertFalse(result["passed"])
        self.assertIn("diff_rmse", result["failed_metrics"])
        self.assertIn("diff_big_avg", result["failed_metrics"])

    def test_maximum_error_uses_ten_times(self):
        g = np.ones(1024, np.float32)
        b = g + 1
        a = g.copy()
        a[0] += 10
        self.assertTrue(metrics.compare_three_way(a, b, g, "float32")["passed"])
        a[0] += 1
        result = metrics.compare_three_way(a, b, g, "float32")
        self.assertEqual(result["failed_metrics"], ["diff_big_max"])

    def test_small_count_gate_is_not_baseline_error_count(self):
        for size, passed in ((99999, True), (100000, False)):
            with self.subTest(size=size):
                g = np.zeros(size, np.float32)
                b, a = g.copy(), g.copy()
                b[:5] = 2e-9
                a[:11] = 2e-9
                result = metrics.compare_three_way(a, b, g, "float32")
                self.assertEqual(result["passed"], passed)
                self.assertEqual("err_small_num" in result["limits"], size >= 100000)

    def test_dtype_dependent_small_value_threshold(self):
        for dtype, small, atol in (
            ("float16", 1e-3, 1e-5),
            ("bfloat16", 1e-3, 1e-5),
            ("float32", 1e-6, 1e-9),
        ):
            m = metrics.error_metrics([0], [0], dtype)
            self.assertEqual((m["small_value"], m["small_value_atol"]), (small, atol))

    def test_exact_small_boundary_is_in_both_error_masks(self):
        g = np.array([1e-3], np.float32)
        m = metrics.error_metrics(g + 1, g, "float16")
        self.assertEqual(m["total_big_num"], 1)
        self.assertEqual(m["total_small_num"], 0)
        self.assertEqual(m["err_small_num"], 1)

    def test_empty_and_all_small_domains_are_finite(self):
        for g in (np.array([], np.float32), np.zeros(4, np.float32)):
            result = metrics.compare_three_way(g, g, g, "float32")
            self.assertTrue(result["passed"])
            json.dumps(result, allow_nan=False)

    def test_nan_and_signed_inf_mismatch_fail_closed(self):
        for a, b, g in (
            ([np.nan], [1], [1]),
            ([1], [np.nan], [1]),
            ([np.inf], [1], [1]),
            ([-np.inf], [np.inf], [np.inf]),
        ):
            self.assertFalse(metrics.compare_three_way(a, b, g, "float32")["passed"])

    def test_reference_clamped_to_output_range(self):
        result = metrics.compare_three_way([np.inf], [np.inf], [70000], "float16")
        self.assertTrue(result["passed"])
        self.assertEqual(result["candidate_metrics"]["num_total_inf"], 1)

    def test_metric_overflow_cannot_silently_pass(self):
        result = metrics.compare_three_way([3e38], [-3e38], [0], "float32")
        self.assertFalse(result["passed"])
        json.dumps(result, allow_nan=False)

    def test_shapes_and_dtype_are_validated(self):
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            metrics.compare_three_way([1, 2], [1], [1], "float32")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            metrics.compare_three_way([1], [1], [1], "int8")


class FakeTensor:
    """Only the CPU output adapter surface; no tensor math is emulated."""

    def __init__(self, values, dtype="float32"):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape, self.dtype = self.values.shape, dtype

    def float(self):
        return self

    def numpy(self):
        return self.values


class TestThreeWayOutputContract(unittest.TestCase):
    def setUp(self):
        self.torch = types.SimpleNamespace(
            Tensor=FakeTensor,
            float32="float32",
            isfinite=lambda tensor: np.isfinite(tensor.values),
        )
        self.patcher = patch.dict(sys.modules, {"torch": self.torch})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        # Tensor contract tests stay runnable without torch; real reference
        # health and dtype/cancellation behavior are covered separately.
        health = patch.object(
            checker, "_reference_health", return_value={"passed": True, "error_msg": ""}
        )
        health.start()
        self.addCleanup(health.stop)

    def compare(self, a, b, g):
        return checker.compare_mc2_outputs(a, b, g, rank=3)

    def test_optional_and_required_none(self):
        t = FakeTensor([1])
        self.assertFalse(self.compare(None, None, None)[0].passed)
        self.assertTrue(
            all(r.passed for r in self.compare((t, None), (t, None), (t, None)))
        )
        self.assertFalse(all(r.passed for r in self.compare((t, None), (t, t), (t, t))))
        self.assertFalse(
            all(r.passed for r in self.compare((t, t), (t, None), (t, None)))
        )

    def test_bad_shape_count_and_dtype(self):
        t = FakeTensor([1])
        variants = (
            ([t], [t, t], [t, t]),
            (FakeTensor([1, 2]), t, t),
            (FakeTensor([1], "float16"), t, t),
        )
        for args in variants:
            result = self.compare(*args)
            self.assertFalse(all(r.passed for r in result))
            self.assertEqual(
                result[0].metadata["failure_type"], "compile_runtime_error"
            )
        invalid = self.compare(t, t, FakeTensor([1], "float16"))[0]
        self.assertEqual(invalid.metadata["failure_type"], "reference_invalid")

    def test_multi_output_failure_is_not_lost(self):
        t = FakeTensor([1])
        results = self.compare((t, FakeTensor([2])), (t, t), (t, t))
        self.assertEqual([r.passed for r in results], [True, False])

    def test_registered_results_round_trip_without_mere_label(self):
        t = FakeTensor([1])
        result = self.compare(t, t, t)[0]
        cls = base.get_output_result_cls("mc2_three_way")
        restored = cls.from_dict(
            json.loads(json.dumps(result.to_dict(), allow_nan=False))
        )
        self.assertEqual(restored.to_dict(), result.to_dict())
        self.assertEqual(restored.metadata["rank"], 3)
        self.assertIn("MC2 three-way", restored.format_summary())
        self.assertNotIn("MERE", restored.format_summary())
        summary = base.AccuracyResult(
            True, output_results=[restored], metadata={"checker_name": "mc2_three_way"}
        ).format_summary()
        self.assertIn("per-output limits", summary)
        self.assertNotIn("threshold=0", summary)
        text = (SOURCE / "eval/results.py").read_text(encoding="utf-8")
        result_class = next(
            node
            for node in ast.parse(text).body
            if isinstance(node, ast.ClassDef) and node.name == "EvalCaseResult"
        )
        from_dict = next(
            node
            for node in result_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "from_dict"
        )
        checker_maps = [
            ast.literal_eval(node.value)
            for node in ast.walk(from_dict)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_checker_modules"
                for target in node.targets
            )
        ]
        self.assertEqual(len(checker_maps), 1)
        self.assertEqual(
            checker_maps[0]["mc2_three_way"], "..checkers.mc2_three_way_checker"
        )


class TestMC2ProcessIsolation(unittest.TestCase):
    def test_cpu_alltoall_uses_each_actual_source_rank(self):
        class Array:
            def __init__(self, values):
                self.values = np.asarray(values)
                self.shape = self.values.shape
                self.dtype = self.values.dtype

            def clone(self):
                return Array(self.values.copy())

            def contiguous(self):
                return self

            def copy_(self, other):
                self.values[...] = other.values

            def __getitem__(self, index):
                return Array(self.values[index])

        sources = [
            Array(np.array([[[1, 2]], [[3, 4]]], np.float32)),
            Array(np.array([[[10, 20]], [[30, 40]]], np.float32)),
        ]
        broadcasts = []

        def broadcast(payload, src, group):
            self.assertEqual(group, "cpu-group")
            broadcasts.append(src)
            payload.copy_(sources[src])

        def broadcast_metadata(objects, src, group):
            self.assertEqual(group, "cpu-group")
            objects[0] = {
                "input_shape": sources[src].shape,
                "output_shape": sources[src].shape,
                "input_dtype": sources[src].dtype,
                "output_dtype": sources[src].dtype,
                "send": [1, 1],
                "recv": [1, 1],
            }

        dist = types.SimpleNamespace(
            broadcast=broadcast,
            broadcast_object_list=broadcast_metadata,
            ReduceOp=types.SimpleNamespace(SUM="sum"),
        )
        stub = types.SimpleNamespace(
            empty=lambda shape, dtype, device: Array(np.empty(shape, dtype=dtype))
        )
        with patch.dict(sys.modules, {"torch": stub}):
            module = load(
                "_mc2_collective_test", SOURCE / "eval/mc2_cpu_collectives.py"
            )
        output = Array(np.zeros_like(sources[0].values))
        cpu = module.CPUCollectives(dist, "cpu-group", rank=1, world_size=2)
        cpu.all_to_all_single(output, sources[1])
        self.assertEqual(broadcasts, [0, 1])
        np.testing.assert_array_equal(output.values, [[[3, 4]], [[30, 40]]])

    def test_mc2_keeps_visible_group_and_regular_keeps_single_card(self):
        tree = ast.parse((SOURCE / "eval/process_pool.py").read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_build_env_for_task"
        )
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        module = ast.Module(body=[method], type_ignores=[])
        scope = {}
        exec(compile(ast.fix_missing_locations(module), "<env-builder>", "exec"), scope)
        build = scope["_build_env_for_task"]
        env = {
            "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7",
            "BENCH_DEVICE_VISIBILITY": "4,5,6,7",
        }
        for mc2, expected in ((True, "4,5,6,7"), (False, "5")):
            task = types.SimpleNamespace(
                device_id=1,
                cases=[types.SimpleNamespace(attrs={"mc2_distributed": mc2})],
            )
            result = build(None, env, task)
            self.assertEqual(result["ASCEND_RT_VISIBLE_DEVICES"], expected)
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "4,5,6,7")

    def test_mc2_restores_physical_device_set_before_rank_spawn(self):
        tree = ast.parse(
            (SOURCE / "eval/mc2_distributed_runner.py").read_text(encoding="utf-8")
        )
        method = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_restore_mc2_visibility"
        )
        method.returns = None
        for arg in method.args.args:
            arg.annotation = None
        scope = {"os": os}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                "<mc2-visibility>",
                "exec",
            ),
            scope,
        )
        restore = scope["_restore_mc2_visibility"]
        for devices in ("2,3", "4,5,6,7", "8,9,10,11,12,13,14,15"):
            with (
                self.subTest(devices=devices),
                patch.dict(
                    os.environ,
                    {
                        "BENCH_DEVICE_VISIBILITY": devices,
                        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
                    },
                ),
            ):
                restore(2)
                self.assertEqual(os.environ["ASCEND_RT_VISIBLE_DEVICES"], devices)
        for devices in ("2", "2,2", "4,5,5,7", "2,x", "2,,3"):
            with (
                self.subTest(devices=devices),
                patch.dict(
                    os.environ,
                    {
                        "BENCH_DEVICE_VISIBILITY": devices,
                        "ASCEND_RT_VISIBLE_DEVICES": "0,1",
                    },
                ),
            ):
                with self.assertRaises(ValueError):
                    restore(2)
                self.assertEqual(os.environ["ASCEND_RT_VISIBLE_DEVICES"], "0,1")

    def test_reference_hook_and_cpu_group_are_not_optional_in_three_way(self):
        source = (SOURCE / "eval/mc2_distributed_runner.py").read_text(encoding="utf-8")
        self.assertIn("requires a task-local mc2_cpu_golden hook", source)
        run_rank = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_rank"
        )
        group_calls = [
            node.value
            for node in ast.walk(run_rank)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and any(
                isinstance(target, ast.Name) and target.id == "cpu_group"
                for target in node.targets
            )
        ]
        self.assertEqual(len(group_calls), 1)
        self.assertEqual(ast.unparse(group_calls[0].func), "dist.new_group")
        self.assertEqual(
            [
                ast.literal_eval(keyword.value)
                for keyword in group_calls[0].keywords
                if keyword.arg == "backend"
            ],
            ["gloo"],
        )
        self.assertLess(
            source.index("cpu_output = cpu_hook"),
            source.index("candidate_output = candidate_fn"),
        )
        self.assertLess(
            source.index("queue.get(timeout="), source.index("proc.join(timeout=")
        )


if __name__ == "__main__":
    unittest.main()
