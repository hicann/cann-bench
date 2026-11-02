#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Storage deduplication must not change reference object aliases."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("safetensors")

from kernel_eval.benches.cann_spec import CannCaseSpec
from kernel_eval.config import Config
from kernel_eval.data.tensor_cache import TensorCache, TensorCacheError
from kernel_eval.eval.evaluator import Evaluator


def test_reference_objects_are_independent_despite_content_dedup(tmp_path):
    case = SimpleNamespace(input_shapes=[[2]], dtypes=["int64"])
    value = torch.tensor([1, 2], dtype=torch.int64)
    other = value.clone()
    cache = TensorCache(tmp_path)
    # Preserve repeated references to each original object, including nested
    # output structures; equal but distinct objects must remain independent.
    path = cache.write({}, case, [value.clone()], (value, [other, value]), (other, value))
    loaded = cache.load({}, case)
    assert loaded.oracle[0] is loaded.oracle[1][1] is loaded.native[1]
    assert loaded.oracle[1][0] is loaded.native[0]
    assert loaded.oracle[0].data_ptr() != loaded.native[0].data_ptr()
    loaded.oracle[0].add_(10)
    assert torch.equal(loaded.native[0], other)
    assert torch.equal(loaded.native[1], value + 10)
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(manifest["files"]["references.safetensors"]["tensors"]) == 1
    assert torch.equal(cache.load({}, case).oracle[0], value)


@pytest.mark.parametrize("overlap", [True, False])
def test_distinct_reference_views_sharing_storage_are_rejected(tmp_path, overlap):
    case = SimpleNamespace(input_shapes=[[2]], dtypes=["int64"])
    value = torch.tensor([1, 2], dtype=torch.int64)
    cache = TensorCache(tmp_path)
    oracle, native = (value, value.view_as(value)) if overlap else (value[:1], value[1:])
    with pytest.raises(TensorCacheError, match="distinct reference tensors sharing storage"):
        cache.write({}, case, [value.clone()], oracle, native)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("precision", ["fp64_cpu", "native_cpu"])
@pytest.mark.parametrize("mode", ["prefer", "require"])
def test_inplace_get_output_preserves_uncached_checking(tmp_path, precision, mode):
    task = tmp_path / "tasks" / "level1" / "alias_clone"
    task.mkdir(parents=True)
    (task / "golden.py").write_text(
        "import torch\n"
        "def alias_clone(x: torch.Tensor) -> torch.Tensor:\n"
        "    return x.clone()\n"
        "def get_output(y):\n"
        "    return y.add_(1)\n", encoding="utf-8")
    (task / "proto.yaml").write_text(
        "operator:\n  name: AliasClone\n  difficulty: L1\n"
        "  inputs:\n  - name: x\n    dtype: [int64]\n"
        "  outputs:\n  - name: y\n    dtype: [int64]\n"
        "  schema: alias_clone(Tensor x) -> Tensor y\n", encoding="utf-8")
    (task / "cases.yaml").write_text("cases: []\n", encoding="utf-8")
    case = CannCaseSpec(case_id="level1/alias_clone_0", case_num=0,
                        operator="AliasClone", rel_path="level1/alias_clone",
                        input_shapes=[[2]], dtypes=["int64"], value_ranges=[[0, 5]], attrs={})
    config = Config(tasks_root=str(tmp_path / "tasks"), device_type="cpu",
                    enable_profiler=False, torch_op_guard_mode="off",
                    tensor_cache_dir=str(tmp_path / "bank"), tensor_cache_mode="off",
                    tensor_cache_tasks=[case.rel_path])
    evaluator = Evaluator(config)
    evaluator._cleanup_memory = lambda: None
    candidate = evaluator.golden_loader.get_golden_function(case.rel_path)
    with patch.object(evaluator.bench_config, "golden_precision", precision):
        original = evaluator.evaluate_case(case, ai_op_func=candidate)
        # native_cpu intentionally reuses one object for both reference roles.
        # An in-place hook therefore modifies it twice; preserve that existing
        # checking outcome rather than silently changing it when cache is on.
        assert original.success == (precision == "fp64_cpu"), original.error_msg
        evaluator.build_tensor_cache(case)
        config.tensor_cache_mode = mode
        with patch.object(evaluator.data_generator, "generate_input_tensors_from_case",
                          side_effect=AssertionError("generated instead of loading cache")), \
             patch.object(evaluator, "_oracle_reference_func", side_effect=AssertionError("oracle executed")), \
             patch.object(evaluator, "_bench_reference_func", side_effect=AssertionError("native executed")):
            cached = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert cached.success == original.success, cached.error_msg
    assert cached.accuracy_result is not None
    assert cached.accuracy_result.passed == original.accuracy_result.passed
    assert cached.accuracy_result.metadata == original.accuracy_result.metadata


@pytest.mark.parametrize("damage", ["missing", "not-map", "unknown-storage", "unused-object", "unused-storage"])
def test_invalid_reference_object_mapping_is_rejected(tmp_path, damage):
    case = SimpleNamespace(input_shapes=[[2]], dtypes=["int64"])
    value = torch.tensor([1, 2], dtype=torch.int64)
    cache = TensorCache(tmp_path)
    path = cache.write({}, case, [value], value.clone(), value.clone() + 1)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = manifest["files"]["references.safetensors"]
    if damage == "missing":
        del spec["tensor_objects"]
    elif damage == "not-map":
        spec["tensor_objects"] = []
    elif damage == "unknown-storage":
        spec["tensor_objects"]["object_000000"] = "not-present"
    elif damage == "unused-object":
        spec["tensor_objects"]["unused"] = "tensor_000000"
    else:
        spec["tensor_objects"]["object_000001"] = "tensor_000000"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(TensorCacheError):
        cache.load({}, case)
