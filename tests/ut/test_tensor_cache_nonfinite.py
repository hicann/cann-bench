#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Special-value contracts remain usable with strict JSON tensor fixtures."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("safetensors")

from kernel_eval.config import Config
from kernel_eval.data.tensor_cache import TensorCache, identity_for
from kernel_eval.eval.evaluator import Evaluator


@pytest.fixture
def evaluator(tmp_path):
    config = Config(device_type="cpu", enable_profiler=False, torch_op_guard_mode="off",
                    tasks_root=str(Path(__file__).resolve().parents[2] / "tasks"),
                    tensor_cache_dir=str(tmp_path / "bank"), tensor_cache_mode="off",
                    tensor_cache_tasks=["level1/exp"])
    result = Evaluator(config)
    result._cleanup_memory = lambda: None
    return result


@pytest.fixture(params=[12, 13], ids=["inf", "nan"])
def special_case(evaluator, request):
    # Keep the published dtype, attrs and range, reducing only input storage.
    task = Path(evaluator.config.tasks_root) / "level1/exp"
    cases = evaluator.case_loader._load_operator_cases(task)
    return replace(next(case for case in cases if case.case_num == request.param), input_shapes=[[8]])


def test_public_special_value_cases_build_and_load(evaluator, special_case):
    path = evaluator.build_tensor_cache(special_case)
    identity = identity_for(evaluator, special_case)
    assert identity == identity_for(evaluator, special_case)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["identity"] == identity
    assert manifest["key"] == TensorCache.key(identity)
    json.dumps(manifest, allow_nan=False)
    loaded = TensorCache(evaluator.config.tensor_cache_dir).load(identity, special_case)
    assert loaded.inputs[0].dtype == getattr(torch, special_case.dtypes[0])
    if special_case.case_num == 12:
        assert torch.isinf(loaded.inputs[0]).any()
    else:
        assert torch.isnan(loaded.inputs[0]).any()


def test_public_special_value_cases_cache_modes_agree(evaluator, special_case):
    candidate = evaluator.golden_loader.get_golden_function(special_case.rel_path)
    assert evaluator.evaluate_case(special_case, ai_op_func=candidate).success
    evaluator.config.tensor_cache_mode = "prefer"
    assert evaluator.evaluate_case(special_case, ai_op_func=candidate).success
    assert not Path(evaluator.config.tensor_cache_dir).exists()
    evaluator.build_tensor_cache(special_case)
    with patch.object(evaluator.data_generator, "generate_input_tensors_from_case",
                      side_effect=AssertionError("cache hit generated inputs")), \
         patch.object(evaluator, "_oracle_reference_func", side_effect=AssertionError("oracle executed")), \
         patch.object(evaluator, "_bench_reference_func", side_effect=AssertionError("native executed")):
        for mode in ("prefer", "require"):
            evaluator.config.tensor_cache_mode = mode
            assert evaluator.evaluate_case(special_case, ai_op_func=candidate).success


def test_nonfinite_identity_has_no_literal_marker_collisions(evaluator, special_case):
    marker = "__cann_bench_nonfinite__"
    literal_float = {marker: ["float", "nan"]}
    literal_dict = {marker: ["dict", literal_float]}
    values = [float("nan"), float("inf"), -float("inf"), "nan", "inf", "-inf",
              1.0, "1.0", literal_float, literal_dict,
              {"nested": literal_dict}, {marker: ["float", "inf"]}]
    identities = [identity_for(evaluator, replace(special_case, attrs={"nested": [value]}))
                  for value in values]
    keys = [TensorCache.key(identity) for identity in identities]
    assert len(set(keys)) == len(values)
    for value, identity in zip(values, identities):
        assert identity == identity_for(evaluator, replace(special_case, attrs={"nested": [value]}))
        assert json.loads(json.dumps(identity, allow_nan=False)) == identity
    assert identities[0] == identity_for(evaluator, replace(special_case, attrs={"nested": [float("nan")]}))


def test_identity_preserves_ordinary_json_values_and_tuple_normalization(evaluator):
    task = Path(evaluator.config.tasks_root) / "level1/exp"
    case = evaluator.case_loader._load_operator_cases(task)[0]
    case = replace(case, attrs={"finite": 1.5, "string": "nan", "items": (1, False, None)},
                   value_ranges=[(-1, 1)])
    identity = identity_for(evaluator, case)
    assert identity["attrs"] == {"finite": 1.5, "string": "nan", "items": [1, False, None]}
    assert identity["value_ranges"] == [[-1, 1]]
    assert case.attrs["items"] == (1, False, None)
