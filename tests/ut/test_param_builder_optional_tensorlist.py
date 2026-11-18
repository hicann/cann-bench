#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the repository LICENSE for the full text.

"""Optional TensorList call contracts, including the LSTM regression in #287."""

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest
import torch
import yaml

from kernel_eval.benches.cann_loader import CannCaseLoader, GoldenLoader
from kernel_eval.data.data_generator import DataGenerator
from kernel_eval.utils.param_builder import ParamBuilder


TASKS = Path(__file__).resolve().parents[2] / "tasks"


def _golden(
    x: torch.Tensor,
    weights: List[torch.Tensor],
    bias: Optional[List[torch.Tensor]] = None,
    h0: Optional[torch.Tensor] = None,
):
    return x, weights, bias, h0


@pytest.mark.parametrize("use_override", [False, True])
def test_optional_tensorlist_none_preserves_following_inputs(use_override):
    shapes = [[2], [[2]], None, [2]]
    tensors = DataGenerator().generate_input_tensors_from_case(
        shapes, ["float32", ["float32"], None, "float32"], value_ranges=[], seed=287
    )
    case = SimpleNamespace(input_shapes=None if use_override else shapes, attrs={})

    params = ParamBuilder().build_call_params(
        _golden, case, tensors, override_shapes=shapes if use_override else None
    )

    assert tensors[2] is None
    assert params["bias"] is None
    assert params["weights"] is tensors[1]
    assert params["h0"] is tensors[3]
    assert _golden(**params)[3] is tensors[3]


@pytest.mark.parametrize("value_kind", ["tensor", "list", "empty_list"])
def test_tensorlist_values_keep_existing_conversion(value_kind):
    x, weight, bias_tensor = (torch.ones(2) for _ in range(3))
    bias = {"tensor": bias_tensor, "list": [bias_tensor], "empty_list": []}[value_kind]
    case = SimpleNamespace(input_shapes=[[2], [2], [[2]]], attrs={})

    params = ParamBuilder().build_call_params(_golden, case, [x, weight, bias])

    assert len(params["weights"]) == 1
    assert params["weights"][0] is weight
    if value_kind == "tensor":
        assert len(params["bias"]) == 1
        assert params["bias"][0] is bias_tensor
    else:
        assert params["bias"] is bias


def test_omitted_optional_tensorlist_uses_default():
    x, weight = torch.ones(2), torch.ones(2)
    case = SimpleNamespace(input_shapes=[[2], [[2]]], attrs={})

    params = ParamBuilder().build_call_params(_golden, case, [x, [weight]])

    assert "bias" not in params
    assert _golden(**params)[2:] == (None, None)


@pytest.fixture(scope="module")
def lstm_contract():
    golden = GoldenLoader(str(TASKS)).get_golden_function("level4/lstm")
    cases = CannCaseLoader(str(TASKS), target_hardware="910b2").scan_by_rel_path("level4/lstm")
    proto = yaml.safe_load((TASKS / "level4/lstm/proto.yaml").read_text(encoding="utf-8"))
    library = torch.library.Library("param_builder_lstm_ut", "DEF")
    library.define(proto["operator"]["schema"])
    # Exercise the real dispatcher schema without running an LSTM kernel.
    library.impl("lstm", lambda x, *args, **kwargs: (x, x, x), "CPU")
    yield golden, {case.case_num: case for case in cases}, torch.ops.param_builder_lstm_ut.lstm.default


@pytest.mark.parametrize("param_name,input_index", [("weight_ih", 1), ("weight_hh", 2)])
def test_required_tensorlist_none_is_preserved_and_rejected_by_schema(lstm_contract, param_name, input_index):
    golden, cases, operator = lstm_contract
    case = cases[11]
    tensors = DataGenerator().generate_input_tensors_from_case(
        case.input_shapes, case.dtypes, case.value_ranges, seed=287
    )
    tensors[input_index] = None

    params = ParamBuilder().build_call_params(golden, case, tensors)

    assert params[param_name] is None
    with pytest.raises(RuntimeError, match=f"argument '{param_name}'"):
        operator(**params)


@pytest.mark.parametrize("case_id", range(1, 21))
def test_lstm_cases_accept_strict_schema(lstm_contract, case_id):
    golden, cases, operator = lstm_contract
    case = cases[case_id]
    tensors = DataGenerator().generate_input_tensors_from_case(
        case.input_shapes, case.dtypes, case.value_ranges, seed=287
    )

    params = ParamBuilder().build_call_params(golden, case, tensors)
    outputs = operator(**params)

    assert outputs[0] is tensors[0]
    assert params["weight_ih"] is tensors[1]
    assert params["weight_hh"] is tensors[2]
    if case_id in (11, 12):
        assert params["bias"] is False
        assert params.get("bias_ih") is None
        assert params.get("bias_hh") is None
    if case_id in (1, 11, 15):
        assert params["weight_hr"] is None
