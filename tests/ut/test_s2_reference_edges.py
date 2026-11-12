# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository for the full text.

"""Public recipe and precision edges for revised S2 references."""

from pathlib import Path

import pytest
import torch
import yaml

from kernel_eval.benches.cann_loader import GoldenLoader

TASKS = Path(__file__).resolve().parents[2] / "tasks"
LOADER = GoldenLoader(str(TASKS))
VM_CASES = yaml.safe_load(
    (TASKS / "level5/tensor_program_vm/cases.yaml").read_text(encoding="utf-8")
)["cases"]


@pytest.mark.parametrize("case", VM_CASES, ids=lambda case: str(case["case_id"]))
def test_public_vm_recipes_prepare_legal_repeatable_inputs(case):
    module = LOADER._load_module("level5/tensor_program_vm")
    p = min(31, case["input_shape"][0][0])
    r, w = [min(value, bound) for value, bound in zip(case["input_shape"][1], (8, 17))]
    generator = torch.Generator().manual_seed(41)
    program = torch.randint(
        0, 1073741824, (p, 4), generator=generator, dtype=torch.int32
    )
    initial = (
        torch.randn(r, w, generator=generator)
        .clamp(-1, 1)
        .to(getattr(torch, case["dtype"][1]))
    )
    original_program, original_initial = program.clone(), initial.clone()
    args = module.get_input(program, initial, **case.get("attrs", {}))
    repeated = module.get_input(program, initial, **case.get("attrs", {}))
    for a, b in zip(args, repeated):
        assert torch.equal(a, b)
    assert args[0].shape == program.shape and args[0].dtype == torch.int32
    assert args[1].shape == initial.shape and args[1].dtype == initial.dtype
    assert bool(((args[0] >= 0) & (args[0] <= 1073741823)).all())
    assert torch.isfinite(module.tensor_program_vm(*args)).all()
    assert torch.equal(program, original_program)
    assert torch.equal(initial, original_initial)


def test_vm_copy_preserves_fp64_reference_bits():
    module = LOADER._load_module("level5/tensor_program_vm")
    initial = torch.tensor([[0.0], [1.0 + 2**-30]], dtype=torch.float64)
    program = torch.tensor([[7, 0, 1, 0]], dtype=torch.int32)
    for function in (module.tensor_program_vm, module.tensor_program_vm_oracle):
        result = function(program, initial)
        assert result.dtype == torch.float64
        assert result[0, 0] == initial[1, 0]
        assert initial[0, 0] == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_kda_zero_update_retains_state_precision(dtype):
    module = LOADER._load_module("level5/kimi_delta_attention")
    state_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    epsilon = 2**-30 if dtype == torch.float64 else 2**-20
    q = torch.ones(1, 1, 1, 1, dtype=dtype)
    k, v = torch.zeros_like(q), torch.zeros_like(q)
    g = torch.zeros_like(q, dtype=state_dtype)
    beta = torch.zeros(1, 1, 1, dtype=state_dtype)
    state = torch.full((1, 1, 1, 1), 1.0 + epsilon, dtype=state_dtype)
    for function in (module.kimi_delta_attention, module.kimi_delta_attention_oracle):
        y, final = function(
            q, k, v, g, beta, state, use_qk_l2norm=False, scaleValue=1.0
        )
        assert y.dtype == dtype and final.dtype == state_dtype
        assert y.is_contiguous() and final.is_contiguous()
        torch.testing.assert_close(final, state, atol=0, rtol=0)


def test_gcd_int32_minimum_has_representable_positive_divisors():
    function = LOADER.get_golden_function("level2/gcd")
    x = torch.tensor([-(2**31)], dtype=torch.int32)
    divisors = torch.tensor([1, 3, 6, 8], dtype=torch.int32)
    assert torch.equal(
        function(x, divisors), torch.tensor([1, 1, 2, 8], dtype=torch.int32)
    )
