#!/usr/bin/python3
# coding=utf-8
import torch
from tasks.level3.weight_quant_batch_matmul.golden import (
    weight_quant_batch_matmul_oracle as oracle,
    weight_quant_batch_matmul_bench as bench,
)


def _inputs(dtype):
    torch.manual_seed(0)
    x = (torch.rand(4, 256) * 2 - 1).to(dtype)
    w = torch.randint(-128, 128, (256, 128), dtype=torch.int8)
    scale = (torch.rand(128) * 0.1 + 0.001).to(dtype)
    return x, w, scale


def test_oracle_dtype_faithful_fp64():
    # oracle at fp64 inputs computes in fp64 (no hidden downcast): output dtype == fp64
    x, w, scale = _inputs(torch.float64)
    y = oracle(x, w, scale)
    assert y.dtype == torch.float64


def test_bench_dequants_to_output_dtype():
    # bench dequants weight to the input(output) dtype (bf16-lossy), accumulates in fp32
    x, w, scale = _inputs(torch.bfloat16)
    y = bench(x, w, scale)
    assert y.dtype == torch.bfloat16
    dq_bf16 = (w.to(torch.bfloat16) * scale.to(torch.bfloat16)).float()  # bf16 dequant, then fp32
    ref = torch.matmul(x.float(), dq_bf16).to(torch.bfloat16)            # fp32 accumulate, round to bf16
    assert torch.equal(y, ref)


def test_oracle_more_accurate_than_bench():
    # same bf16 input values: oracle(fp64 dequant) closer to true than bench(bf16 dequant)
    x16, w, s16 = _inputs(torch.bfloat16)
    o = oracle(x16.double(), w, s16.double()).double()   # fp64 oracle
    p = bench(x16, w, s16).double()                      # bf16-dequant bench
    true_ = torch.matmul(x16.double(), w.double() * s16.double())
    assert (p - true_).abs().sum() > (o - true_).abs().sum()
