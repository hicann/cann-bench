#!/usr/bin/python3
# coding=utf-8
from types import SimpleNamespace


def _make_evaluator():
    # Build a bare Evaluator without running __init__; attach only golden_loader.
    from kernel_eval.eval.evaluator import Evaluator
    return Evaluator.__new__(Evaluator)


def test_oracle_ref_uses_hook_when_present():
    ev = _make_evaluator()
    oracle = lambda *a, **k: "oracle"
    golden = lambda *a, **k: "golden"
    ev.golden_loader = SimpleNamespace(
        get_oracle_function=lambda rp: oracle,
        get_bench_function=lambda rp: None,
    )
    assert ev._oracle_reference_func("level3/weight_quant_batch_matmul", golden) is oracle


def test_oracle_ref_falls_back_to_golden():
    ev = _make_evaluator()
    golden = lambda *a, **k: "golden"
    ev.golden_loader = SimpleNamespace(
        get_oracle_function=lambda rp: None,
        get_bench_function=lambda rp: None,
    )
    assert ev._oracle_reference_func("level3/conv_2d", golden) is golden


def test_bench_ref_uses_hook_when_present():
    ev = _make_evaluator()
    bench = lambda *a, **k: "bench"
    golden = lambda *a, **k: "golden"
    ev.golden_loader = SimpleNamespace(
        get_oracle_function=lambda rp: None,
        get_bench_function=lambda rp: bench,
    )
    assert ev._bench_reference_func("level3/weight_quant_batch_matmul", golden) is bench


def test_bench_ref_falls_back_to_golden():
    ev = _make_evaluator()
    golden = lambda *a, **k: "golden"
    ev.golden_loader = SimpleNamespace(
        get_oracle_function=lambda rp: None,
        get_bench_function=lambda rp: None,
    )
    assert ev._bench_reference_func("level3/conv_2d", golden) is golden
