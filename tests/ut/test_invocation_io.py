# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Real byte cleanup, bounded lifetimes, alias handling and capture boundaries."""

import gc
import weakref

import pytest
import torch

from kernel_eval.eval import invocation_io
from kernel_eval.eval.invocation_io import (
    InvocationIO,
    snapshot_output,
    storage_bytes,
    storage_key,
)
from kernel_eval.eval.restored_inputs import RestoredCall
from kernel_eval.security.torch_op_guard import _pause_state
from tests.ut.test_restored_inputs import mock_evaluator


@pytest.mark.parametrize(
    "op",
    [
        torch.cummin, torch.cummax, torch.min, torch.max, torch.sort, torch.median,
        pytest.param(lambda x, dim: torch.topk(x, 2, dim=dim), id="topk"),
    ],
    ids=lambda op: op.__name__,
)
def test_torch_structured_outputs_are_snapshotted_and_scrubbed(op):
    x = torch.tensor([4.0, 1.0, 3.0, 2.0])
    expected = op(x, dim=0)

    def candidate(input_tensor):
        result = op(input_tensor, dim=0)
        return {"result": result, "same": result, "values": result.values}

    call = RestoredCall(candidate, (x,), {})
    for index in range(2):
        call.prepare(index, "measured")
        output = call()
        result = output["result"]
        saved, count = call.retire(preserve_output=index == 1)
        assert count == sum(
            tensor.numel() * tensor.element_size()
            for tensor in (x, expected.values, expected.indices)
        )
        assert all(not storage_bytes(tensor).any() for tensor in result)
        if index == 0:
            assert saved is None
    assert type(saved["result"]) is type(expected)
    assert saved["result"] is saved["same"]
    assert saved["result"].values is saved["values"]
    assert torch.equal(saved["result"].values, expected.values)
    assert torch.equal(saved["result"].indices, expected.indices)
    call.release()
    assert torch.equal(saved["result"].values, expected.values)
    assert torch.equal(x, torch.tensor([4.0, 1.0, 3.0, 2.0]))


def test_torch_structured_output_preserves_shared_views():
    x = torch.arange(8, dtype=torch.int64)
    output = torch.return_types.cummin((x[1:7:2], x[2:8:2]))
    saved = snapshot_output(output)
    assert type(saved) is type(output)
    assert storage_key(saved.values) == storage_key(saved.indices)
    assert storage_key(saved.values) != storage_key(x)
    assert saved.values.storage_offset() == 1 and saved.indices.storage_offset() == 2
    assert saved.values.stride() == saved.indices.stride() == (2,)
    assert torch.equal(storage_bytes(saved.values), storage_bytes(x))


def test_arbitrary_tuple_subclass_is_not_treated_as_torch_return_type():
    class FakeResult(tuple):
        __module__ = "torch.return_types"

        def __iter__(self):
            raise AssertionError("must not invoke arbitrary container methods")

    x = torch.ones(3)
    io = InvocationIO()
    io.bind_inputs((x,), {})
    io.output = FakeResult((x,))
    with pytest.raises(ValueError, match="does not support FakeResult"):
        io.retire(lambda: None)
    assert not x.any()
    io.release()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int64,
        torch.int32,
        torch.uint8,
        torch.bool,
        torch.complex64,
    ],
)
def test_cleanup_preserves_final_bytes_and_clears_input_output_once(dtype, monkeypatch):
    x = torch.ones(12, dtype=dtype)
    call = RestoredCall(lambda a: {"base": a, "view": a[2:8:2], "same": a}, (x,), {})
    call.prepare(0, "measured")
    output = call()
    expected_bytes = storage_bytes(output["base"]).clone()
    cleared = []
    zero = invocation_io.zero_storage
    monkeypatch.setattr(
        invocation_io,
        "zero_storage",
        lambda raw: (cleared.append(raw.numel()), zero(raw)),
    )
    saved, count = call.retire(preserve_output=True)
    assert len(cleared) == 1 and count == x.numel() * x.element_size()
    assert not storage_bytes(output["base"]).any()
    assert torch.equal(storage_bytes(saved["base"]), expected_bytes)
    assert saved["base"] is saved["same"]
    assert saved["view"].stride() == (2,) and saved["view"].storage_offset() == 2
    assert storage_key(saved["base"]) == storage_key(saved["view"])
    assert torch.equal(x, torch.ones_like(x))
    call.release()
    assert torch.equal(storage_bytes(saved["base"]), expected_bytes)


def test_snapshot_preserves_nan_payload_signed_zero_and_noncontiguous_aliases():
    bits = torch.tensor([0x7FC01234, -2147483648, 0, 0x7F800000], dtype=torch.int32)
    source = bits.view(torch.float32).reshape(2, 2)
    group = [source.t(), source[:, :1]]
    result = snapshot_output({"group": group, "same_group": group})
    assert result["group"] is result["same_group"]
    assert result["group"][0].stride() == source.t().stride()
    assert torch.equal(storage_bytes(result["group"][0]), storage_bytes(source))
    assert storage_key(result["group"][0]) == storage_key(result["group"][1])


def test_previous_io_remains_allocated_and_zero_until_next_call_finishes():
    refs = []
    call = RestoredCall(lambda x: x + 3, (torch.ones(16),), {})
    old_keys = set()
    enabled = gc.isenabled()
    gc.disable()
    try:
        for index in range(40):
            call.prepare(index, "measured")
            call()
            current = {storage_key(call.args[0]), storage_key(call.io.output)}
            assert current.isdisjoint(old_keys)
            assert all(not raw.any() for raw in call.io.previous.values())
            call.retire()
            assert len(call.io.previous) == 2 and not call.io.current
            assert all(ref() is None for ref in refs)
            refs = [weakref.ref(raw) for raw in call.io.previous.values()]
            old_keys = current
        call.release()
        assert all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()


def test_protected_addresses_have_live_storage_until_release():
    call = RestoredCall(lambda x: x + 1, (torch.ones(4),), {})
    refs = [weakref.ref(raw) for raw in call.io.protected.values()]
    assert len(refs) == 2  # temporary caller input and pristine snapshot
    assert all(storage_key(raw) == key for key, raw in call.io.protected.items())
    call.release()
    assert all(ref() is None for ref in refs)


def test_input_storage_is_retained_when_candidate_replaces_container_member():
    held = []

    def candidate(group):
        held.append(group.pop())
        return torch.ones(3)

    call = RestoredCall(candidate, ([torch.ones(3)],), {})
    call.prepare(0, "preflight")
    call()
    _, count = call.retire()
    assert count == 24 and not held[0].any()
    call.release()


@pytest.mark.parametrize("failure", ["candidate", "validator", "profiler", "parse"])
def test_failure_scrubs_known_io_and_has_no_retry(tmp_path, failure):
    evaluator, _ = mock_evaluator(tmp_path)
    held = []
    calls = []

    def candidate(x):
        calls.append(1)
        held.append(x)
        x.fill_(9)
        if failure == "candidate":
            raise RuntimeError("candidate failed")
        out = x + 2
        held.append(out)
        return out

    def validator(*_):
        if failure == "validator":
            raise RuntimeError("validator failed")

    if failure == "profiler":

        def profile(fn, *args, **kwargs):
            fn()
            raise RuntimeError("profiler failed after candidate")

        evaluator._profile = profile
    if failure == "parse":
        evaluator._locate_prof_files = lambda _: (_ for _ in ()).throw(
            RuntimeError("parse failed")
        )
    output, result = evaluator.run_profiled(
        "level1/test_1",
        candidate,
        torch.ones(3),
        warmup=0,
        repeat=3,
        validator=validator,
    )
    assert output is None and result.elapsed_us == 0
    assert result.metadata["perf_integrity_failed"]
    assert len(calls) == (1 if failure in ("candidate", "validator") else 2)
    assert all(not value.any() for value in held)


def test_cleanup_and_final_snapshot_are_outside_capture(tmp_path, monkeypatch):
    evaluator, windows = mock_evaluator(tmp_path)
    zero = invocation_io.zero_storage
    events = []

    def checked_zero(raw):
        assert not windows or windows[-1] == "exit"
        assert getattr(_pause_state, "paused", False)
        events.append("zero")
        zero(raw)

    monkeypatch.setattr(invocation_io, "zero_storage", checked_zero)
    evaluator._synchronize_profile_step = lambda: events.append("sync")
    output, result = evaluator.run_profiled(
        "level1/test_1", lambda x: x + 2, torch.ones(3), warmup=2, repeat=3
    )
    assert result.is_success(), result.error_msg
    assert torch.equal(output, torch.full((3,), 3.0))
    assert result.metadata["io_retired_calls"] == 6
    assert result.metadata["io_scrubbed_bytes"] == 6 * 24
    assert result.elapsed_us == 17  # mock candidate kernels only
    assert events[-1] == "sync"


def test_returning_original_caller_storage_fails_without_clearing_it(tmp_path):
    evaluator, _ = mock_evaluator(tmp_path)
    original = torch.ones(3)
    output, result = evaluator.run_profiled(
        "level1/test_1", lambda x: original, original, warmup=0, repeat=1
    )
    assert output is None and result.metadata["perf_integrity_failed"]
    assert "protected" in result.error_msg
    assert original.tolist() == [1.0, 1.0, 1.0]


def test_reusing_previous_output_storage_fails_isolation(tmp_path):
    evaluator, _ = mock_evaluator(tmp_path)
    fixed = torch.ones(3)
    output, result = evaluator.run_profiled(
        "level1/test_1", lambda x: fixed, torch.ones(3), warmup=0, repeat=2
    )
    assert output is None and result.metadata["perf_integrity_failed"]
    assert "previous invocation storage" in result.error_msg


def test_cleanup_failure_stops_before_next_invocation(tmp_path, monkeypatch):
    evaluator, _ = mock_evaluator(tmp_path)
    calls = []

    def broken_zero(raw):
        raise RuntimeError("memset failed")

    monkeypatch.setattr(invocation_io, "zero_storage", broken_zero)

    def candidate(x):
        calls.append(1)
        return x

    output, result = evaluator.run_profiled(
        "level1/test_1", candidate, torch.ones(3), warmup=0, repeat=2
    )
    assert output is None and result.metadata["perf_integrity_failed"]
    assert calls == [1] and "memset failed" in result.error_msg


def test_invalid_output_still_clears_inputs():
    x = torch.ones(3)
    io = InvocationIO()
    io.bind_inputs((x,), {})
    io.output = object()
    with pytest.raises(ValueError, match="does not support object"):
        io.retire(lambda: None)
    assert not x.any()
    io.release()


def test_failed_cleanup_releases_buffers_without_cyclic_collection(
    tmp_path, monkeypatch
):
    evaluator, _ = mock_evaluator(tmp_path)
    refs = []

    def broken_zero(raw):
        refs.append(weakref.ref(raw))
        raise RuntimeError("cleanup failure")

    monkeypatch.setattr(invocation_io, "zero_storage", broken_zero)
    enabled = gc.isenabled()
    gc.disable()
    try:
        _, result = evaluator.run_profiled(
            "level1/test_1", lambda x: x + 1, torch.ones(3), warmup=0, repeat=2
        )
        assert result.metadata["perf_integrity_failed"]
        assert refs and all(ref() is None for ref in refs)
    finally:
        if enabled:
            gc.enable()
