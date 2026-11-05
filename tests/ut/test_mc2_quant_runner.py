# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Execute the rank runner against simulated HCCL and real CPU tensor math."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d

from kernel_eval.eval import mc2_distributed_runner as runner


@pytest.fixture
def rank_runtime(monkeypatch, tmp_path):
    events = []
    failure = {"phase": None, "cleanup": None}
    groups = {}

    def fail(phase):
        if failure["phase"] == phase:
            raise RuntimeError(f"injected {phase} failure")

    class Group:
        def __init__(self, name):
            self.name = name

        def _get_backend(self, device):
            return self

        def get_hccl_comm_name(self, rank):
            if self.name == "candidate":
                fail("candidate_name")
            return f"{self.name}-hccl-rank{rank}"

    groups["default"] = Group("default")

    def init_process_group(**kwargs):
        assert kwargs["backend"] == "hccl"
        assert kwargs["world_size"] == 2
        events.append(("init", "default"))

    def new_group(*, backend, timeout):
        name = "candidate" if backend == "hccl" else "cpu"
        fail(f"create_{name}")
        groups[name] = Group(name)
        events.append(("create", name))
        return groups[name]

    def destroy_process_group(group=None):
        name = "default" if group is None else group.name
        events.append(("destroy", name))
        if failure["cleanup"] == name:
            raise RuntimeError(f"injected {name} teardown failure")

    def all_reduce(tensor, op=None, group=None):
        name = "default" if group is None else group.name
        assert name != "candidate", (
            "reference collective leaked into the candidate domain"
        )
        events.append(("all_reduce", name))

    def barrier(group=None):
        name = "default" if group is None else group.name
        assert name == "default", "measurement or profiler barrier used another domain"
        events.append(("barrier", name))

    for name, implementation in (
        ("init_process_group", init_process_group),
        ("new_group", new_group),
        ("destroy_process_group", destroy_process_group),
        ("all_reduce", all_reduce),
        ("barrier", barrier),
    ):
        monkeypatch.setattr(dist, name, implementation)
    monkeypatch.setattr(
        distributed_c10d, "_get_default_group", lambda: groups["default"]
    )

    actual_device = torch.device
    monkeypatch.setattr(
        runner.torch,
        "device",
        lambda name: actual_device("cpu" if str(name).startswith("npu") else name),
    )
    npu = SimpleNamespace(
        device_count=lambda: 2,
        get_device_name=lambda rank: "Ascend950DT_9582",
        set_device=lambda rank: None,
        synchronize=lambda device: events.append(("sync", "device")),
    )
    monkeypatch.setattr(runner.torch, "npu", npu, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu=npu))
    monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", "test-preserved")

    def make_inputs(ctx, payload):
        fail("inputs")
        events.append(("inputs", ctx["hcomm_info"]))
        assert ctx["dist"] is dist
        return {
            "x1": torch.tensor([2.0, 3.0]),
            "x2": torch.tensor([1.0, 1.0]),
            "bias": None,
        }

    def npu_reference(ctx, inputs, attrs):
        fail("reference")
        events.append(("reference", ctx["hcomm_info"]))
        result = inputs["x1"].clone()
        ctx["dist"].all_reduce(result)
        return result

    def cpu_reference(ctx, inputs, attrs):
        fail("cpu")
        events.append(("cpu", ctx["hcomm_info"]))
        assert ctx["dist"].group is groups["cpu"]
        result = inputs["x1"].clone()
        ctx["dist"].all_reduce(result)
        return result

    def candidate(x1, x2, hcomm_info, world_size, **kwargs):
        fail("candidate")
        assert world_size == 2
        events.append(("candidate", hcomm_info))
        return x1.clone()

    def call_candidate(function, ctx, inputs, attrs):
        assert ctx["dist"] is dist
        return function(
            inputs["x1"], inputs["x2"], ctx["hcomm_info"], ctx["world_size"]
        )

    module = SimpleNamespace(
        mc2_make_rank_inputs=make_inputs,
        mc2_distributed_golden=npu_reference,
        mc2_cpu_golden=cpu_reference,
        mc2_call_candidate=call_candidate,
    )
    loader = SimpleNamespace(
        _load_module=lambda path: module,
        get_mc2_distributed_golden=lambda path, required: module.mc2_distributed_golden,
    )
    monkeypatch.setattr(runner, "get_golden_loader", lambda *args, **kwargs: loader)
    package = ModuleType("cann_bench")
    for name in (
        "quant_all_reduce",
        "quant_reduce_scatter",
        "quant_grouped_mat_mul_allto_allv",
        "matmul_all_reduce",
    ):
        setattr(package, name, candidate)
    monkeypatch.setitem(sys.modules, "cann_bench", package)

    # Keep _measure and _profile_candidate_rank0 real. Only the device profiler
    # implementation is replaced so its surrounding synchronization is tested.
    def profile_loop(fn, prof_dir, warmup, repeat, profiler_level):
        fail("profile")
        for _ in range(warmup + repeat):
            fn()

    monkeypatch.setattr(runner, "_profile_rank0_loop", profile_loop)

    def run(
        operator="QuantAllReduce",
        isolated=True,
        rank=0,
        mode="three_way",
        perf=True,
        adapter=True,
        supported_devices=("950dt",),
    ):
        if not adapter and hasattr(module, "mc2_call_candidate"):
            del module.mc2_call_candidate
        attrs = {
            "world_size": 2,
            "supported_devices": list(supported_devices) if supported_devices else [],
        }
        if mode is not None:
            attrs["mc2_accuracy_mode"] = mode
        if isolated:
            attrs["mc2_isolated_candidate_group"] = True
        payload = {
            "operator": operator,
            "attrs": attrs,
            "tasks_root": str(tmp_path),
            "reports_dir": str(tmp_path),
            "rel_path": "level4/test_quant",
            "case_id": "test_quant_1",
            "case_num": 1,
            "dtypes": ["float32"],
            "timeout": 30,
        }
        return runner._run_rank(rank, 2, 29500, payload, 1, 2, perf, {})

    return SimpleNamespace(
        run=run,
        events=events,
        failure=failure,
        groups=groups,
        package=package,
        module=module,
        npu=npu,
    )


@pytest.mark.parametrize(
    "device,expected",
    [("Ascend950PR_9579", "auto"), ("Ascend950DT_9582", "auto"), ("Ascend910B2", None)],
)
def test_socket_default_only_applies_to_normalized_a5(
    rank_runtime, monkeypatch, device, expected
):
    import os

    monkeypatch.delenv("HCCL_NPU_SOCKET_PORT_RANGE", raising=False)
    rank_runtime.npu.get_device_name = lambda rank: device
    rank_runtime.run(mode=None, isolated=False, perf=False, supported_devices=None)
    assert os.environ.get("HCCL_NPU_SOCKET_PORT_RANGE") == expected


def test_explicit_socket_configuration_is_preserved(rank_runtime):
    import os

    rank_runtime.run(perf=False)
    assert os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] == "test-preserved"


@pytest.mark.parametrize("mode", [None, "three_way"])
@pytest.mark.parametrize(
    "device,supported,accepted",
    [
        ("Ascend950DT_9582", ["950dt"], True),
        ("Ascend950DT_9582", ["Ascend950DT_9582"], True),
        ("950dt", ["Ascend950DT_9582"], True),
        ("Ascend950DT_9582", ["950pr", "Ascend950DT_9582"], True),
        ("Ascend950DT_9582", ["Ascend950PR_9579"], False),
        ("Ascend950PR_9579", ["950dt"], False),
    ],
)
def test_rank_hardware_gate_matches_case_eligibility(
    rank_runtime, mode, device, supported, accepted
):
    from kernel_eval.utils.case_eligibility import supports_case

    assert supports_case({"supported_devices": supported}, device) is accepted
    rank_runtime.npu.get_device_name = lambda rank: device
    if accepted:
        assert rank_runtime.run(mode=mode, perf=False, supported_devices=supported)[
            "passed"
        ]
    else:
        with pytest.raises(RuntimeError, match="supports"):
            rank_runtime.run(mode=mode, perf=False, supported_devices=supported)
        assert not any(event == "candidate" for event, _ in rank_runtime.events)


def test_invalid_reference_stops_all_candidate_and_performance_calls(rank_runtime):
    rank_runtime.module.mc2_cpu_golden = lambda *args: torch.tensor([20.0, 30.0])
    result = rank_runtime.run()
    assert not result["passed"]
    assert result["failure_type"] == "reference_invalid"
    assert not any(event == "candidate" for event, _ in rank_runtime.events)


@pytest.mark.parametrize("rank", [0, 1])
def test_staged_reference_flows_through_preflight_and_comparison(rank_runtime, rank):
    calls = []

    def references(ctx, inputs, attrs):
        assert ctx["dist"].group is rank_runtime.groups["cpu"]
        calls.append(ctx["rank"])
        # Isolate reference plumbing with a representable FP32 rounding step.
        # A 0.1 offset would be a genuinely invalid staged reference under the
        # fixed physical-dtype health bound, not a plumbing-only fixture.
        return inputs["x1"] + 2**-22, inputs["x1"].clone()

    rank_runtime.module.mc2_cpu_references = references
    rank_runtime.module.mc2_cpu_golden = lambda *args: pytest.fail(
        "duplicate CPU reference calculation"
    )
    result = rank_runtime.run(rank=rank)
    assert result["passed"]
    assert calls == [rank]  # Never included in warmup, performance or profiler loops.
    output = result["output_results"][0]
    assert output["reference_check"]["reference"] == "cpu_staged_cascade"
    assert output["candidate_metrics"]["diff_rmse"] > 0
    assert output["candidate_metrics"] == output["cascade_metrics"]
    assert any(event == "candidate" for event, _ in rank_runtime.events)


@pytest.mark.parametrize("rank", [0, 1])
def test_broken_staged_reference_reaches_consensus_and_stops_candidate(
    rank_runtime, monkeypatch, rank
):
    rank_runtime.module.mc2_cpu_references = lambda *args: (
        torch.tensor([2.0, 3.0]),
        torch.ones(1),
    )
    votes = []
    original = dist.all_reduce

    def all_reduce(tensor, op=None, group=None):
        if tensor.dtype == torch.int32 and group is rank_runtime.groups.get("cpu"):
            votes.append(tensor.item())
        original(tensor, op=op, group=group)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    result = rank_runtime.run(rank=rank)
    assert result["failure_type"] == "reference_invalid"
    assert votes == [1]
    assert not any(event == "candidate" for event, _ in rank_runtime.events)


def test_peer_reference_failure_stops_otherwise_healthy_rank(rank_runtime, monkeypatch):
    original = dist.all_reduce

    def all_reduce(tensor, op=None, group=None):
        if tensor.dtype == torch.int32 and group is rank_runtime.groups.get("cpu"):
            tensor.fill_(1)
        else:
            original(tensor, op=op, group=group)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    result = rank_runtime.run()
    assert result["failure_type"] == "reference_invalid"
    assert "another rank" in result["compare_error"]
    assert not any(event == "candidate" for event, _ in rank_runtime.events)


def test_reference_check_exception_reaches_consensus_before_candidate(
    rank_runtime, monkeypatch
):
    # A malformed cascade dtype used to throw before the shared invalid flag.
    rank_runtime.module.mc2_distributed_golden = lambda *args: torch.tensor(
        [2.0, 3.0], dtype=torch.float64
    )
    consensus = []
    original = dist.all_reduce

    def all_reduce(tensor, op=None, group=None):
        if tensor.dtype == torch.int32 and group is rank_runtime.groups.get("cpu"):
            consensus.append(tensor.item())
        original(tensor, op=op, group=group)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    result = rank_runtime.run()
    assert result["failure_type"] == "reference_invalid"
    assert "reference precision check failed" in result["compare_error"]
    assert consensus == [1]
    assert not any(event == "candidate" for event, _ in rank_runtime.events)


@pytest.mark.parametrize("peer_failure", [False, True])
def test_reference_timing_failure_stops_profiling_on_all_ranks(
    rank_runtime, monkeypatch, peer_failure
):
    calls = 0
    votes = []
    original = dist.all_reduce

    def reference(*args):
        nonlocal calls
        calls += 1
        if calls > 1 and not peer_failure:
            raise RuntimeError("reference timing failure")
        return torch.tensor([2.0, 3.0])

    def all_reduce(tensor, op=None, group=None):
        if tensor.dtype == torch.int32 and group is rank_runtime.groups.get("cpu"):
            votes.append(tensor.item())
            if peer_failure and len(votes) == 2:
                tensor.fill_(1)
        original(tensor, op=op, group=group)

    rank_runtime.module.mc2_distributed_golden = reference
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    with pytest.raises(runner.MC2ReferenceInvalidError, match="reference timing"):
        rank_runtime.run()
    assert votes == [0, 0 if peer_failure else 1]
    # Accuracy and candidate timing finished; no profiler candidate call follows.
    assert sum(event == "candidate" for event, _ in rank_runtime.events) == 4


@pytest.mark.parametrize("operator", ["QuantAllReduce", "QuantReduceScatter"])
@pytest.mark.parametrize("rank", [0, 1])
def test_quant_candidate_has_exclusive_domain_through_timing_and_profiling(
    rank_runtime, operator, rank
):
    result = rank_runtime.run(operator=operator, rank=rank)
    assert result["passed"]
    assert result["elapsed_us"] > 0
    events = rank_runtime.events
    assert [name for event, name in events if event == "create"] == ["candidate", "cpu"]
    assert {name for event, name in events if event == "candidate"} == {
        f"candidate-hccl-rank{rank}"
    }
    for phase in ("inputs", "reference", "cpu"):
        assert {name for event, name in events if event == phase} == {
            f"default-hccl-rank{rank}"
        }
    assert {name for event, name in events if event == "all_reduce"} == {
        "default",
        "cpu",
    }
    # One accuracy call, warmup + repeats for timing, and profiler iterations.
    assert sum(event == "candidate" for event, _ in events) == 7
    assert sum(event == "barrier" for event, _ in events) == 7
    assert [name for event, name in events if event == "destroy"] == [
        "cpu",
        "candidate",
        "default",
    ]


@pytest.mark.parametrize(
    "phase,expected",
    [
        ("create_candidate", ["default"]),
        ("candidate_name", ["candidate", "default"]),
        ("inputs", ["candidate", "default"]),
        ("create_cpu", ["candidate", "default"]),
        ("cpu", ["cpu", "candidate", "default"]),
        ("reference", ["cpu", "candidate", "default"]),
        ("candidate", ["cpu", "candidate", "default"]),
        ("profile", ["cpu", "candidate", "default"]),
    ],
)
def test_setup_and_execution_failure_release_only_created_groups(
    rank_runtime, phase, expected
):
    rank_runtime.failure["phase"] = phase
    with pytest.raises(RuntimeError, match=f"injected {phase} failure"):
        rank_runtime.run()
    assert [
        name for event, name in rank_runtime.events if event == "destroy"
    ] == expected


def test_subgroup_teardown_failure_preserves_original_error_and_cleans_default(
    rank_runtime,
):
    rank_runtime.failure.update(phase="candidate", cleanup="cpu")
    with pytest.raises(RuntimeError, match="injected candidate failure"):
        rank_runtime.run()
    assert [name for event, name in rank_runtime.events if event == "destroy"] == [
        "cpu",
        "candidate",
        "default",
    ]


@pytest.mark.parametrize("mode,groups", [(None, []), ("three_way", ["cpu"])])
def test_existing_builtin_task_keeps_default_candidate_and_reference_domain(
    rank_runtime, mode, groups
):
    result = rank_runtime.run(
        operator="MatmulAllReduce", isolated=False, mode=mode, adapter=False
    )
    assert result["passed"]
    assert [name for event, name in rank_runtime.events if event == "create"] == groups
    assert {name for event, name in rank_runtime.events if event == "candidate"} == {
        "default-hccl-rank0"
    }
    assert [name for event, name in rank_runtime.events if event == "destroy"] == list(
        reversed(groups)
    ) + ["default"]


def test_quant_grouped_matmul_mapping_does_not_force_an_exclusive_group(rank_runtime):
    result = rank_runtime.run(
        operator="QuantGroupedMatMulAlltoAllv", isolated=False, perf=False
    )
    assert result["passed"]
    assert [name for event, name in rank_runtime.events if event == "create"] == ["cpu"]
    assert {name for event, name in rank_runtime.events if event == "candidate"} == {
        "default-hccl-rank0"
    }


@pytest.mark.parametrize(
    "operator,function",
    [
        ("QuantAllReduce", "quant_all_reduce"),
        ("QuantReduceScatter", "quant_reduce_scatter"),
        ("QuantGroupedMatMulAlltoAllv", "quant_grouped_mat_mul_allto_allv"),
    ],
)
def test_new_candidate_names_resolve_to_the_registered_function(
    rank_runtime, operator, function
):
    assert runner._load_candidate(operator) is getattr(rank_runtime.package, function)
