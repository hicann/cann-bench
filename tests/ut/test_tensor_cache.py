#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Real tensor round trips and evaluator equivalence, without an NPU."""

from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("safetensors")

from kernel_eval.benches.cann_spec import CannCaseSpec
from kernel_eval.config import Config
from kernel_eval.data.tensor_cache import (
    DTYPES, TensorCache, TensorCacheError, identity_for, selected, validate_config,
)
from kernel_eval.eval.evaluator import Evaluator


@pytest.fixture
def case():
    return CannCaseSpec(case_id="level4/linear_cross_entropy_18", case_num=18, operator="LinearCrossEntropy",
                        rel_path="level4/linear_cross_entropy",
                        input_shapes=[[2, 4], [7, 4], [2]],
                        dtypes=["bfloat16", "bfloat16", "int32"],
                        value_ranges=[[-1, 1], [-.02, .02], [0, 6]],
                        attrs={"reduction": "mean", "ignore_index": -100})


@pytest.fixture
def evaluator(tmp_path, case):
    config = Config(device_type="cpu", enable_profiler=False, torch_op_guard_mode="off",
                    tensor_cache_dir=str(tmp_path / "bank"), tensor_cache_mode="prefer",
                    tensor_cache_tasks=[case.rel_path])
    evaluator = Evaluator(config)
    evaluator._cleanup_memory = lambda: None
    return evaluator


def minimal_case(dtype="float32", shape=None):
    return SimpleNamespace(input_shapes=[[2] if shape is None else shape], dtypes=[dtype])


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_dtypes_and_scalar_round_trip(tmp_path, dtype):
    dtype_object = getattr(torch, dtype)
    inputs = [torch.tensor([0, 1], dtype=dtype_object)]
    case = minimal_case(dtype)
    cache = TensorCache(tmp_path)
    identity = {"case": "dtype", "dtype": dtype}
    oracle = torch.tensor(1.25, dtype=torch.float64)
    native = torch.tensor(1.125, dtype=torch.float32)
    path = cache.write(identity, case, inputs, oracle, native)
    loaded = cache.load(identity, case)
    assert loaded.inputs[0].dtype == dtype_object
    assert torch.equal(loaded.inputs[0], inputs[0])
    assert loaded.oracle.shape == torch.Size([])
    assert loaded.oracle.dtype == torch.float64 and loaded.oracle.item() == 1.25
    assert loaded.native.dtype == torch.float32 and loaded.native.item() == 1.125
    assert (path / "manifest.json").is_file()
    loaded.inputs[0].zero_()
    assert torch.equal(cache.load(identity, case).inputs[0], inputs[0])


def test_optional_empty_nested_outputs_and_identical_reference_dedup(tmp_path):
    case = SimpleNamespace(input_shapes=[[0, 2], None], dtypes=["float32", "float32"])
    inputs = [torch.empty((0, 2)), None]
    oracle = (torch.tensor([1, 2], dtype=torch.int64), [None, torch.empty(0), True])
    native = (oracle[0].clone(), [None, torch.empty(0), True])
    cache = TensorCache(tmp_path)
    path = cache.write({}, case, inputs, oracle, native)
    loaded = cache.load({}, case)
    assert loaded.inputs[0].shape == (0, 2) and loaded.inputs[1] is None
    assert isinstance(loaded.oracle, tuple) and isinstance(loaded.oracle[1], list)
    assert loaded.oracle[1][0] is None and loaded.oracle[1][1].numel() == 0
    assert torch.equal(loaded.oracle[0], loaded.native[0])
    assert loaded.oracle[0].data_ptr() != loaded.native[0].data_ptr()
    manifest = json.loads((path / "manifest.json").read_text())
    assert len(manifest["files"]["references.safetensors"]["tensors"]) == 2


@pytest.mark.parametrize("damage", ["bytes", "version", "identity", "tree", "dtype", "missing-native", "partial"])
def test_corruption_never_becomes_a_miss(tmp_path, damage):
    cache = TensorCache(tmp_path)
    case = minimal_case()
    x = torch.ones(2)
    path = cache.write({}, case, [x], x.double(), x)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "bytes":
        p = path / "inputs.safetensors"
        data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
    elif damage == "version":
        manifest["version"] = 99
    elif damage == "identity":
        manifest["identity"] = {"different": True}
    elif damage == "tree":
        manifest["files"]["inputs.safetensors"]["tree"] = {"object": "unsafe"}
    elif damage == "dtype":
        manifest["files"]["inputs.safetensors"]["tensors"]["tensor_000000"]["dtype"] = "torch.int64"
    elif damage == "missing-native":
        manifest["files"]["references.safetensors"]["tree"]["tuple"][1] = {"scalar": None}
    else:
        (path / "references.safetensors").unlink()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TensorCacheError):
        cache.load({}, case)
    assert cache.load({"another-seed": 1}, case) is None


def test_input_contract_and_unsupported_layouts(tmp_path):
    cache = TensorCache(tmp_path)
    x = torch.ones(2)
    cache.write({}, minimal_case(), [x], x.double(), x)
    with pytest.raises(TensorCacheError, match="shape/dtype"):
        cache.load({}, minimal_case("int32"))
    with pytest.raises(TensorCacheError, match="contiguous"):
        cache.write({}, minimal_case(shape=[2, 2]), [torch.ones(2, 2).t()], x, x)
    aliased = SimpleNamespace(input_shapes=[[2], [2]], dtypes=["float32", "float32"])
    with pytest.raises(TensorCacheError, match="aliased"):
        cache.write({}, aliased, [x, x], x, x)


def test_failed_publication_leaves_no_partial_entry(tmp_path):
    cache = TensorCache(tmp_path)
    x = torch.ones(2)
    with patch("kernel_eval.data.tensor_cache.os.rename", side_effect=OSError("interrupted")):
        with pytest.raises(OSError, match="interrupted"):
            cache.write({}, minimal_case(), [x], x, x)
    assert list(tmp_path.iterdir()) == []


def test_existing_entry_is_not_overwritten(tmp_path):
    cache = TensorCache(tmp_path)
    x = torch.ones(2)
    path = cache.write({}, minimal_case(), [x], x, x)
    before = (path / "manifest.json").read_bytes()
    cache.write({}, minimal_case(), [x * 2], x * 2, x * 2)
    assert (path / "manifest.json").read_bytes() == before
    assert torch.equal(cache.load({}, minimal_case()).inputs[0], x)
    assert not list(tmp_path.glob(".building-*"))


def test_identity_invalidates_contract_source_seed_distribution_and_policy(evaluator, case, tmp_path):
    before = identity_for(evaluator, case)
    assert before != identity_for(evaluator, replace(case, attrs={"reduction": "sum", "ignore_index": -100}))
    evaluator.config.eval_seed = 7
    assert before != identity_for(evaluator, case)
    evaluator.config.eval_seed = 0
    evaluator.data_generator.input_dist = "normal"
    assert before != identity_for(evaluator, case)
    evaluator.data_generator.input_dist = "uniform"
    with patch.object(evaluator.bench_config, "golden_precision", "native_cpu"):
        assert before != identity_for(evaluator, case)
    task = tmp_path / case.rel_path
    task.mkdir(parents=True)
    (task / "golden.py").write_text("# changed golden source\n")
    evaluator.config.tasks_root = str(tmp_path)
    assert before != identity_for(evaluator, case)


def test_source_digest_reuse_and_edit_add_remove_replace_invalidation(evaluator, case, tmp_path, monkeypatch):
    import kernel_eval.data.tensor_cache as cache_module
    package = tmp_path / "package"
    module = package / "data" / "tensor_cache.py"
    module.parent.mkdir(parents=True)
    module.write_text("# original\n")
    monkeypatch.setattr(cache_module, "__file__", str(module))
    task_root = tmp_path / "tasks"
    task = task_root / case.rel_path
    task.mkdir(parents=True)
    golden = task / "golden.py"
    golden.write_text("# golden A\n")
    evaluator.config.tasks_root = str(task_root)
    with patch.object(cache_module, "file_sha256", wraps=cache_module.file_sha256) as hash_file:
        first = identity_for(evaluator, case)
        reads = hash_file.call_count
        second = identity_for(evaluator, replace(case, case_num=19))
        assert first != second and first["sources"] == second["sources"]
        assert hash_file.call_count == reads  # no source bytes reread for the next case

        # Same-size harness edit with a restored mtime must still invalidate.
        previous = module.stat()
        module.write_text("# modified\n")
        os.utime(module, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        edited = identity_for(evaluator, case)
        assert edited != first and hash_file.call_count == reads + 1

        helper = task / "helper.py"
        helper.write_text("# helper\n")
        added = identity_for(evaluator, case)
        assert added != edited and "task/helper.py" in added["sources"]
        helper.unlink()
        assert identity_for(evaluator, case) == edited

        # Atomic replacement with the original size/mtime also invalidates.
        previous = golden.stat()
        replacement = task / "replacement.tmp"
        replacement.write_text("# golden B\n")
        os.utime(replacement, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        replacement.replace(golden)
        assert identity_for(evaluator, case) != edited


def test_source_change_during_hashing_is_rejected(tmp_path):
    import kernel_eval.data.tensor_cache as cache_module
    source = tmp_path / "golden.py"
    source.write_text("# before\n")
    real_hash = cache_module.file_sha256
    def changing_hash(path):
        digest = real_hash(path)
        path.write_text("# after\n")
        return digest
    with patch.object(cache_module, "file_sha256", side_effect=changing_hash):
        with pytest.raises(TensorCacheError, match="Source changed"):
            cache_module.source_sha256(source)
    assert cache_module.source_sha256(source) == real_hash(source)


def test_selection_and_random_mode_rejection(evaluator, case):
    assert selected(evaluator.config, case)
    assert not selected(evaluator.config, replace(case, rel_path="level1/exp"))
    evaluator.config.eval_seed = None
    with pytest.raises(TensorCacheError, match="deterministic"):
        selected(evaluator.config, case)
    evaluator.config.tensor_cache_mode = "off"
    assert not selected(evaluator.config, case)
    assert not selected(SimpleNamespace(), case)
    evaluator.config.tensor_cache_tasks = ["../escape"]
    with pytest.raises(TensorCacheError, match="canonical"):
        validate_config(evaluator.config)


def test_cached_lce_matches_uncached_and_skips_generator_references(evaluator, case):
    candidate = evaluator.golden_loader.get_golden_function(case.rel_path)
    evaluator.config.tensor_cache_mode = "off"
    original = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert original.success, original.error_msg
    evaluator.build_tensor_cache(case)
    evaluator.config.tensor_cache_mode = "require"
    with patch.object(evaluator.data_generator, "generate_input_tensors_from_case", side_effect=AssertionError("generated")), \
         patch.object(evaluator, "_oracle_reference_func", side_effect=AssertionError("oracle executed")), \
         patch.object(evaluator, "_bench_reference_func", side_effect=AssertionError("native executed")):
        cached = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert cached.success, cached.error_msg
    assert original.accuracy_result.metadata == cached.accuracy_result.metadata
    assert cached.golden_run_result.elapsed_us == 0  # no fake reference timing


def test_require_miss_corruption_and_prefer_fallback(evaluator, case):
    candidate = evaluator.golden_loader.get_golden_function(case.rel_path)
    evaluator.config.tensor_cache_mode = "require"
    with patch.object(evaluator.data_generator, "generate_input_tensors_from_case", side_effect=AssertionError("generated")):
        result = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert not result.success and "Required tensor cache missing" in result.error_msg
    evaluator.config.tensor_cache_mode = "prefer"
    assert evaluator.evaluate_case(case, ai_op_func=candidate).success
    assert not Path(evaluator.config.tensor_cache_dir).exists()  # evaluation is read-only
    path = evaluator.build_tensor_cache(case)
    (path / "inputs.safetensors").write_bytes(b"broken")
    result = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert not result.success and "checksum mismatch" in result.error_msg


def test_selected_unsupported_get_input_fails(evaluator, case):
    with patch.object(evaluator.golden_loader, "get_input_function", return_value=lambda: []):
        with pytest.raises(TensorCacheError, match="get_input"):
            evaluator.build_tensor_cache(case)


@pytest.mark.parametrize("mode", ["prefer", "require"])
def test_get_input_hook_falls_back_only_in_prefer(evaluator, case, mode, capsys):
    candidate = evaluator.golden_loader.get_golden_function(case.rel_path)
    def get_input(hidden, weight, labels, **kwargs):
        # Exercise a real layout transformation through the existing hook path.
        return [hidden.t().contiguous().t(), weight, labels]
    with patch.object(evaluator.golden_loader, "get_input_function", return_value=get_input):
        evaluator.config.tensor_cache_mode = "off"
        original = evaluator.evaluate_case(case, ai_op_func=candidate)
        assert original.success, original.error_msg
        evaluator.config.tensor_cache_mode = mode
        with patch.object(TensorCache, "load", side_effect=AssertionError("unsupported cache read")):
            result = evaluator.evaluate_case(case, ai_op_func=candidate)
    if mode == "prefer":
        assert result.success, result.error_msg
        assert result.accuracy_result.metadata == original.accuracy_result.metadata
        assert "computing without cache" in capsys.readouterr().out
    else:
        assert not result.success and "get_input" in result.error_msg
    assert not Path(evaluator.config.tensor_cache_dir).exists()


@pytest.mark.parametrize("mode", ["prefer", "require"])
@pytest.mark.parametrize("unsupported", ["native_npu", "other_bench"])
def test_unsupported_reference_policy_falls_back_only_in_prefer(evaluator, case, mode, unsupported):
    # Test dispatch on CPU; this does not claim to execute a native NPU reference.
    candidate = evaluator.golden_loader.get_golden_function(case.rel_path)
    owner, attribute, value = ((evaluator.bench_config, "golden_precision", "native_npu")
                               if unsupported == "native_npu" else (evaluator, "bench_name", "other"))
    with patch.object(owner, attribute, value):
        evaluator.config.tensor_cache_mode = "off"
        original = evaluator.evaluate_case(case, ai_op_func=candidate)
        assert original.success, original.error_msg
        evaluator.config.tensor_cache_mode = mode
        with patch.object(TensorCache, "load", side_effect=AssertionError("unsupported cache read")):
            result = evaluator.evaluate_case(case, ai_op_func=candidate)
        with pytest.raises(TensorCacheError, match="CPU references only"):
            evaluator.build_tensor_cache(case)
    if mode == "prefer":
        assert result.success, result.error_msg
        assert result.accuracy_result.metadata == original.accuracy_result.metadata
    else:
        assert not result.success and "CPU references only" in result.error_msg


@pytest.mark.parametrize("invalid", ["missing_source", "cache_inside_source"])
def test_prefer_does_not_hide_invalid_identity(evaluator, case, tmp_path, invalid):
    if invalid == "missing_source":
        evaluator.config.tasks_root = str(tmp_path / "missing")
        message = "Cannot fingerprint task source"
    else:
        evaluator.config.tensor_cache_dir = str(Path(evaluator.config.tasks_root) / case.rel_path / "cache")
        message = "outside the selected task source"
    with patch.object(evaluator.data_generator, "generate_input_tensors_from_case", side_effect=AssertionError("generated")):
        result = evaluator.evaluate_case(case)
    assert not result.success and message in result.error_msg


def test_reference_mutation_is_rejected(evaluator, case):
    golden = evaluator.golden_loader.get_golden_function(case.rel_path)
    def mutating_reference(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                           reduction="mean", ignore_index=-100):
        result = golden(hidden, weight, labels, reduction, ignore_index)
        hidden.zero_()
        return result
    with patch.object(evaluator, "_bench_reference_func", return_value=mutating_reference):
        with pytest.raises(TensorCacheError, match="mutate"):
            evaluator.build_tensor_cache(case)


@pytest.mark.parametrize("mode", ["off", "prefer", "require"])
def test_distributed_selection_delegates_unless_required(evaluator, case, mode, capsys):
    evaluator.config.tensor_cache_mode = mode
    distributed = replace(case, attrs={"mc2_distributed": True})
    with patch("kernel_eval.eval.mc2_distributed_runner.MC2DistributedEvaluator") as runner:
        result = evaluator.evaluate_case(distributed)
        if mode == "require":
            runner.assert_not_called()
            assert not result.success and "distributed" in result.error_msg
            assert result.case_num == 18 and type(result.case_num) is int
            assert result.case_id == case.get_case_id_str()
        else:
            runner.assert_called_once_with(evaluator.config)
            runner.return_value.evaluate_case.assert_called_once_with(
                distributed, custom_thresholds=evaluator._get_merged_thresholds(case.rel_path))
            assert result is runner.return_value.evaluate_case.return_value
            if mode == "prefer":
                assert "computing without cache" in capsys.readouterr().out


def test_build_cli_filters_cases_without_submission(evaluator, case, tmp_path):
    import shutil
    import yaml
    from kernel_eval.cache_cli import create_parser, run
    task_root = tmp_path / "tasks"
    task = task_root / case.rel_path
    task.mkdir(parents=True)
    original = Path(evaluator.config.tasks_root) / case.rel_path
    for name in ("golden.py", "proto.yaml"):
        shutil.copyfile(original / name, task / name)
    definition = dict(operator=case.operator, case_id=18, input_shape=case.input_shapes,
                      dtype=case.dtypes, value_range=case.value_ranges, attrs=case.attrs)
    (task / "cases.yaml").write_text(yaml.safe_dump({"cases": [definition, dict(definition, case_id=19)]}))
    args = create_parser().parse_args(["build", "--tasks-root", str(task_root), "--task", case.rel_path,
                                      "--case-id", "18", "--cache-dir", str(tmp_path / "built")])
    with patch("kernel_eval.eval.evaluator.Evaluator.load_ai_operator", side_effect=AssertionError("candidate loaded")):
        assert run(args) == 0
    entries = list((tmp_path / "built").glob("*/manifest.json"))
    assert len(entries) == 1
    assert json.loads(entries[0].read_text())["identity"]["case_id"].endswith("_18")


def test_fresh_retry_does_not_reuse_cache(evaluator, case):
    candidate = evaluator.golden_loader.get_golden_function(case.rel_path)
    evaluator.build_tensor_cache(case)
    evaluator.config.tensor_cache_mode = "require"
    evaluator.config.enable_accuracy_retry = True
    with patch.object(evaluator.data_generator, "generate_input_tensors_from_case",
                      wraps=evaluator.data_generator.generate_input_tensors_from_case) as generate, \
         patch.object(evaluator, "_oracle_reference_func", wraps=evaluator._oracle_reference_func) as oracle:
        result = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert result.success, result.error_msg
    assert generate.call_count == 1 and oracle.call_count == 1


def test_native_calibration_survives_storage(tmp_path):
    from kernel_eval.utils.compare import _compare_single_tensor
    generator = torch.Generator().manual_seed(0)
    g = torch.cat((torch.ones(2048, dtype=torch.float64),
                   1e-6 * (torch.rand(2048, generator=generator, dtype=torch.float64) + .5)))
    def noisy(seed):
        out = g.clone()
        noise = torch.Generator().manual_seed(seed)
        out[2048:] += 3e-8 * torch.randn(2048, generator=noise, dtype=torch.float64)
        return out.float()
    candidate, native = noisy(11), noisy(22)
    cache = TensorCache(tmp_path)
    cache.write({}, minimal_case(), [torch.ones(2)], g, native)
    loaded = cache.load({}, minimal_case())
    original = _compare_single_tensor(candidate, g, 2**-13, "float32", native_output=native)
    restored = _compare_single_tensor(candidate, loaded.oracle, 2**-13, "float32", native_output=loaded.native)
    cast_only = _compare_single_tensor(candidate, g, 2**-13, "float32", native_output=g.float())
    assert original.passed and restored.passed and not cast_only.passed
    assert original.small_value_cpu_error_count == restored.small_value_cpu_error_count > 0


def test_cli_worker_and_staged_propagation(evaluator, case):
    from kernel_eval.cli import create_parser, _create_config_from_args, _create_config_from_args_for_child
    from kernel_eval.staged_eval import create_parser as staged_parser, _make_config
    from kernel_eval.eval.process_pool import ProcessPoolCoordinator, TaskUnit
    flags = ["--tensor-cache-dir", evaluator.config.tensor_cache_dir, "--tensor-cache-mode", "require",
             "--tensor-cache-task", case.rel_path, "--tensor-cache-task", "level1/exp"]
    parser = create_parser()
    args = parser.parse_args(["eval", "--device", "cpu"] + flags)
    config = _create_config_from_args(args, evaluator.config.tasks_root)
    coordinator = ProcessPoolCoordinator(base_config=config)
    task = TaskUnit(operator=case.operator, rel_path=case.rel_path, cases=[case], device_id=0)
    command = coordinator._build_child_cmd(task, "/tmp/cases.json", "/tmp/results.json")
    args = parser.parse_args(command[command.index("eval-child"):])
    child = _create_config_from_args_for_child(args, evaluator.config.tasks_root)
    assert child.tensor_cache_mode == "require" and child.tensor_cache_tasks == config.tensor_cache_tasks
    staged = _make_config(staged_parser().parse_args(flags), evaluator.config.tasks_root, enable_profiler=True)
    assert staged.tensor_cache_dir == config.tensor_cache_dir and staged.tensor_cache_tasks == config.tensor_cache_tasks


@pytest.mark.parametrize("wrong_dtype", [False, True])
def test_bad_candidate_still_fails_with_cache(evaluator, case, wrong_dtype):
    reference = evaluator.golden_loader.get_golden_function(case.rel_path)
    evaluator.build_tensor_cache(case)
    evaluator.config.tensor_cache_mode = "require"
    def candidate(**kwargs):
        value = reference(**kwargs)
        return value.to(torch.float64) if wrong_dtype else value + 10
    result = evaluator.evaluate_case(case, ai_op_func=candidate)
    assert not result.success
    assert result.accuracy_result is not None and not result.accuracy_result.passed
