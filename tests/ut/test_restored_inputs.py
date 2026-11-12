#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""State restoration, real invocation ownership and fail-closed control tests."""
import csv
from pathlib import Path

import pytest
import torch

from kernel_eval.config import Config
from kernel_eval.base.perf_strategy import ProfFileLocations
from kernel_eval.eval.perf_eval import PerfEvaluator
from kernel_eval.eval.restored_inputs import RestoredCall, PerfIntegrityError, clone_call_inputs
from kernel_eval.security.torch_op_guard import _pause_state


def test_restores_legal_inplace_every_call_and_preserves_nested_tensor_identity():
    x = torch.tensor([1., float('nan'), -0.])
    initial = x.view(torch.uint8).clone()
    def candidate(*, tree):
        assert tree['a'][0] is tree['b']
        tree['b'].fill_(4)
        return tree['b']
    call = RestoredCall(candidate, (), {'tree': {'a': (x,), 'b': x}})
    for index in range(13):
        call.prepare(index, 'measured')
        assert torch.equal(call.kwargs['tree']['b'].view(torch.uint8), initial)
        assert torch.equal(call(), torch.full((3,), 4.))
        with pytest.raises(PerfIntegrityError, match='freshly prepared'):
            call()
        call.retire()
    call.release()
    assert torch.equal(x.view(torch.uint8), initial)


@pytest.mark.parametrize('builder', [
    lambda: (lambda x: [x, x.view_as(x)])(torch.ones(3)),
    lambda: [torch.ones(2, 3).t()],
    lambda: [torch.ones(4)[1:]],
    lambda: [torch.ones(4)[:2]],
    lambda: [object()],
])
def test_unsupported_storage_and_objects_fail_before_candidate(builder):
    with pytest.raises(PerfIntegrityError):
        clone_call_inputs(builder(), {})


def test_cyclic_container_rejected():
    x = []
    x.append(x)
    with pytest.raises(PerfIntegrityError, match='cyclic'):
        clone_call_inputs((x,), {})


def mock_evaluator(tmp_path):
    config = Config(perf_restore_inputs=True, reports_dir=str(tmp_path))
    evaluator = PerfEvaluator(config, freq_boost=False, archive_prof=False)
    evaluator._synchronize_profile_step = lambda: None
    windows = []
    def profile(fn, directory, warmup, repeat, **kwargs):
        assert warmup == 0 and repeat == 1
        assert kwargs == {'run_preflight': False, 'condition_steps': False}
        windows.append('enter')
        fn()
        windows.append('exit')
        path = Path(directory) / 'kernel_details.csv'
        with path.open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=['Name', 'Type', 'Duration(us)', 'Task ID'])
            writer.writeheader()
            # Repeated launches must be summed; candidate names must not be filtered.
            for task, name, duration in [(1, 'same', 2), (2, 'same', 3), (3, 'CannBenchCopy', 5), (4, 'CannBenchWarmup', 7)]:
                writer.writerow({'Name': name, 'Type': name, 'Duration(us)': duration, 'Task ID': task})
        (Path(directory) / 'api_statistic.csv').write_text('API Name,Count\naclrtLaunchKernel,4\n')
    evaluator._profile = profile
    evaluator._locate_prof_files = lambda p: ProfFileLocations(
        csv_path=str(Path(p) / 'kernel_details.csv'),
        api_statistic_path=str(Path(p) / 'api_statistic.csv'))
    return evaluator, windows


def test_real_measured_output_checked_outside_windows_and_each_input_restored(tmp_path):
    evaluator, windows = mock_evaluator(tmp_path)
    seen = []
    x = torch.ones(3)
    def candidate(x):
        assert not getattr(_pause_state, 'paused', False)
        seen.append(x.clone())
        x.add_(2)
        return x
    validations = []
    def validator(output, args, kwargs, index, phase):
        assert getattr(_pause_state, 'paused', False)
        assert not windows or windows[-1] == 'exit'
        assert output is args[0]
        assert torch.equal(output, torch.full((3,), 3.))
        validations.append((index, phase))
    output, result = evaluator.run_profiled('level1/test_1', candidate, x,
                                           warmup=3, repeat=5, validator=validator)
    assert result.is_success(), result.error_msg
    assert len(seen) == 9 and all(torch.equal(t, x) for t in seen)
    assert len(validations) == 9
    assert result.elapsed_us == 17
    assert result.op_times['device_kernels']['same'] == 5
    assert result.metadata['measured_calls'] == result.metadata['validated_calls'] == 5
    assert result.metadata['step_kernel_us'] == [17] * 5
    assert torch.equal(output, torch.full((3,), 3.))
    assert not list(tmp_path.rglob('kernel_details.csv'))


def test_input_factory_is_outside_capture_and_binds_to_actual_output(tmp_path):
    evaluator, windows = mock_evaluator(tmp_path)
    def factory(index, phase):
        assert not windows or windows[-1] == 'exit'
        return (torch.full((3,), index, dtype=torch.float32),), {}
    def validator(output, args, kwargs, index, phase):
        assert torch.equal(output, torch.full((3,), index + 1.))
    _, result = evaluator.run_profiled('level1/test_1', lambda x: x + 1,
                                      torch.ones(3), warmup=0, repeat=2,
                                      input_factory=factory, validator=validator)
    assert result.is_success(), result.error_msg
    assert result.metadata['prepared_calls'] == 3


@pytest.mark.parametrize('steps,expected_us', [
    ([[('A', 100)], [('B', 100)]], 100),
    ([[('A', 100)], [('B', 100)], [('C', 100)]], 100),
    ([[('A', 100), ('B', 1)], [('A', 1), ('B', 100)], [('A', 1), ('B', 1)]], 101),
    ([[('A', 8)], [('A', 10)], [('A', 900)]], 10),
    ([[('A', 40), ('A', 60)], [('B', 100)]], 100),
])
def test_elapsed_aggregates_complete_invocations_with_changing_kernels(
    tmp_path, steps, expected_us,
):
    evaluator, _ = mock_evaluator(tmp_path)
    profile = evaluator._profile
    measured = iter(steps)

    def profile_with_kernels(fn, directory, **kwargs):
        profile(fn, directory, **kwargs)
        with (Path(directory) / 'kernel_details.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Name', 'Duration(us)', 'Task ID'])
            for task_id, (name, duration) in enumerate(next(measured)):
                writer.writerow([name, duration, task_id])

    evaluator._profile = profile_with_kernels

    def factory(index, phase):
        return (torch.full((1,), float(index)),), {}

    _, result = evaluator.run_profiled(
        'level1/test_1', lambda x: x + 1, torch.ones(1),
        warmup=2, repeat=len(steps), input_factory=factory,
    )
    assert result.is_success(), result.error_msg
    assert result.elapsed_us == expected_us
    totals = [sum(duration for _, duration in step) for step in steps]
    assert result.metadata['step_kernel_us'] == totals
    invocations = result.metadata['perf_restore_invocations']
    assert [step['candidate_elapsed_us'] for step in invocations[3:]] == totals
    assert all(step['candidate_elapsed_us'] is None for step in invocations[:3])
    assert result.metadata['elapsed_us_aggregation'] == 'median_of_invocation_totals'
    assert result.metadata['device_kernels_aggregation'] == 'median_when_present_diagnostic_only'
    assert result.to_dict()['elapsed_us'] == expected_us


def test_failed_middle_output_is_sticky_and_not_retried(tmp_path):
    evaluator, windows = mock_evaluator(tmp_path)
    calls = []
    def candidate(x):
        calls.append(1)
        return x
    def validator(output, args, kwargs, index, phase):
        if index == 2:
            raise PerfIntegrityError('wrong measured output')
    output, result = evaluator.run_profiled('level1/test_1', candidate, torch.ones(1),
                                           warmup=0, repeat=5, validator=validator)
    assert output is None and result.elapsed_us == 0
    assert result.metadata['perf_integrity_failed']
    assert len(calls) == 3  # preflight + two measurements, no retry
    assert result.metadata['measured_calls'] == 2


def test_strict_mode_rejects_disabled_profiling_and_opaque_batch(tmp_path):
    evaluator, _ = mock_evaluator(tmp_path)
    evaluator.config.enable_profiler = False
    _, result = evaluator.run_profiled('level1/test_1', lambda x: x, torch.ones(1))
    assert result.metadata['perf_integrity_failed']
    with pytest.raises(PerfIntegrityError, match='opaque batch'):
        evaluator.run_profiled_batch([('level1/test_1', lambda: None)])


def test_hooks_cannot_silently_run_without_restoration(tmp_path):
    evaluator, _ = mock_evaluator(tmp_path)
    evaluator.config.perf_restore_inputs = False
    with pytest.raises(PerfIntegrityError, match='require perf_restore_inputs'):
        evaluator.run_profiled('level1/test_1', lambda: None, validator=lambda *args: None)


def test_input_preparation_and_conditioning_precede_capture(tmp_path):
    evaluator, windows = mock_evaluator(tmp_path)
    evaluator.freq_boost = True
    evaluator._prepare_warmup_tensors = lambda: None
    evaluator._clear_cache = lambda: windows.append('condition')
    original_profile = evaluator._profile
    def profile(*args, **kwargs):
        assert windows[-1] == 'condition'
        return original_profile(*args, **kwargs)
    evaluator._profile = profile
    output, result = evaluator.run_profiled('level1/test_1', lambda x: x,
                                           torch.ones(1), warmup=0, repeat=2)
    assert result.is_success(), result.error_msg
    assert windows == ['condition', 'enter', 'exit', 'condition', 'enter', 'exit']


def test_shared_mutable_container_identity_is_preserved():
    shared = [torch.ones(3)]
    args, kwargs, count = clone_call_inputs((shared,), {'shared': shared})
    assert args[0] is kwargs['shared']
    assert args[0] is not shared
    assert count == 12


def test_missing_memcpy_evidence_fails_closed(tmp_path):
    evaluator, _ = mock_evaluator(tmp_path)
    locate = evaluator._locate_prof_files
    def missing_api(path):
        files = locate(path)
        files.api_statistic_path = None
        return files
    evaluator._locate_prof_files = missing_api
    _, result = evaluator.run_profiled('level1/test_1', lambda x: x,
                                      torch.ones(1), warmup=0, repeat=1)
    assert result.metadata['perf_integrity_failed']
    assert 'api_statistic.csv' in result.error_msg


@pytest.mark.parametrize('rows,repeat,accepted,measured_calls', [
    ([('aclrtMemcpy', 1), ('aclrtMemcpyAsync', 1)], 5, True, 5),
    ([('aclrtMemcpy', 5)], 1, True, 1),
    ([('aclrtMemcpy', 2)], 5, False, 3),
    ([('aclrtMemcpyAsync', 6)], 1, False, 1),
    ([('aclrtMemcpy', 3), ('aclrtMemcpy', 3)], 1, False, 1),
])
def test_memcpy_limit_is_per_api_across_all_measured_calls(
    tmp_path, rows, repeat, accepted, measured_calls,
):
    evaluator, _ = mock_evaluator(tmp_path)
    profile = evaluator._profile

    def profile_with_memcpy(fn, directory, **kwargs):
        profile(fn, directory, **kwargs)
        with (Path(directory) / 'api_statistic.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['API Name', 'Count'])
            writer.writerows(rows)

    evaluator._profile = profile_with_memcpy
    calls = []

    def candidate(x):
        calls.append(1)
        return x

    output, result = evaluator.run_profiled(
        'level1/test_1', candidate, torch.ones(1), warmup=2, repeat=repeat,
    )
    assert result.is_success() == accepted, result.error_msg
    assert result.metadata['measured_calls'] == measured_calls
    assert len(calls) == 3 + measured_calls  # preflight, two warmups; no retry
    counts = {}
    for name, count in rows:
        counts[name] = counts.get(name, 0) + count
    assert result.metadata['candidate_memcpy_counts'] == {
        name: count * measured_calls for name, count in counts.items()
    }
    assert result.metadata['candidate_memcpy_count'] == sum(counts.values()) * measured_calls
    invocations = result.metadata['perf_restore_invocations']
    assert all(step['candidate_memcpy_counts'] is None for step in invocations[:3])
    assert all(step['candidate_memcpy_counts'] == counts for step in invocations[3:])
    if not accepted:
        assert output is None and result.elapsed_us == 0
        assert result.metadata['perf_integrity_failed']
        assert 'excessive aclrtMemcpy' in result.error_msg


def test_cli_and_staged_configuration_keep_flag_for_performance_children(tmp_path):
    from kernel_eval.cli import create_parser, _create_config_from_args_for_child
    from kernel_eval.staged_eval import create_parser as staged_parser, _make_config
    parser = create_parser()
    assert parser.parse_args(['eval', '--perf-restore-inputs']).perf_restore_inputs
    child = parser.parse_args(['eval-child', '--device-id', '0', '--cases-file', 'cases.json', '--output', 'out.json',
                               '--perf-restore-inputs'])
    assert _create_config_from_args_for_child(child, str(tmp_path)).perf_restore_inputs
    staged = staged_parser().parse_args(['--perf-restore-inputs'])
    assert _make_config(staged, str(tmp_path), enable_profiler=False).perf_restore_inputs
    assert _make_config(staged, str(tmp_path), enable_profiler=True).perf_restore_inputs
