# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import json
import csv
import weakref
from pathlib import Path
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import torch
import pytest

from ascend_arena_e2e.ascend_arena.runners import profile as profile_module
from kernel_eval.eval.input_pool import InputPool


@pytest.mark.parametrize("source", [torch.arange(4, dtype=torch.int32), torch.arange(8.0).reshape(2, 4).t()])
@pytest.mark.parametrize("batch", [False, True])
def test_both_measurements_rotate_equal_inputs_without_changing_sampling(monkeypatch, tmp_path, source, batch):
    monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", "page_size:1g")
    calls = []
    prepared_inputs = []
    pool_sizes = []
    pool_refs = []
    session_count = []

    class PoolSpy(InputPool):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pool_sizes.append(self.size())
            pool_refs.append(weakref.ref(self))

    monkeypatch.setattr(sys.modules[InputPool.__module__], "InputPool", PoolSpy)

    class Model:
        def to(self, device):
            return self

        def __call__(self, value):
            calls.append((value.data_ptr(), value.tolist()))
            value.add_(1)
            return value

    def prepare(**kwargs):
        value = source.clone()
        prepared_inputs.append(value)
        return SimpleNamespace(init_inputs_device=[], inputs_device=[value])

    @contextmanager
    def profile(**kwargs):
        count = 2 if batch else 1
        assert kwargs['schedule'] == (0, count * 40)
        assert len(pool_refs) == 4 and all(ref() is not None for ref in pool_refs)
        session_count.append(count)
        target = Path(kwargs['on_trace_ready'])
        target.mkdir(parents=True, exist_ok=True)
        yield SimpleNamespace(step=lambda: None)
        with (target / 'kernel_details.csv').open('w') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Name', 'Type', 'Duration(us)', 'Start Time(us)'])
            for i in range(count * 40):
                writer.writerow(['CannBenchCacheClean', 'N/A', 2, i * 10])
                writer.writerow(['Target', 'N/A', 3, i * 10 + 3])

    def experimental_config(**kwargs):
        assert kwargs['profiler_level'] == 0  # Level1 promotes AiCoreNone to PipeUtilization.
        assert kwargs['aic_metrics'] == 0

    @contextmanager
    def measure(torch_api, samples, inner_iters):
        yield
        samples.append(1.0)

    helpers = ModuleType('cann_bench_utils')
    helpers.__path__ = []
    helpers.cann_bench_warmup = lambda *args: None
    helpers.cann_bench_cache_clean = lambda *args: None
    build_info = ModuleType('cann_bench_utils.build_info')
    build_info.check_build_compatibility = lambda **kwargs: {'cann': '9.1.0'}
    monkeypatch.setitem(sys.modules, 'cann_bench_utils', helpers)
    monkeypatch.setitem(sys.modules, 'cann_bench_utils.build_info', build_info)
    sync = SimpleNamespace(synchronize=lambda *args: None)
    profiler = SimpleNamespace(
        _ExperimentalConfig=experimental_config,
        ProfilerLevel=SimpleNamespace(Level0=0, Level1=1), AiCMetrics=SimpleNamespace(AiCoreNone=0),
        ExportType=SimpleNamespace(Text=0), ProfilerActivity=SimpleNamespace(NPU=0),
        profile=profile, schedule=lambda **kwargs: (kwargs['warmup'], kwargs['active']),
        tensorboard_trace_handler=lambda path: path,
    )
    runtime = {
        'torch': SimpleNamespace(rand=lambda *args, **kwargs: None, float16=torch.float16, npu=sync),
        'torch_npu': SimpleNamespace(profiler=profiler, npu=sync),
        'binding_module': SimpleNamespace(ModelNew=Model), 'reference_module': None,
        'inputs_module': None, 'device': 'test', 'policy': None, 'prepare_case_evaluation': prepare,
    }
    monkeypatch.setattr(profile_module, 'measure_operator_call', measure)
    monkeypatch.setattr(profile_module.time, 'sleep', lambda *args: None)
    runner = profile_module.ProfileRunner(profiler='torch-profiler')
    monkeypatch.setattr(runner, '_verify_profile_output', lambda *args: None)
    runner._execute_torch_profiler_cases(output_dir=tmp_path, cases=[{'case_id': 'case_01'}, {'case_id': 'case_02'}], runtime=runtime, batch=batch)
    assert pool_sizes == [40, 40, 40, 40, 1, 1]
    assert session_count == ([2] if batch else [1, 1])
    assert len(calls) == 162  # Two cases, each with two 10+30 phases and fresh correctness.
    for start in [0, 40, 80, 120]:
        addresses = [address for address, _ in calls[start:start + 40]]
        assert len(set(addresses)) == 40
    original_addresses = {value.data_ptr() for value in prepared_inputs}
    assert not original_addresses.intersection(address for address, _ in calls[:160])
    assert all(value == source.tolist() for _, value in calls)
    assert all(torch.equal(value, source) for value in prepared_inputs)
    result = json.loads((tmp_path / 'case_01/operator_timing.json').read_text())
    assert result['sample_count'] == 30
    assert result['measurement_protocol']['runtime_config']['PYTORCH_NPU_ALLOC_CONF'] == 'page_size:1g'
    assert result['measurement_protocol']['input_pool'] == {
        'policy': 'clone_rotation', 'size': 40, 'max_pool_size': 40, 'max_memory_mb': 512,
    }

    for case in ['case_01', 'case_02']:
        result = runner._extract_metrics(tmp_path / case)
        kernel = next(m for m in result if m['duration_scope'] == 'kernel_total')
        assert kernel['sample_count'] == 30
        assert kernel['duration_samples_us'] == [3] * 30
        assert kernel['measurement_protocol']['profiler_session']['input_lifetime'] == 'all_cases_retained_through_collection'
        assert kernel['measurement_protocol']['profiler_session']['mode'] == ('batch' if batch else 'per_case')
