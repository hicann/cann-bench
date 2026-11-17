# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Regression checks for the Arena pipeline, build inputs and invocation timing."""

import json
import re
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from kernel_eval import optimize
from ascend_arena_e2e.ascend_arena.runners.profile import ProfileRunner, write_operator_timings
from tests.ut.test_ascend_arena_auto_pipeline import _make_optimize_main_fixture, FakeRunner

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('override', [False, True])
def test_configured_pipeline_passes_initialized_workspace_baseline(tmp_path, monkeypatch, arena_workspace_source, override):
    from auto_pipeline import core
    from auto_pipeline.generator.ascend_arena import AscendArenaAgent, workspace
    from auto_pipeline.converter.registry import create_converter

    task = core.ConfiguredTask('expand_into_jagged_permute', 'bench_lab/ascend_arena_bench/expand_into_jagged_permute', tmp_path / 'task with spaces')
    baseline = task.workdir / 'kernel_eval/init_results'

    def initialize(prepared):
        assert prepared.reports_dir / 'init_results' == baseline
        baseline.mkdir(parents=True)
        (baseline / 'status.json').write_text('{"success":true}')
        (baseline / 'result.json').write_text('{}')

    expected = tmp_path / 'explicit baseline' if override else baseline
    commands = []

    def execute(command, **kwargs):
        args = optimize.build_parser().parse_args(command[3:])
        assert Path(args.baseline_dir) == expected
        assert Path(args.reports_dir) == task.reports_dir
        assert (baseline / 'status.json').is_file()
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(workspace, '_run_initial_evaluation', initialize)
    monkeypatch.setattr(core, '_run_captured_process', execute)
    client = core.CannBenchClient(ROOT)
    result = core.BenchmarkPipeline(bench_name='ascend_arena', client=client).run_case(
        selector=task.selector, generator=AscendArenaAgent(cann_bench_root=ROOT, runner=FakeRunner(), device_id=1),
        converter=create_converter('ascend-arena', 'ascend_arena', {}), workdir=task.workdir,
        submission_dir=task.submission_dir, reports_dir=task.reports_dir, timeout_sec=10, device_id=1,
        extra_eval_args=[f'--baseline-dir={expected}'] if override else None,
    )
    assert result.ok and len(commands) == 1


def test_candidate_build_tree_tracks_deleted_and_renamed_files(tmp_path, monkeypatch):
    dependency, task, source, registry, original = _make_optimize_main_fixture(tmp_path)
    (original / 'op_host/old_tiling.cpp').write_text('old tiling')
    (original / 'obsolete_helper.h').write_text('old helper')
    (source / 'op_host/new_tiling.cpp').write_text('new tiling')
    reports = []

    def stage(**kwargs):
        tree = kwargs['standard_dir']
        assert not (tree / 'op_host/old_tiling.cpp').exists()
        assert not (tree / 'obsolete_helper.h').exists()
        assert (tree / 'op_host/new_tiling.cpp').read_text() == 'new tiling'
        assert (tree / 'reference.py').is_file() and (tree / 'cases/proto.yaml').is_file()
        return SimpleNamespace(success=True,
            build_result=SimpleNamespace(success=True, log_path=tmp_path / 'compile.log'),
            profile_result=SimpleNamespace(success=True, log_path=tmp_path / 'profile.log', message='test')), {}

    monkeypatch.setattr(optimize, '_preflight_dependencies', lambda **kwargs: None)
    monkeypatch.setattr(optimize, '_resolve_env_script', lambda value: None)
    monkeypatch.setattr(optimize, '_run_stage', stage)
    monkeypatch.setattr(optimize, '_build_report', lambda **kwargs: reports.append(kwargs))
    assert optimize.main(['--source-dir', str(source), '--task-dir', str(task), '--operator', 'angle_v2',
        '--dependency-root', str(dependency), '--ops-json', str(registry), '--reports-dir', str(tmp_path / 'reports'),
        '--no-perf']) == 0
    changes = reports[0]['source_diff']['changed_files']
    assert {'path': 'op_host/old_tiling.cpp', 'status': 'deleted'} in changes
    assert {'path': 'op_host/new_tiling.cpp', 'status': 'added'} in changes


def test_default_sandbox_is_private_short_and_removed(tmp_path, monkeypatch):
    dependency, task, source, registry, _ = _make_optimize_main_fixture(tmp_path)
    sandboxes = []

    def stage(**kwargs):
        sandbox = kwargs['run_root'].parents[1]
        sandboxes.append(sandbox)
        assert sandbox.name.startswith('cannbench_arena_')
        assert sandbox.stat().st_mode & 0o777 == 0o700
        assert not sandbox.is_relative_to(tmp_path / 'reports with spaces')
        return SimpleNamespace(success=True,
            build_result=SimpleNamespace(success=True, log_path=tmp_path / 'compile.log'),
            profile_result=SimpleNamespace(success=True, log_path=tmp_path / 'profile.log', message='test')), {}

    monkeypatch.setattr(optimize, '_preflight_dependencies', lambda **kwargs: None)
    monkeypatch.setattr(optimize, '_resolve_env_script', lambda value: None)
    monkeypatch.setattr(optimize, '_run_stage', stage)
    monkeypatch.setattr(optimize, '_build_report', lambda **kwargs: None)
    args = ['--source-dir', str(source), '--task-dir', str(task), '--operator', 'angle_v2',
            '--dependency-root', str(dependency), '--ops-json', str(registry),
            '--reports-dir', str(tmp_path / 'reports with spaces'), '--no-perf']
    assert optimize.main(args) == 0
    assert optimize.main(args) == 0
    assert len(set(sandboxes)) == 2
    assert all(not path.exists() for path in sandboxes)


def test_custom_opapi_binds_its_own_symbols_and_keeps_sdk_fallback(tmp_path):
    import os
    import shutil
    import sys

    compiler = shutil.which('c++')
    if sys.platform != 'linux' or compiler is None:
        pytest.skip('CANN library loading requires Linux and a C++ compiler')
    header = (ROOT / 'src/ascend_arena_e2e/ascend_arena/runners/validation/pytorch_npu_helper.hpp').read_text()
    loader = header[header.index('inline const char *GetOpApiLibName(void)'):
                    header.index('inline aclScalar *ConvertTensorToAclScaler')]
    sources = {
        'sdk.cpp': 'extern "C" int shared_impl() { return 1; }\n'
                   'extern "C" int sdk_only() { return 3; }\n',
        'custom.cpp': 'extern "C" int shared_impl() { return 2; }\n'
                      'extern "C" int candidate() { return shared_impl(); }\n',
    }
    for name, source in sources.items():
        path = tmp_path / name
        path.write_text(source)
        library = 'libopapi.so' if name == 'sdk.cpp' else 'libcust_opapi.so'
        subprocess.run([compiler, '-shared', '-fPIC', str(path), '-o', str(tmp_path / library)], check=True)
    driver = tmp_path / 'driver.cpp'
    driver.write_text('''#include <dlfcn.h>
#include <unistd.h>
#include <cstdlib>
#include <string>
#include <iostream>
#define CUST_OPAPI_LIB_PATH "libcust_opapi.so"
#define ASCEND_LOGW(...) do {} while (false)
''' + loader + '''
int main() {
    using Function = int (*)();
    auto candidate = reinterpret_cast<Function>(GetOpApiFuncAddr("candidate"));
    auto fallback = reinterpret_cast<Function>(GetOpApiFuncAddr("sdk_only"));
    if (!candidate || !fallback) return 2;
    std::cout << candidate() << " " << fallback() << "\\n";
    return candidate() == 2 && fallback() == 3 ? 0 : 1;
}
''')
    subprocess.run([compiler, str(driver), '-ldl', '-o', str(tmp_path / 'driver')], check=True)
    env = dict(os.environ, KERNEL_GEN_OPS_CUST_OPAPI_PATH=str(tmp_path / 'libcust_opapi.so'),
               LD_LIBRARY_PATH=str(tmp_path) + os.pathsep + os.environ.get('LD_LIBRARY_PATH', ''))
    result = subprocess.run([str(tmp_path / 'driver')], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == '2 3'


def test_multikernel_latency_uses_complete_invocation_and_trajectory(tmp_path):
    from kernel_eval.optimization_trajectory import _initialized_baselines
    runner = ProfileRunner(repeat=30)
    rows = [{'Op Name': name, 'OP Type': name, 'Task Duration(us)': str(duration)}
            for _ in range(30) for name, duration in [('kernel_a', 5), ('kernel_b', 30)]]
    kernels = runner._aggregate_op_summary(rows)
    assert [metric['sample_count'] for metric in kernels] == [30, 30]
    assert optimize._duration(kernels) == 0  # Old kernel-only reports have no invocation boundaries.
    assert optimize._duration_map({'case': kernels})['case'] is None
    write_operator_timings(tmp_path, [35.] * 30, 30)
    invocation = json.loads((tmp_path / 'operator_timing.json').read_text())
    candidate = kernels + [invocation]
    assert optimize._duration(candidate) == 35
    assert 20 / optimize._duration(candidate) == pytest.approx(20 / 35)
    initialized = tmp_path / 'init_results'
    initialized.mkdir()
    (initialized / 'status.json').write_text('{"success":true}')
    (initialized / 'result.json').write_text(json.dumps({'case': candidate}))
    assert _initialized_baselines(tmp_path)['case']['baseline_us'] == 35


def test_build_skeleton_preserves_space_and_metacharacter_arguments(tmp_path):
    from ascend_arena_e2e.ascend_arena.runners.build import BuildRunner
    patch = ROOT / 'src/ascend_arena_e2e/dependency/ops-skeleton.patch'
    dependency = tmp_path / 'dependencies with spaces'
    skeleton = dependency / 'ops-skeleton'
    for section in re.split(r'(?=^--- a/)', patch.read_text(), flags=re.M):
        if not section:
            continue
        name = section.splitlines()[0].removeprefix('--- a/')
        old_script = ''.join(line[1:] for line in section.splitlines(True)
                             if line.startswith((' ', '-')) and not line.startswith('---'))
        target = skeleton / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(old_script)
    source = tmp_path / 'source'
    (source / 'op_host').mkdir(parents=True)
    (source / 'op_kernel').mkdir()
    runner = BuildRunner(dependency)
    target = runner._resolve_compile_repo_target(compile_repo='ops-skeleton', compile_family='math', compile_op_name='expand_into_jagged_permute')
    output = tmp_path / 'build with spaces'
    runner._prepare_run_local_compile_repo_workspace(src_dir=output, compile_target=target, source_root=source)
    command = '''source "$1/scripts/build_cmake.sh"
BASE_PATH="$1/source path"; BUILD_PATH="$1/build path"; BUILD_OUT_PATH="$1/install path"
CANN_3RD_LIB_PATH="$1/offline opbase"; COMPILED_OPS='expand_into_jagged_permute,group_quant'; BISHENG_FLAGS='-O2 -DNAME=value'
COMPUTE_UNIT=ascend910b; BUILD_MODE=all; ENABLE_PACKAGE=TRUE; ENABLE_OOM=TRUE
cmake() { printf '%s\\0' "$@"; }
assemble_cmake_args
cmake_init
'''
    args = subprocess.check_output(['bash', '-c', command, 'bash', str(output)]).decode().split('\0')
    assert f'-DCMAKE_INSTALL_PREFIX={output}/install path' in args
    assert f'-DCANN_3RD_LIB_PATH={output}/offline opbase' in args
    assert '-DBISHENG_FLAGS=-O2 -DNAME=value' in args
    assert '-DASCEND_OP_NAME=expand_into_jagged_permute;group_quant' in args
    assert args[args.index('-S') + 1] == f'{output}/source path'
    assert args[args.index('-B') + 1] == f'{output}/build path'


def test_operator_timer_measures_whole_call_and_excludes_warmup(tmp_path):
    from ascend_arena_e2e.ascend_arena.runners.profile import measure_operator_call
    clock = [0.0]

    class Event:
        def __init__(self, enable_timing):
            assert enable_timing
        def record(self):
            self.timestamp = clock[0]
        def synchronize(self):
            pass
        def elapsed_time(self, end):
            return (end.timestamp - self.timestamp) / 1000

    torch = SimpleNamespace(npu=SimpleNamespace(Event=Event))
    samples = []
    with measure_operator_call(torch, samples):
        clock[0] += 1000  # warmup
    for _ in range(30):
        with measure_operator_call(torch, samples):
            clock[0] += 5
            clock[0] += 30
    write_operator_timings(tmp_path, samples, 30)
    metric = json.loads((tmp_path / 'operator_timing.json').read_text())
    assert metric['sample_count'] == 30
    assert metric['duration_samples_us'] == [35.] * 30
    assert metric['task_duration_us'] == 35


def test_space_opbase_checkout_is_copied_into_space_free_sandbox(tmp_path):
    dependency = tmp_path / 'dependencies with spaces'
    skeleton = dependency / 'ops-skeleton'
    skeleton.mkdir(parents=True)
    (skeleton / 'CMakeLists.txt').write_text('# skeleton')
    opbase = dependency / 'ops-base'
    (opbase / 'include').mkdir(parents=True)
    subprocess.run(['git', 'init', '--quiet', str(opbase)], check=True)
    (opbase / 'include/op.h').write_text('original source')
    target = tmp_path / 'sandbox'
    optimize._copy_dependency_tree(dependency, target, compile_repo='ops-skeleton')
    assert not (target / 'ops-base').is_symlink()
    assert (target / 'ops-base/include/op.h').read_text() == 'original source'
    (target / 'ops-base/include/op.h').write_text('candidate copy')
    assert (opbase / 'include/op.h').read_text() == 'original source'


def test_legacy_kernel_and_operator_call_timings_are_not_compared(tmp_path):
    from kernel_eval.optimization_trajectory import build_trajectory_data
    initialized = tmp_path / 'init_results'
    initialized.mkdir()
    legacy = {'case': [{'op_name': 'kernel', 'task_duration_us': 20, 'noise_band': 0}]}
    (initialized / 'status.json').write_text('{"success":true}')
    (initialized / 'result.json').write_text(json.dumps(legacy))
    run = tmp_path / 'ascend_arena_expand_into_jagged_permute_test'
    run.mkdir()
    optimize._build_report(output_dir=run, run_name=run.name, operator='expand_into_jagged_permute', baseline=legacy,
        candidate={'case': [{'duration_scope': 'operator_call', 'task_duration_us': 35, 'noise_band': 0}]},
        baseline_status={'success': True}, candidate_status={'success': True}, device_id=1,
        source_diff={'changed_file_count': 1, 'changed_files': [{'path': 'kernel.cpp', 'status': 'modified'}],
                     'unified_diff': '--- a/kernel.cpp\n+++ b/kernel.cpp\n-old\n+new\n'})
    report = json.loads((run / f'{run.name}.json').read_text())
    assert report['cases'][0]['speedup'] is None
    assert report['cases'][0]['timing_comparable'] is False
    trajectory = build_trajectory_data(tmp_path)
    assert trajectory['rounds'][0]['speedup'] is None


def test_trajectory_keeps_legacy_round_separate_from_new_call_baseline(tmp_path):
    from kernel_eval.optimization_trajectory import build_trajectory_data
    initialized = tmp_path / 'init_results'
    initialized.mkdir()
    (initialized / 'status.json').write_text('{"success":true}')
    (initialized / 'result.json').write_text(json.dumps({'case': [
        {'duration_scope': 'operator_call', 'task_duration_us': 100}]}))
    run = tmp_path / 'ascend_arena_expand_into_jagged_permute_legacy'
    run.mkdir()
    optimize._build_report(output_dir=run, run_name=run.name, operator='expand_into_jagged_permute',
        baseline={'case': [{'op_name': 'kernel', 'task_duration_us': 20}]},
        candidate={'case': [{'op_name': 'kernel', 'task_duration_us': 10}]},
        baseline_status={'success': True}, candidate_status={'success': True}, device_id=1,
        source_diff={'changed_file_count': 1, 'changed_files': [{'path': 'kernel.cpp', 'status': 'modified'}],
                     'unified_diff': '--- a/kernel.cpp\n+++ b/kernel.cpp\n-old\n+new\n'})
    assert json.loads((run / f'{run.name}.json').read_text())['cases'][0]['speedup'] == 2
    trajectory = build_trajectory_data(tmp_path)
    assert trajectory['rounds'][0]['speedup'] is None
    assert trajectory['rounds'][0]['cases'][0]['candidate_us'] is None


@pytest.mark.parametrize('backend', ['torch-profiler', 'msprof', 'msprof-op'])
def test_worker_timing_scope_matches_profiler_replay_policy(tmp_path, backend):
    import ast
    runner = ProfileRunner(profiler=backend)
    script = tmp_path / 'worker.py'
    writer = runner._write_torch_profiler_script if backend == 'torch-profiler' else runner._write_msprof_script
    writer(script=script, standard_dir=tmp_path, build_result=SimpleNamespace(build_dir=tmp_path),
           output_dir=tmp_path / 'profile', device_id=1, case={'case_id': 'case'})
    tree = ast.parse(script.read_text())
    compile(tree, str(script), 'exec')
    calls = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert ('measure_operator_call' in calls) is (backend == 'msprof')
    assert ('write_operator_timings' in calls) is (backend == 'msprof')
    if backend == 'torch-profiler':
        assert '_execute_torch_profiler_case' in script.read_text()


@pytest.mark.parametrize('vendors_exists', [False, True])
def test_active_cann_is_selected_without_install_write_permission(tmp_path, monkeypatch, vendors_exists):
    active = tmp_path / 'system-cann'
    active.mkdir()
    (active / 'set_env.sh').write_text('export ASCEND_HOME_PATH=system-cann\n')
    if vendors_exists:
        (active / 'opp/vendors').mkdir(parents=True)
    user = tmp_path / 'user'
    older = user / 'Ascend/cann-9.0.0'
    (older / 'opp/vendors').mkdir(parents=True)
    (older / 'set_env.sh').write_text('export ASCEND_HOME_PATH=cann-9.0.0\n')
    monkeypatch.setenv('ASCEND_HOME_PATH', str(active))
    monkeypatch.setattr(Path, 'home', lambda: user)
    monkeypatch.setattr(optimize.os, 'access', lambda path, mode: path == older / 'opp/vendors')

    assert optimize._resolve_env_script('') == active / 'set_env.sh'
