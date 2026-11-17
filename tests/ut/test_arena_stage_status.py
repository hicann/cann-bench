# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the repository root for the full license text.

import json
import subprocess
import sys
import pytest

from kernel_eval import optimize


def completed_results(root):
    for phase in ('smoke', 'complete', 'profile'):
        (root / f'{phase}-correctness_result.json').write_text(json.dumps({
            'success': True, 'case_results': [
                {'case_id': 'case-1', 'passed': True, 'passed_runs': 3, 'total_runs': 3},
            ],
        }))
    (root / 'result.json').write_text('{"case-1": [{"task_duration_us": 10}]}')
    return {'success': True, 'build_success': True, 'profile_success': True,
            'requested_stages': ['smoke', 'complete', 'profile'], 'profile_cases': 1,
            'compile_log': str(root / 'compile.log'), 'profile_log': str(root / 'profile.log'),
            'profile_message': 'profiled_cases=1'}


def test_success_status_links_retained_results_after_cleanup(tmp_path, monkeypatch):
    status = completed_results(tmp_path)
    for name in optimize._STAGE_LOG_NAMES:
        (tmp_path / name).write_text('done')
    monkeypatch.setenv('ASCEND_ARENA_PROFILE_ARTIFACTS', 'compact')
    (tmp_path / 'profile').mkdir()
    (tmp_path / 'profile/raw.db').touch()
    optimize._write_status_payload(tmp_path / 'status.json', status)
    optimize._cleanup_success_logs(tmp_path, status)
    stored = json.loads((tmp_path / 'status.json').read_text())
    assert stored['all_stages_passed'] is True
    assert list(stored['stages']) == ['compile', 'smoke', 'complete', 'profile_correctness', 'performance']
    for phase in stored['stages'].values():
        assert phase['status'] == 'passed' and phase['success'] is True
        assert (tmp_path / phase['result_json']).is_file()
        assert phase['log'] is None
    assert stored['compile_log'] is None and stored['profile_log'] is None
    assert stored['stages']['smoke']['passed_cases'] == 1
    assert not (tmp_path / 'profile').exists()


def test_no_perf_is_skipped_and_does_not_claim_five_stages_passed(tmp_path):
    status = completed_results(tmp_path)
    status['profile_message'] = 'skipped_performance_disabled'
    status['profile_cases'] = 0
    (tmp_path / 'result.json').write_text('{}')
    optimize._write_status_payload(tmp_path / 'status.json', status)
    assert status['success'] is True
    assert status['all_stages_passed'] is False
    performance = status['stages']['performance']
    assert performance['status'] == 'skipped' and performance['success'] is None
    assert performance['result_json'] is None


@pytest.mark.parametrize('failed', ['compile', 'smoke', 'complete', 'profile_correctness', 'performance'])
def test_failed_stage_keeps_log_and_later_stages_are_not_run(tmp_path, failed):
    names = ['compile', 'smoke', 'complete', 'profile_correctness', 'performance']
    status = completed_results(tmp_path)
    status['success'] = False
    status['profile_success'] = False
    if failed == 'compile':
        status['build_success'] = False
    for name, filename in [('smoke', 'smoke-correctness_result.json'),
                           ('complete', 'complete-correctness_result.json'),
                           ('profile_correctness', 'profile-correctness_result.json')]:
        if names.index(name) > names.index(failed):
            (tmp_path / filename).unlink()
        elif name == failed:
            data = json.loads((tmp_path / filename).read_text())
            data['success'] = False
            data['case_results'][0]['passed'] = False
            (tmp_path / filename).write_text(json.dumps(data))
    (tmp_path / 'compile.log').write_text('diagnostic')
    optimize._write_status_payload(tmp_path / 'status.json', status)
    optimize._cleanup_success_logs(tmp_path, status)
    assert status['stages'][failed]['status'] == 'failed'
    assert status['stages'][failed]['success'] is False
    assert all(status['stages'][n]['status'] == 'not_run' for n in names[names.index(failed) + 1:])
    assert status['all_stages_passed'] is False
    assert (tmp_path / 'compile.log').is_file()


def test_missing_result_is_not_reported_as_passed(tmp_path):
    status = completed_results(tmp_path)
    (tmp_path / 'complete-correctness_result.json').unlink()
    (tmp_path / 'compile.log').write_text('diagnostic')
    optimize._write_status_payload(tmp_path / 'status.json', status)
    optimize._cleanup_success_logs(tmp_path, status)
    assert status['stages']['complete']['status'] == 'unknown'
    assert status['all_stages_passed'] is False
    assert (tmp_path / 'compile.log').is_file()


def test_interruption_preserves_completed_compile_and_smoke(tmp_path):
    status = completed_results(tmp_path)
    optimize._write_status_payload(tmp_path / 'compile_result.json', {'success': True})
    for name in ['complete-correctness_result.json', 'profile-correctness_result.json', 'result.json']:
        (tmp_path / name).unlink()
    status.update(success=False, build_success=False, profile_success=False, interrupted=True)
    optimize._write_status_payload(tmp_path / 'status.json', status)
    assert status['build_success'] is True
    assert status['stages']['compile']['status'] == 'passed'
    assert status['stages']['smoke']['status'] == 'passed'
    assert status['stages']['complete']['status'] == 'interrupted'
    assert status['stages']['profile_correctness']['status'] == 'not_run'
    assert status['stages']['performance']['status'] == 'not_run'


def test_missing_case_failure_is_recorded_from_progress(tmp_path):
    optimize._record_stage_completion(tmp_path, {
        'label': 'smoke_correctness_finished', 'success': False,
        'op_name': 'logit', 'reason': 'skipped_no_smoke_cases',
    })
    status = {'success': False, 'build_success': True, 'profile_success': False}
    optimize._write_status_payload(tmp_path / 'status.json', status)
    assert status['stages']['smoke']['status'] == 'failed'
    assert status['stages']['smoke']['message'] == 'skipped_no_smoke_cases'
    assert status['stages']['complete']['status'] == 'not_run'


def test_baseline_retry_does_not_reuse_previous_success(tmp_path, monkeypatch):
    status = completed_results(tmp_path)
    optimize._write_status_payload(tmp_path / 'status.json', status)
    (tmp_path / 'keep.json').write_text('{"user": "evidence"}')
    original = subprocess.Popen
    script = "print('[progress] {\"label\":\"build_finished\",\"success\":false}')"
    monkeypatch.setattr(optimize.subprocess, 'Popen',
                        lambda command, **kwargs: original([sys.executable, '-c', script], **kwargs))
    result, payload = optimize._run_stage(
        standard_dir=tmp_path, operator='logit', dependency_root=tmp_path,
        run_root=tmp_path / 'run', chip='ascend910b', device_id=1,
        compile_repo='ops-skeleton', compile_family='nn', compile_op_name='logit',
        env_script=None, warmup=1, repeat=1, profile_backend='torch-profiler',
        compile_jobs=1, isolated_build=False, profile_performance=True,
        stages=frozenset({'smoke', 'complete', 'profile'}), result_path=tmp_path / 'result.json',
    )
    refreshed = optimize._write_status(tmp_path / 'status.json', result, payload,
                                      stage='baseline', requested_stages=frozenset({'smoke', 'complete', 'profile'}))
    assert refreshed['stages']['compile']['status'] == 'failed'
    assert all(refreshed['stages'][phase]['status'] == 'not_run'
               for phase in ['smoke', 'complete', 'profile_correctness', 'performance'])
    assert not list(tmp_path.glob('*-correctness_result.json'))
    assert (tmp_path / 'keep.json').is_file()


def test_archived_status_paths_resolve_for_both_implementations(tmp_path):
    from kernel_eval.acceptance_report import write_acceptance_report
    from tests.ut.test_acceptance_report import _make_run

    root = tmp_path / 'run'
    _make_run(root)
    evaluation = root / 'demo_op/kernel_eval'
    sources = []
    for directory in [evaluation / 'init_results', evaluation / 'ascend_arena_demo_op_run/results']:
        status = completed_results(directory)
        for name in optimize._STAGE_LOG_NAMES:
            (directory / name).write_text('diagnostic')
        optimize._write_status_payload(directory / 'status.json', status)
        sources.append((directory / 'status.json').read_bytes())
    output = write_acceptance_report(root)
    for kind, directory, original in zip(['baseline', 'candidate'],
                                         [evaluation / 'init_results', evaluation / 'ascend_arena_demo_op_run/results'], sources):
        path = output.parent / f'evidence/demo_op/{kind}_status.json'
        archived = json.loads(path.read_text())
        for phase in archived['stages'].values():
            assert (path.parent / phase['result_json']).is_file()
            assert (path.parent / phase['log']).is_file()
        assert (directory / 'status.json').read_bytes() == original
