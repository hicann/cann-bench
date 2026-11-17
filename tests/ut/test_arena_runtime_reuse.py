# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import errno
import json

import pytest

from ascend_arena_e2e.ascend_arena.utils import paths
from ascend_arena_e2e.ascend_arena.runners import opp_env


def test_transient_delete_retries_and_exhausts(tmp_path, monkeypatch):
    root = tmp_path / 'tree'
    root.mkdir()
    attempts = []
    real = paths.shutil.rmtree
    def flaky(*args, **kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError(errno.ENOTEMPTY, 'busy')
        real(*args, **kwargs)
    monkeypatch.setattr(paths.shutil, 'rmtree', flaky)
    monkeypatch.setattr(paths.time, 'sleep', lambda _: None)
    paths.remove_tree(root)
    assert len(attempts) == 3 and not root.exists()
    root.mkdir()
    def busy(*args, **kwargs):
        raise OSError(errno.EBUSY, 'busy')
    monkeypatch.setattr(paths.shutil, 'rmtree', busy)
    with pytest.raises(OSError):
        paths.remove_tree(root)


def test_opp_reuses_completed_tree_and_rebuilds_broken_vendor(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    (base / 'built-in').mkdir(parents=True)
    (base / 'built-in/kernel').write_text('kernel')
    custom = tmp_path / 'custom'
    custom.mkdir()
    runtime = tmp_path / 'run/runtime_opp'
    calls = []
    original = opp_env._materialize_opp_child
    def track(src, dst):
        calls.append(src)
        original(src, dst)
    monkeypatch.setattr(opp_env, '_materialize_opp_child', track)
    for _ in range(3):
        opp_env._refresh_runtime_opp_root(base_opp_root=base, runtime_opp_root=runtime, custom_opp_root=custom)
    assert len(calls) == 1
    (runtime / 'vendors/custom').unlink()
    opp_env._refresh_runtime_opp_root(base_opp_root=base, runtime_opp_root=runtime, custom_opp_root=custom)
    assert len(calls) == 2


def test_runtime_opp_selects_candidate_vendor_on_initialization_and_reuse(tmp_path):
    base = tmp_path / 'base'
    (base / 'built-in').mkdir(parents=True)
    custom = tmp_path / 'candidate'
    custom.mkdir()
    env = {'ASCEND_OPP_PATH': str(base), 'ASCEND_CUSTOM_OPP_PATH': '/other/vendor'}
    for _ in range(2):
        env = opp_env.prepare_runtime_opp_env(
            op_run_root=tmp_path / 'run', custom_opp_root=custom, env=env,
        )
        assert env['ASCEND_CUSTOM_OPP_PATH'] == str(custom.resolve())
        assert env['KERNEL_GEN_OPS_CUST_OPAPI_PATH'] == str(custom / 'op_api/lib/libcust_opapi.so')
        assert (tmp_path / 'run/runtime_opp/vendors/config.ini').read_text() == 'load_priority=candidate\n'


def test_archive_checksum_rejected(tmp_path):
    from kernel_eval.optimize import _copy_dependency_tree
    (tmp_path / 'ops-skeleton.tar.gz').write_bytes(b'broken')
    (tmp_path / 'ops-skeleton.json').write_text(json.dumps({'sha256': 'wrong'}))
    with pytest.raises(ValueError, match='checksum'):
        _copy_dependency_tree(tmp_path, tmp_path / 'output', compile_repo='ops-skeleton')


def test_uninitialized_submodule_reports_initialization_command(tmp_path):
    from kernel_eval.optimize import _copy_dependency_tree
    (tmp_path / 'ops-skeleton').mkdir()
    with pytest.raises(FileNotFoundError, match='setup_ascend_arena_dependencies.py'):
        _copy_dependency_tree(tmp_path, tmp_path / 'output', compile_repo='ops-skeleton')


def test_submodule_precedes_legacy_archive_and_excludes_git_metadata(tmp_path):
    from kernel_eval.optimize import _copy_dependency_tree
    repo = tmp_path / 'ops-skeleton'
    repo.mkdir()
    (repo / 'CMakeLists.txt').write_text('# build entry\n')
    (repo / '.git').write_text('gitdir: elsewhere\n')
    (tmp_path / 'ops-skeleton.tar.gz').write_bytes(b'stale archive')
    _copy_dependency_tree(tmp_path, tmp_path / 'output', compile_repo='ops-skeleton')
    assert (tmp_path / 'output/ops-skeleton/CMakeLists.txt').is_file()
    assert not (tmp_path / 'output/ops-skeleton/.git').exists()


def test_archive_parent_traversal_rejected(tmp_path):
    import hashlib
    import tarfile
    from kernel_eval.optimize import _copy_dependency_tree
    archive = tmp_path / 'ops-skeleton.tar.gz'
    with tarfile.open(archive, 'w:gz') as bundle:
        bundle.addfile(tarfile.TarInfo('ops-skeleton/../../outside'))
    (tmp_path / 'ops-skeleton.json').write_text(json.dumps({
        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    }))
    with pytest.raises(ValueError, match='unsafe'):
        _copy_dependency_tree(tmp_path, tmp_path / 'output', compile_repo='ops-skeleton')
    assert not (tmp_path / 'outside').exists()


def test_monorepo_skeleton_and_shared_opbase(tmp_path):
    from kernel_eval.optimize import _copy_dependency_tree
    from ascend_arena_e2e.ascend_arena.runners.build import BuildRunner
    dependency = tmp_path / 'dependency'
    repo = dependency / 'ops-skeleton'
    skeleton = repo / 'ops_optimization_skeleton'
    skeleton.mkdir(parents=True)
    (skeleton / 'CMakeLists.txt').write_text('# skeleton\n')
    (repo / 'unrelated').mkdir()
    (repo / 'unrelated/large_file').write_text('unrelated')
    opbase = dependency / 'ops-base'
    opbase.mkdir()
    (opbase / '.git').write_text('gitdir: elsewhere\n')
    target = tmp_path / 'sandbox'
    _copy_dependency_tree(dependency, target, compile_repo='ops-skeleton')
    assert (target / 'ops-skeleton/CMakeLists.txt').is_file()
    assert not (target / 'ops-skeleton/unrelated').exists()
    assert (target / 'ops-base').resolve() == opbase
    resolved = BuildRunner(dependency)._resolve_compile_repo_target(
        compile_repo='ops-skeleton', compile_family='math', compile_op_name='logit',
    )
    assert resolved.repo_root == skeleton


def test_build_uses_shared_opbase_and_preserves_explicit_override(tmp_path, monkeypatch):
    import os
    import subprocess
    from ascend_arena_e2e.ascend_arena.runners.build import BuildRunner
    dependency = tmp_path / 'deps with spaces'
    (dependency / 'ops-base').mkdir(parents=True)
    (dependency / 'ops-base/.git').write_text('gitdir: elsewhere\n')
    runner = BuildRunner(dependency)
    captured = {}
    monkeypatch.setattr(runner, '_run_shell_command', lambda **kw: captured.update(kw))
    monkeypatch.setattr(runner, '_build_compile_repo_command', lambda **kw: 'true')
    runner._run_run_local_compile_repo_build(repo_root=tmp_path, compile_target=None, chip_name='ascend910b')
    command = captured['command_parts'][0] + '; printf "%s" "$OPBASE_SOURCE_PATH"'
    for override in ('', '/custom opbase'):
        env = {**os.environ, 'OPBASE_SOURCE_PATH': override}
        result = subprocess.check_output(['bash', '-c', command], env=env, text=True)
        assert result == (override or str(dependency / 'ops-base'))
