# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/setup_ascend_arena_dependencies.py'
spec = importlib.util.spec_from_file_location('arena_dependency_setup', SCRIPT)
dependency_setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dependency_setup)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def init_repo(path):
    path.mkdir()
    git(path, 'init', '-q')
    git(path, 'config', 'user.name', 'Test')
    git(path, 'config', 'user.email', 'test@example.com')
    git(path, 'config', 'uploadpack.allowFilter', 'true')


def test_sparse_setup_pins_revision_and_preserves_local_changes(tmp_path, monkeypatch):
    monkeypatch.setenv('GIT_ALLOW_PROTOCOL', 'file')
    source = tmp_path / 'upstream'
    init_repo(source)
    (source / 'README').write_text('older history\n')
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'older history')
    (source / 'ops_optimization_skeleton').mkdir()
    (source / 'ops_optimization_skeleton/CMakeLists.txt').write_text('# pinned\n')
    (source / 'nn/demo/op_kernel').mkdir(parents=True)
    (source / 'nn/demo/op_kernel/kernel.cpp').write_text('int value = 1;\n')
    (source / 'LICENSE').write_text('upstream license\n')
    (source / 'unrelated').mkdir()
    (source / 'unrelated/payload').write_text('do not download this payload')
    git(source, 'add', '.')
    git(source, 'commit', '-qm', 'pinned')
    revision = git(source, 'rev-parse', 'HEAD')
    unwanted = git(source, 'rev-parse', 'HEAD:unrelated/payload')
    (source / 'ops_optimization_skeleton/CMakeLists.txt').write_text('# newer\n')
    git(source, 'commit', '-qam', 'newer')
    root = tmp_path / 'parent'
    init_repo(root)
    entries = []
    for name in ('ops-skeleton', 'ops-base', 'ops-nn', 'ops-transformer'):
        relative = f'{dependency_setup.DEPENDENCY}/{name}'
        entries.append(f'[submodule "{relative}"]\n path = {relative}\n url = {source.as_uri()}\n')
        git(root, 'update-index', '--add', '--cacheinfo', f'160000,{revision},{relative}')
    (root / '.gitmodules').write_text(''.join(entries))
    dependency = root / dependency_setup.DEPENDENCY
    dependency.mkdir(parents=True)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    manifest = {
        'repositories': {name: {'revision': revision, 'paths': ['LICENSE', 'nn/demo/op_kernel/kernel.cpp']}
                         for name in ('ops-nn', 'ops-transformer')},
        'operators': {'demo': {'op_kernel/kernel.cpp': {
            'repository': 'ops-nn', 'source': 'nn/demo/op_kernel/kernel.cpp',
            'source_sha256': digest('int value = 1;\n'), 'sha256': digest('int value = 2;\n'),
        }}},
    }
    (dependency / 'baselines.json').write_text(json.dumps(manifest))
    (dependency / 'baselines.patch').write_text(
        '--- a/demo/op_kernel/kernel.cpp\n+++ b/demo/op_kernel/kernel.cpp\n'
        '@@ -1 +1 @@\n-int value = 1;\n+int value = 2;\n'
    )
    dependency_setup.setup(root)
    checkout = root / dependency_setup.DEPENDENCY / 'ops-skeleton'
    assert git(checkout, 'rev-parse', 'HEAD') == revision
    assert git(checkout, 'rev-list', '--count', 'HEAD') == '1'
    assert not (checkout / 'unrelated').exists()
    missing = git(checkout, 'rev-list', '--objects', '--all', '--missing=print')
    assert f'?{unwanted}' in missing
    for name in ('ops-nn', 'ops-transformer'):
        checkout_source = dependency / name
        assert git(checkout_source, 'rev-parse', 'HEAD') == revision
        assert git(checkout_source, 'rev-list', '--count', 'HEAD') == '1'
        assert not (checkout_source / 'unrelated').exists()
        assert f'?{unwanted}' in git(checkout_source, 'rev-list', '--objects', '--all', '--missing=print')
    baseline = dependency / 'baselines/demo/op_kernel/kernel.cpp'
    assert baseline.read_text() == 'int value = 2;\n'
    assert (dependency / 'ops-nn/nn/demo/op_kernel/kernel.cpp').read_text() == 'int value = 1;\n'
    # Workspace copies must be editable without changing the shared baseline or official checkout.
    link = root / 'demo'
    link.symlink_to(dependency / 'baselines/demo', target_is_directory=True)
    shutil.copytree(link, root / 'workspace', symlinks=True)
    (root / 'workspace/op_kernel/kernel.cpp').write_text('candidate\n')
    assert baseline.read_text() == 'int value = 2;\n'
    dependency_setup.setup(root)
    assert not git(checkout, 'status', '--porcelain')
    entry = checkout / 'ops_optimization_skeleton/CMakeLists.txt'
    entry.write_text('# local edit\n')
    with pytest.raises(RuntimeError, match='local changes'):
        dependency_setup.setup(root)
    assert entry.read_text() == '# local edit\n'
    git(checkout, 'checkout', '--', 'ops_optimization_skeleton/CMakeLists.txt')
    baseline.write_text('baseline edit\n')
    with pytest.raises(RuntimeError, match='Baseline sources differ'):
        dependency_setup.setup(root)
    assert baseline.read_text() == 'baseline edit\n'


@pytest.mark.parametrize('bad_patch', [False, True])
def test_baseline_checksum_or_patch_failure_does_not_publish_partial_sources(tmp_path, bad_patch):
    dependency = tmp_path / dependency_setup.DEPENDENCY
    source = dependency / 'ops-nn/kernel.cpp'
    source.parent.mkdir(parents=True)
    source.write_text('int value = 1;\n')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {'operators': {'demo': {'kernel.cpp': {
        'repository': 'ops-nn', 'source': 'kernel.cpp',
        'source_sha256': digest if bad_patch else 'incorrect', 'sha256': digest,
    }}}}
    (dependency / 'baselines.patch').write_text('invalid patch\n')
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        dependency_setup.materialize_baselines(tmp_path, manifest)
    assert not (dependency / 'baselines').exists()
    assert not list(dependency.glob('.baselines-*'))


def test_baseline_validation_rejects_shared_file_links(tmp_path):
    source = tmp_path / 'official.cpp'
    source.write_text('source\n')
    baseline = tmp_path / 'baselines/demo'
    baseline.mkdir(parents=True)
    (baseline / 'kernel.cpp').symlink_to(source)
    manifest = {'operators': {'demo': {'kernel.cpp': {'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}}}
    with pytest.raises(RuntimeError, match='Baseline sources differ'):
        dependency_setup.verify_baselines(baseline.parent, manifest)
