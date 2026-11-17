# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.fixture
def build_info(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / 'src/cann_bench_utils/cann_bench_utils/build_info.py'
    spec = importlib.util.spec_from_file_location('helper_build_info', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'build_info.py'))
    (tmp_path / 'compiler').mkdir()
    (tmp_path / 'compiler/version.info').write_text('Version=9.1.0\nother=value\n')
    monkeypatch.setenv('ASCEND_HOME_PATH', str(tmp_path))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='2.8.0'))
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace(__version__='2.8.0'))
    for name in ['CMakeLists.txt', 'setup.py', 'build.sh']:
        (tmp_path / name).write_text('source')
    (tmp_path / '_C.so').write_bytes(b'helper binary')
    return module, tmp_path


def metadata(module, root):
    info = module.current_environment()
    info['npu_arch'] = 'ascend910b'
    info['source_sha256'] = module.source_fingerprint(root)
    info['binary_sha256'] = hashlib.sha256((root / '_C.so').read_bytes()).hexdigest()
    (root / '_build_info.json').write_text(json.dumps(info))
    return info


def test_missing_metadata_requires_rebuild(build_info):
    module, _ = build_info
    with pytest.raises(RuntimeError, match='metadata is missing'):
        module.check_build_compatibility()


def test_cann_upgrade_rejects_importable_old_binary(build_info):
    module, root = build_info
    info = metadata(module, root)
    assert info['cann'] == '9.1.0'
    info['cann'] = '9.0.0'
    (root / '_build_info.json').write_text(json.dumps(info))
    with pytest.raises(RuntimeError, match='mismatch: cann'):
        module.check_build_compatibility()


def test_matching_build_and_binary_integrity(build_info):
    module, root = build_info
    info = metadata(module, root)
    assert module.check_build_compatibility() == info
    (root / '_C.so').write_bytes(b'stale binary')
    with pytest.raises(RuntimeError, match='does not match'):
        module.check_build_compatibility()


def test_missing_cann_environment_has_actionable_error(build_info, monkeypatch):
    module, _ = build_info
    monkeypatch.delenv('ASCEND_HOME_PATH')
    with pytest.raises(RuntimeError, match='Load CANN set_env.sh'):
        module.current_environment()


def test_updated_helper_source_requires_rebuild(build_info):
    module, root = build_info
    info = metadata(module, root)
    assert module.check_build_compatibility(source_root=root) == info
    (root / 'CMakeLists.txt').write_text('updated source')
    with pytest.raises(RuntimeError, match='source has changed'):
        module.check_build_compatibility(source_root=root)
