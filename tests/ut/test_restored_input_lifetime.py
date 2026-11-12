# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Strict input graphs release promptly without relying on cyclic collection."""

import gc
import weakref

import pytest
import torch

from kernel_eval.eval.restored_inputs import (
    PerfIntegrityError,
    _map_call_inputs,
    clone_call_inputs,
    clone_reference_inputs,
    migrate_call_inputs,
)


@pytest.fixture
def no_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


@pytest.mark.usefixtures('no_cyclic_gc')
@pytest.mark.parametrize('mode', ['clone', 'migrate', 'reference'])
def test_replaced_graphs_release_immediately_and_preserve_aliases(mode):
    original = torch.ones(4)
    group = [original]
    params = {'x': original, 'group': group, 'same_group': group}
    refs = []
    current = None
    for _ in range(40):
        if mode == 'clone':
            _, current, _ = clone_call_inputs((), params)
        elif mode == 'migrate':
            _, current, _ = migrate_call_inputs((), params, lambda tensor: tensor.clone())
        else:
            converted = clone_reference_inputs([original], params, lambda tensor: tensor.double())
            inputs, current = converted
            assert inputs[0] is current['x']
            del inputs, converted
        assert current['x'] is current['group'][0]
        assert current['group'] is current['same_group']
        assert all(ref() is None for ref in refs)
        refs.append(weakref.ref(current['x']))
    del current
    assert all(ref() is None for ref in refs)
    assert torch.equal(original, torch.ones(4))


@pytest.mark.usefixtures('no_cyclic_gc')
def test_identity_validation_memo_does_not_retain_original_inputs():
    source = torch.ones(4)
    ref = weakref.ref(source)
    args, kwargs, _ = migrate_call_inputs((source,), {'same': source}, lambda tensor: tensor)
    assert args[0] is kwargs['same'] is source
    del source, args, kwargs
    assert ref() is None


@pytest.mark.usefixtures('no_cyclic_gc')
@pytest.mark.parametrize('failure', ['unsupported', 'mapping'])
def test_partial_mapping_error_does_not_retain_cloned_tensors(failure):
    refs = []
    def mapper(tensor):
        if failure == 'mapping' and refs:
            raise RuntimeError('mapping failed')
        copied = tensor.clone()
        refs.append(weakref.ref(copied))
        return copied

    bad = object() if failure == 'unsupported' else torch.zeros(4)
    expected = PerfIntegrityError if failure == 'unsupported' else RuntimeError
    try:
        _map_call_inputs((torch.ones(4), bad), {}, mapper)
    except expected:
        pass
    else:
        pytest.fail('expected partial mapping failure')
    assert len(refs) == 1
    assert refs[0]() is None
