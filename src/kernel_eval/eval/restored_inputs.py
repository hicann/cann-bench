#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Trusted input preparation for per-invocation profiling.

This is state restoration, not a process isolation or unique-address guarantee.
Only ordinary, contiguous, complete dense tensor storages are supported. Distinct
views sharing a storage are rejected; repeated references to the same Tensor are
preserved. No preparation or validation work belongs inside the candidate window.
"""

import torch

from .input_pool import _safe_tensor_clone
from .invocation_io import InvocationIO, collect_storages


class PerfIntegrityError(RuntimeError):
    """A performance integrity requirement failed; never retry with weaker checks."""


def _map_call_inputs(args, kwargs, copy_tensor):
    """Validate supported call data and map each unique tensor exactly once."""
    tensors = {}
    storages = {}
    active = set()
    containers = {}
    total_bytes = 0

    def clone(value):
        nonlocal total_bytes
        if isinstance(value, torch.Tensor):
            if type(value) is not torch.Tensor:
                raise PerfIntegrityError("restored inputs do not support Tensor subclasses")
            if id(value) in tensors:
                return tensors[id(value)]
            if (value.layout != torch.strided or not value.is_contiguous()
                    or value.storage_offset() != 0 or value.is_conj() or value.is_neg()):
                raise PerfIntegrityError("restored inputs require contiguous, offset-zero dense tensors")
            storage = value.untyped_storage()
            size = value.numel() * value.element_size()
            if storage.nbytes() != size:
                raise PerfIntegrityError("restored inputs do not support partial-storage views")
            key = (str(value.device), storage.data_ptr())
            if size and key in storages:
                raise PerfIntegrityError("restored inputs do not support distinct tensors sharing storage")
            storages[key] = id(value)
            copied = copy_tensor(value)
            tensors[id(value)] = copied
            total_bytes += size
            return copied
        if isinstance(value, (list, tuple, dict)):
            if type(value) not in (list, tuple, dict):
                raise PerfIntegrityError("restored inputs do not support custom containers")
            if id(value) in active:
                raise PerfIntegrityError("restored inputs do not support cyclic containers")
            if id(value) in containers:
                return containers[id(value)]
            active.add(id(value))
            try:
                if isinstance(value, dict):
                    if any(type(k) not in (str, int, float, bool, type(None)) for k in value):
                        raise PerfIntegrityError("restored input dictionary keys must be immutable scalars")
                    result = {key: clone(item) for key, item in value.items()}
                else:
                    items = [clone(item) for item in value]
                    result = tuple(items) if isinstance(value, tuple) else items
                containers[id(value)] = result
                return result
            finally:
                active.remove(id(value))
        if value is None or type(value) in (bool, int, float, str, torch.dtype, torch.device):
            return value
        raise PerfIntegrityError(f"restored inputs do not support attribute type {type(value).__name__}")

    try:
        with torch.no_grad():
            copied_args, copied_kwargs = clone((tuple(args), dict(kwargs)))
        return copied_args, copied_kwargs, total_bytes
    finally:
        # The recursive closure owns its own function cell and every memo. Break
        # that cycle explicitly: large CPU/NPU tensors must not await cyclic GC
        # after a fixture is replaced. Clearing memo entries leaves the returned
        # graph and its shared Tensor/container identities intact.
        tensors.clear()
        containers.clear()
        storages.clear()
        active.clear()
        clone = None
        copy_tensor = None


def clone_call_inputs(args, kwargs):
    """Clone supported call data without changing aliases or container types."""
    return _map_call_inputs(
        args, kwargs,
        lambda value: _safe_tensor_clone(value) if value.device.type != "cpu" else value.clone(),
    )


def migrate_call_inputs(args, kwargs, to_device):
    """Validate before migration, then move each distinct Tensor only once.

    ``to_device`` is a trusted tensor mapping, e.g. DeviceManager.to_device.
    This also supports trusted CPU precision conversion without splitting
    repeated Tensor identities. It changes no default DeviceManager behavior.
    """
    _map_call_inputs(args, kwargs, lambda value: value)
    return _map_call_inputs(args, kwargs, to_device)


def clone_reference_inputs(input_tensors, params, transform):
    """Give one reference its own inputs and preserve aliases during conversion."""
    copied, _, _ = clone_call_inputs((input_tensors, params), {})
    converted, _, _ = migrate_call_inputs(copied, {}, transform)
    return converted


class RestoredCall:
    """A callable whose preparation and completion run outside capture.

    ``input_factory(index, phase)`` is trusted and returns (args, kwargs).
    ``validator(output, args, kwargs, index, phase)`` must finish consuming the
    actual returned value before the next prepare; raise on failure. Phases are
    preflight, warmup, measured; indexes are global zero-based call numbers.
    The validator runs after device synchronization, outside candidate guards.
    """

    def __init__(self, func, args, kwargs, *, input_factory=None, validator=None,
                 synchronize=None):
        self.func = func
        self.pristine_args, self.pristine_kwargs, self.input_bytes = clone_call_inputs(args, kwargs)
        self.input_factory = input_factory
        self.validator = validator
        self.synchronize = synchronize or (lambda: None)
        protected = {}
        collect_storages((args, kwargs, self.pristine_args, self.pristine_kwargs), protected)
        self.io = InvocationIO(protected)
        self.pending_io = False
        self.args = self.kwargs = None
        self.prepared = False

    def prepare(self, index, phase):
        if self.pending_io:
            raise PerfIntegrityError("previous invocation I/O must be retired before prepare")
        self.args = self.kwargs = None
        self.prepared = False
        args, kwargs = (self.input_factory(index, phase) if self.input_factory is not None
                        else (self.pristine_args, self.pristine_kwargs))
        self.args, self.kwargs, self.input_bytes = clone_call_inputs(args, kwargs)
        self.pending_io = True
        self.io.bind_inputs(self.args, self.kwargs)
        self.prepared = True

    def __call__(self):
        if not self.prepared:
            raise PerfIntegrityError("candidate invocation requires freshly prepared inputs")
        self.prepared = False
        self.io.output = self.func(*self.args, **self.kwargs)
        return self.io.output

    def complete(self, output, index, phase):
        if self.validator is not None:
            accepted = self.validator(output, self.args, self.kwargs, index, phase)
            if accepted is False:
                raise PerfIntegrityError("performance validator rejected the actual invocation output")

    def release(self):
        try:
            if self.pending_io:
                self.retire()
        finally:
            self.io.release()
            self.args = self.kwargs = None
            self.pristine_args = self.pristine_kwargs = None
            self.prepared = False

    def retire(self, *, preserve_output=False):
        """Consume/clear invocation I/O outside capture; keep only the last result."""
        try:
            return self.io.retire(self.synchronize, preserve_output=preserve_output)
        finally:
            self.pending_io = False
            self.args = self.kwargs = None
            self.prepared = False
