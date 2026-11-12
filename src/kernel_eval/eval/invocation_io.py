# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Storage ownership for one restored invocation, outside candidate timing.

Returned tensors transfer their backing storage to the evaluator for cleanup;
they must not alias persistent model state. Views and input/output aliases are
allowed within an invocation. This does not manage unreturned workspaces or
provide process isolation, an allocator reset, or globally unique addresses.
"""

import torch

from .input_pool import _safe_tensor_clone

# Torch operators return structseq tuple subclasses (cummin, sort, topk, ...).
# Capture exact framework types; arbitrary user-defined containers stay rejected.
_TORCH_RETURN_TYPES = frozenset(
    cls for cls in vars(torch.return_types).values()
    if isinstance(cls, type) and issubclass(cls, tuple)
)


def _map_io(value, tensor_fn):
    """Map output trees, including Torch structseqs, preserving object identity."""
    memo = {}
    active = set()

    def visit(item):
        if isinstance(item, torch.Tensor):
            if (
                type(item) is not torch.Tensor
                or item.layout != torch.strided
                or item.is_conj()
                or item.is_neg()
                or item.is_quantized
                or item.device.type not in ("cpu", "npu", "privateuseone")
            ):
                raise ValueError("I/O cleanup requires ordinary dense CPU/NPU tensors")
            if id(item) not in memo:
                memo[id(item)] = tensor_fn(item)
            return memo[id(item)]
        is_torch_result = type(item) in _TORCH_RETURN_TYPES
        if type(item) in (list, tuple, dict) or is_torch_result:
            if id(item) in active:
                raise ValueError("I/O cleanup does not support cyclic containers")
            if id(item) in memo:
                return memo[id(item)]
            active.add(id(item))
            try:
                if type(item) is dict:
                    if any(
                        type(key) not in (str, int, float, bool, type(None))
                        for key in item
                    ):
                        raise ValueError(
                            "I/O dictionary keys must be immutable scalars"
                        )
                    mapped = {key: visit(child) for key, child in item.items()}
                else:
                    children = [visit(child) for child in item]
                    if is_torch_result:
                        mapped = type(item)(children)
                    else:
                        mapped = tuple(children) if type(item) is tuple else children
                memo[id(item)] = mapped
                return mapped
            finally:
                active.remove(id(item))
        if item is None or type(item) in (
            bool,
            int,
            float,
            str,
            torch.dtype,
            torch.device,
        ):
            return item
        raise ValueError(f"I/O cleanup does not support {type(item).__name__}")

    try:
        return visit(value)
    finally:
        # Do not leave a recursive closure retaining HBM until cyclic GC runs.
        memo.clear()
        active.clear()
        visit = None
        tensor_fn = None


def storage_key(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr()


def storage_bytes(tensor):
    """Retain a byte view of the entire backing storage, including view offsets."""
    storage = tensor.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def collect_storages(value, into, forbidden=()):
    """Retain each backing allocation once, before candidate containers can change."""

    def retain(tensor):
        if tensor.untyped_storage().nbytes():
            key = storage_key(tensor)
            if key in forbidden:
                raise ValueError(
                    "invocation I/O aliases protected or previous invocation storage"
                )
            if key not in into:
                into[key] = storage_bytes(tensor)

    _map_io(value, retain)


def snapshot_output(value):
    """Save final values and view aliases before retiring the candidate's buffers."""
    copies = {}

    def copy_tensor(tensor):
        key = storage_key(tensor)
        # Empty storages share a null pointer; no bytes or alias need preserving.
        if tensor.untyped_storage().nbytes() == 0:
            return torch.empty_strided(
                tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
            )
        if key not in copies:
            raw = storage_bytes(tensor)
            copies[key] = (
                raw.clone() if raw.device.type == "cpu" else _safe_tensor_clone(raw)
            )
        return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
            copies[key].untyped_storage(),
            tensor.storage_offset(),
            tensor.size(),
            tensor.stride(),
        )

    try:
        with torch.no_grad():
            return _map_io(value, copy_tensor)
    finally:
        copies.clear()


def zero_storage(raw):
    """No built-in NPU math kernels: they may have been removed by anti-cheat."""
    if raw.device.type == "cpu":
        raw.zero_()
    else:
        try:
            from cann_bench_utils import cann_bench_zero
        except ImportError as exc:
            raise RuntimeError(
                "I/O cleanup requires rebuilt cann_bench_utils with cann_bench_zero"
            ) from exc
        cann_bench_zero(raw)


class InvocationIO:
    """Keep one scrubbed generation live until the next invocation finishes.

    This prevents adjacent invocations from reusing an allocator-owned I/O
    allocation. Older addresses may recur, but their visible storage was zeroed.
    At most current + previous I/O are retained, independent of repeat count.
    """

    def __init__(self, protected=()):
        # Keep allocations alive, not just their numeric addresses. Otherwise a
        # temporary caller input can die and its address be reassigned to a
        # perfectly valid output, producing a false protected-alias violation.
        self.protected = dict(protected)
        self.current = {}
        self.previous = {}
        self.output = None
        self.peak_tracked_bytes = 0

    def bind_inputs(self, args, kwargs):
        if self.current:
            raise RuntimeError(
                "previous invocation must be retired before preparing inputs"
            )
        collect_storages(
            (args, kwargs), self.current, self.protected.keys() | self.previous.keys()
        )

    def retire(self, synchronize, *, preserve_output=False):
        """Sync, save the final result if requested, zero, sync, then release old I/O.

        Synchronization or cleanup failures propagate; never run another call
        after a partial cleanup. Known inputs are still scrubbed if output tree
        validation fails. References stay live until the final sync completes.
        """
        saved = None
        error = None
        try:
            synchronize()
            try:
                collect_storages(
                    self.output,
                    self.current,
                    self.protected.keys() | self.previous.keys(),
                )
                tracked_bytes = sum(raw.numel() for raw in self.current.values())
                tracked_bytes += sum(raw.numel() for raw in self.previous.values())
                self.peak_tracked_bytes = max(self.peak_tracked_bytes, tracked_bytes)
                if preserve_output:
                    saved = snapshot_output(self.output)
            except Exception as exc:  # noqa: BLE001 - scrub known inputs, then propagate
                error = exc
            # Snapshot D2D reads precede memset on the same current stream.
            with torch.no_grad():
                for raw in self.current.values():
                    try:
                        zero_storage(raw)
                    except Exception as exc:  # noqa: BLE001 - finish other buffers, then propagate
                        if error is None:
                            error = exc
            synchronize()
            if error is not None:
                raise error
            byte_count = sum(raw.numel() for raw in self.current.values())
            self.previous.clear()
            self.previous, self.current = self.current, {}
            return saved, byte_count
        finally:
            self.output = None
            # A caught exception points back to this frame. Do not leave a
            # self-referencing traceback cycle retaining current/saved HBM.
            error = None

    def release(self):
        self.current.clear()
        self.previous.clear()
        self.protected.clear()
        self.output = None
