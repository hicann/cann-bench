#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Versioned, immutable case fixtures. Evaluation never writes this cache.

See docs/design/tensor_cache.md for the on-disk and precision contracts.
Safetensors is imported only when a selected case actually uses the cache.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch


FORMAT = "cann-bench.tensor-cache"
VERSION = 1
DTYPES = ("bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16",
          "int32", "int64", "float16", "bfloat16", "float32", "float64")
FILES = ("inputs.safetensors", "references.safetensors")


class TensorCacheError(ValueError):
    """A selected fixture is invalid or cannot preserve evaluation semantics."""


class TensorCacheUnsupportedError(TensorCacheError):
    """A case cannot use v1 fixtures; prefer may run it without caching."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _identity_value(value):
    """Encode special floats without confusing them with literal contract data.

    Escape any user dictionary containing our marker, including dictionaries
    that already look like an escaped value. Ordinary JSON contracts retain
    their representation, and tuple/list normalization matches the JSON reader.
    This encoding is only for identity; tensor bytes and scalar output rules
    are unchanged.
    """
    marker = "__cann_bench_nonfinite__"
    if isinstance(value, float) and not math.isfinite(value):
        name = "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"
        return {marker: ["float", name]}
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    if isinstance(value, dict):
        encoded = {key: _identity_value(item) for key, item in value.items()}
        return {marker: ["dict", encoded]} if marker in value else encoded
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=4096)
def _source_digest(path, signature):
    digest = file_sha256(path)
    if _source_signature(path) != signature:
        raise TensorCacheError(f"Source changed while fingerprinting: {path}")
    return digest


def source_sha256(path):
    # Recheck metadata each time, including ctime for same-size edits with a
    # restored mtime and inode for replacements. Only source digests are memoized;
    # fixture payload checksums must still scan the files on every load.
    return _source_digest(path, _source_signature(path))


def case_seed(case, eval_seed):
    if eval_seed is None:
        return None
    digest = hashlib.sha256(case.get_case_id_str().encode("utf-8")).digest()
    return (eval_seed + int.from_bytes(digest[:8], "big") % 2**31) % 2**31


def validate_config(config):
    mode = getattr(config, "tensor_cache_mode", "off")
    if mode not in ("off", "prefer", "require"):
        raise TensorCacheError(f"Unknown tensor cache mode: {mode}")
    if mode != "off":
        if not config.tensor_cache_dir or not config.tensor_cache_tasks:
            raise TensorCacheError("Tensor cache requires a directory and explicit --tensor-cache-task selections")
        if config.eval_seed is None:
            raise TensorCacheError("Tensor cache requires a deterministic eval_seed")
    for task in getattr(config, "tensor_cache_tasks", []):
        path = Path(task)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != task or task in ("", "."):
            raise TensorCacheError(f"Cache task must be a canonical tasks-root-relative path: {task!r}")


def selected(config, case):
    validate_config(config)
    return getattr(config, "tensor_cache_mode", "off") != "off" and case.rel_path in config.tensor_cache_tasks


def identity_for(evaluator, case):
    """Conservative content identity: never reuse a stale reference silently."""
    config = evaluator.config
    strategy = getattr(evaluator.bench_config, "golden_precision", "fp64_cpu")
    if evaluator.bench_name != "cann" or strategy not in ("fp64_cpu", "native_cpu"):
        raise TensorCacheUnsupportedError("Tensor cache v1 supports CANN tasks with CPU references only")
    if (case.attrs or {}).get("mc2_distributed", False):
        raise TensorCacheUnsupportedError("Tensor cache v1 does not support distributed cases")
    if evaluator.golden_loader.get_input_function(case.rel_path) is not None:
        raise TensorCacheUnsupportedError("Tensor cache v1 does not support get_input hooks/layout transformations")
    seed = case_seed(case, config.eval_seed)
    if seed is None:
        raise TensorCacheError("Tensor cache requires a deterministic eval_seed")
    task_root = Path(config.tasks_root).resolve()
    task = (task_root / case.rel_path).resolve()
    if not task.is_relative_to(task_root) or not (task / "golden.py").is_file():
        raise TensorCacheError(f"Cannot fingerprint task source: {case.rel_path}")
    if config.tensor_cache_dir and Path(config.tensor_cache_dir).expanduser().resolve().is_relative_to(task):
        raise TensorCacheError("Cache directory must be outside the selected task source directory")
    # Fingerprint all harness Python sources, including shared golden helpers.
    # No Git checkout is required on installed runners. Unrelated harness edits
    # conservatively invalidate fixtures; producer environment is recorded below.
    package = Path(__file__).resolve().parents[1]
    # Rescan names to detect added/removed files; unchanged file contents are
    # hashed once per process rather than once per selected case.
    sources = {"kernel_eval/" + str(p.relative_to(package)): source_sha256(p)
               for p in sorted(package.rglob("*.py"))}
    sources.update({"task/" + str(p.relative_to(task)): source_sha256(p)
                    for p in sorted(task.rglob("*"))
                    if p.is_file() and p.suffix in (".py", ".yaml", ".json")})
    result = dict(format=FORMAT, version=VERSION, bench=evaluator.bench_name,
                  task=case.rel_path, case_id=case.get_case_id_str(), operator=case.operator,
                  input_shapes=case.input_shapes, dtypes=case.dtypes,
                  value_ranges=case.value_ranges, attrs=case.attrs,
                  seed=seed, input_dist=evaluator.data_generator.input_dist,
                  golden_precision=strategy, torch_version=str(torch.__version__), sources=sources)
    # Store the same strict JSON identity used for hashing and load comparison.
    # Special values are valid case contracts, but not JSON numeric literals.
    return json.loads(_json(_identity_value(result)))


def _safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise TensorCacheError("Tensor cache requires safetensors; install the tensor-cache extra") from exc
    return safe_open, save_file


def _validate_tensor(value):
    if (value.device.type != "cpu" or value.layout != torch.strided
            or not value.is_contiguous() or value.is_quantized
            or str(value.dtype).removeprefix("torch.") not in DTYPES):
        raise TensorCacheError("Cache v1 requires contiguous dense CPU tensors in a supported dtype")


class _Encoder:
    def __init__(self, deduplicate=False):
        self.tensors = {}
        self.tensor_objects = {}
        self.object_keys = {}
        self.storage_owners = {}
        self.deduplicate = deduplicate
        self.content_keys = {}

    def encode(self, value):
        if isinstance(value, torch.Tensor):
            _validate_tensor(value)
            object_id = id(value)
            if object_id in self.object_keys:
                return {"tensor": self.object_keys[object_id]}
            if self.deduplicate and value.numel():
                storage = value.untyped_storage().data_ptr()
                owner = self.storage_owners.setdefault(storage, object_id)
                if owner != object_id:
                    raise TensorCacheUnsupportedError(
                        "Cache v1 does not support distinct reference tensors sharing storage")
            key = f"tensor_{len(self.tensors):06d}"
            if self.deduplicate:
                raw = value.detach().reshape(-1).view(torch.uint8).numpy()
                fingerprint = (str(value.dtype), tuple(value.shape), hashlib.sha256(memoryview(raw)).hexdigest())
                key = self.content_keys.setdefault(fingerprint, key)
            # Compact owned stores avoid saving a whole backing allocation;
            # logical object aliases are tracked separately from these bytes.
            if key not in self.tensors:
                self.tensors[key] = value.detach().clone()
            object_key = f"object_{len(self.tensor_objects):06d}"
            self.object_keys[object_id] = object_key
            self.tensor_objects[object_key] = key
            return {"tensor": object_key}
        if value is None or type(value) in (bool, int, str):
            return {"scalar": value}
        if type(value) is float and math.isfinite(value):
            return {"scalar": value}
        if type(value) in (list, tuple):
            return {"tuple" if type(value) is tuple else "list": [self.encode(v) for v in value]}
        raise TensorCacheError(f"Unsupported cache value type: {type(value).__name__}")


def _decode(node, tensors, used, depth=0):
    if depth > 64 or not isinstance(node, dict) or len(node) != 1:
        raise TensorCacheError("Invalid cache value tree")
    kind, value = next(iter(node.items()))
    if kind == "tensor" and isinstance(value, str) and value in tensors:
        used.add(value)
        return tensors[value]
    if kind == "scalar" and (value is None or type(value) in (bool, int, str)
                             or type(value) is float and math.isfinite(value)):
        return value
    if kind in ("list", "tuple") and isinstance(value, list):
        result = [_decode(v, tensors, used, depth + 1) for v in value]
        return tuple(result) if kind == "tuple" else result
    raise TensorCacheError("Invalid cache value tree leaf")


def validate_inputs(inputs, case):
    if not isinstance(inputs, list) or len(inputs) != len(case.input_shapes):
        raise TensorCacheError("Cached input count does not match case")
    if len(case.dtypes) != len(inputs):
        raise TensorCacheError("Cache v1 requires one dtype per input")
    for value, shape, dtype in zip(inputs, case.input_shapes, case.dtypes):
        if shape is None and value is None:
            continue
        if (not isinstance(value, torch.Tensor) or list(value.shape) != list(shape)
                or str(value.dtype).removeprefix("torch.") != dtype):
            raise TensorCacheError("Cached input shape/dtype does not match case")
        _validate_tensor(value)
    storages = [v.untyped_storage().data_ptr() for v in inputs if isinstance(v, torch.Tensor) and v.numel()]
    if len(set(storages)) != len(storages):
        raise TensorCacheError("Cache v1 does not support aliased input storages")


def input_hashes(inputs):
    """Detect reference functions that modify the fixture inputs while building."""
    return [None if value is None else hashlib.sha256(
        memoryview(value.detach().reshape(-1).view(torch.uint8).numpy())).hexdigest() for value in inputs]


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass
class CachedCase:
    inputs: list
    oracle: Any
    native: Any
    key: str


class TensorCache:
    def __init__(self, directory):
        self.root = Path(directory).expanduser().resolve()

    @staticmethod
    def key(identity):
        return hashlib.sha256(_json(identity).encode()).hexdigest()

    def load(self, identity, case):
        """Return None only for an absent entry; corruption always raises."""
        key = self.key(identity)
        directory = self.root / key
        if not directory.exists():
            return None
        try:
            manifest_path = directory / "manifest.json"
            if manifest_path.stat().st_size > 1024**2:
                raise TensorCacheError("Cache manifest exceeds 1 MiB")
            manifest = json.loads(manifest_path.read_text())
            if (manifest["format"] != FORMAT or manifest["version"] != VERSION
                    or manifest["identity"] != identity or manifest["key"] != key
                    or set(manifest["files"]) != set(FILES)):
                raise TensorCacheError("Cache manifest identity/schema mismatch")
            safe_open, _ = _safetensors()
            decoded = {}
            for name in FILES:
                path = directory / name
                spec = manifest["files"][name]
                if path.stat().st_size != spec["bytes"] or file_sha256(path) != spec["sha256"]:
                    raise TensorCacheError(f"Cache checksum mismatch: {name}")
                tensors = {}
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if set(handle.keys()) != set(spec["tensors"]):
                        raise TensorCacheError("Cache tensor names do not match manifest")
                    for tensor_key in handle.keys():
                        value = handle.get_tensor(tensor_key)
                        expected = spec["tensors"][tensor_key]
                        if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                            raise TensorCacheError("Cache tensor shape/dtype does not match manifest")
                        _validate_tensor(value)
                        # Own writable RAM, not retained mmap views. A candidate
                        # mutating inputs cannot alter the persisted fixture.
                        tensors[tensor_key] = value.clone()
                objects = spec["tensor_objects"]
                if (not isinstance(objects, dict)
                        or any(not isinstance(k, str) or not isinstance(v, str) or v not in tensors
                               for k, v in objects.items())):
                    raise TensorCacheError("Invalid cache tensor object mapping")
                if set(objects.values()) != set(tensors):
                    raise TensorCacheError("Unreferenced tensors in cache manifest")
                restored = {}
                physical_used = set()
                for object_key, tensor_key in objects.items():
                    # Content deduplication saves disk space, not writable
                    # object identity. Reuse the first owned copy; subsequent
                    # independent objects need their own storage. Repeated
                    # references to one logical object still alias in _decode.
                    value = tensors[tensor_key]
                    restored[object_key] = value.clone() if tensor_key in physical_used else value
                    physical_used.add(tensor_key)
                used = set()
                decoded[name] = _decode(spec["tree"], restored, used)
                if used != set(restored):
                    raise TensorCacheError("Unreferenced tensor objects in cache manifest")
            inputs = decoded[FILES[0]]
            validate_inputs(inputs, case)
            references = decoded[FILES[1]]
            if (not isinstance(references, tuple) or len(references) != 2
                    or references[0] is None or references[1] is None):
                raise TensorCacheError("Cache requires oracle and native reference roles")
            return CachedCase(inputs, references[0], references[1], key)
        except TensorCacheError:
            raise
        except Exception as exc:
            raise TensorCacheError(f"Invalid tensor cache entry {key}: {exc}") from exc

    def write(self, identity, case, inputs, oracle, native):
        """Publish complete entries atomically; never overwrite an existing entry."""
        validate_inputs(inputs, case)
        if oracle is None or native is None:
            raise TensorCacheError("Cache build requires successful oracle and native references")
        _, save_file = _safetensors()
        key = self.key(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / key
        if destination.exists():
            self.load(identity, case)
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=".building-", dir=self.root))
        try:
            files = {}
            for name, value in zip(FILES, (inputs, (oracle, native))):
                encoder = _Encoder(deduplicate=name == FILES[1])
                tree = encoder.encode(value)
                path = temporary / name
                save_file(encoder.tensors, str(path), metadata={"format": FORMAT, "version": str(VERSION)})
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                files[name] = dict(bytes=path.stat().st_size, sha256=file_sha256(path), tree=tree,
                                   tensor_objects=encoder.tensor_objects,
                                   tensors={k: dict(shape=list(v.shape), dtype=str(v.dtype))
                                            for k, v in encoder.tensors.items()})
                del encoder
            manifest = dict(format=FORMAT, version=VERSION, key=key, identity=identity, files=files,
                            producer=dict(torch=str(torch.__version__), threads=torch.get_num_threads(),
                                          interop_threads=torch.get_num_interop_threads(),
                                          torch_config=torch.__config__.show()))
            with (temporary / "manifest.json").open("w") as stream:
                stream.write(_json(manifest) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(temporary)
            try:
                os.rename(temporary, destination)
                _sync_directory(self.root)
            except OSError:
                # A concurrent trusted builder may have published first. Validate
                # it; an incomplete/corrupt entry is not silently overwritten.
                if self.load(identity, case) is None:
                    raise
            return destination
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def add_cache_arguments(parser):
    parser.add_argument("--tensor-cache-dir", default="", help="Trusted, prebuilt tensor cache directory")
    parser.add_argument("--tensor-cache-mode", choices=("off", "prefer", "require"), default="off",
                        help="off: generate; prefer: compute on miss or unsupported case; "
                             "require: fail on miss or unsupported case")
    parser.add_argument("--tensor-cache-task", action="append", default=[],
                        help="Exact tasks-root-relative task to cache (repeatable); all others generate normally")


def configure_cache(config, args):
    config.tensor_cache_dir = getattr(args, "tensor_cache_dir", "")
    if config.tensor_cache_dir:
        config.tensor_cache_dir = str(Path(config.tensor_cache_dir).expanduser().resolve())
    config.tensor_cache_mode = getattr(args, "tensor_cache_mode", "off")
    config.tensor_cache_tasks = list(getattr(args, "tensor_cache_task", []))
    validate_config(config)
