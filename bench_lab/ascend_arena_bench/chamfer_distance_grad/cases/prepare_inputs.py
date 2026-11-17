from __future__ import annotations

import hashlib
from typing import Any

import torch


_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "int32": torch.int32,
}
_EDGE_WINDOWS = {
    "float16": (-16.0, -8.0, 8.0, 16.0),
    "float32": (-128.0, -16.0, 16.0, 128.0),
}


def _seed(case_id: str, name: str) -> int:
    digest = hashlib.sha256(f"{case_id}:{name}".encode("utf-8")).digest()
    return (torch.initial_seed() ^ int.from_bytes(digest[:8], "little")) % (2**63)


def _generator(case_id: str, name: str) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_seed(case_id, name))
    return gen


def _default_tensor(shape: tuple[int, ...], dtype: torch.dtype, low: float, high: float, gen: torch.Generator) -> torch.Tensor:
    if 0 in shape:
        return torch.empty(shape, dtype=dtype)
    base = torch.rand(shape, generator=gen, dtype=torch.float32)
    return (low + (high - low) * base).to(dtype=dtype)


def _precision_edge_tensor(shape: tuple[int, ...], dtype_name: str, gen: torch.Generator) -> torch.Tensor:
    if 0 in shape:
        return torch.empty(shape, dtype=_DTYPE_MAP[dtype_name])
    neg_low, neg_high, pos_low, pos_high = _EDGE_WINDOWS[dtype_name]
    selector = torch.rand(shape, generator=gen, dtype=torch.float32)
    neg = neg_low + (neg_high - neg_low) * torch.rand(shape, generator=gen, dtype=torch.float32)
    pos = pos_low + (pos_high - pos_low) * torch.rand(shape, generator=gen, dtype=torch.float32)
    return torch.where(selector < 0.5, neg, pos).to(dtype=_DTYPE_MAP[dtype_name])


def _apply_nan_inf_mix(tensor: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    flat = tensor.reshape(-1).clone()
    count = max(1, flat.numel() // 1000)
    perm = torch.randperm(flat.numel(), generator=gen)
    proxy_values = torch.tensor([-12.0, -8.0, 8.0, 12.0], dtype=torch.float32)
    flat[perm[:count]] = proxy_values[torch.arange(count) % proxy_values.numel()].to(flat.dtype)
    return flat.reshape(tensor.shape)


def _apply_outlier_noisy(tensor: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    flat = tensor.reshape(-1).clone()
    count = max(1, flat.numel() // 1000)
    perm = torch.randperm(flat.numel(), generator=gen)
    flat[perm[:count]] *= 10.0
    return flat.reshape(tensor.shape)


def _float_tensor(case_id: str, name: str, spec: dict[str, Any]) -> torch.Tensor:
    dtype_name = spec["dtype"]
    shape = tuple(spec["shape"])
    init = spec.get("input_init", "default")
    gen = _generator(case_id, name)
    if init == "precision_edge":
        return _precision_edge_tensor(shape, dtype_name, gen)
    tensor = _default_tensor(shape, _DTYPE_MAP[dtype_name], float(spec.get("min", -1.0)), float(spec.get("max", 1.0)), gen)
    if init == "nan_inf_mix":
        return _apply_nan_inf_mix(tensor, gen)
    if init == "outlier_noisy":
        return _apply_outlier_noisy(tensor, gen)
    return tensor


def _index_tensor(case_id: str, name: str, spec: dict[str, Any], *, upper: int) -> torch.Tensor:
    shape = tuple(spec["shape"])
    if 0 in shape:
        return torch.empty(shape, dtype=torch.int32)
    if spec.get("input_init") == "identity":
        return torch.arange(shape[-1], dtype=torch.int32).expand(shape).clone()
    gen = _generator(case_id, name)
    high = max(upper, 1)
    return torch.randint(0, high, shape, generator=gen, dtype=torch.int32)


def _to_device(tensor: torch.Tensor, device: Any = None) -> torch.Tensor:
    if device is None:
        return tensor
    return tensor.to(device)


def get_init_inputs(case: dict[str, Any], device: Any = None) -> tuple[()]:
    del case, device
    return ()


def get_inputs(case: dict[str, Any], device: Any = None) -> tuple[torch.Tensor, ...]:
    inputs = case["inputs"]
    case_id = case["case_id"]
    xyz1 = _float_tensor(case_id, "xyz1", inputs["xyz1"])
    xyz2 = _float_tensor(case_id, "xyz2", inputs["xyz2"])
    idx1 = _index_tensor(case_id, "idx1", inputs["idx1"], upper=xyz2.shape[1] if xyz2.ndim >= 2 else 0)
    idx2 = _index_tensor(case_id, "idx2", inputs["idx2"], upper=xyz1.shape[1] if xyz1.ndim >= 2 else 0)
    grad_dist1 = _float_tensor(case_id, "grad_dist1", inputs["grad_dist1"])
    grad_dist2 = _float_tensor(case_id, "grad_dist2", inputs["grad_dist2"])
    return (
        _to_device(xyz1, device),
        _to_device(xyz2, device),
        _to_device(idx1, device),
        _to_device(idx2, device),
        _to_device(grad_dist1, device),
        _to_device(grad_dist2, device),
    )
