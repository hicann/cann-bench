from __future__ import annotations

import hashlib
from typing import Any

import torch


_DTYPE_MAP = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
_EDGE_WINDOWS = {
    "float16": (-16.0, -8.0, 8.0, 16.0),
    "float32": (-128.0, -16.0, 16.0, 128.0),
    "bfloat16": (-64.0, -16.0, 16.0, 64.0),
}


def _generator(case_id: str, name: str) -> torch.Generator:
    gen = torch.Generator(device="cpu")
    digest = hashlib.sha256(f"{case_id}:{name}".encode("utf-8")).digest()
    gen.manual_seed(int.from_bytes(digest[:8], "little"))
    return gen


def _default_tensor(shape: tuple[int, ...], dtype: torch.dtype, low: float, high: float, gen: torch.Generator) -> torch.Tensor:
    if 0 in shape:
        return torch.empty(shape, dtype=dtype)
    base = torch.rand(shape, generator=gen, dtype=torch.float32)
    return (low + (high - low) * base).to(dtype=dtype)


def _precision_edge_tensor(shape: tuple[int, ...], dtype_name: str, gen: torch.Generator) -> torch.Tensor:
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
    proxies = torch.tensor([-12.0, -8.0, 8.0, 12.0], dtype=torch.float32)
    chosen = proxies[torch.randint(0, proxies.numel(), (count,), generator=gen)]
    flat[perm[:count]] = chosen.to(dtype=flat.dtype)
    return flat.reshape(tensor.shape)


def _apply_outlier_noisy(tensor: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    flat = tensor.reshape(-1).clone()
    count = max(1, flat.numel() // 1000)
    flat[torch.randperm(flat.numel(), generator=gen)[:count]] *= 10.0
    return flat.reshape(tensor.shape)


def _make_tensor(case_id: str, name: str, spec: dict[str, Any], *, device: Any = None) -> torch.Tensor:
    shape = tuple(int(dim) for dim in spec["shape"])
    dtype_name = str(spec["dtype"])
    gen = _generator(case_id, name)
    init = str(spec.get("input_init", "default"))
    if init == "precision_edge":
        tensor = _precision_edge_tensor(shape, dtype_name, gen)
    else:
        tensor = _default_tensor(shape, _DTYPE_MAP[dtype_name], float(spec.get("min", -1.0)), float(spec.get("max", 1.0)), gen)
        if init == "nan_inf_mix":
            tensor = _apply_nan_inf_mix(tensor, gen)
        elif init == "outlier_noisy":
            tensor = _apply_outlier_noisy(tensor, gen)
    return tensor if device is None else tensor.to(device)


def get_inputs(case: dict[str, Any], device: Any = None) -> tuple[torch.Tensor]:
    return (_make_tensor(str(case["case_id"]), "x", case["inputs"]["x"], device=device),)


def get_init_inputs(case: dict[str, Any], device: Any = None) -> tuple[list[int]]:
    del device
    output_size = case.get("attrs", {}).get("output_size", [])
    if isinstance(output_size, dict):
        output_size = output_size.get("value", [])
    if isinstance(output_size, str):
        output_size = [int(value) for value in output_size.replace("x", ",").split(",") if value]
    return ([int(value) for value in output_size],)
