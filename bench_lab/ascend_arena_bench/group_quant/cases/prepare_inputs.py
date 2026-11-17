from __future__ import annotations

from typing import Any

import torch


_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _resolve_device(device: Any) -> torch.device:
    return torch.device("cpu") if device is None else torch.device(device)


def _to_device(tensor: torch.Tensor, device: Any) -> torch.Tensor:
    return tensor.to(_resolve_device(device))


def _rand_float(shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
    if 0 in shape:
        return torch.empty(shape, dtype=torch.float32)
    return torch.rand(shape, dtype=torch.float32) * (high - low) + low


def _safe_precision_edge(shape: tuple[int, ...], dtype_name: str) -> torch.Tensor:
    if dtype_name == "float16":
        low, high = 8.0, 16.0
    elif dtype_name == "bfloat16":
        low, high = 16.0, 64.0
    else:
        low, high = 16.0, 128.0
    base = _rand_float(shape, low, high)
    signs = torch.where(torch.rand(shape, dtype=torch.float32) > 0.5, 1.0, -1.0)
    return base * signs


def _apply_input_init(data: torch.Tensor, spec: dict[str, Any]) -> torch.Tensor:
    mode = spec.get("input_init", "default")
    if mode == "default":
        return data
    flat = data.reshape(-1)
    if flat.numel() == 0:
        return data
    if mode == "nan_inf_mix":
        count = max(1, flat.numel() // 1000)
        perm = torch.randperm(flat.numel())[:count]
        proxies = torch.tensor([8.0, -8.0, 12.0, -12.0], dtype=torch.float32)
        flat[perm] = proxies[torch.arange(count) % proxies.numel()]
        return data
    if mode == "precision_edge":
        return _safe_precision_edge(tuple(data.shape), str(spec["dtype"]))
    if mode == "outlier_noisy":
        count = max(1, flat.numel() // 1000)
        perm = torch.randperm(flat.numel())[:count]
        flat[perm] *= 10.0
        return data
    raise ValueError(f"unsupported input_init: {mode}")


def _rand_tensor(spec: dict[str, Any]) -> torch.Tensor:
    shape = tuple(int(v) for v in spec["shape"])
    dtype_name = str(spec["dtype"])
    dtype = _DTYPE[dtype_name]
    low = float(spec.get("min", -1.0))
    high = float(spec.get("max", 1.0))
    data = _rand_float(shape, low, high)
    data = _apply_input_init(data, spec)
    return data.to(dtype=dtype)


def _build_group_index(spec: dict[str, Any], rows: int) -> torch.Tensor:
    count = int(spec["shape"][0])
    dtype = _DTYPE[str(spec["dtype"])]
    if count == 0:
        return torch.empty((0,), dtype=dtype)
    if rows == 0:
        return torch.zeros((count,), dtype=dtype)
    values = torch.linspace(1, rows, steps=count, dtype=torch.float32).floor().to(torch.int64)
    values = torch.clamp(values, min=0, max=rows)
    values, _ = torch.sort(values)
    values[-1] = rows
    return values.to(dtype=dtype)


def get_init_inputs(case: dict[str, Any], device: Any = None) -> tuple[()]:
    del case, device
    return ()


def get_inputs(case: dict[str, Any], device: Any = None) -> tuple[torch.Tensor, ...]:
    specs = case["inputs"]
    x = _rand_tensor(specs["x"])
    scale = _rand_tensor(specs["scale"]).abs().clamp_min_(1.0e-4)
    group_index = _build_group_index(specs["group_index"], rows=x.shape[0])
    tensors: list[torch.Tensor] = [
        _to_device(x, device),
        _to_device(scale, device),
        _to_device(group_index, device),
    ]
    if "offset" in specs:
        offset = _rand_tensor(specs["offset"]).reshape(-1)[:1]
        tensors.append(_to_device(offset, device))
    return tuple(tensors)
