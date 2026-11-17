from __future__ import annotations

from typing import Any

import torch


def _dtype(name: str) -> torch.dtype:
    aliases = {"float": "float32", "half": "float16", "long": "int64", "string": "int64"}
    return getattr(torch, aliases.get(name.lower(), name.lower()))


def _default_tensor(shape: tuple[int, ...], dtype: torch.dtype, low: float, high: float) -> torch.Tensor:
    data = low + (high - low) * torch.rand(shape, dtype=torch.float32, device="cpu")
    return data.to(dtype)


def _precision_edge_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    low: float | None = None,
    high: float | None = None,
) -> torch.Tensor:
    if low is not None and high is not None:
        # 显式声明了取值范围的 case 以 JSON 为准，避免 cos/sin 等小值输入被极值生成器放大。
        return _default_tensor(shape, dtype, low, high)
    if dtype == torch.float16:
        base = _default_tensor(shape, torch.float32, 60000.0, 65000.0)
    elif dtype == torch.bfloat16:
        base = _default_tensor(shape, torch.float32, 1.0e30, 1.0e38)
    else:
        base = _default_tensor(shape, torch.float32, 1.0e30, 3.0e38)
    sign = torch.where(torch.rand(shape, device="cpu") > 0.5, 1.0, -1.0)
    return (base * sign).to(dtype)


def _inject_nan_inf(data: torch.Tensor) -> torch.Tensor:
    flat = data.reshape(-1)
    count = max(1, flat.numel() // 100)
    idx = torch.randperm(flat.numel(), device="cpu")[:count]
    payload = torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=torch.float32, device="cpu")
    flat[idx] = payload[torch.arange(count, device="cpu") % payload.numel()].to(flat.dtype)
    return data


def _inject_outlier(data: torch.Tensor) -> torch.Tensor:
    flat = data.reshape(-1)
    count = max(1, flat.numel() // 1000)
    idx = torch.randperm(flat.numel(), device="cpu")[:count]
    flat[idx] = flat[idx] * 1000
    return data


def _rand_tensor(spec: dict[str, Any]) -> torch.Tensor:
    shape = tuple(spec["shape"])
    dtype = _dtype(spec["dtype"])
    init = spec.get("input_init", "default")
    low = -1.0 if spec.get("min") is None else float(spec["min"])
    high = 1.0 if spec.get("max") is None else float(spec["max"])
    if init == "precision_edge":
        has_explicit_range = "min" in spec and "max" in spec
        return _precision_edge_tensor(
            shape,
            dtype,
            low=low if has_explicit_range else None,
            high=high if has_explicit_range else None,
        )
    data = _default_tensor(shape, dtype, low, high)
    if init == "nan_inf_mix":
        return _inject_nan_inf(data)
    if init == "outlier_noisy":
        return _inject_outlier(data)
    return data


def _to_device(value: torch.Tensor, device=None) -> torch.Tensor:
    if device is None:
        return value
    if isinstance(device, str) and device == "cpu":
        return value.to("cpu")
    return value.to(device)


def get_init_inputs(case: dict[str, Any], device=None):
    del device
    attrs = case.get("attrs", {})
    return [int(attrs["layout"]["value"]), str(attrs["rotary_mode"]["value"])]


def get_inputs(case: dict[str, Any], device=None):
    inputs = case.get("inputs", {})
    query = _to_device(_rand_tensor(inputs["query"]), device)
    key = _to_device(_rand_tensor(inputs["key"]), device)
    cos = _to_device(_rand_tensor(inputs["cos"]), device)
    sin = _to_device(_rand_tensor(inputs["sin"]), device)
    return query, key, cos, sin
