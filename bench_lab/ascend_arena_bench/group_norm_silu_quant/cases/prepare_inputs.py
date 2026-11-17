from typing import Any

import torch


def _torch_dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _attr_value(spec: Any):
    if isinstance(spec, dict) and "value" in spec:
        return spec["value"]
    return spec


def _cpu_uniform(shape, low: float, high: float) -> torch.Tensor:
    if any(int(dim) == 0 for dim in shape):
        return torch.empty(shape, dtype=torch.float32)
    return torch.empty(shape, dtype=torch.float32).uniform_(low, high)


def _finite_uniform(shape, low: Any, high: Any, fallback_low: float = -1.0, fallback_high: float = 1.0) -> torch.Tensor:
    try:
        lo = float(low)
        hi = float(high)
        if not torch.isfinite(torch.tensor([lo, hi], dtype=torch.float32)).all():
            raise ValueError("non-finite range")
    except (TypeError, ValueError):
        lo = fallback_low
        hi = fallback_high
    return _cpu_uniform(shape, lo, hi)


def _apply_nan_inf_mix(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return tensor
    count = max(1, flat.numel() // 100)
    indices = torch.randperm(flat.numel())[:count]
    pattern = torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=torch.float32)
    flat[indices] = pattern.repeat((count + 2) // 3)[:count]
    return tensor


def _apply_outlier_noisy(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if flat.numel() == 0:
        return tensor
    count = max(1, flat.numel() // 1000)
    indices = torch.randperm(flat.numel())[:count]
    flat[indices] = flat[indices] * 1000.0
    return tensor


def _precision_edge(shape, dtype_name: str) -> torch.Tensor:
    base = _cpu_uniform(shape, 0.0, 1.0)
    sign = torch.where(base > 0.5, 1.0, -1.0)
    if dtype_name == "float16":
        return sign * (60000.0 + 5000.0 * base)
    if dtype_name == "bfloat16":
        return sign * (1e30 + (1e38 - 1e30) * base)
    return 1e3 + 1e4 * base


def _resolve_init_mode(case: dict, input_name: str, spec: dict) -> str:
    if "input_init" in spec:
        return str(spec["input_init"])
    return "default"


def _make_tensor(case: dict, input_name: str, spec: dict, device="cpu") -> torch.Tensor:
    shape = tuple(int(dim) for dim in spec["shape"])
    dtype_name = str(spec["dtype"])
    init_mode = _resolve_init_mode(case, input_name, spec)
    if "value" in spec:
        tensor = torch.tensor(spec["value"], dtype=torch.float32).reshape(shape)
    elif init_mode == "precision_edge":
        tensor = _precision_edge(shape, dtype_name)
    else:
        low = spec.get("min", -1.0)
        high = spec.get("max", 1.0)
        tensor = _finite_uniform(shape, low, high)
        if init_mode == "nan_inf_mix":
            tensor = _apply_nan_inf_mix(tensor)
        elif init_mode == "outlier_noisy":
            tensor = _apply_outlier_noisy(tensor)
        elif init_mode != "default":
            raise ValueError(f"unsupported input_init: {init_mode}")
    return tensor.to(dtype=_torch_dtype(dtype_name), device=device)


def get_init_inputs(case: dict, device="cpu"):
    del device
    attrs = case["attrs"]
    return (
        int(_attr_value(attrs["group"])),
        float(_attr_value(attrs["eps"])),
        bool(_attr_value(attrs["activate_silu"])),
    )


def get_inputs(case: dict, device="cpu"):
    inputs = case["inputs"]
    return (
        _make_tensor(case, "self", inputs["self"], device=device),
        _make_tensor(case, "gamma", inputs["gamma"], device=device),
        _make_tensor(case, "beta", inputs["beta"], device=device),
        _make_tensor(case, "quant_scale", inputs["quant_scale"], device=device),
    )
