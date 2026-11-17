import torch


def _move_to_device(tensor, device=None):
    if device is None:
        return tensor
    return tensor.to(device)


def _build_default_tensor(spec):
    shape = tuple(spec["shape"])
    dtype = getattr(torch, spec["dtype"])
    low = spec.get("min", -1.0)
    high = spec.get("max", 1.0)
    if dtype.is_floating_point:
        base = torch.rand(shape, device="cpu", dtype=torch.float32)
        return (low + (high - low) * base).to(dtype=dtype)
    return torch.randint(int(low), int(high) + 1, shape, device="cpu", dtype=dtype)


def _init_default(spec, *, device=None):
    return _move_to_device(_build_default_tensor(spec), device)


def _init_nan_inf_mix(spec, *, device=None):
    tensor = _build_default_tensor(spec)
    if tensor.numel() == 0:
        return _move_to_device(tensor, device)
    flat = tensor.reshape(-1)
    picks = max(1, tensor.numel() // 100)
    index = torch.randperm(tensor.numel(), device="cpu")[:picks]
    pool = torch.tensor([float("nan"), float("inf"), float("-inf")], dtype=tensor.dtype)
    flat[index] = pool[torch.randint(0, 3, (picks,), device="cpu")]
    return _move_to_device(tensor, device)


def _init_precision_edge(spec, *, device=None):
    shape = tuple(spec["shape"])
    dtype = getattr(torch, spec["dtype"])
    base = torch.rand(shape, device="cpu", dtype=torch.float32)
    sign = (torch.randint(0, 2, shape, device="cpu") * 2 - 1).to(torch.float32)
    if dtype == torch.float16:
        magnitude = 60000.0 + 5000.0 * base
    elif dtype == torch.bfloat16:
        magnitude = 1e30 + (1e38 - 1e30) * base
    else:
        magnitude = 1e30 + (1e37 - 1e30) * base
    tensor = (sign * magnitude).to(dtype=dtype)
    return _move_to_device(tensor, device)


def _init_outlier_noisy(spec, *, device=None):
    tensor = _build_default_tensor(spec)
    if tensor.numel() == 0:
        return _move_to_device(tensor, device)
    flat = tensor.reshape(-1)
    picks = max(1, tensor.numel() // 1000)
    index = torch.randperm(tensor.numel(), device="cpu")[:picks]
    flat[index] = flat[index] * 1000.0
    return _move_to_device(tensor, device)


_INIT_DISPATCH = {
    "default": _init_default,
    "nan_inf_mix": _init_nan_inf_mix,
    "precision_edge": _init_precision_edge,
    "outlier_noisy": _init_outlier_noisy,
}


def _build_x(spec, *, device=None):
    init_mode = spec.get("input_init", "default")
    return _INIT_DISPATCH[init_mode](spec, device=device)


def _normalize_dim(rank, dim):
    return dim if dim >= 0 else rank + dim


def _build_index(case, *, device=None):
    x_shape = tuple(case["inputs"]["x"]["shape"])
    index_spec = case["inputs"]["index"]
    index_shape = tuple(index_spec["shape"])
    index_dtype = getattr(torch, index_spec["dtype"])
    dim = _normalize_dim(len(x_shape), int(case["attrs"]["dim"]["value"]))
    dim_size = int(x_shape[dim])
    if dim_size == 0:
        index = torch.empty(index_shape, device="cpu", dtype=index_dtype)
    else:
        index = torch.randint(0, dim_size, index_shape, device="cpu", dtype=index_dtype)
    return _move_to_device(index, device)


def get_init_inputs(case, device=None):
    del device
    return [int(case["attrs"]["dim"]["value"])]


def get_inputs(case, device=None):
    return (
        _build_x(case["inputs"]["x"], device=device),
        _build_index(case, device=device),
    )
