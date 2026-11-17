import torch


def _to_device(tensor: torch.Tensor, device=None) -> torch.Tensor:
    return tensor if device is None else tensor.to(device)


def _resolve_mode(case_id: str, spec: dict) -> str:
    init = str(spec.get("input_init", "")).strip()
    if init:
        return init
    if "numeric_nan_inf" in case_id:
        return "nan_inf_mix"
    if "numeric_precision_edge" in case_id:
        return "precision_edge"
    if "numeric_outlier" in case_id:
        return "outlier_noisy"
    return "default"


def _partition_lengths(num_parts: int, total: int, *, mode: str, variant: str) -> torch.Tensor:
    if num_parts == 0:
        return torch.zeros((0,), dtype=torch.int64)
    if total < num_parts:
        raise ValueError(f"output_size={total} must be >= num_parts={num_parts}")
    if total == 0:
        return torch.zeros((num_parts,), dtype=torch.int64)

    lengths = torch.ones((num_parts,), dtype=torch.int64)
    remaining = total - num_parts
    if remaining == 0:
        return lengths

    if mode == "default":
        weights = torch.rand((num_parts,), dtype=torch.float64)
        extra = torch.floor(weights / weights.sum() * remaining).to(torch.int64)
        lengths += extra
    elif mode == "nan_inf_mix":
        heavy = torch.arange(num_parts, dtype=torch.int64) % 2
        extra = torch.floor(heavy.to(torch.float64) / max(float(heavy.sum().item()), 1.0) * remaining).to(torch.int64)
        lengths += extra
    elif mode == "precision_edge":
        lengths[-1] += remaining
    elif mode == "outlier_noisy":
        focus = 0 if variant == "input" else num_parts // 2
        spike = max(1, (remaining * 3) // 4)
        lengths[focus] += spike
        remaining -= spike
        if remaining > 0:
            extra = torch.floor(torch.rand((num_parts,), dtype=torch.float64) * (remaining / max(num_parts, 1))).to(torch.int64)
            lengths += extra
    else:
        raise ValueError(f"unsupported input_init: {mode}")

    deficit = total - int(lengths.sum().item())
    if deficit > 0:
        order = torch.arange(num_parts, dtype=torch.int64)
        if variant == "output":
            order = torch.flip(order, dims=[0])
        for index in range(deficit):
            lengths[order[index % num_parts]] += 1
    elif deficit < 0:
        order = torch.arange(num_parts - 1, -1, -1, dtype=torch.int64)
        for index in order.tolist():
            removable = min(int(lengths[index].item()) - 1, -deficit)
            if removable > 0:
                lengths[index] -= removable
                deficit += removable
            if deficit == 0:
                break
    return lengths


def _build_offsets(num_parts: int, total: int, *, mode: str, variant: str) -> torch.Tensor:
    lengths = _partition_lengths(num_parts, total, mode=mode, variant=variant)
    offsets = torch.zeros((num_parts + 1,), dtype=torch.int64)
    if num_parts > 0:
        offsets[1:] = torch.cumsum(lengths, dim=0)
    return offsets.to(torch.int32)


def _build_input_offsets_from_output(
    permute: torch.Tensor,
    output_offsets: torch.Tensor,
) -> torch.Tensor:
    num_parts = int(permute.numel())
    if num_parts == 0:
        return torch.zeros((1,), dtype=torch.int32)

    output_lengths = output_offsets[1:].to(torch.int64) - output_offsets[:-1].to(torch.int64)
    source_lengths = torch.zeros((num_parts,), dtype=torch.int64)
    for dst_index in range(num_parts):
        source_lengths[int(permute[dst_index].item())] = output_lengths[dst_index]

    input_offsets = torch.zeros((num_parts + 1,), dtype=torch.int64)
    input_offsets[1:] = torch.cumsum(source_lengths, dim=0)
    return input_offsets.to(torch.int32)


def _build_permute(num_parts: int, *, mode: str) -> torch.Tensor:
    if num_parts == 0:
        return torch.zeros((0,), dtype=torch.int32)
    if mode == "precision_edge":
        permute = torch.arange(num_parts - 1, -1, -1, dtype=torch.int64)
    elif mode == "outlier_noisy":
        permute = torch.roll(torch.arange(num_parts, dtype=torch.int64), shifts=1)
    else:
        permute = torch.randperm(num_parts, dtype=torch.int64)
    return permute.to(torch.int32)


def get_init_inputs(case, device=None):
    return [int(case["attrs"]["output_size"]["value"])]


def get_inputs(case, device=None):
    case_id = str(case["case_id"])
    specs = case["inputs"]
    num_parts = int(specs["permute"]["shape"][0])
    output_size = int(case["attrs"]["output_size"]["value"])
    mode = _resolve_mode(case_id, specs["permute"])
    permute = _build_permute(num_parts, mode=mode)
    output_offsets = _build_offsets(num_parts, output_size, mode=mode, variant="output")
    input_offsets = _build_input_offsets_from_output(permute, output_offsets)
    return (
        _to_device(permute, device=device),
        _to_device(input_offsets, device=device),
        _to_device(output_offsets, device=device),
    )
