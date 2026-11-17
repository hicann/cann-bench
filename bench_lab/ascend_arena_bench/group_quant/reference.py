from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dst_type: int = 2):
        super().__init__()
        self.dst_type = int(dst_type)
        if self.dst_type != 2:
            raise ValueError("group_quant standardized reference currently supports dst_type=2 (INT8) only")

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        group_index: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rows, _ = x.shape
        result = torch.empty_like(x, dtype=torch.float32, device="cpu")
        x_cpu = x.detach().to("cpu", dtype=torch.float32)
        scale_cpu = scale.detach().to("cpu", dtype=torch.float32)
        group_cpu = group_index.detach().to("cpu", dtype=torch.int64)
        bias = 0.0 if offset is None else float(offset.detach().to("cpu", dtype=torch.float32).reshape(-1)[0])

        start = 0
        for group_id, stop_tensor in enumerate(group_cpu):
            stop = int(stop_tensor.item())
            stop = max(start, min(stop, rows))
            if stop > start:
                result[start:stop] = torch.round(x_cpu[start:stop] * scale_cpu[group_id] + bias)
            start = stop
        if start < rows:
            result[start:] = torch.round(x_cpu[start:] * scale_cpu[-1] + bias)
        return result.clamp_(-128, 127).to(dtype=torch.int8)
