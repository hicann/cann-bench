from __future__ import annotations

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor, rotary_mode: str) -> torch.Tensor:
    if rotary_mode == "half":
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    if rotary_mode == "quarter":
        q1, q2, q3, q4 = torch.chunk(x, 4, dim=-1)
        return torch.cat((-q2, q1, -q4, q3), dim=-1)
    if rotary_mode == "interleave":
        even = x[..., ::2]
        odd = x[..., 1::2]
        stacked = torch.stack((-odd, even), dim=-1)
        return stacked.reshape_as(x)
    raise ValueError(f"unsupported rotary_mode: {rotary_mode}")


def _forward_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_mode: str,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the kernel formula using an explicit intermediate dtype."""
    query_work = query.to(compute_dtype)
    key_work = key.to(compute_dtype)
    cos_work = cos.to(compute_dtype)
    sin_work = sin.to(compute_dtype)
    query_out = query_work * cos_work + _rotate_half(query_work, rotary_mode) * sin_work
    key_out = key_work * cos_work + _rotate_half(key_work, rotary_mode) * sin_work
    return query_out, key_out


class OracleModel(nn.Module):
    def __init__(self, layout: int = 1, rotary_mode: str = "half") -> None:
        super().__init__()
        self.layout = int(layout)
        self.rotary_mode = str(rotary_mode)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _forward_impl(query, key, cos, sin, self.rotary_mode, torch.float64)


class BenchModel(OracleModel):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query.dtype == torch.bfloat16:
            # The BF16 kernels cast all operands to FP32 before Mul/Add.
            compute_dtype = torch.float32
        else:
            # FP16 and FP32 kernels perform Mul/Add in their input dtype.
            compute_dtype = query.dtype
        query_out, key_out = _forward_impl(query, key, cos, sin, self.rotary_mode, compute_dtype)
        return query_out.to(query.dtype), key_out.to(key.dtype)


class Model(OracleModel):
    pass
