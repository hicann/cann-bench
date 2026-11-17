from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_output_size(values: Iterable[int]) -> list[int]:
    output_size = [int(value) for value in values]
    if len(output_size) != 3:
        raise ValueError(f"output_size must contain exactly 3 integers, got {output_size}")
    return output_size


class Model(nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = _normalize_output_size(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool3d(x, self.output_size)
