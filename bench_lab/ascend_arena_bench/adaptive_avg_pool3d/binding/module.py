from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _normalize_output_size(values: Iterable[int]) -> list[int]:
    output_size = [int(value) for value in values]
    if len(output_size) != 3:
        raise ValueError(f"output_size must contain exactly 3 integers, got {output_size}")
    return output_size


class ModelNew(nn.Module):
    def __init__(self, output_size) -> None:
        super().__init__()
        self.output_size = _normalize_output_size(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops

        # pybind.cpp 将函数导出为 custom_pybind_api（而非 adaptive_avg_pool3d）
        return kernel_gen_ops.custom_pybind_api(x, self.output_size)
