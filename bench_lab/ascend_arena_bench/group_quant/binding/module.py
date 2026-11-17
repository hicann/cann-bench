from __future__ import annotations

import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, dst_type: int = 2):
        super().__init__()
        self.dst_type = int(dst_type)
        if self.dst_type != 2:
            raise ValueError("group_quant standardized binding currently supports dst_type=2 (INT8) only")

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        group_index: torch.Tensor,
        offset: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.group_quant(x, scale, group_index, offset, self.dst_type)
