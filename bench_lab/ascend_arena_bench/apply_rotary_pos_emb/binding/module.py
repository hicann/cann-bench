from __future__ import annotations

import torch
import torch.nn as nn


class ModelNew(nn.Module):
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
        import kernel_gen_ops

        # pybind.cpp 将函数导出为 custom_pybind_api（而非 apply_rotary_pos_emb）
        return kernel_gen_ops.custom_pybind_api(
            query,
            key,
            cos,
            sin,
            self.layout,
            self.rotary_mode,
        )
