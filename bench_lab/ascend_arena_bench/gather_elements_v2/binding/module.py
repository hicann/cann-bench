import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.gather_elements_v2(x, index, self.dim)
