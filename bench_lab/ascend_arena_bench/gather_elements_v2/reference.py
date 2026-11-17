import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, dim: int = 0):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, dim=self.dim, index=index.to(dtype=torch.int64))
