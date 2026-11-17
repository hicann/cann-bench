import torch
import torch.nn as nn
import torch.nn.functional as F


def _quant_scale_view(x: torch.Tensor, quant_scale: torch.Tensor) -> torch.Tensor:
    if quant_scale.numel() == 1:
        return quant_scale.reshape(1)
    shape = [1] * x.dim()
    shape[1] = x.shape[1]
    return quant_scale.reshape(shape)


class Model(nn.Module):
    def __init__(self, group: int, eps: float, activate_silu: bool = True):
        super().__init__()
        self.group = int(group)
        self.eps = float(eps)
        self.activate_silu = bool(activate_silu)

    def forward(
        self,
        self_tensor: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        quant_scale: torch.Tensor,
    ):
        x32 = self_tensor.to(torch.float32)
        gamma32 = gamma.to(torch.float32)
        beta32 = beta.to(torch.float32)
        quant32 = quant_scale.to(torch.float32)

        reshaped = x32.reshape(x32.shape[0], self.group, -1)
        mean = reshaped.mean(dim=-1)
        var = reshaped.var(dim=-1, unbiased=False)
        rstd = torch.rsqrt(var + self.eps)

        normalized = F.group_norm(x32, self.group, gamma32, beta32, self.eps)
        if self.activate_silu:
            normalized = F.silu(normalized)
        scaled = normalized / _quant_scale_view(x32, quant32)
        out = torch.clamp(torch.round(scaled), -128, 127).to(torch.int8)

        return out, mean.to(self_tensor.dtype), rstd.to(self_tensor.dtype)
