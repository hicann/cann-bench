import torch
import torch.nn as nn


class ModelNew(nn.Module):
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
        import kernel_gen_ops

        return kernel_gen_ops.group_norm_silu_quant(
            self_tensor,
            gamma,
            beta,
            quant_scale,
            self.group,
            self.eps,
            self.activate_silu,
        )
