import torch
import torch.nn as nn
import torch_npu  # noqa: F401


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        idx1: torch.Tensor,
        idx2: torch.Tensor,
        grad_dist1: torch.Tensor,
        grad_dist2: torch.Tensor,
    ):
        import kernel_gen_ops

        return kernel_gen_ops.chamfer_distance_grad(
            xyz1,
            xyz2,
            idx1,
            idx2,
            grad_dist1,
            grad_dist2,
        )
