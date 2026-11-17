import torch
import torch.nn as nn


class Model(nn.Module):
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
        xyz1_cpu = xyz1.to("cpu", dtype=torch.float32)
        xyz2_cpu = xyz2.to("cpu", dtype=torch.float32)
        idx1_cpu = idx1.to("cpu", dtype=torch.int64)
        idx2_cpu = idx2.to("cpu", dtype=torch.int64)
        grad1_cpu = grad_dist1.to("cpu", dtype=torch.float32)
        grad2_cpu = grad_dist2.to("cpu", dtype=torch.float32)

        grad_xyz1 = torch.zeros_like(xyz1_cpu)
        grad_xyz2 = torch.zeros_like(xyz2_cpu)

        for batch in range(grad1_cpu.shape[0]):
            for point in range(grad1_cpu.shape[1]):
                dst = int(idx1_cpu[batch, point].item())
                delta = 2.0 * grad1_cpu[batch, point] * (xyz1_cpu[batch, point] - xyz2_cpu[batch, dst])
                grad_xyz1[batch, point] += delta
                grad_xyz2[batch, dst] -= delta
            for point in range(grad2_cpu.shape[1]):
                dst = int(idx2_cpu[batch, point].item())
                delta = 2.0 * grad2_cpu[batch, point] * (xyz2_cpu[batch, point] - xyz1_cpu[batch, dst])
                grad_xyz2[batch, point] += delta
                grad_xyz1[batch, dst] -= delta

        return grad_xyz1.to(dtype=xyz1.dtype), grad_xyz2.to(dtype=xyz2.dtype)
