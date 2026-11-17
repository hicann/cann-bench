import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = int(output_size)

    def forward(
        self,
        permute: torch.Tensor,
        input_offsets: torch.Tensor,
        output_offsets: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.empty((self.output_size,), dtype=permute.dtype, device=permute.device)
        for index in range(int(permute.numel())):
            out_start = int(output_offsets[index].item())
            out_end = int(output_offsets[index + 1].item())
            in_start = int(input_offsets[int(permute[index].item())].item())
            result[out_start:out_end] = torch.arange(
                in_start,
                in_start + (out_end - out_start),
                dtype=permute.dtype,
                device=permute.device,
            )
        return result
