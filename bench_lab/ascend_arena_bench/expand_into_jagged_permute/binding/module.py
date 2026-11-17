import torch
import torch.nn as nn
import torch_npu


class ModelNew(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.output_size = int(output_size)

    def forward(
        self,
        permute: torch.Tensor,
        input_offsets: torch.Tensor,
        output_offsets: torch.Tensor,
    ) -> torch.Tensor:
        import kernel_gen_ops

        return kernel_gen_ops.expand_into_jagged_permute(
            permute,
            input_offsets,
            output_offsets,
            self.output_size,
        )
