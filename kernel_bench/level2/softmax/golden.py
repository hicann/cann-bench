import torch

"""
Softmax 算子 Torch Golden 参考实现

沿指定维度计算 Softmax 归一化

公式:
    y_i = exp(x_i) / sum(exp(x_j))

参考 PyTorch API: torch.nn.functional.softmax
    https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html

Parameters:
    - x: 任意维度输入张量
    - dim: int, 默认 -1 - 计算 Softmax 的维度
"""


def softmax(
    x: torch.Tensor,
    dim: int = -1
) -> torch.Tensor:
    """
    沿指定维度计算 Softmax 归一化

    Args:
        x: 输入张量，任意 shape
        dim: 计算 Softmax 的维度，默认为 -1（最后一维）

    Returns:
        Softmax 归一化后的张量，shape 与输入相同
        输出元素值在 [0, 1] 范围内，且沿 dim 维度求和为 1

    Examples:
        >>> x = torch.randn(1024, 2048)
        >>> y = softmax(x, dim=-1)
    """
    y = torch.nn.functional.softmax(x, dim=dim)

    return y
