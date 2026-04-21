import torch

"""
GroupNorm 算子 Torch Golden 参考实现

计算分组归一化

公式:
    y = (x - mean) / sqrt(var + eps) * gamma + beta

参考 PyTorch API: torch.nn.functional.group_norm
    https://pytorch.org/docs/stable/generated/torch.nn.functional.group_norm.html

Parameters:
    - x: (N, C, ...) 输入张量，N=batch size, C=通道数
    - gamma: (C,) 缩放参数
    - beta: (C,) 偏置参数
    - num_groups: int - 分组数，C 必须能被 num_groups 整除
    - epsilon: float, 默认 1e-5 - 数值稳定性参数
"""


def group_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
    epsilon: float = 1e-5
) -> torch.Tensor:
    """
    计算分组归一化

    Args:
        x: 输入张量，shape (N, C, ...) 或 (N, C)
           N = batch size, C = 通道数
           C 必须能被 num_groups 整除
        gamma: 缩放参数，shape (C,)
        beta: 偏置参数，shape (C,)
        num_groups: 分组数，将 C 个通道分为 num_groups 组
                    每组内独立计算均值和方差
        epsilon: 数值稳定性参数，防止除零
                 默认值 1e-5

    Returns:
        分组归一化后的张量，shape 与输入相同

    Examples:
        >>> x = torch.randn(8, 32, 64, 64)
        >>> gamma = torch.ones(32)
        >>> beta = torch.zeros(32)
        >>> y = group_norm(x, gamma, beta, num_groups=8, epsilon=1e-5)
    """
    y = torch.nn.functional.group_norm(
        input=x,
        num_groups=num_groups,
        weight=gamma,
        bias=beta,
        eps=epsilon
    )

    return y
