import torch

"""
RmsNorm 算子 Torch Golden 参考实现

计算 RMS (均方根) 归一化

公式:
    y = x / sqrt(mean(x^2) + eps) * gamma

参考论文: Root Mean Square Layer Normalization
    https://arxiv.org/abs/1910.07467

Parameters:
    - x: (..., D) 输入张量，最后一维为归一化维度
    - gamma: (D,) 缩放参数
    - epsilon: float, 默认 1e-6 - 数值稳定性参数
"""


def rms_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    计算 RMS (均方根) 归一化

    Args:
        x: 输入张量，shape (..., D)
           最后一维 D 为归一化维度
        gamma: 缩放参数，shape (D,)
               与输入最后一维大小相同
        epsilon: 数值稳定性参数，防止除零
                 默认值 1e-6

    Returns:
        RMS 归一化后的张量，shape 与输入相同

    Examples:
        >>> x = torch.randn(32, 128, 4096)
        >>> gamma = torch.ones(4096)
        >>> y = rms_norm(x, gamma, epsilon=1e-6)
    """
    # 计算均方根
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
    # 归一化并乘以缩放参数
    y = x / rms * gamma

    return y
