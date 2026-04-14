import torch

"""
MhcSinkhorn算子Torch Golden参考实现

MHC(Multi-Head Communication) Sinkhorn 算子，用于实现基于 Sinkhorn 迭代的软路由分配
公式: P = Sinkhorn(exp(logits / temperature), n_iters)
"""
def mhc_sinkhorn(
    logits: torch.Tensor, temperature: float = 1.0, n_iters: int = 3
) -> torch.Tensor:
    """
    MHC Sinkhorn 软路由分配算子

    公式: P = Sinkhorn(exp(logits / temperature), n_iters)

    Args:
        logits: 输入 logits 张量，shape [B, S, E]
        temperature: Sinkhorn 温度参数
        n_iters: Sinkhorn 迭代次数

    Returns:
        routing_weights: 路由权重张量，shape [B, S, E]
    """
    # Apply temperature scaling
    log_alpha = logits / temperature
    # Sinkhorn iterations in log-domain
    for _ in range(n_iters):
        # Row normalization (log-domain)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization (log-domain)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)
