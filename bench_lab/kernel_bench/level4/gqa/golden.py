import torch

"""
GQA算子Torch Golden参考实现

分组查询注意力 (Grouped Query Attention)，多个 query head 共享一组 key/value head，减少 KV cache 内存占用
公式:
    对于第 i 个 query head，使用第 floor(i * numKVHeads / numHeads) 个 KV head
    head_i = softmax(Q_i @ K_g(i)^T / sqrt(d_k)) @ V_g(i)
    GQA = Concat(head_1, ..., head_h) @ W_o
"""
def gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight_o: torch.Tensor | None = None,
    numHeads: int = 1,
    numKVHeads: int = 1,
    scaleValue: float = -1.0
) -> torch.Tensor:
    """
    分组查询注意力 (Grouped Query Attention)

    公式:
        对于第 i 个 query head，使用第 floor(i * numKVHeads / numHeads) 个 KV head
        head_i = softmax(Q_i @ K_g(i)^T / sqrt(d_k)) @ V_g(i)
        GQA = Concat(head_1, ..., head_h) @ W_o

    Args:
        query: 查询张量 [B, S, N_q, D]，N_q = numHeads
        key: 键张量 [B, S_kv, N_kv, D]，N_kv = numKVHeads
        value: 值张量 [B, S_kv, N_kv, D]
        weight_o: 输出投影权重 [N_q * D, N_q * D] (可为None表示不做输出投影)
        numHeads: query 头数
        numKVHeads: KV 头数，需整除 numHeads
        scaleValue: 缩放因子，<=0时自动使用 1/sqrt(d_k)

    Returns:
        输出张量 [B, S, N_q, D]
    """
    B, S, N_q, D = query.shape
    S_kv = key.shape[1]
    N_kv = key.shape[2]

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 扩展 KV heads 以匹配 Q heads: 每个 KV head 重复 numHeads // numKVHeads 次
    num_repeats = numHeads // numKVHeads
    # [B, S_kv, N_kv, D] -> [B, S_kv, N_kv, 1, D] -> [B, S_kv, N_kv, num_repeats, D] -> [B, S_kv, N_q, D]
    key = key.unsqueeze(3).expand(B, S_kv, N_kv, num_repeats, D).reshape(B, S_kv, N_q, D)
    value = value.unsqueeze(3).expand(B, S_kv, N_kv, num_repeats, D).reshape(B, S_kv, N_q, D)

    # 转置为 [B, N_q, S, D] 和 [B, N_q, S_kv, D]
    q = query.transpose(1, 2)   # [B, N_q, S, D]
    k = key.transpose(1, 2)     # [B, N_q, S_kv, D]
    v = value.transpose(1, 2)   # [B, N_q, S_kv, D]

    # 缩放点积注意力: [B, N_q, S, D] @ [B, N_q, D, S_kv] -> [B, N_q, S, S_kv]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # 加权求和: [B, N_q, S, S_kv] @ [B, N_q, S_kv, D] -> [B, N_q, S, D]
    attn_output = torch.matmul(attn_weights, v)

    # 转回 [B, S, N_q, D]
    y = attn_output.transpose(1, 2)

    # 输出投影 (可选)
    if weight_o is not None:
        # [B, S, N_q * D] @ [N_q * D, N_q * D] -> [B, S, N_q * D] -> [B, S, N_q, D]
        y_flat = y.reshape(B, S, N_q * D)
        y_flat = torch.nn.functional.linear(y_flat, weight_o)
        y = y_flat.reshape(B, S, N_q, D)

    return y
