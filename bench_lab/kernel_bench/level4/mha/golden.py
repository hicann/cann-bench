import torch

"""
MHA算子Torch Golden参考实现

多头注意力机制 (Multi-Head Attention)，将输入通过多个注意力头并行计算后拼接输出
公式:
    head_i = Attention(Q_i, K_i, V_i) = softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
    MHA(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
"""
def mha(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    weight_o: torch.Tensor,
    bias_q: torch.Tensor | None = None,
    bias_k: torch.Tensor | None = None,
    bias_v: torch.Tensor | None = None,
    bias_o: torch.Tensor | None = None,
    numHeads: int = 1,
    scaleValue: float = -1.0,
    dropoutRate: float = 0.0
) -> torch.Tensor:
    """
    多头注意力机制 (Multi-Head Attention)

    公式:
        head_i = softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
        MHA = Concat(head_1, ..., head_h) @ W_o

    Args:
        query: 查询张量 [B, S, D]
        key: 键张量 [B, S_kv, D]
        value: 值张量 [B, S_kv, D]
        weight_q: Query 投影权重 [D, D]
        weight_k: Key 投影权重 [D, D]
        weight_v: Value 投影权重 [D, D]
        weight_o: 输出投影权重 [D, D]
        bias_q: Query 投影偏置 [D] (可选)
        bias_k: Key 投影偏置 [D] (可选)
        bias_v: Value 投影偏置 [D] (可选)
        bias_o: 输出投影偏置 [D] (可选)
        numHeads: 注意力头数
        scaleValue: 缩放因子，<=0时自动使用 1/sqrt(d_k)
        dropoutRate: dropout比率

    Returns:
        输出张量 [B, S, D]
    """
    B, S, D = query.shape
    S_kv = key.shape[1]
    d_k = D // numHeads

    if scaleValue <= 0:
        scaleValue = 1.0 / (d_k ** 0.5)

    # 线性投影: [B, S, D] @ [D, D] -> [B, S, D]
    Q = torch.nn.functional.linear(query, weight_q, bias_q)
    K = torch.nn.functional.linear(key, weight_k, bias_k)
    V = torch.nn.functional.linear(value, weight_v, bias_v)

    # 重塑为多头: [B, S, D] -> [B, S, numHeads, d_k] -> [B, numHeads, S, d_k]
    Q = Q.reshape(B, S, numHeads, d_k).transpose(1, 2)
    K = K.reshape(B, S_kv, numHeads, d_k).transpose(1, 2)
    V = V.reshape(B, S_kv, numHeads, d_k).transpose(1, 2)

    # 缩放点积注意力: [B, numHeads, S, d_k] @ [B, numHeads, d_k, S_kv] -> [B, numHeads, S, S_kv]
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # Dropout (仅训练时生效，golden 中不应用)
    # attn_weights = torch.nn.functional.dropout(attn_weights, p=dropoutRate)

    # 加权求和: [B, numHeads, S, S_kv] @ [B, numHeads, S_kv, d_k] -> [B, numHeads, S, d_k]
    attn_output = torch.matmul(attn_weights, V)

    # 拼接多头: [B, numHeads, S, d_k] -> [B, S, numHeads, d_k] -> [B, S, D]
    attn_output = attn_output.transpose(1, 2).reshape(B, S, D)

    # 输出投影: [B, S, D] @ [D, D] -> [B, S, D]
    y = torch.nn.functional.linear(attn_output, weight_o, bias_o)
    return y
