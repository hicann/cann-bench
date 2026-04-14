import torch

"""
MLA算子Torch Golden参考实现

多头潜在注意力 (Multi-Head Latent Attention)，通过低秩压缩 KV 缓存降低推理内存，使用潜在向量 (latent vector) 进行注意力计算
公式:
    c_kv = x @ W_dkv  (低秩压缩)
    K = c_kv @ W_uk   (解压缩得到 key)
    V = c_kv @ W_uv   (解压缩得到 value)
    y = softmax(Q @ K^T / sqrt(d)) @ V
"""
def mla(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    w_uk: torch.Tensor,
    w_uv: torch.Tensor,
    numHeads: int = 1,
    scaleValue: float = -1.0
) -> torch.Tensor:
    """
    多头潜在注意力 (Multi-Head Latent Attention)

    公式:
        K = compressed_kv @ W_uk  (解压缩得到 key)
        V = compressed_kv @ W_uv  (解压缩得到 value)
        y = softmax(Q @ K^T / sqrt(d)) @ V

    Args:
        query: 查询张量 [B, S, N, D]
        compressed_kv: 低秩压缩的 KV 缓存 [B, S_kv, D_c]，D_c < N*D
        w_uk: key 解压缩权重 [D_c, N, D]
        w_uv: value 解压缩权重 [D_c, N, D]
        numHeads: 注意力头数
        scaleValue: 缩放因子，<=0时自动使用 1/sqrt(D)

    Returns:
        输出张量 [B, S, N, D]
    """
    B, S_kv, D_c = compressed_kv.shape
    B, S, N, D = query.shape

    if scaleValue <= 0:
        scaleValue = 1.0 / (D ** 0.5)

    # 解压缩 KV: [B, S_kv, D_c] @ [D_c, N*D] -> [B, S_kv, N, D]
    key = torch.matmul(compressed_kv, w_uk.reshape(D_c, N * D)).reshape(B, S_kv, N, D)
    value = torch.matmul(compressed_kv, w_uv.reshape(D_c, N * D)).reshape(B, S_kv, N, D)

    # 转置为注意力计算格式
    q = query.transpose(1, 2)   # [B, N, S, D]
    k = key.transpose(1, 2)     # [B, N, S_kv, D]
    v = value.transpose(1, 2)   # [B, N, S_kv, D]

    # 缩放点积注意力: [B, N, S, D] @ [B, N, D, S_kv] -> [B, N, S, S_kv]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scaleValue
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)

    # 加权求和: [B, N, S, S_kv] @ [B, N, S_kv, D] -> [B, N, S, D]
    out = torch.matmul(attn_weights, v)

    # 转回 [B, S, N, D]
    y = out.transpose(1, 2)
    return y
