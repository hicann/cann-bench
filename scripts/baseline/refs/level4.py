"""level4 reference implementations on NPU. All level4 ops are L4 fused composites."""
import torch


# Lazy-built 2048×2048 upper-triangular causal mask, reused across all attention
# refs (sparse_mode=3 in npu_fusion_attention requires this compressed mask).
_CAUSAL_MASK_2048 = {}


def _get_causal_mask_2048(device):
    key = str(device)
    if key not in _CAUSAL_MASK_2048:
        m = torch.ones(2048, 2048, dtype=torch.uint8, device=device).triu_(diagonal=1)
        _CAUSAL_MASK_2048[key] = m
    return _CAUSAL_MASK_2048[key]


# Cached zero-filled rope tensors keyed by (device, dtype, shape). The SFA ref
# uses this for Dk=512 cases that need a synthesized zero rope — caching keeps
# the kernel breakdown to k=1 (a fresh torch.zeros each call would inject a
# ZerosLike kernel into the timed region).
_ZERO_ROPE_CACHE = {}


def _get_zero_rope(shape, dtype, device):
    key = (str(device), dtype, tuple(shape))
    t = _ZERO_ROPE_CACHE.get(key)
    if t is None:
        t = torch.zeros(*shape, dtype=dtype, device=device)
        _ZERO_ROPE_CACHE[key] = t
    return t


def gqa_ref(inputs, attrs):
    """Grouped-query attention via `npu_fused_infer_attention_score`.

    Inference-tuned fused attention API. Auto-dispatches:
      - Sq == 1 → IncreFlashAttention branch (decode-optimized)
      - Sq >  1 → PromptFlashAttention branch
    Native GQA support via `num_key_value_heads` — kernel folds the head
    ratio internally (no Python-side broadcast of K/V).

    fp32 cases unsupported by this API → returns None (baseline empty).
    is_causal=True maps to sparse_mode=3 (rightDownCausal); only effective
    for Sq > 1 (causal is a no-op for Sq=1)."""
    import torch_npu
    q, k, v = inputs[0], inputs[1], inputs[2]
    if q.dtype not in (torch.float16, torch.bfloat16):
        return None
    scale = attrs.get("scaleValue", -1.0)
    if scale is None or scale < 0:
        scale = (q.shape[-1]) ** -0.5
    Nq = q.shape[2]
    Nkv = k.shape[2]
    Sq = q.shape[1]
    kwargs = dict(
        num_heads=Nq, num_key_value_heads=Nkv,
        input_layout="BSND", scale=scale,
    )
    if bool(attrs.get("is_causal", False)) and Sq > 1:
        kwargs["atten_mask"] = _get_causal_mask_2048(q.device)
        kwargs["sparse_mode"] = 3
    try:
        out = torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)
    except Exception:
        return None
    return out[0]


def grouped_matmul_swiglu_quant_ref(inputs, attrs):
    """torch_npu.npu_grouped_matmul_swiglu_quant_v2 takes:
      x (int8 [M,K]), [weight] (FRACTAL_NZ list of 3D [E,K,N]),
      [weight_scale] (list of 2D [E,N]), x_scale (1D [M]),
      group_list (int64 1D [E], cumsum form).
    cases.yaml provides x/weight/weight_scale/x_scale as inputs (4 tensors)
    and supplies group_list as an attribute list (cumsum-form ints).
    """
    import torch_npu
    if not hasattr(torch_npu, "npu_grouped_matmul_swiglu_quant_v2"):
        return None
    x, w, ws, xs = inputs[0], inputs[1], inputs[2], inputs[3]
    gl_list = attrs.get("group_list")
    if gl_list is None:
        raise ValueError("grouped_matmul_swiglu_quant_ref: attrs.group_list missing")
    gl = torch.tensor(gl_list, dtype=torch.int64, device=x.device)
    # cast weight to FRACTAL_NZ; inputs from builder are ND int8
    try:
        w_nz = torch_npu.npu_format_cast(w, 29)
    except Exception:
        w_nz = w
    return torch_npu.npu_grouped_matmul_swiglu_quant_v2(x, [w_nz], [ws], xs, gl)


def gru_ref(inputs, attrs):
    """torch.nn.functional has no direct GRU functional. Use torch.nn.GRU
    by re-binding parameters into a temporary module."""
    import torch.nn as nn
    x = inputs[0]  # [seq, batch, input]
    hidden_size = attrs.get("hidden_size") or attrs.get("hiddenSize")
    num_layers = int(attrs.get("num_layers", 1))
    bidirectional = bool(attrs.get("bidirectional", False))
    if hidden_size is None:
        # infer from W_ih shape
        w_ih = inputs[1] if not isinstance(inputs[1], list) else inputs[1][0]
        hidden_size = w_ih.shape[0] // 3
    layer = nn.GRU(input_size=x.shape[-1], hidden_size=int(hidden_size),
                   num_layers=num_layers, bidirectional=bidirectional, batch_first=False)
    layer = layer.to(device=x.device, dtype=x.dtype)
    out, h = layer(x)
    return out, h


def lstm_ref(inputs, attrs):
    """ACL launch-mode does not support LSTM (segfaults instead of raising).
    Return None so the baseline harness skips it gracefully."""
    return None


def mha_ref(inputs, attrs):
    """Multi-head attention via `npu_fused_infer_attention_score`.

    Inference-tuned API: auto-dispatches Sq=1 → IncreFA, Sq>1 → PromptFA.
    fp32 cases unsupported → returns None (baseline empty).
    is_causal=True → sparse_mode=3 (rightDownCausal), only for Sq > 1."""
    import torch_npu
    q, k, v = inputs[0], inputs[1], inputs[2]
    if q.dtype not in (torch.float16, torch.bfloat16):
        return None
    scale = attrs.get("scaleValue", -1.0)
    if scale is None or scale < 0:
        scale = (q.shape[-1]) ** -0.5
    N = q.shape[2]
    Sq = q.shape[1]
    kwargs = dict(
        num_heads=N, num_key_value_heads=N,
        input_layout="BSND", scale=scale,
    )
    if bool(attrs.get("is_causal", False)) and Sq > 1:
        kwargs["atten_mask"] = _get_causal_mask_2048(q.device)
        kwargs["sparse_mode"] = 3
    try:
        out = torch_npu.npu_fused_infer_attention_score(q, k, v, **kwargs)
    except Exception:
        return None
    return out[0]


def mla_ref(inputs, attrs):
    """Multi-head latent attention via `npu_fused_infer_attention_score`.

    Uses the API's native `query_rope` / `key_rope` parameters — no
    Python-side concat. The IncreFA branch (Sq=1) and PromptFA branch
    (Sq>1) both handle nope/rope split + Dqk≠Dv natively.

    Cases that the rope-mode API constraints reject return None (empty
    baseline). Specifically rope mode requires:
      - dtype ∈ {fp16, bf16}                  (fp32 not supported)
      - d_nope == 512, d_rope == 64           (other dims not supported)
      - Nq ∈ {32, 64, 128}

    sparse_mode dispatch:
      - is_causal=False or Sq==1   → sparse=0 (no mask; works empirically
        for Sq>1 too despite doc saying otherwise)
      - is_causal=True + Sq <= 16  → sparse=3 (rightDownCausal, 2048² mask)
      - is_causal=True + Sq >  16  → sparse=4 (band mode, pre=large next=0)"""
    import torch_npu
    q_nope, q_rope, k_nope, k_rope, v = (
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
    )
    if q_nope.dtype not in (torch.float16, torch.bfloat16):
        return None
    d_nope = q_nope.shape[-1]
    d_rope = q_rope.shape[-1]
    if d_nope != 512 or d_rope != 64:
        return None
    layout = attrs.get("inputLayout", "BSND")
    if layout == "BSND":
        Sq, Nq = q_nope.shape[1], q_nope.shape[2]
    elif layout == "BNSD":
        Nq, Sq = q_nope.shape[1], q_nope.shape[2]
    else:
        return None
    if Nq not in (32, 64, 128):
        return None
    is_causal = bool(attrs.get("is_causal", False)) and Sq > 1

    scale = attrs.get("scaleValue", -1.0)
    if scale is None or scale < 0:
        scale = (d_nope + d_rope) ** -0.5

    kwargs = dict(
        query_rope=q_rope, key_rope=k_rope,
        num_heads=Nq, num_key_value_heads=k_nope.shape[2 if layout == "BSND" else 1],
        input_layout=layout, scale=scale,
    )
    if is_causal:
        kwargs["atten_mask"] = _get_causal_mask_2048(q_nope.device)
        if Sq <= 16:
            kwargs["sparse_mode"] = 3
        else:
            # Band-mode shaped as causal: pre_tokens = all past, next = 0
            kwargs["sparse_mode"] = 4
            kwargs["pre_tokens"] = 2147483647
            kwargs["next_tokens"] = 0
    try:
        out = torch_npu.npu_fused_infer_attention_score(
            q_nope, k_nope, v, **kwargs
        )
    except Exception:
        return None
    return out[0]


_MLA_PROLOG_CACHE: dict = {}


def mla_prolog_ref(inputs, attrs):
    """`torch_npu.npu_mla_prolog_v3` (Multi-Latent-Attention prologue, paged).

    cases.csv carries only the 9 model-state tensors (token_x + 4 weights +
    2 rmsnorm gammas + 2 rope tables). The kernel additionally needs
    `kv_cache` / `kr_cache` paged-attention blocks and a `cache_index`, which
    we allocate once per case (Skv=2048 / BlockSize=128 / Nkv=1 decode
    setup) and reuse across trials so they don't pollute the msprof window.
    Three of the four weight matrices are pre-cast to FRACTAL_NZ once and
    cached for the same reason — in production these would already be NZ.
    """
    import math
    import torch_npu
    if not hasattr(torch_npu, "npu_mla_prolog_v3"):
        return None

    (token_x, w_dq, w_uq_qr, w_uk, w_dkv_kr,
     gamma_cq, gamma_ckv, rope_sin, rope_cos) = inputs[:9]

    cache_key = (id(token_x), id(w_dq), id(w_uq_qr), id(w_dkv_kr))
    derived = _MLA_PROLOG_CACHE.get(cache_key)
    if derived is None:
        try:
            w_dq_nz     = torch_npu.npu_format_cast(w_dq.contiguous(), 29)
            w_uq_qr_nz  = torch_npu.npu_format_cast(w_uq_qr.contiguous(), 29)
            w_dkv_kr_nz = torch_npu.npu_format_cast(w_dkv_kr.contiguous(), 29)
        except Exception:
            w_dq_nz, w_uq_qr_nz, w_dkv_kr_nz = w_dq, w_uq_qr, w_dkv_kr

        B = token_x.shape[0]
        S = token_x.shape[1] if token_x.dim() == 3 else 1
        Hckv = gamma_ckv.shape[0]
        Dr   = rope_sin.shape[-1]
        Nkv, Skv, BlockSize = 1, 2048, 128
        BlockNum = max(1, math.ceil(B * Skv / BlockSize))
        dev, dt = token_x.device, token_x.dtype

        kv_cache    = torch.empty(BlockNum, BlockSize, Nkv, Hckv, dtype=dt, device=dev)
        kr_cache    = torch.empty(BlockNum, BlockSize, Nkv, Dr,   dtype=dt, device=dev)
        cache_index = torch.zeros((B, S), dtype=torch.int64, device=dev)

        derived = (w_dq_nz, w_uq_qr_nz, w_dkv_kr_nz,
                   kv_cache, kr_cache, cache_index)
        _MLA_PROLOG_CACHE[cache_key] = derived

    w_dq_nz, w_uq_qr_nz, w_dkv_kr_nz, kv_cache, kr_cache, cache_index = derived
    eps_cq  = float(attrs.get("rmsnorm_epsilon_cq", 1e-5))
    eps_ckv = float(attrs.get("rmsnorm_epsilon_ckv", 1e-5))
    try:
        out = torch_npu.npu_mla_prolog_v3(
            token_x, w_dq_nz, w_uq_qr_nz, w_uk, w_dkv_kr_nz,
            gamma_cq, gamma_ckv, rope_sin, rope_cos,
            kv_cache, kr_cache,
            cache_index=cache_index,
            rmsnorm_epsilon_cq=eps_cq,
            rmsnorm_epsilon_ckv=eps_ckv,
            cache_mode="PA_BSND",
        )
    except Exception:
        return None
    return out[0] if isinstance(out, tuple) else out


def sparse_flash_attention_ref(inputs, attrs):
    """Sparse Flash Attention via torch_npu.npu_sparse_flash_attention.

    CANN 9.0.0-beta.2 only implements the MLA template (SparseFlashAttentionMla);
    no GQA/MHA kernel exists. Kernel constraints (from tiling validation):
      - attention_mode == 2 (MLA only)
      - kv_head_num == 1
      - qk_head_dim == 512, rope_head_dim == 64
      - layout_query/kv ∈ {BSND, TND} (no BNSD)
      - sparse_block_size ∈ {1, 2, 4, ..., 128}

    For cases that fit MLA shape (Nkv=1, Dk∈{512,576}), this fn adapts:
      - Transpose BNSD → BSND when needed
      - Dk=576 → slice q[...,:512] + q[...,512:576] as nope/rope
      - Dk=512 → synthesize random rope[..., 64]
    Cases outside these constraints return None (no baseline measurable)."""
    import torch_npu
    if not hasattr(torch_npu, "npu_sparse_flash_attention"):
        return None
    q, k, v, sparse_idx = inputs[0], inputs[1], inputs[2], inputs[3]
    scale_value = float(attrs.get("scaleValue", -1.0))
    layout_orig = attrs.get("inputLayout", "BSND")
    if layout_orig not in ("BSND", "BNSD"):
        return None

    # Probe Nkv and Dk before any GPU work so unsupported cases (Nkv!=1 or Dk
    # outside {512,576}) exit immediately and don't leave Transpose/Slice in
    # the profile.
    if layout_orig == "BSND":
        _, _, Nkv_probe, _ = k.shape
        Dk_probe = q.shape[-1]
    else:  # BNSD
        _, Nkv_probe, _, _ = k.shape
        Dk_probe = q.shape[-1]
    if Nkv_probe != 1 or Dk_probe not in (512, 576):
        return None

    # Transpose BNSD → BSND if needed (kernel only accepts BSND/TND)
    if layout_orig == "BNSD":
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        sparse_idx = sparse_idx.transpose(1, 2).contiguous()

    B, Sq, Nq, Dk = q.shape
    _, Skv, Nkv, _ = k.shape
    if scale_value < 0:
        scale_value = Dk ** -0.5

    # Adapt Dk to kernel-required {nope=512, rope=64}
    if Dk == 576:
        q_nope = q[..., :512].contiguous()
        q_rope = q[..., 512:576].contiguous()
        k_nope = k[..., :512].contiguous()
        k_rope = k[..., 512:576].contiguous()
    elif Dk == 512:
        q_nope, k_nope = q, k
        # torch.empty would leak uninitialized memory into the kernel (often
        # NaN); zero-filled is correct. Cached so only the first call allocates.
        q_rope = _get_zero_rope((B, Sq, Nq, 64), q.dtype, q.device)
        k_rope = _get_zero_rope((B, Skv, Nkv, 64), k.dtype, k.device)
    else:
        return None

    # is_causal=True maps directly to sparse_mode=3 (rightDownCausal) in
    # npu_sparse_flash_attention; the kernel handles the diagonal natively
    # and no atten_mask is required (unlike npu_fusion_attention).
    sm = 3 if bool(attrs.get("is_causal", False)) else 0
    try:
        out = torch_npu.npu_sparse_flash_attention(
            q_nope, k_nope, v, sparse_idx, scale_value,
            attention_mode=2,
            query_rope=q_rope, key_rope=k_rope,
            sparse_block_size=64, sparse_mode=sm,
            layout_query="BSND", layout_kv="BSND",
        )
    except Exception:
        return None
    out_t = out[0] if isinstance(out, tuple) else out
    # Kernel only outputs BSND. Convert back to BNSD when caller specified it,
    # so the ref output layout matches the golden's (which honors inputLayout).
    if layout_orig == "BNSD":
        out_t = out_t.transpose(1, 2).contiguous()
    return out_t


REGISTRY = {
    "level4/gqa": gqa_ref,
    "level4/grouped_matmul_swiglu_quant": grouped_matmul_swiglu_quant_ref,
    "level4/gru": gru_ref,
    "level4/lstm": lstm_ref,
    "level4/mha": mha_ref,
    "level4/mla": mla_ref,
    "level4/mla_prolog": mla_prolog_ref,
    "level4/sparse_flash_attention": sparse_flash_attention_ref,
}
