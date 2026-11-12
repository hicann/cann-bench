# CompressedSparseAttentionCore 算子 API 描述

版本：2026-10-30 质量修订。level5 任务定义保持不变；本修订检查 CPU 参考、合法输入域和测试证据，不声明新的 NPU 精度、时延或硬件下界测量。

## 1. 算子简介

V4 CSA核心子图：压缩KV稀疏条目与128滑窗条目共用带sink的softmax；不含压缩器或索引搜索。

参考计算流程：合法压缩块索引 → 128滑窗与稀疏压缩KV联合读取 → QK点积 → 含sink的稳定归一化 → 共享KV加权聚合。

### 本次评测边界

包含两个KV时间域的联合读取、128滑窗、压缩块完成边界、-1 padding和每头sink。
不含压缩器、Lightning Indexer/Top-k搜索、Q/O投影、RMSNorm、RoPE/逆RoPE、FP4/FP8缓存及持久缓存更新。
输出是官方core attention位置的子图结果，不是完整CSA层。compressed_kv已经是压缩结果，不做平均池化替代官方压缩器。

## 2. 算子定义


固定压缩比 $r=4$、滑窗 $W=128$。第 $i$ 个查询位置 $t=query\_start+i$。
原始缓存起点 $a=\max(0,query\_start-127)$，长度 $L_w=S+\min(query\_start,127)$。
压缩块 $j$ 在位置 $4j+3$ 完成，合法索引满足 $4(j+1)\le t+1$；例如位置3可以读取块0。
把原始窗口 $[\max(0,t-127),t]$ 和合法压缩条目连接为 $E_t$，两种表示即使覆盖相同原始token也都保留。
\[
z_{h,j}=\mathrm{scale}\,q_{t,h}^T E_{t,j},\quad
m_h=\max(\max_j z_{h,j},s_h),\quad
y_{t,h}=\frac{\sum_j e^{z_{h,j}-m_h}E_{t,j}}
{\sum_j e^{z_{h,j}-m_h}+e^{s_h-m_h}}.
\]
两个分支必须共用同一个分母，不能分别softmax后相加；sink没有对应的非零value。


## 3. 接口规范

```text
compressed_sparse_attention_core(Tensor query, Tensor compressed_kv, Tensor window_kv, Tensor compressed_indices, Tensor attn_sink, int query_start=0, float scaleValue=-1.0) -> Tensor y
```

### 输入

| 参数 | Shape | dtype | 含义 |
|---|---|---|---|
| `query` | `[B,S,H,D]` | float16 / bfloat16 / float32 | 已完成投影和位置处理的query |
| `compressed_kv` | `[B,C,D]` | float16 / bfloat16 / float32 | 已压缩且已完成位置处理的共享K=V；C=floor((query_start+S)/4) |
| `window_kv` | `[B,Lw,D]` | float16 / bfloat16 / float32 | 时间连续的共享K=V原始缓存；起点max(0,query_start-127) |
| `compressed_indices` | `[B,S,K]` | int32 | 每行有效压缩块id唯一且已完成；-1为padding |
| `attn_sink` | `[H]` | float32 | 各头sink logit；仅进入softmax分母 |

### 输出

| 参数 | Shape | dtype | 含义 |
|---|---|---|---|
| `y` | `[B,S,H,D]` | float16 / bfloat16 / float32 | 逆RoPE/输出投影之前的核心输出；dtype同query |

### 支持范围与约束

B∈[1,4]，S∈[1,128]，H∈[1,128]，D∈[1,512]，K∈[1,1024]，query_start∈[0,32768]。
C=floor((query_start+S)/4)，可为0；K可大于当时可见块数，多余槽位填-1。
每行有效压缩id无重复、在[0,C)内且满足块完成边界。无效/重复/未来id不属于Core合法输入域。
query与两个KV同dtype、同D，每个KV缓存只有一个共享头且K=V；输入已包含需要的位置处理，不要求两种KV数值相等。
query/KV有限且在[-8,8]；sink为FP32且在[-20,20]；scaleValue有限，正值≤1，否则自动1/sqrt(D)。
本版FP16/FP32、缩小D/H/K与显式scale是benchmark的可移植扩展；官方V4核心参考是BF16、D=512。

接口只定义有限合法输入，不把预期异常混入20个计分case；本版不做NaN/Inf语义承诺。代码中的形状检查不意味着穷举检查了全部合法域。

## 4. 精度要求

Golden/bench 对 FP16/BF16/FP32 输入采用 FP32 核心计算；开发者 FP64 输入全程采用 FP64，y转回输入dtype。oracle由评测器提供FP64输入，输出保持FP64，不允许假FP64回落。
FP32 输入的 QK 点积使用分块显式 FP32 乘法与归约，限制 BLAS 累加顺序对大 logits 和输出相消的影响；FP16/BF16 路径仍采用 FP32 核心计算，FP64 oracle 保持真实 FP64。测试值域与精度阈值不变。
独立CPU自测使用：FP32 `rtol=1e-4, atol=1e-5`；FP16 `rtol=3e-3, atol=3e-4`；BF16 `rtol=2e-2, atol=2e-3`。
这些是本包验证门限，不覆盖仓库正式精度检查器。正式上线仍需用仓库checker和同精度参考验证NPU候选，不写入未经真机校准的算子级阈值。
本实现不承诺与公开低精度kernel逐位相同，也不能用该串行/分块PyTorch Golden的速度作为有竞争力的性能baseline。

## 5. 标准 Golden 代码

下方内容与同目录golden.py逐字一致，入口、oracle及bench签名保持一致。

```python

import math

import torch


COMPRESSION_RATIO = 4
WINDOW_SIZE = 128


def _validate(
    query,
    compressed_kv,
    window_kv,
    compressed_indices,
    attn_sink,
    query_start,
    scaleValue,
):
    if query.ndim != 4 or compressed_kv.ndim != 3 or window_kv.ndim != 3:
        raise ValueError("Expected query[B,S,H,D] and two KV tensors[B,N,D]")
    b, s, h, d = query.shape
    if min(b, s, h, d) < 1 or query_start < 0:
        raise ValueError("Nonempty query and nonnegative query_start required")
    start = max(0, query_start - WINDOW_SIZE + 1)
    if window_kv.shape != (b, query_start + s - start, d):
        raise ValueError("window_kv must cover [max(0,query_start-127),query_start+S)")
    if compressed_kv.shape != (b, (query_start + s) // COMPRESSION_RATIO, d):
        raise ValueError("compressed_kv must contain all completed prefix blocks")
    if compressed_indices.ndim != 3 or compressed_indices.shape[:2] != (b, s):
        raise ValueError("compressed_indices must have shape [B,S,K]")
    if compressed_indices.shape[-1] < 1 or attn_sink.shape != (h,):
        raise ValueError("K must be positive and attn_sink must have shape [H]")
    if compressed_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("Indices must be integer tensors")
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("query must use a supported floating-point dtype")
    if compressed_kv.dtype != query.dtype or window_kv.dtype != query.dtype:
        # The internal FP32 working query is passed only to _compute, not here.
        raise TypeError("Query and both KV inputs must share dtype")
    if attn_sink.dtype not in (torch.float32, torch.float64):
        raise TypeError("attn_sink must be FP32, or FP64 for the oracle")
    if isinstance(query_start, bool) or not isinstance(query_start, int):
        raise TypeError("query_start must be an integer")
    if (
        isinstance(scaleValue, bool)
        or not isinstance(scaleValue, (float, int))
        or not math.isfinite(scaleValue)
    ):
        raise ValueError("scaleValue must be finite")
    if any(
        x.device != query.device
        for x in (compressed_kv, window_kv, compressed_indices, attn_sink)
    ):
        raise ValueError("All inputs must be on the same device")


def _fp32_query_key(query, entries):
    """Explicit FP32 products and reductions; bound the temporary by two tiles.

    FP32 public outputs face the strictest checker. Large finite logits make
    BLAS-dependent dot accumulation visible after softmax and cancellation.
    The reduction stays FP32 and does not change any input values or threshold.
    """
    b, s, h, _ = query.shape
    count = entries.shape[2]
    scores = query.new_empty((b, s, h, count))
    for head in range(0, h, 8):
        for entry in range(0, count, 64):
            products = (
                query[:, :, head : head + 8, None, :]
                * entries[:, :, None, entry : entry + 64, :]
            )
            scores[:, :, head : head + 8, entry : entry + 64] = products.sum(dim=-1)
    return scores


def _compute(
    query,
    compressed_kv,
    window_kv,
    compressed_indices,
    attn_sink,
    query_start,
    scaleValue,
    explicit_fp32_dot=False,
):
    """Bound temporary memory by query tiling; arithmetic follows query dtype."""
    b, s, h, d = query.shape
    c = compressed_kv.shape[1]
    scale = d**-0.5 if scaleValue <= 0 else scaleValue
    start = max(0, query_start - WINDOW_SIZE + 1)
    batch = torch.arange(b, device=query.device)[:, None, None]
    offsets = torch.arange(WINDOW_SIZE, device=query.device)
    outputs = []
    for lo in range(0, s, 16):
        hi = min(lo + 16, s)
        positions = query_start + torch.arange(lo, hi, device=query.device)
        # A fixed-width window is padded on the left near the start of a sequence.
        raw_positions = positions[:, None] - WINDOW_SIZE + 1 + offsets[None, :]
        raw_valid = raw_positions >= 0
        raw_ids = (raw_positions - start).clamp(min=0)
        raw = window_kv[batch, raw_ids[None, :, :], :].to(query.dtype)
        ids = compressed_indices[:, lo:hi, :].long()
        valid = (ids >= 0) & (ids < c)
        valid = valid & ((ids + 1) * COMPRESSION_RATIO <= positions[None, :, None] + 1)
        if c:
            comp = compressed_kv[batch, ids.clamp(min=0, max=c - 1), :].to(query.dtype)
        else:
            comp = torch.zeros((*ids.shape, d), dtype=query.dtype, device=query.device)
        entries = torch.cat((raw, comp), dim=2)
        mask = torch.cat((raw_valid[None, :, :].expand(b, -1, -1), valid), dim=2)
        scores = (
            _fp32_query_key(query[:, lo:hi], entries)
            if explicit_fp32_dot
            else torch.einsum("bshd,bsed->bshe", query[:, lo:hi], entries)
        ) * scale
        scores = scores.masked_fill(~mask[:, :, None, :], -torch.inf)
        sink = attn_sink.to(query.dtype)[None, None, :, None]
        maximum = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)
        weights = torch.exp(scores - maximum)
        denominator = weights.sum(dim=-1, keepdim=True) + torch.exp(sink - maximum)
        outputs.append(torch.einsum("bshe,bsed->bshd", weights / denominator, entries))
    return torch.cat(outputs, dim=1).contiguous()


def compressed_sparse_attention_core(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    query_start: int = 0,
    scaleValue: float = -1.0,
) -> torch.Tensor:
    """Return [B,S,H,D] in query dtype; use FP32, or genuine FP64, arithmetic.

    window_kv has absolute origin max(0,query_start-127); compressed block j has
    completion position 4*j+3. Valid indices are unique per row; -1 is padding.
    Inputs have already received any model projection/normalization/RoPE.
    """
    _validate(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )
    accumulation_dtype = (
        torch.float64 if query.dtype == torch.float64 else torch.float32
    )
    return _compute(
        query.to(accumulation_dtype),
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
        explicit_fp32_dot=query.dtype == torch.float32,
    ).to(query.dtype)


def compressed_sparse_attention_core_oracle(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    query_start: int = 0,
    scaleValue: float = -1.0,
) -> torch.Tensor:
    """Dtype-preserving hook: evaluator FP64 inputs yield a genuine FP64 result."""
    _validate(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )
    return _compute(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )


def compressed_sparse_attention_core_bench(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    query_start: int = 0,
    scaleValue: float = -1.0,
) -> torch.Tensor:
    """Same-precision correctness reference; NOT a measured performance baseline."""
    return compressed_sparse_attention_core(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )


def get_input(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    window_kv: torch.Tensor,
    compressed_indices: torch.Tensor,
    attn_sink: torch.Tensor,
    query_start: int = 0,
    scaleValue: float = -1.0,
    **kwargs,
):
    """Make causal, unique index rows from seeded generator inputs, for ALL paths.

    An all--1 source row stays all padding. Other rows include the latest completed
    block and block zero when space permits, followed by input-dependent remote
    blocks. Short prefixes pad with -1. No randomness is introduced here and the
    seeded source indices affect selection. This is untimed structural input setup,
    not index search inside the operator. No external files are consulted.
    """
    _validate(
        query,
        compressed_kv,
        window_kv,
        compressed_indices,
        attn_sink,
        query_start,
        scaleValue,
    )
    b, s, k = compressed_indices.shape
    c = compressed_kv.shape[1]
    source = compressed_indices.detach().cpu().tolist()
    result = torch.full((b, s, k), -1, dtype=compressed_indices.dtype)
    for bi in range(b):
        for si in range(s):
            limit = min(c, (query_start + si + 1) // COMPRESSION_RATIO)
            row = source[bi][si]
            if not limit or all(x == -1 for x in row):
                continue
            count = min(k, limit)
            choices = [limit - 1, 0] + [abs(x) % limit for x in row if x != -1]
            choices += [(i * limit) // count for i in range(count)]
            seen, selected = set(), []
            for value in choices:
                if value not in seen:
                    seen.add(value)
                    selected.append(value)
                if len(selected) == count:
                    break
            result[bi, si, :count] = torch.tensor(selected, dtype=result.dtype)
    return [
        query,
        compressed_kv,
        window_kv,
        result.to(compressed_indices.device),
        attn_sink,
    ]
```

## 6. 测试用例与验收

恰好20个公开Core候选用例；Smoke为ID 1、9、15、18的子集，不能重复进入计分分母。另有80个内部用例（ID 21–100），公开与内部输入域均在本次修订中检查；边界用例单独归类，不作为代表性工作量的证据。
用途配额为4热点/4形态/3尾块/3精度/3属性/2边界/1性能。热点是模型核心维度或缩小的代表性情景，并非平台实测热点排名。

| ID | 主要目的 | 输入尺寸摘要 | dtype | 设计要点 |
|---:|---|---|---|---|
| 1 | hotspot | [1, 16, 64, 512]; t0=0; K=4 | bfloat16 | V4-Flash-head-dim-short-prefill |
| 2 | hotspot | [1, 1, 128, 512]; t0=4095; K=1024 | bfloat16 | V4-Pro-head-dim-topk-decode |
| 3 | hotspot | [1, 16, 64, 512]; t0=512; K=128 | bfloat16 | V4-head-dim-history-chunk |
| 4 | hotspot | [2, 1, 64, 512]; t0=8192; K=512 | bfloat16 | V4-Flash-batched-decode |
| 5 | shape | [1, 1, 1, 8]; t0=0; K=1 | float32 | tiny-no-compressed-block |
| 6 | shape | [2, 64, 8, 64]; t0=0; K=16 | float16 | small-prefill-batch |
| 7 | shape | [1, 1, 16, 128]; t0=32768; K=128 | bfloat16 | 32K-context-sparse-decode |
| 8 | shape | [1, 64, 16, 128]; t0=512; K=128 | float16 | multi-query-shared-kv |
| 9 | tail | [1, 3, 3, 63]; t0=127; K=31 | float16 | tail-window-eviction-D63-K31 |
| 10 | tail | [1, 7, 5, 65]; t0=130; K=33 | bfloat16 | tail-query-D65-K33 |
| 11 | tail | [2, 5, 7, 31]; t0=3; K=3 | float32 | tail-block-completion-D31 |
| 12 | dtype | [1, 17, 4, 64]; t0=128; K=32 | float32 | fp32-reference-path |
| 13 | dtype | [1, 8, 4, 64]; t0=512; K=64 | float16 | large-logits-stable-softmax |
| 14 | dtype | [1, 8, 4, 64]; t0=128; K=32 | bfloat16 | small-values-sink-dominance |
| 15 | attr | [1, 9, 2, 32]; t0=127; K=32 | float16 | explicit-positive-scale |
| 16 | attr | [1, 8, 2, 16]; t0=0; K=4 | float32 | zero-origin-padding |
| 17 | attr | [1, 1, 2, 32]; t0=3; K=1 | bfloat16 | first-completed-block-visible |
| 18 | boundary | [1, 4, 3, 16]; t0=127; K=32 | float32 | zero-query-constant-kv-analytic |
| 19 | boundary | [1, 8, 4, 64]; t0=512; K=64 | float16 | all-compressed-padding-window-only |
| 20 | perf | [1, 128, 32, 128]; t0=2048; K=256 | bfloat16 | bounded-union-attention-throughput |

### 输入生成与公平性

CSA使用golden.py的get_input规整结构索引，对Golden、候选、oracle使用同一组规整后输入。
全-1源行保持padding；否则按可见块数生成唯一索引，优先最新已完成块和块0，随后使用随机输入驱动的远距块并补足，早期不足K时填-1。
构造规则不读取case_id或外部文件、不生成新随机数；原seed变化可改变远距选择。输入准备不计入kernel耗时。
测试固定seed保证可复现，并以不同seed复测非结构数值。CSV与YAML严格对齐，无baseline_perf_us或t_hw_us字段。

### 成本边界

20例最大输入输出合计约 4.504 MiB，全部小于1GB。本版不为达到大张量占比强行扩大注意力/递推任务，S/M/L按本算子的工作量相对分档；这是有界Core设计，不代表执行成本通过评审。
百万token、超大batch、完整模型层以及压缩器/indexer融合留待Extended方案；NPU时间和峰值内存仍需人工复核。

### 当前发布条件

本次交付算子定义、参考实现和20个公开用例。NPU精度、性能baseline和硬件下界尚未验证；本PR不包含Ascend C实现，不改变现有主线计分集合。
已有公开实现不因更换名称或shape而消失。本题目范围及用例设计不构成不可复用公开代码的承诺。

### 参考文档

- [DeepSeek V4，v1，§2.3](https://arxiv.org/html/2606.19348v1#S2.SS3)
- [官方模型代码，固定提交](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/5607980f3a4b8ea0371b9f11e1848ac41f14979e/inference/model.py)
- [官方core kernel](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/5607980f3a4b8ea0371b9f11e1848ac41f14979e/inference/kernel.py)
