# FlashAttentionScoreGradEnhance 算子 API 描述

## 1. 算子简介

FlashAttentionScoreGradEnhance 是 `FlashAttentionScoreEnhance` 的反向算子，根据正向输入、上游梯度 `dy` 以及正向 softmax 中间统计量，计算 query、key 和 value 的梯度。可选 rope 输入参与正向等价逻辑重建，但当前 Python 接口不单独返回 rope 梯度。

**主要应用场景**：
- 大语言模型训练中的 FlashAttention 反向
- MLA nope/rope 分离结构的梯度计算
- 使用正向 `softmax_max` / `softmax_sum` 中间结果的高效 attention 反向

**算子特征**：
- 难度等级：L4（FusedComposite）
- 默认布局为 `input_layout="TND"`，同时保留 `BSND` 布局语义
- 支持 float16 / bfloat16 输入
- 输出 `dq`、`dk`、`dv`

## 2. 算子定义

### 数学公式

正向 attention 为：

$$
O = Softmax(Mask(scale \cdot QK^T + pse))V
$$

反向给定上游梯度 `dy = dL/dO`，计算：

$$
dQ = \frac{\partial L}{\partial Q},\quad dK = \frac{\partial L}{\partial K},\quad dV = \frac{\partial L}{\partial V}
$$

当传入 rope 时，实际参与 QK 的输入为：

$$
Q'=[Q,Q_{rope}],\quad K'=[K,K_{rope}]
$$

### 步骤说明

1. 按 `input_layout` 和 actual sequence 字段还原 batch 内有效序列。
2. 若传入 `query_rope` 和 `key_rope`，拼接后参与 QK 梯度计算。
3. 复用正向一致的 `pse`、`atten_mask`、`sparse_mode`、`pre_tokens`、`next_tokens` 语义重建 attention score。
4. 根据 `dy` 对 softmax attention 反向传播，得到 `dq`、`dk`、`dv`。
5. 返回当前 Python 接口暴露的 `dq`、`dk`、`dv` 三个梯度。

## 3. 接口规范

### 算子原型

```python
cann_bench.flash_attention_score_grad_enhance(
    Tensor query,
    Tensor key,
    Tensor value,
    Tensor dy,
    int head_num,
    str input_layout="TND",
    Tensor? pse=None,
    Tensor? padding_mask=None,
    Tensor? atten_mask=None,
    Tensor? softmax_max=None,
    Tensor? softmax_sum=None,
    Tensor? softmax_in=None,
    Tensor? attention_in=None,
    Tensor? sink_tensor=None,
    Tensor? query_rope=None,
    Tensor? key_rope=None,
    float scale_value=1.0,
    float keep_prob=1.0,
    int pre_tokens=2147483647,
    int next_tokens=2147483647,
    int inner_precise=0,
    prefix=None,
    Tensor|list|None actual_seq_qlen=None,
    Tensor|list|None actual_seq_kvlen=None,
    int sparse_mode=0,
    int sink_num=0,
    int pse_type=1,
    str softmaxInLayout="",
    Tensor|list|None q_start_idx=None,
    Tensor|list|None kv_start_idx=None,
) -> (Tensor dq, Tensor dk, Tensor dv)
```

> **计分接口以 `proto.yaml` 为准，共 8 个输入**：`query, key, value, dy, pse, atten_mask, query_rope, key_rope`（golden 亦只消费这 8 个输入与属性；`softmax_max` / `softmax_sum` 由 golden 按等价正向逻辑重建，不作为输入传递）。本节签名中的其余参数为 ascend 兼容接口说明，评测不传递。

### 输入参数说明

| 参数 | 类型 | 必需或默认值 | dtype | shape | 描述 |
|------|------|--------------|-------|-------|------|
| query | Tensor | 必选 | float16 / bfloat16 | `TND: [Tq,N,D]`；`BSND: [B,Sq,N,D]` | 正向 Query 输入 |
| key | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,D]`；`BSND: [B,Skv,Nkv,D]` | 正向 Key 输入 |
| value | Tensor | 必选 | float16 / bfloat16 | `TND: [Tkv,Nkv,Dv]`；`BSND: [B,Skv,Nkv,Dv]` | 正向 Value 输入 |
| dy | Tensor | 必选 | float16 / bfloat16 | 与正向 `attention_out` 一致 | 上游梯度 |
| head_num | int | 必选 | - | 标量 | query head 数 |
| input_layout | str | 默认 `TND` | - | - | 输入布局，支持 `TND` 和 `BSND` |
| pse | Tensor? | 默认 `None` | float16 / bfloat16 / float32 | 可广播到 `[B,N,Sq,Skv]` | 正向使用的 position bias |
| padding_mask | Tensor? | 默认 `None` | - | 预留 | 接口保留字段 |
| atten_mask | Tensor? | 默认 `None` | bool / uint8 | 可广播到 `[B,N,Sq,Skv]` 或 `[Sq,Skv]` | 正向使用的 attention mask |
| softmax_max | Tensor? | 默认 `None` | float32 | 与正向统计输出一致 | 正向 `softmax_max` 中间结果，标准 Golden 按等价正向逻辑重建 |
| softmax_sum | Tensor? | 默认 `None` | float32 | 与正向统计输出一致 | 正向 `softmax_sum` 中间结果，标准 Golden 按等价正向逻辑重建 |
| softmax_in | Tensor? | 默认 `None` | - | 预留 | softmax 输入保留字段 |
| attention_in | Tensor? | 默认 `None` | float16 / bfloat16 | 与正向 `attention_out` 一致 | 正向 attention 输出保留字段 |
| sink_tensor | Tensor? | 默认 `None` | - | 预留 | sink 相关保留字段，配合 `sink_num` 保持接口兼容 |
| query_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 query 前缀维一致 | 正向 query rope 输入 |
| key_rope | Tensor? | 默认 `None` | float16 / bfloat16 | 与 key 前缀维一致 | 正向 key rope 输入 |
| scale_value | float | 默认 `1.0` | - | 标量 | 正向 QK 缩放系数 |
| keep_prob | float | 默认 `1.0` | - | 标量 | 正向 dropout 保留概率 |
| pre_tokens | int | 默认 `2147483647` | - | 标量 | 正向滑窗左侧可见 token 数 |
| next_tokens | int | 默认 `2147483647` | - | 标量 | 正向滑窗右侧可见 token 数 |
| inner_precise | int | 默认 `0` | - | 标量 | 内部精度控制字段 |
| prefix | Any? | 默认 `None` | - | - | 接口保留字段 |
| actual_seq_qlen | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | query 实际长度；TND 时可按前缀和解释 |
| actual_seq_kvlen | Tensor/list? | 默认 `None` | int32 / int64 | `[B]` | key/value 实际长度；TND 时可按前缀和解释 |
| sparse_mode | int | 默认 `0` | - | 标量 | 正向 sparse/mask 模式 |
| sink_num | int | 默认 `0` | - | 标量 | sink token 组数；TND 布局下生效（语义见规则与约束），BSND 布局忽略 |
| pse_type | int | 默认 `1` | - | 标量 | pse 类型标识 |
| softmaxInLayout | str | 默认 `""` | - | - | softmax 中间输入布局控制字段 |
| q_start_idx | Tensor/list? | 默认 `None` | int32 / int64 | - | query 起始索引保留字段 |
| kv_start_idx | Tensor/list? | 默认 `None` | int32 / int64 | - | key/value 起始索引保留字段 |

### 输出

| 参数 | dtype | shape | 描述 |
|------|-------|-------|------|
| dq | float16 / bfloat16 | 与 query 一致 | Query 梯度 |
| dk | float16 / bfloat16 | 与 key 一致 | Key 梯度 |
| dv | float16 / bfloat16 | 与 value 一致 | Value 梯度 |

### 规则与约束

- `softmax_max` 和 `softmax_sum` 应来自同一次 `FlashAttentionScoreEnhance` 正向输出。
- **sink 语义（TND 专用，golden 已实现）**：`sink_num > 0` 时，K/V（及 `key_rope`）的**前 `sink_num × 64` 行**为 sink 行，自常规 K/V 序列中划出；每个 query 行除其带状可见窗口外，对全部 sink 行额外可见，sink 行正常参与 softmax 归一化与 dk/dv 梯度回传。BSND 布局忽略 `sink_num`。`sink_tensor` 不参与计算，仅为接口兼容字段。
- **计分 dtype**：`query/key/value/dy` 为 float16 / bfloat16；proto 中的 float32 声明仅为接口兼容，计分 case 不使用。
- `softmax_in` 为接口保留参数，标准 Golden 不依赖该输入。
- 反向使用的 `pse`、`atten_mask`、`sparse_mode`、`pre_tokens`、`next_tokens` 应与正向保持一致。
- `query_rope` 与 `key_rope` 需要成对传入；当前 Python 接口仅返回 `dq`、`dk`、`dv`。
- 当 key/value head 数小于 `head_num` 时，按 GQA/MQA 语义 repeat KV head。

### 完整计分语义

- 计分用例的八个输入依次为 `query, key, value, dy, pse, atten_mask, query_rope, key_rope`。原型中其余 Tensor 参数是兼容字段或由预处理生成的正向统计量；它们不改变这里定义的数学结果。
- `TND` 的 Q 和普通 KV 分别按 `actual_seq_qlen` / `actual_seq_kvlen` 切段。列表既可为每段长度，也可为总量匹配的非递减前缀和。TND 的 Q 长度和必须为 `Tq`，普通 KV 长度和必须为 `Tkv - 64*sink_num`。Q 和 KV 的段数必须相等，每段长度均大于零。
- `BSND` 为 `[B, S, N, D]`；未提供 actual sequence 列表时，各 batch 使用完整 `S`。只支持 `TND` 和 `BSND`，`keep_prob=1.0`；不测试 dropout。
- `Hq` 必须等于 `head_num`，且为 `Hkv` 的整数倍。第 `h` 个 Q head 使用第 `floor(h/(Hq/Hkv))` 个 KV head，其 KV 梯度对同组 Q heads 求和。V 的最后一维可以独立于 Q/K 维度。
- 对每段，令 query 行为 `i`、普通 key 列为 `j`、该段长度为 `Sq, Sk`。`sparse_mode=2` 可见 `j<=i`（左上对齐因果），`sparse_mode=3` 可见 `j<=Sk-Sq+i`（右下对齐因果）；`sparse_mode=0/4` 可见 `Sk-Sq+i-pre_tokens <= j <= Sk-Sq+i+next_tokens`。边界裁剪到 `[0,Sk)`。全注意力使用 mode 0 和足够大的 pre/next。
- mode 0 的非空 `atten_mask` 以非零表示屏蔽；mode 2/3/4 保留候选侧的压缩 mask，不将其当作每段 `[Sq,Sk]` 的逻辑表。未传 mask 时，预处理为这些稀疏模式生成固定 `[2048,2048]` uint8 因果压缩模板；CPU 分块参考按各段长度及 sparse/pre/next 属性重建逻辑可见性，序列超过 2048 时也不扩展候选输入。二维逻辑 mask/PSE 使用其末尾 `[Sq,Sk]` 区域，四维逻辑 mask/PSE 使用对应 batch 与尾部区域，可对 head 广播。sink 列的 PSE 为零、不会被普通序列 mask 屏蔽。
- 被屏蔽的 score 替换为 `-40000.0` 后做稳定 softmax；屏蔽位置对 Q/K 的导数为零。计分用例保留单 token 边界，但不会把整例设计成全部 Q/K 梯度为零的单 token/identity attention。
- Rope 仅拼接参与 score，不改变主 Q/K 的输入和输出 shape。返回的 `dq/dk/dv` 与原始 `query/key/value` shape 一致，sink K/V 梯度累加来自所有段的贡献。
- CPU 标准答案采用分块解析反向，限制临时 score/probability 存储；保留 FP64 真值，以独立检查 FP32 累加与相消误差。同精度参考从原始 FP16/BF16 量化输入提升到 FP32 计算，比较仍使用原输出 dtype 对 golden 截断与原 dtype 阈值。

## 4. 精度要求

采用生态算子精度标准进行验证。

**误差指标**：

1. 平均相对误差（MERE）

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 |
|----------|---------|----------|---------|
| Threshold | 2^-10 | 2^-7 | 2^-13 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。

## 5. 标准 Golden 代码

标准 Golden 参考实现位于同目录 `golden.py` 的 `flash_attention_score_grad_enhance` 函数，使用分块 softmax 及解析反向公式重建 attention，并返回原始 Q/K/V 的梯度。独立验收另外使用 Torch autograd 重建正向进行复核。

## 6. 额外信息

### 算子调用示例

```python

from typing import Optional

import torch

F = torch.nn.functional
_BENCH_PRECISION = False


def _default_lengths(
    total: Optional[int], batch: Optional[int], default_mode: Optional[str]
):
    if total is None:
        return []
    batch = max(int(batch or 1), 1)
    total = max(int(total), 0)
    if default_mode is None or default_mode == "split_total":
        base = total // batch
        out = [base] * batch
        for idx in range(total - base * batch):
            out[idx] += 1
        return out
    if default_mode == "per_batch_full":
        return [total] * batch
    raise ValueError(f"unsupported default_mode: {default_mode}")


def as_int_list(
    value,
    total: Optional[int] = None,
    batch: Optional[int] = None,
    is_tnd: bool = True,
    *,
    default_mode: Optional[str] = None,
    allow_scalar: bool = True,
    empty_uses_default: bool = True,
):
    if value is None:
        return (
            _default_lengths(total, batch, default_mode) if empty_uses_default else []
        )
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif allow_scalar and isinstance(value, (int, float)):
        value = [value]
    out = [int(x) for x in value]
    if not out:
        return (
            _default_lengths(total, batch, default_mode) if empty_uses_default else []
        )
    if not is_tnd or total is None:
        return out
    if all(a <= b for a, b in zip(out, out[1:])) and out[-1] == int(total):
        prev = 0
        lengths = []
        for item in out:
            lengths.append(max(int(item) - prev, 0))
            prev = int(item)
        return lengths
    if sum(out) == int(total):
        return out
    raise ValueError(
        "actual_seq list is inconsistent with the packed total: expected a "
        f"non-decreasing prefix ending at {int(total)} or lengths summing to "
        f"{int(total)}, got {out} (sum={sum(out)})"
    )


def slice_optional(x: Optional[torch.Tensor], b: int, q_len: int, kv_len: int, device):
    if x is None or not isinstance(x, torch.Tensor) or x.numel() <= 1:
        return None
    x = x.detach().to(device=device)
    if x.dim() == 2:
        return x[-q_len:, -kv_len:]
    if x.dim() >= 4:
        b_idx = min(b, x.shape[0] - 1)
        return x[b_idx : b_idx + 1, :, -q_len:, -kv_len:]
    return x


def _attention_compute(
    query, key, value, dy=None, *, calc_dtype=torch.float32, calc_device="cpu", **kwargs
):
    """Bounded-memory reference using the softmax-backward equations.

    Query blocks and head blocks bound temporary storage. No dense attention
    graph is retained across segments. FP64 oracle and FP32 bench use the same
    equations; tests cross-check an independently differentiated forward.
    """
    layout = kwargs.get("input_layout", "TND")
    if layout not in ("TND", "BSND"):
        raise ValueError("only TND and BSND are supported")
    if float(kwargs.get("keep_prob", 1.0)) != 1.0:
        raise ValueError("only keep_prob=1.0 is supported")
    hq, hk = query.shape[-2], key.shape[-2]
    if int(kwargs.get("head_num", hq)) != hq or hq % hk:
        raise ValueError("head_num must match query heads and be divisible by KV heads")
    if key.shape[:-1] != value.shape[:-1]:
        raise ValueError("key/value sequence and head dimensions must match")
    q = query.detach().to(device=calc_device, dtype=calc_dtype)
    k = key.detach().to(device=calc_device, dtype=calc_dtype)
    v = value.detach().to(device=calc_device, dtype=calc_dtype)
    qr, kr = kwargs.get("query_rope"), kwargs.get("key_rope")
    paired_rope = (
        isinstance(qr, torch.Tensor)
        and qr.numel()
        and isinstance(kr, torch.Tensor)
        and kr.numel()
    )
    if bool(isinstance(qr, torch.Tensor) and qr.numel()) != bool(
        isinstance(kr, torch.Tensor) and kr.numel()
    ):
        raise ValueError("query_rope and key_rope must be paired")
    if paired_rope:
        q = torch.cat((q, qr.detach().to(device=calc_device, dtype=calc_dtype)), -1)
        k = torch.cat((k, kr.detach().to(device=calc_device, dtype=calc_dtype)), -1)
    sink = int(kwargs.get("sink_num", 0) or 0) * 64 if layout == "TND" else 0
    if sink < 0 or (sink and sink >= k.shape[0]):
        raise ValueError("sink rows must leave a nonempty ordinary KV sequence")
    aql = kwargs.get("actual_seq_qlen")
    batch = (len(aql) if aql is not None else 1) if layout == "TND" else query.shape[0]
    qt = query.shape[0] if layout == "TND" else query.shape[1]
    kt = key.shape[0] - sink if layout == "TND" else key.shape[1]
    qlens = as_int_list(
        kwargs.get("actual_seq_qlen"),
        qt,
        batch or 1,
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    klens = as_int_list(
        kwargs.get("actual_seq_kvlen"),
        kt,
        len(qlens),
        layout == "TND",
        default_mode="split_total" if layout == "TND" else "per_batch_full",
    )
    if len(qlens) != len(klens) or any(x <= 0 for x in qlens + klens):
        raise ValueError("Q/KV must have matching positive sequence lengths")
    if layout == "BSND" and (any(x > qt for x in qlens) or any(x > kt for x in klens)):
        raise ValueError("actual lengths exceed BSND padded extents")
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    output = q.new_zeros((*query.shape[:-1], value.shape[-1]))
    if dy is not None:
        dy = dy.detach().to(device=calc_device, dtype=calc_dtype)
    scale = float(kwargs.get("scale", kwargs.get("scale_value", 1.0)))
    mode = int(kwargs.get("sparse_mode", 0))
    if mode not in (0, 2, 3, 4):
        raise ValueError("sparse_mode must be 0, 2, 3 or 4")
    pre, nxt = (
        int(kwargs.get("pre_tokens", 2**31 - 1)),
        int(kwargs.get("next_tokens", 2**31 - 1)),
    )
    pse, mask = kwargs.get("pse"), kwargs.get("atten_mask")
    qoff, koff = 0, sink
    maxparts, sumparts = [], []
    for b, (ql, kl) in enumerate(zip(qlens, klens)):
        qslice = slice(qoff, qoff + ql) if layout == "TND" else (b, slice(0, ql))
        kslice = slice(koff, koff + kl) if layout == "TND" else (b, slice(0, kl))
        qb, kb, vb = (
            q[qslice].transpose(0, 1),
            k[kslice].transpose(0, 1),
            v[kslice].transpose(0, 1),
        )
        if sink:
            kb = torch.cat((k[:sink].transpose(0, 1), kb), 1)
            vb = torch.cat((v[:sink].transpose(0, 1), vb), 1)
        dqb, dkb, dvb = torch.zeros_like(qb), torch.zeros_like(kb), torch.zeros_like(vb)
        ob = q.new_zeros((hq, ql, value.shape[-1]))
        rowmax = q.new_empty((hq, ql, 1))
        rowsum = torch.empty_like(rowmax)
        dyb = dy[qslice].transpose(0, 1) if dy is not None else None
        bias = slice_optional(pse, b, ql, kl, calc_device)
        # Sparse modes carry a compressed candidate mask. Logical visibility
        # is reconstructed per sequence in the bounded reference.
        explicit = slice_optional(mask, b, ql, kl, calc_device) if mode == 0 else None
        for h0 in range(0, hq, 4):
            h1 = min(h0 + 4, hq)
            headmap = (torch.arange(h0, h1) // (hq // hk)).to(device=calc_device)
            kh, vh = kb[headmap], vb[headmap]
            for q0 in range(0, ql, 256):
                q1 = min(q0 + 256, ql)
                qc = qb[h0:h1, q0:q1]
                score = (qc @ kh.transpose(-1, -2)) * scale
                if bias is not None:
                    bs = bias.reshape(-1, ql, kl)
                    bs = bs if bs.shape[0] == 1 else bs[h0:h1]
                    bs = bs[:, q0:q1].to(calc_dtype)
                    score = score + (F.pad(bs, (sink, 0)) if sink else bs)
                rows = torch.arange(q0, q1, device=calc_device).view(-1, 1)
                cols = torch.arange(kl, device=calc_device).view(1, -1)
                if explicit is not None:
                    blocked = explicit.reshape(-1, ql, kl)
                    blocked = blocked if blocked.shape[0] == 1 else blocked[h0:h1]
                    blocked = blocked[:, q0:q1].bool()
                elif mode == 2:
                    blocked = cols > rows
                elif mode == 3:
                    blocked = cols > kl - ql + rows
                else:
                    center = kl - ql + rows
                    blocked = (cols < center - pre) | (cols > center + nxt)
                if sink:
                    blocked = F.pad(blocked, (sink, 0), value=False)
                score = score.masked_fill(blocked, -40000.0)
                rm = score.max(-1, keepdim=True).values
                ex = torch.exp(score - rm)
                rs = ex.sum(-1, keepdim=True)
                prob = ex / rs
                ob[h0:h1, q0:q1] = prob @ vh
                rowmax[h0:h1, q0:q1], rowsum[h0:h1, q0:q1] = rm, rs
                if dyb is not None:
                    grad = dyb[h0:h1, q0:q1]
                    dp = grad @ vh.transpose(-1, -2)
                    ds = prob * (dp - (dp * prob).sum(-1, keepdim=True))
                    ds.masked_fill_(blocked, 0)
                    dqb[h0:h1, q0:q1] = (ds @ kh) * scale
                    dkb.index_add_(0, headmap, (ds.transpose(-1, -2) @ qc) * scale)
                    dvb.index_add_(0, headmap, prob.transpose(-1, -2) @ grad)
        output[qslice] = ob.transpose(0, 1)
        if dyb is not None:
            dq[qslice] = dqb.transpose(0, 1)
            if sink:
                dk[:sink] += dkb[:, :sink].transpose(0, 1)
                dv[:sink] += dvb[:, :sink].transpose(0, 1)
            dk[kslice] += dkb[:, sink:].transpose(0, 1)
            dv[kslice] += dvb[:, sink:].transpose(0, 1)
        # Packed auxiliary stats address [valid token, head, lane]. Each
        # sequence/batch contributes its valid query tokens in input order.
        # Flattening [head, token, lane] before the final reshape permutes rows.
        stats_max = rowmax.transpose(0, 1)
        stats_sum = rowsum.transpose(0, 1)
        maxparts.append(stats_max.expand(-1, -1, 8).contiguous().reshape(-1))
        sumparts.append(stats_sum.expand(-1, -1, 8).contiguous().reshape(-1))
        qoff, koff = qoff + ql, koff + kl
    softmax_max = torch.cat(maxparts).reshape(-1, hq, 8).float()
    softmax_sum = torch.cat(sumparts).reshape(-1, hq, 8).float()
    if dy is None:
        return output.to(query.dtype), softmax_max, softmax_sum
    # Rope channels affect score, but only gradients of the original Q/K are exposed.
    return (
        dq[..., : query.shape[-1]],
        dk[..., : key.shape[-1]],
        dv,
        dq[..., query.shape[-1] :],
        dk[..., key.shape[-1] :],
    )


def _attention_forward_inputs(
    query,
    key,
    value,
    pse=None,
    atten_mask=None,
    query_rope=None,
    key_rope=None,
    **kwargs,
):
    return _attention_compute(
        query,
        key,
        value,
        calc_dtype=torch.float32,
        pse=pse,
        atten_mask=atten_mask,
        query_rope=query_rope,
        key_rope=key_rope,
        **kwargs,
    )


def attention_grad(query, key, value, dy, **kwargs):
    with torch.enable_grad():
        return _attention_grad_impl(query, key, value, dy, **kwargs)


def _attention_grad_impl(query, key, value, dy, **kwargs):
    # The input hook can leave the original dtype in the CPU oracle call.
    # Select the oracle precision here so that it stays distinct from the bench.
    native = query.device.type == "npu" and query.dtype in (torch.float16, torch.bfloat16)
    calc_device = query.device if native else torch.device("cpu")
    calc_dtype = query.dtype if native else (
        torch.float32 if _BENCH_PRECISION else torch.float64
    )
    return _attention_compute(
        query, key, value, dy, calc_dtype=calc_dtype, calc_device=calc_device, **kwargs
    )


def flash_attention_score_grad_enhance(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    head_num: int,
    input_layout: str = "TND",
    pse: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    softmax_max: Optional[torch.Tensor] = None,
    softmax_sum: Optional[torch.Tensor] = None,
    softmax_in: Optional[torch.Tensor] = None,
    attention_in: Optional[torch.Tensor] = None,
    sink_tensor: Optional[torch.Tensor] = None,
    query_rope: Optional[torch.Tensor] = None,
    key_rope: Optional[torch.Tensor] = None,
    scale_value: float = 1.0,
    scale: float = None,
    keep_prob: float = 1.0,
    pre_tokens: int = (1 << 31) - 1,
    next_tokens: int = (1 << 31) - 1,
    inner_precise: int = 0,
    prefix=None,
    actual_seq_qlen=None,
    actual_seq_kvlen=None,
    sparse_mode: int = 0,
    sink_num: int = 0,
    pse_type: int = 1,
    softmaxInLayout: str = "",
    q_start_idx=None,
    kv_start_idx=None,
    **kwargs,
):
    del (
        padding_mask,
        softmax_max,
        softmax_sum,
        softmax_in,
        attention_in,
        sink_tensor,
        inner_precise,
    )
    del prefix, pse_type, softmaxInLayout, q_start_idx, kv_start_idx, kwargs
    if scale is not None:
        scale_value = scale
    grads = attention_grad(
        query,
        key,
        value,
        dy,
        head_num=head_num,
        input_layout=input_layout,
        pse=pse,
        atten_mask=atten_mask,
        query_rope=query_rope,
        key_rope=key_rope,
        scale=scale_value,
        keep_prob=keep_prob,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        sparse_mode=sparse_mode,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        sink_num=sink_num,
    )
    # ST only asserts dq/dk/dv; the current Python wrapper exposes those three.
    return grads[:3]


def flash_attention_score_grad_enhance_bench(*args, **kwargs):
    """同精度参考钩子（evaluator 的 b，见 cann_loader.get_bench_function）。

    与 golden 同一实现、同一签名，但强制 float32 累加精度计算——即修复前
    golden 的口径——作为"正确同精度实现应有的误差下限"，使小值域/相消/正常
    值域兜底的参考错误计数不再恒为 0（issue #311）。非对某颗硬件的复刻；
    更高内部精度的实现会 meet-or-exceed 此下限、照常通过。
    """
    global _BENCH_PRECISION
    _BENCH_PRECISION = True
    try:
        return flash_attention_score_grad_enhance(*args, **kwargs)
    finally:
        _BENCH_PRECISION = False


def get_input(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dy: torch.Tensor,
    pse=None,
    atten_mask=None,
    query_rope=None,
    key_rope=None,
    **kwargs,
):
    # Preserve the published Q/K shapes. Rope participates in the score only;
    # sink rows are already part of the case's physical K/V tensors.
    if (
        not getattr(query, "is_meta", False)
        and atten_mask is None
        and int(kwargs.get("sparse_mode", 0)) in (2, 3, 4)
    ):
        # Preserve master's compressed-mask candidate contract, without
        # expanding it to the actual (possibly >2048) sequence lengths.
        atten_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.uint8), diagonal=1)
    attention_in = softmax_max = softmax_sum = None
    if not getattr(query, "is_meta", False):
        attention_in, softmax_max, softmax_sum = _attention_forward_inputs(
            query,
            key,
            value,
            pse=pse,
            atten_mask=atten_mask,
            query_rope=query_rope,
            key_rope=key_rope,
            **kwargs,
        )
    # Preserve supplied mask storage. Sparse2/3/4 decode visibility only in
    # the CPU reference; mode0 keeps the explicit logical mask semantics.
    return [
        query,
        key,
        value,
        dy,
        pse,
        None,
        atten_mask,
        softmax_max,
        softmax_sum,
        None,
        attention_in,
        None,
        query_rope,
        key_rope,
    ]
```

CPU 主参考使用 FP64，CPU 同精度参考使用 FP32。NPU 上的 FP16 和 BF16 输入沿用输入精度在设备上计算。
