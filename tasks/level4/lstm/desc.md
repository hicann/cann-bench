# LSTM 算子 API 描述

## 1. 算子简介

Long Short-Term Memory 循环神经网络算子，通过输入门、遗忘门、输出门和细胞状态的门控机制实现长距离依赖建模，支持多层堆叠、双向处理、投影降维和可选偏置。

**主要应用场景**：
- 自然语言处理中的序列到序列建模（机器翻译、文本生成）
- 语音识别与合成中的时序特征建模
- 时间序列预测与长距离依赖建模
- 需要投影降维（proj_size）的大规模隐藏状态场景

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（x, weight_ih, weight_hh, 可选 bias_ih, bias_hh, h0, c0）三输出（y, hn, cn）
- 支持多层堆叠、双向处理、batch_first 格式和投影降维；本基准为推理模式，不执行层间 Dropout

## 2. 算子定义

### 数学公式

对于每个时间步 $t$：

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad \text{（输入门）}
$$

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad \text{（遗忘门）}
$$

$$
g_t = \tanh(W_g x_t + U_g h_{t-1} + b_g) \quad \text{（候选细胞状态）}
$$

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad \text{（输出门）}
$$

$$
c_t = f_t \odot c_{t-1} + i_t \odot g_t \quad \text{（细胞状态）}
$$

$$
h_t' = o_t \odot \tanh(c_t) \quad \text{（隐藏状态）}
$$

当 `projSize > 0`（LSTM with Projection / LSTMP）时，隐藏状态再经投影矩阵 $W_{hr}$ 降维至 projSize 维；`projSize = 0` 时 $h_t = h_t'$：

$$
h_t = W_{hr}\, h_t' = W_{hr}\,(o_t \odot \tanh(c_t)) \quad \text{（投影隐藏状态，维度 projSize）}
$$

其中：
- $i_t, f_t, o_t$ 分别为输入门、遗忘门、输出门
- $g_t$ 为候选细胞状态
- $c_t$ 为细胞状态，$h_t$ 为隐藏状态（$h_t'$ 为投影前隐藏状态，仅 projSize>0 时区分）
- $W_{hr}$ 为投影矩阵（shape $(\text{projSize}, \text{hiddenSize})$，仅 projSize>0 / LSTMP 时存在）
- $\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘法

## 3. 接口规范

### 算子原型

```python
cann_bench.lstm(Tensor x, TensorList weight_ih, TensorList weight_hh, int inputSize, int hiddenSize, int numLayers, bool bias=True, bool batchFirst=False, float dropout=0.0, bool bidirectional=False, int projSize=0, TensorList? bias_ih=None, TensorList? bias_hh=None, TensorList? weight_hr=None, Tensor? h0=None, Tensor? c0=None) -> (Tensor y, Tensor hn, Tensor cn)
```

### 输入参数说明

下文记 `num_directions = 2 if bidirectional else 1`，
`H_out = projSize if projSize > 0 else hiddenSize`。隐藏状态使用 `H_out`，细胞状态始终使用 `hiddenSize`。

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入序列张量，shape 为 (S, B, input_size) 或 (B, S, input_size) |
| weight_ih | TensorList | 必选 | 输入到隐藏层权重列表，每层/每个方向独立 tensor。详见权重列表格式 |
| weight_hh | TensorList | 必选 | 隐藏层到隐藏层权重列表，每个 tensor shape 为 (4*hiddenSize, H_out) |
| inputSize | int | 必选 | 输入特征维度 |
| hiddenSize | int | 必选 | 细胞状态及投影前隐藏状态的特征维度 |
| numLayers | int | 1 | 循环层数 |
| bias | bool | true | 是否使用偏置 |
| batchFirst | bool | false | 输入是否为 (B, S, input_size) 格式 |
| dropout | float | 0.0 | 保留层间 Dropout 概率参数；推理模式下不生效 |
| bidirectional | bool | false | 是否双向 LSTM |
| projSize | int | 0 | 投影维度（>0 时启用 LSTM with Projection） |
| bias_ih | TensorList | None | 输入到隐藏层偏置列表（可选），每个 tensor shape 为 (4*hiddenSize) |
| bias_hh | TensorList | None | 隐藏层到隐藏层偏置列表（可选），每个 tensor shape 为 (4*hiddenSize) |
| weight_hr | TensorList | None | 投影矩阵列表（可选，仅 projSize>0 / LSTMP 需要），每层/每个方向一个 tensor，shape 为 (projSize, hiddenSize) |
| h0 | Tensor | None | 初始隐藏状态（可选，默认全 0），shape 为 (numLayers * num_directions, B, H_out) |
| c0 | Tensor | None | 初始细胞状态（可选，默认全 0），shape 为 (numLayers * num_directions, B, hiddenSize) |

### 权重列表格式

LSTM 有 4 个门（i, f, g, o），每个门都需要独立的权重矩阵。权重以 TensorList 形式传入，每层、每个方向为独立的 tensor。

**TensorList 长度计算：**

- `len(weight_ih) = numLayers * num_directions`
- `len(weight_hh) = numLayers * num_directions`
- `len(weight_hr) = numLayers * num_directions`（`projSize > 0` 时）
- `len(bias_ih) = numLayers * num_directions`（如有偏置）
- `len(bias_hh) = numLayers * num_directions`（如有偏置）

**排列顺序：**

```
[weight_ih_l0, weight_ih_l0_reverse, weight_ih_l1, weight_ih_l1_reverse, ...]  (bidirectional=true)
[weight_ih_l0, weight_ih_l1, ...]  (bidirectional=false)
```

**每个 tensor shape：**

| 参数 | Layer 0 | Layer k (k>0) |
|------|---------|---------------|
| weight_ih (单向) | (4*hiddenSize, inputSize) | (4*hiddenSize, H_out) |
| weight_ih (双向) | (4*hiddenSize, inputSize) | (4*hiddenSize, 2*H_out) |
| weight_hh | (4*hiddenSize, H_out) | (4*hiddenSize, H_out) |
| weight_hr（仅 projSize>0） | (projSize, hiddenSize) | (projSize, hiddenSize) |
| bias_ih/bias_hh | (4*hiddenSize) | (4*hiddenSize) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (S, B, num_directions * H_out) 或 (B, S, num_directions * H_out) | 与输入 x 相同 | 输出序列 |
| hn | (numLayers * num_directions, B, H_out) | 与输入 x 相同 | 最终隐藏状态 |
| cn | (numLayers * num_directions, B, hiddenSize) | 与输入 x 相同 | 最终细胞状态 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor 的 dtype 必须一致
- LSTM 有 4 个门（i, f, g, o），因此权重矩阵行数为 4*hiddenSize
- 多层 LSTM 时，Layer k 的输入来自前一层的输出，因此 weight_ih 的列维度需要调整
- 当 `bias=true` 时，`bias_ih` 和 `bias_hh` 必须提供
- 当 `bidirectional=true` 时，num_directions=2，否则为 1
- 本算子执行推理前向，所有层均禁用 Dropout。即使 `numLayers > 1` 且 `dropout > 0`，也不生成或应用随机掩码。
- `projSize > 0` 时须满足 `projSize < hiddenSize`，并提供 `weight_hr`；隐藏状态有效维度为 `H_out=projSize`，细胞状态维度仍为 `hiddenSize`。
- `batchFirst=true` 时，输入 x 的 shape 为 (B, S, input_size)，输出 y 的 shape 为 (B, S, num_directions * H_out)

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `S`（序列长度） | 1 ~ 2048 | cases.csv 实测 5 ~ 200 |
| `B`（batch） | 1 ~ 256 | cases.csv 实测 2 ~ 32 |
| `inputSize` | 1 ~ 2048 | cases.csv 实测 32 ~ 512 |
| `hiddenSize` | 1 ~ 2048 | cases.csv 实测 16 ~ 256；weight 行数 = 4*hiddenSize |
| `numLayers` | 1 ~ 8 | cases.csv 实测 1 ~ 3 |
| `bias` | {true, false} | cases.csv 覆盖两种；为 true 时 bias_ih/bias_hh 必填 |
| `batchFirst` | {true, false} | cases.csv 覆盖两种；true 时 x 为 (B, S, inputSize) |
| `dropout` | 0.0 ~ 1.0 | cases.csv 实测 0.0 ~ 0.1；参数保留，但推理模式下不生效 |
| `bidirectional` | {true, false} | cases.csv 覆盖两种；true 时 num_directions=2 |
| `projSize` | 0，或 0 < projSize < hiddenSize | cases.csv 实测 0 / 16；>0 时启用投影降维 |
| `h0`, `c0` 维度 0 | = numLayers * num_directions | 可选；缺省全 0 |

约束：`weight_ih[k]` shape 为 (4*hiddenSize, inputSize)（第 0 层）或 (4*hiddenSize, num_directions*H_out)（其它层）；`weight_hh[k]` shape 为 (4*hiddenSize, H_out)；`bias_ih[k]`、`bias_hh[k]` shape 为 (4*hiddenSize)；TensorList 长度 = numLayers * num_directions。

## 4. 精度要求

采用[生态算子精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md)进行验证。

**误差指标**：

1. 平均相对误差（MERE）：采样点中相对误差平均值

   $$
   \text{MERE} = \text{avg}(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

2. 最大相对误差（MARE）：采样点中相对误差最大值

   $$
   \text{MARE} = \max(\frac{\text{abs}(actual - golden)}{\text{abs}(golden)+\text{1e-7}})
   $$

**通过标准**：

| 数据类型 | FLOAT16 | BFLOAT16 | FLOAT32 | HiFLOAT32 | FLOAT8 E4M3 | FLOAT8 E5M2 |
|----------|---------|----------|---------|-----------|-------------|-------------|
| **通过阈值(Threshold)** | 2^-10 | 2^-7 | 2^-13 | 2^-11 | 2^-3 | 2^-2 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过。


## 5. 标准 Golden 代码

```python
import torch
from typing import List, Optional, Tuple

def lstm(
    x: torch.Tensor,
    weight_ih: List[torch.Tensor],
    weight_hh: List[torch.Tensor],
    inputSize: int,
    hiddenSize: int,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False,
    projSize: int = 0,
    bias_ih: Optional[List[torch.Tensor]] = None,
    bias_hh: Optional[List[torch.Tensor]] = None,
    weight_hr: Optional[List[torch.Tensor]] = None,
    h0: Optional[torch.Tensor] = None,
    c0: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    LSTM 前向计算（对标 PyTorch torch.nn.LSTM）

    Args:
        x: 输入序列 (S, B, inputSize) 或 (B, S, inputSize) if batch_first
        weight_ih: TensorList，每层每方向一个 [4*hiddenSize, input_dim] tensor
        weight_hh: TensorList，每层每方向一个 [4*hiddenSize, hiddenSize or projSize] tensor
        inputSize: 输入特征维度
        hiddenSize: 隐藏状态维度
        numLayers: 层数
        bias: 是否使用偏置
        batchFirst: 输入格式是否为 (batch, seq, feature)
        dropout: 保留层间 dropout 参数；推理模式下不生效
        bidirectional: 是否双向
        projSize: 投影维度 (>0 时启用 LSTM with Projection / LSTMP)
        bias_ih: TensorList?, 每层每方向一个 [4*hiddenSize] tensor
        bias_hh: TensorList?, 每层每方向一个 [4*hiddenSize] tensor
        weight_hr: TensorList?, 投影矩阵，每层每方向一个 [projSize, hiddenSize]
            tensor；仅 projSize>0 时需要 (LSTMP)
        h0: 初始隐藏状态 [numLayers*num_directions, B, hiddenSize or projSize]
        c0: 初始细胞状态 [numLayers*num_directions, B, hiddenSize]

    Returns:
        y: 输出序列
        hn: 最终隐藏状态
        cn: 最终细胞状态
    """
    num_directions = 2 if bidirectional else 1
    gate_size = 4 * hiddenSize  # LSTM: i, f, g, o
    effective_hidden = projSize if projSize > 0 else hiddenSize

    if projSize > 0 and weight_hr is None:
        raise ValueError("projSize>0 (LSTMP) requires weight_hr (projection matrix list)")

    lstm_layer = torch.nn.LSTM(
        input_size=inputSize,
        hidden_size=hiddenSize,
        num_layers=numLayers,
        bias=bias,
        batch_first=batchFirst,
        dropout=dropout if numLayers > 1 else 0.0,
        bidirectional=bidirectional,
        proj_size=projSize if projSize > 0 else 0,
    )

    input_dtype = x.dtype
    lstm_layer = lstm_layer.float().to(x.device)
    # Inference golden: disable inter-layer dropout so the reference is deterministic
    # (nn.Module defaults to train mode; for numLayers>1 + dropout>0 the forward would
    # apply a random mask each call -> nondeterministic hn/cn that no kernel can match).
    lstm_layer.eval()

    # 计算每层的输入维度
    layer_inputs = [inputSize]
    for layer in range(1, numLayers):
        layer_inputs.append(effective_hidden * num_directions)

    # 设置权重参数（TensorList 格式）。projSize>0 时必须设上投影权重 weight_hr，
    # 否则 nn.LSTM 用随机初始化的投影矩阵 -> 参考输出非确定。
    with torch.no_grad():
        for layer in range(numLayers):
            layer_input = layer_inputs[layer]
            for d in range(num_directions):
                idx = layer * num_directions + d
                suffix = f"l{layer}" if d == 0 else f"l{layer}_reverse"

                # 从 TensorList 中取对应 tensor
                wi = weight_ih[idx][:gate_size, :layer_input]
                wh = weight_hh[idx][:gate_size, :effective_hidden]

                getattr(lstm_layer, f'weight_ih_{suffix}').copy_(wi.float())
                getattr(lstm_layer, f'weight_hh_{suffix}').copy_(wh.float())

                if projSize > 0:
                    wr = weight_hr[idx][:projSize, :hiddenSize]
                    getattr(lstm_layer, f'weight_hr_{suffix}').copy_(wr.float())

                if bias and bias_ih is not None and bias_hh is not None:
                    bi = bias_ih[idx][:gate_size]
                    bh = bias_hh[idx][:gate_size]
                    getattr(lstm_layer, f'bias_ih_{suffix}').copy_(bi.float())
                    getattr(lstm_layer, f'bias_hh_{suffix}').copy_(bh.float())

    x_float = x.float()
    if h0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        h0 = torch.zeros(numLayers * num_directions, batch_size, effective_hidden,
                         dtype=torch.float32, device=x.device)
    else:
        h0 = h0.float()

    if c0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        c0 = torch.zeros(numLayers * num_directions, batch_size, hiddenSize,
                         dtype=torch.float32, device=x.device)
    else:
        c0 = c0.float()

    y, (hn, cn) = lstm_layer(x_float, (h0, c0))
    y = y.to(input_dtype)
    hn = hn.to(input_dtype)
    cn = cn.to(input_dtype)

    return y, hn, cn
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 单层单向 LSTM（TensorList 格式）
seq_len, batch, input_size, hidden_size = 20, 8, 128, 256
x = torch.randn(seq_len, batch, input_size, dtype=torch.float32, device="npu")
weight_ih = [torch.randn(4 * hidden_size, input_size, dtype=torch.float32, device="npu")]
weight_hh = [torch.randn(4 * hidden_size, hidden_size, dtype=torch.float32, device="npu")]
bias_ih = [torch.randn(4 * hidden_size, dtype=torch.float32, device="npu")]
bias_hh = [torch.randn(4 * hidden_size, dtype=torch.float32, device="npu")]
y, hn, cn = cann_bench.lstm(x, weight_ih, weight_hh,
                               inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                               bias=True, batchFirst=False, bidirectional=False,
                               bias_ih=bias_ih, bias_hh=bias_hh)

# 双向 LSTM
weight_ih_bi = [torch.randn(4 * hidden_size, input_size), torch.randn(4 * hidden_size, input_size)]
weight_hh_bi = [torch.randn(4 * hidden_size, hidden_size), torch.randn(4 * hidden_size, hidden_size)]
y_bi, hn_bi, cn_bi = cann_bench.lstm(x, weight_ih_bi, weight_hh_bi,
                                        inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                                        bias=False, bidirectional=True)
```
