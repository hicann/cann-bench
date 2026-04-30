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
- 支持多层堆叠、双向处理、batch_first 格式、层间 Dropout、投影降维

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
h_t = o_t \odot \tanh(c_t) \quad \text{（隐藏状态）}
$$

其中：
- $i_t, f_t, o_t$ 分别为输入门、遗忘门、输出门
- $g_t$ 为候选细胞状态
- $c_t$ 为细胞状态，$h_t$ 为隐藏状态
- $\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘法

## 3. 接口规范

### 算子原型

```python
cann_bench.lstm(Tensor x, TensorList weight_ih, TensorList weight_hh, TensorList? bias_ih, TensorList? bias_hh, Tensor? h0, Tensor? c0, int inputSize, int hiddenSize, int numLayers, bool bias, bool batchFirst, float dropout, bool bidirectional, int projSize) -> (Tensor y, Tensor hn, Tensor cn)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入序列张量，shape 为 (S, B, input_size) 或 (B, S, input_size) |
| weight_ih | TensorList | 必选 | 输入到隐藏层权重列表，每层/每个方向独立 tensor。详见权重列表格式 |
| weight_hh | TensorList | 必选 | 隐藏层到隐藏层权重列表，每个 tensor shape 为 (4*hiddenSize, hiddenSize) |
| bias_ih | TensorList | None | 输入到隐藏层偏置列表（可选），每个 tensor shape 为 (4*hiddenSize) |
| bias_hh | TensorList | None | 隐藏层到隐藏层偏置列表（可选），每个 tensor shape 为 (4*hiddenSize) |
| h0 | Tensor | None | 初始隐藏状态（可选，默认全 0），shape 为 (num_layers * num_directions, B, hiddenSize) |
| c0 | Tensor | None | 初始细胞状态（可选，默认全 0），shape 为 (num_layers * num_directions, B, hiddenSize) |
| inputSize | int | 必选 | 输入特征维度 |
| hiddenSize | int | 必选 | 隐藏状态特征维度 |
| numLayers | int | 1 | 循环层数 |
| bias | bool | true | 是否使用偏置 |
| batchFirst | bool | false | 输入是否为 (B, S, input_size) 格式 |
| dropout | float | 0.0 | Dropout 概率（层间） |
| bidirectional | bool | false | 是否双向 LSTM |
| projSize | int | 0 | 投影维度（>0 时启用 LSTM with Projection） |

### 权重列表格式

LSTM 有 4 个门（i, f, g, o），每个门都需要独立的权重矩阵。权重以 TensorList 形式传入，每层、每个方向为独立的 tensor。

**TensorList 长度计算：**

- `len(weight_ih) = numLayers * num_directions`
- `len(weight_hh) = numLayers * num_directions`
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
| weight_ih (单向) | (4*hiddenSize, inputSize) | (4*hiddenSize, hiddenSize) |
| weight_ih (双向) | (4*hiddenSize, inputSize) | (4*hiddenSize, 2*hiddenSize) |
| weight_hh | (4*hiddenSize, hiddenSize) | (4*hiddenSize, hiddenSize 或 projSize) |
| bias_ih/bias_hh | (4*hiddenSize) | (4*hiddenSize) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (S, B, num_directions * hiddenSize) 或 (B, S, num_directions * hiddenSize) | 与输入 x 相同 | 输出序列 |
| hn | (num_layers * num_directions, B, hiddenSize 或 projSize) | 与输入 x 相同 | 最终隐藏状态 |
| cn | (num_layers * num_directions, B, hiddenSize) | 与输入 x 相同 | 最终细胞状态 |

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
- `dropout` 仅在 `numLayers > 1` 时生效，作用于层间（非最后一层）
- `projSize > 0` 时启用投影降维，隐藏状态的有效维度变为 projSize，weight_hh 列维度变为 projSize
- `batchFirst=true` 时，输入 x 的 shape 为 (B, S, input_size)，输出 y 的 shape 为 (B, S, num_directions * hiddenSize)
- PyTorch LSTM 内部使用 float32 计算，float16/bfloat16 输入会转换为 float32 后计算，结果再转回原 dtype

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
    bias_ih: Optional[List[torch.Tensor]] = None,
    bias_hh: Optional[List[torch.Tensor]] = None,
    h0: Optional[torch.Tensor] = None,
    c0: Optional[torch.Tensor] = None,
    inputSize: int = 0,
    hiddenSize: int = 0,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False,
    projSize: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_directions = 2 if bidirectional else 1
    effective_hidden_size = projSize if projSize > 0 else hiddenSize
    lstm_layer = torch.nn.LSTM(
        input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers,
        bias=bias, batch_first=batchFirst,
        dropout=dropout if numLayers > 1 else 0.0, bidirectional=bidirectional,
        proj_size=projSize if projSize > 0 else 0
    )
    input_dtype = x.dtype
    lstm_layer = lstm_layer.float()

    with torch.no_grad():
        for layer in range(numLayers):
            getattr(lstm_layer, f'weight_ih_l{layer}').copy_(weight_ih[layer * num_directions].float())
            getattr(lstm_layer, f'weight_hh_l{layer}').copy_(weight_hh[layer * num_directions].float())
            if bias and bias_ih is not None:
                getattr(lstm_layer, f'bias_ih_l{layer}').copy_(bias_ih[layer * num_directions].float())
            if bias and bias_hh is not None:
                getattr(lstm_layer, f'bias_hh_l{layer}').copy_(bias_hh[layer * num_directions].float())
            if bidirectional:
                getattr(lstm_layer, f'weight_ih_l{layer}_reverse').copy_(weight_ih[layer * num_directions + 1].float())
                getattr(lstm_layer, f'weight_hh_l{layer}_reverse').copy_(weight_hh[layer * num_directions + 1].float())
                if bias and bias_ih is not None:
                    getattr(lstm_layer, f'bias_ih_l{layer}_reverse').copy_(bias_ih[layer * num_directions + 1].float())
                if bias and bias_hh is not None:
                    getattr(lstm_layer, f'bias_hh_l{layer}_reverse').copy_(bias_hh[layer * num_directions + 1].float())

    x_float = x.float()
    if h0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        h0 = torch.zeros(numLayers * num_directions, batch_size, effective_hidden_size, dtype=torch.float32)
    else:
        h0 = h0.float()
    if c0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        c0 = torch.zeros(numLayers * num_directions, batch_size, hiddenSize, dtype=torch.float32)
    else:
        c0 = c0.float()
    y, (hn, cn) = lstm_layer(x_float, (h0, c0))
    return y.to(input_dtype), hn.to(input_dtype), cn.to(input_dtype)
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
y, hn, cn = cann_bench.lstm(x, weight_ih, weight_hh, bias_ih, bias_hh, None, None,
                               inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                               bias=True, batchFirst=False, bidirectional=False)

# 双向 LSTM
weight_ih_bi = [torch.randn(4 * hidden_size, input_size), torch.randn(4 * hidden_size, input_size)]
weight_hh_bi = [torch.randn(4 * hidden_size, hidden_size), torch.randn(4 * hidden_size, hidden_size)]
y_bi, hn_bi, cn_bi = cann_bench.lstm(x, weight_ih_bi, weight_hh_bi, None, None, None, None,
                                        inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                                        bias=False, bidirectional=True)
```
