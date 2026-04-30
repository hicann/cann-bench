# GRU 算子 API 描述

## 1. 算子简介

Gated Recurrent Unit 循环神经网络算子，实现带门控机制的循环单元，通过更新门和重置门控制信息流动，支持多层堆叠、双向处理和可选偏置。

**主要应用场景**：
- 自然语言处理中的序列建模（机器翻译、文本分类）
- 语音识别中的时序特征提取
- 时间序列预测与异常检测
- 作为 LSTM 的轻量替代方案（参数更少，无细胞状态）

**算子特征**：
- 难度等级：L4（FusedComposite）
- 多输入（x, weight_ih, weight_hh, 可选 bias_ih, bias_hh, h0）双输出（y, hn）
- 支持多层堆叠、双向处理、batch_first 格式、层间 Dropout

## 2. 算子定义

### 数学公式

对于每个时间步 $t$：

$$
z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

$$
r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

$$
n_t = \tanh(W_n x_t + r_t \odot (U_n h_{t-1} + b_n))
$$

$$
h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}
$$

其中：
- $z_t$ 为更新门，控制前一时刻隐藏状态的保留比例
- $r_t$ 为重置门，控制前一时刻隐藏状态对候选状态的影响
- $n_t$ 为候选隐藏状态
- $h_t$ 为当前时刻的隐藏状态
- $\sigma$ 为 sigmoid 函数，$\odot$ 为逐元素乘法

## 3. 接口规范

### 算子原型

```python
cann_bench.gru(Tensor x, TensorList weight_ih, TensorList weight_hh, TensorList? bias_ih, TensorList? bias_hh, Tensor? h0, int inputSize, int hiddenSize, int numLayers, bool bias, bool batchFirst, float dropout, bool bidirectional) -> (Tensor y, Tensor hn)
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入序列张量，shape 为 (S, B, input_size) 或 (B, S, input_size) |
| weight_ih | TensorList | 必选 | 输入到隐藏层权重列表，每层/每个方向独立 tensor。详见权重列表格式 |
| weight_hh | TensorList | 必选 | 隐藏层到隐藏层权重列表，每个 tensor shape 为 (3*hiddenSize, hiddenSize) |
| bias_ih | TensorList | None | 输入到隐藏层偏置列表（可选），每个 tensor shape 为 (3*hiddenSize) |
| bias_hh | TensorList | None | 隐藏层到隐藏层偏置列表（可选），每个 tensor shape 为 (3*hiddenSize) |
| h0 | Tensor | None | 初始隐藏状态（可选，默认全 0），shape 为 (num_layers * num_directions, B, hiddenSize) |
| inputSize | int | 必选 | 输入特征维度 |
| hiddenSize | int | 必选 | 隐藏状态特征维度 |
| numLayers | int | 1 | 循环层数 |
| bias | bool | true | 是否使用偏置 |
| batchFirst | bool | false | 输入是否为 (B, S, input_size) 格式 |
| dropout | float | 0.0 | Dropout 概率 |
| bidirectional | bool | false | 是否双向 GRU |

### 权重列表格式

GRU 有 3 个门（z, r, n），每个门都需要独立的权重矩阵。权重以 TensorList 形式传入，每层、每个方向为独立的 tensor。

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
| weight_ih (单向) | (3*hiddenSize, inputSize) | (3*hiddenSize, hiddenSize) |
| weight_ih (双向) | (3*hiddenSize, inputSize) | (3*hiddenSize, 2*hiddenSize) |
| weight_hh | (3*hiddenSize, hiddenSize) | (3*hiddenSize, hiddenSize) |
| bias_ih/bias_hh | (3*hiddenSize) | (3*hiddenSize) |

**示例：**

- 单层单向: `weight_ih = [tensor(3*H, inputSize)]`, `weight_hh = [tensor(3*H, H)]`
- 单层双向: `weight_ih = [tensor(3*H, inputSize), tensor(3*H, inputSize)]` (forward + reverse)
- 两层双向: `weight_ih = [tensor(3*H, inputSize), tensor(3*H, inputSize), tensor(3*H, 2*H), tensor(3*H, 2*H)]`

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | (S, B, num_directions * hiddenSize) 或 (B, S, num_directions * hiddenSize) | 与输入 x 相同 | 输出序列 |
| hn | (num_layers * num_directions, B, hiddenSize) | 与输入 x 相同 | 最终隐藏状态 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |
| bfloat16 | bfloat16 |

### 规则与约束

- 所有输入 Tensor 的 dtype 必须一致
- GRU 有 3 个门（z, r, n），因此权重矩阵行数为 3*hiddenSize
- 多层 GRU 时，Layer k 的输入来自前一层的输出，因此 weight_ih 的列维度需要调整
- 当 `bias=true` 时，`bias_ih` 和 `bias_hh` 必须提供
- 当 `bidirectional=true` 时，num_directions=2，否则为 1
- `dropout` 仅在 `numLayers > 1` 时生效，作用于层间（非最后一层）
- `batchFirst=true` 时，输入 x 的 shape 为 (B, S, input_size)，输出 y 的 shape 为 (B, S, num_directions * hiddenSize)
- PyTorch GRU 内部使用 float32 计算，float16/bfloat16 输入会转换为 float32 后计算，结果再转回原 dtype

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

def gru(
    x: torch.Tensor,
    weight_ih: List[torch.Tensor],
    weight_hh: List[torch.Tensor],
    bias_ih: Optional[List[torch.Tensor]] = None,
    bias_hh: Optional[List[torch.Tensor]] = None,
    h0: Optional[torch.Tensor] = None,
    inputSize: int = 0,
    hiddenSize: int = 0,
    numLayers: int = 1,
    bias: bool = True,
    batchFirst: bool = False,
    dropout: float = 0.0,
    bidirectional: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_directions = 2 if bidirectional else 1
    gru_layer = torch.nn.GRU(
        input_size=inputSize, hidden_size=hiddenSize, num_layers=numLayers,
        bias=bias, batch_first=batchFirst,
        dropout=dropout if numLayers > 1 else 0.0, bidirectional=bidirectional
    )
    input_dtype = x.dtype
    gru_layer = gru_layer.float()

    with torch.no_grad():
        for layer in range(numLayers):
            getattr(gru_layer, f'weight_ih_l{layer}').copy_(weight_ih[layer * num_directions].float())
            getattr(gru_layer, f'weight_hh_l{layer}').copy_(weight_hh[layer * num_directions].float())
            if bias and bias_ih is not None:
                getattr(gru_layer, f'bias_ih_l{layer}').copy_(bias_ih[layer * num_directions].float())
            if bias and bias_hh is not None:
                getattr(gru_layer, f'bias_hh_l{layer}').copy_(bias_hh[layer * num_directions].float())
            if bidirectional:
                getattr(gru_layer, f'weight_ih_l{layer}_reverse').copy_(weight_ih[layer * num_directions + 1].float())
                getattr(gru_layer, f'weight_hh_l{layer}_reverse').copy_(weight_hh[layer * num_directions + 1].float())
                if bias and bias_ih is not None:
                    getattr(gru_layer, f'bias_ih_l{layer}_reverse').copy_(bias_ih[layer * num_directions + 1].float())
                if bias and bias_hh is not None:
                    getattr(gru_layer, f'bias_hh_l{layer}_reverse').copy_(bias_hh[layer * num_directions + 1].float())

    x_float = x.float()
    if h0 is None:
        batch_size = x.shape[1] if not batchFirst else x.shape[0]
        h0 = torch.zeros(numLayers * num_directions, batch_size, hiddenSize, dtype=torch.float32)
    else:
        h0 = h0.float()
    y, hn = gru_layer(x_float, h0)
    return y.to(input_dtype), hn.to(input_dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 单层单向 GRU（TensorList 格式）
seq_len, batch, input_size, hidden_size = 20, 8, 128, 256
x = torch.randn(seq_len, batch, input_size, dtype=torch.float32, device="npu")
weight_ih = [torch.randn(3 * hidden_size, input_size, dtype=torch.float32, device="npu")]
weight_hh = [torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32, device="npu")]
bias_ih = [torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")]
bias_hh = [torch.randn(3 * hidden_size, dtype=torch.float32, device="npu")]
y, hn = cann_bench.gru(x, weight_ih, weight_hh, bias_ih, bias_hh, None,
                          inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                          bias=True, batchFirst=False, dropout=0.0, bidirectional=False)

# 双向 GRU
weight_ih_bi = [torch.randn(3 * hidden_size, input_size, dtype=torch.float32, device="npu"),
                torch.randn(3 * hidden_size, input_size, dtype=torch.float32, device="npu")]
weight_hh_bi = [torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32, device="npu"),
                torch.randn(3 * hidden_size, hidden_size, dtype=torch.float32, device="npu")]
y_bi, hn_bi = cann_bench.gru(x, weight_ih_bi, weight_hh_bi, None, None, None,
                                inputSize=input_size, hiddenSize=hidden_size, numLayers=1,
                                bias=False, batchFirst=False, dropout=0.0, bidirectional=True)
```
