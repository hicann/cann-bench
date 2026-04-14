# Engram 算子 API 描述

## 1. 算子简介

Engram 算子，实现记忆增强的注意力机制中的记忆编码与检索融合操作，将输入特征与记忆库进行交互计算。

**主要应用场景**：
- 记忆增强型 Transformer 中的外部记忆检索
- 长序列建模中的记忆库交互
- 基于注意力的知识检索与融合

**算子特征**：
- 难度等级：L3（VVFusion）
- 双输入单输出，输入特征 [B, S, D] 与记忆库 [B, M, D] 进行注意力交互，输出增强后的特征

## 2. 算子定义

### 数学公式

$$
y = x + \alpha \cdot \text{softmax}\left(\frac{x \cdot \text{memory}^T}{\sqrt{d}}\right) \cdot \text{memory}
$$

### 处理流程

1. 计算缩放因子：若 $\text{scale} \leq 0$，则 $\text{scale} = \frac{1}{\sqrt{D}}$
2. 计算注意力得分：$\text{scores} = x \cdot \text{memory}^T \cdot \text{scale}$，shape 为 $[B, S, M]$
3. 对 scores 沿最后一维执行 Softmax 得到注意力权重：$\text{attn} = \text{softmax}(\text{scores}, \text{dim}=-1)$
4. 加权求和记忆内容：$\text{mem\_out} = \text{attn} \cdot \text{memory}$，shape 为 $[B, S, D]$
5. 残差连接：$y = x + \alpha \cdot \text{mem\_out}$

## 3. 接口规范

### 算子原型

```python
ascend_bench.engram(Tensor x, Tensor memory, float alpha, float scale) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征张量，shape [B, S, D] |
| memory | Tensor | 必选 | 记忆库张量，shape [B, M, D] |
| alpha | float | 1.0 | 记忆增强系数 |
| scale | float | -1.0 | 缩放因子，<=0 表示自动使用 1/sqrt(D) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [B, S, D] | 与输入 x 相同 | 记忆增强后的输出张量 |

### 数据类型

| 输入 (x) dtype | 输入 (memory) dtype | 输出 dtype |
|---------------|-------------------|-----------|
| float16 | float16 | float16 |
| bfloat16 | bfloat16 | bfloat16 |
| float32 | float32 | float32 |

### 规则与约束

- 输入 `x` 必须为 3D 张量，shape 为 [B, S, D]
- 输入 `memory` 必须为 3D 张量，shape 为 [B, M, D]
- `x` 和 `memory` 的 batch 维度 B 和特征维度 D 必须一致
- `x` 和 `memory` 的 dtype 必须相同
- `alpha` 为浮点数，控制记忆增强的强度
- `scale` 为浮点数，<=0 时自动计算为 1/sqrt(D)
- 输出 `y` 的 shape 和 dtype 与输入 `x` 完全一致

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比，需满足以下误差阈值：

| 数据类型 | 验证方式 | rtol | atol |
|---------|---------|------|------|
| float16 | 相对误差 | 1e-3 | 1e-3 |
| float32 | 相对误差 | 1e-4 | 1e-4 |
| bfloat16 | 相对误差 | 4e-3 | 4e-3 |

**对比公式**：

$$
|output - golden| \leq atol + rtol \times |golden|
$$

## 5. 标准 Golden 代码

```python
import torch

"""
Engram算子Torch Golden参考实现

记忆增强的注意力机制中的记忆编码与检索融合操作
公式: y = x + alpha * softmax(x @ memory^T / sqrt(d)) @ memory
"""
def engram(
    x: torch.Tensor, memory: torch.Tensor, alpha: float = 1.0, scale: float = -1.0
) -> torch.Tensor:
    """
    Engram 记忆增强注意力算子

    公式: y = x + alpha * softmax(x @ memory^T / sqrt(d)) @ memory

    Args:
        x: 输入特征张量，shape [B, S, D]
        memory: 记忆库张量，shape [B, M, D]
        alpha: 记忆增强系数
        scale: 缩放因子，<=0 表示自动使用 1/sqrt(D)

    Returns:
        y: 记忆增强后的输出张量，shape [B, S, D]
    """
    d = x.shape[-1]
    if scale <= 0:
        scale = 1.0 / (d ** 0.5)
    scores = torch.matmul(x, memory.transpose(-2, -1)) * scale
    attn = torch.nn.functional.softmax(scores, dim=-1)
    mem_out = torch.matmul(attn, memory)
    return x + alpha * mem_out
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

x = torch.randn(4, 512, 256, dtype=torch.float16, device="npu")
memory = torch.randn(4, 128, 256, dtype=torch.float16, device="npu")
y = ascend_bench.engram(x, memory, alpha=1.0, scale=-1.0)

# 显式指定 scale
y = ascend_bench.engram(x, memory, alpha=0.5, scale=0.0625)

# 强记忆增强
y = ascend_bench.engram(x, memory, alpha=2.0, scale=-1.0)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **Softmax**：Engram 内部使用 Softmax 计算注意力权重
- **GroupedMatmul**：分组矩阵乘法，可用于批量化的记忆检索场景
- **MhcSinkhorn**：同为 VVFusion 类算子，用于 MoE 软路由分配
