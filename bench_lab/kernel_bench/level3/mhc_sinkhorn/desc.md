# MhcSinkhorn 算子 API 描述

## 1. 算子简介

MHC(Multi-Head Communication) Sinkhorn 算子，用于实现基于 Sinkhorn 迭代的软路由分配，常用于 MoE (Mixture of Experts) 模型中将 token 分配给不同专家。

**主要应用场景**：
- MoE 模型中 token 到专家的软路由分配
- 基于 Sinkhorn 迭代的双随机矩阵近似
- 稀疏激活 Transformer 中的负载均衡路由策略

**算子特征**：
- 难度等级：L3（VVFusion）
- 单输入单输出，支持 3D 输入 [B, S, E]，通过温度缩放和 Sinkhorn 迭代实现软路由权重计算

## 2. 算子定义

### 数学公式

$$
P = \text{Sinkhorn}(\exp(\text{logits} / \text{temperature}), \text{n\_iters})
$$

### 处理流程

1. 对输入 logits 进行温度缩放：$\text{log\_alpha} = \text{logits} / \text{temperature}$
2. 进行 $\text{n\_iters}$ 次 Sinkhorn 迭代：
   - 行归一化（log 域）：$\text{log\_alpha} = \text{log\_alpha} - \text{logsumexp}(\text{log\_alpha}, \text{dim}=-1)$
   - 列归一化（log 域）：$\text{log\_alpha} = \text{log\_alpha} - \text{logsumexp}(\text{log\_alpha}, \text{dim}=-2)$
3. 返回 $\exp(\text{log\_alpha})$ 作为路由权重

## 3. 接口规范

### 算子原型

```python
ascend_bench.mhc_sinkhorn(Tensor logits, float temperature, int n_iters) -> Tensor routing_weights
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| logits | Tensor | 必选 | 输入 logits 张量，shape [B, S, E] |
| temperature | float | 1.0 | Sinkhorn 温度参数 |
| n_iters | int | 3 | Sinkhorn 迭代次数 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| routing_weights | [B, S, E] | 与输入 logits 相同 | Sinkhorn 迭代后的路由权重张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 |
| bfloat16 | bfloat16 |
| float32 | float32 |

### 规则与约束

- 输入 `logits` 必须为 3D 张量，shape 为 [B, S, E]
- `temperature` 必须为正浮点数，控制路由的"软硬"程度（越小越接近硬路由）
- `n_iters` 必须为正整数，控制 Sinkhorn 迭代的收敛程度
- 输出 `routing_weights` 的 shape 与输入 `logits` 完全一致
- Sinkhorn 迭代在 log 域进行以保证数值稳定性

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
MhcSinkhorn算子Torch Golden参考实现

MHC(Multi-Head Communication) Sinkhorn 算子，用于实现基于 Sinkhorn 迭代的软路由分配
公式: P = Sinkhorn(exp(logits / temperature), n_iters)
"""
def mhc_sinkhorn(
    logits: torch.Tensor, temperature: float = 1.0, n_iters: int = 3
) -> torch.Tensor:
    """
    MHC Sinkhorn 软路由分配算子

    公式: P = Sinkhorn(exp(logits / temperature), n_iters)

    Args:
        logits: 输入 logits 张量，shape [B, S, E]
        temperature: Sinkhorn 温度参数
        n_iters: Sinkhorn 迭代次数

    Returns:
        routing_weights: 路由权重张量，shape [B, S, E]
    """
    # Apply temperature scaling
    log_alpha = logits / temperature
    # Sinkhorn iterations in log-domain
    for _ in range(n_iters):
        # Row normalization (log-domain)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        # Column normalization (log-domain)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import ascend_bench

logits = torch.randn(4, 512, 32, dtype=torch.float16, device="npu")
routing_weights = ascend_bench.mhc_sinkhorn(logits, temperature=1.0, n_iters=3)

# 低温度（趋近硬路由）
routing_weights = ascend_bench.mhc_sinkhorn(logits, temperature=0.1, n_iters=5)

# 高温度（趋近均匀分配）
routing_weights = ascend_bench.mhc_sinkhorn(logits, temperature=2.0, n_iters=3)
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，所有用例的 baseline_perf_us 均为 None，性能基线数据尚未测量。

### 相关算子

- **MoeGatingTopKSoftmax**：MoE 门控网络中 Softmax 和 TopK 的融合算子，用于选取 TopK 专家
- **MoeReRouting**：MoE token 重排算子，根据路由结果重新排列 token
- **MoeFinalizeRoutingV2**：MoE 路由合并算子，使用路由权重对专家输出进行加权求和
