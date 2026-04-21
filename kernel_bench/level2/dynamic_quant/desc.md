# DynamicQuant 算子 API 描述

## 1. 算子简介

对输入张量进行 per-token 对称动态量化。

**主要应用场景**：
- 大语言模型推理加速中的动态量化（W8A8 / W4A8 等方案）
- KV Cache 量化压缩以节省显存
- 模型部署阶段的在线量化处理

**算子特征**：
- 难度等级：L2（FusedComposite）
- 单输入单输出，涉及求最大值、缩放、四舍五入等多步融合计算
- 输入支持 2-8 维张量

## 2. 算子定义

### 数学公式

$$
scaleOut = \frac{\max_{row}(|x|)}{dtypeMax}
$$

$$
yOut = \text{round}\left(\frac{x}{scaleOut}\right)
$$

其中：
- $\max_{row}(|x|)$ 表示沿指定维度（axis）取绝对值的最大值
- $dtypeMax$ 为目标量化数据类型的最大值（如 int8 对应 127）
- $\text{round}$ 为四舍五入到最近整数

## 3. 接口规范

### 算子原型

```python
cann_bench.dynamic_quant(Tensor x, int axis=-1, int dst_type=0) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入张量 |
| axis | int64 | -1 | 计算 scale 和 zero_point 的维度，默认为最后一个维度 |
| dst_type | int64 | 0 | 目标数据类型 |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | 与输入 x 相同 | float16 / bfloat16 / int8 | 量化后的张量 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float16 | float16 / int8 |
| bfloat16 | bfloat16 / int8 |
| float32 | float32 / int8 |

### 规则与约束

- 输入支持 2-8 维张量
- `axis` 指定沿哪个维度计算 scale，默认为最后一个维度（-1），支持负数索引
- `dst_type` 控制目标量化数据类型
- 输出 shape 与输入 shape 完全一致
- 量化为对称量化（zero_point 为 0），scale 基于绝对值最大值计算
- 当输入值全为 0 时，scale 为 0，需注意除零处理

## 4. 精度要求

计算结果与 PyTorch Golden 实现逐元素对比：

| 数据类型 | 验证方式 | 阈值 |
|---------|---------|------|
| float16（dst_type 为浮点透传） | 相对误差：`\|output-golden\| ≤ atol + rtol×\|golden\|` | rtol=1e-3, atol=1e-3 |
| bfloat16（dst_type 为浮点透传） | 相对误差：`\|output-golden\| ≤ atol + rtol×\|golden\|` | rtol=4e-3, atol=4e-3 |
| float32（dst_type 为浮点透传） | 相对误差 | rtol=1e-4, atol=1e-4 |
| int8（dst_type=0） | 允许量化边界 off-by-1，最大绝对偏差 ≤ 1；off-by-1 元素占比 | < 1e-4 |

**说明**：int8 量化输出允许因 float32 累加顺序差异在 round 时舍到相邻整数；出现 |Δ|≥2 的元素直接判负。

## 5. 标准 Golden 代码

```python
import torch

"""
DynamicQuant算子Torch Golden参考实现

对输入张量进行per-token对称动态量化
公式: scaleOut = row_max(abs(x)) / dtypeMax, yOut = round(x / scaleOut)
"""
def dynamic_quant(
    x: torch.Tensor, axis: int = -1, dst_type: int = 0
) -> torch.Tensor:
    """
    对输入张量进行per-token对称动态量化
    
    公式: scaleOut = row_max(abs(x)) / dtypeMax, yOut = round(x / scaleOut)
    
    Args:
        x: 输入张量
        axis: 计算scale和zero_point的维度，默认为最后一个维度
        dst_type: 目标数据类型
    
    Returns:
        量化后的张量
    """

    scale_out = torch.max(torch.abs(x), dim=axis, keepdim=True)[0] / 127.0
    y = torch.round(x / scale_out)
    return y
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(1024, 1024, dtype=torch.float16, device="npu")
y = cann_bench.dynamic_quant(x, axis=-1, dst_type=0)   # 沿最后一维 per-token 量化

x = torch.randn(2, 8, 256, 256, dtype=torch.bfloat16, device="npu")
y = cann_bench.dynamic_quant(x, axis=-1, dst_type=0)   # 4D 张量 per-token 量化

x = torch.randn(512, 2048, dtype=torch.float16, device="npu")
y = cann_bench.dynamic_quant(x, axis=0, dst_type=0)    # 沿第 0 维 per-channel 量化
```

### 性能基线参考

基于 cases.yaml 中 20 个测试用例，当前所有用例的 baseline_perf_us 均为 0.0，性能基线数据待补充。

### 相关算子

- **ApplyAdamW**：同为 FusedComposite 类别的多步融合计算
- **CrossEntropyLoss**：同涉及数值稳定性处理的复合算子
- **Gather**：索引提取算子，量化后常需配合 Gather 进行查表操作
