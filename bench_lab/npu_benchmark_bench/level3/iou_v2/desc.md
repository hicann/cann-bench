# IouV2 算子 API 描述

## 1. 算子简介

IouV2 计算预测框（bbox）与真实框（gtbox）之间的交并比（IoU）或交占前景比（IOF），支持 aligned 逐行配对与全对配对两种模式。本算子对标 ops-cv `iou_v2` 融合实现，一次完成全部框对的重叠面积与归一化。

**主要应用场景**：
- 目标检测模型训练/推理中的框匹配与正负样本划分
- 检测结果评测（mAP 计算中的 IoU 判定）
- 非极大值抑制、框回归损失等需要框重叠度量的场景

**算子特征**：
- 难度等级：L3（Vision）
- 双张量输入 + 3 个属性（`mode` / `eps` / `aligned`），单输出
- `aligned=False` 时通过广播计算全部 `[N, M]` 框对，`aligned=True` 时逐行配对输出 `[N, 1]`
- 输入框格式为 `[x1, y1, x2, y2]`，要求 `x1 <= x2`、`y1 <= y2`

## 2. 算子定义

### 数学公式

设框 $b_1=(x_1,y_1,x_2,y_2)$、$b_2=(x_3,y_3,x_4,y_4)$，则：

$$
\text{inter} = \max(0, \min(x_2,x_4)-\max(x_1,x_3)) \cdot \max(0, \min(y_2,y_4)-\max(y_1,y_3))
$$

$$
\text{area}_1 = (x_2-x_1)(y_2-y_1),\qquad \text{area}_2 = (x_4-x_3)(y_4-y_3)
$$

$$
\text{iou: } y = \frac{\text{inter}}{\text{area}_1 + \text{area}_2 - \text{inter} + \epsilon}
\qquad
\text{iof: } y = \frac{\text{inter}}{\text{area}_2 + \epsilon}
$$

`aligned=False` 时对每个 $(n, m)$ 组合分别计算并输出 `[N, M]`；`aligned=True` 时仅计算同下标框对并输出 `[N, 1]`。

### 实现建议

- `aligned=False` 通过 `bboxes[:, None, :]` 与 `gtboxes[None, :, :]` 广播一次向量化算出全部框对，无需 Python 循环。
- 重叠宽高须先用 `clamp(min=0)` 截断为 0，避免不相交框产生负面积。
- `iof` 的分母使用 gtbox 面积 `area2`（与源实现一致）。

## 3. 接口规范

### 算子原型

```python
cann_bench.iou_v2(Tensor bboxes, Tensor gtboxes, string mode="iou",
                  float eps=1e-05, bool aligned=False) -> Tensor y
```

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| bboxes | Tensor | 必选 | 预测框 [N, 4]，每行为 (x1, y1, x2, y2) |
| gtboxes | Tensor | 必选 | 真实框 [M, 4]，每行为 (x3, y3, x4, y4)；aligned=True 时 shape 为 [N, 4] |
| mode | string | "iou" | 计算模式，"iou" 表示交并比，"iof" 表示交占前景比 |
| eps | float | 1e-05 | 分母保护项 |
| aligned | bool | false | 是否按行对应计算；True 时输出 [N, 1]，否则输出 [N, M] |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [N, M] 或 [N, 1] | float32 | IoU/IOF 结果 |

### 数据类型

| 输入 (bboxes) dtype | 输入 (gtboxes) dtype | 输出 dtype |
|-------------------|---------------------|-----------|
| float32 | float32 | float32 |

### 规则与约束

- `bboxes` 与 `gtboxes` 形状分别为 `[N, 4]`、`[M, 4]`，最后一维必须为 4
- 每个框需满足 `x1 <= x2`、`y1 <= y2`；`get_input` 会对随机输入排序坐标对，保证良定义
- `aligned=True` 时 `M` 必须等于 `N`，输出 `[N, 1]`
- `mode` 仅支持 `"iou"` 与 `"iof"`，其他取值报错
- `eps` 为分母保护项，通常取 `1e-5`

## 4. 标准 Golden 代码

```python
#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import torch

"""
IouV2 算子 Torch Golden 参考实现

计算预测框与真实框之间的 IoU 或 IOF。
公式:
    inter = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
    area1 = (x2 - x1) * (y2 - y1), area2 = (x4 - x3) * (y4 - y3)
    iou: y = inter / (area1 + area2 - inter + eps)
    iof: y = inter / (area2 + eps)
"""


def iou_v2(
    bboxes: torch.Tensor,
    gtboxes: torch.Tensor,
    mode: str = "iou",
    eps: float = 1e-5,
    aligned: bool = False,
) -> torch.Tensor:
    """
    计算预测框与真实框之间的 IoU 或 IOF

    公式:
        inter = max(0, min(x2, x4) - max(x1, x3)) * max(0, min(y2, y4) - max(y1, y3))
        area1 = (x2 - x1) * (y2 - y1), area2 = (x4 - x3) * (y4 - y3)
        iou: y = inter / (area1 + area2 - inter + eps)
        iof: y = inter / (area2 + eps)

    Args:
        bboxes: 预测框 [N, 4]，每行为 (x1, y1, x2, y2)
        gtboxes: 真实框 [M, 4]，每行为 (x3, y3, x4, y4)；aligned=True 时 shape 为 [N, 4]
        mode: "iou" 或 "iof"
        eps: 分母保护项
        aligned: True 时按行对应计算，输出 [N, 1]；否则输出 [N, M]

    Returns:
        IoU/IOF 结果，shape 为 [N, M]（或 aligned 时 [N, 1]）
    """
    if aligned:
        b1 = bboxes
        b2 = gtboxes
    else:
        b1 = bboxes.unsqueeze(1)            # [N, 1, 4]
        b2 = gtboxes.unsqueeze(0)           # [1, M, 4]

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    inter_w = torch.clamp(torch.min(b1[..., 2], b2[..., 2]) - torch.max(b1[..., 0], b2[..., 0]), min=0.0)
    inter_h = torch.clamp(torch.min(b1[..., 3], b2[..., 3]) - torch.max(b1[..., 1], b2[..., 1]), min=0.0)
    inter = inter_w * inter_h

    if mode == "iou":
        denom = area1 + area2 - inter
    elif mode == "iof":
        denom = area2                      # 前景面积取 gtbox 面积
    else:
        raise ValueError(f"unsupported mode: {mode}")
    out = inter / (denom + eps)
    if aligned:
        out = out.unsqueeze(-1)             # aligned 输出 [N, 1]
    return out


def get_input(
    bboxes: torch.Tensor,
    gtboxes: torch.Tensor,
    mode: str = "iou",
    eps: float = 1e-5,
    aligned: bool = False,
    **kwargs,
):
    """规整测试输入，使 IoU/IOF 良定义且可被真实单算子复现。

    cases.yaml 的 value_range 无法表达“框有序”这一结构化约束，随机取值会产生
    x1>x2 / y1>y2 的退化框（负面积）。这里对每个框独立排序 x 对与 y 对，保证
    x1<=x2、y1<=y2，同时替换 golden 与候选的输入，保证比较公平。

    Returns:
        [bboxes, gtboxes]，顺序与 iou_v2 签名的张量输入顺序一致。
    """

    def _normalize(boxes):
        x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
        return torch.stack([x1, y1, x2, y2], dim=1)

    return [_normalize(bboxes), _normalize(gtboxes)]
```

## 5. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

# 全对配对：bboxes [N,4] 与 gtboxes [M,4] -> 输出 [N,M]
bboxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
gtboxes = torch.tensor([[1.0, 1.0, 2.0, 2.0], [0.0, 0.0, 3.0, 3.0]])
y = cann_bench.iou_v2(bboxes, gtboxes, mode="iou", eps=1e-5, aligned=False)
# y.shape == (2, 2)

# 逐行配对：aligned=True，输出 [N,1]
y = cann_bench.iou_v2(bboxes, bboxes, mode="iof", eps=1e-5, aligned=True)
# y.shape == (2, 1)
```
