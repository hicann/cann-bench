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
