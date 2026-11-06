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
CropAndResize 算子 Torch Golden 参考实现

按归一化框 [y1, x1, y2, x2] 从 NHWC 输入中裁剪并 resize 到 crop_size。
公式:
    in_y = y1 * (H - 1) + py * (y2 - y1) * (H - 1) / (crop_h - 1)
    in_x = x1 * (W - 1) + px * (x2 - x1) * (W - 1) / (crop_w - 1)
    y = bilinear(x[b], in_y, in_x)  (越界位置取 extrapolation_value)
"""


def crop_and_resize(
    x: torch.Tensor,
    boxes: torch.Tensor,
    box_index: torch.Tensor,
    crop_size: list,
    extrapolation_value: float = 0.0,
    method: str = "bilinear",
) -> torch.Tensor:
    """
    按归一化框裁剪并 resize

    公式:
        in_y = y1 * (H - 1) + py * (y2 - y1) * (H - 1) / (crop_h - 1)
        in_x = x1 * (W - 1) + px * (x2 - x1) * (W - 1) / (crop_w - 1)
        y = bilinear(x[b], in_y, in_x)  (越界位置取 extrapolation_value)

    Args:
        x: 输入特征图 [N, H, W, C]（NHWC）
        boxes: 归一化裁剪框 [num_boxes, 4]，每行为 (y1, x1, y2, x2)
        box_index: 每个框对应的 batch 索引 [num_boxes]
        crop_size: 裁剪输出尺寸 [crop_h, crop_w]
        extrapolation_value: 采样位置越界时使用的外插值
        method: 插值方式，"bilinear" / "nearest" / "bilinear_v2"

    Returns:
        输出张量 [num_boxes, crop_h, crop_w, C]，dtype 与 boxes 一致
    """
    return _crop_and_resize(x, boxes, box_index, crop_size,
                            extrapolation_value, method)


def _crop_and_resize(x, boxes, box_index, crop_size,
                     extrapolation_value, method):
    n, h, w, c = x.shape
    num_boxes = boxes.shape[0]
    crop_h = int(crop_size[0])
    crop_w = int(crop_size[1])
    device = x.device

    xb = x[box_index]                                   # [num_boxes, H, W, C]
    xb = xb.to(torch.float32)                           # 中间一律 float32 保精度
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    py = torch.arange(crop_h, device=device, dtype=torch.float32)
    px = torch.arange(crop_w, device=device, dtype=torch.float32)

    if method in ("bilinear", "nearest"):
        if crop_h > 1:
            height_scale = (y2 - y1) * (h - 1) / (crop_h - 1)
            in_y = (y1 * (h - 1))[:, None] + py[None, :] * height_scale[:, None]
        else:
            in_y = 0.5 * (y1 + y2) * (h - 1)
            in_y = in_y[:, None].expand(num_boxes, crop_h)
        if crop_w > 1:
            width_scale = (x2 - x1) * (w - 1) / (crop_w - 1)
            in_x = (x1 * (w - 1))[:, None] + px[None, :] * width_scale[:, None]
        else:
            in_x = 0.5 * (x1 + x2) * (w - 1)
            in_x = in_x[:, None].expand(num_boxes, crop_w)

        valid = ((in_y[:, :, None] >= 0) & (in_y[:, :, None] <= h - 1)
                 & (in_x[:, None, :] >= 0) & (in_x[:, None, :] <= w - 1))

        if method == "bilinear":
            top = torch.floor(in_y)
            bottom = torch.ceil(in_y)
            left = torch.floor(in_x)
            right = torch.ceil(in_x)
            y_ratio = in_y - top
            x_ratio = in_x - left
            top_c = top.clamp(0, h - 1).long()[:, :, None]
            bottom_c = bottom.clamp(0, h - 1).long()[:, :, None]
            left_c = left.clamp(0, w - 1).long()[:, None, :]
            right_c = right.clamp(0, w - 1).long()[:, None, :]
            kk = torch.arange(num_boxes, device=device)[:, None, None]
            v_tl = xb[kk, top_c, left_c]
            v_tr = xb[kk, top_c, right_c]
            v_bl = xb[kk, bottom_c, left_c]
            v_br = xb[kk, bottom_c, right_c]
            wy = y_ratio[:, :, None]
            wx = x_ratio[:, None, :]
            val = (((1.0 - wy) * (1.0 - wx)).unsqueeze(-1) * v_tl
                   + ((1.0 - wy) * wx).unsqueeze(-1) * v_tr
                   + (wy * (1.0 - wx)).unsqueeze(-1) * v_bl
                   + (wy * wx).unsqueeze(-1) * v_br)
        else:  # nearest：四舍五入 = floor(v + 0.5)
            iy = torch.floor(in_y + 0.5).clamp(0, h - 1).long()[:, :, None]
            ix = torch.floor(in_x + 0.5).clamp(0, w - 1).long()[:, None, :]
            kk = torch.arange(num_boxes, device=device)[:, None, None]
            val = xb[kk, iy, ix]                        # [num_boxes, crop_h, crop_w, C]

        out = torch.where(valid.unsqueeze(-1), val,
                          torch.full_like(val, extrapolation_value))
    elif method == "bilinear_v2":
        y1o = torch.floor(y1 * h).long()
        x1o = torch.floor(x1 * w).long()
        y2o = torch.floor(y2 * h).long()
        x2o = torch.floor(x2 * w).long()
        ch = (y2o - y1o + 1).clamp(min=1).float()       # [num_boxes]
        cw = (x2o - x1o + 1).clamp(min=1).float()
        r_y = (py[None, :] + 0.5) * ch[:, None] / crop_h - 0.5
        r_x = (px[None, :] + 0.5) * cw[:, None] / crop_w - 0.5
        zero_h = torch.zeros_like(ch[:, None] - 1)
        zero_w = torch.zeros_like(cw[:, None] - 1)
        lower_y = torch.floor(r_y).clamp(zero_h, ch[:, None] - 1).long()
        upper_y = torch.ceil(r_y).clamp(zero_h, ch[:, None] - 1).long()
        lower_x = torch.floor(r_x).clamp(zero_w, cw[:, None] - 1).long()
        upper_x = torch.ceil(r_x).clamp(zero_w, cw[:, None] - 1).long()
        y_ratio = (r_y - lower_y.float())[:, :, None]
        x_ratio = (r_x - lower_x.float())[:, None, :]
        gy0 = (y1o[:, None] + lower_y).clamp(0, h - 1)[:, :, None]
        gy1 = (y1o[:, None] + upper_y).clamp(0, h - 1)[:, :, None]
        gx0 = (x1o[:, None] + lower_x).clamp(0, w - 1)[:, None, :]
        gx1 = (x1o[:, None] + upper_x).clamp(0, w - 1)[:, None, :]
        kk = torch.arange(num_boxes, device=device)[:, None, None]
        v_tl = xb[kk, gy0, gx0]
        v_tr = xb[kk, gy0, gx1]
        v_bl = xb[kk, gy1, gx0]
        v_br = xb[kk, gy1, gx1]
        out = (((1.0 - y_ratio) * (1.0 - x_ratio)).unsqueeze(-1) * v_tl
               + ((1.0 - y_ratio) * x_ratio).unsqueeze(-1) * v_tr
               + (y_ratio * (1.0 - x_ratio)).unsqueeze(-1) * v_bl
               + (y_ratio * x_ratio).unsqueeze(-1) * v_br)
    else:
        raise ValueError(f"unsupported method: {method}")

    return out.to(boxes.dtype)


def get_input(
    x: torch.Tensor,
    boxes: torch.Tensor,
    box_index: torch.Tensor,
    crop_size: list,
    extrapolation_value: float = 0.0,
    method: str = "bilinear",
    **kwargs,
):
    """规整测试输入，使裁剪框良定义且可被真实单算子复现。

    cases.yaml 的 value_range 无法表达“框有序且落在 [0,1]”这一结构化约束。
    这里重建成归一化合法框（y1<y2、x1<x2，坐标位于 [0,1]）以及界内的
    batch 索引，同时替换 golden 与候选的输入，保证比较公平。

    Returns:
        [x, boxes, box_index]，顺序与 crop_and_resize 签名的张量输入顺序一致。
    """
    num_boxes = boxes.shape[0]
    batch = x.shape[0]
    g = torch.Generator().manual_seed(0)  # 确定性，跨 eval 复现

    y1 = torch.rand(num_boxes, 1, generator=g) * 0.7
    y2 = y1 + torch.rand(num_boxes, 1, generator=g) * 0.3
    x1 = torch.rand(num_boxes, 1, generator=g) * 0.7
    x2 = x1 + torch.rand(num_boxes, 1, generator=g) * 0.3
    new_boxes = torch.cat([y1, x1, y2, x2], dim=1).to(boxes.device, boxes.dtype)

    new_box_index = torch.randint(0, batch, (num_boxes,), generator=g).to(
        box_index.device, box_index.dtype
    )
    return [x, new_boxes, new_box_index]
