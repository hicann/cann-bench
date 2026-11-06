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
DeformableOffsets 算子 Torch Golden 参考实现

偏移量驱动的图像采样：把 offsets 中的元素视为输入图像按行展开 [H*W*C] 的线性索引，
从 images[b] 连续取出 C 个通道值。
公式:
    idx = offsets[b, p, h, w]
    y[b, p, h, w, c] = images.reshape(B, H * W * C)[b, idx + c], c = 0..C-1
"""


def deformable_offsets(
    images: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """
    偏移量驱动的图像采样

    公式:
        idx = offsets[b, p, h, w]
        y[b, p, h, w, c] = images.reshape(B, H * W * C)[b, idx + c], c = 0..C-1

    Args:
        images: 输入图像 [B, H, W, C]（NHWC）
        offsets: 采样偏移 [B, P, new_h, new_w]，每个元素为 images[b] 展平 [H*W*C] 的线性索引

    Returns:
        输出张量 [B, P, new_h, new_w, C]，dtype 与 images 一致
    """
    b, h, w, ch = images.shape
    _, points, new_h, new_w = offsets.shape
    device = images.device

    images_flat = images.reshape(b, h * w * ch)             # [B, H*W*C]
    idx = offsets.to(torch.int64).reshape(b, -1)            # [B, P*new_h*new_w]
    # 每个偏移指向一个线性位置，连续读 ch 个通道
    warp_parts = [
        torch.gather(images_flat, 1, idx + c) for c in range(ch)
    ]
    warp_flat = torch.stack(warp_parts, dim=-1)             # [B, M, C]
    return warp_flat.reshape(b, points, new_h, new_w, ch)


def get_input(
    images: torch.Tensor,
    offsets: torch.Tensor,
    **kwargs,
):
    """规整 offsets 到合法线性索引范围，使采样不越界。

    offsets 必须是 [0, H*W*C - C] 内的整数，才能保证 idx + c (c < C) 不越过
    images[b] 的展平长度 H*W*C。这里对随机生成的 offsets 做 clamp，同时替换
    golden 与候选的输入，保证比较公平。

    Returns:
        [images, offsets]，顺序与 deformable_offsets 签名的张量输入顺序一致。
    """
    b, h, w, ch = images.shape
    max_index = h * w * ch - ch
    offsets = offsets.clamp(min=0, max=max_index)
    return [images, offsets]
