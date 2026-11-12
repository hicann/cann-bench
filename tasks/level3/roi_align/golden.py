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

# Portions of _bilinear_interpolate and _roi_align_fallback were adapted
# from torchvision and are licensed under BSD-3-Clause.
# Reference: https://github.com/pytorch/vision/blob/v0.28.0/torchvision/ops/roi_align.py
#
# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch

"""
ROIAlign operator Torch golden reference.

Region-of-interest pooling for feature maps of non-uniform input size.
Formula: y = roi_align(x, boxes, output_size)

The signature matches torch_npu.npu_roi_align: boxes is an (N, 5) tensor whose
column 0 is the batch index (same dtype as the features) and columns 1-4 are the
(x0, y0, x1, y1) pixel coordinates.

Prefers torchvision.ops.roi_align; when torchvision is unavailable it falls back
to a pure-Python implementation mirroring the torchvision reference.
"""

try:
    from torchvision.ops import roi_align as _tv_roi_align

    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False


def _bilinear_interpolate(
    input,
    roi_batch_ind,
    y,
    x,
    ymask,
    xmask,
):
    """Bilinear interpolation, mirroring torchvision's _bilinear_interpolate."""
    _, channels, height, width = input.size()
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)
    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)
    ly = y - y_low
    lx = x - x_low
    hy = 1.0 - ly
    hx = 1.0 - lx

    def masked_index(y_idx, x_idx):
        if ymask is not None:
            assert xmask is not None
            y_idx = torch.where(ymask[:, None, :], y_idx, 0)
            x_idx = torch.where(xmask[:, None, :], x_idx, 0)
        return input[
            roi_batch_ind[:, None, None, None, None, None],
            torch.arange(channels, device=input.device)[
                None, :, None, None, None, None
            ],
            y_idx[:, None, :, None, :, None],
            x_idx[:, None, None, :, None, :],
        ]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)

    def outer_prod(y_t, x_t):
        return y_t[:, None, :, None, :, None] * x_t[:, None, None, :, None, :]

    w1 = outer_prod(hy, hx)
    w2 = outer_prod(hy, lx)
    w3 = outer_prod(ly, hx)
    w4 = outer_prod(ly, lx)
    return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4


def _roi_align_fallback(
    x,
    boxes,
    outputHeight,
    outputWidth,
    spatial_scale,
    sampling_ratio,
    aligned,
):
    """Pure-Python fallback mirroring torchvision.ops.roi_align's reference path."""
    orig_dtype = x.dtype
    # Preserve the evaluator FP64 oracle; low-precision inputs still accumulate in FP32.
    calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    x_calc = x.to(calc_dtype)
    boxes_calc = boxes.to(calc_dtype)
    _, channels, height, width = x_calc.size()
    roi_batch_ind = boxes_calc[:, 0].long()
    offset = 0.5 if aligned else 0.0
    roi_start_w = boxes_calc[:, 1] * spatial_scale - offset
    roi_start_h = boxes_calc[:, 2] * spatial_scale - offset
    roi_end_w = boxes_calc[:, 3] * spatial_scale - offset
    roi_end_h = boxes_calc[:, 4] * spatial_scale - offset
    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = roi_width.clamp(min=1.0)
        roi_height = roi_height.clamp(min=1.0)
    bin_size_h = roi_height / outputHeight
    bin_size_w = roi_width / outputWidth
    exact_sampling = sampling_ratio > 0
    roi_bin_grid_h = (
        sampling_ratio if exact_sampling else torch.ceil(roi_height / outputHeight)
    )
    roi_bin_grid_w = (
        sampling_ratio if exact_sampling else torch.ceil(roi_width / outputWidth)
    )
    if exact_sampling:
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
        iy = torch.arange(roi_bin_grid_h, device=x.device)
        ix = torch.arange(roi_bin_grid_w, device=x.device)
        ymask = None
        xmask = None
    else:
        count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)
        # Crop only masked-out sampling slots; preserve the old H/W limit.
        gh = min(height, max(0, int(roi_bin_grid_h.max().item())))
        gw = min(width, max(0, int(roi_bin_grid_w.max().item())))
        iy = torch.arange(gh, device=x.device)
        ix = torch.arange(gw, device=x.device)
        ymask = iy[None, :] < roi_bin_grid_h[:, None]
        xmask = ix[None, :] < roi_bin_grid_w[:, None]

    def from_K(t):
        return t[:, None, None]

    y = (
        from_K(roi_start_h)
        + torch.arange(outputHeight, device=x.device)[None, :, None]
        * from_K(bin_size_h)
        + (iy[None, None, :] + 0.5).to(x_calc.dtype)
        * from_K(bin_size_h / roi_bin_grid_h)
    )
    x_pos = (
        from_K(roi_start_w)
        + torch.arange(outputWidth, device=x.device)[None, :, None] * from_K(bin_size_w)
        + (ix[None, None, :] + 0.5).to(x_calc.dtype)
        * from_K(bin_size_w / roi_bin_grid_w)
    )

    # Interpolation materializes [K,C,PH,PW,IY,IX]. Crop masked sampling
    # slots and chunk ROIs; reduction order may differ within tolerance.
    # Other live tensors contribute to total memory beyond the val budget.
    n_rois = y.shape[0]
    elems_per_roi = max(
        1, x_calc.shape[1] * y.shape[1] * y.shape[2] * x_pos.shape[1] * x_pos.shape[2]
    )
    # Budget a single val tensor at 32 MiB; this is not a total RSS guarantee.
    chunk = max(1, (32 * 1024 * 1024) // (x_calc.element_size() * elems_per_roi))
    parts = []
    for st in range(0, n_rois, chunk):
        en = st + chunk
        val = _bilinear_interpolate(
            x_calc,
            roi_batch_ind[st:en],
            y[st:en],
            x_pos[st:en],
            None if ymask is None else ymask[st:en],
            None if xmask is None else xmask[st:en],
        )
        if not exact_sampling:
            val = torch.where(ymask[st:en][:, None, None, None, :, None], val, 0)
            val = torch.where(xmask[st:en][:, None, None, None, None, :], val, 0)
        out_chunk = val.sum((-1, -2))
        if isinstance(count, torch.Tensor):
            out_chunk = out_chunk / count[st:en][:, None, None, None]
        else:
            out_chunk = out_chunk / count
        parts.append(out_chunk)
    output = torch.cat(parts, dim=0)

    return output.to(orig_dtype)


def roi_align(
    x: torch.Tensor,
    boxes: torch.Tensor,
    outputHeight: int,
    outputWidth: int,
    spatial_scale: float,
    sampling_ratio: int = -1,
    aligned: bool = False,
) -> torch.Tensor:
    """Region-of-interest pooling for feature maps of non-uniform input size.

    Formula: y = roi_align(x, boxes, output_size)

    Args:
        x: input feature map [B, C, H, W]
        boxes: ROI boxes [N, 5], each row (batch_idx, x0, y0, x1, y1); col 0 is the
            integer batch index (same dtype as the features)
        outputHeight: output height
        outputWidth: output width
        spatial_scale: spatial scale factor (maps box coords to feature resolution)
        sampling_ratio: sampling ratio (-1 or 0 -> auto ceil(roi_h/oh))
        aligned: whether to align (aligned=True shifts ROI coords by -0.5 pixel)

    Returns:
        output tensor [N, C, outputHeight, outputWidth]
    """
    if HAS_TORCHVISION:
        return _tv_roi_align(
            x,
            boxes,
            (outputHeight, outputWidth),
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio,
            aligned=aligned,
        )
    return _roi_align_fallback(
        x,
        boxes,
        outputHeight,
        outputWidth,
        spatial_scale,
        sampling_ratio,
        aligned,
    )


def get_input(
    x: torch.Tensor,
    boxes: torch.Tensor,
    outputHeight: int,
    outputWidth: int,
    spatial_scale: float,
    sampling_ratio: int = -1,
    aligned: bool = False,
    **kwargs,
):
    """Input preprocessing: rebuild legal ROI boxes (feature map x kept as-is).

    A cases.yaml value_range is a single [min, max] interval and cannot express the
    structured boxes (N,5) contract (col0 an integer batch index, col1-4 satisfying
    x1>x0/y1>y0 and landing inside the feature map after scaling). Spread over [-1,1]
    by the generic generator, col0 becomes non-integer/negative and boxes degenerate.
    The golden tolerates this (.long() truncation + clamp) but torch_npu.npu_roi_align
    does not, so its accuracy explodes. get_input regenerates legal boxes here.

    kernel_eval calls this with input names + attrs as keywords and uses the return
    value (a list in golden-signature Tensor order) to replace the inputs of BOTH the
    golden and the candidate, so the comparison stays fair.

    Args:
        x: feature map [B, C, H, W] (returned as-is, including inf/nan stress cases)
        boxes: original boxes [N, 5] (only N/dtype/device used; contents rebuilt)
        spatial_scale: coord->feature-map scale, used to bound the coordinate range

    Returns:
        [x, new_boxes], ordered to match the golden signature's (x, boxes, ...) tensors.
    """
    B, _, H, W = x.shape
    N = boxes.shape[0]
    g = torch.Generator().manual_seed(0)  # deterministic for reproducibility

    # col0: integer batch index in [0, B); fp16 represents it exactly for B <= 4
    batch_idx = torch.randint(0, B, (N,), generator=g).float()

    # col1-4: pixel coords kept non-degenerate and inside the map after *spatial_scale
    max_x = (W - 1) / spatial_scale
    max_y = (H - 1) / spatial_scale
    x0 = torch.rand(N, generator=g) * 0.5 * max_x
    y0 = torch.rand(N, generator=g) * 0.5 * max_y
    x1 = (
        x0 + (0.1 + 0.4 * torch.rand(N, generator=g)) * max_x
    )  # width in (0.1,0.5)*max -> x1>x0, x1<=max
    y1 = y0 + (0.1 + 0.4 * torch.rand(N, generator=g)) * max_y

    new_boxes = torch.stack([batch_idx, x0, y0, x1, y1], dim=1).to(
        boxes.device, boxes.dtype
    )
    return [x, new_boxes]
