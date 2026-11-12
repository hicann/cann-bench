#!/usr/bin/python3
# coding=utf-8

import torch


def image_resize_normalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    out_h: int = 224,
    out_w: int = 224,
    align_corners: bool = False,
) -> torch.Tensor:
    if x.dim() != 4:
        raise ValueError("x must be [B,C,H,W]")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("out_h/out_w must be positive")
    channels = x.shape[1]
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError("mean/std must be [C]")
    if (
        not torch.isfinite(x).all()
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
    ):
        raise ValueError("x/mean/std must contain finite values")
    if torch.any(std <= 0):
        raise ValueError("std must be strictly positive")
    out_dtype = x.dtype
    # The FP64 oracle must preserve feature, mean and scale precision.
    calc_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    resized = torch.nn.functional.interpolate(
        x.to(calc_dtype),
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=align_corners,
    )
    denom = std.to(calc_dtype).reshape(1, channels, 1, 1)
    y = (resized - mean.to(calc_dtype).reshape(1, channels, 1, 1)) / denom
    return y.to(out_dtype)
