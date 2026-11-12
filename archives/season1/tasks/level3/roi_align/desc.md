# ROIAlign 算子 API 描述

## 1. 算子简介

池化层，用于非均匀输入尺寸的特征图。

**主要应用场景**：
- 目标检测中对候选区域（Region of Interest）进行特征提取
- Faster R-CNN、Mask R-CNN 等两阶段检测框架中的特征对齐
- 实例分割中将不同大小的 ROI 映射到固定尺寸的特征表示

**算子特征**：
- 难度等级：L3（FusedComposite）
- 双输入（特征图 x 和 ROI 框 rois）单输出，支持双线性插值模式

## 2. 算子定义

### 数学公式

$$
y = \text{roi\_align}(x, \text{boxes}, \text{output\_size})
$$

对于每个 box，将其映射到输入特征图上的区域（通过 spatial_scale 缩放），然后将该区域划分为 outputHeight x outputWidth 个 bin，在每个 bin 内通过双线性插值采样后进行平均池化，得到固定尺寸的输出。

## 3. 接口规范

### 算子原型

```python
cann_bench.roi_align(Tensor x, Tensor boxes, int outputHeight, int outputWidth,
                     float spatial_scale, int sampling_ratio, bool aligned) -> Tensor y
```

本算子签名与 `torch_npu.npu_roi_align` 一致：`boxes` 为 (numBoxes, 5) 张量，第 0 列为
batch 索引（与 features 同 dtype，整数值），第 1-4 列为 (x0, y0, x1, y1) 像素坐标。

### 输入参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| x | Tensor | 必选 | 输入特征图，shape 为 [B, C, H, W] |
| boxes | Tensor | 必选 | ROI 框，shape 为 [numBoxes, 5]，每行 (batch_idx, x0, y0, x1, y1)；col 0 为整数值的 batch 索引，取值 ∈ [0, B-1] |
| outputHeight | int | 必选 | 输出高度 |
| outputWidth | int | 必选 | 输出宽度 |
| spatial_scale | float | 必选 | 空间缩放因子（用于将 boxes 坐标乘到特征图分辨率） |
| sampling_ratio | int | -1 | 采样比率 (-1 或 0 时自动计算 ceil(roi_h/oh)) |
| aligned | bool | false | 是否对齐 (aligned=True 时 boxes 坐标偏移 -0.5 像素) |

### 输出

| 参数 | Shape | dtype | 描述 |
|------|-------|-------|------|
| y | [numBoxes, C, outputHeight, outputWidth] | 与输入 x 相同 | 输出张量，ROI 对齐结果 |

### 数据类型

| 输入 dtype | 输出 dtype |
|-----------|-----------|
| float32 | float32 |
| float16 | float16 |

### 规则与约束

- 输入特征图 x 的 shape 为 [B, C, H, W]，即 batch、通道、高、宽四维格式
- ROI 框 boxes 的 shape 为 [numBoxes, 5]，每行 (batch_idx, x0, y0, x1, y1)
- x 和 boxes 的 dtype 需一致
- boxes 第 0 列须为整数值的 batch 索引，取值 ∈ [0, B-1]；fp16 下 B 须 ≤ 1024 以保证整数精度
- outputHeight 和 outputWidth 需为正整数
- spatial_scale 用于将 ROI 坐标从原图尺度映射到特征图尺度
- sampling_ratio 为 -1 或 0 时自动计算采样点数

### 支持范围

输入 tensor 各维度与参数的支持范围：

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `B`（batch） | 1 ~ 64 | cases.csv 实测 2 ~ 4；fp16 下要求 `B ≤ 1024` 以保证 `batch_idx` 在 fp16 中能精确表示 |
| `C`（通道） | 1 ~ 2048 | cases.csv 实测 127 ~ 1023 |
| `H`（特征图高） | 8 ~ 512 | cases.csv 实测 31 ~ 128 |
| `W`（特征图宽） | 8 ~ 512 | cases.csv 实测 31 ~ 128 |
| `numBoxes`（ROI 数） | 1 ~ 2048 | cases.csv 实测 50 ~ 256 |
| `boxes.shape[1]` | = 5 | 固定 `(batch_idx, x0, y0, x1, y1)` |
| `outputHeight` | 1 ~ 64 | cases.csv 实测 1 ~ 28 |
| `outputWidth` | 1 ~ 64 | cases.csv 实测 1 ~ 28 |
| `spatial_scale` | (0, 1] | cases.csv 实测 0.0625 / 0.125 / 0.25 / 1.0 |
| `sampling_ratio` | -1 ~ 16 | -1 / 0 表示自动（取 `ceil(roi_h/oh)`）；cases.csv 实测 0 / 1 / 2 / 4 |
| `aligned` | `false` / `true` | cases.csv 全部覆盖 |

约束：
- shape 关系：`x.shape == [B, C, H, W]`、`boxes.shape == [numBoxes, 5]`，输出 `y.shape == [numBoxes, C, outputHeight, outputWidth]`。
- dtype 关系：`x.dtype == boxes.dtype`（必须同为 float16 或 float32）；输出 `y.dtype` 与 `x.dtype` 一致。
- `boxes` 数据约束：第 0 列 `batch_idx` 必须为整数值且 ∈ [0, B-1]；fp16 下 B ≤ 1024，否则 batch_idx 在 fp16 中无法精确表示。
- 几何约束：`(x0, y0, x1, y1)` 是原图像素坐标；乘以 `spatial_scale` 映射到特征图坐标后，结果应落入 `[0, W) × [0, H)`；超出范围的部分由 bilinear 插值的 clamp 行为决定（输出靠近边界值或 0）。
- `outputHeight`、`outputWidth` 必须为正整数；`sampling_ratio` 取 -1 或 0 时自动按 `ceil(roi_h/oh)` 计算采样点数，否则使用指定值（必须为正整数）。

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

> 优先使用 `torchvision.ops.roi_align`，不可用时使用纯 Python fallback。

```python
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

try:
    from torchvision.ops import roi_align as _tv_roi_align
    HAS_TORCHVISION = True
except Exception:
    HAS_TORCHVISION = False


def _bilinear_interpolate(input, roi_batch_ind, y, x, ymask, xmask):
    _, channels, height, width = input.size()
    y = y.clamp(min=0); x = x.clamp(min=0)
    y_low = y.int(); x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)
    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)
    ly = y - y_low; lx = x - x_low; hy = 1.0 - ly; hx = 1.0 - lx
    def masked_index(y_idx, x_idx):
        if ymask is not None:
            y_idx = torch.where(ymask[:, None, :], y_idx, 0)
            x_idx = torch.where(xmask[:, None, :], x_idx, 0)
        return input[roi_batch_ind[:, None, None, None, None, None],
                     torch.arange(channels, device=input.device)[None, :, None, None, None, None],
                     y_idx[:, None, :, None, :, None], x_idx[:, None, None, :, None, :]]
    v1 = masked_index(y_low, x_low); v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low); v4 = masked_index(y_high, x_high)
    def outer_prod(y_t, x_t):
        return y_t[:, None, :, None, :, None] * x_t[:, None, None, :, None, :]
    return outer_prod(hy, hx)*v1 + outer_prod(hy, lx)*v2 + outer_prod(ly, hx)*v3 + outer_prod(ly, lx)*v4


def _roi_align_fallback(
    x, boxes, outputHeight, outputWidth, spatial_scale, sampling_ratio, aligned,
):
    """纯 Python fallback，对标 torchvision.ops.roi_align 的纯 Python 参考实现。"""
    orig_dtype = x.dtype
    x_fp32 = x.float(); boxes_fp32 = boxes.float()
    _, channels, height, width = x_fp32.size()
    roi_batch_ind = boxes_fp32[:, 0].long()
    offset = 0.5 if aligned else 0.0
    roi_start_w = boxes_fp32[:, 1] * spatial_scale - offset
    roi_start_h = boxes_fp32[:, 2] * spatial_scale - offset
    roi_end_w = boxes_fp32[:, 3] * spatial_scale - offset
    roi_end_h = boxes_fp32[:, 4] * spatial_scale - offset
    roi_width = roi_end_w - roi_start_w; roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = roi_width.clamp(min=1.0); roi_height = roi_height.clamp(min=1.0)
    bin_size_h = roi_height / outputHeight; bin_size_w = roi_width / outputWidth
    exact_sampling = sampling_ratio > 0
    roi_bin_grid_h = sampling_ratio if exact_sampling else torch.ceil(roi_height / outputHeight)
    roi_bin_grid_w = sampling_ratio if exact_sampling else torch.ceil(roi_width / outputWidth)
    if exact_sampling:
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
        iy = torch.arange(roi_bin_grid_h, device=x.device); ix = torch.arange(roi_bin_grid_w, device=x.device)
        ymask = None; xmask = None
    else:
        count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)
        iy = torch.arange(height, device=x.device); ix = torch.arange(width, device=x.device)
        ymask = iy[None, :] < roi_bin_grid_h[:, None]; xmask = ix[None, :] < roi_bin_grid_w[:, None]
    def from_K(t): return t[:, None, None]
    y = from_K(roi_start_h) + torch.arange(outputHeight, device=x.device)[None, :, None] * from_K(bin_size_h) + (iy[None, None, :] + 0.5).to(x_fp32.dtype) * from_K(bin_size_h / roi_bin_grid_h)
    x_pos = from_K(roi_start_w) + torch.arange(outputWidth, device=x.device)[None, :, None] * from_K(bin_size_w) + (ix[None, None, :] + 0.5).to(x_fp32.dtype) * from_K(bin_size_w / roi_bin_grid_w)
    val = _bilinear_interpolate(x_fp32, roi_batch_ind, y, x_pos, ymask, xmask)
    if not exact_sampling:
        val = torch.where(ymask[:, None, None, None, :, None], val, 0)
        val = torch.where(xmask[:, None, None, None, None, :], val, 0)
    output = val.sum((-1, -2))
    if isinstance(count, torch.Tensor): output = output / count[:, None, None, None]
    else: output = output / count
    return output.to(orig_dtype)


def roi_align(
    x: torch.Tensor, boxes: torch.Tensor, outputHeight: int, outputWidth: int,
    spatial_scale: float, sampling_ratio: int = -1, aligned: bool = False,
) -> torch.Tensor:
    """
    池化层，用于非均匀输入尺寸的特征图
    公式: y = roi_align(x, boxes, output_size)
    """
    if HAS_TORCHVISION:
        return _tv_roi_align(x, boxes, (outputHeight, outputWidth),
            spatial_scale=spatial_scale, sampling_ratio=sampling_ratio, aligned=aligned)
    return _roi_align_fallback(x, boxes, outputHeight, outputWidth,
        spatial_scale, sampling_ratio, aligned)
```

## 6. 额外信息

### 算子调用示例

```python
import torch
import cann_bench

x = torch.randn(2, 256, 64, 64, dtype=torch.float32, device="npu")
boxes = torch.tensor([[0, 10.0, 10.0, 50.0, 50.0], [1, 20.0, 20.0, 60.0, 60.0]],
                     dtype=torch.float32, device="npu")
y = cann_bench.roi_align(x, boxes, 7, 7, 0.0625, 2, False)
```

### NPU 实现路径与基线测量说明

torch_npu 上对应 API 为
`torch_npu.npu_roi_align(features, boxes, spatial_scale, pooled_height, pooled_width, sample_num, roi_end_mode)`。
`aligned` 与 `roi_end_mode` 的映射遵循官方 `torch_npu/contrib/module/roi_align.py`：`aligned=True → roi_end_mode=3`，
`aligned=False → roi_end_mode=0`。

**`baseline_perf_us` = `torch_npu.npu_roi_align(...)` 单次调用在 device 端启动的全部 kernel 的时长之和（msprof pattern-interval 中位数）。**
不做计时窗口外的预处理（不预转 NC1HWC0、不预拼 rois、不预 Cast），不做后处理过滤。也就是说：包装层的「fusion pass」——
torch_npu binding 调 `aclnnRoiAlign` V1 时由 `libopapi.so` 内部展开的 `Reshape + Cast + ConcatD + (Trans/Slice) + ROIAlign`——
统统计入 baseline。这是第三方调用该 API 时实际看到的成本，不能减。

#### V1 vs V2 背景（reference only，不影响 baseline 值）

`torch_npu.npu_roi_align` 当前绑定到 `aclnnRoiAlign` V1（`libopapi.so::aclnnRoiAlign`）。
反汇编 `libopapi.so` 显示 V1 内部用 `Reshape + Cast (fp16→fp32) + ConcatD` 把用户传入的 (N,5)
拆成 (rois (N,4), batchIndices int64) 再喂 kernel；V2（`aclnnRoiAlignV2`，接受 (N,5) 直通）
未在 `torch_npu` 暴露任何 Python 入口。故 fp16 调用路径上的 `Cast×2` 是 V1 实现的固有开销，
不属于 ROIAlign kernel 的纯计算，但作为本 baseline 的可观测成本被计入。

实测在 op_summary 中观察到的 kernel 链（全部计入 baseline）：

| dtype | C 通道 | kernel 链 |
|---|---|---|
| fp32 | 任意 | `TransData×1 + ROIAlign×1` |
| fp16 | C%16==0 | `Cast×2 + TransData×1 + ROIAlign×1` |
| fp16 | C%16≠0 | `Cast×2 + TransData×3 + ROIAlign×1 + Slice×1` |

`note` 后缀 `| baseline_kernels: <完整 kernel 链>` 明确列出该 case 实际启动的所有 kernel。

### boxes 数据的取值约束（cases.yaml 的语义补充）

`boxes` 第 0 列（batch_idx）必须为整数值且 ∈ [0, B-1]。cases.yaml 的 `value_range` 是均匀随机采样，
直接随机会产生越界。harness 在 `inputs.py:_apply_op_aliases` 把 `boxes[:, 0]` 强制设为 `arange(N) % B`
（确保每个 ROI 指向合法 batch、覆盖均匀）。这是**输入数据合法性整形**，第三方 kernel 在自己实现中
对相同输入约束也需要承认 batch_idx 必须有效——不属于 baseline 计时窗口的工作量。
