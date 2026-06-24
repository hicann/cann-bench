# ResizeNearestNeighborV2Grad

## 算子简介

ResizeNearestNeighborV2Grad 是 [ResizeNearestNeighborV2](../resize_nearest_neighbor_v2/README.md) 的反向传播算子，属于图像处理类（Image）算子，难度等级 L2。它根据最近邻插值的坐标映射关系，将输出空间上的梯度散射回原始输入空间并累加，广泛应用于语义分割、目标检测等需要可学习上/下采样的网络中。

- 支持硬件：Ascend 950PR / 950DT
本评测集仅支持 `align_corners=false`：PyTorch aten 的 `upsample_nearest2d_backward` / `_upsample_nearest_exact2d_backward` 接口本身无 `align_corners` 语义，为避免 golden 与真实 NPU 语义不一致，评测集中不再声明 `align_corners=true` 的支持。

## 算子定义

### 正向映射回顾

正向 ResizeNearestNeighborV2 将输入图像从空间尺寸 `(H_in, W_in)` 缩放至 `(H_out, W_out)`，每个输出像素复制自某个输入像素：

```
scaleH = H_in / H_out
scaleW = W_in / W_out
```

默认情况（`align_corners=false, half_pixel_centers=false`）：

```
h_src = min(floor(h_out * scaleH), H_in - 1)
w_src = min(floor(w_out * scaleW), W_in - 1)
```

### 反向传播

反向时，输出梯度 `grads(N, C, H_out, W_out)` 按照正向的最近邻映射反向散射：

```
y(N, C, h_src, w_src) += grads(N, C, h_out, w_out)
```

其中 `(h_src, w_src)` 由上述映射公式计算。

### 不同属性组合下的映射

| half_pixel_centers | h_src 计算方式 |
|--------------------|----------------|
| false              | `min(floor(h_out * scaleH), H_in - 1)` |
| true               | `min(floor((h_out + 0.5) * scaleH), H_in - 1)` |

## 接口规范

### 算子原型

```
resize_nearest_neighbor_v2_grad(Tensor grads, Tensor size, bool half_pixel_centers=false, float[] scales=[0.0, 0.0]) -> Tensor y
```

### 输入

| 名称 | 类型 | 描述 |
|------|------|------|
| grads | float16/float32/bfloat16 | 正向输出端梯度，4D，NCHW 或 NHWC。 |
| size | int32 | 原始输入图像尺寸，1D，长度为 2，`[H_in, W_in]`。 |

### 输出

| 名称 | 类型 | 描述 |
|------|------|------|
| y | float16/float32/bfloat16 | 正向输入端梯度，4D，与 grads 同格式、同 dtype。 |

### 属性

| 名称 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| half_pixel_centers | bool | false | 是否将像素中心置于半像素坐标。 |
| scales | list_float | [0.0, 0.0] | 空间尺寸乘数 `[scaleH, scaleW]`，为 0 时由 size 与 grads 尺寸推导。 |

## 支持范围

- **维度**：仅支持 4D 张量。
- **数据格式**：NCHW（本评测集默认）或 NHWC。
- **数据类型**：float16、float32、bfloat16。
- **尺寸约束**：`H_out` 与 `W_out` 由 `grads` 的后两维确定；`H_in` 与 `W_in` 由 `size` 确定。
- **属性约束**：本评测集不支持 `align_corners=true`，仅按 `align_corners=false` 计算与评测。

## Golden 实现

基于 PyTorch aten 接口（两者均只对应 `align_corners=false` 的最近邻反向语义）：

- `half_pixel_centers=false`： `torch.ops.aten.upsample_nearest2d_backward`
- `half_pixel_centers=true`： `torch.ops.aten._upsample_nearest_exact2d_backward`

实现要点：

1. 将 `size` 转换为 `[N, C, H_in, W_in]` 的 4D `input_size`。
2. 当输入输出尺寸相同时直接返回 `grads`。
3. 低精度（fp16/bf16）输入在 fp32 下计算，再 cast 回原 dtype。
4. `get_input` 预处理函数负责将 cases.yaml 中的占位 size 替换为 attrs 中真实的 `input_size`。
5. 本实现不提供 `align_corners` 参数，统一按 `align_corners=false` 计算。

## 评测说明

本评测集包含 20 个 case，覆盖：

- 上采样（2x、3x）、下采样（0.5x、4x）、等尺寸
- 对齐/非对齐、对称/非对称尺寸
- `half_pixel_centers` false/true 两种属性
- float16、float32、bfloat16 三种数据类型
- 极小图、大图等边界场景

> 注：原 case 3/12/18/20 曾设置 `align_corners=true`，但 golden 实际按 `align_corners=false` 计算，为避免与文档/NPU 语义冲突，已将它们改为 `align_corners=false`，并从 proto/desc 中移除 `align_corners` 属性声明。

性能基线通过 `torch.ops.aten.upsample_nearest2d_backward` / `_upsample_nearest_exact2d_backward` 在 NPU 上实测获得。
