# AscendAntiQuantV2

## 1. 算子简介

AscendAntiQuantV2 是**量化（Quantization）**类算子，难度等级 **L2**。它根据输入的 `scale` 和可选的 `offset` 对量化张量 `x` 进行反量化（dequantize），输出浮点张量 `y`。典型应用场景包括：

- 将 LLM 权重或激活值从低比特量化格式（INT8、INT4、FP8 等）恢复为 FP16/BF16 进行计算；
- 在推理部署阶段对量化模型进行反量化还原。

算子特征：

- 支持 `x` 为 INT8、INT4（打包在 INT32 中）、HIFLOAT8、FLOAT8_E4M3、FLOAT8_E5M2；
- `scale` 与 `offset` 为浮点张量，支持 per-channel 或 per-tensor；
- 通过 `dst_type` 选择输出为 FLOAT16 或 BFLOAT16；
- 通过 `sqrt_mode` 控制 scale 是否平方，但当前 `torch_npu.npu_anti_quant` Python 接口未暴露该参数，因此 cann-bench 用例中仅测试 `sqrt_mode=False`。
- 支持硬件：Ascend 950PR / 950DT

## 2. 算子定义

设输入为 `x`、scale 为 `s`、offset 为 `o`（可为空），输出为 `y`，`dst_type` 指定输出浮点类型。

- `sqrt_mode=False` 且 `offset=None`：

  ```text
  y = cast_to_dst_type(x * s)
  ```

- `sqrt_mode=False` 且 `offset != None`：

  ```text
  y = cast_to_dst_type((x + o) * s)
  ```

- `sqrt_mode=True` 且 `offset=None`：

  ```text
  y = cast_to_dst_type(x * s * s)
  ```

- `sqrt_mode=True` 且 `offset != None`：

  ```text
  y = cast_to_dst_type((x + o) * s * s)
  ```

其中 `cast_to_dst_type(...)` 表示按 `dst_type` 截断为 FLOAT16（`dst_type=1`）或 BFLOAT16（`dst_type=27`）。

## 3. 接口规范

### 算子原型

```text
ascend_anti_quant_v2(Tensor x, Tensor scale, Tensor? offset=None, int dst_type=1, bool sqrt_mode=False) -> Tensor y
```

### 输入参数

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 |
|--------|----------------|------|----------|
| x | 输入 | 待反量化输入；不支持空 Tensor；INT4 时尾轴须为偶数 | INT4、INT8、HIFLOAT8、FLOAT8_E4M3、FLOAT8_E5M2 |
| scale | 输入 | 反量化 scale；维度数与 x 相同或 1 维；最多一个非 1 维且位于 -1/-2 轴 | FLOAT32、BFLOAT16 |
| offset | 可选输入 | 反量化 offset；数据类型和 shape 须与 scale 一致 | FLOAT32、BFLOAT16 |
| dst_type | 属性 | 输出数据类型，1=FLOAT16，27=BFLOAT16 | INT64 |
| sqrt_mode | 属性 | scale 是否平方；当前 torch_npu Python 接口未暴露该参数 | BOOL |
| y | 输出 | 反量化结果，shape 与 x 一致 | FLOAT16、BFLOAT16 |

## 4. 支持范围

- **维度支持**：`x` 支持 1-8 维；`y` 与 `x` 同 shape。
- **scale/offset 广播**：
  - `scale` 维度数与 `x` 相同，或为 1 维；
  - 最多只有一个非 1 维度，且须位于 `x` 的 `-1` 或 `-2` 轴；
  - `offset` 数据类型和 shape 须与 `scale` 保持一致。
- **INT4 打包约束**：
  - `x` 为 INT32 时视为 INT4 打包，每 8 个 int4 值打包在一个 int32 中；
  - 概念上 INT4 张量的尾轴大小须为 8 的倍数，对应 INT32 尾轴大小为原尾轴除以 8；
  - 一维 `scale` 长度须为 `x` 概念尾轴大小的 8 倍（即 INT32 尾轴大小的 8 倍）。
- **dst_type 取值**：仅支持 `1`（FLOAT16）和 `27`（BFLOAT16）。
- **sqrt_mode 取值**：理论上支持 `true` / `false`，但当前 `torch_npu.npu_anti_quant` 未暴露该参数，评测用例均按 `false` 构造。
- **实际 cases 覆盖**：
  - INT8 输入：case 1-10、15-20，覆盖 2D/3D/4D/5D、per-channel/per-tensor、有/无 offset、FLOAT16/BFLOAT16 输出；
  - INT4 打包输入：case 11-14，覆盖 2D/3D/4D、per-channel、有/无 offset、FLOAT16/BFLOAT16 输出；
  - 边界 case：case 9/10/19 覆盖小 batch/单元素/前维广播，case 20 覆盖全零输入。

## 5. Golden 实现

`golden.py` 中 `ascend_anti_quant_v2()` 的 CPU 参考逻辑如下：

1. 若 `x.dtype == torch.int32`，调用 `_unpack_int4()` 将 INT32 打包数据解包为 INT8（每个 int32 含 8 个 int4，按低 nibble 在前解析）。
2. 将 `x` 与 `scale` 提升为 FP32 进行计算；若提供 `offset`，先执行 `x + offset`。
3. 计算 `res = x * scale`；若 `sqrt_mode=True`，再乘一次 `scale`。
4. 按 `dst_type` 将结果转换为 `torch.float16` 或 `torch.bfloat16` 后返回。

`get_input()` 是 `kernel_eval` 框架在生成输入后调用的预处理钩子，用于：

- 将概念上的 INT4 输入（以 int32 生成）通过 `_pack_int4()` 打包为 NPU 期望的 INT32 格式；
- 将 `scale` 取绝对值并 clamp 到最小 `0.001`，保证数值合法性。
