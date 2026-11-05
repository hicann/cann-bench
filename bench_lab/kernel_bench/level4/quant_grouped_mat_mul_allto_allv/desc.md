# QuantGroupedMatMulAlltoAllv 算子 API 描述

## 1. 算子简介

QuantGroupedMatMulAlltoAllv 将 MXFP8 分组矩阵乘、按目标 rank 重排与 AllToAllv 通信融合，支持可选的本地共享专家 MX MatMul。各专家发送多少行由显式计数指定，不执行 Dispatch/Combine 或 TopK 路由流程。

本接口对应 Ascend 950DT（A5、arch35）上的原生 aclnnQuantGroupedMatMulAlltoAllv，选用 FLOAT8_E4M3FN/FLOAT8_E5M2 数据与 E8M0 缩放因子的 MX 子集。原生 per-token/per-channel 模式当前不支持，pertensor 模式使用 HIFLOAT8；这些模式不在本接口范围内。

## 2. 算子定义

设 p=world_size，每个 rank 有 e 个专家，C[s][r][j] 为源 rank s 的本地专家 j 发给目标 rank r 的行数。输入按 [expert][destination] 存储；第 j 组行数为 sum_r C[s][r][j]。

每 32 个 K 轴元素共享一个 E8M0 缩放值。编码字节 c 表示 2^(c-127)，本集合使用有限正常指数。gmm_x_scale 的下标为 [row,floor(k/64),floor((k%64)/32)]。权重 scale 按专家、K 组与输出列分别取值；物理布局随转置属性变化，见下表。

对源 rank s 的专家 j 先将 FP8 输入与权重按对应 K32 scale 精确反量化，再执行矩阵乘。得到的 FP32 结果转换为 out_dtype 后，将 [expert][destination] 重排为 [destination][expert] 并发送。目标 rank r 按 [source][expert] 拼接各段：

$$
Y_r = \operatorname{Concat}_{s=0}^{p-1}\operatorname{Concat}_{j=0}^{e-1}
\left(\operatorname{Cast}_{out\_dtype}(D(X_{s,j,r})D(W_{s,j}))\right)
$$

D 表示按 K32 分组反量化。输出不跨 rank 求和；通信不量化，传输的已是 FP16/BF16 结果。共享分支为 mm_y=Cast(D(mm_x)@D(mm_weight))，仅使用当前 rank 的输入。

## 3. 接口规范

### 算子原型

```python
cann_bench.quant_grouped_mat_mul_allto_allv(Tensor gmm_x, Tensor gmm_weight, Tensor gmm_x_scale, Tensor gmm_weight_scale, str hcomm_info, int world_size, int[] send_counts, int[] recv_counts, Tensor? mm_x=None, Tensor? mm_weight=None, Tensor? mm_x_scale=None, Tensor? mm_weight_scale=None, bool trans_gmm_weight=False, bool trans_mm_weight=False, str out_dtype="bfloat16", int gmm_x_quant_mode=6, int gmm_weight_quant_mode=6, int mm_x_quant_mode=0, int mm_weight_quant_mode=0, int comm_quant_mode=0, int group_size=4295032864, int scale_dtype=293) -> (Tensor y, Tensor? mm_y)
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `gmm_x` | FP8 Tensor，[A,K] | 连续输入，按本地专家、目标 rank 排列。 |
| `gmm_weight` | FP8 Tensor，[e,K,N] 或 [e,N,K] | 连续物理权重，后者用于 trans_gmm_weight=True。 |
| `gmm_x_scale` | uint8 Tensor，[A,ceil(K/64),2] | E8M0 指数编码，每个末维元素对应 32 个 K 元素。 |
| `gmm_weight_scale` | uint8 Tensor | 非转置为 [e,ceil(K/64),N,2]；转置为 [e,N,ceil(K/64),2]。 |
| `hcomm_info` | str | 已初始化的 HCCL 通信域名称。 |
| `world_size` | int | 本集合使用双卡，必须与实际通信域一致。 |
| `send_counts` | int[]，长度 p*e | C[当前rank] 按 [destination][expert] 展平，元素为直接行计数。 |
| `recv_counts` | int[]，长度 p*e | C[source][当前rank] 按 [source][expert] 展平。 |
| `mm_x` | 可选 FP8 Tensor，[BS,H2] | 共享输入，dtype 与 gmm_x 相同。 |
| `mm_weight` | 可选 FP8 Tensor，[H2,N2] 或 [N2,H2] | 共享权重，dtype 与 gmm_weight 相同。 |
| `mm_x_scale` | 可选 uint8 Tensor，[BS,ceil(H2/64),2] | 共享输入的 E8M0 scale。 |
| `mm_weight_scale` | 可选 uint8 Tensor | 非转置为 [ceil(H2/64),N2,2]；转置为 [N2,ceil(H2/64),2]。 |
| `trans_gmm_weight` | bool | 默认 False；同时决定专家权重和权重 scale 的物理布局。 |
| `trans_mm_weight` | bool | 默认 False；同时决定共享权重和权重 scale 的布局。 |
| `out_dtype` | str | float16 或 bfloat16，默认 bfloat16；同时用于 y 和 mm_y。 |
| `gmm_x_quant_mode` | int | 固定为 6，表示 MX。 |
| `gmm_weight_quant_mode` | int | 固定为 6。 |
| `mm_x_quant_mode` | int | 无共享分支时为 0，启用共享分支时必须为 6。 |
| `mm_weight_quant_mode` | int | 与 mm_x_quant_mode 相同。 |
| `comm_quant_mode` | int | 固定为 0，不压缩通信结果。 |
| `group_size` | int | 固定 4295032864，即 32 + (1<<16) + (1<<32)，表示 [1,1,32]。 |
| `scale_dtype` | int | 固定 293，将全部 uint8 scale 存储解释为 FLOAT8_E8M0。 |

### 输出

始终返回二元组 (y,mm_y)。y 为 [sum(recv_counts),N]，按源 rank、本地源专家顺序排列，dtype 为 out_dtype。共享分支开启时 mm_y 为 [BS,N2]，dtype 与 y 相同；否则 mm_y 必须为 None。

### 规则与约束

- 仅适用于 Ascend 950DT 的双卡 MXFP8 子集。gmm_x/gmm_weight 可分别为 E4M3FN 或 E5M2，也可混用；不接受 HIFLOAT8、FP4 或 INT8。
- 输入为连续 ND 物理张量，权重转置由属性表达。权重与 scale 必须同时使用对应布局；不能仅转置权重而保留旧 scale 轴。
- 1<=e<=32、p*e<=256；0<A,sum(recv_counts)<52428800；0<K,N<65536。每条通信边计数非负且发送端、接收端完全匹配，允许零计数。本集合保持每个 rank 总发送和总接收行数一致。
- shared 的四个张量必须同时存在或同时为 None。开启时 0<H2<=12288、0<N2<65536，sum(recv_counts) 能被 BS 整除，且商在 [2,8]；两分支使用同一量化模式和输出类型。
- E8M0 字节是指数编码，不是数值 scale。scale_dtype=293 是接口适配属性；调用原生 ACLNN 时须将相应存储描述为 ACL_FLOAT8_E8M0，不能当作普通 UINT8 数值转换。
- 原生接口的 sendCountsTensorOptional、recvCountsTensorOptional、commQuantScaleOptional 在本子集中均为空；commQuantDtypeOptional 使用未指定值，commQuantMode=0。基础 ACLNN 接口采用 AI_CPU 通信。
- scale 布局以源码 arch35 的 mx_quant_grouped_mat_mul_allto_allv_tiling.cpp 中 CheckMxQuantGmmScaleShapes 与 CheckMxQuantMmScaleShapes 为准。README 关于转置 scale 的部分文字与这两处检查相反，本接口遵循实际校验。
- 来源：ops-transformer/mc2/quant_grouped_mat_mul_allto_allv/op_api/aclnn_quant_grouped_mat_mul_allto_allv.cpp 和 op_host/op_tiling/arch35/mx_quant_grouped_mat_mul_allto_allv_tiling.cpp。

## 4. 精度要求

所有 rank 的输出均须满足形状、dtype 和可选输出契约。启用三方比较时，以相同输入的 CPU FP32 结果为参考：候选的大值域最大误差不超过设备级联的 10 倍，平均误差、误差总和及全量 RMSE 不超过级联的 2 倍。

另允许有限值逐元素满足 `abs(candidate-golden) <= atol + rtol*abs(golden)`，避免零级联误差退化为逐位相等要求。有效比较类型 FP16/BF16/FP32 的 `rtol` 分别为 `2^-10`、`2^-7`、`2^-23`，`atol` 分别为 `1e-5`、`1e-5`、`1e-9`。显式 FP32 比较和 FP8 辅助输出使用 FP32 容差，不能按输入 FP8 类型放宽。

FP16/BF16 的小值域分界为 `1e-3`，FP32 为 `1e-6`。小值元素不少于 100000 个时，候选的小值超差数不得超过级联的 2 倍；计数边界及绝对误差阈值保持不变。结构、NaN/正负无穷掩码及有限误差统计检查始终有效，容差分支不能绕过这些检查。

级联参考先接受按真实输出类型的独立固定精度校验；参考失效时评分不可用，不归为候选编译错误。未启用三方模式的历史用例保持原相对误差流程。完整判定与比较类型规则见[三方精度指南](../../../../docs/guide/mc2_accuracy.md)。

## 5. 标准 Golden 代码

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


"""Ascend 950DT MXFP8 grouped matmul, unpermute and AlltoAllv."""

import torch


def _dtype(name):
    return {
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float8_e5m2": torch.float8_e5m2,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "uint8": torch.uint8,
    }[name]


def _layout(ctx, attrs):
    p, rank, e = int(ctx["world_size"]), int(ctx["rank"]), int(attrs["e"])
    counts = attrs["token_counts"]
    if p != 2 or attrs["world_size"] != p or not 0 <= rank < p or not 1 <= e <= 32:
        raise ValueError(
            "This A5 MXFP8 suite requires two ranks and 1..32 local experts"
        )
    if len(counts) != p or any(len(row) != p for row in counts):
        raise ValueError("token_counts must be C[source][destination][expert]")
    if any(
        len(edge) != e or any(type(v) is not int or v < 0 for v in edge)
        for row in counts
        for edge in row
    ):
        raise ValueError("token_counts must contain nonnegative integer counts")
    send = [v for edge in counts[rank] for v in edge]
    recv = [v for source in range(p) for v in counts[source][rank]]
    if not 0 < sum(send) < 52428800 or not 0 < sum(recv) < 52428800:
        raise ValueError("Each rank must send and receive nonempty token tensors")
    return p, rank, e, counts, send, recv


def mc2_make_rank_inputs(ctx, case_payload):
    attrs = case_payload["attrs"]
    shapes, dtypes, ranges = (
        case_payload["input_shapes"],
        case_payload["dtypes"],
        case_payload["value_ranges"],
    )
    _, rank, _, _, send, recv = _layout(ctx, attrs)
    names = (
        "gmm_x",
        "gmm_weight",
        "send_counts",
        "recv_counts",
        "mm_x",
        "mm_weight",
        "gmm_x_scale",
        "gmm_weight_scale",
        "mm_x_scale",
        "mm_weight_scale",
    )
    result = {"send_counts": send, "recv_counts": recv}
    for index, name in enumerate(names):
        if index in (2, 3):
            continue
        shape = shapes[index]
        if shape is None:
            result[name] = None
            continue
        low, high = ranges[index]
        seed = int(attrs["seed"]) + rank * 1000003 + index * 1009
        if name.endswith("_scale"):
            if not 1 <= low <= high <= 254:
                raise ValueError("Scale codes must be finite normal E8M0 exponents")
            count = 1
            for dimension in shape:
                count *= dimension
            codes = torch.arange(count, dtype=torch.int64)
            value = (
                (int(low) + (codes + seed) % (int(high) - int(low) + 1))
                .reshape(shape)
                .to(torch.uint8)
            )
        else:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            value = (
                torch.empty(shape, dtype=torch.float32)
                .uniform_(low, high, generator=generator)
                .to(_dtype(dtypes[index]))
            )
        result[name] = value.to(ctx["device"])
    if result["gmm_x"].shape[0] != sum(send):
        raise ValueError("gmm_x row count must equal sum(send_counts)")
    return result


def _mx_rows(values, codes):
    """Values are [...,rows,K]; E8M0 codes are [...,rows,ceil(K/64),2]."""
    k = values.shape[-1]
    if codes.dtype != torch.uint8 or tuple(codes.shape) != tuple(values.shape[:-1]) + (
        (k + 63) // 64,
        2,
    ):
        raise ValueError("MX row scales must be uint8 [...,rows,ceil(K/64),2]")
    scales = (
        torch.exp2(codes.float() - 127.0)
        .flatten(-2)
        .repeat_interleave(32, dim=-1)[..., :k]
    )
    return values.float() * scales


def _mx_weight(weight, scale, transposed):
    # Physical NT: [e,K,N], [e,ceil(K/64),N,2]; T: [e,N,K], [e,N,ceil(K/64),2].
    # The same last-axis rule handles the optional 2D shared weight.
    rows = weight if transposed else weight.transpose(-2, -1)
    codes = scale if transposed else scale.transpose(-3, -2)
    return _mx_rows(rows, codes).transpose(-2, -1)


def _reference(ctx, inputs, attrs, round_output):
    p, rank, e, counts, send, recv = _layout(ctx, attrs)
    if list(inputs["send_counts"]) != send or list(inputs["recv_counts"]) != recv:
        raise ValueError("Local counts must match C[source][destination][expert]")
    if attrs["gmm_x_quant_mode"] != 6 or attrs["gmm_weight_quant_mode"] != 6:
        raise ValueError("This contract supports MX quant mode 6 only")
    if attrs["group_size"] != 4295032864 or attrs["comm_quant_mode"] != 0:
        raise ValueError("MX groups are [1,1,32] and communication is not quantized")
    x = _mx_rows(inputs["gmm_x"], inputs["gmm_x_scale"])
    weight = _mx_weight(
        inputs["gmm_weight"], inputs["gmm_weight_scale"], attrs["trans_gmm_weight"]
    )
    output_dtype = _dtype(attrs["out_dtype"]) if round_output else torch.float32
    by_expert, offset = [], 0
    for expert in range(e):
        rows = sum(counts[rank][destination][expert] for destination in range(p))
        product = torch.matmul(x[offset : offset + rows], weight[expert]).to(
            output_dtype
        )
        by_expert.append(
            list(
                product.split(
                    [counts[rank][destination][expert] for destination in range(p)],
                    dim=0,
                )
            )
        )
        offset += rows
    sent = torch.cat(
        [
            by_expert[expert][destination]
            for destination in range(p)
            for expert in range(e)
        ],
        dim=0,
    ).contiguous()
    y = torch.empty((sum(recv), weight.shape[-1]), dtype=output_dtype, device=x.device)
    ctx["dist"].all_to_all_single(
        y,
        input=sent,
        input_split_sizes=[sum(edge) for edge in counts[rank]],
        output_split_sizes=[sum(counts[source][rank]) for source in range(p)],
    )
    mm_y = None
    if attrs["mm_out"]:
        mm_x = _mx_rows(inputs["mm_x"], inputs["mm_x_scale"])
        mm_weight = _mx_weight(
            inputs["mm_weight"], inputs["mm_weight_scale"], attrs["trans_mm_weight"]
        )
        mm_y = torch.matmul(mm_x, mm_weight).to(output_dtype)
    return y, mm_y


def mc2_distributed_golden(ctx, inputs, attrs):
    """NPU cascade: exact MX dequantization, FP32 matmul, then output rounding before communication."""
    return _reference(ctx, inputs, attrs, round_output=True)


def mc2_cpu_golden(ctx, inputs, attrs):
    """CPU reference communicates FP32 results and does not round the outputs."""
    return _reference(ctx, inputs, attrs, round_output=False)


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    return candidate(
        inputs["gmm_x"],
        inputs["gmm_weight"],
        inputs["gmm_x_scale"],
        inputs["gmm_weight_scale"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        inputs["send_counts"],
        inputs["recv_counts"],
        mm_x=inputs.get("mm_x"),
        mm_weight=inputs.get("mm_weight"),
        mm_x_scale=inputs.get("mm_x_scale"),
        mm_weight_scale=inputs.get("mm_weight_scale"),
        trans_gmm_weight=attrs["trans_gmm_weight"],
        trans_mm_weight=attrs["trans_mm_weight"],
        out_dtype=attrs["out_dtype"],
        gmm_x_quant_mode=attrs["gmm_x_quant_mode"],
        gmm_weight_quant_mode=attrs["gmm_weight_quant_mode"],
        mm_x_quant_mode=attrs["mm_x_quant_mode"],
        mm_weight_quant_mode=attrs["mm_weight_quant_mode"],
        comm_quant_mode=attrs["comm_quant_mode"],
        group_size=attrs["group_size"],
        scale_dtype=attrs["scale_dtype"],
    )
```

## 6. 算子调用示例

假设 Ascend 950DT 的 device 和双卡通信域已初始化。各 rank 的每个专家各向两个目标发送 4 行，输入按 [expert][destination] 排列。

```python
import torch
import cann_bench

x = torch.randn(16, 128, device=device).to(torch.float8_e4m3fn)
w = torch.randn(2, 128, 64, device=device).to(torch.float8_e5m2)
xs = torch.full((16, 2, 2), 126, dtype=torch.uint8, device=device)
ws = torch.full((2, 2, 64, 2), 127, dtype=torch.uint8, device=device)
y, mm_y = cann_bench.quant_grouped_mat_mul_allto_allv(
    x, w, xs, ws, hcomm_info, world_size, [4, 4, 4, 4], [4, 4, 4, 4],
    out_dtype="bfloat16", gmm_x_quant_mode=6, gmm_weight_quant_mode=6,
    group_size=4295032864, scale_dtype=293,
)
# y: [16,64] BF16；mm_y: None
```
