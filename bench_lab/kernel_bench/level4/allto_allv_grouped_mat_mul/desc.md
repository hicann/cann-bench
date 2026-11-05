# AlltoAllvGroupedMatMul 算子 API 描述

## 1. 算子简介

AlltoAllvGroupedMatMul 将可变长度 AllToAllv 通信、专家维重排和分组矩阵乘融合执行。每个 rank 将 token 按目标 rank 和专家发送；接收端把源 rank 优先的数据重排为专家优先布局，再使用本地专家权重计算。可同时执行独立的共享专家矩阵乘，并可返回重排后的中间输入。

计算语义对应 `aclnnAlltoAllvGroupedMatMul`，以下定义 Ascend 950DT（A5、arch35）上的 FP16/BF16 接口。

## 2. 算子定义

设 `p=world_size`，每个 rank 有 `e` 个专家，源 rank 为 `s`，目标 rank 为 `r`。令 `C[s][r][j]` 表示源 rank `s` 发给目标 rank `r` 上专家 `j` 的 token 行数。

源输入 `X_s` 的布局为 `[destination rank][local expert][token]`：先按目标 rank 从 0 到 `p-1` 排列，再按专家从 0 到 `e-1` 排列。记其中发往 `(r,j)` 的连续行块为 `B_{s,r,j}`。

$$
Q_{r,j}=\operatorname{Concat}_{s=0}^{p-1}B_{s,r,j},\qquad
Y_{r,j}=Q_{r,j}W_{r,j}
$$

$$
Q_r=\operatorname{Concat}_{j=0}^{e-1}Q_{r,j},\qquad
Y_r=\operatorname{Concat}_{j=0}^{e-1}Y_{r,j}
$$

拼接均沿行维进行。AllToAllv 接收结果初始为 `[source rank][expert][token]`，重排后为 `[expert][source rank][token]`；每个块内部的 token 顺序保持不变。这是行数据的重分布，不进行 token 求和、复制或加权。

本 rank 的发送行数 `T_r=sum(send_counts)`，接收行数 `A_r=sum(recv_counts)`。有效专家权重为 `[e,H1,N1]`，因此 `Q_r` 为 `[A_r,H1]`，`Y_r` 为 `[A_r,N1]`。`trans_gmm_weight=True` 时对物理权重 `[e,N1,H1]` 的最后两维转置。

若同时提供共享专家输入，则另行计算 `mm_y=mm_x @ W_mm`。`trans_mm_weight=True` 时 `W_mm=mm_weight.T`；否则直接使用 `mm_weight`。共享专家结果不参与通信、重排或路由专家求和。

## 3. 接口规范

### 算子原型

```python
cann_bench.allto_allv_grouped_mat_mul(
    Tensor gmm_x, Tensor gmm_weight, str hcomm_info, int world_size,
    int[] send_counts, int[] recv_counts, Tensor? mm_x=None,
    Tensor? mm_weight=None, bool trans_gmm_weight=False,
    bool trans_mm_weight=False, bool permute_out_flag=False
) -> (Tensor gmm_y, Tensor? mm_y, Tensor? permute_out)
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `gmm_x` | FP16/BF16 Tensor，`[T,H1]` | 按目标 rank、专家连续分块的发送 token。 |
| `gmm_weight` | Tensor，`[e,H1,N1]` 或 `[e,N1,H1]` | 当前 rank 的专家权重；后一布局用于转置模式。 |
| `hcomm_info` | str | 已初始化的 HCCL 通信域名称。 |
| `world_size` | int | 实际通信域大小。 |
| `send_counts` | `list[int]`，长度 `p*e` | `send_counts[r*e+j]=C[self][r][j]`，单位为 token 行。 |
| `recv_counts` | `list[int]`，长度 `p*e` | `recv_counts[s*e+j]=C[s][self][j]`，单位为 token 行。 |
| `mm_x` | 可选 Tensor，`[BS,H2]` | 本地共享专家输入；默认 `None`。 |
| `mm_weight` | 可选 Tensor，`[H2,N2]` 或 `[N2,H2]` | 本地共享专家权重；默认 `None`。 |
| `trans_gmm_weight` | bool | 是否转置专家权重的最后两维；默认 `False`。 |
| `trans_mm_weight` | bool | 是否转置共享专家权重；默认 `False`。 |
| `permute_out_flag` | bool | 是否返回专家优先排列的通信输入；默认 `False`。 |

### 输出

始终返回三元组 `(gmm_y, mm_y, permute_out)`。

| 输出 | 形状 | 数据类型 | 说明 |
| --- | --- | --- | --- |
| `gmm_y` | `[A,N1]` | 与 `gmm_x` 相同 | 按专家、源 rank 排列的矩阵乘结果。 |
| `mm_y` | `[BS,N2]` | 与 `gmm_x` 相同 | 未提供共享专家输入时为 `None`。 |
| `permute_out` | `[A,H1]` | 与 `gmm_x` 相同 | `permute_out_flag=False` 时为 `None`；否则为矩阵乘前的重排输入。 |

### 规则与约束

- 支持 Ascend 950DT，通信域大小为 2、4、8、16、32 或 64。CCU 通信限单机 UB 域内；AI_CPU 通信可跨机 UB 域内互联。
- 本接口限定 `1<=e<=32` 且 `p*e<=256`；各 rank 的专家数、`H1`、`N1`、dtype 和通信属性必须一致，所有 rank 按相同顺序参与通信。
- `send_counts` 和 `recv_counts` 均为非负整数直接计数，允许某条边或某个专家的计数为零；不使用累计前缀计数。对每个 `(s,r,j)`，源端发送计数必须等于接收端对应计数。
- `0<T<52428800`、`0<A<52428800`、`1<=H1,N1<=65535`。输入行数等于 `sum(send_counts)`，输出行数等于 `sum(recv_counts)`；不接受整张卡总输入或总输出为空。
- 所有非空输入必须为相同的 FP16/BF16 类型、ND 格式，并位于当前 rank 的同一设备。本接口使用连续的物理输入；转置通过属性表达。
- `mm_x` 与 `mm_weight` 必须同时提供或同时为 `None`。提供共享专家时，`BS>0`、`1<=H2<=12288`、`1<=N2<=65535`，且 `T` 必须能被 `BS` 整除，`T/BS` 位于 `[2,8]`。
- 不支持偏置、激活函数、量化参数或通信计数 Tensor。专家权重不跨 rank 共享，每个接收 rank 使用自己的权重。

规格依据：`ops-transformer/mc2/allto_allv_grouped_mat_mul/README.md`、`op_host/op_tiling/arch35/allto_allv_grouped_mat_mul_tiling_a5.cpp` 和公共 tiling 参数校验；重排语义同时对应 `mc2_test/op_class/aclnnAlltoAllvGroupedMatMul.py`。

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


"""Ascend 950DT AlltoAllv -> Permute -> GroupedMatMul references."""

import torch


def _layout(ctx, attrs):
    p, rank, e = int(ctx["world_size"]), int(ctx["rank"]), int(attrs["e"])
    counts = attrs["token_counts"]
    if (
        p not in (2, 4, 8, 16, 32, 64)
        or int(attrs["world_size"]) != p
        or not 0 <= rank < p
    ):
        raise ValueError(
            "world_size and rank must match the Ascend 950DT communication group"
        )
    if not 1 <= e <= 32 or p * e > 256:
        raise ValueError(
            "this contract requires 1 <= e <= 32 and world_size * e <= 256"
        )
    if len(counts) != p or any(len(row) != p for row in counts):
        raise ValueError("token_counts must have shape [world_size, world_size, e]")
    for row in counts:
        for edge in row:
            if len(edge) != e or any(
                type(value) is not int or value < 0 for value in edge
            ):
                raise ValueError(
                    "token counts must be nonnegative integers, one per expert"
                )
    send = [value for edge in counts[rank] for value in edge]
    recv = [value for source in range(p) for value in counts[source][rank]]
    if not 0 < sum(send) < 52428800 or not 0 < sum(recv) < 52428800:
        raise ValueError("each rank must send and receive a nonempty token tensor")
    return p, rank, e, counts, send, recv


def _validate_inputs(ctx, inputs, attrs):
    p, rank, e, counts, send, recv = _layout(ctx, attrs)
    x, physical = inputs["gmm_x"], inputs["gmm_weight"]
    weight = (
        physical.transpose(1, 2)
        if bool(attrs.get("trans_gmm_weight", False))
        else physical
    )
    if x.ndim != 2 or weight.ndim != 3 or weight.shape[0] != e:
        raise ValueError("gmm_x must be 2D and gmm_weight must contain e matrices")
    if x.shape[0] != sum(send) or x.shape[1] != weight.shape[1]:
        raise ValueError(
            "gmm_x rows must equal sum(send_counts), with matching contraction dimensions"
        )
    if not 0 < x.shape[1] < 65536 or not 0 < weight.shape[2] < 65536:
        raise ValueError("H1 and N1 must be in [1,65535]")
    if list(inputs["send_counts"]) != send or list(inputs["recv_counts"]) != recv:
        raise ValueError(
            "rank-local counts disagree with token_counts[source][destination][expert]"
        )
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("this contract requires FP16 or BF16 inputs")
    mm_x, mm_weight = inputs.get("mm_x"), inputs.get("mm_weight")
    if (mm_x is not None) != bool(attrs.get("mm_out", False)) or (mm_x is None) != (
        mm_weight is None
    ):
        raise ValueError(
            "mm_x and mm_weight must both be present exactly when mm_out is enabled"
        )
    tensors = [x, physical]
    if mm_x is not None:
        effective_mm = (
            mm_weight.t() if bool(attrs.get("trans_mm_weight", False)) else mm_weight
        )
        if (
            mm_x.ndim != 2
            or effective_mm.ndim != 2
            or mm_x.shape[1] != effective_mm.shape[0]
        ):
            raise ValueError(
                "shared expert matrices must be 2D with matching contraction dimensions"
            )
        bs = mm_x.shape[0]
        if bs <= 0 or x.shape[0] % bs or not 2 <= x.shape[0] // bs <= 8:
            raise ValueError(
                "shared expert BS must divide sent token rows with a ratio in [2,8]"
            )
        if not 0 < mm_x.shape[1] <= 12288 or not 0 < effective_mm.shape[1] < 65536:
            raise ValueError("shared expert H2 and N2 exceed the supported range")
        tensors += [mm_x, mm_weight]
    if any(
        value.dtype != x.dtype or value.device != x.device or not value.is_contiguous()
        for value in tensors
    ):
        raise ValueError(
            "physical inputs must be contiguous tensors of the same dtype on one device"
        )
    return p, rank, e, counts, send, recv


def _make_tensor(shape, dtype, value_range, seed, device):
    if shape is None:
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    lo, hi = value_range or [-1.0, 1.0]
    return (
        torch.empty(shape, dtype=torch.float32)
        .uniform_(float(lo), float(hi), generator=generator)
        .to(dtype)
        .to(device)
    )


def mc2_make_rank_inputs(ctx, case_payload):
    attrs, shapes, dtypes = (
        case_payload["attrs"],
        case_payload["input_shapes"],
        case_payload["dtypes"],
    )
    ranges = case_payload.get("value_ranges") or [None] * 6
    _, rank, _, _, send, recv = _layout(ctx, attrs)
    seed = int(attrs.get("seed", 1)) + rank * 1000003
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtypes[0]]
    inputs = {
        "gmm_x": _make_tensor(shapes[0], dtype, ranges[0], seed, ctx["device"]),
        "gmm_weight": _make_tensor(
            shapes[1], dtype, ranges[1], seed + 101, ctx["device"]
        ),
        "send_counts": send,
        "recv_counts": recv,
        "mm_x": _make_tensor(shapes[4], dtype, ranges[4], seed + 211, ctx["device"]),
        "mm_weight": _make_tensor(
            shapes[5], dtype, ranges[5], seed + 307, ctx["device"]
        ),
    }
    _validate_inputs(ctx, inputs, attrs)
    return inputs


def mc2_distributed_golden(ctx, inputs, attrs):
    """Real AlltoAllv followed by a permutation and native-dtype matrix products."""
    p, rank, e, counts, send, recv = _validate_inputs(ctx, inputs, attrs)
    x = inputs["gmm_x"]
    received = x.new_empty((sum(recv), x.shape[1]))
    ctx["dist"].all_to_all_single(
        received,
        input=x,
        output_split_sizes=[sum(counts[s][rank]) for s in range(p)],
        input_split_sizes=[sum(edge) for edge in counts[rank]],
    )
    offsets, offset = [], 0
    for count in recv:
        offsets.append(offset)
        offset += count
    permutation = []
    for expert in range(e):
        for source in range(p):
            block = source * e + expert
            permutation.extend(range(offsets[block], offsets[block] + recv[block]))
    indices = torch.tensor(permutation, dtype=torch.int64, device=x.device)
    permuted = received.index_select(0, indices)
    weight = inputs["gmm_weight"]
    if bool(attrs.get("trans_gmm_weight", False)):
        weight = weight.transpose(1, 2)
    products, start = [], 0
    for expert in range(e):
        rows = sum(counts[source][rank][expert] for source in range(p))
        if rows:
            products.append(
                torch.matmul(permuted[start : start + rows], weight[expert])
            )
        start += rows
    y = torch.cat(products, dim=0)
    mm_y = None
    if inputs.get("mm_x") is not None:
        mm_weight = (
            inputs["mm_weight"].t()
            if bool(attrs.get("trans_mm_weight", False))
            else inputs["mm_weight"]
        )
        mm_y = torch.matmul(inputs["mm_x"], mm_weight)
    return y, mm_y, permuted if bool(attrs.get("permute_out_flag", False)) else None


def mc2_cpu_golden(ctx, inputs, attrs):
    """FP32 source/expert slices define the reference independently of the permutation."""
    p, rank, e, counts, send, recv = _validate_inputs(ctx, inputs, attrs)
    x, weight = inputs["gmm_x"].float(), inputs["gmm_weight"].float()
    if bool(attrs.get("trans_gmm_weight", False)):
        weight = weight.transpose(1, 2)
    received = torch.empty(
        (sum(recv), x.shape[1]), dtype=torch.float32, device=x.device
    )
    ctx["dist"].all_to_all_single(
        received,
        input=x,
        output_split_sizes=[sum(counts[source][rank]) for source in range(p)],
        input_split_sizes=[sum(edge) for edge in counts[rank]],
    )
    by_source = list(
        received.split([sum(counts[source][rank]) for source in range(p)], dim=0)
    )
    products, ordered_inputs = [], []
    for expert in range(e):
        for source in range(p):
            begin = sum(counts[source][rank][:expert])
            end = begin + counts[source][rank][expert]
            block = by_source[source][begin:end]
            ordered_inputs.append(block)
            products.append(torch.matmul(block, weight[expert]))
    mm_y = None
    if inputs.get("mm_x") is not None:
        mm_weight = inputs["mm_weight"].float()
        if bool(attrs.get("trans_mm_weight", False)):
            mm_weight = mm_weight.t()
        mm_y = torch.matmul(inputs["mm_x"].float(), mm_weight)
    permuted = (
        torch.cat(ordered_inputs, dim=0)
        if bool(attrs.get("permute_out_flag", False))
        else None
    )
    return torch.cat(products, dim=0), mm_y, permuted


def mc2_call_candidate(candidate, ctx, inputs, attrs):
    return candidate(
        inputs["gmm_x"],
        inputs["gmm_weight"],
        ctx["hcomm_info"],
        int(ctx["world_size"]),
        inputs["send_counts"],
        inputs["recv_counts"],
        mm_x=inputs.get("mm_x"),
        mm_weight=inputs.get("mm_weight"),
        trans_gmm_weight=bool(attrs.get("trans_gmm_weight", False)),
        trans_mm_weight=bool(attrs.get("trans_mm_weight", False)),
        permute_out_flag=bool(attrs.get("permute_out_flag", False)),
    )
```

## 6. 算子调用示例

以下示例在 Ascend 950DT 上执行。假设双卡通信域已经初始化，`device` 为当前 rank 的 NPU，`hcomm_info` 为域名称，`world_size=2`。每个 rank 有两个专家，每条源、目标、专家边发送 8 行，各 rank 均执行：

```python
import torch
import cann_bench

gmm_x = torch.randn(32, 256, dtype=torch.float16, device=device)
gmm_weight = torch.randn(2, 256, 128, dtype=torch.float16, device=device)
gmm_y, mm_y, permute_out = cann_bench.allto_allv_grouped_mat_mul(
    gmm_x, gmm_weight, hcomm_info, world_size,
    [8, 8, 8, 8], [8, 8, 8, 8],
    mm_x=None, mm_weight=None,
    trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=True,
)
# gmm_y: [32,128], mm_y: None, permute_out: [32,256]
```
