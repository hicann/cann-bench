# GroupedMatMulAlltoAllv 算子 API 描述

## 1. 算子简介

GroupedMatMulAlltoAllv 先执行本地分组矩阵乘，将结果按目标 rank 重排，再通过 AllToAllv 发送。可选共享专家 MatMul 在本地计算，返回独立输出。

以下描述 Ascend 950DT（A5、arch35）的非量化 FP16/BF16 接口。各段 token 数显式给定，不执行 TopK 选择或 Dispatch/Combine 路由。

## 2. 算子定义

设 p 为 rank 数，每个 rank 有 e 个专家。全局表 C[source][destination][expert] 表示源 rank 的某个本地专家向目标 rank 发送的行数。

源 rank s 的 gmm_x 按 [expert][destination] 存储。第 j 个专家的行数为 sum_d C[s][d][j]，与本地 W[s,j] 相乘，再按 [destination][expert] 重排发送。目标 rank r 按 [source][expert] 拼接接收结果：

$$
Y_r = \operatorname{Concat}_{s=0}^{p-1}\operatorname{Concat}_{j=0}^{e-1}(X_{s,j,r}W_{s,j})
$$

X[s,j,r] 为专家 j 输入中发送给 r 的连续行段。每行只传递一次，不做跨 rank 求和。A[s]=sum(d,j) C[s][d][j] 为输入总行数，BSK[r]=sum(s,j) C[s][r][j] 为输出总行数。

共享输出为 mm_y=mm_x@mm_weight，使用当前 rank 的本地数据。转置属性为真时，先将对应权重恢复到有效矩阵形状。

## 3. 接口规范

### 算子原型

```python
cann_bench.grouped_mat_mul_allto_allv(Tensor gmm_x, Tensor gmm_weight, str hcomm_info, int world_size, int[] send_counts, int[] recv_counts, Tensor? mm_x=None, Tensor? mm_weight=None, bool trans_gmm_weight=False, bool trans_mm_weight=False) -> (Tensor y, Tensor? mm_y)
```

### 输入与属性

| 参数 | 类型 / 形状 | 说明 |
| --- | --- | --- |
| `gmm_x` | FP16/BF16 Tensor，[A,H1] | 按本地专家分组，每组内按目标 rank 排列。 |
| `gmm_weight` | Tensor，[e,H1,N1] 或 [e,N1,H1] | 每个专家一份权重，与 gmm_x 类型相同。 |
| `hcomm_info` | str | 已初始化 HCCL 通信域名称。 |
| `world_size` | int | 必须等于实际通信域的 rank 数。 |
| `send_counts` | int[]，长度 p*e | 按 [destination][expert] 展平 C[当前rank]，总和为 A。 |
| `recv_counts` | int[]，长度 p*e | 按 [source][expert] 展平 C[source][当前rank]，总和为 BSK。 |
| `mm_x` | 可选 Tensor，[BS,H2] | 与 mm_weight 同时提供或同时为 None。 |
| `mm_weight` | 可选 Tensor，[H2,N2] 或 [N2,H2] | 共享权重，类型与 gmm_x 相同。 |
| `trans_gmm_weight` | bool | 默认 False；True 表示权重存储为 [e,N1,H1]。 |
| `trans_mm_weight` | bool | 默认 False；True 表示共享权重存储为 [N2,H2]。 |

### 输出

始终返回二元组 (y,mm_y)。y 的形状为 [BSK,N1]，类型与 gmm_x 相同；mm_y 的形状为 [BS,N2]，类型与 mm_x 相同。没有共享输入时，mm_y 必须为 None。

### 规则与约束

- A5 本集合仅适用于 Ascend 950DT。支持 2、4、8、16、32、64 卡；本集合使用 2 卡。所有 rank 按相同顺序参加集合通信。
- 数据张量使用 ND 格式，均为相同的 FP16 或 BF16。专家数 1<=e<=32，p*e<=256。
- 0<A,BSK<52428800，0<H1,N1,N2<65536；共享输入开启时 0<H2<=12288。
- 每一通信边的 send_counts 与对端 recv_counts 必须一致，单个计数允许为零。输入按专家分组，不能直接当作按目标 rank 分组的数据发送。
- 共享输入开启时，BSK 必须能被 BS 整除，BSK/BS 在 [2,8] 内。共享输出不参加 AllToAllv。
- 各 rank 可使用不同权重、数据和计数。本集合保持每个 rank 的总发送和接收行数相同，覆盖边间不均匀与零计数。
- A5 支持 AI_CPU 和 CCU 通信；CCU 仅支持单机 UB 域内互联，AI_CPU 可支持跨机 UB 域内互联。
- 来源：ops-transformer/mc2/grouped_mat_mul_allto_allv/README.md；op_host/grouped_mat_mul_allto_allv_def.cpp 的 ascend950 注册，以及 op_host/op_tiling/grouped_mat_mul_allto_allv_tiling.cpp 的平台校验。

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


from typing import Any, Dict, List

import torch


def grouped_mat_mul_allto_allv(
    gmm_x,
    gmm_weight,
    hcomm_info: str = "",
    world_size: int = 1,
    send_counts: List[int] = None,
    recv_counts: List[int] = None,
    mm_x=None,
    mm_weight=None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
):
    """Single-process smoke golden.

    Distributed cases use ``mc2_distributed_golden`` because the real semantics
    include all-to-all communication.
    """
    del hcomm_info, world_size, send_counts, recv_counts
    weight = gmm_weight.transpose(1, 2) if trans_gmm_weight else gmm_weight
    y = torch.matmul(gmm_x.float(), weight[0].float())
    mm_y = None
    if mm_x is not None and mm_weight is not None:
        w = mm_weight.t() if trans_mm_weight else mm_weight
        mm_y = torch.matmul(mm_x.float(), w.float())
    return y, mm_y


def mc2_make_rank_inputs(
    ctx: Dict[str, Any], case_payload: Dict[str, Any]
) -> Dict[str, Any]:
    if case_payload["attrs"].get("mc2_accuracy_mode") == "three_way":
        return _make_a5_rank_inputs(ctx, case_payload)
    rank = int(ctx["rank"])
    device = ctx["device"]
    attrs = case_payload["attrs"]
    shapes = case_payload["input_shapes"]
    dtypes = case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    world_size = int(ctx["world_size"])
    exp_per_card = int(attrs.get("e", shapes[1][0]))
    seed = int(attrs.get("seed", 1))

    rank_rows = _generate_positive_integers(world_size, int(shapes[0][0]), seed)
    exp_token_nums, _ = _generate_gmm_alltoallv_matrix(
        rank_rows, exp_per_card, seed=seed
    )
    send_counts = _get_send_counts(exp_token_nums, rank, exp_per_card, world_size)
    recv_counts = [int(x) for x in exp_token_nums[rank]]
    input_splits = _gen_output_splits(exp_token_nums, rank, exp_per_card, world_size)
    output_splits = _gen_input_splits(exp_token_nums, rank, exp_per_card, world_size)
    group_list = _get_group_list(exp_token_nums, rank, exp_per_card, world_size)

    gmm_x_shape = [sum(send_counts), int(shapes[0][1])]
    gmm_x = _make_tensor(gmm_x_shape, dtypes[0], ranges[0], seed, device)
    gmm_weight = _make_tensor(shapes[1], dtypes[1], ranges[1], seed, device)

    mm_out = bool(attrs.get("mm_out", False))
    mm_x = (
        _make_tensor(shapes[4], dtypes[4], ranges[4], seed, device) if mm_out else None
    )
    mm_weight = (
        _make_tensor(shapes[5], dtypes[5], ranges[5], seed, device) if mm_out else None
    )

    return {
        "gmm_x": gmm_x,
        "gmm_weight": gmm_weight,
        "send_counts": send_counts,
        "recv_counts": recv_counts,
        "input_splits": input_splits,
        "output_splits": output_splits,
        "group_list": group_list,
        "exp_token_nums": exp_token_nums,
        "mm_x": mm_x,
        "mm_weight": mm_weight,
    }


def mc2_call_candidate(
    candidate, ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
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
    )


def mc2_distributed_golden(
    ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    if attrs.get("mc2_accuracy_mode") == "three_way":
        return _a5_gmm_alltoallv(ctx, inputs, attrs, cpu_reference=False)
    return _legacy_distributed_golden(ctx, inputs, attrs)


def mc2_cpu_golden(ctx, inputs, attrs):
    """A5 FP32 reference: grouped matmul, reorder and real variable all-to-all."""
    return _a5_gmm_alltoallv(ctx, inputs, attrs, cpu_reference=True)


def _make_a5_rank_inputs(ctx, case_payload):
    """C[source][destination][expert] determines all static communication counts."""
    attrs = case_payload["attrs"]
    shapes, dtypes = case_payload["input_shapes"], case_payload["dtypes"]
    ranges = case_payload["value_ranges"]
    counts = attrs["token_counts"]
    rank, world_size, experts = (
        int(ctx["rank"]),
        int(ctx["world_size"]),
        int(attrs["e"]),
    )
    send_counts = [
        int(counts[rank][dest][expert])
        for dest in range(world_size)
        for expert in range(experts)
    ]
    recv_counts = [
        int(counts[source][rank][expert])
        for source in range(world_size)
        for expert in range(experts)
    ]
    if sum(send_counts) != int(shapes[0][0]):
        raise ValueError("token_counts must match the declared local gmm_x row count")
    group_list = [
        sum(counts[rank][dest][expert] for dest in range(world_size))
        for expert in range(experts)
    ]
    seed, device = int(attrs["seed"]) + rank * 1000003, ctx["device"]
    return {
        "gmm_x": _make_tensor(shapes[0], dtypes[0], ranges[0], seed, device),
        "gmm_weight": _make_tensor(
            shapes[1], dtypes[1], ranges[1], seed + 1009, device
        ),
        "send_counts": send_counts,
        "recv_counts": recv_counts,
        "group_list": group_list,
        "input_splits": [sum(counts[rank][dest]) for dest in range(world_size)],
        "output_splits": [sum(counts[source][rank]) for source in range(world_size)],
        "mm_x": (
            _make_tensor(shapes[4], dtypes[4], ranges[4], seed + 2017, device)
            if attrs.get("mm_out", False)
            else None
        ),
        "mm_weight": (
            _make_tensor(shapes[5], dtypes[5], ranges[5], seed + 3011, device)
            if attrs.get("mm_out", False)
            else None
        ),
    }


def _a5_unpermute(gmm_out, send_counts, world_size, experts):
    """Convert [expert][destination] rows into [destination][expert] rows."""
    by_expert = []
    offset = 0
    for expert in range(experts):
        chunks = []
        for dest in range(world_size):
            rows = int(send_counts[dest * experts + expert])
            chunks.append(gmm_out[offset : offset + rows])
            offset += rows
        by_expert.append(chunks)
    return torch.cat(
        [
            by_expert[expert][dest]
            for dest in range(world_size)
            for expert in range(experts)
        ],
        dim=0,
    ).contiguous()


def _a5_gmm_alltoallv(ctx, inputs, attrs, cpu_reference):
    """Compute using input precision on NPU and unrounded FP32 on CPU."""
    gmm_x, weights = inputs["gmm_x"], inputs["gmm_weight"]
    if cpu_reference:
        gmm_x, weights = gmm_x.float(), weights.float()
    if attrs.get("trans_gmm_weight", False):
        weights = weights.transpose(1, 2)
    gmm_out = _grouped_matmul(gmm_x, weights, inputs["group_list"])
    sent = _a5_unpermute(
        gmm_out, inputs["send_counts"], int(ctx["world_size"]), int(attrs["e"])
    )
    y = torch.empty(
        (sum(inputs["recv_counts"]), weights.shape[-1]),
        dtype=gmm_out.dtype,
        device=gmm_out.device,
    )
    ctx["dist"].all_to_all_single(
        y,
        input=sent,
        input_split_sizes=inputs["input_splits"],
        output_split_sizes=inputs["output_splits"],
    )
    mm_y = None
    if inputs.get("mm_x") is not None and inputs.get("mm_weight") is not None:
        mm_x, mm_weight = inputs["mm_x"], inputs["mm_weight"]
        if cpu_reference:
            mm_x, mm_weight = mm_x.float(), mm_weight.float()
        if attrs.get("trans_mm_weight", False):
            mm_weight = mm_weight.t()
        mm_y = torch.matmul(mm_x, mm_weight)
    return y, mm_y


def _legacy_distributed_golden(
    ctx: Dict[str, Any], inputs: Dict[str, Any], attrs: Dict[str, Any]
):
    dist = ctx["dist"]
    gmm_x = inputs["gmm_x"]
    gmm_weight = inputs["gmm_weight"]
    if bool(attrs.get("trans_gmm_weight", False)):
        gmm_weight = gmm_weight.transpose(1, 2)

    gmm_out = _grouped_matmul(gmm_x.float(), gmm_weight.float(), inputs["group_list"])
    unpermuted = _unpermute(
        gmm_out,
        int(ctx["world_size"]),
        int(attrs.get("e", gmm_weight.shape[0])),
        inputs["exp_token_nums"],
        int(ctx["rank"]),
        gmm_out.device,
    )
    y = torch.empty(
        (sum(inputs["output_splits"]), gmm_out.shape[-1]),
        dtype=torch.float32,
        device=gmm_out.device,
    )
    dist.all_to_all_single(
        y,
        input=unpermuted,
        output_split_sizes=inputs["output_splits"],
        input_split_sizes=inputs["input_splits"],
    )

    mm_y = None
    mm_x = inputs.get("mm_x")
    mm_weight = inputs.get("mm_weight")
    if mm_x is not None and mm_weight is not None:
        if bool(attrs.get("trans_mm_weight", False)):
            mm_weight = mm_weight.t()
        mm_y = torch.matmul(mm_x.float(), mm_weight.float())
    return y, mm_y


def _make_tensor(shape, dtype_name: str, value_range, seed: int, device):
    torch.manual_seed(int(seed))
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    dtype = dtype_map.get(str(dtype_name).lower(), torch.float32)
    lo, hi = value_range if value_range is not None else [0, 1]
    if dtype.is_floating_point:
        tensor = (
            torch.empty(shape, dtype=torch.float32)
            .uniform_(float(lo), float(hi))
            .to(dtype)
        )
    else:
        tensor = torch.randint(int(lo), int(hi) + 1, shape, dtype=dtype)
    return tensor.to(device)


def _grouped_matmul(gmm_x, gmm_weight, group_list):
    parts = torch.split(gmm_x, [int(x) for x in group_list], dim=0)
    outputs = []
    for idx, part in enumerate(parts):
        if part.numel() == 0:
            continue
        outputs.append(torch.matmul(part, gmm_weight[idx]))
    return (
        torch.cat(outputs, dim=0)
        if outputs
        else gmm_x.new_empty((0, gmm_weight.shape[-1]))
    )


def _unpermute(
    tokens, world_size: int, exp_per_card: int, exp_token_nums, rank: int, device
):
    counts = torch.zeros(world_size, exp_per_card, dtype=torch.int64, device="cpu")
    for src in range(world_size):
        counts[src] = torch.tensor(
            exp_token_nums[src][rank * exp_per_card : (rank + 1) * exp_per_card],
            dtype=torch.int64,
        )
    by_expert = counts.t()
    offsets = [0] + torch.cumsum(by_expert.sum(dim=1), dim=0).long().tolist()[:-1]
    cumsums = torch.cumsum(by_expert, dim=1)
    selected = []
    for src in range(world_size):
        for expert in range(exp_per_card):
            start = 0 if src == 0 else int(cumsums[expert][src - 1])
            end = int(cumsums[expert][src])
            indices = torch.arange(
                start + offsets[expert],
                end + offsets[expert],
                dtype=torch.long,
                device=device,
            )
            selected.append(tokens.index_select(dim=0, index=indices))
    return torch.cat(selected, dim=0) if selected else tokens


def _get_send_counts(
    exp_token_nums, rank: int, exp_per_card: int, world_size: int
) -> List[int]:
    counts = []
    for src in range(world_size):
        counts.extend(
            int(x)
            for x in exp_token_nums[src][
                rank * exp_per_card : (rank + 1) * exp_per_card
            ]
        )
    return counts


def _gen_input_splits(
    exp_token_nums, rank: int, exp_per_card: int, world_size: int
) -> List[int]:
    row = exp_token_nums[rank]
    return [
        sum(int(x) for x in row[i * exp_per_card : (i + 1) * exp_per_card])
        for i in range(world_size)
    ]


def _gen_output_splits(
    exp_token_nums, rank: int, exp_per_card: int, world_size: int
) -> List[int]:
    return [
        sum(
            int(x)
            for x in exp_token_nums[i][rank * exp_per_card : (rank + 1) * exp_per_card]
        )
        for i in range(world_size)
    ]


def _get_group_list(
    exp_token_nums, rank: int, exp_per_card: int, world_size: int
) -> List[int]:
    counts = torch.zeros(world_size, exp_per_card, dtype=torch.int64, device="cpu")
    for src in range(world_size):
        counts[src] = torch.tensor(
            exp_token_nums[src][rank * exp_per_card : (rank + 1) * exp_per_card],
            dtype=torch.int64,
        )
    return counts.sum(dim=0).long().tolist()


def _generate_positive_integers(n: int, sum_total: int, seed: int):
    if sum_total % n != 0:
        raise ValueError(f"sum_total={sum_total} must be divisible by n={n}")
    quotient = sum_total // n
    if quotient < n:
        raise ValueError(f"sum_total must satisfy sum_total >= n^2, got {sum_total}")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    remain = quotient - n
    if remain == 0:
        return torch.full((n,), n, dtype=torch.int64)
    probs = torch.ones(n, dtype=torch.float32) / n
    samples = torch.multinomial(
        probs, num_samples=remain, replacement=True, generator=rng
    )
    counts = torch.bincount(samples, minlength=n).long()
    return (counts + 1) * n


def _generate_gmm_alltoallv_matrix(
    a_array, exp_per_card: int, recv_count_zero_rank_list=None, seed: int = 1
):
    del recv_count_zero_rank_list
    world_size = len(a_array)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    k_values = [int(a) // world_size for a in a_array]
    blocks = []
    special = any(
        k < exp_per_card or int(a) % world_size != 0 for a, k in zip(a_array, k_values)
    )
    probs = torch.ones(exp_per_card, dtype=torch.float32) / exp_per_card
    for k in k_values:
        block = torch.zeros(exp_per_card, world_size, dtype=torch.int64)
        for col in range(world_size):
            if special:
                n_samples = max(k, 0)
                if n_samples == 0:
                    block[:, col] = 0
                else:
                    samples = torch.multinomial(
                        probs, num_samples=n_samples, replacement=True, generator=rng
                    )
                    block[:, col] = torch.bincount(
                        samples, minlength=exp_per_card
                    ).long()
            elif k == 0:
                block[:, col] = 1
            else:
                n_samples = k - exp_per_card
                samples = torch.multinomial(
                    probs, num_samples=n_samples, replacement=True, generator=rng
                )
                block[:, col] = (
                    torch.bincount(samples, minlength=exp_per_card).long() + 1
                )
        blocks.append(block)
    matrix = torch.cat(blocks, dim=0)
    return matrix.t().tolist(), matrix.t().to(torch.int64)
```

## 6. 算子调用示例

假设 Ascend 950DT 的当前 rank 设备 device 与通信域已初始化，world_size=2；两个 rank 使用相同计数和调用顺序。

```python
import torch
import cann_bench

# 每个专家有 8 行，依次发给 rank 0 和 rank 1 各 4 行。
gmm_x = torch.randn(16, 128, dtype=torch.float16, device=device)
gmm_weight = torch.randn(2, 128, 64, dtype=torch.float16, device=device)
send_counts = [4, 4, 4, 4]
recv_counts = [4, 4, 4, 4]
y, mm_y = cann_bench.grouped_mat_mul_allto_allv(
    gmm_x, gmm_weight, hcomm_info, world_size, send_counts, recv_counts,
    mm_x=None, mm_weight=None, trans_gmm_weight=False, trans_mm_weight=False,
)
# y: [16,64] FP16；mm_y: None
```
