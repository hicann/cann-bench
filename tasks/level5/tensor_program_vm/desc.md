# TensorProgramVm 算子 API 描述

## 1. 算子简介

张量程序虚拟机：**kernel 即解释器，程序即数据**。输入不是固定的算子参数，而是一段待执行的指令序列 `program`——kernel 必须在片上译码这段程序，并按指令逐条驱动寄存器堆（R 个宽 W 的行向量寄存器）上的向量运算。指令间存在真依赖（后一条指令可以读前一条刚写入的寄存器），执行顺序是规格的一部分。

这一形态是新一代 megakernel 推理基础设施的核心机制：Hazy Research 的低延迟 megakernel 在每个计算单元上运行一个持久化解释器，把整网前向编译成指令流按序消费；Mirage MPK 把 LLM 推理编译成单个持久 kernel，由片上调度器逐任务译码分发（见 §6 参考文献）。这类系统消灭了逐算子 kernel launch 的开销与中间结果的 GM 往返，本算子把其中"片上解释器"这一关键机制抽象为可评测的算子。

**主要应用场景**：
- Megakernel / 持久化 kernel 推理系统的片上指令流解释与任务调度
- 以数据驱动控制流的算子融合运行时（指令序列由上层编译器生成，kernel 侧只负责高效执行）
- 张量程序的硬件行为建模与调度器验证

**算子特征**：
- 难度等级：L5（FusedComposite）
- 双输入（program int32、init_regs 浮点）单输出（final_regs）
- 控制流由数据决定：8 条指令的 ISA，经 mod 译码后任意非负 int32 程序都合法
- 饱和算术（sat = clamp ±10000）保证任意随机程序数值有界、结果良定义

**为何是 L5**：
- **规格自足，不考知识**：全部 ISA 语义、译码规则与常数在本文档给全（§2），实现不需要查任何资料；难点不在数学而在"数据依赖的控制流 + 真依赖指令流"的工程映射
- **正确 ≠ 快（性能墙）**：按 §2 逐指令直译、每条指令下发一次独立 kernel 是完全正确的实现，能通过全部精度用例，但 P 条指令意味着 P 次 kernel launch 与 P 次寄存器堆的 GM 往返，性能只能锚定朴素 baseline（预期得分 ~0.5 档）。要逼近硬件下界，必须把整个执行做成**持久化解释器**：程序只从 GM 读一次，寄存器堆全程驻留片上，标量单元译码指令流驱动向量单元执行——这正是 megakernel 解释器的核心机制。指令间真依赖（含 MULADD 的读改写）决定了不能简单按指令并行，可并行维度必须从数据布局中挖掘
- **实现完整规格**：实现必须遵循完整 ISA、译码 mod 语义、MULADD 读写顺序、ROLL 方向、MEANB 广播和饱和边界，不应依赖公开样本的特定配置。
- **精度契约明确**：内部计算精度为 fp32（§2），阈值按实测误差下限收紧（§4），数值行为无自由发挥空间

## 2. 算子定义

### 执行模型

寄存器堆 $\mathrm{regs} \in \mathbb{R}^{R \times W}$，初值为 `init_regs`。对 $t = 0, 1, \dots, P-1$ **严格按序**执行 `program` 的第 $t$ 行；第 $t+1$ 条指令读取的寄存器值是第 $t$ 条执行完后的状态（真依赖）。全部 P 条指令执行完后的寄存器堆即输出 `final_regs`。

### 译码规则

`program` 的每行 4 个 int32（任意非负整数都是合法指令）：

$$
\mathrm{op} = \mathrm{program}[t,0] \bmod 8, \quad
\mathrm{dst} = \mathrm{program}[t,1] \bmod R, \quad
a = \mathrm{program}[t,2] \bmod R, \quad
b = \mathrm{program}[t,3] \bmod R
$$

program 取值非负（评测取值范围 [0, 1073741823]），mod 无符号歧义。

### ISA（8 条指令）

所有操作数为寄存器堆的行向量 $[W]$。饱和函数 $\mathrm{sat}(x) = \mathrm{clamp}(x, -10000.0, 10000.0)$ 逐元素作用。

| op | 名称 | 语义 | 说明 |
|----|------|------|------|
| 0 | ADD | regs[dst] = sat(regs[a] + regs[b]) | 逐元素加，饱和 |
| 1 | MUL | regs[dst] = sat(regs[a] * regs[b]) | 逐元素乘，饱和 |
| 2 | RELU | regs[dst] = max(regs[a], 0) | b 忽略 |
| 3 | MEANB | regs[dst] = broadcast(mean(regs[a])) | 沿 W 求均值后广播为常数行；b 忽略 |
| 4 | MAX | regs[dst] = elementwise_max(regs[a], regs[b]) | 逐元素取大 |
| 5 | ROLL | regs[dst][j] = regs[a][(j−1+W) mod W] | 向右循环移 1 位；b 忽略 |
| 6 | MULADD | regs[dst] = sat(regs[dst] + regs[a] * regs[b]) | **读 dst，写 dst**（读改写） |
| 7 | COPY | regs[dst] = regs[a] | b 忽略 |

**本算子的精确约定**：
- **饱和语义**：sat 仅作用于 ADD / MUL / MULADD 三条会扩大数值范围的指令。这保证任意随机程序下所有中间值有界（$|x| \le \max(10000, \max|{\rm init}|)$）、结果良定义；RELU / MEANB / MAX / ROLL / COPY 不会扩大值域，不做饱和
- **MULADD 读改写**：先读 dst 的当前值，加上 regs[a] ⊙ regs[b]，饱和后写回 dst——它是唯一读三个寄存器的指令
- **ROLL 方向**：向右循环移 1 位，即输出下标 j 取输入下标 (j−1+W) mod W；连续 W 次 ROLL 恒等
- **MEANB**：先沿 W 求算术均值（一个标量），再广播成整行；对常数行幂等（数学恒等，浮点下可差 1 ulp）
- **dst = a 或 dst = b 合法**：语义仍是"先读源、后写 dst"（每条指令的读全部发生在写之前）
- **计算精度**：全部指令运算以 fp32 执行；bfloat16 / float16 输入先升 fp32，寄存器堆以 fp32 保存，仅最终输出转回输入 dtype（这是规格的一部分，golden 与评测均按此定义）

## 3. 接口规范

### 算子原型

```python
tensor_program_vm(Tensor program, Tensor init_regs) -> Tensor final_regs
```

### 输入参数说明

| 参数 | 类型 | 必需 | dtype | shape | 描述 |
|------|------|------|-------|-------|------|
| program | Tensor | 是 | int32 | [P, 4] | 指令序列，每行经 mod 译码为 (op, dst, a, b)，任意非负整数合法（评测取值范围 [0, 1073741823]） |
| init_regs | Tensor | 是 | float32/float16/bfloat16 | [R, W] | 寄存器堆初值（公开样本取值范围见 cases.csv） |

### 输出

| 名称 | dtype | shape | 描述 |
|------|-------|-------|------|
| final_regs | 与 init_regs 一致 | [R, W] | 全部 P 条指令执行完后的寄存器堆 |

### 数据类型

| init_regs dtype | final_regs dtype | 内部计算 |
|-----------------|------------------|----------|
| bfloat16 | bfloat16 | fp32（寄存器堆全程 fp32 驻留，仅输出转回） |
| float16 | float16 | fp32 |
| float32 | float32 | fp32 |

### 规则与约束

- program 取值非负（由 value_range 保证）；经 mod 译码后**任意程序都合法**，算子不做合法性检查
- 指令必须严格按 t = 0..P-1 的顺序生效：第 t+1 条指令读到的寄存器状态 = 第 t 条执行完后的状态
- 每条指令内部"先读后写"：dst 与 a / b 重合时，源操作数取写入前的值
- 寄存器堆与全部指令运算以 fp32 保存/执行（§2 计算精度约定）
- MEANB 沿 W 的求和顺序不限（不同归约树均可），误差由 §4 阈值覆盖
- 未被任何指令写入的寄存器在输出中保持初值
- 输出须为 contiguous 张量

### 支持范围

| 维度 / 参数 | 范围 | 备注 |
|---|---|---|
| `P`（程序长度） | 1 ~ 4096 | cases.csv 实测 1 ~ 4096 |
| `R`（寄存器数） | 1 ~ 64 | cases.csv 实测 1 ~ 64 |
| `W`（寄存器宽度） | 1 ~ 8192 | cases.csv 实测 1 ~ 8192 |
| dtype | bfloat16 / float16 / float32 | 三种均覆盖 |
| `program` 取值 | [0, 1073741823] | 公开样本见 cases.csv；实现应支持规格域内的任意合法程序 |
| `init_regs` 有限值 | [-1, 1] | 浮点输入先按声明 dtype 量化，再按 fp32 ISA 执行 |
| 非有限值 | NaN、+Inf、-Inf | IEEE 边界语义见下文；不得假设输入全为有限数 |

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

参考计算区分 FP64 真值与 FP32 同精度路径。实现按 FP32 指令语义运行，精度要求如下。

| 数据类型 | FLOAT32 | FLOAT16 | BFLOAT16 |
|----------|---------|---------|----------|
| **通过阈值(Threshold)** | 0.0001 | 0.001 | 0.008 |

当平均相对误差 MERE < Threshold，最大相对误差 MARE < 10 * Threshold 时判定为通过；小值域与相消位置沿用框架相对于同精度参考的兜底规则，以容纳 FP32 归约顺序 / FMA 的合理差异。RELU / MUL 衰减链产生的小值域场景由评测框架的兜底标准处理。**达标前提**：按 §2 约定以 fp32 执行全部指令运算（寄存器堆 fp32 驻留）。

## 5. 标准 Golden 代码

```python

import torch

"""
TensorProgramVm 算子 Torch Golden 参考实现

张量程序虚拟机：kernel 即解释器，程序即数据。输入 program [P, 4] int32 是一段
待执行的指令序列，init_regs [R, W] 是寄存器堆（R 个行向量寄存器，每个宽 W）。
对 t = 0..P-1 顺序执行，译码规则（mod 保证任意非负 int32 都是合法指令）:
    op  = program[t, 0] mod 8
    dst = program[t, 1] mod R
    a   = program[t, 2] mod R
    b   = program[t, 3] mod R
ISA（所有操作数为寄存器堆的行向量 [W]，sat(x) = clamp(x, -10000, 10000)）:
    0 ADD:    regs[dst] = sat(regs[a] + regs[b])
    1 MUL:    regs[dst] = sat(regs[a] * regs[b])
    2 RELU:   regs[dst] = max(regs[a], 0)                （b 忽略）
    3 MEANB:  regs[dst] = broadcast(mean(regs[a]))       （沿 W 求均值后广播；b 忽略）
    4 MAX:    regs[dst] = elementwise_max(regs[a], regs[b])
    5 ROLL:   regs[dst][j] = regs[a][(j-1+W) mod W]      （向右循环移 1 位；b 忽略）
    6 MULADD: regs[dst] = sat(regs[dst] + regs[a] * regs[b])   （读 dst，写 dst）
    7 COPY:   regs[dst] = regs[a]                        （b 忽略）
输出 final_regs = 全部指令执行完后的寄存器堆 [R, W]。
饱和仅作用于 ADD/MUL/MULADD，保证任意随机程序数值有界（|值| ≤ max(10000, |init|)）、
结果良定义。指令间存在真依赖（后一条可能读前一条的结果），golden 逐指令顺序执行；
每条指令是 [W] 向量操作，寄存器堆全程张量化。
plain golden 对量化输入升 fp32 计算（bench 语义：寄存器堆全程 fp32 驻留，仅输出转回输入
dtype）；tensor_program_vm_oracle 跟随输入精度（golden_precision=fp64_cpu 下即为
fp64 真值）。框架传入 fp64 时 plain golden 也保留 fp64，避免同精度参考误当高精度真值。
"""

_SAT_BOUND = 10000.0


def _tensor_program_vm_core(program, init_regs, compute_dtype):
    """核心计算：以 compute_dtype 精度逐指令顺序执行，返回 final_regs [R, W]。"""
    P = program.shape[0]
    R, W = init_regs.shape

    prog = program.to(torch.int64)
    op_v = (prog[:, 0] % 8).tolist()
    dst_v = (prog[:, 1] % R).tolist()
    a_v = (prog[:, 2] % R).tolist()
    b_v = (prog[:, 3] % R).tolist()

    hi = torch.tensor(_SAT_BOUND, dtype=compute_dtype, device=init_regs.device)
    lo = torch.tensor(-_SAT_BOUND, dtype=compute_dtype, device=init_regs.device)
    zero = torch.zeros((), dtype=compute_dtype, device=init_regs.device)

    regs = init_regs.to(compute_dtype).clone()
    for t in range(P):
        op, dst, a, b = op_v[t], dst_v[t], a_v[t], b_v[t]
        if op == 0:  # ADD
            regs[dst] = torch.clamp(regs[a] + regs[b], lo, hi)
        elif op == 1:  # MUL
            regs[dst] = torch.clamp(regs[a] * regs[b], lo, hi)
        elif op == 2:  # RELU
            regs[dst] = torch.maximum(regs[a], zero)
        elif op == 3:  # MEANB（沿 W 求均值后广播）
            regs[dst] = regs[a].mean().expand(W).clone()
        elif op == 4:  # MAX
            regs[dst] = torch.maximum(regs[a], regs[b])
        elif op == 5:  # ROLL（向右循环移 1 位）
            regs[dst] = torch.roll(regs[a], shifts=1, dims=0)
        elif op == 6:  # MULADD（读 dst，写 dst）
            regs[dst] = torch.clamp(regs[dst] + regs[a] * regs[b], lo, hi)
        else:  # 7 COPY
            regs[dst] = regs[a].clone()
    return regs


def tensor_program_vm(program: torch.Tensor, init_regs: torch.Tensor) -> torch.Tensor:
    """
    张量程序虚拟机 golden reference（逐指令顺序执行；plain golden = bench：fp32 寄存器堆）

    Args:
        program: [P, 4] int32 指令序列，任意非负整数合法（评测取值范围 [0, 1073741823]），
            每行经 mod 译码为 (op, dst, a, b)
        init_regs: [R, W] float32/float16/bfloat16 寄存器堆初值（评测取值范围 [-1, 1]）

    Returns:
        final_regs: [R, W] 全部指令执行完后的寄存器堆，dtype 与 init_regs 一致
    """
    compute_dtype = torch.float64 if init_regs.dtype == torch.float64 else torch.float32
    out = _tensor_program_vm_core(program, init_regs, compute_dtype)
    return out.to(init_regs.dtype)


def tensor_program_vm_oracle(
    program: torch.Tensor, init_regs: torch.Tensor
) -> torch.Tensor:
    """Oracle (g)：dtype-agnostic，计算精度跟随输入（fp64_cpu 下为 fp64 真值）。框架传入 fp64 时 plain golden 也保留 fp64，避免同精度参考误当高精度真值。"""
    return _tensor_program_vm_core(program, init_regs, init_regs.dtype)


def tensor_program_vm_bench(
    program: torch.Tensor, init_regs: torch.Tensor
) -> torch.Tensor:
    """Quantized-input FP32 ISA reference for the checker's native error floor."""
    return _tensor_program_vm_core(program, init_regs, torch.float32).to(
        init_regs.dtype
    )


def get_input(
    program: torch.Tensor,
    init_regs: torch.Tensor,
    program_pattern="raw",
    program_opcode=0,
    **kwargs,
):
    """Construct legal data-dependent programs; test metadata is not an op argument.

    The ISA and callable interface are unchanged. Bounded stream families retain
    input-dependent read-only rows. Long streams use noncontracting ADD,
    sign-MUL and ROLL; short streams separately cover the full ISA.
    Arithmetic chains read the preceding result; MULADD reads old destination.
    All float preprocessing is fp32; only the constructed input is quantized back.
    """
    if program_pattern == "raw":
        return program, init_regs
    p, r = program.shape[0], init_regs.shape[0]
    if p < 1 or r < 1:
        raise ValueError("positive program/register dimensions required")
    code = program.to(torch.int64).clone()
    initial = init_regs.float().clone()
    # Rebuild bounded register values from the supplied input values.
    # Keep finite values inside the documented domain while preserving IEEE cases.
    initial = torch.where(torch.isfinite(initial), initial.clamp(-1, 1), initial)
    base = code.clone()
    if program_pattern in (
        "special_copy",
        "special_add",
        "single",
        "scalar",
        "roll_boundary",
        "mean_boundary",
        "saturation_add",
        "saturation_muladd",
        "copy_boundary",
    ):
        previous = int(base[0, 2] % r)
        for t in range(p):
            if program_pattern == "special_copy":
                op, dst, left, right = 7, 0, min(1, r - 1), 0
            elif program_pattern == "special_add":
                op, dst, left, right = 0, 0, min(1, r - 1), min(2, r - 1)
            elif program_pattern == "single":
                op, dst, left, right = (
                    int(program_opcode) % 8,
                    int(base[t, 1] % r),
                    int(base[t, 2] % r),
                    int(base[t, 3] % r),
                )
            elif program_pattern == "scalar":
                op, dst, left, right = (0, 1, 5)[t % 3], 0, 0, 0
            elif program_pattern == "roll_boundary":
                op, dst, left, right = 5, min(1, r - 1), min(1, r - 1), 0
            elif program_pattern == "mean_boundary":
                op, dst, left, right = 3, 0, 0, 0
            elif program_pattern == "copy_boundary":
                op, dst, left, right = 7, min(1, r - 1), 1073741823 % r, 1073741823 % r
            else:
                op, dst, left, right = (
                    (0 if program_pattern == "saturation_add" else 6),
                    0,
                    0,
                    0,
                )
            if program_pattern == "single" and int(program_opcode) % 8 == 3 and r >= 2:
                dst, left, right = 1, 0, 0
                initial[1] = torch.where(
                    initial[0] >= 0,
                    -torch.full_like(initial[0], 0.5),
                    torch.full_like(initial[0], 0.5),
                )
            code[t] = torch.tensor(
                [op, dst, left, right], dtype=torch.int64, device=code.device
            )
            if program_pattern == "copy_boundary":
                code[t, 0] = 1073741823
                for column, operand in ((1, dst), (2, left), (3, right)):
                    code[t, column] = ((1073741823 - operand) // r) * r + operand
    elif program_pattern.startswith("affine_"):
        if r < 3:
            raise ValueError("long affine stream needs two read-only rows")
        active = r - 2
        initial[:active] = torch.round(initial[:active] * 8) / 8
        initial[-2] = (1 + torch.floor(initial[-2].abs() * 3)) / 8
        initial[-1] = torch.where(
            initial[-1] >= 0,
            torch.ones_like(initial[-1]),
            -torch.ones_like(initial[-1]),
        )
        # Random instruction selection avoids periodic whole-program identities.
        # ADD increments are <=1; MUL uses exactly +/-1; ROLL is a permutation.
        # Thus |state|<=1+P<=4097 and no contraction/clamp can erase a prefix.
        choices = torch.tensor([0, 1, 5], dtype=torch.int64, device=code.device)
        op = choices[base[:, 0] % 3]
        dst = base[:, 1] % active
        first = int(base[0, 2] % active)
        if program_pattern == "affine_alias":
            dst = dst.clone()
            dst[0] = first
            dst[2::2] = dst[1:-1:2].clone()
        source = torch.empty_like(dst)
        source[0] = first
        source[1:] = dst[:-1]
        right = torch.where(op == 1, r - 1, r - 2 + base[:, 3] % 2)
        if program_pattern == "affine_reduce":
            op[-1] = 3
        values = torch.stack((op, dst, source, right), 1)
        moduli = torch.tensor([8, r, r, r], dtype=torch.int64, device=code.device)
        limits = (1073741823 - (moduli - 1)) // moduli
        code = ((base // moduli) % (limits + 1)) * moduli + values
    else:
        if r < 3:
            raise ValueError("bounded stream requires at least three registers")
        allowed = {
            "mixed": (0, 1, 2, 3, 4, 5, 6, 7),
            "dependency": (0, 5, 6, 1),
            "reduction": (3, 0, 5, 6),
            "alias": (0, 1, 2, 3, 4, 5, 6, 7),
            "mixed_small": (0, 5, 6, 1),
        }
        if program_pattern not in allowed:
            raise ValueError("unknown program construction pattern")
        active = r - 2
        scale = initial.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        initial[-2] = initial[-2].abs() * 0.25 + scale * 0.25
        initial[-1] = -initial[-1].abs() * 0.25 - scale * 0.125
        previous = int(base[0, 2] % active)
        schedule = allowed[program_pattern]
        phase = int(base[0, 0] % len(schedule))
        for t in range(p):
            op = schedule[(t + phase) % len(schedule)]
            dst = (
                previous
                if program_pattern == "alias" and t % 2 == 0
                else int(base[t, 1] % active)
            )
            right = r - 2 + int(base[t, 3] % 2)
            # Higher multiples exercise modulo decoding without changing meaning.
            operands = (op, dst, previous, right)
            moduli = (8, r, r, r)
            for column, (operand, modulus) in enumerate(zip(operands, moduli)):
                quotient = int(base[t, column]) // modulus
                limit = (1073741823 - (modulus - 1)) // modulus
                code[t, column] = (quotient % (limit + 1)) * modulus + operand
            previous = dst
    return code.to(program.dtype), initial.to(init_regs.dtype)
```

## 6. 额外信息

### 算子调用示例

```python
import torch

P, R, W = 1024, 32, 4096

program = torch.randint(0, 1073741824, (P, 4), dtype=torch.int32, device="npu")
init_regs = (torch.rand(R, W, dtype=torch.bfloat16, device="npu") * 2 - 1)

final_regs = tensor_program_vm(program, init_regs)
# final_regs.shape: [R, W]，dtype 与 init_regs 一致
```

### 可用于自检的性质（均已数值验证）

- 程序 `[[7, 1, 0, 0]]`（COPY r1←r0）执行一次与两次结果相同，且 final_regs[1] == init_regs[0]
- 对同一寄存器连续 W 次 ROLL 后恒等；单次 ROLL 满足 out[j] = in[(j−1+W) mod W]
- MEANB 幂等（对已广播的常数行再做 MEANB 结果不变，浮点下至多差 1 ulp）
- init_regs 全 1 时，程序 `[[0,0,0,0]] * 20`（r0 = r0 + r0 连加）使 r0 严格按 2 的幂增长并在 +10000 饱和；全 −1 时对称地在 −10000 饱和
- op=8 与 op=0 等价、dst=R+1 与 dst=1 等价（mod 译码）
- 未被写入的寄存器保持初值

### 关于长随机程序的数值形态

全域随机程序（op 均匀分布）中 ADD 的饱和增长与 MAX 的传播使寄存器值随 P 增大向吸收态（±10000、0）收敛——这是 ISA 的固有性质，不是实现缺陷；实现必须遵循 ISA 对任意合法程序的定义。

### 参考文献

- Hazy Research (Stanford), "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B" (2025)：单 kernel 内的持久化指令解释器，每个计算单元按序消费指令流（本算子"kernel 即解释器"形态的直接原型）
- Mirage Persistent Kernel (MPK), "Compiling LLMs into a MegaKernel: A Path to Low-Latency Inference" (2025, CMU/Mirage 项目)：把 LLM 推理编译为单个持久 kernel，片上调度器逐任务译码分发（SM 级任务调度的原型）
- Gupta, K. et al. (2012). "A Study of Persistent Threads Style GPU Programming for GPGPU Workloads". InPar 2012（持久化线程编程模型的早期系统化研究）


### IEEE 边界和测试数据构造

NaN 按 PyTorch 的逐元素算术、maximum、mean 规则传播；clamp 不把 NaN 变成有限数。
ADD/MUL/MULADD 的 ±Inf 结果按既定 ±10000 边界饱和，Inf 与反号 Inf 相加产生 NaN。
COPY/ROLL 保留对应值，MEANB 对混合非有限值遵循相同算术规则。NaN 位置和 Inf 符号
均属于正确性契约。+0 与 -0 按数值相等比较。

常规 CPU 参考先保留 BF16/FP16 输入量化，再以 FP32 执行全部 ISA，仅最终输出转换回
原 dtype；MERE/MARE 的数值计算可在 FP32 中进行，阈值不变。FP64 参考用于独立
评测高精度真值，不改变该算子的 FP32 执行规格；框架把浮点输入升为 FP64 后，
golden 保留该精度；同精度参考由显式 _bench 钩子按 FP32 计算。

题包的 get_input 钩子只构造合法 program/init_regs 数据，不增加算子调用参数或更改 ISA。
长流保留从输入导出的只读幅度行与正负一符号行，随机交错 ADD、乘正负一和 ROLL，
活动初值量化到 1/8 格点，幅度为输入导出的 1/8、1/4、3/8 或 1/2；连续值覆盖保留在短流。
源寄存器承接前一条结果。P≤4096 时绝对值不超过 4097，不会靠收缩或饱和遗忘前缀；
归约家族在最后增加一次 MEANB。这里不声称仿射流不可并行重写，只要求不能直接忽略
前缀。完整八条 ISA 在短混合流、单指令、别名、归约、非有限传播及饱和边界中覆盖。
短程序中的幂等边界不冒充长负载。实现仍须支持完整规格内任意
合法输入，不得依赖测试构造模式。
