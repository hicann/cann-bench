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
