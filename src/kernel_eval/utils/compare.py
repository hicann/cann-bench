#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
张量精度对比引擎

职责：
1. 提供 compare_tensors() — 张量对比主入口（支持多输出）
2. SingleOutputResult / CompareResult 数据类
3. 正常值域 MERE/MARE 三方比值 + 小值域/相消兜底判定

从 utils/precision.py 拆分出来；阈值查询依赖 utils/thresholds.py。
"""

import logging
import math
import traceback
from typing import Union, Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field

import torch

from ..base.result import FAILURE_TYPE_REFERENCE_INVALID
from .tensor_utils import non_contiguous_reason
from .thresholds import (
    get_threshold,
    get_small_value_threshold,
    get_small_value_error,
    get_cancel_boundary,
    get_cancel_zero_threshold,
)


_logger = logging.getLogger(__name__)

# Normal-region three-way MARE/MERE ratio limits.
_NORMAL_MARE_RATIO_LIMIT = 5.0
_NORMAL_MERE_RATIO_LIMIT = 1.5
# Only absorb FP64 reduction/division rounding at the closed ratio limits.
_NORMAL_RATIO_REL_TOL = 1e-12


# Integer dtypes that should use absolute-difference / exact comparison
# rather than the floating-point MERE/MARE path. Must stay in sync with
# the int threshold entries in `thresholds.py` (which declares all 8).
_INTEGER_DTYPES = (
    torch.int8, torch.int16, torch.int32, torch.int64,
    torch.uint8, torch.uint16, torch.uint32, torch.uint64,
)

# Bit-exact 浮点路径支持的 fp -> int view 映射（字节级比较）。
# 提到模块级：device 归一化需要在进入分支前判断是否走 bit-exact 路径。
_BIT_VIEW = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


@dataclass
class SingleOutputResult:
    """单个输出的对比结果（通用容器，各 checker 通过 metadata 扩展特有指标）"""
    index: int                      # 输出索引
    name: str = ""                  # 输出名称（可选）
    dtype: str = ""                 # 数据类型
    passed: bool = True
    mismatch_count: int = 0         # 不匹配元素数
    total_count: int = 0            # 总元素数
    error_msg: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'index': self.index,
            'name': self.name,
            'dtype': self.dtype,
            'passed': self.passed,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
            'error_msg': self.error_msg,
        }
        d.update(self.metadata)
        return d

    def format_summary(self) -> str:
        """格式化单输出判定摘要（通用实现，各 checker 子类覆盖此方法）"""
        dtype_str = f"{self.dtype}[{self.name or self.index}]"
        if self.passed:
            return f"{dtype_str}: ✅"
        else:
            return f"{dtype_str}: ❌ {self.error_msg}"


@dataclass
class CompareResult:
    """对比结果（支持多输出算子）"""
    passed: bool
    dtype: str
    threshold: float  # 精度阈值（聚合值）
    mere: float = 0.0  # 平均相对误差（聚合值）
    mare: float = 0.0  # 最大相对误差（聚合值）
    max_diff: float = 0.0
    mean_diff: float = 0.0
    mismatch_count: int = 0
    total_count: int = 0
    mismatch_ratio: float = 0.0
    small_value_error_count: int = 0  # 小值域 NPU 错误计数
    small_value_cpu_error_count: int = 0  # 小值域 CPU 错误计数
    small_value_total_count: int = 0  # 小值域总计数
    cancel_error_count: int = 0  # 相消位置 NPU 错误计数
    cancel_cpu_error_count: int = 0  # 相消位置 CPU 错误计数
    cancel_total_count: int = 0  # 相消位置总计数
    small_value_passed: bool = True  # 小值域兜底判定是否通过
    cancel_passed: bool = True       # 相消兜底判定是否通过
    normal_total_count: int = 0      # 兜底时统计正常域元素数；整体提前通过时未统计，保持 0
    normal_passed: bool = True        # 正常值域同精度兜底判定是否通过
    normal_mare: float = 0.0         # 正常域最大相对误差；整体提前通过时未统计，保持 0
    normal_mare_ratio: Optional[float] = None
    normal_mere_ratio: Optional[float] = None
    error_msg: Optional[str] = None
    output_results: List[SingleOutputResult] = field(default_factory=list)  # 各输出独立结果
    failure_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            'passed': self.passed,
            'dtype': self.dtype,
            'threshold': self.threshold,
            'mere': self.mere,
            'mare': self.mare,
            'max_diff': self.max_diff,
            'mean_diff': self.mean_diff,
            'mismatch_count': self.mismatch_count,
            'total_count': self.total_count,
            'mismatch_ratio': self.mismatch_ratio,
            'small_value_error_count': self.small_value_error_count,
            'small_value_cpu_error_count': self.small_value_cpu_error_count,
            'small_value_total_count': self.small_value_total_count,
            'cancel_error_count': self.cancel_error_count,
            'cancel_cpu_error_count': self.cancel_cpu_error_count,
            'cancel_total_count': self.cancel_total_count,
            'small_value_passed': self.small_value_passed,
            'cancel_passed': self.cancel_passed,
            'normal_total_count': self.normal_total_count,
            'normal_passed': self.normal_passed,
            'normal_mare': self.normal_mare,
            'normal_mare_ratio': self.normal_mare_ratio,
            'normal_mere_ratio': self.normal_mere_ratio,
            'error_msg': self.error_msg,
            'output_results': [r.to_dict() for r in self.output_results],
        }
        if self.failure_type is not None:
            result['failure_type'] = self.failure_type
        return result

    def format_all_outputs(self) -> str:
        """格式化所有输出判定结果（用于日志）"""
        lines = []
        for r in self.output_results:
            lines.append(f"  - {r.format_summary()}")
        return "\n".join(lines)


def _normalize_outputs(output: Any) -> List[torch.Tensor]:
    """将输出标准化为张量列表。

    F089: 旧版静默丢弃非 tensor 元素（str / None / int / 嵌套非 tensor），
    若 output 和 golden 各自 normalize 后 len 恰好相等但**对应位置不同**，
    后续 zip 比较会张冠李戴。改为保留 None 占位 + 丢弃元素超过 0 时 logger 警告。
    调用方拿到 None 占位时按"该位置无法对比"处理。
    """
    if isinstance(output, torch.Tensor):
        return [output]
    elif isinstance(output, (tuple, list)):
        result: List[Any] = []
        dropped = 0
        for item in output:
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        result.append(sub_item)
                    else:
                        result.append(None)
                        dropped += 1
            else:
                # 保留 None 占位维持索引对齐
                result.append(None)
                if item is not None:
                    dropped += 1
        if dropped > 0:
            _logger.warning(
                "_normalize_outputs: 丢弃 %d 个非 tensor 元素（替换为 None 占位 "
                "维持索引对齐）。若 output / golden 含同类型非 tensor 数据需调用方"
                "在外层先做语义比较。",
                dropped,
            )
        return result
    else:
        return []


# 分块归约的块长（元素数）。fp64 下 4Mi 元素 = 32MB，足够摊薄 per-chunk 开销，
# 又远小于隐藏集巨型 case 的整段长度。
_MASKED_CHUNK = 1 << 22


class _MaskedStats:
    """``values[mask]`` 的 count / mean / max，但不物化那份 gather。

    ``relative_error[valid_mask]`` 这类写法会分配一份与输入等长的 fp64 张量，而调用点
    只消费三个标量。分块之后额外驻留是 O(chunk) 而非 O(N)——2.15B 元素的 case 上，
    这类 gather 一共占 16 B/元素。

    求和顺序因此从整段 pairwise 变成「块内 pairwise + 块间顺序累加」，fp64 下 mere 会有
    个位数 ULP 的偏差（实测最大相对偏差 6.5e-16）。mere/mare 是与阈值比较的统计量，
    这个量级不影响任何判定；120 个 case 的语料上 passed / small_value_passed /
    cancel_passed / normal_passed 全部不变。
    """

    __slots__ = ("count", "_sum", "_max")

    def __init__(self, values: torch.Tensor, mask: torch.Tensor,
                 chunk: int = _MASKED_CHUNK) -> None:
        flat_values = values.reshape(-1)
        flat_mask = mask.reshape(-1)
        self.count = 0
        self._sum = 0.0
        self._max: Optional[float] = None
        for start in range(0, flat_values.numel(), chunk):
            selected = flat_values[start:start + chunk][flat_mask[start:start + chunk]]
            if selected.numel() == 0:
                continue
            self.count += selected.numel()
            self._sum += float(selected.sum())
            block_max = float(selected.max())
            self._max = block_max if self._max is None else max(self._max, block_max)
            del selected  # 下一块分配前释放，最多驻留一块 gather

    def __len__(self) -> int:
        return self.count

    def mean(self) -> float:
        return self._sum / self.count if self.count else 0.0

    def max(self) -> float:
        return self._max if self._max is not None else 0.0


def _normal_error_ratio(candidate: float, native: float, floor: float) -> Optional[float]:
    """Return a finite ratio, or None when the ratio is unavailable."""
    if not math.isfinite(candidate) or not math.isfinite(native):
        return None
    if candidate == 0 and native == 0:
        return 1.0
    denominator = max(native, floor)
    if denominator <= 0:
        return None
    ratio = candidate / denominator
    return ratio if math.isfinite(ratio) else None


def _normal_ratio_passed(ratio: Optional[float], limit: float) -> bool:
    return ratio is not None and math.isfinite(ratio) and (
        ratio <= limit
        or math.isclose(ratio, limit, rel_tol=_NORMAL_RATIO_REL_TOL, abs_tol=0.0)
    )


def _normal_failure_message(mare_ratio, mere_ratio):
    mare = f"{mare_ratio:.6g}" if mare_ratio is not None else "不可用（非有限比值）"
    mere = f"{mere_ratio:.6g}" if mere_ratio is not None else "不可用（非有限比值）"
    return (
        f"正常值域三方比对失败: MARE_ratio={mare} "
        f"(limit={_NORMAL_MARE_RATIO_LIMIT}), MERE_ratio={mere} "
        f"(limit={_NORMAL_MERE_RATIO_LIMIT})"
    )


def _compare_single_tensor(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold: float,
    dtype: str,
    native_output: Optional[torch.Tensor] = None,
) -> CompareResult:
    """对比单个张量（计算MERE/MARE + 小值域处理）

    生态算子精度标准：
    1. 平均相对误差（MERE）= avg(|actual - golden| / (|golden| + 1e-7))
    2. 最大相对误差（MARE）= max(|actual - golden| / (|golden| + 1e-7))
    3. 整体 MERE < threshold 且 MARE < 10 * threshold 时直接通过。
       否则有 native 时，正常值域兜底按比值阈值判定：MARE_ratio <= 5 且 MERE_ratio <= 1.5；
       分母以对应 dtype 的 small_value_threshold 夹底。
       无 native 时保留候选与 golden 的原两方比较策略。

    小值域处理（来自 docs/kernel_bench_design_v1.0.md）：
    当 |golden| < small_value_threshold 时，采用小值域通过标准：
    - ErrorCount = 统计满足 (|golden| < threshold 且 |actual - golden| > error) 的位置数
    - 通过标准: ErrorCount_npu / max(ErrorCount_native, 1) ≤ 2

    Args:
        output: NPU/算子输出张量
        golden: Golden参考输出（FP64精度）
        threshold: 精度阈值
        dtype: 数据类型字符串
        native_output: 同精度参考输出（可选）
                       如果不提供，使用 golden 截断到目标精度作为参考
    """
    # 检查形状 (比 shape 不碰数据, 无需先落 CPU)
    if output.shape != golden.shape:
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=f"形状不匹配: output={output.shape}, golden={golden.shape}"
        )

    # 检查候选输出的内存布局 (issue #146)。必须在下面的 .cpu() 之前做 --
    # 跨设备拷贝会归一化部分非连续张量的 stride, 检查放到之后就漏了。
    layout_error = non_contiguous_reason(output)
    if layout_error is not None:
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=layout_error,
        )

    # Golden runs on CPU while the AI op runs on NPU.
    # Normalize golden to CPU so subtract/equal don't trip on mixed devices.
    if golden.is_cuda or golden.device.type == "npu":
        golden = golden.cpu()

    # output 的 host 化按比较路径区分（内存优化：避免全量 host 副本重复驻留）：
    # - bit-exact / 整数路径需要原生 dtype 的 host 副本，在此 .cpu()；
    # - 浮点 MERE/MARE 路径在下方直接单次完成 device→host + fp64 转换，
    #   不再额外驻留一份原生 dtype 的全量 host 临时张量（巨型 case 可省数 GB 峰值）。
    needs_native_host_copy = (
        output.dtype in _INTEGER_DTYPES
        or (threshold == 0 and output.is_floating_point() and output.dtype in _BIT_VIEW)
    )
    if needs_native_host_copy and (output.is_cuda or output.device.type == "npu"):
        output = output.cpu()

    # Bit-exact 浮点路径: 当 threshold == 0 且为浮点 dtype 时, 通过 .view(int dtype)
    # 做字节级比较, 这样 +0.0 / -0.0 不会被 IEEE 754 相等性当作同值放过, NaN payload
    # 也按字节区分。整数 dtype 已经天然字节唯一, 走下方 torch.equal 路径即可。
    if threshold == 0 and output.is_floating_point():
        int_dtype = _BIT_VIEW.get(output.dtype)
        if int_dtype is not None:
            golden_cast = golden.to(output.dtype).contiguous()
            # F093: 检测 golden 在 cast 到更窄 output.dtype 时是否新产生 inf
            # (fp64 → fp16 时 |x|>65504 → inf)。若如此，bit-exact 比较的 fail
            # 实际是数据范围超 dtype，而非计算错误。给出明确告警避免误判方向。
            new_inf = torch.isinf(golden_cast) & ~torch.isinf(golden)
            if new_inf.any().item():
                _logger.warning(
                    "compare(bit-exact): golden cast 到 %s 时产生 %d 处新 inf "
                    "(数据范围超 output dtype)，bit-exact 失败可能源自数据溢出而非"
                    "计算错误。",
                    output.dtype, int(new_inf.sum().item()),
                )
            output_c = output.contiguous()
            out_bits = output_c.view(int_dtype)
            gold_bits = golden_cast.view(int_dtype)
            if torch.equal(out_bits, gold_bits):
                return CompareResult(
                    passed=True,
                    dtype=dtype,
                    threshold=threshold,
                    mere=0.0,
                    mare=0.0,
                    max_diff=0.0,
                    mean_diff=0.0,
                    mismatch_count=0,
                    total_count=output.numel(),
                    mismatch_ratio=0.0,
                    cancel_error_count=0,
                    cancel_cpu_error_count=0,
                    cancel_total_count=0,
                )
            mismatch_mask = out_bits != gold_bits
            mismatch_count = int(mismatch_mask.sum())
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=0.0,
                mean_diff=0.0,
                mismatch_count=mismatch_count,
                total_count=output.numel(),
                mismatch_ratio=mismatch_count / output.numel() if output.numel() > 0 else 0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
                error_msg=f"bit-exact 比较失败: {mismatch_count}/{output.numel()} 个元素字节不等 (包括 ±0.0、NaN payload、Inf 符号等差异)",
            )
        # 不支持 view 的浮点类型 (例如 fp8) 继续走下方 MERE/MARE 路径

    # 对于整数类型，使用绝对差值容差比较
    if output.dtype in _INTEGER_DTYPES:
        if torch.equal(output, golden):
            return CompareResult(
                passed=True,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=0.0,
                mean_diff=0.0,
                mismatch_count=0,
                total_count=output.numel(),
                mismatch_ratio=0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
            )

        # 不完全相等时，检查差值是否在容差范围内
        diff = torch.abs(output.long() - golden.long())
        mismatch_mask = diff > max(threshold, 0)
        mismatch_count = int(mismatch_mask.sum())

        diff_float = diff.float()
        if mismatch_count == 0:
            return CompareResult(
                passed=True,
                dtype=dtype,
                threshold=threshold,
                mere=0.0,
                mare=0.0,
                max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
                mean_diff=float(diff_float.mean()) if diff.numel() > 0 else 0.0,
                mismatch_count=0,
                total_count=output.numel(),
                mismatch_ratio=0.0,
                cancel_error_count=0,
                cancel_cpu_error_count=0,
                cancel_total_count=0,
            )

        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            mere=0.0,
            mare=0.0,
            max_diff=float(diff.max()) if diff.numel() > 0 else 0.0,
            mean_diff=float(diff_float.mean()) if diff.numel() > 0 else 0.0,
            mismatch_count=mismatch_count,
            total_count=output.numel(),
            mismatch_ratio=mismatch_count / output.numel() if output.numel() > 0 else 0.0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
            error_msg=f"整数类型差值超出容差({threshold}): {mismatch_count}/{output.numel()} 个元素不匹配",
        )

    # 浮点类型对比
    # Golden 主动截断到 output.dtype，模拟算子输出的精度限制
    target_dtype = output.dtype
    golden_truncated = golden.to(target_dtype).double()
    # 单次完成 device→host + fp64 转换（等价于旧的 .cpu() 再 .double()，数值不变），
    # 消除一份原生 dtype 的全量 host 临时副本；调用方持有的原张量不受影响。
    output_fp64 = output.to(device="cpu", dtype=torch.float64)

    # 处理 NaN
    if torch.any(torch.isnan(output_fp64)) or torch.any(torch.isnan(golden_truncated)):
        nan_out = torch.isnan(output_fp64)
        nan_gold = torch.isnan(golden_truncated)
        if not torch.all(nan_out == nan_gold):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg="NaN位置不匹配"
            )

    # 处理 Inf
    # 饱和边界处理：当一方 inf 另一方有限值时，将 inf 替换为 dtype 最大有限值后继续比较。
    # 这处理 fp16 饱和边界场景（NPU fp32→fp16 截断到 inf，golden fp64→fp16 未越界）。
    # 替换后由 MRE/MARE 决定 pass/fail——只有底层数值本身足够接近才能通过。
    replaced_inf = False
    inf_match_mask = torch.zeros_like(output_fp64, dtype=torch.bool)  # 初始化 inf 匹配掩码
    if torch.any(torch.isinf(output_fp64)) or torch.any(torch.isinf(golden_truncated)):
        inf_out = torch.isinf(output_fp64)
        inf_gold = torch.isinf(golden_truncated)
        inf_mismatch = inf_out != inf_gold

        if torch.any(inf_mismatch):
            # 获取目标 dtype 的最大有限值
            if target_dtype == torch.float16:
                max_finite = float(torch.finfo(torch.float16).max)  # 65504.0
            elif target_dtype == torch.bfloat16:
                max_finite = float(torch.finfo(torch.bfloat16).max)  # ~3.389e38
            elif target_dtype == torch.float32:
                max_finite = float(torch.finfo(torch.float32).max)
            else:
                max_finite = float(torch.finfo(target_dtype).max)

            mismatch_count = int(inf_mismatch.sum())
            # F087: 旧版无条件 print 到 stdout，批量评测污染输出 / 干扰 JSON 管道。
            # 改 logging.info 让用户/CI 通过 LOG_LEVEL 控制是否显示。
            _logger.info(
                "[inf_sat] %d element(s) saturated to inf on one side only, "
                "replacing inf with %s and continuing comparison",
                mismatch_count, max_finite,
            )

            # 替换 inf 为最大有限值（保留符号）
            if torch.any(inf_out & ~inf_gold):
                mask = inf_out & ~inf_gold
                output_fp64[mask] = torch.sign(output_fp64[mask]) * max_finite
                replaced_inf = True
            if torch.any(inf_gold & ~inf_out):
                mask = inf_gold & ~inf_out
                golden_truncated[mask] = torch.sign(golden_truncated[mask]) * max_finite
                replaced_inf = True

        # 双方都 inf 且符号相同的位置，直接视为匹配并排除后续比较
        both_inf = inf_out & inf_gold
        if torch.any(both_inf):
            if not torch.all(torch.sign(output_fp64[both_inf]) == torch.sign(golden_truncated[both_inf])):
                return CompareResult(
                    passed=False,
                    dtype=dtype,
                    threshold=threshold,
                    error_msg="Inf符号不匹配"
                )
            inf_match_mask[both_inf] = True  # 更新 inf 匹配掩码

    # 计算相对误差（标准公式）
    # 公式: |actual - golden| / (|golden| + 1e-7)
    # abs 就地写在减法的临时上：减法结果没有别的引用，省一份全量 fp64
    diff = (output_fp64 - golden_truncated).abs_()
    golden_abs = torch.abs(golden_truncated)
    denominator = golden_abs + 1e-7  # 防止除0

    relative_error = diff / denominator
    del denominator  # 只用于这一次除法；巨型 case 上留着它就是 8 B/元素的净损耗

    # 排除 NaN、Inf 和匹配的 Inf 位置
    valid_mask = ~(torch.isnan(relative_error) | torch.isinf(relative_error) | inf_match_mask)
    del inf_match_mask
    valid_relative_error = _MaskedStats(relative_error, valid_mask)

    if len(valid_relative_error) == 0:
        # F090: 旧版"全 NaN/Inf 匹配 = pass" 会掩盖"两个算子在同位置都产生 NaN
        # 但根因不同（一个溢出 / 一个除零）"的真实 bug。改为：
        # - 全部位置都是 NaN 且位置匹配 → 视为可疑（log warn，仍 passed=True
        #   维持兼容；调用方可基于 mismatch_count 决定是否进一步排查）
        # - 全部位置都是 Inf 且符号匹配 → 同上（已由 inf_match_mask 保证）
        # 若调用方需要更严格的判定，可在外层检查 (mere==0 and total>0)。
        has_nan = torch.isnan(output_fp64).any().item() if output_fp64.numel() > 0 else False
        if has_nan:
            _logger.warning(
                "compare: 所有有效位置均为 NaN/Inf 且匹配；两个算子均输出 NaN 的"
                "极端情况可能掩盖独立的数值 bug，请人工核查 case 输入分布。"
                " (total=%d, dtype=%s)",
                output_fp64.numel(), dtype,
            )
        return CompareResult(
            passed=True,
            dtype=dtype,
            threshold=threshold,
            mere=0.0,
            mare=0.0,
            max_diff=0.0,
            mean_diff=0.0,
            mismatch_count=0,
            total_count=output_fp64.numel(),
            mismatch_ratio=0.0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
        )

    mere = float(valid_relative_error.mean())
    mare = float(valid_relative_error.max())

    # 计算绝对差异统计
    valid_diff = _MaskedStats(diff, valid_mask)
    max_diff = valid_diff.max() if len(valid_diff) > 0 else 0.0
    mean_diff = valid_diff.mean() if len(valid_diff) > 0 else 0.0
    del valid_diff
    total_count = output_fp64.numel()

    # ============================================================
    # 第一阶段：整体相对误差判定（优先判定）
    # ============================================================
    # 原则：相对误差是主要判定标准，如果整体相对误差达标，直接通过
    # 小值域/相消判定只是为了处理"相对误差可能不合理"的特殊情况
    mare_threshold = 10 * threshold

    # 整体相对误差就是上面算过的 mere/mare（同一组有效位置、同一次归约），换个名字是为了
    # 和下面按值域细分的显示用 MERE/MARE 区分开
    overall_mere = mere
    overall_mare = mare
    del valid_relative_error

    # 如果整体相对误差通过，直接返回，不需要小值域/相消判定
    if overall_mere < threshold and overall_mare < mare_threshold:
        return CompareResult(
            passed=True,
            dtype=dtype,
            threshold=threshold,
            mere=overall_mere,
            mare=overall_mare,
            max_diff=max_diff,
            mean_diff=mean_diff,
            mismatch_count=0,
            total_count=total_count,
            mismatch_ratio=0.0,
            small_value_error_count=0,
            small_value_cpu_error_count=0,
            small_value_total_count=0,
            cancel_error_count=0,
            cancel_cpu_error_count=0,
            cancel_total_count=0,
        )

    # ============================================================
    # 第二阶段：分析失败原因，使用兜底判定
    # ============================================================
    # 整体相对误差不通过，需要分析是哪些位置导致的，并判断是否属于特殊情况

    # 找出相对误差超标的点
    mismatch_mask = relative_error > mare_threshold
    mismatch_mask[~valid_mask] = False
    mismatch_count = int(mismatch_mask.sum())
    del mismatch_mask

    # === 小值域处理 ===
    # 获取小值域阈值
    small_value_threshold = get_small_value_threshold(dtype)
    small_value_error = get_small_value_error(dtype)

    # 筛选小值域位置: |golden| < small_value_threshold
    small_value_mask = golden_abs < small_value_threshold
    small_value_mask[~valid_mask] = False  # 排除NaN/Inf
    small_value_total_count = int(small_value_mask.sum())

    # 小值域 NPU 错误计数: |golden| < small_value_threshold 且 |output - golden| > small_value_error
    small_value_npu_error_mask = small_value_mask & (diff > small_value_error)
    small_value_error_count = int(small_value_npu_error_mask.sum())
    del small_value_npu_error_mask, diff  # diff 到此为止只被这一处消费

    # 小值域同精度参考错误计数
    # 重要：native 和 NPU 的比较基准必须一致，都使用 golden_truncated（FP64 → target_dtype → FP64）
    # 这样才能公平比较两者在相同精度限制下与理论真值的误差差异
    if native_output is not None:
        # F088: native_output 应在 CPU 上但调用方契约不强制；防御性 device→host +
        # fp64 单次转换（等价于旧的 .cpu() 再 .double()，数值不变），
        # 避免 NPU 上的 .double() 失败或与 golden_truncated 跨设备运算，
        # 同时消除一份原生 dtype 的全量 host 临时副本。
        cpu_output_fp64 = native_output.to(device="cpu", dtype=torch.float64)
        # 同精度差异也与截断后的 golden 比较，保持基准一致
        cpu_diff = (cpu_output_fp64 - golden_truncated).abs_()
    else:
        # golden 截断到目标精度后再升到 FP64，这就是同精度下的"理想"输出
        cpu_output_fp64 = golden.to(target_dtype).double()
        # 与截断后的 golden 比较（此时 cpu_output_fp64 == golden_truncated，diff = 0）
        cpu_diff = (cpu_output_fp64 - golden_truncated).abs_()
    # 两个分支都只拿 cpu_output_fp64 算 cpu_diff；golden_truncated 也到此为止
    del cpu_output_fp64, golden_truncated

    # 同精度小值域错误计数: |golden| < threshold 且 |native_output - golden_truncated| > error
    cpu_small_value_mask = small_value_mask  # 直接使用 NPU 的小值域 mask，保证一致
    cpu_small_value_error_mask = cpu_small_value_mask & (cpu_diff > small_value_error)
    small_value_cpu_error_count = int(cpu_small_value_error_mask.sum())
    del cpu_small_value_error_mask

    # === 小值域兜底判定 ===
    # 判定标准严格对齐 docs/spec/benchmark_spec.md「小值域通过标准」：
    #     ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2
    # 即 CPU 无小值域错误时分母取 1，NPU 至多 2 个小值域错误仍判通过。
    if small_value_total_count > 0:
        max_cpu_error = max(small_value_cpu_error_count, 1)
        small_value_ratio = small_value_error_count / max_cpu_error
        small_value_passed = small_value_ratio <= 2
    else:
        small_value_passed = True

    # === 相消位置处理 ===
    # 获取相消阈值
    cancel_boundary = get_cancel_boundary(dtype)
    cancel_zero_threshold = get_cancel_zero_threshold(dtype)

    # 检测相消位置
    output_near_zero = torch.abs(output_fp64) < cancel_zero_threshold
    del output_fp64  # total_count 已在上面取过；后续只用掩码
    golden_in_cancel_range = (golden_abs < cancel_boundary) & (golden_abs >= small_value_threshold)
    cancel_mask = output_near_zero & golden_in_cancel_range & valid_mask
    del output_near_zero, golden_in_cancel_range
    cancel_total_count = int(cancel_mask.sum())

    # 相消位置"错误"判断：相对误差超过 mare_threshold
    cancel_npu_error_mask = cancel_mask & (relative_error > mare_threshold)
    cancel_error_count = int(cancel_npu_error_mask.sum())
    del cancel_npu_error_mask

    # CPU 相对误差超标计数
    # 就地除：cpu_diff 之后不再被读，复用它的存储省一份全量 fp64
    cpu_relative_error = cpu_diff.div_(golden_abs + 1e-7)
    del cpu_diff, golden_abs
    cancel_cpu_error_mask = cancel_mask & (cpu_relative_error > mare_threshold)
    cancel_cpu_error_count = int(cancel_cpu_error_mask.sum())
    del cancel_cpu_error_mask

    # 相消位置兜底判定：与小值域相同标准，对齐 benchmark_spec.md「相消位置通过标准」：
    #     ErrorCount_npu / max(ErrorCount_cpu, 1) ≤ 2
    if cancel_total_count > 0:
        max_cpu_cancel_error = max(cancel_cpu_error_count, 1)
        cancel_ratio = cancel_error_count / max_cpu_cancel_error
        cancel_passed = cancel_ratio <= 2
    else:
        cancel_passed = True

    # === 正常值域判定：先检查候选误差，未达标再做同精度兜底 ===
    normal_region_mask = ~small_value_mask & ~cancel_mask & valid_mask
    del small_value_mask, cancel_mask, valid_mask
    # 复用已有误差数组和 mask 分块归约；候选达标时跳过 native 归约及比值计算。
    normal_relative_error = _MaskedStats(relative_error, normal_region_mask)
    normal_total_count = len(normal_relative_error)
    normal_mare = normal_relative_error.max()
    normal_mere = normal_relative_error.mean()
    normal_mare_ratio = normal_mere_ratio = None
    normal_error_msg = None
    failure_type = None
    if native_output is not None and normal_total_count:
        if normal_mere < threshold and normal_mare < mare_threshold:
            normal_passed = True
        else:
            normal_native_error = _MaskedStats(cpu_relative_error, normal_region_mask)
            native_mare = normal_native_error.max()
            native_mere = normal_native_error.mean()
            if not math.isfinite(native_mare) or not math.isfinite(native_mere):
                # A nonfinite reference cannot serve as an infinitely loose baseline.
                normal_passed = False
                failure_type = FAILURE_TYPE_REFERENCE_INVALID
                normal_error_msg = (
                    "正常值域同精度参考失效: native MARE/MERE 含 NaN/Inf "
                    f"(MARE={native_mare}, MERE={native_mere})，无法进行三方比对"
                )
            else:
                # Tighten only the ratio floor for a stricter effective threshold;
                # retain the dtype threshold for small-value region partitioning.
                ratio_floor = small_value_threshold
                default_threshold = get_threshold(dtype)
                if 0 < threshold < default_threshold:
                    ratio_floor *= threshold / default_threshold
                normal_mare_ratio = _normal_error_ratio(normal_mare, native_mare, ratio_floor)
                normal_mere_ratio = _normal_error_ratio(normal_mere, native_mere, ratio_floor)
                normal_passed = (
                    _normal_ratio_passed(normal_mare_ratio, _NORMAL_MARE_RATIO_LIMIT)
                    and _normal_ratio_passed(normal_mere_ratio, _NORMAL_MERE_RATIO_LIMIT)
                )
                if not normal_passed:
                    normal_error_msg = _normal_failure_message(normal_mare_ratio, normal_mere_ratio)
    else:
        # 无 native 时沿用候选与 golden 的原两方比较严格判定。
        normal_passed = normal_mare <= mare_threshold
    del cpu_relative_error  # 正常域判定后立即释放；归约仅保留标量统计

    # 最终判定：三个值域各自兜底判定，全部通过才算通过。
    passed = normal_passed and small_value_passed and cancel_passed

    # 显示用 MERE/MARE
    del relative_error, normal_region_mask
    if not normal_passed:
        # 正常值域导致失败：显示排除小值域/相消后的 MERE/MARE
        if len(normal_relative_error) > 0:
            display_mere = normal_relative_error.mean()
            display_mare = normal_relative_error.max()
        else:
            display_mere = overall_mere
            display_mare = overall_mare
    elif passed:
        # 全部通过：显示排除小值域/相消后的 MERE/MARE，
        # 避免 overall 值被小值域的巨大相对误差拉高，造成"MARE=0.9 却通过"的误解
        if len(normal_relative_error) > 0:
            display_mere = normal_relative_error.mean()
            display_mare = normal_relative_error.max()
        else:
            display_mere = 0.0
            display_mare = 0.0
    else:
        # 正常值域通过、但小值域/相消兜底失败：显示 overall MERE/MARE，
        # 让用户看到实际误差大小（失败源于小值域/相消而非正常值域）。
        display_mere = overall_mere
        display_mare = overall_mare

    return CompareResult(
        passed=passed,
        dtype=dtype,
        threshold=threshold,
        mere=display_mere,
        mare=display_mare,
        max_diff=max_diff,
        mean_diff=mean_diff,
        mismatch_count=mismatch_count,
        total_count=total_count,
        mismatch_ratio=mismatch_count / total_count if total_count > 0 else 0.0,
        small_value_error_count=small_value_error_count,
        small_value_cpu_error_count=small_value_cpu_error_count,
        small_value_total_count=small_value_total_count,
        cancel_error_count=cancel_error_count,
        cancel_cpu_error_count=cancel_cpu_error_count,
        cancel_total_count=cancel_total_count,
        small_value_passed=small_value_passed,
        cancel_passed=cancel_passed,
        normal_total_count=normal_total_count,
        normal_passed=normal_passed,
        normal_mare=normal_mare,
        normal_mare_ratio=normal_mare_ratio,
        normal_mere_ratio=normal_mere_ratio,
        error_msg=normal_error_msg,
        failure_type=failure_type,
    )


def compare_tensors(
    output: Union[torch.Tensor, Tuple, List],
    golden: Union[torch.Tensor, Tuple, List],
    dtype: str = 'float32',
    threshold: Optional[float] = None,
    native_output: Optional[Union[torch.Tensor, Tuple, List]] = None,
    ignore_output_indices: Optional[List[int]] = None,
    custom_thresholds: Optional[Dict[str, float]] = None,
) -> CompareResult:
    """
    对比输出张量与Golden参考结果（采用MERE/MARE标准 + 小值域处理）

    Args:
        output: 算子输出（单个张量或多张量）
        golden: Golden参考输出（单个张量或多张量），通常为 FP64 精度
        dtype: 数据类型字符串
        threshold: 精度阈值（可选，默认根据dtype自动选择）
        native_output: 同精度参考输出（可选，用于正常值域三方比值及小值域/相消计数）
                       如果不提供，则使用 golden 截断到目标精度作为参考
        ignore_output_indices: 需要忽略对比的输出索引列表
        custom_thresholds: 自定义精度阈值表（优先级高于默认阈值）

    Returns:
        CompareResult: 对比结果

    通过条件:
        正常值域（有 native）: MERE < threshold 且 MARE < 10 * threshold 时通过，
                             否则要求 MARE_ratio <= 5 且 MERE_ratio <= 1.5
        正常值域（无 native）: 保留候选与 golden 的原两方比较严格判定
        小值域: ErrorCount_npu / max(ErrorCount_native, 1) ≤ 2
    """
    # 获取阈值（优先使用自定义阈值）
    if custom_thresholds is None:
        custom_thresholds = {}

    def _get_output_threshold(dtype_str: str) -> float:
        """获取单个输出的阈值（优先自定义，其次默认）"""
        dtype_lower = dtype_str.lower()
        if dtype_lower in custom_thresholds:
            return custom_thresholds[dtype_lower]
        return get_threshold(dtype_str)

    if threshold is None:
        threshold = _get_output_threshold(dtype)

    try:
        # 处理多输出情况
        outputs = _normalize_outputs(output)
        goldens = _normalize_outputs(golden)
        native_outputs = _normalize_outputs(native_output) if native_output is not None else None

        if len(outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"输出数量不匹配: output={len(outputs)}, golden={len(goldens)}"
            )

        if native_outputs is not None and len(native_outputs) != len(goldens):
            return CompareResult(
                passed=False,
                dtype=dtype,
                threshold=threshold,
                error_msg=f"同精度输出数量不匹配: native_output={len(native_outputs)}, golden={len(goldens)}"
            )

        # 逐个对比
        all_passed = True
        mere_sum = 0.0
        mare_max = 0.0
        max_diff = 0.0
        mean_diff = 0.0
        mismatch_count = 0
        total_count = 0
        small_value_error_count = 0
        small_value_cpu_error_count = 0
        small_value_total_count = 0
        cancel_error_count = 0
        cancel_cpu_error_count = 0
        cancel_total_count = 0
        all_small_value_passed = True
        all_cancel_passed = True
        normal_total_count = 0
        all_normal_passed = True
        normal_mare = 0.0
        normal_mare_ratios = []
        normal_mere_ratios = []

        # 记录每个输出的独立判定结果
        single_output_results: List[SingleOutputResult] = []

        for i, (out_tensor, gold_tensor) in enumerate(zip(outputs, goldens)):
            # 处理可选输出：双方均为 None 则跳过，否则报错
            if out_tensor is None or gold_tensor is None:
                both_none = out_tensor is None and gold_tensor is None
                single_output_results.append(SingleOutputResult(
                    index=i,
                    name="",
                    dtype="none",
                    passed=both_none,
                    error_msg="" if both_none else f"output[{i}] is None (candidate={'None' if out_tensor is None else 'Tensor'}, golden={'None' if gold_tensor is None else 'Tensor'})",
                    metadata={'dtype_category': 'none'},
                ))
                continue

            # 跳过不需要对比的输出
            if ignore_output_indices and i in ignore_output_indices:
                # 创建跳过标记的 SingleOutputResult
                single_output_results.append(SingleOutputResult(
                    index=i,
                    name="",  # 名称由调用方填充
                    dtype=str(out_tensor.dtype).replace('torch.', ''),
                    passed=True,  # 跳过的输出视为通过
                    error_msg="(跳过对比)",
                    metadata={'dtype_category': 'int' if out_tensor.dtype in _INTEGER_DTYPES else 'float'},
                ))
                continue

            # 根据每个输出的实际 dtype 获取阈值（优先自定义阈值）
            out_dtype_str = str(out_tensor.dtype).replace('torch.', '')
            out_threshold = _get_output_threshold(out_dtype_str)
            out_dtype_category = 'int' if out_tensor.dtype in _INTEGER_DTYPES else 'float'

            native_tensor = native_outputs[i] if native_outputs is not None else None
            result = _compare_single_tensor(out_tensor, gold_tensor, out_threshold, out_dtype_str, native_tensor)

            # 转换 CompareResult 到 SingleOutputResult
            single_result = SingleOutputResult(
                index=i,
                name="",  # 名称由调用方填充
                dtype=out_dtype_str,
                passed=result.passed,
                mismatch_count=result.mismatch_count,
                total_count=result.total_count,
                error_msg=result.error_msg or "",
                metadata={
                    'dtype_category': out_dtype_category,
                    'threshold': out_threshold,
                    'mere': result.mere,
                    'mare': result.mare,
                    'max_diff': result.max_diff,
                    'mean_diff': result.mean_diff,
                    'max_abs_diff': int(result.max_diff) if out_dtype_category == 'int' else 0,
                    'small_value_error_count': result.small_value_error_count,
                    'small_value_cpu_error_count': result.small_value_cpu_error_count,
                    'small_value_total_count': result.small_value_total_count,
                    'cancel_error_count': result.cancel_error_count,
                    'cancel_cpu_error_count': result.cancel_cpu_error_count,
                    'cancel_total_count': result.cancel_total_count,
                    'small_value_passed': result.small_value_passed,
                    'cancel_passed': result.cancel_passed,
                    'normal_total_count': result.normal_total_count,
                    'normal_passed': result.normal_passed,
                    'normal_mare': result.normal_mare,
                    'normal_mare_ratio': result.normal_mare_ratio,
                    'normal_mere_ratio': result.normal_mere_ratio,
                    'normal_mare_ratio_limit': _NORMAL_MARE_RATIO_LIMIT,
                    'normal_mere_ratio_limit': _NORMAL_MERE_RATIO_LIMIT,
                },
            )
            single_output_results.append(single_result)
            if result.failure_type is not None:
                single_result.metadata['failure_type'] = result.failure_type

            is_passed = result.passed
            all_passed = all_passed and is_passed
            mere_sum += result.mere * result.total_count
            mare_max = max(mare_max, result.mare)
            max_diff = max(max_diff, result.max_diff)
            mean_diff += result.mean_diff * result.total_count
            mismatch_count += result.mismatch_count
            total_count += result.total_count
            small_value_error_count += result.small_value_error_count
            small_value_cpu_error_count += result.small_value_cpu_error_count
            small_value_total_count += result.small_value_total_count
            cancel_error_count += result.cancel_error_count
            cancel_cpu_error_count += result.cancel_cpu_error_count
            cancel_total_count += result.cancel_total_count
            all_small_value_passed = all_small_value_passed and result.small_value_passed
            all_cancel_passed = all_cancel_passed and result.cancel_passed
            normal_total_count += result.normal_total_count
            all_normal_passed = all_normal_passed and result.normal_passed
            normal_mare = max(normal_mare, result.normal_mare)
            if result.normal_mare_ratio is not None:
                normal_mare_ratios.append(result.normal_mare_ratio)
            if result.normal_mere_ratio is not None:
                normal_mere_ratios.append(result.normal_mere_ratio)

        if total_count > 0:
            mere = mere_sum / total_count
            mean_diff = mean_diff / total_count
        else:
            mere = 0.0

        # 最终通过条件
        passed = all_passed

        # 确定返回的 dtype 和 threshold
        # 如果有失败输出，返回第一个失败输出的阈值信息
        # 否则返回第一个输出的阈值信息
        result_dtype = dtype
        result_threshold = threshold
        for sr in single_output_results:
            if not sr.passed and not sr.error_msg.startswith("(跳过"):
                result_dtype = sr.dtype
                result_threshold = sr.metadata.get('threshold', threshold)
                break

        invalid_reference = next((sr for sr in single_output_results
                                  if sr.metadata.get('failure_type') == FAILURE_TYPE_REFERENCE_INVALID), None)
        return CompareResult(
            passed=passed,
            dtype=result_dtype,
            threshold=result_threshold,
            mere=mere,
            mare=mare_max,
            max_diff=max_diff,
            mean_diff=mean_diff,
            mismatch_count=mismatch_count,
            total_count=total_count,
            mismatch_ratio=mismatch_count / total_count if total_count > 0 else 0.0,
            small_value_error_count=small_value_error_count,
            small_value_cpu_error_count=small_value_cpu_error_count,
            small_value_total_count=small_value_total_count,
            cancel_error_count=cancel_error_count,
            cancel_cpu_error_count=cancel_cpu_error_count,
            cancel_total_count=cancel_total_count,
            small_value_passed=all_small_value_passed,
            cancel_passed=all_cancel_passed,
            normal_total_count=normal_total_count,
            normal_passed=all_normal_passed,
            normal_mare=normal_mare,
            normal_mare_ratio=max(normal_mare_ratios, default=None),
            normal_mere_ratio=max(normal_mere_ratios, default=None),
            output_results=single_output_results,  # 新增：各输出独立结果
            failure_type=FAILURE_TYPE_REFERENCE_INVALID if invalid_reference is not None else None,
            error_msg=invalid_reference.error_msg if invalid_reference is not None else None,
        )

    except Exception as e:
        # 顶层 except 会吞掉所有内部逻辑（in-place mutation、shape 比较、
        # MERE/MARE 计算等）的 traceback。把堆栈附在 error_msg 末尾，方便排查。
        return CompareResult(
            passed=False,
            dtype=dtype,
            threshold=threshold,
            error_msg=f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )


def compare_with_custom_threshold(
    output: torch.Tensor,
    golden: torch.Tensor,
    threshold_dict: Dict[str, float] = None,
) -> CompareResult:
    """
    使用自定义阈值表对比

    Args:
        output: 算子输出张量
        golden: Golden参考输出张量
        threshold_dict: 自定义阈值表，格式为 {dtype: threshold}

    Returns:
        CompareResult: 对比结果
    """
    from .thresholds import PRECISION_THRESHOLDS as _DEFAULT_THRESHOLDS

    if threshold_dict is None:
        threshold_dict = _DEFAULT_THRESHOLDS

    # 从张量推断dtype
    dtype_str = str(output.dtype).replace('torch.', '')
    threshold = get_threshold(dtype_str)
    if dtype_str.lower() in threshold_dict:
        threshold = threshold_dict[dtype_str.lower()]

    return compare_tensors(output, golden, dtype_str, threshold)
