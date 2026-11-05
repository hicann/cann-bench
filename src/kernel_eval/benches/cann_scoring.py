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

"""
CANN 评分模块

包含：
- 权重常量 (WEIGHT_COMPILATION, WEIGHT_FUNCTION, WEIGHT_PERFORMANCE —— 1
  档的兼容别名；权重档的单一事实来源在 base/score_weights.py)
- OperatorScoreInfo: 算子级得分信息
- per_case_sol_score: 单用例 SOL-Score 计算
- aggregate_eq4: 单算子综合分聚合
- ScoringCalculator: 评分计算器
- CannScoringScheme, SimpleComparisonScheme, RecordingOnlyScheme: 评分方案实现

评分公式 (docs/spec/benchmark_spec.md §3.3 / Eq. 3, 4, 5):
- 单用例性能得分: score_i = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))
- 单算子综合评分: EachOperatorScore =
      [ w_c · δ_pass + w_b · δ_all + Σ_i δ_acc,i · (w_f + w_p · score_i) / len(cases) ] · 100
  权重档: 2（tasks 2.x）w_c=0.05, w_f=0.20, w_b=0.05, w_p=0.70；
  1（历史档，可切换复算）w_c=0.2, w_f=0.3, w_p=0.5, w_b=0 (sum=1, 单算子满分=100)
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..base.perf_metric import OPERATOR_SCORE_CAP, compute_perf_score
from ..base.score_weights import resolve_score_weights
from ..base.scoring import ScoringScheme, CaseScoreInfo
from ..base.result import (
    FAILURE_TYPE_REFERENCE_INVALID,
    PerfResult,
    is_compile_runtime_case_failure,
)
from ..eval.results import EvalOperatorResult  # 直接导入模块，避免循环导入


_logger = logging.getLogger(__name__)


# === 权重配置（bench.tex §3.5 = 1 档）===
# 仅作兼容别名保留——权重档的单一事实来源在 base/score_weights.py，
# aggregate_eq4 未显式传权重时按环境变量 CANN_BENCH_SCORE_WEIGHTS 解析
# （评测入口从实际题集 VERSION 大版本注入默认档）。

WEIGHT_COMPILATION = 0.2  # w_c (1)
WEIGHT_FUNCTION = 0.3  # w_f (1)
WEIGHT_PERFORMANCE = 0.5  # w_p (1)

NO_NPU_PERF_ERROR_CODE = "no_npu_kernel_detected"
NO_NPU_PERF_ERROR = "未检测到 NPU 算子执行，疑似 CPU fallback，反作弊触发。"


# === 算子级得分信息 ===


@dataclass
class OperatorScoreInfo:
    """算子级得分信息（per operator）

    包含编译/运行、精度、性能三轴得分，以及 per-case 调试用分数列表。
    """

    operator: str = ""
    rel_path: str = ""
    pass_rate: float = 0.0
    avg_speedup: float = 0.0  # 诊断保留
    compile_passed: bool = False
    passed_cases: int = 0
    total_cases: int = 0
    # bench.tex 三轴得分（已按 w_c/w_f/w_p 加权后的贡献，并已归一化到 0-100 量纲）
    compilation_score: Optional[float] = 0.0
    compile_runtime_fail_cases: int = 0
    function_score: Optional[float] = 0.0
    bonus_score: Optional[float] = (
        0.0  # 全用例精度通过 bonus（w_b·δ_all·100；1 档 w_b=0 恒为 0）
    )
    performance_score: Optional[float] = 0.0
    total_score: Optional[float] = 0.0  # 单算子综合得分，[0, 100]
    # 调试用：每个用例的 hardware-anchored 分数，None 表示数据不全或未通过精度门
    per_case_scores: List[Optional[float]] = field(default_factory=list)
    score_error_code: Optional[str] = None
    score_error: Optional[str] = None
    zeroed_by_no_npu_perf: bool = False
    reference_invalid_cases: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operator": self.operator,
            "rel_path": self.rel_path,
            "pass_rate": self.pass_rate,
            "avg_speedup": self.avg_speedup,
            "compile_passed": self.compile_passed,
            "passed_cases": self.passed_cases,
            "total_cases": self.total_cases,
            "compilation_score": self.compilation_score,
            "compile_runtime_score": self.compilation_score,
            "compile_runtime_fail_cases": self.compile_runtime_fail_cases,
            "function_score": self.function_score,
            "bonus_score": self.bonus_score,
            "performance_score": self.performance_score,
            "total_score": self.total_score,
            "per_case_scores": self.per_case_scores,
            "score_error_code": self.score_error_code,
            "score_error": self.score_error,
            "zeroed_by_no_npu_perf": self.zeroed_by_no_npu_perf,
            "reference_invalid_cases": self.reference_invalid_cases,
        }


# === 单用例 SOL-Score 计算 ===


def per_case_sol_score(
    t_baseline: float,
    t_cand: float,
    t_hw: float,
    rel_path: Optional[str] = None,
    metric: Optional[str] = None,
) -> Optional[float]:
    """bench.tex Eq. 3。T_cand 或 T_HW 异常 / 公式无定义时返回 None。

    公式本体委托 ``base.perf_metric.compute_perf_score``：默认 ``eq3``
    （减法版，与历史逐位一致）；``metric="eq3_log"`` 或环境变量
    ``CANN_BENCH_PERF_METRIC=eq3_log`` 切换对数同构版（评估中）。
    本函数保留全部锚点守卫与日志语义。

    注意 Eq.3 是**饱和型**指标，不是加速比（speedup）：它衡量候选逼近硬件
    理论上界 T_HW 的程度，而非比 baseline 快多少倍。baseline 很慢
    （T_baseline ≫ T_HW）时，即使 speedup 巨大，本分数也只趋近 1 而非线性放大。

    设计取舍：
    - 不对返回值做上界截断——当 T_cand < T_HW 时 score > 1.0 的"超额分"
      是有意保留的，用以激励算法突破当前硬件下界（T_cand < T_HW 走 HAP 奖励
      分支，受 PER_CASE_SCORE_CAP 封顶；单算子分可 >100，受 OPERATOR_SCORE_CAP 封顶）。
    - T_baseline < T_HW 视为基线/T_HW 标定可疑（应在用例侧人工核查），
      但仍计算分数继续输出，仅 warn 一次（per rel_path × 数值组合）。
    - **T_baseline 缺失（≤ 0）但 T_HW 已知时**，按 fallback 规则取
      ``max(T_HW * 10, 10)`` 作为代理基线继续打分（F054：fallback 路径也 warn）；
      fallback 不回写到 cases，仅在运行时使用（约定：让缺基线的 case 也能拿到一个
      合理的相对分数，不会因为基线漏填整体被静默置 0）。

    Args:
        rel_path: 算子相对路径，用于日志去重 key（F058）+ 错误溯源。可选。

    F057: 返回 None 的三种成因分别 warn，避免被 aggregate_eq4 统一吞为"缺锚点"。
    """
    # F057 成因 (a): T_cand 或 T_HW 异常
    if math.isnan(t_cand) or math.isnan(t_hw) or t_cand <= 0 or t_hw <= 0:
        _warn_invalid_anchor(rel_path, t_cand, t_hw)
        return None
    if math.isnan(t_baseline) or t_baseline <= 0:
        # F057 成因 (c) + F054: baseline 缺失走 fallback，warn 一次让用户感知
        _warn_fallback_baseline(rel_path, t_hw)
        t_baseline = _fallback_baseline_from_hw(t_hw)
    elif t_baseline < t_hw:
        _warn_baseline_below_hw(t_baseline, t_hw, rel_path)
    score = compute_perf_score(t_baseline, t_cand, t_hw, metric=metric)
    if score is None:
        # F057 成因 (b): 公式在该锚点组合下无定义——
        # eq3: denom ≤ 0（罕见，但 T_cand < T_HW 且 T_baseline ≈ T_HW 时可能）；
        # eq3_log: B ≤ H（lg(B/H) 非正）或对数分母 ≤ 0，同类边界。
        _warn_denom_nonpositive(rel_path, t_baseline, t_cand, t_hw)
        return None
    return score


def _fallback_baseline_from_hw(t_hw: float) -> float:
    """缺基线时的代理 baseline：max(T_HW * 10, 10 us)。"""
    return max(t_hw * 10.0, 10.0)


# 同一 (rel_path, T_baseline, T_HW) 组合只提示一次，避免成批用例时刷屏。
# F058: key 加入 rel_path 标识，不同算子的相同数值组合不会互相抑制。
_BASELINE_HW_WARNED: set = set()
_FALLBACK_BASELINE_WARNED: set = set()
_INVALID_ANCHOR_WARNED: set = set()
_DENOM_WARNED: set = set()
_PERF_MISSING_WARNED_RUN: set = set()


def _warn_baseline_below_hw(
    t_baseline: float, t_hw: float, rel_path: Optional[str] = None
) -> None:
    """F059: 改用 _logger.warning 输出到 stderr（默认 logging handler），不污染 stdout。"""
    key = (rel_path, round(t_baseline, 4), round(t_hw, 4))
    if key in _BASELINE_HW_WARNED:
        return
    _BASELINE_HW_WARNED.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%sper_case_sol_score: T_baseline (%.4f us) < T_HW (%.4f us)。"
        "请核查该用例 baseline_perf_us / t_hw_us 是否标定正确。",
        op_prefix,
        t_baseline,
        t_hw,
    )


def _warn_fallback_baseline(rel_path: Optional[str], t_hw: float) -> None:
    key = (rel_path, round(t_hw, 4))
    if key in _FALLBACK_BASELINE_WARNED:
        return
    _FALLBACK_BASELINE_WARNED.add(key)
    proxy = _fallback_baseline_from_hw(t_hw)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%sper_case_sol_score: baseline_perf_us 缺失/≤0，使用代理基线 "
        "max(T_HW*10, 10) = %.2f us（T_HW=%.4f）。该用例的分数基于代理基线计算，"
        "精度可能降低；请补 cases.yaml 的 baseline_perf_us。",
        op_prefix,
        proxy,
        t_hw,
    )


def _warn_invalid_anchor(rel_path: Optional[str], t_cand: float, t_hw: float) -> None:
    key = (rel_path, round(t_cand, 4), round(t_hw, 4))
    if key in _INVALID_ANCHOR_WARNED:
        return
    _INVALID_ANCHOR_WARNED.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%sper_case_sol_score: T_cand (%.4f us) 或 T_HW (%.4f us) 非正，"
        "本 case 性能项按 None 处理。",
        op_prefix,
        t_cand,
        t_hw,
    )


def _warn_denom_nonpositive(
    rel_path: Optional[str], t_baseline: float, t_cand: float, t_hw: float
) -> None:
    key = (rel_path, round(t_baseline, 4), round(t_cand, 4), round(t_hw, 4))
    if key in _DENOM_WARNED:
        return
    _DENOM_WARNED.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%sper_case_sol_score: 性能分公式在该锚点组合下无定义 "
        "(T_baseline=%.4f us, T_cand=%.4f us, T_HW=%.4f us)。本 case 性能项 None。"
        "eq3: denom = (T_cand-T_HW)+(T_baseline-T_HW) ≤ 0（T_baseline 与 T_HW 几乎"
        "相等且 T_cand 远小于 T_HW）；eq3_log: B ≤ H 或对数分母 ≤ 0。",
        op_prefix,
        t_baseline,
        t_cand,
        t_hw,
    )


def _warn_perf_anchor_missing(
    n_func_pass: int, n_perf_missing: int, rel_path: Optional[str] = None
) -> None:
    """通知调用者：精度通过的 case 中有 N 个缺基线/T_HW 锚点，按 §3.3 计为 0 性能分。"""
    key = (rel_path, n_func_pass, n_perf_missing)
    if key in _PERF_MISSING_WARNED_RUN:
        return
    _PERF_MISSING_WARNED_RUN.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%saggregate_eq4: %d / %d 个精度通过的 case 缺 baseline_perf_us / t_hw_us 锚点，"
        "按 spec §3.3 按 0 计入性能项。若分数偏低，请先核查这些 case 的基线是否已填充。",
        op_prefix,
        n_perf_missing,
        n_func_pass,
    )


_NO_NPU_PERF_WARNED_RUN: set = set()


def _warn_no_npu_perf(
    n_func_pass: int, n_no_perf_pass: int, rel_path: Optional[str] = None
) -> None:
    """精度通过但 profiler 已产出 CSV 却无 NPU kernel（疑似 CPU fallback），整算子置 0 分。"""
    key = (rel_path, n_func_pass, n_no_perf_pass)
    if key in _NO_NPU_PERF_WARNED_RUN:
        return
    _NO_NPU_PERF_WARNED_RUN.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%saggregate_eq4: %d / %d 个精度通过的 case 已采集 kernel_details.csv "
        "但未检测到 NPU 算子性能数据，疑似 CPU fallback 或未执行提交的 NPU kernel，"
        "整算子按 0 分处理。",
        op_prefix,
        n_no_perf_pass,
        n_func_pass,
    )


_PERF_COLLECTION_FAILED_WARNED_RUN: set = set()


def _warn_perf_collection_failed(
    n_collection_failed: int, n_func_pass: int, rel_path: Optional[str] = None
) -> None:
    """精度通过但 profiler 采集功能异常（未产出 CSV 等），该 case 性能项按 0 计入，
    不触发反作弊。"""
    key = (rel_path, n_func_pass, n_collection_failed)
    if key in _PERF_COLLECTION_FAILED_WARNED_RUN:
        return
    _PERF_COLLECTION_FAILED_WARNED_RUN.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%saggregate_eq4: %d / %d 个精度通过的 case 因 profiler 采集功能异常"
        "（未产出 kernel_details.csv 等）无性能数据，该 case 性能项按 0 计入，"
        "不触发反作弊。请排查 profiler/环境问题。",
        op_prefix,
        n_collection_failed,
        n_func_pass,
    )


# === 单算子综合分聚合 ===


def aggregate_eq4(
    compile_passed: bool,
    total_cases: int,
    case_scores: List[Tuple[bool, Optional[float]]],
    wc: Optional[float] = None,
    wf: Optional[float] = None,
    wp: Optional[float] = None,
    rel_path: Optional[str] = None,
    n_no_perf_pass: int = 0,
    n_compile_runtime_fail: int = 0,
    wb: Optional[float] = None,
    weights: Optional[str] = None,
    n_reference_invalid: int = 0,
) -> Dict[str, Any]:
    """Eq.4 单算子综合分聚合——单一事实来源。

    EachOperatorScore =
        [ w_c·δ_compile_runtime + w_b·δ_all + Σ_i δ_acc,i (w_f + w_p·score_i) / N ] · 100

    其中 δ_all = ∏_i δ_acc,i（全部 N=total_cases 个用例精度通过才为 1，
    漏跑视为未通过）；反作弊置零路径同样清零 bonus。

    ``wc/wf/wp/wb`` 传 None 时按 ``weights`` 权重档解析：显式
    ``weights="1"`` / ``weights="2"`` > ``CANN_BENCH_SCORE_WEIGHTS`` >
    评测入口从实际 bench_root/metadata/VERSION 大版本注入的默认档。
    无来源时抛 ValueError。档 "1" 为 0.2/0.3/0.5/0，档 "2" 为
    0.05/0.20/0.70/0.05；显式单项权重优先。与 eq3 / eq3_log 正交。

    n_no_perf_pass 仅统计 kernel_details.csv 已产出但无 NPU kernel 的 case
    （疑似 CPU fallback，触发反作弊）。profiler 采集功能异常（未产出 CSV）
    的 case 不计入此参数，由调用方在 calculate_operator_score 中提前识别并
    归入 n_perf_missing，不触发反作弊。
    """
    # 完整显式数值权重不依赖默认档，保留外部复算脚本的调用语义。
    if weights is not None or any(value is None for value in (wc, wf, wp, wb)):
        profile = resolve_score_weights(weights)
        wc = profile.wc if wc is None else wc
        wf = profile.wf if wf is None else wf
        wp = profile.wp if wp is None else wp
        wb = profile.wb if wb is None else wb

    if n_reference_invalid:
        return {
            "compilation_score": None,
            "function_score": None,
            "bonus_score": None,
            "performance_score": None,
            "total_score": None,
            "per_case_scores": [None] * total_cases,
            "n_reference_invalid": n_reference_invalid,
            "n_compile_runtime_fail": n_compile_runtime_fail,
            "score_error_code": FAILURE_TYPE_REFERENCE_INVALID,
            "score_error": "参考失效，本次算子评分不可判定；请修复参考后重新评测。",
        }
    n_func_pass = 0
    n_perf_missing = 0
    perf_score_sum = 0.0
    per_case_scores: List[Optional[float]] = []

    if compile_passed:
        for success, score_i in case_scores:
            if not success:
                per_case_scores.append(None)
                continue
            n_func_pass += 1
            per_case_scores.append(score_i)
            if score_i is None or math.isnan(score_i):
                n_perf_missing += 1
                perf_score_sum += 0.0
            else:
                perf_score_sum += score_i
    else:
        per_case_scores = [None] * total_cases

    if n_perf_missing > 0:
        _warn_perf_anchor_missing(n_func_pass, n_perf_missing, rel_path)

    if compile_passed and n_no_perf_pass > 0:
        _warn_no_npu_perf(n_func_pass, n_no_perf_pass, rel_path)
        return {
            "compilation_score": 0.0,
            "function_score": 0.0,
            "bonus_score": 0.0,
            "performance_score": 0.0,
            "total_score": 0.0,
            "per_case_scores": per_case_scores,
            "n_func_pass": n_func_pass,
            "n_no_perf_pass": n_no_perf_pass,
            "n_compile_runtime_fail": n_compile_runtime_fail,
            "score_error_code": NO_NPU_PERF_ERROR_CODE,
            "score_error": NO_NPU_PERF_ERROR,
            "zeroed_by_no_npu_perf": True,
        }

    compile_runtime_pass_cases = 0
    if compile_passed and total_cases > 0:
        compile_runtime_pass_cases = max(total_cases - n_compile_runtime_fail, 0)
    delta_all = (
        1 if (compile_passed and total_cases > 0 and n_func_pass == total_cases) else 0
    )
    compilation_score = (
        (compile_runtime_pass_cases * wc / total_cases) * 100.0
        if total_cases > 0
        else 0.0
    )
    function_score = (n_func_pass * wf / total_cases) * 100.0
    bonus_score = wb * delta_all * 100.0
    performance_score = (perf_score_sum * wp / total_cases) * 100.0
    total_score = compilation_score + function_score + bonus_score + performance_score
    # HAP > 1 的奖励分允许 total_score 突破名义满分 100，但以 OPERATOR_SCORE_CAP
    # 封顶，保证单算子对 Level 分 / 总分的贡献有界；未截断值随结果一并返回。
    total_score_uncapped = total_score
    if total_score > OPERATOR_SCORE_CAP:
        _logger.warning(
            "aggregate_eq4: %s total_score %.2f 超过 OPERATOR_SCORE_CAP=%.0f，已封顶。",
            rel_path,
            total_score_uncapped,
            OPERATOR_SCORE_CAP,
        )
        total_score = OPERATOR_SCORE_CAP

    return {
        "compilation_score": compilation_score,
        "function_score": function_score,
        "bonus_score": bonus_score,
        "performance_score": performance_score,
        "total_score": total_score,
        "total_score_uncapped": total_score_uncapped,
        "per_case_scores": per_case_scores,
        "n_func_pass": n_func_pass,
        "n_no_perf_pass": n_no_perf_pass,
        "n_compile_runtime_fail": n_compile_runtime_fail,
        "score_error_code": None,
        "score_error": None,
        "zeroed_by_no_npu_perf": False,
    }


# === 评分计算器 ===


class ScoringCalculator:
    """评分计算器

    ``wc/wf/wp/wb`` 传 None（默认）时按 ``weights`` 权重档在**每次聚合时**
    解析（支持环境变量 ``CANN_BENCH_SCORE_WEIGHTS`` 运行期切换）；显式传入
    的值优先于权重档。
    """

    def __init__(
        self,
        wc: Optional[float] = None,
        wf: Optional[float] = None,
        wp: Optional[float] = None,
        wb: Optional[float] = None,
        weights: Optional[str] = None,
    ):
        self.wc = wc
        self.wf = wf
        self.wp = wp
        self.wb = wb
        self.weights = weights

    def _effective_weights(self):
        """解析当前生效的 (wc, wf, wp, wb)——仅用于展示/分解，聚合在
        aggregate_eq4 内部自行解析（同一规则）。"""
        explicit = (self.wc, self.wf, self.wp, self.wb)
        if self.weights is None and all(value is not None for value in explicit):
            return explicit
        profile = resolve_score_weights(self.weights)
        return (
            profile.wc if self.wc is None else self.wc,
            profile.wf if self.wf is None else self.wf,
            profile.wp if self.wp is None else self.wp,
            profile.wb if self.wb is None else self.wb,
        )

    def calculate_operator_score(self, result: EvalOperatorResult) -> OperatorScoreInfo:
        """单算子综合得分 (Eq. 4)。"""
        compile_passed = result.compile_passed
        declared = result.total_cases
        run = len(result.results)

        # F062: 空壳算子（0 声明 + 0 实测）直接 0 分
        if declared == 0 and run == 0:
            return OperatorScoreInfo(
                operator=result.operator,
                rel_path=result.rel_path,
                pass_rate=0.0,
                avg_speedup=0.0,
                compile_passed=compile_passed,
                passed_cases=0,
                total_cases=0,
                compilation_score=0.0,
                function_score=0.0,
                performance_score=0.0,
                total_score=0.0,
                per_case_scores=[],
            )

        total_cases = max(declared, run, 1)

        case_scores: List[Tuple[bool, Optional[float]]] = []
        n_no_perf_pass = 0
        n_compile_runtime_fail = 0
        n_collection_failed = 0
        n_func_pass = 0
        for case in result.results:
            if not case.success:
                if is_compile_runtime_case_failure(case):
                    n_compile_runtime_fail += 1
                case_scores.append((case.success, None))
                continue
            n_func_pass += 1
            if case.perf_result is None or case.perf_result.elapsed_us <= 0:
                # 区分采集功能异常与 CPU fallback：
                # - perf_collection_failed=True（或 perf_result=None）：profiler 未产出
                #   kernel_details.csv 等，属采集功能异常，不触发反作弊。
                # - perf_collection_failed=False：CSV 已产出但无有效 NPU kernel，
                #   疑似 CPU fallback，触发反作弊。
                is_collection_failure = (
                    case.perf_result is None
                    or case.perf_result.metadata.get("perf_collection_failed", False)
                )
                if is_collection_failure:
                    n_collection_failed += 1
                    case_scores.append((True, None))
                    continue
                n_no_perf_pass += 1
                case_scores.append((True, None))
                continue
            score_i = per_case_sol_score(
                case.baseline_perf_us,
                case.perf_result.elapsed_us,
                case.t_hw_us,
                rel_path=result.rel_path,
            )
            case_scores.append((True, score_i))

        if n_collection_failed > 0:
            _warn_perf_collection_failed(
                n_collection_failed,
                n_func_pass,
                result.rel_path,
            )

        agg = aggregate_eq4(
            compile_passed=compile_passed,
            total_cases=total_cases,
            case_scores=case_scores,
            wc=self.wc,
            wf=self.wf,
            wp=self.wp,
            rel_path=result.rel_path,
            n_no_perf_pass=n_no_perf_pass,
            n_compile_runtime_fail=n_compile_runtime_fail,
            wb=self.wb,
            weights=self.weights,
            n_reference_invalid=result.reference_invalid_cases,
        )

        return OperatorScoreInfo(
            operator=result.operator,
            rel_path=result.rel_path,
            pass_rate=result.pass_rate,
            avg_speedup=result.avg_speedup,
            compile_passed=compile_passed,
            passed_cases=result.passed_cases,
            total_cases=total_cases,
            compilation_score=agg["compilation_score"],
            compile_runtime_fail_cases=agg.get("n_compile_runtime_fail", 0),
            function_score=agg["function_score"],
            bonus_score=agg.get("bonus_score", 0.0),
            performance_score=agg["performance_score"],
            total_score=agg["total_score"],
            per_case_scores=agg["per_case_scores"],
            score_error_code=agg.get("score_error_code"),
            score_error=agg.get("score_error"),
            zeroed_by_no_npu_perf=bool(agg.get("zeroed_by_no_npu_perf")),
            reference_invalid_cases=agg.get("n_reference_invalid", 0),
        )

    def calculate_overall_score(
        self, score_infos: List[OperatorScoreInfo]
    ) -> Optional[float]:
        """benchmark 总分 = Σ Level 得分 = Σ EachOperatorScore (Eq. 5)."""
        if any(info.total_score is None for info in score_infos):
            return None
        return sum(info.total_score for info in score_infos)

    def calculate_average_score(
        self, score_infos: List[OperatorScoreInfo]
    ) -> Optional[float]:
        """benchmark 平均分 = 总分 / 算子数。"""
        if not score_infos:
            return 0.0
        total = self.calculate_overall_score(score_infos)
        return total / len(score_infos) if total is not None else None

    def calculate_level_score(
        self, score_infos: List[OperatorScoreInfo], level: str
    ) -> Optional[float]:
        """指定 Level 的得分。"""
        return self.calculate_overall_score(
            [
                info
                for info in score_infos
                if info.rel_path == level or info.rel_path.split("/", 1)[0] == level
            ]
        )

    def list_levels(self, score_infos: List[OperatorScoreInfo]) -> List[str]:
        """收集所有出现过的 Level 标签（按出现顺序去重）。"""
        seen: List[str] = []
        for info in score_infos:
            level = info.rel_path.split("/", 1)[0] if info.rel_path else "unknown"
            if level not in seen:
                seen.append(level)
        return seen

    def calculate_ranking(
        self, score_infos: List[OperatorScoreInfo]
    ) -> List[Dict[str, Any]]:
        """计算算子排名"""
        sorted_infos = sorted(
            (info for info in score_infos if info.total_score is not None),
            key=lambda x: x.total_score,
            reverse=True,
        )
        ranking = []
        for i, info in enumerate(sorted_infos, 1):
            ranking.append(
                {
                    "rank": i,
                    "operator": info.operator,
                    "rel_path": info.rel_path,
                    "score": info.total_score,
                    "pass_rate": info.pass_rate,
                    "avg_speedup": info.avg_speedup,
                    "compile_passed": info.compile_passed,
                }
            )
        return ranking

    def get_score_breakdown(self, result: EvalOperatorResult) -> Dict[str, Any]:
        """获取得分分解详情"""
        score_info = self.calculate_operator_score(result)
        wc, wf, wp, wb = self._effective_weights()
        return {
            "operator": score_info.operator,
            "rel_path": score_info.rel_path,
            "compile_passed": score_info.compile_passed,
            "total_cases": score_info.total_cases,
            "passed_cases": score_info.passed_cases,
            "pass_rate": score_info.pass_rate,
            "avg_speedup": score_info.avg_speedup,
            "compilation_score": {
                "formula": "w_c · (N - compile_runtime_fail_cases) / N · 100",
                "weight": wc,
                "delta_pass": 1 if score_info.compile_passed else 0,
                "compile_runtime_fail_cases": score_info.compile_runtime_fail_cases,
                "score": score_info.compilation_score,
            },
            "function_score": {
                "formula": "(Σ δ_acc,i · w_f / N) · 100",
                "weight": wf,
                "passed_cases": score_info.passed_cases,
                "total_cases": score_info.total_cases,
                "score": score_info.function_score,
            },
            "bonus_score": {
                "formula": "w_b · δ_all · 100, δ_all = ∏ δ_acc,i",
                "weight": wb,
                "score": score_info.bonus_score,
            },
            "performance_score": {
                "formula": "(Σ δ_acc,i · w_p · score_i / N) · 100",
                "weight": wp,
                "per_case_scores": score_info.per_case_scores,
                "score": score_info.performance_score,
            },
            "total_score": {
                "formula": "[ w_c · δ_pass + w_b · δ_all + Σ δ_acc,i (w_f + w_p · score_i) / N ] · 100",
                "compilation": score_info.compilation_score,
                "function": score_info.function_score,
                "bonus": score_info.bonus_score,
                "performance": score_info.performance_score,
                "score": score_info.total_score,
            },
        }


# === 评分方案实现 ===


class CannScoringScheme(ScoringScheme):
    """CANN 评分方案

    使用 SOL-Score 公式计算性能得分：
    - baseline: 从用例定义获取（baseline_perf_us）
    - t_hw: 从用例定义获取（t_hw_us）
    - 评分公式: (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))
    """

    def get_scheme_name(self) -> str:
        return "cann"

    def get_scheme_description(self) -> str:
        return "CANN SOL-Score 方案：baseline对比 + 理论硬件下界"

    def prepare_baseline(self, case_spec: Any) -> float:
        """从用例定义获取基线时间"""
        if hasattr(case_spec, "baseline_perf_us"):
            return float(case_spec.baseline_perf_us)
        if hasattr(case_spec, "metadata"):
            return float(case_spec.metadata.get("baseline_perf_us", 0.0))
        if isinstance(case_spec, dict):
            return float(case_spec.get("baseline_perf_us", 0.0))
        return 0.0

    def get_t_hw(self, case_spec: Any) -> float:
        """从用例定义获取理论硬件下界"""
        if hasattr(case_spec, "t_hw_us"):
            return float(case_spec.t_hw_us)
        if hasattr(case_spec, "metadata"):
            return float(case_spec.metadata.get("t_hw_us", 0.0))
        if isinstance(case_spec, dict):
            return float(case_spec.get("t_hw_us", 0.0))
        return 0.0

    def calculate_case_score(
        self, perf_result: PerfResult, baseline_us: float
    ) -> Optional[float]:
        """计算单个用例的 SOL-Score"""
        if perf_result.metadata.get("failure_type") == FAILURE_TYPE_REFERENCE_INVALID:
            return None
        t_hw_us = perf_result.metadata.get("t_hw_us", 0.0)
        return per_case_sol_score(baseline_us, perf_result.elapsed_us, t_hw_us)

    def aggregate_operator_scores(
        self,
        case_scores: List[CaseScoreInfo],
        compile_passed: bool = True,
        total_cases: int = None,
    ) -> Optional[float]:
        """聚合算子综合得分；参考失效时不可评分。"""
        if any(
            s.metadata.get("failure_type") == FAILURE_TYPE_REFERENCE_INVALID
            for s in case_scores
        ):
            return None
        if total_cases is None:
            total_cases = len(case_scores)

        if total_cases <= 0:
            return 0.0

        passed_cases = [s for s in case_scores if s.passed]
        n_passed = len(passed_cases)

        compilation_contrib = WEIGHT_COMPILATION if compile_passed else 0.0
        function_contrib = WEIGHT_FUNCTION * n_passed / total_cases

        perf_scores = []
        for s in passed_cases:
            if s.score is not None and s.score > 0:
                perf_scores.append(s.score)

        if perf_scores:
            avg_perf_score = sum(perf_scores) / len(perf_scores)
            perf_contrib = WEIGHT_PERFORMANCE * avg_perf_score * n_passed / total_cases
        else:
            perf_contrib = 0.0

        total_score = (compilation_contrib + function_contrib + perf_contrib) * 100
        return round(total_score, 4)


class SimpleComparisonScheme(ScoringScheme):
    """简单对比评分方案

    使用加速比作为得分指标：
    - score = baseline / elapsed
    - 无 SOL-Score（不考虑理论硬件下界）
    """

    def get_scheme_name(self) -> str:
        return "simple_comparison"

    def get_scheme_description(self) -> str:
        return "简单对比方案：加速比 = baseline / elapsed"

    def prepare_baseline(self, case_spec: Any) -> float:
        """从用例定义获取基线时间"""
        if hasattr(case_spec, "baseline_perf_us"):
            return float(case_spec.baseline_perf_us)
        if hasattr(case_spec, "metadata"):
            return float(case_spec.metadata.get("baseline_perf_us", 0.0))
        if isinstance(case_spec, dict):
            return float(case_spec.get("baseline_perf_us", 0.0))
        return 0.0

    def calculate_case_score(
        self, perf_result: PerfResult, baseline_us: float
    ) -> Optional[float]:
        """计算加速比"""
        elapsed_us = perf_result.elapsed_us
        if baseline_us <= 0 or elapsed_us <= 0:
            return None
        return baseline_us / elapsed_us

    def aggregate_operator_scores(
        self,
        case_scores: List[CaseScoreInfo],
        compile_passed: bool = True,
        total_cases: int = None,
    ) -> float:
        """聚合算子得分（使用平均加速比）"""
        passed_cases = [s for s in case_scores if s.passed]
        if not passed_cases:
            return 0.0

        speedups = [s.score for s in passed_cases if s.score is not None]
        if not speedups:
            return 0.0

        avg_speedup = sum(speedups) / len(speedups)
        normalized_score = min(avg_speedup * 10, 100)
        return round(normalized_score, 4)


class RecordingOnlyScheme(ScoringScheme):
    """仅记录方案

    不进行评分，仅记录性能数据：
    - elapsed_us 记录在结果中
    - score = None
    """

    def get_scheme_name(self) -> str:
        return "recording_only"

    def get_scheme_description(self) -> str:
        return "仅记录方案：不评分，仅记录 elapsed_us"

    def prepare_baseline(self, case_spec: Any) -> float:
        """无基线"""
        return 0.0

    def calculate_case_score(
        self, perf_result: PerfResult, baseline_us: float
    ) -> Optional[float]:
        """不计算得分"""
        return None

    def aggregate_operator_scores(
        self,
        case_scores: List[CaseScoreInfo],
        compile_passed: bool = True,
        total_cases: int = None,
    ) -> float:
        """不计算聚合得分"""
        return 0.0


# === 注册由 benches/cann.py 负责 ===
