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
评分计算器 (docs/spec/benchmark_spec.md §3.3 / Eq. 3, 4, 5)

- 单用例性能得分: score_i = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))
- 单算子综合评分: EachOperatorScore =
      [ w_c · δ_pass + Σ_i δ_acc,i · (w_f + w_p · score_i) / len(cases) ] · 100
  权重: w_c=0.2, w_f=0.3, w_p=0.5  (sum=1, 单算子满分=100；T_cand<T_HW 时允许 >100)
- Level 得分: 给定 level 标签下所有算子分数之和；总分 = Σ Level 得分
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from ..eval.evaluator import EvalOperatorResult


_logger = logging.getLogger(__name__)


# 权重配置（bench.tex §3.5）
WEIGHT_COMPILATION = 0.2  # w_c
WEIGHT_FUNCTION = 0.3     # w_f
WEIGHT_PERFORMANCE = 0.5  # w_p

NO_NPU_PERF_ERROR_CODE = "no_npu_kernel_detected"
NO_NPU_PERF_ERROR = "未检测到 NPU 算子执行，疑似 CPU fallback，反作弊触发。"


@dataclass
class OperatorScoreInfo:
    """算子级得分信息（per operator）

    包含编译/功能/性能三轴得分，以及 per-case 调试用分数列表。
    """

    operator: str = ""
    rel_path: str = ""
    pass_rate: float = 0.0
    avg_speedup: float = 0.0  # 诊断保留
    compile_passed: bool = False
    passed_cases: int = 0
    total_cases: int = 0
    # bench.tex 三轴得分（已按 w_c/w_f/w_p 加权后的贡献，并已归一化到 0-100 量纲）
    compilation_score: float = 0.0
    function_score: float = 0.0
    performance_score: float = 0.0
    total_score: float = 0.0  # 单算子综合得分，常规区间 [0, 100]；T_cand<T_HW 时可 >100（不截断）
    # 调试用：每个用例的 hardware-anchored 分数，None 表示数据不全或未通过功能门
    per_case_scores: List[Optional[float]] = field(default_factory=list)
    score_error_code: Optional[str] = None
    score_error: Optional[str] = None
    zeroed_by_no_npu_perf: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operator': self.operator,
            'rel_path': self.rel_path,
            'pass_rate': self.pass_rate,
            'avg_speedup': self.avg_speedup,
            'compile_passed': self.compile_passed,
            'passed_cases': self.passed_cases,
            'total_cases': self.total_cases,
            'compilation_score': self.compilation_score,
            'function_score': self.function_score,
            'performance_score': self.performance_score,
            'total_score': self.total_score,
            'per_case_scores': self.per_case_scores,
            'score_error_code': self.score_error_code,
            'score_error': self.score_error,
            'zeroed_by_no_npu_perf': self.zeroed_by_no_npu_perf,
        }


def per_case_sol_score(
    t_baseline: float,
    t_cand: float,
    t_hw: float,
    rel_path: Optional[str] = None,
) -> Optional[float]:
    """bench.tex Eq. 3。T_cand 或 T_HW 异常 / 分母 ≤ 0 时返回 None。

    设计取舍：
    - 不对返回值做上界截断——当 T_cand < T_HW 时 score > 1.0 的"超额分"
      是有意保留的，用以激励算法突破当前硬件下界。
    - T_baseline < T_HW 视为基线/T_HW 标定可疑（应在用例侧人工核查），
      但仍计算分数继续输出，仅 warn 一次（per rel_path × 数值组合）。
    - **T_baseline 缺失（≤ 0）但 T_HW 已知时**，按 fallback 规则取
      ``max(T_HW * 10, 10)`` 作为代理基线继续打分（F054：fallback 路径也 warn）；
      fallback 不回写到 cases，仅在运行时使用（约定：让缺基线的 case 也能拿到一个
      合理的相对分数，不会因为基线漏填整体被静默置 0）。

    Args:
        rel_path: 算子相对路径，用于日志去重 key（F058）+ 错误溯源。可选。

    F057: 返回 None 的三种成因分别 warn，避免被 aggregate_eq4 统一吞为"缺锚点"。
    F646: NaN inputs traverse `<= 0` guards transparently because every IEEE 754
    comparison against NaN returns False. Explicitly reject NaN up front so it
    cannot propagate into the aggregated total_score.
    """
    # F646: reject NaN inputs explicitly — IEEE 754 makes every comparison
    # against NaN false (`NaN <= 0` is False), so the subsequent guards alone
    # would let NaN pass through and contaminate `perf_score_sum`.
    if math.isnan(t_baseline) or math.isnan(t_cand) or math.isnan(t_hw):
        _warn_invalid_anchor(rel_path, t_cand, t_hw)
        return None
    # F057 成因 (a): T_cand 或 T_HW 异常
    if t_cand <= 0 or t_hw <= 0:
        _warn_invalid_anchor(rel_path, t_cand, t_hw)
        return None
    if t_baseline <= 0:
        # F057 成因 (c) + F054: baseline 缺失走 fallback，warn 一次让用户感知
        _warn_fallback_baseline(rel_path, t_hw)
        t_baseline = _fallback_baseline_from_hw(t_hw)
    elif t_baseline < t_hw:
        _warn_baseline_below_hw(t_baseline, t_hw, rel_path)
    denom = (t_cand - t_hw) + (t_baseline - t_hw)
    if denom <= 0:
        # F057 成因 (b): denom ≤ 0（罕见，但 T_cand < T_HW 且 T_baseline ≈ T_HW 时可能）
        _warn_denom_nonpositive(rel_path, t_baseline, t_cand, t_hw)
        return None
    return (t_baseline - t_hw) / denom


def _fallback_baseline_from_hw(t_hw: float) -> float:
    """缺基线时的代理 baseline：max(T_HW * 10, 10 us)。"""
    return max(t_hw * 10.0, 10.0)


# 同一 (rel_path, T_baseline, T_HW) 组合只提示一次，避免成批用例时刷屏。
# F058: key 加入 rel_path 标识，不同算子的相同数值组合不会互相抑制。
_BASELINE_HW_WARNED: set = set()


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
        op_prefix, t_baseline, t_hw,
    )


# F054 + F057 (c): fallback baseline 走代理值，独立 warn 让用户感知"分数基于代理基线"
_FALLBACK_BASELINE_WARNED: set = set()


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
        op_prefix, proxy, t_hw,
    )


# F057 (a): T_cand 或 T_HW 异常
_INVALID_ANCHOR_WARNED: set = set()


def _warn_invalid_anchor(rel_path: Optional[str], t_cand: float, t_hw: float) -> None:
    key = (rel_path, round(t_cand, 4), round(t_hw, 4))
    if key in _INVALID_ANCHOR_WARNED:
        return
    _INVALID_ANCHOR_WARNED.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%sper_case_sol_score: T_cand (%.4f us) 或 T_HW (%.4f us) 非正，"
        "本 case 性能项按 None 处理。",
        op_prefix, t_cand, t_hw,
    )


# F057 (b): denom ≤ 0
_DENOM_WARNED: set = set()


def _warn_denom_nonpositive(
    rel_path: Optional[str], t_baseline: float, t_cand: float, t_hw: float
) -> None:
    key = (rel_path, round(t_baseline, 4), round(t_cand, 4), round(t_hw, 4))
    if key in _DENOM_WARNED:
        return
    _DENOM_WARNED.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%sper_case_sol_score: denom = (T_cand-T_HW) + (T_baseline-T_HW) ≤ 0 "
        "(T_baseline=%.4f us, T_cand=%.4f us, T_HW=%.4f us)。本 case 性能项 None。"
        "可能 T_baseline 与 T_HW 几乎相等且 T_cand 远小于 T_HW。",
        op_prefix, t_baseline, t_cand, t_hw,
    )


# 一次 aggregate_eq4 调用中只打一次缺锚点告警
# F061: key 加 rel_path，不同算子的相同 (通过数, 缺失数) 组合不会互相抑制
_PERF_MISSING_WARNED_RUN: set = set()


def _warn_perf_anchor_missing(
    n_func_pass: int, n_perf_missing: int, rel_path: Optional[str] = None
) -> None:
    """通知调用者：功能通过的 case 中有 N 个缺基线/T_HW 锚点，按 §3.3 计为 0 性能分。

    避免静默"系统性低估"——按 spec 设计 missing anchor → 0 是有意的，
    但用户至少应该看到这条信息，知道分数偏低是因为基线缺失，不是 kernel 慢。
    F059: 改 _logger.warning 输出到 stderr。
    """
    key = (rel_path, n_func_pass, n_perf_missing)
    if key in _PERF_MISSING_WARNED_RUN:
        return
    _PERF_MISSING_WARNED_RUN.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%saggregate_eq4: %d / %d 个功能通过的 case 缺 baseline_perf_us / t_hw_us 锚点，"
        "按 spec §3.3 按 0 计入性能项。若分数偏低，请先核查这些 case 的基线是否已填充。",
        op_prefix, n_perf_missing, n_func_pass,
    )


_NO_NPU_PERF_WARNED_RUN: set = set()


def _warn_no_npu_perf(
    n_func_pass: int, n_no_perf_pass: int, rel_path: Optional[str] = None
) -> None:
    """功能通过但 profiler 没有采到 NPU kernel 时间，整算子置 0 分。"""
    key = (rel_path, n_func_pass, n_no_perf_pass)
    if key in _NO_NPU_PERF_WARNED_RUN:
        return
    _NO_NPU_PERF_WARNED_RUN.add(key)
    op_prefix = f"[{rel_path}] " if rel_path else ""
    _logger.warning(
        "%saggregate_eq4: %d / %d 个功能通过的 case 未检测到 NPU 算子性能数据，"
        "疑似 CPU fallback 或未执行提交的 NPU kernel，整算子按 0 分处理。",
        op_prefix, n_no_perf_pass, n_func_pass,
    )



def aggregate_eq4(
    compile_passed: bool,
    total_cases: int,
    case_scores: List[Tuple[bool, Optional[float]]],
    wc: float = WEIGHT_COMPILATION,
    wf: float = WEIGHT_FUNCTION,
    wp: float = WEIGHT_PERFORMANCE,
    rel_path: Optional[str] = None,
    n_no_perf_pass: int = 0,
) -> Dict[str, Any]:
    """Eq.4 单算子综合分聚合——单一事实来源。

    EachOperatorScore = [ w_c·δ_pass + Σ_i δ_acc,i (w_f + w_p·score_i) / N ] · 100

    **F021 设计意图（必读）**：

    权重 ``wc=0.2 / wf=0.3 / wp=0.5`` 是 **运行通过路径** 下的权重分配，**不**
    意味着编译失败仅扣 20%。根据 spec §3.x，``δ_pass=0`` 时**所有** ``δ_acc,i ≡ 0``，
    意味着：

        编译失败 → compilation_score = 0
                  → function_score    = 0   (无 case 能通过 functional check)
                  → performance_score = 0   (无 case 能通过 perf 评测)
                  → total_score       = 0

    即"编译失败 = 不可评测 = 0 分"。这是有意设计：未编译的 kernel 没有任何
    可信的数值可供性能/功能评判，部分给分会让"完全不能跑"的提交看起来比"跑通
    但全错"的提交分数还高。

    如果调用方需要"编译失败时仍给 wf+wp 部分分"的语义，请**明确**修改 spec
    并重新设计 Eq.4，不要靠改 wc 来"近似"。

    Args:
        compile_passed: δ_pass=1 时为 True；δ_pass=0 时所有 δ_acc,i ≡ 0
            （详见上"设计意图"段）。
        total_cases: 分母 N。调用方负责保证 ≥ 1。
        case_scores: 列表，每项为 ``(success, score_i_or_None)``——
            ``success`` 对应 δ_acc,i；``score_i_or_None`` 缺锚点时为 None
            （按 §3.3 极限按 0 计入性能项，不影响功能项）。
        wc, wf, wp: 权重，默认 0.2 / 0.3 / 0.5。
        n_no_perf_pass: 功能通过但没有有效 NPU 性能数据的 case 数。
            这与 score_i=None 的"缺 baseline/T_HW 锚点"不同：缺锚点只扣该
            case 性能分；没有 NPU 性能数据说明可能 CPU fallback 或没有执行
            提交的 NPU kernel，整算子按反作弊规则置 0 分。

    Returns:
        dict，键: ``compilation_score`` / ``function_score`` /
        ``performance_score`` / ``total_score`` / ``per_case_scores``。
        所有分项已乘 100。
    """
    delta_pass = 1 if compile_passed else 0
    n_func_pass = 0
    n_perf_missing = 0   # 功能通过但缺基线/T_HW 锚点导致性能分按 0 计入
    perf_score_sum = 0.0
    per_case_scores: List[Optional[float]] = []

    if compile_passed:
        for success, score_i in case_scores:
            if not success:
                per_case_scores.append(None)
                continue
            n_func_pass += 1
            # F646: belt-and-suspenders — `per_case_sol_score` now rejects NaN
            # upfront, but defend the aggregator against callers that pass a
            # hand-rolled `case_scores` list with NaN, which would otherwise
            # poison `perf_score_sum` (NaN + anything = NaN).
            if score_i is None or math.isnan(score_i):
                per_case_scores.append(None if score_i is None else score_i)
                n_perf_missing += 1
                perf_score_sum += 0.0
            else:
                per_case_scores.append(score_i)
                perf_score_sum += score_i
    else:
        per_case_scores = [None] * total_cases

    # 通过的 case 中如果有缺锚点的，打印一次警告，避免性能分被静默"系统性低估"
    if n_perf_missing > 0:
        _warn_perf_anchor_missing(n_func_pass, n_perf_missing, rel_path)

    if compile_passed and n_no_perf_pass > 0:
        _warn_no_npu_perf(n_func_pass, n_no_perf_pass, rel_path)
        return {
            "compilation_score": 0.0,
            "function_score": 0.0,
            "performance_score": 0.0,
            "total_score": 0.0,
            "per_case_scores": per_case_scores,
            "n_func_pass": n_func_pass,
            "n_no_perf_pass": n_no_perf_pass,
            "score_error_code": NO_NPU_PERF_ERROR_CODE,
            "score_error": NO_NPU_PERF_ERROR,
            "zeroed_by_no_npu_perf": True,
        }

    compilation_score = wc * delta_pass * 100.0
    function_score = (n_func_pass * wf / total_cases) * 100.0
    performance_score = (perf_score_sum * wp / total_cases) * 100.0
    total_score = compilation_score + function_score + performance_score

    return {
        "compilation_score": compilation_score,
        "function_score": function_score,
        "performance_score": performance_score,
        "total_score": total_score,
        "per_case_scores": per_case_scores,
        "n_func_pass": n_func_pass,
        "n_no_perf_pass": n_no_perf_pass,
        "score_error_code": None,
        "score_error": None,
        "zeroed_by_no_npu_perf": False,
    }


class ScoringCalculator:
    """评分计算器"""

    def __init__(
        self,
        wc: float = WEIGHT_COMPILATION,
        wf: float = WEIGHT_FUNCTION,
        wp: float = WEIGHT_PERFORMANCE,
    ):
        self.wc = wc
        self.wf = wf
        self.wp = wp

    def calculate_operator_score(self, result: EvalOperatorResult) -> OperatorScoreInfo:
        """单算子综合得分 (Eq. 4)。

        N = max(声明 total_cases, len(results), 1)；
        实际 Eq.4 聚合走 aggregate_eq4——dict 输入路径与本路径共用同一份实现。

        F062: 空壳算子（声明 0 case + 实测 0 results）即使 compile_passed
        也应得 0 分。旧版 N=1 给 compilation_score=wc*100=20 是 white-elephant
        scoring，违反"无用例 → 不可评测 → 0 分"语义。
        """
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
        for case in result.results:
            if not case.success:
                case_scores.append((case.success, None))
                continue
            if case.perf_result is None or case.perf_result.elapsed_us <= 0:
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

        agg = aggregate_eq4(
            compile_passed=compile_passed,
            total_cases=total_cases,
            case_scores=case_scores,
            wc=self.wc, wf=self.wf, wp=self.wp,
            rel_path=result.rel_path,
            n_no_perf_pass=n_no_perf_pass,
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
            function_score=agg["function_score"],
            performance_score=agg["performance_score"],
            total_score=agg["total_score"],
            per_case_scores=agg["per_case_scores"],
            score_error_code=agg.get("score_error_code"),
            score_error=agg.get("score_error"),
            zeroed_by_no_npu_perf=bool(agg.get("zeroed_by_no_npu_perf")),
        )

    def calculate_overall_score(self, score_infos: List[OperatorScoreInfo]) -> float:
        """benchmark 总分 = Σ Level 得分 = Σ EachOperatorScore (Eq. 5)."""
        return sum(info.total_score for info in score_infos)

    def calculate_average_score(self, score_infos: List[OperatorScoreInfo]) -> float:
        """benchmark 平均分 = 总分 / 算子数（常规区间每算子满分约 100，T_cand<T_HW 时可 >100，跨时间可比）。

        Eq. 5 的总分会随新增算子线性膨胀（每个算子满分 +100），新旧 benchmark
        分数不可比。平均分（normalized to per-operator 100 满分）可以在算子集
        发生变化时仍保留可比性，作为额外报告维度——不替代 Eq. 5 总分。
        """
        if not score_infos:
            return 0.0
        return sum(info.total_score for info in score_infos) / len(score_infos)

    def calculate_level_score(self, score_infos: List[OperatorScoreInfo], level: str) -> float:
        """指定 Level 的得分 = Σ EachOperatorScore over ops in that level (Eq. 5)。

        Args:
            score_infos: 全部算子的 OperatorScoreInfo 列表。
            level: Level 标签，例如 ``"level1"``、``"level3"``——按 ``rel_path``
                **首段精确匹配**。

        Returns:
            该 level 下所有算子综合得分之和；无匹配算子时返回 0.0。

        F080: 旧版 `startswith(f"{level}/")` 若引入 level10 / level11 时 level1
        会误匹配（"level1/" 也是 "level10/..." 的前缀）。改首段 `split('/', 1)[0]`
        精确比较。
        """
        return sum(
            info.total_score for info in score_infos
            if info.rel_path == level or info.rel_path.split('/', 1)[0] == level
        )

    def list_levels(self, score_infos: List[OperatorScoreInfo]) -> List[str]:
        """收集所有出现过的 Level 标签（按出现顺序去重）。"""
        seen: List[str] = []
        for info in score_infos:
            level = info.rel_path.split('/', 1)[0] if info.rel_path else "unknown"
            if level not in seen:
                seen.append(level)
        return seen

    def calculate_ranking(self, score_infos: List[OperatorScoreInfo]) -> List[Dict[str, Any]]:
        """计算算子排名"""
        sorted_infos = sorted(score_infos, key=lambda x: x.total_score, reverse=True)
        ranking = []
        for i, info in enumerate(sorted_infos, 1):
            ranking.append({
                'rank': i,
                'operator': info.operator,
                'rel_path': info.rel_path,
                'score': info.total_score,
                'pass_rate': info.pass_rate,
                'avg_speedup': info.avg_speedup,
                'compile_passed': info.compile_passed,
            })
        return ranking

    def get_score_breakdown(self, result: EvalOperatorResult) -> Dict[str, Any]:
        """获取得分分解详情"""
        score_info = self.calculate_operator_score(result)
        return {
            'operator': score_info.operator,
            'rel_path': score_info.rel_path,
            'compile_passed': score_info.compile_passed,
            'total_cases': score_info.total_cases,
            'passed_cases': score_info.passed_cases,
            'pass_rate': score_info.pass_rate,
            'avg_speedup': score_info.avg_speedup,
            'compilation_score': {
                'formula': 'w_c · δ_pass · 100',
                'weight': self.wc,
                'delta_pass': 1 if score_info.compile_passed else 0,
                'score': score_info.compilation_score,
            },
            'function_score': {
                'formula': '(Σ δ_acc,i · w_f / N) · 100',
                'weight': self.wf,
                'passed_cases': score_info.passed_cases,
                'total_cases': score_info.total_cases,
                'score': score_info.function_score,
            },
            'performance_score': {
                'formula': '(Σ δ_acc,i · w_p · score_i / N) · 100, score_i = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))',
                'weight': self.wp,
                'per_case_scores': score_info.per_case_scores,
                'score': score_info.performance_score,
            },
            'total_score': {
                'formula': '[ w_c · δ_pass + Σ δ_acc,i (w_f + w_p · score_i) / N ] · 100',
                'compilation': score_info.compilation_score,
                'function': score_info.function_score,
                'performance': score_info.performance_score,
                'score': score_info.total_score,
            },
        }
