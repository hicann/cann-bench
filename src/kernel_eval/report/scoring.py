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
评分计算器 (bench.tex §3.3 / Eq. 3, 4)

- 单用例性能得分: score_i = (T_baseline - T_HW) / ((T_cand - T_HW) + (T_baseline - T_HW))
- 单算子综合评分: EachOperatorScore =
      [ w_c · δ_pass + Σ_i δ_acc,i · (w_f + w_p · score_i) / len(cases) ] · 100
  权重: w_c=0.2, w_f=0.3, w_p=0.5  (sum=1, 满分=100)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ..eval.evaluator import EvalOperatorResult


# 权重配置（bench.tex §3.5）
WEIGHT_COMPILATION = 0.2  # w_c
WEIGHT_FUNCTION = 0.3     # w_f
WEIGHT_PERFORMANCE = 0.5  # w_p


@dataclass
class ScoreInfo:
    """得分信息（per operator）"""
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
    total_score: float = 0.0  # 单算子综合得分，[0, 100]
    # 调试用：每个用例的 SOL 分数，None 表示数据不全或未通过功能门
    per_case_scores: List[Optional[float]] = field(default_factory=list)

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
        }


def per_case_sol_score(t_baseline: float, t_cand: float, t_hw: float) -> Optional[float]:
    """bench.tex Eq. 3。任一锚点缺失或分母 ≤ 0 时返回 None。"""
    if t_baseline <= 0 or t_cand <= 0 or t_hw <= 0:
        return None
    denom = (t_cand - t_hw) + (t_baseline - t_hw)
    if denom <= 0:
        return None
    return (t_baseline - t_hw) / denom


# 历史别名，保留供 scoring 模块内部使用
_per_case_sol_score = per_case_sol_score


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

    def calculate_operator_score(self, result: EvalOperatorResult) -> ScoreInfo:
        """单算子综合得分 (bench.tex Eq. 4)。

        EachOperatorScore = [ w_c·δ_pass + Σ_i δ_acc,i (w_f + w_p·score_i) / N ] · 100
        其中 N = len(cases)；δ_pass=0 时 δ_acc,i ≡ 0。
        """
        compile_passed = result.compile_passed
        delta_pass = 1 if compile_passed else 0
        total_cases = max(result.total_cases, len(result.results), 1)

        per_case_scores: List[Optional[float]] = []
        n_func_pass = 0      # Σ δ_acc,i 用于功能分
        perf_score_sum = 0.0 # Σ δ_acc,i · score_i 用于性能分

        if compile_passed:
            for case in result.results:
                if not case.success:
                    # 功能未通过：δ_acc,i = 0，per_case_scores 记 None 便于审阅
                    per_case_scores.append(None)
                    continue
                n_func_pass += 1
                score_i = _per_case_sol_score(
                    case.baseline_perf_us,
                    case.perf_result.elapsed_us if case.perf_result else 0.0,
                    case.t_hw_us,
                ) if case.perf_result else None
                per_case_scores.append(score_i)
                # 缺锚点（baseline / t_hw / 实测）时按 0 计入性能项，
                # 与 bench.tex §3.3 "T_cand→∞ ⇒ score→0" 极限一致；功能项不受影响。
                perf_score_sum += score_i if score_i is not None else 0.0
        else:
            per_case_scores = [None] * total_cases

        compilation_score = self.wc * delta_pass * 100.0
        function_score = (n_func_pass * self.wf / total_cases) * 100.0
        performance_score = (perf_score_sum * self.wp / total_cases) * 100.0
        total_score = compilation_score + function_score + performance_score

        return ScoreInfo(
            operator=result.operator,
            rel_path=result.rel_path,
            pass_rate=result.pass_rate,
            avg_speedup=result.avg_speedup,
            compile_passed=compile_passed,
            passed_cases=result.passed_cases,
            total_cases=total_cases,
            compilation_score=compilation_score,
            function_score=function_score,
            performance_score=performance_score,
            total_score=total_score,
            per_case_scores=per_case_scores,
        )

    def calculate_overall_score(self, score_infos: List[ScoreInfo]) -> float:
        """benchmark 总分 = Σ EachOperatorScore (bench.tex Eq. 5)."""
        return sum(info.total_score for info in score_infos)

    def calculate_level_score(self, score_infos: List[ScoreInfo]) -> float:
        """Level-N 得分 = Σ EachOperatorScore (bench.tex Eq. 5)."""
        return sum(info.total_score for info in score_infos)

    def calculate_ranking(self, score_infos: List[ScoreInfo]) -> List[Dict[str, Any]]:
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
