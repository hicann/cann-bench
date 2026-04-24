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
评分计算器

职责：
1. 计算功能得分（Pass@1 × Wf）
2. 计算性能得分（SpeedUp × Wp）
3. 计算综合评分
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from ..eval.evaluator import EvalOperatorResult


# 权重配置
WEIGHT_FUNCTION = 3  # 功能得分权重 Wf
WEIGHT_PERFORMANCE = 5  # 性能得分权重 Wp


@dataclass
class ScoreInfo:
    """得分信息"""
    operator: str
    level: int
    pass_rate: float
    avg_speedup: float
    function_score: float
    performance_score: float
    total_score: float
    passed_cases: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operator': self.operator,
            'level': self.level,
            'pass_rate': self.pass_rate,
            'avg_speedup': self.avg_speedup,
            'function_score': self.function_score,
            'performance_score': self.performance_score,
            'total_score': self.total_score,
            'passed_cases': self.passed_cases,
        }


class ScoringCalculator:
    """评分计算器"""

    def __init__(self, wf: float = WEIGHT_FUNCTION, wp: float = WEIGHT_PERFORMANCE):
        """
        Args:
            wf: 功能得分权重
            wp: 性能得分权重
        """
        self.wf = wf
        self.wp = wp

    def calculate_operator_score(self, result: EvalOperatorResult) -> ScoreInfo:
        """计算单个算子的得分

        评分公式：
        - 功能得分 = Pass@1 × Wf
        - 性能得分 = SpeedUp × Wp
        - 总得分 = 功能通过用例数 × (功能得分 + 性能得分)
        """
        pass_rate = result.pass_rate
        avg_speedup = result.avg_speedup
        passed_cases = result.passed_cases

        # 功能得分：通过率 × 功能权重
        function_score = pass_rate * self.wf

        # 性能得分：加速比 × 性能权重
        # 加速比 > 1 表示优于基准，给予正向得分
        # 加速比 <= 1 表示不优于基准，得分为0或负
        performance_score = max(0, avg_speedup) * self.wp

        # 综合评分：通过用例数 × (功能得分 + 性能得分)
        # 只有通过的用例才能贡献得分
        total_score = passed_cases * (function_score + performance_score)

        return ScoreInfo(
            operator=result.operator,
            level=result.level,
            pass_rate=pass_rate,
            avg_speedup=avg_speedup,
            function_score=function_score,
            performance_score=performance_score,
            total_score=total_score,
            passed_cases=passed_cases,
        )

    def calculate_overall_score(self, score_infos: List[ScoreInfo]) -> float:
        """计算综合得分

        综合得分 = 所有算子总得分的平均值
        """
        if not score_infos:
            return 0.0

        total_score = sum(info.total_score for info in score_infos)
        return total_score / len(score_infos)

    def calculate_ranking(self, score_infos: List[ScoreInfo]) -> List[Dict[str, Any]]:
        """计算算子排名"""
        sorted_infos = sorted(score_infos, key=lambda x: x.total_score, reverse=True)

        ranking = []
        for i, info in enumerate(sorted_infos, 1):
            ranking.append({
                'rank': i,
                'operator': info.operator,
                'level': info.level,
                'score': info.total_score,
                'pass_rate': info.pass_rate,
                'avg_speedup': info.avg_speedup,
            })

        return ranking

    def get_score_breakdown(self, result: EvalOperatorResult) -> Dict[str, Any]:
        """获取得分分解详情"""
        score_info = self.calculate_operator_score(result)

        return {
            'operator': score_info.operator,
            'level': score_info.level,
            'total_cases': result.total_cases,
            'passed_cases': score_info.passed_cases,
            'pass_rate': score_info.pass_rate,
            'avg_speedup': score_info.avg_speedup,
            'function_score': {
                'formula': 'Pass@1 × Wf',
                'pass_rate': score_info.pass_rate,
                'weight': self.wf,
                'score': score_info.function_score,
            },
            'performance_score': {
                'formula': 'SpeedUp × Wp',
                'speedup': score_info.avg_speedup,
                'weight': self.wp,
                'score': score_info.performance_score,
            },
            'total_score': {
                'formula': '通过用例数 × (功能得分 + 性能得分)',
                'passed_cases': score_info.passed_cases,
                'function_score': score_info.function_score,
                'performance_score': score_info.performance_score,
                'score': score_info.total_score,
            },
        }