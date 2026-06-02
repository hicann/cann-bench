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
PerfEvaluator 单元测试

测试对象：kernel_eval.eval.perf_eval.PerfEvaluator
核心功能：
1. run_profiled 临时目录清理（finally 块）
2. profiler 异常后资源不泄漏
"""

from unittest.mock import patch

from kernel_eval.config import Config
from kernel_eval.eval.perf_eval import PerfEvaluator


class TestProfileOperatorTempDirCleanup:
    """测试 run_profiled 的临时目录清理"""

    def test_temp_dir_cleaned_on_profiler_exception(self):
        """profiler 抛异常后 finally 块仍执行清理，且 error 信息不被覆盖"""
        config = Config(enable_profiler=True)
        evaluator = PerfEvaluator(config, archive_prof=False, freq_boost=False)

        def dummy_func():
            pass

        with patch.object(evaluator, '_profile', side_effect=RuntimeError("crash")), \
             patch.object(evaluator, '_parse_case_id', return_value=('L1/Add', '0001')), \
             patch('shutil.rmtree') as mock_rmtree:
            outputs, result = evaluator.run_profiled(
                "L1/Add/0001", dummy_func, warmup=1, repeat=2,
            )
            # profiler 异常信息应被保留，不被 CSV 解析覆盖
            assert "crash" in (result.error_msg or "")
            # finally 块应执行了清理
            assert mock_rmtree.call_count >= 1

    def test_temp_dir_cleaned_even_when_csv_walk_throws(self):
        """CSV 遍历抛异常后 finally 块仍执行清理"""
        config = Config(enable_profiler=True)
        evaluator = PerfEvaluator(config, archive_prof=False, freq_boost=False)

        def dummy_func():
            pass

        with patch.object(evaluator, '_profile', return_value=None), \
             patch.object(evaluator, '_parse_case_id', return_value=('L1/Add', '0001')), \
             patch('os.walk', side_effect=OSError("walk failed")), \
             patch('shutil.rmtree') as mock_rmtree:
            try:
                evaluator.run_profiled("L1/Add/0001", dummy_func, warmup=1, repeat=2)
            except OSError:
                pass
            assert mock_rmtree.call_count >= 1

    def test_archive_mode_does_not_cleanup(self):
        """archive_prof=True 时 finally 块不应清理目录"""
        config = Config(enable_profiler=True)
        evaluator = PerfEvaluator(config, archive_prof=True, freq_boost=False)

        def dummy_func():
            pass

        with patch.object(evaluator, '_profile', return_value=None), \
             patch.object(evaluator, '_parse_case_id', return_value=('L1/Add', '0001')), \
             patch('os.makedirs', return_value=None), \
             patch('os.listdir', return_value=[]), \
             patch('shutil.rmtree') as mock_rmtree:
            evaluator.run_profiled("L1/Add/0001", dummy_func, warmup=1, repeat=2)
            mock_rmtree.assert_not_called()


class TestMeasureSimple:
    """测试 _measure_simple 方法（CPU 计时路径）"""

    def test_measure_simple_basic(self):
        """enable_profiler=False 时走简单计时路径"""
        config = Config(enable_profiler=False)
        evaluator = PerfEvaluator(config)

        def add(a, b):
            return a + b

        outputs, result = evaluator.run_profiled(
            "L1/Add/0001", add, 1.0, 2.0,
            warmup=2, repeat=3,
        )

        assert result.elapsed_us >= 0
        assert result.error_msg is None
