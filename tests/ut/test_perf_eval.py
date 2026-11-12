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
3. perf_metric_strategy 从 Config 自取
4. profiler 采集失败后的自动重试机制
"""

import csv
import json
from unittest.mock import patch

from kernel_eval.config import Config
from kernel_eval.eval.perf_eval import PerfEvaluator
from kernel_eval.base.perf_strategy import ProfFileLocations


class TestProfileOperatorTempDirCleanup:
    """测试 run_profiled 的临时目录清理"""

    def test_temp_dir_cleaned_on_profiler_exception(self):
        """profiler 抛异常后 finally 块仍执行清理，异常信息保留在 metadata"""
        # PerfEvaluator 从 Config.perf_metric_strategy_override 获取策略
        config = Config(perf_restore_inputs=False, enable_profiler=True, perf_metric_strategy_override="kernel_details")
        evaluator = PerfEvaluator(config, archive_prof=False, freq_boost=False)

        def dummy_func():
            pass

        with patch.object(evaluator, '_profile', side_effect=RuntimeError("crash")), \
             patch.object(evaluator, '_parse_case_id', return_value=('L1/Add', '0001')), \
             patch('shutil.rmtree') as mock_rmtree:
            outputs, result = evaluator.run_profiled(
                "L1/Add/0001", dummy_func, warmup=1, repeat=2,
            )
            # profiler 异常信息保留在 metadata 中，error_msg 由策略报告
            assert result.metadata["profile_exception"] == "crash"
            # finally 块应执行了清理
            assert mock_rmtree.call_count >= 1

    def test_temp_dir_cleaned_even_when_csv_walk_throws(self):
        """CSV 遍历抛异常后 finally 块仍执行清理"""
        config = Config(perf_restore_inputs=False, enable_profiler=True, perf_metric_strategy_override="kernel_details")
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
        config = Config(perf_restore_inputs=False, enable_profiler=True, perf_metric_strategy_override="kernel_details")
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
        config = Config(perf_restore_inputs=False, enable_profiler=False)
        evaluator = PerfEvaluator(config)

        def add(a, b):
            return a + b

        outputs, result = evaluator.run_profiled(
            "L1/Add/0001", add, 1.0, 2.0,
            warmup=2, repeat=3,
        )

        assert result.elapsed_us >= 0
        assert result.error_msg is None


def test_run_profiled_uses_trace_view_strategy():
    """Config.perf_metric_strategy_override="trace_view" 时使用 trace_view 口径"""
    # PerfEvaluator 从 Config.perf_metric_strategy_override 自取策略
    config = Config(perf_restore_inputs=False, enable_profiler=True, perf_metric_strategy_override="trace_view")
    evaluator = PerfEvaluator(config, archive_prof=False, freq_boost=False)
    trace_strategy = evaluator.perf_metric_strategy  # 从 registry 取到的 TraceViewStrategy 实例

    def dummy_func():
        return "ok"

    def run_stub(fn, prof_dir, warmup, repeat):
        fn()

    # Mock TraceViewStrategy.parse to simulate successful trace_view parsing
    with patch.object(evaluator, "_profile", side_effect=run_stub), \
         patch.object(evaluator, "_parse_case_id", return_value=("L1/Add", "0001")), \
         patch.object(trace_strategy, "parse") as mock_parse:
        # Simulate TraceViewStrategy.parse filling the result
        def fake_parse(prof_files, result):
            result.elapsed_us = 12.35
            result.op_times = {"trace_view": {
                "aicore_e2e": 12.35,
                "aicpukernel_gap": 1.23,
                "aicore_e2e_jitter": 0.04,
            }}
            result.metadata["perf_source"] = "trace_view"
            result.metadata["elapsed_us_source"] = "trace_view.aicore_e2e"
            return result
        mock_parse.side_effect = fake_parse

        outputs, result = evaluator.run_profiled(
            "L1/Add/0001", dummy_func, warmup=1, repeat=2,
        )

    assert outputs == "ok"
    assert result.elapsed_us == 12.35
    assert result.op_times == {
        "trace_view": {
            "aicore_e2e": 12.35,
            "aicpukernel_gap": 1.23,
            "aicore_e2e_jitter": 0.04,
        }
    }
    assert result.metadata["perf_source"] == "trace_view"
    assert result.metadata["elapsed_us_source"] == "trace_view.aicore_e2e"
    mock_parse.assert_called_once()


class TestProfilerStepSynchronization:
    """测试 profiler step 边界同步"""

    def test_profile_step_synchronizes_before_step(self):
        """候选算子返回后应先等待 NPU stream，再推进 profiler step"""
        calls = []

        class FakeDeviceManager:
            def synchronize(self):
                calls.append("sync")

        class FakeProfiler:
            def step(self):
                calls.append("step")

        config = Config(perf_restore_inputs=False, enable_profiler=True, perf_metric_strategy_override="kernel_details")
        evaluator = PerfEvaluator(config, device_manager=FakeDeviceManager(), archive_prof=False, freq_boost=False)

        def fn():
            calls.append("fn")

        exc = evaluator._run_profile_step(fn, FakeProfiler())

        assert exc is None
        assert calls == ["fn", "sync", "step"]

    def test_profile_step_advances_step_when_sync_fails(self):
        """同步暴露异步执行错误时仍推进 step，让 profiler 上下文干净退出"""
        calls = []

        class FakeDeviceManager:
            def synchronize(self):
                calls.append("sync")
                raise RuntimeError("device failed")

        class FakeProfiler:
            def step(self):
                calls.append("step")

        config = Config(perf_restore_inputs=False, enable_profiler=True, perf_metric_strategy_override="kernel_details")
        evaluator = PerfEvaluator(config, device_manager=FakeDeviceManager(), archive_prof=False, freq_boost=False)

        def fn():
            calls.append("fn")

        exc = evaluator._run_profile_step(fn, FakeProfiler())

        assert isinstance(exc, RuntimeError)
        assert str(exc) == "device failed"
        assert calls == ["fn", "sync", "step"]


class TestProfiledBatch:
    @staticmethod
    def _write_csv(path):
        fields = [
            "Start Time(us)", "Name", "Type", "Duration(us)",
            "Input Shapes",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            clock = 0
            for case_idx, durations in enumerate(((10, 14), (20, 40))):
                for duration in durations:
                    writer.writerow({
                        "Start Time(us)": str(clock),
                        "Name": "cache",
                        "Type": "CannBenchCacheClean",
                        "Duration(us)": "90",
                        "Input Shapes": "",
                    })
                    clock += 1
                    writer.writerow({
                        "Start Time(us)": str(clock),
                        "Name": f"case{case_idx}_custom",
                        "Type": "custom",
                        "Duration(us)": str(duration),
                        "Input Shapes": "",
                    })
                    clock += 1

    @staticmethod
    def _host_ranges(n_cases, warmup, repeat, width=20):
        trace = []
        steps_per_case = warmup + repeat
        for case_index in range(n_cases):
            for step_index in range(steps_per_case):
                ordinal = case_index * steps_per_case + step_index
                trace.append({
                    "name": (
                        f"CannBenchBatchStep/case={case_index}/"
                        f"step={step_index}"
                    ),
                    "ph": "X",
                    "pid": 7,
                    "tid": 11,
                    "ts": str(ordinal * width),
                    "dur": str(width - 1),
                })
        return trace

    def test_batch_returns_per_case_medians(self, tmp_path):
        csv_path = tmp_path / "kernel_details.csv"
        self._write_csv(csv_path)
        config = Config(perf_restore_inputs=False,
            enable_profiler=True,
            perf_metric_strategy_override="kernel_details",
        )
        evaluator = PerfEvaluator(
            config, archive_prof=False, freq_boost=False,
        )
        prof_files = ProfFileLocations(csv_path=str(csv_path))

        with patch.object(evaluator, "_profile_batch"), \
             patch.object(
                 evaluator, "_locate_prof_files", return_value=prof_files,
             ):
            results = evaluator.run_profiled_batch(
                [("level1/exp_1", lambda: None),
                 ("level1/exp_2", lambda: None)],
                warmup=0,
                repeat=2,
            )

        assert results["level1/exp_1"].elapsed_us == 12.0
        assert results["level1/exp_2"].elapsed_us == 30.0
        assert results["level1/exp_1"].metadata["perf_batch_used"] is True

    def test_batch_prepares_deferred_calls_before_profiler(self):
        config = Config(perf_restore_inputs=False,
            enable_profiler=True,
            perf_metric_strategy_override="kernel_details",
        )
        evaluator = PerfEvaluator(
            config, archive_prof=False, freq_boost=False,
        )
        events = []

        class DeferredCall:
            def __init__(self, name):
                self.name = name
                self.prepared = False

            def prepare(self):
                events.append(f"prepare:{self.name}")
                self.prepared = True

            def __call__(self):
                assert self.prepared
                events.append(f"call:{self.name}")

            def release(self):
                events.append(f"release:{self.name}")

        first = DeferredCall("first")
        second = DeferredCall("second")

        def profile_stub(dispatcher, prof_dir, warmup, repeat,
                         include_host=False):
            assert include_host is True
            events.append("profile:start")
            dispatcher()  # Profiler's unconditional pre-flight call.
            for _ in range(repeat):
                dispatcher()

        with patch.object(evaluator, "_profile", side_effect=profile_stub):
            evaluator._profile_batch(
                [first, second], "unused", warmup=0, repeat=2,
            )

        assert events[:3] == [
            "prepare:first", "prepare:second", "profile:start",
        ]
        assert events.count("call:first") == 2
        assert events.count("call:second") == 2

    def test_aggregate_memcpy_forces_safe_fallback_signal(self, tmp_path):
        csv_path = tmp_path / "kernel_details.csv"
        api_path = tmp_path / "api_statistic.csv"
        self._write_csv(csv_path)
        with open(api_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["API Name", "Count", "Avg(us)", "Time(us)"],
            )
            writer.writeheader()
            writer.writerow({
                "API Name": "aclrtMemcpyAsync",
                "Count": "6",
                "Avg(us)": "1",
                "Time(us)": "6",
            })
        config = Config(perf_restore_inputs=False,
            enable_profiler=True,
            perf_metric_strategy_override="kernel_details",
        )
        evaluator = PerfEvaluator(
            config, archive_prof=False, freq_boost=False,
        )
        prof_files = ProfFileLocations(
            csv_path=str(csv_path),
            api_statistic_path=str(api_path),
        )

        with patch.object(evaluator, "_profile_batch"), \
             patch.object(
                 evaluator, "_locate_prof_files", return_value=prof_files,
             ):
            results = evaluator.run_profiled_batch(
                [("level1/exp_1", lambda: None),
                 ("level1/exp_2", lambda: None)],
                warmup=0,
                repeat=2,
            )

        assert all(result.elapsed_us == 0 for result in results.values())
        assert all(
            "per-case fallback" in result.error_msg
            for result in results.values()
        )

    def test_memcpy_is_validated_per_case_from_trace(self, tmp_path):
        api_path = tmp_path / "api_statistic.csv"
        trace_path = tmp_path / "trace_view.json"
        with open(api_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["API Name", "Count"])
            writer.writeheader()
            writer.writerow({"API Name": "aclrtMemcpy", "Count": "10"})

        trace = self._host_ranges(n_cases=2, warmup=0, repeat=2)
        # Two cases, two steps per case, five memcpy calls in each case.
        for timestamp in (1, 2, 3, 21, 22, 41, 42, 43, 61, 62):
            trace.append({
                "name": "AscendCL@aclrtMemcpy",
                "pid": 7,
                "tid": 11,
                "ts": str(timestamp),
            })
        trace_path.write_text(json.dumps(trace), encoding="utf-8")

        violation, reason, fallback_indices = (
            PerfEvaluator._batch_has_excessive_memcpy(
                str(api_path), str(trace_path),
                n_cases=2, warmup=0, repeat=2,
            )
        )

        assert violation is False
        assert reason == ""
        assert fallback_indices == []

    def test_memcpy_variants_keep_single_case_threshold_semantics(
        self, tmp_path,
    ):
        api_path = tmp_path / "api_statistic.csv"
        trace_path = tmp_path / "trace_view.json"
        with open(api_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["API Name", "Count"])
            writer.writeheader()
            writer.writerow({"API Name": "aclrtMemcpy", "Count": "8"})
            writer.writerow({"API Name": "aclrtMemcpyAsync", "Count": "8"})

        trace = self._host_ranges(n_cases=2, warmup=0, repeat=2)
        for case_offset in (0, 40):
            for api_name, offsets in (
                ("aclrtMemcpy", (1, 2, 21, 22)),
                ("aclrtMemcpyAsync", (3, 4, 23, 24)),
            ):
                trace.extend({
                    "name": f"AscendCL@{api_name}",
                    "pid": 7,
                    "tid": 11,
                    "ts": str(case_offset + offset),
                } for offset in offsets)
        trace_path.write_text(json.dumps(trace), encoding="utf-8")

        violation, reason, fallback_indices = (
            PerfEvaluator._batch_has_excessive_memcpy(
                str(api_path), str(trace_path),
                n_cases=2, warmup=0, repeat=2,
            )
        )

        assert violation is False
        assert reason == ""
        assert fallback_indices == []

    def test_memcpy_warmup_steps_are_excluded_from_limit(self, tmp_path):
        api_path = tmp_path / "api_statistic.csv"
        trace_path = tmp_path / "trace_view.json"
        with open(api_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["API Name", "Count"])
            writer.writeheader()
            writer.writerow({"API Name": "aclrtMemcpy", "Count": "16"})

        trace = self._host_ranges(
            n_cases=2, warmup=3, repeat=5, width=20,
        )
        for step in range(16):
            timestamp = step * 20
            trace.append({
                "name": "AscendCL@aclrtMemcpy",
                "pid": 7,
                "tid": 11,
                "ts": str(timestamp + 1),
            })
        trace_path.write_text(json.dumps(trace), encoding="utf-8")

        violation, reason, fallback_indices = (
            PerfEvaluator._batch_has_excessive_memcpy(
                str(api_path), str(trace_path),
                n_cases=2, warmup=3, repeat=5,
            )
        )

        assert violation is False
        assert reason == ""
        assert fallback_indices == []

    def test_memcpy_trace_rejects_one_over_limit_case(self, tmp_path):
        api_path = tmp_path / "api_statistic.csv"
        trace_path = tmp_path / "trace_view.json"
        with open(api_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["API Name", "Count"])
            writer.writeheader()
            writer.writerow({"API Name": "aclrtMemcpyAsync", "Count": "7"})

        trace = self._host_ranges(n_cases=2, warmup=0, repeat=2)
        trace.extend(
            {
                "name": "AscendCL@aclrtMemcpyAsync",
                "pid": 7,
                "tid": 11,
                "ts": str(timestamp),
            }
            for timestamp in (1, 2, 3, 4, 5, 6, 41)
        )
        trace_path.write_text(json.dumps(trace), encoding="utf-8")

        violation, reason, fallback_indices = (
            PerfEvaluator._batch_has_excessive_memcpy(
                str(api_path), str(trace_path),
                n_cases=2, warmup=0, repeat=2,
            )
        )

        assert violation is True
        assert "case[0].aclrtMemcpyAsync=6" in reason
        assert fallback_indices == [0]

    def test_memcpy_outside_host_ranges_fails_closed(self, tmp_path):
        api_path = tmp_path / "api_statistic.csv"
        trace_path = tmp_path / "trace_view.json"
        with open(api_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["API Name", "Count"])
            writer.writeheader()
            writer.writerow({"API Name": "aclrtMemcpy", "Count": "6"})

        trace = self._host_ranges(n_cases=2, warmup=0, repeat=2)
        trace.extend({
            "name": "AscendCL@aclrtMemcpy",
            "pid": 7,
            "tid": 11,
            "ts": str(timestamp),
        } for timestamp in (1, 2, 21, 22, 41, 1000))
        trace_path.write_text(json.dumps(trace), encoding="utf-8")

        violation, reason, fallback_indices = (
            PerfEvaluator._batch_has_excessive_memcpy(
                str(api_path), str(trace_path),
                n_cases=2, warmup=0, repeat=2,
            )
        )

        assert violation is True
        assert "exactly one batch host range" in reason
        assert fallback_indices is None

    def test_missing_host_range_fails_closed(self, tmp_path):
        api_path = tmp_path / "api_statistic.csv"
        trace_path = tmp_path / "trace_view.json"
        with open(api_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["API Name", "Count"])
            writer.writeheader()
            writer.writerow({"API Name": "aclrtMemcpy", "Count": "6"})

        trace = self._host_ranges(n_cases=2, warmup=0, repeat=2)[:-1]
        trace.extend({
            "name": "AscendCL@aclrtMemcpy",
            "pid": 7,
            "tid": 11,
            "ts": str(timestamp),
        } for timestamp in (1, 2, 21, 22, 41, 42))
        trace_path.write_text(json.dumps(trace), encoding="utf-8")

        violation, reason, fallback_indices = (
            PerfEvaluator._batch_has_excessive_memcpy(
                str(api_path), str(trace_path),
                n_cases=2, warmup=0, repeat=2,
            )
        )

        assert violation is True
        assert "host range count 3 != expected 4" in reason
        assert fallback_indices is None

    def test_batch_falls_back_only_memcpy_offender(self, tmp_path):
        csv_path = tmp_path / "kernel_details.csv"
        api_path = tmp_path / "api_statistic.csv"
        trace_path = tmp_path / "trace_view.json"
        self._write_csv(csv_path)
        with open(api_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["API Name", "Count"])
            writer.writeheader()
            writer.writerow({"API Name": "aclrtMemcpy", "Count": "7"})
        trace = self._host_ranges(n_cases=2, warmup=0, repeat=2)
        trace.extend(
            {
                "name": "AscendCL@aclrtMemcpy",
                "pid": 7,
                "tid": 11,
                "ts": str(timestamp),
            }
            for timestamp in (1, 2, 3, 4, 5, 6, 41)
        )
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        config = Config(perf_restore_inputs=False,
            enable_profiler=True,
            perf_metric_strategy_override="kernel_details",
        )
        evaluator = PerfEvaluator(
            config, archive_prof=False, freq_boost=False,
        )
        prof_files = ProfFileLocations(
            csv_path=str(csv_path),
            api_statistic_path=str(api_path),
            trace_view_path=str(trace_path),
        )

        with patch.object(evaluator, "_profile_batch"), \
             patch.object(
                 evaluator, "_locate_prof_files", return_value=prof_files,
             ):
            results = evaluator.run_profiled_batch(
                [("level1/exp_1", lambda: None),
                 ("level1/exp_2", lambda: None)],
                warmup=0,
                repeat=2,
            )

        offender = results["level1/exp_1"]
        retained = results["level1/exp_2"]
        assert offender.elapsed_us == 0
        assert "case[0].aclrtMemcpy=6" in offender.error_msg
        assert offender.metadata["perf_batch_failed"] is True
        assert retained.elapsed_us == 30.0
        assert retained.metadata["perf_batch_used"] is True


class TestProfilerRetry:
    """Test profiler collection retry mechanism.

    When kernel_details.csv is missing or elapsed_us <= 0 but the op executed
    successfully (Host side dispatched the kernel — last_outputs is not None),
    the profiler collection should be automatically retried instead of
    directly scoring 0.
    """

    def _make_evaluator(self, max_retries=2, retry_delay=0.0):
        config = Config(perf_restore_inputs=False,
            enable_profiler=True,
            perf_metric_strategy_override="kernel_details",
            profiler_max_retries=max_retries,
            profiler_retry_delay=retry_delay,
        )
        return PerfEvaluator(config, archive_prof=False, freq_boost=False)

    @staticmethod
    def _run_stub(fn, prof_dir, warmup, repeat):
        fn()

    def test_retry_succeeds_on_second_attempt(self):
        """First attempt elapsed_us=0, retry produces valid data."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        parse_calls = [0]

        def fake_parse(prof_files, result):
            parse_calls[0] += 1
            if parse_calls[0] == 1:
                result.elapsed_us = 0.0
                result.error_msg = "no csv"
            else:
                result.elapsed_us = 42.0
                result.error_msg = None
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub), \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 42.0
        assert result.metadata.get("profiler_retries_used") == 1
        assert parse_calls[0] == 2

    def test_retry_when_profile_raises_but_op_ran(self):
        """_profile raises but last_outputs set (op ran in pre-flight) → retry."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        profile_calls = [0]

        def profile_stub(fn, prof_dir, warmup, repeat):
            fn()
            profile_calls[0] += 1
            if profile_calls[0] == 1:
                raise RuntimeError("profiler crashed")

        def fake_parse(prof_files, result):
            result.elapsed_us = 77.0
            return result

        with patch.object(evaluator, '_profile', side_effect=profile_stub), \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 77.0
        assert result.metadata.get("profiler_retries_used") == 1
        assert profile_calls[0] == 2

    def test_no_retry_when_op_fails(self):
        """When _profile raises and last_outputs is None, no retry."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)

        with patch.object(evaluator, '_profile',
                          side_effect=RuntimeError("crash")) as mock_profile, \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch('time.sleep'), \
             patch('shutil.rmtree'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 0.0
        assert result.metadata.get("profiler_retries_used") is None
        assert result.metadata.get("profiler_retries_exhausted") is None
        assert mock_profile.call_count == 1

    def test_no_retry_when_first_attempt_succeeds(self):
        """When first attempt produces valid elapsed_us, no retry."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        def fake_parse(prof_files, result):
            result.elapsed_us = 99.0
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub) as mock_profile, \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 99.0
        assert result.metadata.get("profiler_retries_used") is None
        assert mock_profile.call_count == 1

    def test_retries_exhausted(self):
        """All attempts fail → profiler_retries_exhausted=True."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        def fake_parse(prof_files, result):
            result.elapsed_us = 0.0
            result.error_msg = "no csv"
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub) as mock_profile, \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'), \
             patch('shutil.rmtree'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 0.0
        assert result.metadata.get("profiler_retries_exhausted") is True
        assert len(result.metadata.get("profiler_retry_reasons", [])) == 2
        assert mock_profile.call_count == 3

    def test_no_retry_when_disabled(self):
        """profiler_max_retries=0 means no retry."""
        evaluator = self._make_evaluator(max_retries=0, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        def fake_parse(prof_files, result):
            result.elapsed_us = 0.0
            result.error_msg = "no csv"
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub) as mock_profile, \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch('time.sleep'), \
             patch('shutil.rmtree'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 0.0
        assert result.metadata.get("profiler_retries_exhausted") is None
        assert mock_profile.call_count == 1

    def test_prof_dir_cleaned_between_retries(self):
        """_clean_prof_dir_contents called before each retry attempt."""
        evaluator = self._make_evaluator(max_retries=2, retry_delay=0.0)
        strategy = evaluator.perf_metric_strategy

        parse_calls = [0]

        def fake_parse(prof_files, result):
            parse_calls[0] += 1
            if parse_calls[0] <= 2:
                result.elapsed_us = 0.0
                result.error_msg = "no csv"
            else:
                result.elapsed_us = 50.0
                result.error_msg = None
            return result

        with patch.object(evaluator, '_profile', side_effect=self._run_stub), \
             patch.object(evaluator, '_parse_case_id',
                          return_value=('L1/Add', '0001')), \
             patch.object(evaluator, '_locate_prof_files',
                          return_value=ProfFileLocations()), \
             patch.object(strategy, 'parse', side_effect=fake_parse), \
             patch.object(evaluator, '_clean_prof_dir_contents') as mock_clean, \
             patch('time.sleep'):
            outputs, result = evaluator.run_profiled(
                "L1/Add_0001", lambda: "ok", warmup=1, repeat=2,
            )

        assert result.elapsed_us == 50.0
        assert mock_clean.call_count == 2
