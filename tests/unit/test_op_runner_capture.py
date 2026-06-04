#!/usr/bin/python3
# coding=utf-8

"""
单元测试：op_runner 捕获算子输出功能
"""

import pytest
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

from kernel_eval.eval.op_runner import OpRunner, OpRunResult, capture_output
from kernel_eval.utils.device_manager import DeviceManager


class TestCaptureOutput:
    """测试 capture_output 上下文管理器"""

    def test_capture_stdout(self):
        """测试捕获 stdout"""
        with capture_output() as (cap_out, cap_err):
            print("test output")

        assert "test output" in cap_out.getvalue()
        assert cap_err.getvalue() == ""

    def test_capture_stderr(self):
        """测试捕获 stderr"""
        with capture_output() as (cap_out, cap_err):
            print("error message", file=sys.stderr)

        assert cap_out.getvalue() == ""
        assert "error message" in cap_err.getvalue()

    def test_capture_both(self):
        """测试同时捕获 stdout 和 stderr"""
        with capture_output() as (cap_out, cap_err):
            print("stdout message")
            print("stderr message", file=sys.stderr)

        assert "stdout message" in cap_out.getvalue()
        assert "stderr message" in cap_err.getvalue()

    def test_restore_after_context(self):
        """测试上下文退出后恢复原始 stdout/stderr"""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        with capture_output() as (cap_out, cap_err):
            pass

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


class TestOpRunnerCapture:
    """测试 OpRunner 捕获算子输出功能"""

    def setup_method(self):
        """测试前准备"""
        self.device_manager = MagicMock(spec=DeviceManager)
        self.device_manager.is_npu_mode.return_value = False
        self.device_manager.to_device_batch.side_effect = lambda x: x
        self.device_manager.get_device.return_value = "cpu"
        self.device_manager.synchronize.return_value = None

        self.runner = OpRunner(self.device_manager)

    def test_capture_output_on_success(self):
        """测试成功执行时捕获输出"""
        import torch

        def success_func(x):
            print("operator log message")
            return x * 2

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.runner._run_simple(
            success_func,
            {"x": input_tensor},
            [input_tensor]
        )

        assert result.success is True
        assert result.captured_output is None  # 成功时不保存捕获输出
        assert torch.allclose(result.outputs, torch.tensor([2.0, 4.0, 6.0]))

    def test_capture_output_on_failure(self):
        """测试失败执行时捕获输出"""
        import torch

        def fail_func(x):
            print("error: unsupported dtype", file=sys.stderr)
            raise RuntimeError("operator failed")

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.runner._run_simple(
            fail_func,
            {"x": input_tensor},
            [input_tensor]
        )

        assert result.success is False
        assert result.captured_output is not None
        assert "error: unsupported dtype" in result.captured_output
        assert "operator failed" in result.error
        assert "算子执行期间输出" in result.error

    def test_capture_output_on_none_return(self):
        """测试算子返回 None 时捕获输出"""
        import torch

        def none_func(x):
            print("warning: bfloat16 not supported", file=sys.stderr)
            return None

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.runner._run_simple(
            none_func,
            {"x": input_tensor},
            [input_tensor]
        )

        assert result.success is False
        assert result.captured_output is not None
        assert "bfloat16 not supported" in result.captured_output
        assert "算子执行返回 None" in result.error

    def test_no_capture_on_clean_failure(self):
        """测试失败但无输出时不保存 captured_output"""
        import torch

        def silent_fail_func(x):
            raise RuntimeError("silent error")

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = self.runner._run_simple(
            silent_fail_func,
            {"x": input_tensor},
            [input_tensor]
        )

        assert result.success is False
        assert result.captured_output is None  # 无输出时为 None
        assert "silent error" in result.error


class TestOpRunnerProfilerPath:
    """测试 OpRunner profiler 路径的错误捕获"""

    def test_profiler_path_catches_error_from_perf_result(self):
        """测试 profiler 路径从 perf_result.error_msg 捕获真实错误"""
        import torch
        from kernel_eval.eval.perf_eval import PerfResult

        device_manager = MagicMock(spec=DeviceManager)
        device_manager.is_npu_mode.return_value = True
        device_manager.to_device_batch.side_effect = lambda x: x
        device_manager.get_device.return_value = "npu:0"

        perf_evaluator = MagicMock()
        perf_evaluator.config.enable_profiler = True

        # 模拟 run_profiled 返回 (None, perf_result_with_error)
        # 这模拟了算子在 profiler 中执行失败的情况
        perf_result = PerfResult(
            error_msg="call aclnnExp failed, detail:AclNN_Parameter_Error(EZ1001): Dtype not supported"
        )
        perf_evaluator.run_profiled.return_value = (None, perf_result)
        perf_evaluator.wait_all.return_value = None

        runner = OpRunner(device_manager, perf_evaluator)

        def mock_func(x):
            return x * 2

        input_tensor = torch.tensor([1.0, 2.0, 3.0])
        result = runner.run(
            mock_func,
            {"x": input_tensor},
            "test_case_1",
            [input_tensor],
            enable_profiler=True
        )

        assert result.success is False
        # 关键断言：错误信息应包含 perf_result.error_msg 的内容
        assert "AclNN_Parameter_Error" in result.error
        assert "Dtype not supported" in result.error
        assert "EZ1001" in result.error
        # 不应是泛化的 "算子执行返回 None" 错误
        assert "算子执行返回 None" not in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
