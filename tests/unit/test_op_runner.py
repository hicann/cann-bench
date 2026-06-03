from types import SimpleNamespace

from kernel_eval.base.result import PerfResult
from kernel_eval.eval.op_runner import OpRunner


class FakeDeviceManager:
    def to_device_batch(self, tensors):
        return tensors

    def is_npu_mode(self):
        return True

    def get_device(self):
        return "npu:0"

    def synchronize(self):
        return None


class FakePerfEvaluator:
    config = SimpleNamespace(enable_profiler=True)

    def run_profiled(self, case_id, func, **kwargs):
        return None, PerfResult(
            error_msg="TypeError: raw callable failure",
            metadata={
                "profile_exception_traceback": (
                    "Traceback (most recent call last):\n"
                    "  File \"candidate.py\", line 1, in run\n"
                    "TypeError: raw callable failure\n"
                ),
            },
        )

    def wait_all(self):
        return None


def test_profiled_ai_error_is_not_treated_as_success():
    runner = OpRunner(FakeDeviceManager(), FakePerfEvaluator())

    result = runner.run(lambda: None, {}, "sigmoid_8", [], enable_profiler=True)

    assert result.success is False
    assert "raw callable failure" in result.error
    assert "Traceback (most recent call last)" in result.error
    assert result.traceback is not None
