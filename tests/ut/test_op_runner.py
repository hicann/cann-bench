from contextlib import nullcontext
from types import SimpleNamespace

from kernel_eval.base.result import PerfResult
from kernel_eval.eval.op_runner import OpRunner


class FakeDeviceManager:
    def to_device(self, tensor):
        return tensor

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


class FakeDeferredCall:
    def __init__(self):
        self.release_count = 0

    def __call__(self):
        return None

    def release(self):
        self.release_count += 1


class FakeBatchPerfEvaluator:
    config = SimpleNamespace(enable_profiler=True)
    freq_boost = False

    def run_profiled_batch(self, pending):
        return {
            case_id: PerfResult(error_msg="batch invariant: delimiter mismatch")
            for case_id, _ in pending
        }

    def run_profiled(self, case_id, func):
        return None, PerfResult(elapsed_us=1.0)

    def wait_all(self):
        return None


def test_batch_fallback_reason_is_printed_once_with_case_count(capsys):
    runner = OpRunner(FakeDeviceManager(), FakeBatchPerfEvaluator())
    runner._profile_guard = lambda: (nullcontext(), nullcontext())
    runner._perf_batch_active = True
    runner._perf_batch_cases = {
        "case_1": FakeDeferredCall(),
        "case_2": FakeDeferredCall(),
    }

    results = runner.finalize_perf_batch()

    output = capsys.readouterr().out
    assert "批量性能采集回退原因 (2 个 case)" in output
    assert "batch invariant: delimiter mismatch" in output
    assert all(
        result.metadata["perf_batch_fallback_reason"]
        == "batch invariant: delimiter mismatch"
        for result in results.values()
    )


class FakePartialBatchPerfEvaluator(FakeBatchPerfEvaluator):
    def __init__(self):
        self.per_case_calls = []

    def run_profiled_batch(self, pending):
        return {
            "case_1": PerfResult(
                elapsed_us=12.0,
                metadata={"perf_batch_used": True},
            ),
            "case_2": PerfResult(
                error_msg="batch anti-cheat requires per-case fallback"
            ),
        }

    def run_profiled(self, case_id, func):
        self.per_case_calls.append(case_id)
        return None, PerfResult(elapsed_us=2.0)


def test_batch_falls_back_only_failed_case(capsys):
    perf = FakePartialBatchPerfEvaluator()
    runner = OpRunner(FakeDeviceManager(), perf)
    runner._profile_guard = lambda: (nullcontext(), nullcontext())
    runner._perf_batch_active = True
    runner._perf_batch_cases = {
        "case_1": FakeDeferredCall(),
        "case_2": FakeDeferredCall(),
    }

    results = runner.finalize_perf_batch()

    output = capsys.readouterr().out
    assert "批量性能采集回退原因 (1 个 case)" in output
    assert perf.per_case_calls == ["case_2"]
    assert results["case_1"].elapsed_us == 12.0
    assert results["case_2"].elapsed_us == 2.0
    assert results["case_2"].metadata["perf_batch_fallback"] is True
