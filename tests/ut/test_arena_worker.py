# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See LICENSE in the repository root for the full license text.

import subprocess
import sys
import time

import pytest

from ascend_arena_e2e.worker import stream_worker


def test_silent_worker_timeout_reaps_process():
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                          stdout=subprocess.PIPE, start_new_session=True) as proc:
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            list(stream_worker(proc, 0.2))
        assert proc.poll() is not None
        assert time.monotonic() - start < 7


def test_worker_streams_before_process_exit():
    with subprocess.Popen([sys.executable, "-u", "-c",
                           "import time; print('ready'); time.sleep(60)"],
                          stdout=subprocess.PIPE, start_new_session=True) as proc:
        stream = stream_worker(proc, 10)
        assert next(stream) == "ready\n"
        assert proc.poll() is None
        stream.close()
        assert proc.poll() is not None


def test_shared_thresholds_are_identical_objects():
    from ascend_arena_e2e.ascend_arena.evaluator import thresholds as arena
    from kernel_eval.utils import thresholds as common
    assert arena.PRECISION_THRESHOLDS is common.PRECISION_THRESHOLDS
