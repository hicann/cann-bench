# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Bounded streaming of an evaluation worker's output."""

import os
import selectors
import signal
import time

from kernel_eval.eval.subprocess_utils import _terminate_process_group


def stream_worker(proc, timeout):
    deadline = time.monotonic() + timeout
    pending = b""
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"evaluation worker exceeded {timeout:g}s")
                if not selector.select(min(remaining, 0.2)):
                    continue
                data = os.read(proc.stdout.fileno(), 65536)
                if not data:
                    if pending:
                        yield pending.decode("utf-8", errors="replace")
                    proc.wait(timeout=max(0.001, deadline - time.monotonic()))
                    return
                pending += data
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    yield line.decode("utf-8", errors="replace") + "\n"
    finally:
        _terminate_process_group(proc, grace_sec=5)
        # Reap descendants even when the leader exited before its children.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
