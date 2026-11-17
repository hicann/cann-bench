# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Subprocess helpers with hard timeouts and process-group cleanup.

Wraps shell-based subprocess invocations so that a hang in the spawned bash
(or any of its descendants — msprof, python profile scripts, etc.) does not
indefinitely hold an NPU device slot. On timeout we SIGTERM the whole session,
wait briefly, then SIGKILL.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Mapping

DEFAULT_BUILD_TIMEOUT_SEC = int(os.environ.get("ASCEND_BUILD_TIMEOUT", "1800"))
DEFAULT_CORRECTNESS_TIMEOUT_SEC = int(os.environ.get("ASCEND_CORRECTNESS_TIMEOUT", "600"))
DEFAULT_PROFILE_TIMEOUT_SEC = int(os.environ.get("ASCEND_PROFILE_TIMEOUT", "600"))

_SIGTERM_GRACE_SEC = 5.0
_SIGKILL_REAP_SEC = 5.0


def run_shell_with_timeout(
    command: str,
    *,
    timeout: float,
    executable: str = "/bin/bash",
    env: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a shell command with a hard timeout.

    On timeout, sends SIGTERM to the whole process group (started via
    setsid), waits up to _SIGTERM_GRACE_SEC, then SIGKILLs and reaps.
    The returned CompletedProcess has returncode=-signal.SIGKILL and the
    stderr tail annotated with the timeout marker so callers can detect it.
    """
    proc = subprocess.Popen(
        command,
        shell=True,
        executable=executable,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=dict(env) if env is not None else None,
        cwd=cwd,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            args=command, returncode=proc.returncode, stdout=stdout, stderr=stderr
        )
    except subprocess.TimeoutExpired:
        stdout, stderr = _terminate_pg(proc, timeout)
        marker = f"\n[ASCEND_TIMEOUT] killed after {timeout:.0f}s\n"
        return subprocess.CompletedProcess(
            args=command,
            returncode=-signal.SIGKILL,
            stdout=stdout or "",
            stderr=(stderr or "") + marker,
        )


def _terminate_pg(proc: subprocess.Popen[str], timeout: float) -> tuple[str, str]:
    pid = proc.pid
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        return proc.communicate(timeout=_SIGTERM_GRACE_SEC)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        try:
            return proc.communicate(timeout=_SIGKILL_REAP_SEC)
        except subprocess.TimeoutExpired:
            return "", ""
