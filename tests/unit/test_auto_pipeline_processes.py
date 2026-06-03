import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from auto_pipeline.core import cleanup_process_on_exit, terminate_process_family


def test_cleanup_process_on_exit_kills_detached_descendants(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    parent_code = (
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "    start_new_session=True,\n"
        ")\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        "print('ready', flush=True)\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        child_pid = _wait_for_pid_file(child_pid_file, process)
        assert _pid_is_alive(child_pid)

        with pytest.raises(KeyboardInterrupt):
            with cleanup_process_on_exit(process, grace_sec=0.2):
                raise KeyboardInterrupt

        assert _wait_until(lambda: process.poll() is not None)
        assert _wait_until(lambda: not _pid_is_alive(child_pid))
    finally:
        terminate_process_family(process, grace_sec=0.1)


def test_cleanup_process_on_exit_kills_env_matched_orphan_after_parent_exits(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    token = f"token-{time.time_ns()}"
    parent_code = (
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(60)'],\n"
        "    start_new_session=True,\n"
        ")\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
    )
    env = dict(os.environ)
    env["AUTO_PIPELINE_TEST_TOKEN"] = token
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )

    try:
        child_pid = _wait_for_pid_file(child_pid_file, process)
        assert _pid_is_alive(child_pid)
        with cleanup_process_on_exit(
            process,
            grace_sec=0.2,
            match_environ={"AUTO_PIPELINE_TEST_TOKEN": token},
        ):
            process.wait(timeout=5)
        assert _wait_until(lambda: not _pid_is_alive(child_pid))
    finally:
        terminate_process_family(process, grace_sec=0.1)


def _wait_for_pid_file(path: Path, process: subprocess.Popen, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return int(path.read_text(encoding="utf-8").strip())
        if process.poll() is not None:
            _stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(f"parent process exited early: {stderr}")
        time.sleep(0.05)
    terminate_process_family(process, grace_sec=0.1)
    _stdout, stderr = process.communicate(timeout=1)
    raise AssertionError(f"timed out waiting for child pid file: {stderr}")


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _pid_is_alive(pid: int) -> bool:
    try:
        content = Path("/proc") / str(pid) / "stat"
        fields = content.read_text(encoding="utf-8").rsplit(") ", 1)[1].split()
    except (IndexError, OSError):
        return False
    return fields[0] != "Z"
