import os
import subprocess
import sys
import time

import auto_pipeline.state as pipeline_state


def test_list_tasks_preserves_declared_order(tmp_path):
    root = tmp_path / "cann-bench"
    pipeline_state.upsert_run(
        "run-1",
        {
            "tasks_declared": [
                {"task_id": "zeta", "name": "zeta", "selector": "level1/zeta"},
                {"task_id": "alpha", "name": "alpha", "selector": "level1/alpha"},
            ]
        },
        cann_bench_root=root,
    )
    pipeline_state.update_task("run-1", "alpha", {"status": "pending"}, cann_bench_root=root)
    pipeline_state.update_task("run-1", "zeta", {"status": "pending"}, cann_bench_root=root)

    tasks = pipeline_state.list_tasks("run-1", cann_bench_root=root)

    assert [task["task_id"] for task in tasks] == ["zeta", "alpha"]


def test_list_tasks_prefers_task_index_over_filename_order(tmp_path):
    root = tmp_path / "cann-bench"
    pipeline_state.upsert_run("run-1", {"tasks_declared": []}, cann_bench_root=root)
    pipeline_state.update_task("run-1", "alpha", {"task_index": 1, "status": "pending"}, cann_bench_root=root)
    pipeline_state.update_task("run-1", "zeta", {"task_index": 0, "status": "pending"}, cann_bench_root=root)

    tasks = pipeline_state.list_tasks("run-1", cann_bench_root=root)

    assert [task["task_id"] for task in tasks] == ["zeta", "alpha"]


def test_kill_run_refuses_pid_identity_mismatch(monkeypatch):
    calls = []
    callbacks = []

    monkeypatch.setattr(pipeline_state, "_pid_identity_status", lambda _pid, _start_time: "mismatch")
    monkeypatch.setattr(pipeline_state.os, "kill", lambda *args: calls.append(args))

    result = pipeline_state.kill_run(
        {
            "run_id": "run-1",
            "pid": 12345,
            "pid_start_time": 67890,
        },
        grace_sec=0,
        on_signal_start=lambda: callbacks.append("called"),
    )

    assert result["ok"] is False
    assert calls == []
    assert callbacks == []
    assert "identity mismatch" in "\n".join(result["errors"])


def test_kill_run_calls_signal_start_hook_before_first_signal(monkeypatch):
    calls = []

    monkeypatch.setattr(pipeline_state, "_pid_identity_status", lambda _pid, _start_time: "alive")
    monkeypatch.setattr(pipeline_state, "_pid_matches_start_time", lambda _pid, _start_time: False)
    monkeypatch.setattr(pipeline_state.os, "kill", lambda *args: calls.append(("kill", args)))

    result = pipeline_state.kill_run(
        {
            "run_id": "run-1",
            "pid": 12345,
            "pid_start_time": 67890,
        },
        grace_sec=0,
        on_signal_start=lambda: calls.append(("callback", None)),
    )

    assert result["ok"] is True
    assert calls[0] == ("callback", None)
    assert calls[1] == ("kill", (12345, pipeline_state.signal.SIGTERM))


def test_kill_run_cleans_descendant_tree(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    parent_code = (
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid), encoding='utf-8')\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", parent_code], start_new_session=True)

    try:
        child_pid = _wait_for_pid_file(child_pid_file, process)
        identity = pipeline_state.process_identity(process.pid)
        result = pipeline_state.kill_run(
            {
                "run_id": "run-1",
                **identity,
            },
            grace_sec=0.1,
        )

        assert result["ok"] is True
        assert _wait_until(lambda: process.poll() is not None)
        assert _wait_until(lambda: not _pid_is_alive(child_pid))
        assert any(f"pgid:{identity['pgid']}" in item for item in result["sent"])
        assert any(f"snapshot-pid:{child_pid}" in item for item in result["sent"])
    finally:
        _cleanup_process(process)


def test_kill_run_cleans_env_matched_orphan(tmp_path):
    env = dict(os.environ)
    env["AUTO_PIPELINE_RUN_ID"] = "run-1"
    env["BENCHMARK_OUTPUT_DIR"] = str(tmp_path / "work" / "artifact")
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=env, start_new_session=True)

    try:
        assert _wait_until(lambda: _pid_is_alive(process.pid))
        result = pipeline_state.kill_run(
            {
                "run_id": "run-1",
                "cleanup_env": [
                    {"BENCHMARK_OUTPUT_DIR": env["BENCHMARK_OUTPUT_DIR"]},
                ],
            },
            grace_sec=0.1,
        )

        assert result["ok"] is True
        assert _wait_until(lambda: not _pid_is_alive(process.pid))
        assert any(f"snapshot-pid:{process.pid}" in item for item in result["sent"])
    finally:
        _cleanup_process(process)


def _wait_for_pid_file(path, process: subprocess.Popen, timeout: float = 5.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return int(path.read_text(encoding="utf-8").strip())
        if process.poll() is not None:
            raise AssertionError("parent process exited before writing child pid")
        time.sleep(0.05)
    raise AssertionError("timed out waiting for child pid")


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    stat = pipeline_state._read_proc_stat(pid)
    return stat is not None and stat[2] != "Z"


def _cleanup_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    _cleanup_pid(process.pid)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _cleanup_pid(pid: int) -> None:
    try:
        os.kill(pid, 9)
    except OSError:
        pass
