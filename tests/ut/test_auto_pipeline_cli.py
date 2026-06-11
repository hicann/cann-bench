import argparse
import os
from pathlib import Path

import auto_pipeline.cli as cli


def _parse_run_args(*extra: str):
    return cli.create_run_parser().parse_args([
        "--config",
        "config.yaml",
        "--workspace",
        "workspace",
        *extra,
    ])


def test_run_parser_defaults_to_background_mode():
    args = _parse_run_args()

    assert args.foreground is False


def test_run_parser_accepts_explicit_foreground_mode():
    args = _parse_run_args("--foreground")

    assert args.foreground is True


def test_background_child_runs_in_foreground_to_avoid_respawn():
    args = _parse_run_args("--devices", "2", "--parallel", "1")

    child_args = cli._run_args(args, run_id="run-1", output="out", foreground=True)

    assert "--foreground" in child_args
    assert "--background" not in child_args


def test_main_run_defaults_to_background(monkeypatch):
    calls = {}
    args = _parse_run_args()

    def fake_start_background(run_args, *, run_id):
        calls["run_args"] = run_args
        calls["run_id"] = run_id
        return 0

    def fail_run_from_config(*_args, **_kwargs):
        raise AssertionError("default run should start in background")

    monkeypatch.setattr(cli.pipeline_state, "new_run_id", lambda: "run-1")
    monkeypatch.setattr(cli, "_start_background", fake_start_background)
    monkeypatch.setattr(cli, "run_from_config", fail_run_from_config)

    assert cli._main_run(args, ["--config", "config.yaml", "--workspace", "workspace"]) == 0
    assert calls["run_args"] is args
    assert calls["run_id"] == "run-1"


def test_main_run_foreground_executes_pipeline(monkeypatch):
    calls = {}
    args = _parse_run_args("--foreground", "--output", "out")

    def fail_start_background(*_args, **_kwargs):
        raise AssertionError("foreground run should not start background process")

    def fake_run_from_config(config_path, *, runtime):
        calls["config_path"] = config_path
        calls["runtime"] = runtime
        return 17

    monkeypatch.setattr(cli.pipeline_state, "new_run_id", lambda: "run-1")
    monkeypatch.setattr(cli, "_start_background", fail_start_background)
    monkeypatch.setattr(cli, "run_from_config", fake_run_from_config)

    assert cli._main_run(
        args,
        ["--config", "config.yaml", "--workspace", "workspace", "--foreground", "--output", "out"],
    ) == 17
    assert calls["config_path"] == Path("config.yaml")
    assert calls["runtime"]["run_id"] == "run-1"
    assert calls["runtime"]["output"] == "out"
    assert "--foreground" in calls["runtime"]["command"]


def test_watch_monitor_clears_screen_on_refresh(monkeypatch, capsys):
    args = argparse.Namespace(run_id="run-1", output=None, latest=False, json=False)
    run = {"run_id": "run-1", "status": "running", "tasks": []}

    monkeypatch.setattr(cli, "_select_run", lambda _args: run)

    def stop_after_first_refresh(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", stop_after_first_refresh)

    assert cli._watch_run(args, interval=0.1, clear=True, stop_when_terminal=False) == 0
    captured = capsys.readouterr()

    assert captured.out.startswith(cli._CLEAR_SCREEN)
    assert "run_id = run-1" in captured.out
    assert "monitor closed; background run is untouched" in captured.err


def test_watch_monitor_stops_when_auto_started_run_is_terminal(monkeypatch, capsys):
    args = argparse.Namespace(run_id="run-1", output=None, latest=False, json=False)
    run = {"run_id": "run-1", "status": "success", "tasks": []}

    monkeypatch.setattr(cli, "_select_run", lambda _args: run)

    def fail_sleep(_seconds):
        raise AssertionError("terminal auto monitor should not sleep again")

    monkeypatch.setattr(cli.time, "sleep", fail_sleep)

    assert cli._watch_run(args, interval=0.1, clear=True, stop_when_terminal=True) == 0
    captured = capsys.readouterr()

    assert captured.out.startswith(cli._CLEAR_SCREEN)
    assert "status = success" in captured.out


def test_start_background_opens_monitor_by_default(monkeypatch, tmp_path):
    args = _parse_run_args("--output", str(tmp_path / "out"), "--monitor-interval", "0.25")
    calls = {}

    class FakeProcess:
        pid = 12345

    def fake_popen(command, **kwargs):
        calls["command"] = command
        calls["popen_kwargs"] = kwargs
        return FakeProcess()

    def fake_watch(selector, *, interval, clear, stop_when_terminal):
        calls["selector"] = selector
        calls["interval"] = interval
        calls["clear"] = clear
        calls["stop_when_terminal"] = stop_when_terminal
        return 0

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli.os, "getpgid", lambda pid: 12345)
    monkeypatch.setattr(cli.pipeline_state, "run_dir", lambda run_id: tmp_path / run_id)
    monkeypatch.setattr(cli.pipeline_state, "upsert_run", lambda run_id, payload: calls.setdefault("run_payload", payload))
    monkeypatch.setattr(cli, "_watch_run", fake_watch)

    assert cli._start_background(args, run_id="run-1") == 0

    assert calls["command"][:4] == [cli.sys.executable, "-m", "auto_pipeline.cli", "run"]
    assert "--foreground" in calls["command"]
    assert calls["selector"].run_id == "run-1"
    assert calls["interval"] == 0.25
    assert calls["clear"] is True
    assert calls["stop_when_terminal"] is True
    assert calls["run_payload"]["pid"] == 12345


def test_start_background_can_skip_auto_monitor(monkeypatch, tmp_path, capsys):
    args = _parse_run_args("--output", str(tmp_path / "out"), "--no-monitor")
    calls = {}

    class FakeProcess:
        pid = 12345

    monkeypatch.setattr(cli.subprocess, "Popen", lambda command, **kwargs: FakeProcess())
    monkeypatch.setattr(cli.os, "getpgid", lambda pid: 12345)
    monkeypatch.setattr(cli.pipeline_state, "run_dir", lambda run_id: tmp_path / run_id)
    monkeypatch.setattr(cli.pipeline_state, "upsert_run", lambda run_id, payload: calls.setdefault("run_payload", payload))
    monkeypatch.setattr(cli, "_watch_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("monitor disabled")))

    assert cli._start_background(args, run_id="run-1") == 0
    captured = capsys.readouterr()

    assert "run_id: run-1" in captured.out
    assert "pid: 12345" in captured.out


def test_main_monitor_defaults_to_watch_and_exits_after_terminal_snapshot(monkeypatch, capsys):
    args = argparse.Namespace(run_id="run-1", output=None, latest=False, json=False, once=False, interval=0.1)
    run = {"run_id": "run-1", "status": "failed", "tasks": []}

    monkeypatch.setattr(cli, "_select_run", lambda _args: run)

    def fail_sleep(_seconds):
        raise AssertionError("terminal monitor should render once and exit")

    monkeypatch.setattr(cli.time, "sleep", fail_sleep)

    assert cli._main_monitor(args) == 0
    captured = capsys.readouterr()

    assert captured.out.startswith(cli._CLEAR_SCREEN)
    assert "status = failed" in captured.out


def test_main_monitor_once_renders_single_snapshot(monkeypatch, capsys):
    args = argparse.Namespace(run_id="run-1", output=None, latest=False, json=False, once=True, interval=0.1)
    run = {"run_id": "run-1", "status": "running", "tasks": []}

    monkeypatch.setattr(cli, "_select_run", lambda _args: run)

    def fail_sleep(_seconds):
        raise AssertionError("once monitor should not sleep")

    monkeypatch.setattr(cli.time, "sleep", fail_sleep)

    assert cli._main_monitor(args) == 0
    captured = capsys.readouterr()

    assert not captured.out.startswith(cli._CLEAR_SCREEN)
    assert "status = running" in captured.out


def test_format_tables_include_elapsed_duration():
    run = {
        "run_id": "run-1",
        "status": "success",
        "created_at": "2026-06-08T00:00:00Z",
        "completed_at": "2026-06-08T01:02:03Z",
        "updated_at": "2026-06-08T01:02:03Z",
        "pid": 123,
        "output": "out",
        "summary": {"total": 1, "running": 0, "success": 1, "failed": 0},
        "tasks": [
            {
                "task_id": "exp",
                "status": "success",
                "stage": "eval",
                "created_at": "2026-06-08T00:00:00Z",
                "updated_at": "2026-06-08T01:02:03Z",
                "token_usage": {"total": 12},
                "stage_times": {
                    "generation": {
                        "started_at": "2026-06-08T00:00:00Z",
                        "ended_at": "2026-06-08T00:10:00Z",
                    },
                    "conversion": {
                        "started_at": "2026-06-08T00:10:00Z",
                        "ended_at": "2026-06-08T00:30:00Z",
                    },
                    "eval": {
                        "started_at": "2026-06-08T00:30:00Z",
                        "ended_at": "2026-06-08T01:02:03Z",
                    },
                },
            }
        ],
    }

    runs_table = cli._format_runs_table([run])
    monitor_table = cli._format_monitor_table(run)
    display_updated = cli._format_display_timestamp("2026-06-08T01:02:03Z")

    assert "ELAPSED" in runs_table
    assert "GEN" in monitor_table
    assert "CONVERT" in monitor_table
    assert "EVAL" in monitor_table
    assert "TOTAL" in monitor_table
    assert "PID" not in monitor_table
    assert "LOG/RESULT" not in monitor_table
    assert "01:02:03" in runs_table
    assert display_updated in runs_table
    assert "2026-06-08T01:02:03Z" not in runs_table
    assert "00:10:00" in monitor_table
    assert "00:20:00" in monitor_table
    assert "00:32:03" in monitor_table
    assert "01:02:03" in monitor_table
    assert display_updated in monitor_table
    assert "2026-06-08T01:02:03Z" not in monitor_table


def test_task_total_elapsed_ignores_pending_wait_time():
    task = {
        "task_id": "waiting",
        "status": "pending",
        "created_at": "2026-06-08T00:00:00Z",
        "updated_at": "2026-06-08T00:05:43Z",
    }

    assert (
        cli._format_task_total_elapsed(task, now=cli._parse_utc_timestamp("2026-06-08T00:05:43Z"))
        == "00:00:00"
    )


def test_task_total_elapsed_includes_active_generation_stage(monkeypatch):
    now = cli._parse_utc_timestamp("2026-06-08T00:05:43Z")
    assert now is not None
    monkeypatch.setattr(cli.time, "time", lambda: now)
    task = {
        "task_id": "exp",
        "status": "running",
        "stage": "generation",
        "stage_times": {
            "generation": {"started_at": "2026-06-08T00:00:00Z"},
        },
    }

    assert cli._format_stage_elapsed_for_task(task, "generation") == "00:05:43"
    assert cli._format_task_total_elapsed(task) == "00:05:43"


def test_monitor_table_total_matches_stage_time_not_wait(monkeypatch):
    now = cli._parse_utc_timestamp("2026-06-08T00:05:43Z")
    assert now is not None
    monkeypatch.setattr(cli.time, "time", lambda: now)
    run = {
        "run_id": "run-1",
        "status": "running",
        "tasks": [
            {
                "task_id": "waiting",
                "status": "pending",
                "stage": "pending",
                "created_at": "2026-06-08T00:00:00Z",
                "updated_at": "2026-06-08T00:05:43Z",
            },
            {
                "task_id": "exp",
                "status": "running",
                "stage": "generation",
                "stage_times": {
                    "generation": {"started_at": "2026-06-08T00:00:00Z"},
                },
                "updated_at": "2026-06-08T00:05:43Z",
            },
        ],
    }

    table = cli._format_monitor_table(run)
    rows = {
        cells[0]: cells
        for cells in (line.split() for line in table.splitlines())
        if cells and cells[0] in {"waiting", "exp"}
    }

    assert rows["waiting"][2:6] == ["00:00:00", "00:00:00", "00:00:00", "00:00:00"]
    assert rows["exp"][2:6] == ["00:05:43", "00:00:00", "00:00:00", "00:05:43"]


def test_format_display_timestamp_uses_local_timezone(monkeypatch):
    old_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "CST-8")
    if hasattr(cli.time, "tzset"):
        cli.time.tzset()
    try:
        assert cli._format_display_timestamp("2026-06-07T18:44:28Z") == "2026-06-08 02:44:28"
    finally:
        if old_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", old_tz)
        if hasattr(cli.time, "tzset"):
            cli.time.tzset()
