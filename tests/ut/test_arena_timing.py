import json

import pytest

from ascend_arena_e2e.timing import timed_run


def test_timing_preserves_success_failure_and_resume(tmp_path, capsys):
    path = tmp_path / "timing.jsonl"
    with timed_run(path, "task", "logit") as record:
        record["returncode"] = 0
    with pytest.raises(ValueError):
        with timed_run(path, "evaluation", "logit"):
            raise ValueError("compile failed")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["start", "end", "start", "end"]
    assert rows[0]["id"] == rows[1]["id"] != rows[2]["id"] == rows[3]["id"]
    assert rows[1]["returncode"] == 0
    assert rows[3]["returncode"] == 1
    assert all(row["duration_seconds"] >= 0 and row["ended_at"] >= row["started_at"]
               for row in rows if row["event"] == "end")
    output = capsys.readouterr().err
    assert "任务开始" in output and "评测结束" in output


def test_evaluation_records_failed_preflight(tmp_path, monkeypatch):
    from kernel_eval import optimize

    def fail(args):
        raise SystemExit(2)

    monkeypatch.setattr(optimize, "_evaluate", fail)
    with pytest.raises(SystemExit):
        optimize.main(["--operator", "logit", "--reports-dir", str(tmp_path)])
    rows = [json.loads(line) for line in (tmp_path / "timing.jsonl").read_text().splitlines()]
    assert rows[-1]["returncode"] == 2
    assert rows[-1]["phase"] == "comparison"
