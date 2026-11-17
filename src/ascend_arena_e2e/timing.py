"""Wall-clock events for task, agent, and evaluation accounting."""

import json
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


@contextmanager
def timed_run(path: Path, kind: str, operator: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"id": uuid.uuid4().hex, "kind": kind, "operator": operator,
              "started_at": datetime.now().astimezone().isoformat(), "returncode": 1}
    started = time.monotonic()

    def emit(event):
        payload = {**record, "event": event}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        label = {"task": "任务", "agent": "Agent", "evaluation": "评测"}[kind]
        stamp = payload.get("ended_at", payload["started_at"])
        suffix = (f" 耗时={payload['duration_seconds']:.3f}s 退出码={payload['returncode']}"
                  if event == "end" else "")
        print(f"[{operator}] {label}{'开始' if event == 'start' else '结束'} {stamp}{suffix}", file=sys.stderr, flush=True)

    emit("start")
    try:
        yield record
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            record["returncode"] = 130
        elif isinstance(exc, SystemExit) and isinstance(exc.code, int):
            record["returncode"] = exc.code
        raise
    finally:
        record.update(ended_at=datetime.now().astimezone().isoformat(),
                      duration_seconds=round(time.monotonic() - started, 6))
        emit("end")
