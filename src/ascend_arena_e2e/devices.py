"""Read-only NPU occupancy checks; never claim exclusive ownership of a device."""

import os
import subprocess
from datetime import datetime, timezone


def parse_device_mapping(output):
    mapping = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            npu_id, chip_id, logic_id = map(int, fields[:3])
        except ValueError:
            continue
        mapping[logic_id] = (npu_id, chip_id)
    return mapping


def npu_smi(*arguments):
    try:
        result = subprocess.run(["npu-smi", *arguments], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"npu-smi unavailable: {type(exc).__name__}") from exc
    if result.returncode:
        raise RuntimeError(f"npu-smi {' '.join(arguments)} failed: {(result.stderr or result.stdout).strip()}")
    return result


def device_snapshot(device_id, *, query=npu_smi, mapping=None):
    record = {"device_id": device_id, "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        mapping = parse_device_mapping(query("info", "-m").stdout) if mapping is None else mapping
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").strip()
        logic_id = int(visible.split(",")[device_id]) if visible else device_id
        npu_id, chip_id = mapping[logic_id]
        result = query("info", "-t", "proc-mem", "-i", str(npu_id), "-c", str(chip_id))
        output = result.stdout + result.stderr
        record.update(logic_id=logic_id, npu_id=npu_id, chip_id=chip_id,
                      state="idle" if "no process in device" in output.lower() else "busy",
                      output=output.strip())
    except (RuntimeError, ValueError, IndexError, KeyError) as exc:
        record.update(state="unknown", error=str(exc))
    return record


def discover_idle_devices(*, query=npu_smi, log=print):
    mapping = parse_device_mapping(query("info", "-m").stdout)
    if not mapping:
        raise RuntimeError("could not parse any NPU from `npu-smi info -m`")
    idle, busy = [], []
    for logic_id, (npu_id, chip_id) in sorted(mapping.items()):
        try:
            result = query("info", "-t", "proc-mem", "-i", str(npu_id), "-c", str(chip_id))
            available = "no process in device" in (result.stdout + result.stderr).lower()
        except RuntimeError as exc:
            log(f"[devices] NPU {logic_id} skipped: {exc}")
            available = False
        (idle if available else busy).append(logic_id)
    return idle, busy
