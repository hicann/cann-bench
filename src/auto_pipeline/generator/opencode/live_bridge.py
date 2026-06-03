"""Live OpenCode event bridge for subagent progress.

OpenCode's CLI JSON stream only exposes the root session.  A small server
plugin can see the internal event bus and write session/part events to JSONL;
this module tails that JSONL and writes lightweight per-session append-only
Markdown streams while the run is still active.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote


PLUGIN_PATH = Path(__file__).with_name("opencode_live_bridge_plugin.js")
_TAIL_POLL_SEC = 0.25


class OpencodeLiveBridge:
    """Tail OpenCode plugin JSONL output and render live session streams."""

    def __init__(self, *, output_dir: Path) -> None:
        self.live_dir = Path(output_dir) / "opencode-live"
        self.events_file = self.live_dir / "events.jsonl"
        self.nodes_dir = self.live_dir / "nodes"
        self.session_tree_file = self.live_dir / "session_tree.tsv"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self._error: Optional[str] = None
        self._reset_state()

    def configure_env(self, env: dict[str, str]) -> None:
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        env["OPENCODE_SUBAGENT_BRIDGE_LOG"] = str(self.events_file)
        env["OPENCODE_CONFIG_CONTENT"] = _opencode_config_with_plugin(
            env.get("OPENCODE_CONFIG_CONTENT"),
            PLUGIN_PATH,
        )

    def start(self) -> None:
        with self._lock:
            self._reset_state()
            self._prepare_live_outputs()
        self._stop.clear()
        self._thread = threading.Thread(target=self._tail_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.finalize()

    def finalize(self) -> None:
        with self._lock:
            self._drain_events_file(render=True)
            if self._sessions:
                self._render_session_tree()
            self._close_open_tool_blocks()

    def summary(self) -> dict[str, object]:
        with self._lock:
            sessions = dict(self._sessions)
            event_count = self._event_count
            root_sessions = [sid for sid, info in sessions.items() if not info.get("parentID")]
            subagent_sessions = [sid for sid, info in sessions.items() if info.get("parentID")]
            status = "missing"
            if self.events_file.is_file():
                status = "captured" if event_count else "empty"
            if self._error:
                status = "error"
            return {
                "status": status,
                "live_dir": str(self.live_dir),
                "events_file": str(self.events_file),
                "session_tree_file": str(self.session_tree_file),
                "node_session_count": len(sessions),
                "subagent_session_count": len(subagent_sessions),
                "root_session_ids": root_sessions,
                "subagent_session_ids": subagent_sessions,
                "subagent_titles": [
                    str(sessions[sid].get("title") or sid)
                    for sid in subagent_sessions
                ],
                "event_count": event_count,
                "error": self._error or "",
            }

    def _reset_state(self) -> None:
        self._sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._parts: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._tool_outputs: dict[str, str] = {}
        self._event_count = 0
        self._tail_offset = 0
        self._session_paths: dict[str, Path] = {}
        self._initialized_sessions: set[str] = set()
        self._seen_text_parts: set[tuple[str, str]] = set()
        self._seen_tool_parts: set[tuple[str, str]] = set()
        self._tool_output_seen: set[tuple[str, str]] = set()
        self._open_tool_blocks: set[tuple[str, str]] = set()
        self._part_status: dict[tuple[str, str], str] = {}
        self._session_tree_dirty = True

    def _prepare_live_outputs(self) -> None:
        self.live_dir.mkdir(parents=True, exist_ok=True)
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        for path in [self.events_file, self.session_tree_file]:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for path in list(self.nodes_dir.glob("*.live.md")) + list(self.nodes_dir.glob("*.live.md.tmp")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _tail_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.events_file.is_file():
                    time.sleep(_TAIL_POLL_SEC)
                    continue
                self._drain_events_file(render=True)
            except Exception as exc:  # pragma: no cover - defensive observability path
                self._error = str(exc)
                time.sleep(_TAIL_POLL_SEC)
            time.sleep(_TAIL_POLL_SEC)

    def _drain_events_file(self, *, render: bool) -> None:
        if not self.events_file.is_file():
            return
        size = self.events_file.stat().st_size
        if size < self._tail_offset:
            self._tail_offset = 0
        with self.events_file.open("r", encoding="utf-8") as handle:
            handle.seek(self._tail_offset)
            for line in handle:
                self._process_jsonl_line(line, render=render)
            self._tail_offset = handle.tell()

    def _process_jsonl_line(self, line: str, *, render: bool = True) -> None:
        line = line.strip()
        if not line:
            return
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            return
        if record.get("kind") != "event":
            return
        with self._lock:
            changed_session = self._ingest_event(record)
            if render and changed_session:
                if self._session_tree_dirty:
                    self._render_session_tree()
                    self._session_tree_dirty = False
                self._append_event_markdown(changed_session, record)

    def _ingest_event(self, record: Mapping[str, Any]) -> Optional[str]:
        self._event_count += 1
        session_info = record.get("session")
        if isinstance(session_info, Mapping) and session_info.get("id"):
            session_id = str(session_info["id"])
            existing = self._sessions.get(session_id, {})
            merged = {**existing, **dict(session_info)}
            if record.get("parentID") and not merged.get("parentID"):
                merged["parentID"] = record.get("parentID")
            if merged != existing:
                self._session_tree_dirty = True
            self._sessions[session_id] = merged

        delta = record.get("delta")
        if isinstance(delta, Mapping) and delta.get("partID"):
            session_id = str(delta.get("sessionID") or record.get("sessionID") or "")
            if not session_id:
                return None
            if session_id not in self._sessions:
                self._sessions[session_id] = {"id": session_id}
                self._session_tree_dirty = True
            part_id = str(delta["partID"])
            key = (session_id, part_id)
            existing = self._parts.get(key, {})
            text = delta.get("text")
            field = str(delta.get("field") or "text")
            merged = {
                **existing,
                "id": part_id,
                "sessionID": session_id,
                "messageID": delta.get("messageID") or existing.get("messageID"),
            }
            if field == "text" and isinstance(text, str) and text:
                merged["text"] = str(merged.get("text") or "") + text
            self._parts[key] = merged
            return session_id

        part = record.get("part")
        if not isinstance(part, Mapping) or not part.get("id"):
            session_id = record.get("sessionID")
            return str(session_id) if isinstance(session_id, str) else None

        session_id = str(part.get("sessionID") or record.get("sessionID") or "")
        if not session_id:
            return None
        if session_id not in self._sessions:
            self._sessions[session_id] = {"id": session_id}
            self._session_tree_dirty = True

        part_id = str(part["id"])
        key = (session_id, part_id)
        existing = self._parts.get(key, {})
        incoming = dict(part)
        if incoming.get("type") != "tool":
            incoming.pop("text", None)
        merged = {**existing, **incoming}

        call_id = str(merged.get("callID") or "")
        if call_id:
            delta = part.get("outputDelta")
            if isinstance(delta, str) and delta:
                self._tool_outputs[call_id] = self._tool_outputs.get(call_id, "") + delta
            output = part.get("output")
            if isinstance(output, str) and output:
                self._tool_outputs[call_id] = output
            if call_id in self._tool_outputs:
                merged["liveOutput"] = self._tool_outputs[call_id]

        self._parts[key] = merged
        return session_id

    def _render_all(self) -> None:
        self._render_session_tree()
        for session_id in self._sessions:
            self._ensure_session_header(session_id)

    def _render_session_tree(self) -> None:
        rows = [
            "order\trole\tsession_id\tparent_id\ttitle\tmd_file",
        ]
        for index, (session_id, info) in enumerate(self._sessions.items()):
            role = "subagent" if info.get("parentID") else "root"
            md_path = self._session_paths.get(session_id) or self._session_file(session_id, info)
            md_file = _relative_to(md_path, self.live_dir)
            rows.append(
                "\t".join(
                    [
                        str(index),
                        role,
                        session_id,
                        str(info.get("parentID") or ""),
                        _tsv(str(info.get("title") or "")),
                        md_file,
                    ]
                )
            )
        _write_text(self.session_tree_file, "\n".join(rows) + "\n")

    def _ensure_session_header(self, session_id: str) -> Path:
        info = self._sessions.get(session_id, {"id": session_id})
        path = self._session_paths.get(session_id)
        if path is None:
            path = self._session_file(session_id, info)
            self._session_paths[session_id] = path
        if session_id in self._initialized_sessions:
            return path

        role = "Subagent" if info.get("parentID") else "Root"
        lines = [
            f"# {role}: {info.get('title') or session_id}",
            "",
            f"**Session ID:** {session_id}",
            "",
        ]
        if info.get("parentID"):
            lines.extend([f"**Parent Session ID:** {info.get('parentID')}", ""])
        _append_text(path, "\n".join(lines).rstrip() + "\n")
        self._initialized_sessions.add(session_id)
        return path

    def _append_event_markdown(self, session_id: str, record: Mapping[str, Any]) -> None:
        self._ensure_session_header(session_id)
        delta = record.get("delta")
        if isinstance(delta, Mapping) and delta.get("partID"):
            self._append_text_delta(session_id, delta)
            return

        part = record.get("part")
        if not isinstance(part, Mapping) or not part.get("id"):
            return
        if part.get("type") == "tool":
            self._append_tool_part(session_id, part)

    def _append_text_delta(self, session_id: str, delta: Mapping[str, Any]) -> None:
        field = str(delta.get("field") or "text")
        text = delta.get("text")
        if field != "text" or not isinstance(text, str) or not text:
            return
        part_id = str(delta["partID"])
        key = (session_id, part_id)
        path = self._ensure_session_header(session_id)
        prefix = "\n\n" if key not in self._seen_text_parts else ""
        self._seen_text_parts.add(key)
        _append_text(path, prefix + text)

    def _append_tool_part(self, session_id: str, part: Mapping[str, Any]) -> None:
        part_id = str(part["id"])
        key = (session_id, part_id)
        path = self._ensure_session_header(session_id)
        tool = str(part.get("tool") or "tool")
        status = str(part.get("status") or "")
        previous_status = self._part_status.get(key)
        is_new_part = key not in self._seen_tool_parts
        chunks: list[str] = []

        if is_new_part:
            chunks.append(f"\n\n## Tool: {tool}\n\n")
            if status:
                chunks.append(f"Status: {status}\n\n")
            raw_input = part.get("input")
            if raw_input:
                chunks.append(f"Input:\n```json\n{raw_input}\n```\n\n")
            self._seen_tool_parts.add(key)

        output_delta = part.get("outputDelta")
        if not isinstance(output_delta, str) or not output_delta:
            output_delta = part.get("output") if key not in self._tool_output_seen else None
        if isinstance(output_delta, str) and output_delta:
            if key not in self._open_tool_blocks:
                chunks.append("Output:\n```text\n")
                self._open_tool_blocks.add(key)
            chunks.append(output_delta)
            self._tool_output_seen.add(key)

        terminal_statuses = {"completed", "error", "failed", "cancelled", "canceled"}
        status_changed = bool(status and previous_status is not None and status != previous_status)
        if status:
            self._part_status[key] = status
        if status in terminal_statuses and key in self._open_tool_blocks:
            chunks.append("\n```\n")
            self._open_tool_blocks.remove(key)
        if status_changed:
            chunks.append(f"\nStatus: {status}\n")

        if chunks:
            _append_text(path, "".join(chunks))

    def _close_open_tool_blocks(self) -> None:
        for session_id, part_id in list(self._open_tool_blocks):
            path = self._session_paths.get(session_id)
            if path is not None:
                _append_text(path, "\n```\n")
            self._open_tool_blocks.remove((session_id, part_id))

    def _session_file(self, session_id: str, info: Mapping[str, Any]) -> Path:
        role = "subagent" if info.get("parentID") else "root"
        title = _safe_name(str(info.get("title") or session_id))
        return self.nodes_dir / f"{role}__{session_id}__{title}.live.md"


def _opencode_config_with_plugin(existing: Optional[str], plugin_path: Path) -> str:
    plugin_uri = _file_uri(plugin_path)
    data: dict[str, Any] = {}
    if existing:
        try:
            parsed = json.loads(existing)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            data.update(parsed)
    plugins = data.get("plugin")
    if not isinstance(plugins, list):
        plugins = []
    if plugin_uri not in plugins:
        plugins.append(plugin_uri)
    data["plugin"] = plugins
    return json.dumps(data, sort_keys=True)


def _file_uri(path: Path) -> str:
    resolved = path.expanduser().resolve()
    return "file://" + quote(str(resolved))


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return (safe or "session")[:96]


def _tsv(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ")


def _relative_to(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
