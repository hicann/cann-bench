"""OpenCode session export helpers.

The OpenCode CLI export can truncate large sessions in some environments.  For
benchmark logs we prefer OpenCode's sqlite storage, where the root session and
child/subagent sessions can be reconstructed into a complete transcript tree.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


SESSION_ID_RE = re.compile(r"\bses_[A-Za-z0-9]+\b")
DEFAULT_EXPORT_TIMEOUT_SEC = 120
_TOKEN_FIELDS = ("total", "input", "output", "reasoning")
_CACHE_TOKEN_FIELDS = ("read", "write")


@dataclass
class OpencodeExportResult:
    """Result of one OpenCode transcript export attempt."""

    session_id: Optional[str] = None
    markdown_file: Optional[Path] = None
    json_file: Optional[Path] = None
    export_dir: Optional[Path] = None
    session_tree_file: Optional[Path] = None
    nodes_dir: Optional[Path] = None
    status: str = "skipped"
    message: str = ""
    session_updated_at_ms: Optional[int] = None
    tree_updated_at_ms: Optional[int] = None
    node_session_count: int = 0
    token_usage: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "exported" and self.markdown_file is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "markdown_file": str(self.markdown_file) if self.markdown_file else None,
            "json_file": str(self.json_file) if self.json_file else None,
            "export_dir": str(self.export_dir) if self.export_dir else None,
            "session_tree_file": str(self.session_tree_file) if self.session_tree_file else None,
            "nodes_dir": str(self.nodes_dir) if self.nodes_dir else None,
            "status": self.status,
            "message": self.message,
            "session_updated_at_ms": self.session_updated_at_ms,
            "tree_updated_at_ms": self.tree_updated_at_ms,
            "node_session_count": self.node_session_count,
            "token_usage": self.token_usage,
        }


def empty_token_usage() -> dict[str, Any]:
    return {
        "supported": False,
        "message_count": 0,
        "session_count": 0,
        "total": 0,
        "input": 0,
        "output": 0,
        "reasoning": 0,
        "cache": {"read": 0, "write": 0},
        "cost": 0.0,
        "by_model": {},
    }


def merge_token_usage(usages: Sequence[Optional[Mapping[str, Any]]]) -> dict[str, Any]:
    merged = empty_token_usage()
    by_model: dict[str, dict[str, Any]] = {}
    for usage in usages:
        if not isinstance(usage, Mapping):
            continue
        if usage.get("supported"):
            merged["supported"] = True
        merged["message_count"] += _as_int(usage.get("message_count"))
        merged["session_count"] += _as_int(usage.get("session_count"))
        for key in _TOKEN_FIELDS:
            merged[key] += _as_int(usage.get(key))
        cache = usage.get("cache") if isinstance(usage.get("cache"), Mapping) else {}
        for key in _CACHE_TOKEN_FIELDS:
            merged["cache"][key] += _as_int(cache.get(key))
        merged["cost"] += _as_float(usage.get("cost"))

        raw_by_model = usage.get("by_model")
        if not isinstance(raw_by_model, Mapping):
            continue
        for model_key, model_usage in raw_by_model.items():
            if not isinstance(model_usage, Mapping):
                continue
            target = by_model.setdefault(str(model_key), empty_token_usage())
            target["supported"] = True
            target["message_count"] += _as_int(model_usage.get("message_count"))
            target["session_count"] += _as_int(model_usage.get("session_count"))
            for token_key in _TOKEN_FIELDS:
                target[token_key] += _as_int(model_usage.get(token_key))
            model_cache = (
                model_usage.get("cache")
                if isinstance(model_usage.get("cache"), Mapping)
                else {}
            )
            for cache_key in _CACHE_TOKEN_FIELDS:
                target["cache"][cache_key] += _as_int(model_cache.get(cache_key))
            target["cost"] += _as_float(model_usage.get("cost"))

    merged["cost"] = round(float(merged["cost"]), 12)
    for model_usage in by_model.values():
        model_usage["cost"] = round(float(model_usage["cost"]), 12)
        model_usage["by_model"] = {}
    merged["by_model"] = dict(sorted(by_model.items()))
    return merged


def make_session_title(op_name: str, phase: str = "pypto") -> str:
    """Return a unique, searchable title for an OpenCode run."""

    safe_op = _safe_title_part(op_name, limit=64)
    safe_phase = _safe_title_part(phase, limit=32)
    millis = int(time.time() * 1000)
    unique = uuid.uuid4().hex[:8]
    return f"auto-pipeline:{safe_phase}:{safe_op}:{os.getpid()}:{threading.get_ident()}:{millis}:{unique}"


def find_session_ids(text: str) -> list[str]:
    out: list[str] = []
    seen = set()
    for match in SESSION_ID_RE.finditer(text or ""):
        session_id = match.group(0)
        if session_id not in seen:
            seen.add(session_id)
            out.append(session_id)
    return out


def export_session_from_log(
    *,
    log_file: Optional[Path],
    output_file: Path,
    output_dir: Optional[Path] = None,
    session_title: str = "",
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    allow_latest_fallback: bool = False,
    raw_json_file: Optional[Path] = None,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> OpencodeExportResult:
    session_id = resolve_session_id(
        log_file=log_file,
        session_title=session_title,
        opencode_bin=opencode_bin,
        cwd=cwd,
        allow_latest_fallback=allow_latest_fallback,
        timeout_sec=timeout_sec,
    )
    if not session_id:
        return OpencodeExportResult(
            status="skipped",
            message="failed to resolve OpenCode session id",
            token_usage=empty_token_usage(),
        )
    return export_session_to_markdown(
        session_id=session_id,
        output_file=output_file,
        output_dir=output_dir,
        opencode_bin=opencode_bin,
        cwd=cwd,
        raw_json_file=raw_json_file,
        timeout_sec=timeout_sec,
    )


def resolve_session_id(
    *,
    log_file: Optional[Path] = None,
    session_title: str = "",
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    allow_latest_fallback: bool = False,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> Optional[str]:
    if session_title:
        by_title = resolve_session_id_by_title(
            session_title,
            opencode_bin=opencode_bin,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
        if by_title:
            return by_title

    if log_file is not None and log_file.exists():
        try:
            ids = find_session_ids(log_file.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            ids = []
        if ids:
            return ids[0]

    if allow_latest_fallback:
        return latest_session_id(
            opencode_bin=opencode_bin,
            cwd=cwd,
            timeout_sec=timeout_sec,
        )
    return None


def resolve_session_id_by_title(
    session_title: str,
    *,
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> Optional[str]:
    try:
        opencode = _resolve_opencode(opencode_bin)
        completed = subprocess.run(
            [opencode, "session", "list"],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        for line in _decode_process_output(completed.stdout).splitlines():
            if session_title not in line:
                continue
            ids = find_session_ids(line)
            if ids:
                return ids[0]
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        pass

    return _resolve_session_id_by_title_from_db(session_title, cwd=cwd)


def latest_session_id(
    *,
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> Optional[str]:
    try:
        opencode = _resolve_opencode(opencode_bin)
        completed = subprocess.run(
            [opencode, "session", "list"],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    ids = find_session_ids(_decode_process_output(completed.stdout))
    return ids[0] if ids else None


def export_session_to_markdown(
    *,
    session_id: str,
    output_file: Path,
    output_dir: Optional[Path] = None,
    opencode_bin: str = "",
    cwd: Optional[Path] = None,
    raw_json_file: Optional[Path] = None,
    timeout_sec: int = DEFAULT_EXPORT_TIMEOUT_SEC,
) -> OpencodeExportResult:
    data = _load_session_export_from_db(session_id)
    if data is not None:
        return _write_session_export(
            session_id=session_id,
            data=data,
            output_file=output_file,
            output_dir=output_dir,
            raw_json_file=raw_json_file,
            message="Markdown transcript exported from OpenCode sqlite storage.",
        )

    try:
        opencode = _resolve_opencode(opencode_bin)
        completed = subprocess.run(
            [opencode, "export", session_id],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
    except FileNotFoundError as exc:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode executable not found: {exc}",
            token_usage=empty_token_usage(),
        )
    except subprocess.TimeoutExpired:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode export timed out after {timeout_sec}s",
            token_usage=empty_token_usage(),
        )
    except OSError as exc:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode export failed to start: {exc}",
            token_usage=empty_token_usage(),
        )

    if completed.returncode != 0:
        detail = (
            _decode_process_output(completed.stderr)
            or _decode_process_output(completed.stdout)
        ).strip()
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode export exited code={completed.returncode}: {detail[:500]}",
            token_usage=empty_token_usage(),
        )

    try:
        data = json.loads(_decode_process_output(completed.stdout))
    except json.JSONDecodeError as exc:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"opencode export JSON parse failed: {exc}",
            token_usage=empty_token_usage(),
        )

    if not isinstance(data, dict):
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message="opencode export returned non-object JSON",
            token_usage=empty_token_usage(),
        )

    return _write_session_export(
        session_id=session_id,
        data=data,
        output_file=output_file,
        output_dir=output_dir,
        raw_json_file=raw_json_file,
        message="Markdown transcript exported from opencode export.",
    )


def append_export_result_to_log(
    log_file: Optional[Path],
    result: OpencodeExportResult,
    *,
    label: str,
) -> None:
    if log_file is None:
        return
    try:
        payload = result.to_dict()
        payload["label"] = label
        payload["log_file"] = str(log_file)
        status_json = log_file.parent / f"{label}_session_export.json"
        status_log = log_file.parent / f"{label}_session_export.log"
        status_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        status_log.write_text(_format_export_status_log(result, label=label), encoding="utf-8")
        with log_file.open("a", encoding="utf-8") as handle:
            if result.ok:
                handle.write(
                    f"\n[opencode export:{label}] session_id={result.session_id} "
                    f"markdown={result.markdown_file} json={result.json_file} "
                    f"sessions={result.node_session_count} "
                    f"tokens={result.token_usage.get('total', 0)}\n"
                )
            else:
                handle.write(
                    f"\n[opencode export:{label}] status={result.status} "
                    f"session_id={result.session_id or '<unknown>'} "
                    f"message={result.message}\n"
                )
    except OSError:
        return


def render_transcript(data: Mapping[str, Any]) -> str:
    info = data.get("info") if isinstance(data.get("info"), Mapping) else {}
    messages = data.get("messages") if isinstance(data.get("messages"), list) else []
    children = data.get("children") if isinstance(data.get("children"), list) else []
    title = info.get("title") or info.get("id") or "OpenCode Session"
    session_id = info.get("id") or ""
    time_block = info.get("time") if isinstance(info.get("time"), Mapping) else {}

    lines: list[str] = [
        f"# {title}",
        "",
        f"**Session ID:** {session_id}",
        "",
    ]
    if info.get("parent_id"):
        lines.extend([f"**Parent Session ID:** {info.get('parent_id')}", ""])
    if info.get("directory"):
        lines.extend([f"**Directory:** {info.get('directory')}", ""])
    lines.extend(
        [
            f"**Created:** {_format_timestamp(time_block.get('created'))}",
            "",
            f"**Updated:** {_format_timestamp(time_block.get('updated'))}",
            "",
            "---",
            "",
        ]
    )

    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        msg_info = msg.get("info") if isinstance(msg.get("info"), Mapping) else {}
        parts = msg.get("parts") if isinstance(msg.get("parts"), list) else []
        lines.append(_format_message(msg_info, parts).rstrip())
        lines.extend(["---", ""])

    if children:
        lines.extend(["# Subagent Sessions", ""])
        for child in children:
            if isinstance(child, Mapping):
                lines.extend(_render_subagent_session(child, heading_level=2))
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_single_session(data: Mapping[str, Any], *, subagent: bool = False) -> str:
    node = _session_without_children(data)
    if subagent:
        return "\n".join(_render_subagent_session(node, heading_level=1)).rstrip() + "\n"
    return render_transcript(node)


def summarize_session_token_usage(data: Mapping[str, Any]) -> dict[str, Any]:
    usages: list[dict[str, Any]] = []

    def visit(node: Mapping[str, Any]) -> None:
        session_usage = _summarize_single_session_token_usage(node)
        session_usage["session_count"] = 1
        usages.append(session_usage)
        children = node.get("children") if isinstance(node.get("children"), list) else []
        for child in children:
            if isinstance(child, Mapping):
                visit(child)

    visit(data)
    return merge_token_usage(usages)


def _write_session_export(
    *,
    session_id: str,
    data: dict[str, Any],
    output_file: Path,
    output_dir: Optional[Path],
    raw_json_file: Optional[Path],
    message: str,
) -> OpencodeExportResult:
    try:
        rendered = render_transcript(data)
        json_text = json.dumps(data, indent=2, ensure_ascii=False)
        output_file = output_file.resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(rendered, encoding="utf-8")

        json_file = raw_json_file.resolve() if raw_json_file is not None else output_file.with_suffix(".json")
        json_file.parent.mkdir(parents=True, exist_ok=True)
        json_file.write_text(json_text, encoding="utf-8")

        export_dir = output_dir.resolve() if output_dir is not None else output_file.parent
        nodes_dir: Optional[Path] = None
        session_tree_file: Optional[Path] = None
        node_count = _node_session_count(data)
        if output_dir is not None:
            export_dir.mkdir(parents=True, exist_ok=True)
            root_md_file = export_dir / "root_full.md"
            root_json_file = export_dir / "root_full.json"
            root_md_file.write_text(rendered, encoding="utf-8")
            root_json_file.write_text(json_text, encoding="utf-8")
            nodes_dir = export_dir / "nodes"
            nodes_dir.mkdir(parents=True, exist_ok=True)
            rows = _write_session_nodes(data, nodes_dir=nodes_dir, base_dir=export_dir)
            session_tree_file = export_dir / "session_tree.tsv"
            _write_session_tree(rows, session_tree_file)
            node_count = len(rows)
    except OSError as exc:
        return OpencodeExportResult(
            session_id=session_id,
            status="error",
            message=f"session export write failed: {exc}",
            token_usage=empty_token_usage(),
        )

    return OpencodeExportResult(
        session_id=session_id,
        markdown_file=output_file,
        json_file=json_file,
        export_dir=export_dir,
        session_tree_file=session_tree_file,
        nodes_dir=nodes_dir,
        status="exported",
        message=message,
        session_updated_at_ms=_session_updated_at_ms(data),
        tree_updated_at_ms=_tree_updated_at_ms(data),
        node_session_count=node_count,
        token_usage=summarize_session_token_usage(data),
    )


def _write_session_nodes(
    data: Mapping[str, Any],
    *,
    nodes_dir: Path,
    base_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order, depth, node in _flatten_session_tree(data):
        info = node.get("info") if isinstance(node.get("info"), Mapping) else {}
        node_session_id = str(info.get("id") or f"unknown_{order}")
        role = "root" if depth == 0 else "subagent"
        basename = _safe_file_part(
            "__".join(
                [
                    f"{order:04d}",
                    f"depth{depth:02d}",
                    role,
                    node_session_id,
                    str(info.get("title") or ""),
                ]
            ),
            limit=180,
        )
        node_json_file = nodes_dir / f"{basename}.json"
        node_md_file = nodes_dir / f"{basename}.md"
        node_data = _session_without_children(node)
        node_json_file.write_text(
            json.dumps(node_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        node_md_file.write_text(
            render_single_session(node, subagent=depth > 0),
            encoding="utf-8",
        )
        single_usage = _summarize_single_session_token_usage(node_data)
        rows.append(
            {
                "order": order,
                "depth": depth,
                "role": role,
                "session_id": node_session_id,
                "parent_id": info.get("parent_id") or "",
                "title": info.get("title") or "",
                "created": _format_timestamp((info.get("time") or {}).get("created")),
                "updated": _format_timestamp((info.get("time") or {}).get("updated")),
                "md_file": str(node_md_file.relative_to(base_dir)),
                "json_file": str(node_json_file.relative_to(base_dir)),
                "token_total": single_usage.get("total", 0),
                "token_input": single_usage.get("input", 0),
                "token_output": single_usage.get("output", 0),
                "token_reasoning": single_usage.get("reasoning", 0),
                "token_cache_read": (single_usage.get("cache") or {}).get("read", 0),
                "token_cache_write": (single_usage.get("cache") or {}).get("write", 0),
                "cost": single_usage.get("cost", 0.0),
            }
        )
    return rows


def _write_session_tree(rows: list[dict[str, Any]], session_tree_file: Path) -> None:
    fields = [
        "order",
        "depth",
        "role",
        "session_id",
        "parent_id",
        "title",
        "created",
        "updated",
        "md_file",
        "json_file",
        "token_total",
        "token_input",
        "token_output",
        "token_reasoning",
        "token_cache_read",
        "token_cache_write",
        "cost",
    ]
    with session_tree_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_session_tree(data: Mapping[str, Any]) -> list[tuple[int, int, Mapping[str, Any]]]:
    out: list[tuple[int, int, Mapping[str, Any]]] = []

    def visit(node: Mapping[str, Any], depth: int) -> None:
        out.append((len(out), depth, node))
        children = node.get("children") if isinstance(node.get("children"), list) else []
        for child in children:
            if isinstance(child, Mapping):
                visit(child, depth + 1)

    visit(data, 0)
    return out


def _node_session_count(data: Mapping[str, Any]) -> int:
    return len(_flatten_session_tree(data))


def _session_without_children(data: Mapping[str, Any]) -> dict[str, Any]:
    node = dict(data)
    node.pop("children", None)
    return node


def _summarize_single_session_token_usage(data: Mapping[str, Any]) -> dict[str, Any]:
    usages: list[dict[str, Any]] = []
    messages = data.get("messages") if isinstance(data.get("messages"), list) else []
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        usage = _message_token_usage(msg)
        if usage.get("supported"):
            usages.append(usage)
    return merge_token_usage(usages)


def _message_token_usage(msg: Mapping[str, Any]) -> dict[str, Any]:
    info = msg.get("info") if isinstance(msg.get("info"), Mapping) else {}
    parts = msg.get("parts") if isinstance(msg.get("parts"), list) else []
    tokens = info.get("tokens") if isinstance(info.get("tokens"), Mapping) else None
    cost = info.get("cost")

    if tokens is None:
        part_usages: list[dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, Mapping):
                continue
            part_tokens = part.get("tokens")
            if isinstance(part_tokens, Mapping):
                part_usages.append(_usage_from_tokens(part_tokens, part.get("cost")))
        usage = merge_token_usage(part_usages)
        if not usage.get("supported"):
            return usage
    else:
        usage = _usage_from_tokens(tokens, cost)

    usage["message_count"] = 1
    model_key = _message_model_key(info)
    if model_key:
        model_usage = dict(usage)
        model_usage["by_model"] = {}
        usage["by_model"] = {model_key: model_usage}
    return usage


def _usage_from_tokens(tokens: Mapping[str, Any], cost: Any = None) -> dict[str, Any]:
    usage = empty_token_usage()
    usage["supported"] = True
    for key in _TOKEN_FIELDS:
        usage[key] = _as_int(tokens.get(key))
    cache = tokens.get("cache") if isinstance(tokens.get("cache"), Mapping) else {}
    usage["cache"]["read"] = _as_int(cache.get("read") or tokens.get("cacheRead"))
    usage["cache"]["write"] = _as_int(cache.get("write") or tokens.get("cacheWrite"))
    if usage["total"] <= 0:
        usage["total"] = (
            usage["input"]
            + usage["output"]
            + usage["reasoning"]
            + usage["cache"]["read"]
            + usage["cache"]["write"]
        )
    usage["cost"] = _as_float(cost)
    return usage


def _message_model_key(info: Mapping[str, Any]) -> str:
    provider = str(info.get("providerID") or "").strip()
    model = str(info.get("modelID") or "").strip()
    if not model:
        model_info = info.get("model")
        if isinstance(model_info, Mapping):
            provider = provider or str(model_info.get("providerID") or "").strip()
            model = str(model_info.get("modelID") or "").strip()
    if provider and model:
        return f"{provider}/{model}"
    return model or provider


def _session_updated_at_ms(data: Mapping[str, Any]) -> Optional[int]:
    info = data.get("info") if isinstance(data.get("info"), Mapping) else {}
    time_block = info.get("time") if isinstance(info.get("time"), Mapping) else {}
    value = time_block.get("updated")
    return value if isinstance(value, int) else None


def _tree_updated_at_ms(data: Mapping[str, Any]) -> Optional[int]:
    values: list[int] = []

    def visit(node: Mapping[str, Any]) -> None:
        updated = _session_updated_at_ms(node)
        if updated is not None:
            values.append(updated)
        children = node.get("children") if isinstance(node.get("children"), list) else []
        for child in children:
            if isinstance(child, Mapping):
                visit(child)

    visit(data)
    return max(values) if values else None


def _render_subagent_session(data: Mapping[str, Any], *, heading_level: int) -> list[str]:
    info = data.get("info") if isinstance(data.get("info"), Mapping) else {}
    messages = data.get("messages") if isinstance(data.get("messages"), list) else []
    children = data.get("children") if isinstance(data.get("children"), list) else []
    title = info.get("title") or info.get("id") or "OpenCode Subagent Session"
    session_id = info.get("id") or ""
    time_block = info.get("time") if isinstance(info.get("time"), Mapping) else {}
    marker = "#" * min(max(heading_level, 1), 6)
    lines: list[str] = [
        f"{marker} Subagent: {title}",
        "",
        f"**Session ID:** {session_id}",
        "",
    ]
    if info.get("parent_id"):
        lines.extend([f"**Parent Session ID:** {info.get('parent_id')}", ""])
    if info.get("directory"):
        lines.extend([f"**Directory:** {info.get('directory')}", ""])
    lines.extend(
        [
            f"**Created:** {_format_timestamp(time_block.get('created'))}",
            "",
            f"**Updated:** {_format_timestamp(time_block.get('updated'))}",
            "",
            "---",
            "",
        ]
    )
    message_heading_level = min(heading_level + 1, 6)
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        msg_info = msg.get("info") if isinstance(msg.get("info"), Mapping) else {}
        parts = msg.get("parts") if isinstance(msg.get("parts"), list) else []
        lines.append(_format_message(msg_info, parts, heading_level=message_heading_level).rstrip())
        lines.extend(["---", ""])

    for child in children:
        if isinstance(child, Mapping):
            lines.extend(_render_subagent_session(child, heading_level=min(heading_level + 1, 6)))
            lines.append("")
    return lines


def _format_message(
    info: Mapping[str, Any],
    parts: Sequence[Any],
    *,
    heading_level: int = 2,
) -> str:
    role = info.get("role")
    marker = "#" * min(max(heading_level, 1), 6)
    chunks: list[str] = [f"{marker} User\n" if role == "user" else _assistant_header(info, heading_level=heading_level)]
    for part in parts:
        if not isinstance(part, Mapping):
            continue
        rendered = _format_part(part)
        if rendered:
            chunks.append(rendered)
    return "\n".join(chunk.rstrip() for chunk in chunks) + "\n"


def _assistant_header(info: Mapping[str, Any], *, heading_level: int = 2) -> str:
    pieces: list[str] = []
    agent = info.get("agent")
    model = info.get("modelID")
    duration = _message_duration(info.get("time") if isinstance(info.get("time"), Mapping) else {})
    marker = "#" * min(max(heading_level, 1), 6)
    if agent:
        pieces.append(_titlecase(str(agent)))
    if model:
        pieces.append(str(model))
    if duration:
        pieces.append(duration)
    if pieces:
        return f"{marker} Assistant ({' - '.join(pieces)})\n"
    return f"{marker} Assistant\n"


def _format_part(part: Mapping[str, Any]) -> str:
    part_type = part.get("type")
    if part_type == "text" and not part.get("synthetic"):
        return str(part.get("text") or "") + "\n"

    if part_type == "reasoning":
        text = str(part.get("text") or "").strip()
        return f"_Thinking:_\n\n{text}\n" if text else ""

    if part_type == "tool":
        tool_name = part.get("tool") or "(unknown)"
        state = part.get("state") if isinstance(part.get("state"), Mapping) else {}
        chunks = [f"**Tool: {tool_name}**\n"]
        if "input" in state and state.get("input") is not None:
            chunks.append("**Input:**\n")
            chunks.append(_fenced(json.dumps(state.get("input"), indent=2, ensure_ascii=False), lang="json"))
        if state.get("status") == "completed" and state.get("output") is not None:
            chunks.append("**Output:**\n")
            chunks.append(_fenced(_stringify_block(state.get("output"))))
        if state.get("status") == "error" and state.get("error") is not None:
            chunks.append("**Error:**\n")
            chunks.append(_fenced(_stringify_block(state.get("error"))))
        return "\n".join(chunk.rstrip() for chunk in chunks) + "\n"

    return ""


def _format_export_status_log(result: OpencodeExportResult, *, label: str) -> str:
    token_usage = result.token_usage or {}
    cache = token_usage.get("cache") if isinstance(token_usage.get("cache"), Mapping) else {}
    return "\n".join(
        [
            f"label={label}",
            f"status={result.status}",
            f"session_id={result.session_id or ''}",
            f"markdown_file={result.markdown_file or ''}",
            f"json_file={result.json_file or ''}",
            f"export_dir={result.export_dir or ''}",
            f"session_tree_file={result.session_tree_file or ''}",
            f"nodes_dir={result.nodes_dir or ''}",
            f"node_session_count={result.node_session_count}",
            f"token_supported={bool(token_usage.get('supported'))}",
            f"token_total={token_usage.get('total', 0)}",
            f"token_input={token_usage.get('input', 0)}",
            f"token_output={token_usage.get('output', 0)}",
            f"token_reasoning={token_usage.get('reasoning', 0)}",
            f"token_cache_read={cache.get('read', 0)}",
            f"token_cache_write={cache.get('write', 0)}",
            f"token_cost={token_usage.get('cost', 0.0)}",
            f"message={result.message}",
            "",
        ]
    )


def _fenced(text: str, lang: str = "") -> str:
    longest = 0
    for match in re.finditer(r"`+", text or ""):
        longest = max(longest, len(match.group(0)))
    fence = "`" * max(3, longest + 1)
    suffix = lang if lang else ""
    return f"{fence}{suffix}\n{text}\n{fence}\n"


def _message_duration(time_block: Mapping[str, Any]) -> str:
    created = time_block.get("created")
    completed = time_block.get("completed")
    if not isinstance(created, (int, float)) or not isinstance(completed, (int, float)):
        return ""
    return f"{(completed - created) / 1000:.1f}s"


def _format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        seconds = value / 1000 if abs(value) > 100000000000 else value
        try:
            return dt.datetime.fromtimestamp(seconds).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        except (OverflowError, OSError, ValueError):
            return str(value)
    return str(value)


def _decode_process_output(data: Any) -> str:
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    return bytes(data).decode("utf-8", errors="replace")


def _stringify_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def _load_session_export_from_db(session_id: str) -> Optional[dict[str, Any]]:
    for db_path in _opencode_db_paths():
        data = _load_session_export_from_db_path(db_path, session_id)
        if data is not None:
            return data
    return None


def _load_session_export_from_db_path(db_path: Path, session_id: str) -> Optional[dict[str, Any]]:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0) as conn:
            return _load_session_export_from_conn(conn, session_id, seen=set())
    except sqlite3.Error:
        return None


def _load_session_export_from_conn(
    conn: sqlite3.Connection,
    session_id: str,
    *,
    seen: set[str],
) -> Optional[dict[str, Any]]:
    if session_id in seen:
        return None
    seen.add(session_id)

    session_cols = _table_columns(conn, "session")
    parent_expr = "parent_id" if "parent_id" in session_cols else "null as parent_id"
    try:
        session = conn.execute(
            "select id, title, directory, version, time_created, time_updated, "
            f"{parent_expr} from session where id = ?",
            (session_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    if session is None:
        return None

    try:
        message_rows = conn.execute(
            "select id, data from message where session_id = ? order by time_created asc, id asc",
            (session_id,),
        ).fetchall()
        part_rows = conn.execute(
            "select message_id, data from part where session_id = ? order by time_created asc, id asc",
            (session_id,),
        ).fetchall()
    except sqlite3.Error:
        return None

    parts_by_message: dict[str, list[dict[str, Any]]] = {}
    for message_id, part_text in part_rows:
        part_data = _loads_json_object(part_text)
        if part_data is not None:
            parts_by_message.setdefault(str(message_id), []).append(part_data)

    messages: list[dict[str, Any]] = []
    for message_id, message_text in message_rows:
        message_info = _loads_json_object(message_text)
        if message_info is None:
            continue
        message_info.setdefault("id", str(message_id))
        messages.append(
            {
                "info": message_info,
                "parts": parts_by_message.get(str(message_id), []),
            }
        )

    sid, title, directory, version, created, updated, parent_id = session
    data: dict[str, Any] = {
        "info": {
            "id": sid,
            "parent_id": parent_id,
            "title": title,
            "directory": directory,
            "version": version,
            "time": {"created": created, "updated": updated},
        },
        "messages": messages,
    }

    child_rows: list[Any] = []
    if "parent_id" in session_cols:
        try:
            child_rows = conn.execute(
                "select id from session where parent_id = ? order by time_created asc, id asc",
                (session_id,),
            ).fetchall()
        except sqlite3.Error:
            child_rows = []

    children: list[dict[str, Any]] = []
    for (child_id,) in child_rows:
        child_data = _load_session_export_from_conn(conn, str(child_id), seen=seen)
        if child_data is not None:
            children.append(child_data)
    if children:
        data["children"] = children
    return data


def _resolve_session_id_by_title_from_db(session_title: str, *, cwd: Optional[Path]) -> Optional[str]:
    for db_path in _opencode_db_paths():
        session_id = _resolve_session_id_by_title_from_db_path(db_path, session_title, cwd=cwd)
        if session_id:
            return session_id
    return None


def _resolve_session_id_by_title_from_db_path(
    db_path: Path,
    session_title: str,
    *,
    cwd: Optional[Path],
) -> Optional[str]:
    if not db_path.exists():
        return None
    cwd_resolved = str(cwd.resolve()) if cwd else ""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0) as conn:
            rows = conn.execute(
                "select id, directory from session where title = ? order by time_created desc limit 20",
                (session_title,),
            ).fetchall()
    except sqlite3.Error:
        return None

    if not rows:
        return None
    if cwd_resolved:
        for session_id, directory in rows:
            try:
                if str(Path(directory).resolve()) == cwd_resolved:
                    return str(session_id)
            except (TypeError, OSError):
                continue
    return str(rows[0][0])


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"pragma table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()


def _loads_json_object(text: Any) -> Optional[dict[str, Any]]:
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _opencode_db_paths() -> list[Path]:
    explicit = os.environ.get("OPENCODE_DB")
    if explicit:
        return [Path(explicit).expanduser()]

    candidates: list[Path] = []
    data_home = os.environ.get("XDG_DATA_HOME")
    if data_home:
        candidates.append(Path(data_home).expanduser() / "opencode" / "opencode.db")
    candidates.append(Path.home() / ".local" / "share" / "opencode" / "opencode.db")

    out: list[Path] = []
    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _resolve_opencode(opencode_bin: str = "") -> str:
    if opencode_bin:
        candidate = Path(opencode_bin).expanduser()
        if candidate.exists():
            return str(candidate)
        found = shutil.which(opencode_bin)
        if found:
            return found
        raise FileNotFoundError(opencode_bin)
    found = shutil.which("opencode")
    if not found:
        raise FileNotFoundError("opencode")
    return found


def _titlecase(value: str) -> str:
    out: list[str] = []
    for part in re.split(r"([\s_-]+)", value):
        if not part or re.fullmatch(r"[\s_-]+", part):
            out.append(part)
        else:
            out.append(part[0].upper() + part[1:])
    return "".join(out)


def _safe_title_part(value: str, *, limit: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", value or "unknown").strip("_")
    return (cleaned or "unknown")[:limit]


def _safe_file_part(value: str, *, limit: int) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown").strip("_")
    return (cleaned or "unknown")[:limit]


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
