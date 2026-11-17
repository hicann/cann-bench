#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------



"""Build the standalone optimization trajectory report for Ascend Arena runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict

REPORT_NAME = "optimization_trajectory.html"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _round_report(run_dir: Path) -> Dict[str, Any]:
    preferred = run_dir / f"{run_dir.name}.json"
    candidates = [preferred] if preferred.is_file() else []
    candidates.extend(
        path for path in sorted(run_dir.glob("*.json"))
        if path != preferred and path.name != "comparison.json"
    )
    for path in candidates:
        report = _load_json(path)
        if isinstance(report.get("source_diff"), dict) and isinstance(report.get("cases"), list):
            report["_run_name"] = str(report.get("eval_code") or run_dir.name)
            return report
    return {}


def _source_signature(report: Dict[str, Any]) -> str:
    source_diff = report.get("source_diff", {})
    diff_text = str(source_diff.get("unified_diff") or "")
    if diff_text:
        state = diff_text
    else:
        state = json.dumps(source_diff.get("changed_files", []), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(state.encode("utf-8")).hexdigest()


def _changed_file_count(report: Dict[str, Any]) -> int:
    source_diff = report.get("source_diff", {})
    changed_files = source_diff.get("changed_files")
    if isinstance(changed_files, list):
        return sum(
            isinstance(item, dict) and Path(str(item.get("path") or "")).name != ".source"
            for item in changed_files
        )
    try:
        return int(source_diff.get("changed_file_count", 0))
    except (TypeError, ValueError):
        return 0


def _positive_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def _initialized_baselines(reports_root: Path) -> Dict[str, Dict[str, Any]]:
    from kernel_eval.optimize import _duration, _timing_scope, _measurement_protocol

    status = _load_json(reports_root / "init_results" / "status.json")
    payload = _load_json(reports_root / "init_results" / "result.json")
    if status.get("success") is not True:
        return {}
    baselines = {}
    for case_id, records in payload.items():
        duration = _duration(records)
        if duration:
            baselines[str(case_id)] = {
                "case_id": str(case_id),
                "baseline_us": duration,
                "timing_scope": _timing_scope(records),
                "measurement_protocol": _measurement_protocol(records),
                "run_name": "init_results",
            }
    return baselines


def _case_rows(report: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for item in report.get("cases", []):
        if (not isinstance(item, dict) or not item.get("case_id")
                or item.get("timing_comparable") is False):
            continue
        rows.append({
            "case_id": str(item["case_id"]),
            "baseline_us": _positive_float(item.get("baseline_us")),
            "candidate_us": _positive_float(item.get("candidate_us")),
            "timing_scope": item.get("timing_scope", "kernel"),
            "measurement_protocol": item.get("measurement_protocol"),
        })
    return rows


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _fallback_file_diffs(source_diff: Dict[str, Any]) -> list[Dict[str, str]]:
    diff_text = str(source_diff.get("unified_diff") or "")
    changed_files = source_diff.get("changed_files", [])
    starts = []
    for item in changed_files if isinstance(changed_files, list) else []:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = str(item["path"])
        if Path(path).name == ".source":
            continue
        positions = [
            position for position in (
                diff_text.find(f"--- a/{path}\n"),
                diff_text.find(f"diff --git a/{path} b/{path}\n"),
            )
            if position >= 0
        ]
        starts.append((min(positions) if positions else len(diff_text), item))

    starts.sort(key=lambda value: value[0])
    file_diffs = []
    for index, (start, item) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(diff_text)
        block = diff_text[start:end].rstrip() if start < len(diff_text) else ""
        file_diffs.append({
            "path": str(item["path"]),
            "status": str(item.get("status") or "modified"),
            "diff": block or "# Diff content unavailable.",
        })
    return file_diffs


def _file_diffs(report: Dict[str, Any]) -> list[Dict[str, str]]:
    source_diff = report.get("source_diff", {})
    embedded = source_diff.get("file_diffs", [])
    if isinstance(embedded, list) and embedded:
        return [
            {
                "path": str(item.get("path") or "unknown"),
                "status": str(item.get("status") or "modified"),
                "diff": str(item.get("diff") or "# Diff content unavailable."),
            }
            for item in embedded
            if isinstance(item, dict) and Path(str(item.get("path") or "")).name != ".source"
        ]
    return _fallback_file_diffs(source_diff)


def _trace_events(reports_root: Path) -> list[Dict[str, Any]]:
    """Keep every measured invocation, independently of code-change rounds."""
    events = []
    for path in (reports_root / "timing.jsonl", reports_root.parent / "artifact/timing.jsonl"):
        if not path.is_file():
            continue
        records = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A running writer may leave a partial last line.
            if isinstance(row, dict) and row.get("id") and row.get("kind") in {"agent", "evaluation"}:
                records[row["id"]] = row
        for row in records.values():
            row = dict(row, status="incomplete", speedup=None)
            if row.get("kind") == "evaluation" and row.get("event") == "end":
                name = str(row.get("run_name") or "")
                if name and Path(name).name == name:
                    run = reports_root / name
                    status_paths = ([run / "status.json"] if row.get("phase") == "init" else
                                    [run / "results/status.json"])
                    if row.get("phase") == "comparison":
                        status_paths.insert(0, run / "baseline_results/status.json")
                    statuses = [_load_json(p) for p in status_paths]
                    if any(s.get("build_success") is False and not s.get("interrupted") for s in statuses):
                        row["status"] = "compile_failed"
                    elif row.get("returncode") == 0 and all(s.get("success") is True for s in statuses):
                        row["status"] = "success"
                        report = _round_report(run)
                        row["speedup"] = report.get("summary", {}).get("average_speedup")
                    elif any(s.get("build_success") is True for s in statuses):
                        row["status"] = "evaluation_failed"
            events.append(row)
    return sorted(events, key=lambda row: str(row.get("started_at", "")))


def build_trajectory_data(reports_root: Path) -> Dict[str, Any]:
    """Collect valid code-change rounds and their rolling baseline comparisons."""
    reports_root = Path(reports_root)
    reports = []
    if reports_root.is_dir():
        for run_dir in reports_root.iterdir():
            if run_dir.is_dir() and not run_dir.name.startswith("."):
                report = _round_report(run_dir)
                if report:
                    reports.append(report)
    reports.sort(key=lambda item: (str(item.get("timestamp") or ""), str(item.get("_run_name") or "")))

    baseline_by_case = _initialized_baselines(reports_root)
    has_initialized_baseline = bool(baseline_by_case)
    rounds = []
    previous_signature = ""
    for report in reports:
        run_name = str(report["_run_name"])
        for case in _case_rows(report):
            baseline_us = case["baseline_us"]
            previous = baseline_by_case.get(case["case_id"])
            if not has_initialized_baseline and baseline_us and (
                previous is None or baseline_us < previous["baseline_us"]
            ):
                baseline_by_case[case["case_id"]] = {
                    "case_id": case["case_id"],
                    "baseline_us": baseline_us,
                    "timing_scope": case["timing_scope"],
                    "measurement_protocol": case.get("measurement_protocol"),
                    "run_name": run_name,
                }

        signature = _source_signature(report)
        changed = _changed_file_count(report) > 0
        same_source = bool(rounds) and signature == previous_signature
        previous_signature = signature
        if not changed:
            continue

        candidate_by_case = {
            item["case_id"]: item["candidate_us"] for item in _case_rows(report)
            if item["timing_scope"] == baseline_by_case.get(item["case_id"], {}).get("timing_scope", "kernel")
            and item.get("measurement_protocol") == baseline_by_case.get(item["case_id"], {}).get("measurement_protocol")
        }
        case_ids = sorted(set(baseline_by_case) | set(candidate_by_case), key=_natural_key)
        cases = [
            {
                "index": index,
                "case_id": case_id,
                "baseline_us": baseline_by_case.get(case_id, {}).get("baseline_us") or None,
                "candidate_us": candidate_by_case.get(case_id) or None,
            }
            for index, case_id in enumerate(case_ids, start=1)
        ]
        succeeded = (
            report.get("candidate_status", {}).get("success") is True
            and report.get("baseline_status", {}).get("success") is True
        )
        speedups = [
            item["baseline_us"] / item["candidate_us"]
            for item in cases
            if item["baseline_us"] and item["candidate_us"]
        ]
        round_data = {
            "round": len(rounds) + 1,
            "run_name": run_name,
            "timestamp": str(report.get("timestamp") or ""),
            "success": succeeded,
            "speedup": mean(speedups) if succeeded and speedups else None,
            "cases": cases,
            "file_diffs": _file_diffs(report),
            "measurement": report.get("measurement", {}),
        }
        if same_source:
            # A later scored evaluation completes the same source round.
            if rounds[-1]["speedup"] is None and round_data["speedup"] is not None:
                round_data["round"] = rounds[-1]["round"]
                rounds[-1] = round_data
        else:
            rounds.append(round_data)

    baselines = sorted(baseline_by_case.values(), key=lambda item: _natural_key(item["case_id"]))
    operator = next((str(item.get("operator")) for item in reversed(reports) if item.get("operator")), "")
    return {
        "operator": operator,
        "generated_from_runs": len(reports),
        "baselines": [
            {"index": index, **item} for index, item in enumerate(baselines, start=1)
        ],
        "rounds": rounds,
        "trace": _trace_events(reports_root),
    }


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def render_optimization_trajectory(data: Dict[str, Any]) -> str:
    payload = _json_for_script(data)
    return _HTML_TEMPLATE.replace("__TRAJECTORY_DATA__", payload)


def write_optimization_trajectory_report(reports_root: Path) -> Path:
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)
    output = reports_root / REPORT_NAME
    output.write_text(render_optimization_trajectory(build_trajectory_data(reports_root)), encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the kernel optimization trajectory HTML report.")
    parser.add_argument("reports_root", type=Path, help="kernel_eval directory containing evaluation runs")
    args = parser.parse_args(argv)
    print(write_optimization_trajectory_report(args.reports_root))
    return 0


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kernel Optimization Trajectory</title>
  <style>
    :root {
      color-scheme: light;
      --page: #f4f6f5;
      --surface: #ffffff;
      --surface-subtle: #f8faf9;
      --ink: #18201e;
      --muted: #5d6966;
      --line: #d6ddda;
      --grid: #e7ecea;
      --primary: #007f73;
      --primary-dark: #005f57;
      --before: #d98b20;
      --after: #007f73;
      --failure: #c23b4a;
      --focus: #0b63ce;
      --code: #171b1a;
      --code-ink: #e8eeec;
      --added: #173c2d;
      --deleted: #482328;
      --hunk: #253a45;
      --radius: 6px;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin: 0; background: var(--page); color: var(--ink); font-size: 16px; line-height: 1.55; }
    button, table { font: inherit; }
    .skip-link { position: fixed; left: 16px; top: 8px; z-index: 20; transform: translateY(-160%); padding: 10px 14px; background: var(--ink); color: white; border-radius: 4px; }
    .skip-link:focus { transform: translateY(0); }
    header { border-bottom: 1px solid var(--line); background: var(--surface); }
    .header-inner, main { width: min(1280px, calc(100% - 40px)); margin: 0 auto; }
    .header-inner { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; padding: 24px 0 20px; }
    .eyebrow { margin: 0 0 4px; color: var(--primary-dark); font-size: 0.78rem; font-weight: 700; text-transform: uppercase; }
    h1, h2, h3, p { margin-top: 0; }
    h1 { margin-bottom: 0; font-size: clamp(1.65rem, 3vw, 2.35rem); line-height: 1.15; letter-spacing: 0; }
    h2 { margin-bottom: 6px; font-size: 1.28rem; letter-spacing: 0; }
    h3 { margin-bottom: 8px; font-size: 1rem; letter-spacing: 0; }
    .header-meta { color: var(--muted); text-align: right; }
    .header-meta strong { display: block; color: var(--ink); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92rem; }
    main { padding: 28px 0 56px; }
    section.band { padding: 28px 0; border-bottom: 1px solid var(--line); }
    section.band:first-child { padding-top: 0; }
    .section-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 24px; margin-bottom: 18px; }
    .section-head p { max-width: 680px; margin-bottom: 0; color: var(--muted); }
    .table-shell { overflow-x: auto; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { padding: 11px 14px; border-bottom: 1px solid var(--grid); text-align: left; vertical-align: top; }
    th { background: var(--surface-subtle); color: #485451; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; }
    tbody tr:last-child td { border-bottom: 0; }
    td.number, th.number { text-align: right; }
    td.mono { overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; }
    .summary-strip { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
    .metric { min-width: 0; padding: 18px 20px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric span { display: block; color: var(--muted); font-size: 0.8rem; }
    .metric strong { display: block; margin-top: 2px; font-size: 1.45rem; font-variant-numeric: tabular-nums; }
    .chart-shell { position: relative; min-height: 360px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); }
    svg.chart { display: block; width: 100%; height: 360px; overflow: visible; }
    .axis-label, .tick-label, .point-label, .bar-label { fill: var(--muted); font-family: ui-sans-serif, system-ui, sans-serif; }
    .axis-label { font-size: 12px; font-weight: 700; }
    .tick-label { font-size: 11px; }
    .point-label, .bar-label { fill: var(--ink); font-size: 11px; font-weight: 700; }
    .grid-line { stroke: var(--grid); stroke-width: 1; }
    .axis-line { stroke: #aab5b1; stroke-width: 1; }
    .trend-line { fill: none; stroke: var(--primary); stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }
    .point-hit { cursor: pointer; fill: transparent; }
    .point-visible { fill: var(--surface); stroke: var(--primary); stroke-width: 3; pointer-events: none; transition: fill 180ms ease-out, stroke 180ms ease-out; }
    .point.failed .point-visible { fill: none; stroke: var(--failure); }
    .point.selected .point-visible { fill: var(--primary); stroke: var(--primary-dark); }
    .point.failed.selected .point-visible { fill: none; stroke: #8e2633; }
    .point:focus { outline: none; }
    .point:focus .point-visible { stroke: var(--focus); stroke-width: 5; }
    .chart-tooltip { position: absolute; z-index: 4; min-width: 150px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 4px; background: var(--ink); color: white; box-shadow: 0 6px 20px rgba(24, 32, 30, 0.18); font-size: 0.8rem; pointer-events: none; transform: translate(-50%, calc(-100% - 12px)); }
    .chart-tooltip[hidden] { display: none; }
    .case-tooltip { min-width: 220px; }
    .trace-row { display:flex; align-items:center; gap:12px; margin:14px 0; }
    .trace-row > strong { flex:0 0 90px; font-size:13px; }
    .trace-track { position:relative; height:32px; flex:1; background:#eef1f4; border-radius:4px; overflow:hidden; }
    .trace-block { position:absolute; top:5px; height:22px; min-width:3px; border:1px solid white; padding:0; cursor:crosshair; border-radius:3px; }
    .trace-block:hover,.trace-block:focus { outline:2px solid #243b53; outline-offset:-2px; }
    .trace-axis { font-size:12px; color:var(--muted); overflow-wrap:anywhere; }
    #trace-tooltip { position:fixed; z-index:100; width:max-content; max-width:calc(100vw - 16px); padding:12px; border-radius:6px; background:#202c39; color:white; white-space:pre-wrap; overflow-wrap:anywhere; font:13px/1.7 monospace; pointer-events:none; box-shadow:0 4px 18px #0003; }
    .tooltip-title { margin-bottom: 7px; color: #ffffff; font-weight: 700; overflow-wrap: anywhere; }
    .tooltip-row { display: flex; align-items: baseline; justify-content: space-between; gap: 20px; color: #d8e1de; }
    .tooltip-row strong { color: #ffffff; font-variant-numeric: tabular-nums; }
    .legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 0; padding: 0; list-style: none; color: var(--muted); font-size: 0.84rem; }
    .legend li { display: inline-flex; align-items: center; gap: 7px; }
    .swatch { width: 20px; height: 8px; border-radius: 2px; background: var(--after); }
    .swatch.before { background: var(--before); }
    .swatch.failure { width: 10px; height: 10px; background: none; color: var(--failure); font-size: 18px; line-height: 10px; }
    .round-heading { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-bottom: 18px; }
    .round-heading h2 { margin: 0; }
    .badge { display: inline-flex; align-items: center; min-height: 28px; padding: 3px 9px; border: 1px solid #9ac8c1; border-radius: 999px; background: #e7f5f2; color: var(--primary-dark); font-size: 0.78rem; font-weight: 700; }
    .badge.failure { border-color: #e2a3ab; background: #fcecef; color: #922936; }
    .round-id { flex: 1 1 100%; margin: 0; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.78rem; overflow-wrap: anywhere; }
    .bar-before { fill: var(--before); }
    .bar-after { fill: var(--after); }
    .bar-mark { cursor: pointer; transition: opacity 180ms ease-out; }
    .bar-mark:hover { opacity: 0.82; }
    .bar-mark:focus { outline: none; stroke: var(--focus); stroke-width: 3; }
    .diff-list { display: grid; gap: 16px; }
    .diff-file { overflow: hidden; border: 1px solid #303735; border-radius: var(--radius); background: var(--code); }
    .diff-file header { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 11px 14px; border-bottom-color: #303735; background: #202624; color: var(--code-ink); }
    .diff-file h3 { margin: 0; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.86rem; overflow-wrap: anywhere; }
    .diff-status { color: #b9c6c2; font-size: 0.72rem; text-transform: uppercase; }
    pre { max-height: 520px; margin: 0; overflow: auto; padding: 10px 0; color: var(--code-ink); font: 0.78rem/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; tab-size: 4; }
    code { display: block; min-width: max-content; }
    .diff-line { display: block; min-height: 1.55em; padding: 0 14px; white-space: pre; }
    .diff-line.add { background: var(--added); color: #bdebd0; }
    .diff-line.del { background: var(--deleted); color: #ffc6cc; }
    .diff-line.hunk { background: var(--hunk); color: #b9dff2; }
    .diff-line.meta { color: #aab7b3; }
    .empty { padding: 36px 20px; border: 1px dashed #aab5b1; border-radius: var(--radius); background: var(--surface); color: var(--muted); text-align: center; }
    .visually-hidden { position: absolute; width: 1px; height: 1px; margin: -1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
    :focus-visible { outline: 3px solid var(--focus); outline-offset: 3px; }

    @media (max-width: 720px) {
      .header-inner, main { width: min(100% - 24px, 1280px); }
      .header-inner, .section-head { align-items: flex-start; flex-direction: column; gap: 10px; }
      .header-meta { text-align: left; }
      .summary-strip { grid-template-columns: 1fr; }
      .metric { border-right: 0; border-bottom: 1px solid var(--line); }
      .metric:last-child { border-bottom: 0; }
      th, td { padding: 10px; }
      svg.chart { height: 330px; }
      .chart-shell { min-height: 330px; }
    }

    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; }
    }

    @media print {
      body { background: white; }
      .header-inner, main { width: 100%; }
      .chart-tooltip { display: none !important; }
      section.band { break-inside: avoid; }
      pre { max-height: none; }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#main">跳到主要内容</a>
  <header>
    <div class="header-inner">
      <div>
        <p class="eyebrow">CANN Bench · Performance</p>
        <h1>Kernel Optimization Trajectory</h1>
      </div>
      <div class="header-meta"><span>Operator</span><strong id="operator-name">—</strong></div>
    </div>
  </header>

  <main id="main">
    <section class="band" aria-labelledby="trace-title">
      <div class="section-head"><div><h2 id="trace-title">执行链路追踪</h2>
        <p>悬停或聚焦色块可查看起止时间、时长和评测结果。</p></div></div>
      <div id="execution-trace"></div>
      <div id="trace-tooltip" role="tooltip" hidden></div>
    </section>
    <section class="band" aria-labelledby="baseline-title">
      <div class="section-head">
        <div><h2 id="baseline-title">Case Baseline</h2><p>截至最新评测，每个 case 观测到的最小时延。</p></div>
      </div>
      <div id="baseline-table"></div>
    </section>

    <section class="band" aria-labelledby="trajectory-title">
      <div class="section-head">
        <div><h2 id="trajectory-title">Optimization Trajectory</h2><p>整体加速比采用逐 case 加速比的算术平均；未采集到的性能显示为 N/A。</p></div>
        <ul class="legend" aria-label="图例"><li><span class="swatch"></span>整体加速比</li><li><span class="swatch failure">×</span>失败轮次（0x）</li></ul>
      </div>
      <div class="summary-strip" id="summary-strip"></div>
      <div class="chart-shell" style="margin-top:16px">
        <svg id="trajectory-chart" class="chart" role="img" aria-labelledby="trajectory-title trajectory-summary"></svg>
        <div id="trajectory-tooltip" class="chart-tooltip" role="tooltip" hidden></div>
      </div>
      <p id="trajectory-summary" class="visually-hidden"></p>
      <div id="trajectory-table" class="visually-hidden"></div>
    </section>

    <section class="band" id="round-detail" aria-labelledby="round-title">
      <div id="round-heading" class="round-heading"></div>
      <div class="section-head">
        <div><h2 id="case-chart-title">Case Latency</h2><p>逐 case rolling baseline 与当前优化实现时延，单位为微秒。</p></div>
        <ul class="legend" aria-label="柱状图图例"><li><span class="swatch before"></span>优化前</li><li><span class="swatch"></span>优化后</li></ul>
      </div>
      <div id="case-chart-wrap" class="chart-shell">
        <svg id="case-chart" class="chart" role="img" aria-labelledby="case-chart-title case-chart-summary"></svg>
        <div id="case-tooltip" class="chart-tooltip case-tooltip" role="tooltip" hidden></div>
      </div>
      <p id="case-chart-summary" class="visually-hidden"></p>
    </section>

    <section class="band" aria-labelledby="diff-title">
      <div class="section-head"><div><h2 id="diff-title">Code Diff</h2><p id="diff-count"></p></div></div>
      <div id="diff-list" class="diff-list"></div>
    </section>
  </main>

  <script id="trajectory-data" type="application/json">__TRAJECTORY_DATA__</script>
  <script>
    (() => {
      const data = JSON.parse(document.getElementById('trajectory-data').textContent);
      const rounds = data.rounds || [];
      function renderTrace() {
        const host = document.getElementById('execution-trace');
        const tooltip = document.getElementById('trace-tooltip');
        const items = (data.trace || []).filter(x => Number.isFinite(Date.parse(x.started_at)));
        if (!items.length) { host.textContent = '暂无计时记录；历史评测未记录的时间不会推算。'; return; }
        const start = Math.min(...items.map(x => Date.parse(x.started_at)));
        const end = Math.max(...items.map(x => Date.parse(x.ended_at || x.started_at)));
        const span = Math.max(1, end - start);
        const evaluations = items.filter(item => item.kind === 'evaluation').map(item =>
          [Date.parse(item.started_at), Date.parse(item.ended_at || item.started_at)]).sort((a, b) => a[0] - b[0]);
        const merge = intervals => intervals.reduce((result, interval) => {
          const last = result[result.length - 1];
          if (last && interval[0] <= last[1]) last[1] = Math.max(last[1], interval[1]);
          else result.push([...interval]);
          return result;
        }, []);
        const evaluationIntervals = merge(evaluations);
        const segments = items.flatMap(item => {
          if (item.kind !== 'agent' || !item.ended_at) return [item];
          let cursor = Date.parse(item.started_at);
          const finish = Date.parse(item.ended_at), parts = [];
          const append = (a, b) => { if (b > a) parts.push({...item,
            started_at: new Date(a).toISOString(), ended_at: new Date(b).toISOString(), duration_seconds: (b-a)/1000}); };
          for (const [a, b] of evaluationIntervals) {
            if (b <= cursor || a >= finish) continue;
            append(cursor, Math.min(a, finish)); cursor = Math.min(finish, Math.max(cursor, b));
          }
          append(cursor, finish);
          return parts;
        });
        const colors = {agent:'#9270bd', compile_failed:'#d64a50', evaluation_failed:'#dcaa28', success:'#38a878', incomplete:'#8a939f'};
        const states = {compile_failed:'编译失败', evaluation_failed:'评测失败', success:'评测成功', incomplete:'运行中或记录不完整'};
        for (const [kind, label] of [['evaluation', '执行评测'], ['agent', 'Agent 优化']]) {
          const row = document.createElement('div');
          row.className = 'trace-row';
          const name = document.createElement('strong'); name.textContent = label;
          const track = document.createElement('div'); track.className = 'trace-track';
          for (const item of segments.filter(x => x.kind === kind)) {
            const bar = document.createElement('button');
            bar.type = 'button'; bar.className = 'trace-block';
            bar.dataset.status = kind === 'agent' ? 'agent' : item.status;
            bar.style.background = colors[bar.dataset.status] || colors.incomplete;
            bar.style.left = `min(${(Date.parse(item.started_at) - start) / span * 100}%, calc(100% - 3px))`;
            bar.style.width = `${Math.max(0.3, (Date.parse(item.ended_at || item.started_at) - Date.parse(item.started_at)) / span * 100)}%`;
            let details = `开始：${item.started_at}\n结束：${item.ended_at || 'N/A'}\n时长：${Number.isFinite(item.duration_seconds) ? item.duration_seconds.toFixed(3) + ' 秒' : 'N/A'}`;
            if (kind === 'evaluation') details += `\n阶段：${item.phase || 'N/A'}\n状态：${states[item.status]}\n加速比：${item.status === 'success' && Number.isFinite(item.speedup) && item.speedup > 0 ? item.speedup.toFixed(4) + 'x' : 'N/A'}`;
            bar.setAttribute('aria-label', details);
            const show = event => {
              tooltip.textContent = details; tooltip.hidden = false;
              const rect = bar.getBoundingClientRect();
              const x = event.clientX ?? rect.left;
              const y = event.clientY ?? rect.bottom;
              tooltip.style.left = `${Math.max(8, Math.min(x + 12, window.innerWidth - tooltip.offsetWidth - 8))}px`;
              tooltip.style.top = `${Math.max(8, Math.min(y + 12, window.innerHeight - tooltip.offsetHeight - 8))}px`;
            };
            bar.addEventListener('mousemove', show); bar.addEventListener('focus', show);
            bar.addEventListener('mouseleave', () => tooltip.hidden = true);
            bar.addEventListener('blur', () => tooltip.hidden = true);
            track.appendChild(bar);
          }
          row.append(name, track); host.appendChild(row);
        }
        const axis = document.createElement('p');
        axis.className = 'trace-axis';
        const totals = ['agent', 'evaluation'].map(kind => merge(segments.filter(item => item.kind === kind && item.ended_at)
          .map(item => [Date.parse(item.started_at), Date.parse(item.ended_at)]).sort((a,b) => a[0]-b[0]))
          .reduce((sum, [a,b]) => sum + Math.max(0,b-a)/1000, 0));
        const total = (end - start) / 1000;
        const percent = value => total > 0 ? (value / total * 100).toFixed(2) : '0.00';
        axis.textContent = `${new Date(start).toLocaleString()} → ${new Date(end).toLocaleString()} · 总跨度 ${total.toFixed(3)} 秒 · Agent 优化 ${totals[0].toFixed(3)} 秒（${percent(totals[0])}%） · 执行评测 ${totals[1].toFixed(3)} 秒（${percent(totals[1])}%）`;
        host.appendChild(axis);
      }
      let selectedIndex = Math.max(0, rounds.length - 1);
      const number = new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 4 });
      const valid = value => Number.isFinite(Number(value)) && Number(value) > 0;
      const latency = value => valid(value) ? `${number.format(value)} μs` : 'N/A';
      const speedup = value => value === 0 || valid(value) ? `${number.format(value)}x` : 'N/A';
      const roundSpeedup = item => item.success ? item.speedup : 0;
      const el = id => document.getElementById(id);
      const svgNode = (name, attrs = {}) => {
        const node = document.createElementNS('http://www.w3.org/2000/svg', name);
        Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
        return node;
      };
      const appendText = (parent, text, attrs = {}) => {
        const node = svgNode('text', attrs);
        node.textContent = text;
        parent.appendChild(node);
        return node;
      };

      el('operator-name').textContent = data.operator || 'Unknown';

      function renderBaselineTable() {
        const rows = data.baselines || [];
        if (!rows.length) {
          el('baseline-table').innerHTML = '<div class="empty">暂无 baseline 性能数据</div>';
          return;
        }
        const shell = document.createElement('div');
        shell.className = 'table-shell';
        const table = document.createElement('table');
        table.innerHTML = '<thead><tr><th class="number">#</th><th>Case ID</th><th class="number">Baseline</th></tr></thead>';
        const tbody = document.createElement('tbody');
        rows.forEach(row => {
          const tr = document.createElement('tr');
          const values = [row.index, row.case_id, latency(row.baseline_us)];
          values.forEach((value, index) => {
            const td = document.createElement('td');
            td.textContent = value;
            if (index === 0 || index === 2) td.className = 'number';
            if (index === 1) td.classList.add('mono');
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        shell.appendChild(table);
        el('baseline-table').replaceChildren(shell);
      }

      function renderSummary() {
        const successful = rounds.filter(item => item.success && valid(item.speedup));
        const best = successful.reduce((value, item) => Math.max(value, Number(item.speedup)), null);
        const metrics = [
          ['有效优化轮次', rounds.length],
          ['成功轮次', successful.length],
          ['最佳整体加速比', speedup(best)],
        ];
        el('summary-strip').replaceChildren(...metrics.map(([label, value]) => {
          const item = document.createElement('div');
          item.className = 'metric';
          const caption = document.createElement('span');
          const strong = document.createElement('strong');
          caption.textContent = label;
          strong.textContent = value;
          item.append(caption, strong);
          return item;
        }));
      }

      function lineChart() {
        const svg = el('trajectory-chart');
        svg.replaceChildren();
        if (!rounds.length) {
          svg.setAttribute('viewBox', '0 0 800 360');
          appendText(svg, '暂无有效优化轮次', { x: 400, y: 180, 'text-anchor': 'middle', class: 'tick-label' });
          el('trajectory-summary').textContent = '尚无候选评测轮次；工作区准备或基线采集不代表优化失败。';
          return;
        }
        const width = Math.max(320, svg.clientWidth || 800);
        const height = svg.clientHeight || 360;
        const margin = { top: 40, right: 28, bottom: 55, left: 66 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        const measured = rounds.filter(item => valid(item.speedup));
        const maxValue = Math.max(1, ...measured.map(item => Number(item.speedup)));
        const ceiling = Math.ceil(maxValue * 4) / 4;
        const x = index => margin.left + (rounds.length === 1 ? innerWidth / 2 : index * innerWidth / (rounds.length - 1));
        const y = value => margin.top + innerHeight - value / ceiling * innerHeight;
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

        for (let tick = 0; tick <= 4; tick += 1) {
          const value = ceiling * tick / 4;
          const py = y(value);
          svg.appendChild(svgNode('line', { x1: margin.left, y1: py, x2: width - margin.right, y2: py, class: 'grid-line' }));
          appendText(svg, number.format(value), { x: margin.left - 12, y: py + 4, 'text-anchor': 'end', class: 'tick-label' });
        }
        svg.appendChild(svgNode('line', { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: 'axis-line' }));
        svg.appendChild(svgNode('line', { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: 'axis-line' }));
        appendText(svg, '整体加速比', { x: 18, y: margin.top + innerHeight / 2, transform: `rotate(-90 18 ${margin.top + innerHeight / 2})`, 'text-anchor': 'middle', class: 'axis-label' });
        appendText(svg, '优化轮次', { x: margin.left + innerWidth / 2, y: height - 12, 'text-anchor': 'middle', class: 'axis-label' });

        let drawing = false;
        const path = rounds.map((item, index) => {
          const value = roundSpeedup(item);
          if (!valid(value) && value !== 0) { drawing = false; return ''; }
          const command = drawing ? 'L' : 'M'; drawing = true;
          return `${command} ${x(index)} ${y(Number(value))}`;
        }).filter(Boolean).join(' ');
        svg.appendChild(svgNode('path', { d: path, class: 'trend-line' }));
        rounds.forEach((item, index) => {
          const px = x(index);
          const py = valid(item.speedup) ? y(Number(item.speedup)) : y(0);
          const group = svgNode('g', {
            class: `point${item.success ? '' : ' failed'}${index === selectedIndex ? ' selected' : ''}`,
            tabindex: '0', role: 'button', 'aria-label': `第 ${item.round} 轮，${item.success ? `加速比 ${speedup(item.speedup)}` : '失败，加速比 0x'}`,
          });
          group.dataset.index = index;
          group.appendChild(svgNode('circle', { cx: px, cy: py, r: 22, class: 'point-hit' }));
          group.appendChild(item.success
            ? svgNode('circle', { cx: px, cy: py, r: 6, class: 'point-visible' })
            : svgNode('path', { d: `M ${px - 6} ${py - 6} L ${px + 6} ${py + 6} M ${px - 6} ${py + 6} L ${px + 6} ${py - 6}`, class: 'point-visible' }));
          group.addEventListener('click', () => selectRound(index));
          group.addEventListener('keydown', event => {
            if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); selectRound(index); }
            if (event.key === 'ArrowRight') { event.preventDefault(); selectRound(Math.min(rounds.length - 1, index + 1), true); }
            if (event.key === 'ArrowLeft') { event.preventDefault(); selectRound(Math.max(0, index - 1), true); }
          });
          group.addEventListener('mouseenter', event => showTooltip(event.currentTarget, item, px, py));
          group.addEventListener('focus', event => showTooltip(event.currentTarget, item, px, py));
          group.addEventListener('mouseleave', hideTooltip);
          group.addEventListener('blur', hideTooltip);
          svg.appendChild(group);
          appendText(svg, speedup(roundSpeedup(item)), { x: px, y: Math.max(16, py - 13), 'text-anchor': 'middle', class: 'point-label' });
          const labelEvery = Math.max(1, Math.ceil(rounds.length / Math.max(4, Math.floor(innerWidth / 62))));
          if (index % labelEvery === 0 || index === rounds.length - 1) {
            appendText(svg, item.round, { x: px, y: height - margin.bottom + 22, 'text-anchor': 'middle', class: 'tick-label' });
          }
        });
        el('trajectory-summary').textContent = `共 ${rounds.length} 个有效优化轮次，最佳整体加速比 ${speedup(measured.length ? Math.max(...measured.map(item => Number(item.speedup))) : null)}。`;
      }

      function showTooltip(target, item, x, y) {
        const tooltip = el('trajectory-tooltip');
        tooltip.textContent = `第 ${item.round} 轮 · ${item.success ? speedup(item.speedup) : '失败 · 0x'}`;
        tooltip.hidden = false;
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
        target.setAttribute('aria-describedby', 'trajectory-tooltip');
      }

      function hideTooltip(event) {
        el('trajectory-tooltip').hidden = true;
        if (event && event.currentTarget) event.currentTarget.removeAttribute('aria-describedby');
      }

      function renderTrajectoryTable() {
        const table = document.createElement('table');
        table.innerHTML = '<caption>优化轨迹数据</caption><thead><tr><th>轮次</th><th>状态</th><th>整体加速比</th></tr></thead>';
        const tbody = document.createElement('tbody');
        rounds.forEach(item => {
          const row = document.createElement('tr');
          [item.round, item.success ? '成功' : '失败', speedup(roundSpeedup(item))].forEach(value => {
            const cell = document.createElement('td'); cell.textContent = value; row.appendChild(cell);
          });
          tbody.appendChild(row);
        });
        table.appendChild(tbody);
        el('trajectory-table').replaceChildren(table);
      }

      function selectRound(index, focusPoint = false) {
        selectedIndex = index;
        lineChart();
        renderRound();
        if (focusPoint) {
          const point = el('trajectory-chart').querySelector(`.point[data-index="${index}"]`);
          if (point) point.focus();
        }
      }

      function renderRound() {
        if (!rounds.length) {
          el('round-heading').innerHTML = '<h2 id="round-title">Round Detail</h2>';
          el('case-chart-wrap').innerHTML = '<div class="empty">暂无轮次详情</div>';
          el('diff-count').textContent = '暂无代码变更';
          el('diff-list').innerHTML = '<div class="empty">暂无 code diff</div>';
          return;
        }
        const round = rounds[selectedIndex];
        const heading = el('round-heading');
        heading.replaceChildren();
        const title = document.createElement('h2');
        title.id = 'round-title';
        title.textContent = `Round ${round.round}`;
        const badge = document.createElement('span');
        badge.className = `badge${round.success ? '' : ' failure'}`;
        badge.textContent = round.success ? speedup(round.speedup) : 'FAILED · 0x';
        const id = document.createElement('p');
        id.className = 'round-id';
        id.textContent = `${round.run_name}${round.timestamp ? ` · ${round.timestamp}` : ''}`;
        const measurement = round.measurement || {};
        id.textContent += measurement.significant == null ? ' · 噪声未标定' :
          measurement.significant ? ' · 全部 case 收益超过观测噪声' : ' · 收益未全部超过观测噪声';
        heading.append(title, badge, id);
        barChart(round);
        renderDiff(round);
      }

      function barChart(round) {
        let wrap = el('case-chart-wrap');
        el('case-tooltip').hidden = true;
        if (!el('case-chart')) {
          wrap.replaceChildren();
          const replacement = svgNode('svg', { id: 'case-chart', class: 'chart', role: 'img', 'aria-labelledby': 'case-chart-title case-chart-summary' });
          wrap.appendChild(replacement);
        }
        const svg = el('case-chart');
        svg.replaceChildren();
        if (!round.cases.length) {
          svg.setAttribute('viewBox', '0 0 800 360');
          appendText(svg, '本轮无 case 性能数据', { x: 400, y: 180, 'text-anchor': 'middle', class: 'tick-label' });
          return;
        }
        const width = Math.max(320, svg.clientWidth || 800);
        const height = svg.clientHeight || 360;
        const margin = { top: 36, right: 24, bottom: 55, left: 92 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        const maxValue = Math.max(1, ...round.cases.flatMap(item => [item.baseline_us, item.candidate_us]).filter(valid).map(Number));
        const ceiling = maxValue * 1.12;
        const groupWidth = innerWidth / round.cases.length;
        const barWidth = Math.min(26, Math.max(4, groupWidth * 0.3));
        const y = value => margin.top + innerHeight - value / ceiling * innerHeight;
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        for (let tick = 0; tick <= 4; tick += 1) {
          const value = ceiling * tick / 4;
          const py = y(value);
          svg.appendChild(svgNode('line', { x1: margin.left, y1: py, x2: width - margin.right, y2: py, class: 'grid-line' }));
          appendText(svg, value.toFixed(2), { x: margin.left - 10, y: py + 4, 'text-anchor': 'end', class: 'tick-label' });
        }
        svg.appendChild(svgNode('line', { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: 'axis-line' }));
        svg.appendChild(svgNode('line', { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: 'axis-line' }));
        appendText(svg, '时延 (μs)', { x: 18, y: margin.top + innerHeight / 2, transform: `rotate(-90 18 ${margin.top + innerHeight / 2})`, 'text-anchor': 'middle', class: 'axis-label' });
        appendText(svg, 'Case Index', { x: margin.left + innerWidth / 2, y: height - 12, 'text-anchor': 'middle', class: 'axis-label' });
        round.cases.forEach((item, index) => {
          const center = margin.left + groupWidth * (index + 0.5);
          const values = [
            [item.baseline_us, center - barWidth - 2, 'bar-before'],
            [item.candidate_us, center + 2, 'bar-after'],
          ];
          values.forEach(([value, x, className]) => {
            if (!valid(value)) return;
            const top = y(value);
            const ratio = valid(item.baseline_us) && valid(item.candidate_us) ? item.baseline_us / item.candidate_us : null;
            const bar = svgNode('rect', {
              x, y: top, width: barWidth, height: Math.max(0, height - margin.bottom - top), rx: 2,
              class: `${className} bar-mark`, tabindex: '0', role: 'button',
              'aria-label': `Case ${item.index}，优化前 ${latency(item.baseline_us)}，优化后 ${latency(item.candidate_us)}，加速比 ${speedup(ratio)}`,
            });
            bar.addEventListener('mouseenter', event => showCaseTooltip(event.currentTarget, item, center, top));
            bar.addEventListener('focus', event => showCaseTooltip(event.currentTarget, item, center, top));
            bar.addEventListener('click', event => showCaseTooltip(event.currentTarget, item, center, top));
            bar.addEventListener('mouseleave', hideCaseTooltip);
            bar.addEventListener('blur', hideCaseTooltip);
            svg.appendChild(bar);
          });
          appendText(svg, item.index, { x: center, y: height - margin.bottom + 21, 'text-anchor': 'middle', class: 'tick-label' });
        });
        el('case-chart-summary').textContent = `第 ${round.round} 轮包含 ${round.cases.length} 个 case 的优化前后时延对比。`;
      }

      function showCaseTooltip(target, item, x, y) {
        const tooltip = el('case-tooltip');
        const ratio = valid(item.baseline_us) && valid(item.candidate_us) ? item.baseline_us / item.candidate_us : null;
        tooltip.replaceChildren();
        const title = document.createElement('div');
        title.className = 'tooltip-title';
        title.textContent = `Case ${item.index} · ${item.case_id}`;
        tooltip.appendChild(title);
        [['优化前', latency(item.baseline_us)], ['优化后', latency(item.candidate_us)], ['加速比', speedup(ratio)]].forEach(([label, value]) => {
          const row = document.createElement('div');
          row.className = 'tooltip-row';
          const caption = document.createElement('span');
          const strong = document.createElement('strong');
          caption.textContent = label;
          strong.textContent = value;
          row.append(caption, strong);
          tooltip.appendChild(row);
        });
        tooltip.hidden = false;
        const halfWidth = tooltip.offsetWidth / 2 + 8;
        const boundedX = Math.max(halfWidth, Math.min(x, tooltip.parentElement.clientWidth - halfWidth));
        const boundedY = Math.max(tooltip.offsetHeight + 16, y);
        tooltip.style.left = `${boundedX}px`;
        tooltip.style.top = `${boundedY}px`;
        target.setAttribute('aria-describedby', 'case-tooltip');
      }

      function hideCaseTooltip(event) {
        el('case-tooltip').hidden = true;
        if (event && event.currentTarget) event.currentTarget.removeAttribute('aria-describedby');
      }

      function displayDiffLine(line) {
        if (line.startsWith('--- a/')) return `--- 优化前：${line.slice(6)}`;
        if (line.startsWith('+++ b/')) return `+++ 优化后：${line.slice(6)}`;
        const gitHeader = line.match(/^diff --git a\/(.+) b\/(.+)$/);
        if (gitHeader && gitHeader[1] === gitHeader[2]) return `diff ${gitHeader[1]}（优化前 → 优化后）`;
        const binary = line.match(/^Binary files a\/(.+) and b\/(.+) differ$/);
        if (binary && binary[1] === binary[2]) return `二进制文件 ${binary[1]} 在优化前后存在差异`;
        return line;
      }

      function renderDiff(round) {
        const diffs = round.file_diffs || [];
        el('diff-count').textContent = `${diffs.length} 个变更文件`;
        const list = el('diff-list');
        list.replaceChildren();
        if (!diffs.length) {
          list.innerHTML = '<div class="empty">本轮未记录可展示的 code diff</div>';
          return;
        }
        diffs.forEach(file => {
          const section = document.createElement('section');
          section.className = 'diff-file';
          const header = document.createElement('header');
          const title = document.createElement('h3');
          const status = document.createElement('span');
          title.textContent = file.path;
          status.className = 'diff-status';
          status.textContent = file.status;
          header.append(title, status);
          const pre = document.createElement('pre');
          const code = document.createElement('code');
          String(file.diff || '').split('\n').forEach(line => {
            const span = document.createElement('span');
            span.className = 'diff-line';
            if (line.startsWith('+') && !line.startsWith('+++')) span.classList.add('add');
            else if (line.startsWith('-') && !line.startsWith('---')) span.classList.add('del');
            else if (line.startsWith('@@')) span.classList.add('hunk');
            else if (line.startsWith('---') || line.startsWith('+++') || line.startsWith('diff ')) span.classList.add('meta');
            span.textContent = displayDiffLine(line) || ' ';
            code.appendChild(span);
          });
          pre.appendChild(code);
          section.append(header, pre);
          list.appendChild(section);
        });
      }

      let resizeTimer;
      window.addEventListener('resize', () => {
        window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(() => { lineChart(); if (rounds.length) barChart(rounds[selectedIndex]); }, 120);
      });

      renderTrace();
      renderBaselineTable();
      renderSummary();
      renderTrajectoryTable();
      lineChart();
      renderRound();
    })();
  </script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
