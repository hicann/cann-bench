# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Invocation-level device work from explicit cache-clean boundaries."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


HELPER_KERNELS = ("CannBenchWarmup", "CannBenchCacheClean")


def measurement_protocol(warmup, repeat, inner_iters=1, helper_build=None, input_pool=None, runtime_config=None, case=None):
    protocol = {
        "version": "arena_kernel_total_v4",
        "helper_build": helper_build,
        "input_pool": input_pool,
        "runtime_config": runtime_config,
        "cache_policy": "clean_before_each_sample",
        "device_warmup": "before_each_measurement_phase",
        "profiler": "npu_only_level0_aicore_none",
        "aggregation": "median_of_invocation_kernel_sums",
        "operator_call": "independent_unprofiled_npu_events",
        "warmup": warmup,
        "repeat": repeat,
        "calls_per_round": inner_iters,
    }
    if case is not None:
        spec = {key: case.get(key) for key in ("case_id", "inputs", "attrs")}
        protocol["case_spec_sha256"] = hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return protocol


def _read_invocation_rounds(path: Path, expected, inner_iters):
    """Keep every launch; missing invocation boundaries invalidate the measurement."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    launches = []
    for row in rows:
        try:
            start = float(row["Start Time(us)"])
            duration = float(row["Duration(us)"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("kernel CSV requires finite start times and durations") from exc
        if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("kernel CSV contains an invalid launch duration or timestamp")
        launches.append({
            "name": row.get("Name") or row.get("Type") or "",
            "type": row.get("Type") or "",
            "start_us": start,
            "duration_us": duration,
            "stream_id": row.get("Stream ID"),
            "device_id": row.get("Device_id"),
        })
    launches.sort(key=lambda row: row["start_us"])
    boundaries = [i for i, row in enumerate(launches)
                  if "CannBenchCacheClean" in row["type"] or "CannBenchCacheClean" in row["name"]]
    if len(boundaries) != expected:
        raise ValueError(f"cache-clean boundary count {len(boundaries)} != expected {expected}")

    def target(row):
        return not any(token in row["type"] or token in row["name"] for token in HELPER_KERNELS)

    if any(target(row) for row in launches[:boundaries[0]]):
        raise ValueError("target launch precedes the first invocation boundary")
    rounds = []
    for index, boundary in enumerate(boundaries):
        stop = boundaries[index + 1] if index + 1 < len(boundaries) else len(launches)
        kernels = [row for row in launches[boundary + 1:stop] if target(row)]
        if not kernels:
            raise ValueError(f"invocation {index + 1} has no target kernels")
        end = launches[boundaries[index + 1]]["start_us"] if index + 1 < len(boundaries) else math.inf
        if any(row["start_us"] + row["duration_us"] > end + 0.1 for row in kernels):
            raise ValueError("a target kernel crosses an invocation boundary")
        rounds.append({"total_us": math.fsum(row["duration_us"] for row in kernels) / inner_iters,
                       "launches": kernels})
    return rounds


def _summarize_rounds(rounds, protocol):
    values = [row["total_us"] for row in rounds]
    center = statistics.median(values)
    return {
        "duration_scope": "kernel_total",
        "task_duration_us": center,
        "duration_samples_us": values,
        "duration_statistic": "median_of_invocation_kernel_sums",
        "sample_count": len(values),
        "noise_band": max(abs(value / center - 1) for value in values),
        "measurement_protocol": protocol,
        "rounds": rounds,
    }


def parse_kernel_totals(path: Path, *, warmup, repeat, inner_iters=1, helper_build=None, input_pool=None, runtime_config=None):
    rounds = _read_invocation_rounds(path, repeat, inner_iters)
    protocol = measurement_protocol(warmup, repeat, inner_iters, helper_build, input_pool, runtime_config)
    return _summarize_rounds(rounds, protocol)


def parse_kernel_batches(path: Path, *, case_ids, warmup, repeat, inner_iters=1,
                         helper_build=None, input_pool=None, runtime_config=None, session_mode="batch"):
    if not case_ids or len(set(case_ids)) != len(case_ids) or warmup < 0 or repeat < 1:
        raise ValueError("profiler session requires unique cases and valid sampling counts")
    width = warmup + repeat
    rounds = _read_invocation_rounds(path, len(case_ids) * width, inner_iters)
    protocol = measurement_protocol(warmup, repeat, inner_iters, helper_build, input_pool, runtime_config)
    protocol.update(version="arena_kernel_total_v5", profiler_session={
        "mode": session_mode, "case_ids": list(case_ids), "warmup_recorded": True,
        "input_lifetime": "all_cases_retained_through_collection",
    })
    return {case: _summarize_rounds(rounds[i * width + warmup:(i + 1) * width], protocol)
            for i, case in enumerate(case_ids)}
