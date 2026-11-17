# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Test case loading, normalization, and selection.

Merged from runners/case_loader.py and runners/case_selection.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FLOAT_DTYPES = {"float16", "float32", "float64", "bfloat16", "half", "double", "float"}
BOOL_DTYPES = {"bool"}
INT_DTYPES = {"int8", "uint8", "int16", "int32", "int64", "long", "short", "byte"}

# Three-stage case file names (current)
SMOKE_CASES_FILENAME = "smoke_cases.json"
COMPLETE_CASES_FILENAME = "complete_cases.json"
PROFILE_CASES_FILENAME = "profile_cases.json"

# Legacy file names kept for backward compatibility so op directories that
# predate the 3-stage rename continue to work.
LEGACY_SMOKE_FILENAME = "smoke_test_cases.json"
LEGACY_COMPLETE_FILENAME = "full_test_cases.json"
LEGACY_PROFILE_FILENAME = "test_cases.json"

# Map each case_filter to (primary, *fallbacks) filenames. The first existing
# file wins; if none exist, the primary is returned so callers see a missing
# file path instead of None.
_FILTER_FILENAMES: dict[str, tuple[str, ...]] = {
    "smoke": (SMOKE_CASES_FILENAME, LEGACY_SMOKE_FILENAME),
    "complete": (COMPLETE_CASES_FILENAME, LEGACY_COMPLETE_FILENAME),
    "full": (COMPLETE_CASES_FILENAME, LEGACY_COMPLETE_FILENAME),
    "perf": (COMPLETE_CASES_FILENAME, PROFILE_CASES_FILENAME, LEGACY_COMPLETE_FILENAME),
    "profile": (PROFILE_CASES_FILENAME, LEGACY_PROFILE_FILENAME),
    "non_smoke": (PROFILE_CASES_FILENAME, LEGACY_PROFILE_FILENAME),
    "all": (PROFILE_CASES_FILENAME, LEGACY_PROFILE_FILENAME),
}

# Role sets for case selection
SMOKE_ROLES = {"smoke-test", "smoke_test", "smoke"}
COMPLETE_ROLES = {
    "complete-test", "complete_test", "complete",
    "full-test", "full_test", "full",
}
PERF_ROLES = {"perf-anchored", "perf_anchored", "perf-anchor"}


# ---------------------------------------------------------------------------
# Case Loading
# ---------------------------------------------------------------------------

def load_test_cases(path: Path) -> list[dict[str, Any]]:
    """Load and normalize test cases from a JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cases", payload.get("test_cases"))
    if not isinstance(payload, list):
        raise ValueError(f"cases file must contain a top-level list or a cases list: {path}")
    return [_normalize_case(item, index=index) for index, item in enumerate(payload)]


def resolve_cases_path(cases_dir: Path, case_filter: str) -> Path:
    """Resolve the correct cases file based on filter.

    Three-stage file layout (current):
    - "smoke"    -> smoke_cases.json    (Stage 1: small shapes, quick debug)
    - "complete" -> complete_cases.json (Stage 2: agent-authored ~20 cases
                                         covering all dimensions via note-axes)
    - "profile"  -> profile_cases.json  (Stage 3: 5 large shapes for perf)

    Legacy filters "full" -> complete, "non_smoke"/"all" -> profile.
    If the primary filename does not exist, legacy filenames are tried in order
    so existing op directories keep working.
    """
    candidates = _FILTER_FILENAMES.get(case_filter)
    if not candidates:
        # Unknown filter: default to profile file for backward compatibility.
        candidates = _FILTER_FILENAMES["profile"]
    for name in candidates:
        path = cases_dir / name
        if path.exists():
            return path
    return cases_dir / candidates[0]


def load_all_cases(cases_dir: Path) -> list[dict[str, Any]]:
    """Load cases from all stage files if they exist (current + legacy)."""
    all_cases: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    filenames = (
        SMOKE_CASES_FILENAME,
        COMPLETE_CASES_FILENAME,
        PROFILE_CASES_FILENAME,
        LEGACY_SMOKE_FILENAME,
        LEGACY_COMPLETE_FILENAME,
        LEGACY_PROFILE_FILENAME,
    )
    for filename in filenames:
        path = cases_dir / filename
        if path.exists() and path not in seen_paths:
            seen_paths.add(path)
            all_cases.extend(load_test_cases(path))
    return all_cases


# ---------------------------------------------------------------------------
# Case Selection
# ---------------------------------------------------------------------------

def is_smoke_case(case: dict[str, Any]) -> bool:
    """Return True if the case is tagged as a smoke test."""
    roles = case.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    if isinstance(roles, list) and any(str(role).strip().lower() in SMOKE_ROLES for role in roles):
        return True
    case_id = str(case.get("case_id", "")).lower()
    return case_id == "smoke" or case_id.startswith("smoke_") or "_smoke" in case_id


def is_complete_case(case: dict[str, Any]) -> bool:
    """Return True if the case is tagged as a complete/full test."""
    roles = case.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    if isinstance(roles, list) and any(str(role).strip().lower() in COMPLETE_ROLES for role in roles):
        return True
    case_id = str(case.get("case_id", "")).lower()
    return (
        case_id.startswith("complete_") or "_complete_" in case_id
        or case_id.startswith("full_") or "_full_" in case_id
    )


# Legacy alias kept so external callers that still use is_full_case keep
# working. Prefer is_complete_case in new code.
is_full_case = is_complete_case


def is_perf_case(case: dict[str, Any]) -> bool:
    """Return True if the case is tagged with perf-anchored role."""
    roles = case.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    if isinstance(roles, list) and any(str(role).strip().lower() in PERF_ROLES for role in roles):
        return True
    case_id = str(case.get("case_id", "")).lower()
    return case_id.startswith("profile_") or "_profile_" in case_id


def select_cases(cases: list[dict[str, Any]], case_filter: str = "all") -> list[dict[str, Any]]:
    """Filter cases by role/stage."""
    if case_filter == "all":
        return list(cases)
    if case_filter == "smoke":
        return [case for case in cases if is_smoke_case(case)]
    if case_filter in ("complete", "full"):
        return [case for case in cases if is_complete_case(case)]
    if case_filter == "perf":
        return [case for case in cases if is_perf_case(case)]
    if case_filter in ("profile", "non_smoke"):
        # Profile cases are anything that is not smoke and not complete.
        return [
            case for case in cases
            if not is_smoke_case(case) and not is_complete_case(case)
        ]
    raise ValueError(f"unsupported case_filter: {case_filter}")


# ---------------------------------------------------------------------------
# Internal normalization helpers
# ---------------------------------------------------------------------------

def _normalize_case(raw_case: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(raw_case, dict):
        raise ValueError(f"case at index {index} must be an object")
    if "case_id" not in raw_case:
        raise ValueError(f"case at index {index} must define case_id")

    case_id = str(raw_case["case_id"])
    inputs = _normalize_section(raw_case.get("inputs", {}), case_id=case_id, section_name="inputs")
    attrs = _normalize_section(raw_case.get("attrs", {}), case_id=case_id, section_name="attrs")
    normalized_case = {
        "case_id": case_id,
        "inputs": inputs,
        "attrs": attrs,
    }
    for key in ("roles", "tolerance", "meta"):
        if key in raw_case:
            normalized_case[key] = raw_case[key]
    return normalized_case


def _normalize_section(raw_section: Any, *, case_id: str, section_name: str) -> dict[str, dict[str, Any]]:
    if raw_section in ({}, None):
        return {}
    if not isinstance(raw_section, dict):
        raise ValueError(f"{section_name} for case {case_id} must be an object")

    normalized: dict[str, dict[str, Any]] = {}
    for name, spec in raw_section.items():
        normalized[str(name)] = _normalize_spec(spec, case_id=case_id, name=str(name), section_name=section_name)
    return normalized


def _normalize_spec(spec: Any, *, case_id: str, name: str, section_name: str) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError(f"{section_name}.{name} for case {case_id} must be an object")
    dtype = _require_text(spec, "dtype", case_id=case_id, name=name, section_name=section_name)

    if section_name == "inputs":
        shape = _normalize_shape(spec.get("shape"), case_id=case_id, name=name, section_name=section_name, required=True)
        normalized = {"shape": shape, "dtype": dtype}
        if "value" in spec:
            normalized["value"] = spec["value"]
            return normalized
        normalized["min"] = spec.get("min", _default_min(dtype))
        normalized["max"] = spec.get("max", _default_max(dtype))
        for key in ("input_init", "scale"):
            if key in spec:
                normalized[key] = spec[key]
        return normalized

    if "value" not in spec:
        raise ValueError(f"{section_name}.{name} for case {case_id} must define value")
    shape = _normalize_shape(spec.get("shape", []), case_id=case_id, name=name, section_name=section_name, required=False)
    return {
        "shape": shape,
        "dtype": dtype,
        "value": spec["value"],
    }


def _require_text(spec: dict[str, Any], key: str, *, case_id: str, name: str, section_name: str) -> str:
    value = spec.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section_name}.{name} for case {case_id} must define non-empty {key}")
    return value.strip()


def _normalize_shape(
    value: Any,
    *,
    case_id: str,
    name: str,
    section_name: str,
    required: bool,
) -> list[int]:
    if value is None:
        if required:
            raise ValueError(f"{section_name}.{name} for case {case_id} must define shape")
        return []
    if not isinstance(value, list):
        raise ValueError(f"{section_name}.{name} for case {case_id} shape must be a list")

    shape: list[int] = []
    for dim in value:
        if not isinstance(dim, int):
            raise ValueError(f"{section_name}.{name} for case {case_id} shape entries must be int")
        shape.append(dim)
    return shape


def _default_min(dtype: str) -> Any:
    normalized = dtype.lower()
    if normalized in BOOL_DTYPES:
        return False
    if normalized in FLOAT_DTYPES:
        return -1.0
    if normalized in INT_DTYPES:
        return 0
    raise ValueError(f"unsupported dtype for randomized inputs: {dtype}")


def _default_max(dtype: str) -> Any:
    normalized = dtype.lower()
    if normalized in BOOL_DTYPES:
        return True
    if normalized in FLOAT_DTYPES:
        return 1.0
    if normalized in INT_DTYPES:
        return 7
    raise ValueError(f"unsupported dtype for randomized inputs: {dtype}")
