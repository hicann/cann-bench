#!/usr/bin/python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check directory-keyed metadata without requiring torch or an NPU."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUITES = ("kernel_bench", "omni-ops_bench")


def _metadata_operators():
    for suite in SUITES:
        bench_root = ROOT / "bench_lab" / suite
        for path in sorted((bench_root / "metadata").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            for key, value in data.items():
                if key.startswith("_"):
                    continue
                entries = value.items() if key.startswith("level") else [(key, value)]
                for name, cases in entries:
                    relative = f"{key}/{name}" if key.startswith("level") else name
                    yield pytest.param(
                        bench_root,
                        path.stem,
                        relative,
                        cases,
                        id=f"{suite}/{path.stem}/{relative}",
                    )


@pytest.fixture
def baseline_store_class(monkeypatch):
    # Import the real standard-library-only modules in an isolated package.
    # Importing kernel_eval.utils normally also imports torch-dependent utilities.
    package_name = "_bench_metadata_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "src/kernel_eval/utils")]
    monkeypatch.setitem(sys.modules, package_name, package)
    for name in ("baseline_resolver", "baseline_store"):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name,
            ROOT / "src/kernel_eval/utils" / f"{name}.py",
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
    return module.BaselineStore


@pytest.mark.parametrize("has_target_file", [False, True])
def test_default_platform_fallback_keeps_values_without_warning(
    baseline_store_class,
    tmp_path,
    caplog,
    has_target_file,
):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "910b2.json").write_text(
        json.dumps(
            {
                "level1": {"exp": {"1": {"baseline_perf_us": 12.5, "t_hw_us": 1.25}}},
            }
        ),
        encoding="utf-8",
    )
    if has_target_file:
        (metadata / "950dt.json").write_text(
            json.dumps(
                {
                    "level1": {
                        "exp": {"2": {"baseline_perf_us": 6.0, "t_hw_us": 0.75}}
                    },
                }
            ),
            encoding="utf-8",
        )
    store = baseline_store_class(tmp_path, project_root=tmp_path, hardware="950dt")
    with caplog.at_level("WARNING"):
        store.load()
        assert store.get_perf("level1/exp", 1) == 12.5
        assert store.get_t_hw("level1/exp", 1) == 1.25
        assert store.has_baseline("level1/exp", 1)
        if has_target_file:
            assert store.get_perf("level1/exp", 2) == 6.0
            assert store.get_t_hw("level1/exp", 2) == 0.75
    assert not caplog.records
    sources = store.get_sources("level1/exp", 1)
    for source in sources.values():
        assert source == {
            "requested_hardware": "950dt",
            "source_hardware": "910b2",
            "fallback_used": True,
            "source": "metadata/910b2.json",
        }


def test_baseline_sources_follow_each_field(baseline_store_class, tmp_path):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    for platform, entries in {
        "950dt": {"1": {"baseline_perf_us": 8.0}},
        "910b2": {"1": {"baseline_perf_us": 12.0, "t_hw_us": 2.0}},
    }.items():
        (metadata / f"{platform}.json").write_text(
            json.dumps(
                {
                    "level1": {"exp": entries},
                }
            ),
            encoding="utf-8",
        )
    store = baseline_store_class(tmp_path, hardware="Ascend950DT_9582")
    store.load()
    assert (store.get_perf("level1/exp", 1), store.get_t_hw("level1/exp", 1)) == (8, 2)
    sources = store.get_sources("level1/exp", 1)
    assert sources["baseline_perf_us"]["source_hardware"] == "950dt"
    assert sources["baseline_perf_us"]["fallback_used"] is False
    assert sources["t_hw_us"]["source_hardware"] == "910b2"
    assert sources["t_hw_us"]["fallback_used"] is True
    assert store.get_sources("level1/exp", 999) == {}


def test_multihardware_value_source_is_not_inferred_from_filename(
    baseline_store_class, tmp_path
):
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    (metadata / "950dt.json").write_text(
        json.dumps(
            {
                "level1": {"exp": {"1": {"baseline_perf_us": {"950pr": 8.0}}}},
            }
        ),
        encoding="utf-8",
    )
    store = baseline_store_class(tmp_path, hardware="950dt")
    store.load()
    assert store.get_perf("level1/exp", 1, hardware="950pr") == 8
    source = store.get_sources("level1/exp", 1, hardware="950pr")["baseline_perf_us"]
    assert source["source_hardware"] == "950pr"
    assert source["source"] == "metadata/950dt.json"
    assert source["fallback_used"] is False


@pytest.mark.parametrize(
    "bench_root,hardware,relative,cases", list(_metadata_operators())
)
def test_metadata_keys_resolve_to_delivery_and_performance(
    baseline_store_class,
    bench_root,
    hardware,
    relative,
    cases,
):
    folder = bench_root / relative
    assert (folder / "proto.yaml").is_file(), f"Stale metadata path: {folder}"
    assert cases, f"Empty metadata for {relative}"
    # Start at the delivery directory to cover the runner's upward search.
    store = baseline_store_class(folder, project_root=ROOT, hardware=hardware)
    store.load()
    for case_id, expected in cases.items():
        assert store._resolve_entry(relative, int(case_id)) == expected
        for field, getter in (
            ("baseline_perf_us", store.get_perf),
            ("t_hw_us", store.get_t_hw),
        ):
            value = expected.get(field)
            if isinstance(value, dict):
                value = value.get(hardware)
            # Null/zero means unmeasured; do not invent measurements or disable
            # the production loader's intentional default-platform fallback.
            if value is not None and value > 0:
                assert getter(relative, int(case_id)) == pytest.approx(value)
        baseline = expected.get("baseline_perf_us")
        if isinstance(baseline, dict):
            baseline = baseline.get(hardware)
        if baseline is not None:
            assert store.has_baseline(relative, int(case_id))
