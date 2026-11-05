# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""MC2 metadata keeps measured baselines separate from theoretical anchors."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from docs.hap_thw_model import compute_mc2_a5 as anchors
from docs.hap_thw_model import compute_mc2_pr_baselines as measurements


def _metadata(platform):
    return json.loads(
        (anchors.BENCH / "metadata" / f"{platform}.json").read_text(encoding="utf-8")
    )


@pytest.fixture
def pr_snapshot(monkeypatch, tmp_path):
    cases = list(anchors.a5_cases())
    monkeypatch.setattr(anchors, "a5_cases", lambda bench=tmp_path: iter(cases))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    path = metadata / "950pr.json"
    path.write_bytes((anchors.BENCH / "metadata/950pr.json").read_bytes())
    return tmp_path, path, cases


def _write(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def test_published_resource_metadata_passes_no_write_self_check():
    # Check the delivered anchors, without regenerating fingerprints first.
    anchors.generate()


def test_pr_measured_source_is_distinct_from_non_mc2_measurements():
    meta = _metadata("950pr")["_metadata"]
    source = meta["mc2_baseline_source"]
    assert source["kind"] == "measured"
    assert source["case_count"] == 180
    assert len(source["case_ids"]) == 9
    assert all("/" in key for key in source["case_ids"])
    assert source["source_hardware"] == source["target_hardware"] == "950pr"
    assert source["provenance_evidence"]["kind"] == "measurement_declaration"
    assert source["provenance_evidence"]["statement"]
    assert not measurements.PROXY_FIELDS.intersection(source)
    assert "exclude_mc2" in meta["baseline_source"]["scope"]
    assert "measured" in meta["baseline_source"]
    assert "theoretical" in meta["mc2_hap"]["confidence"]


def test_published_measurement_self_check_does_not_rewrite_files(capsys):
    paths = [anchors.BENCH / "metadata" / f"{hw}.json" for hw in ("950pr", "950dt")]
    original = [path.read_bytes() for path in paths]
    measurements.verify()
    assert [path.read_bytes() for path in paths] == original
    assert "case coverage and fingerprints verified" in capsys.readouterr().out


def test_measurement_check_needs_neither_dt_data_nor_resource_model(
    monkeypatch, pr_snapshot
):
    bench, path, _ = pr_snapshot
    assert not (bench / "metadata/950dt.json").exists()

    def unexpected_model_call(*args, **kwargs):
        pytest.fail(
            "Measurement verification must not compute a theoretical or proxy latency"
        )

    for name in ("compute", "generate", "model_metadata"):
        monkeypatch.setattr(anchors, name, unexpected_model_call)
    # Even a changed theoretical bound must not alter or clamp a measurement.
    data = json.loads(path.read_text(encoding="utf-8"))
    data["level4"]["allto_all_matmul"]["21"]["t_hw_us"] = 1e9
    _write(path, data)
    original = path.read_bytes()
    measurements.verify(bench)
    assert path.read_bytes() == original


def test_changed_measurement_is_rejected_without_repairing_it(pr_snapshot):
    bench, path, _ = pr_snapshot
    data = json.loads(path.read_text(encoding="utf-8"))
    data["level4"]["allto_all_matmul"]["21"]["baseline_perf_us"] += 0.01
    _write(path, data)
    altered = path.read_bytes()
    with pytest.raises(ValueError, match="Stale MC2 950PR baseline_sha256"):
        measurements.verify(bench)
    assert path.read_bytes() == altered


def test_theoretical_write_preserves_all_measured_baselines(pr_snapshot):
    bench, _, _ = pr_snapshot
    dt_path = bench / "metadata/950dt.json"
    dt_path.write_bytes((anchors.BENCH / "metadata/950dt.json").read_bytes())
    paths = [bench / "metadata" / f"{hw}.json" for hw in ("950pr", "950dt")]
    before = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    anchors.generate(write=True, bench=bench)
    # The published T_HW values already match, so a theoretical refresh must
    # preserve the entire file, including every measured value and its source.
    assert [json.loads(path.read_text(encoding="utf-8")) for path in paths] == before
    measurements.verify(bench)


@pytest.mark.parametrize(
    "value", [0, -1, True, None, "12.3", float("nan"), float("inf")]
)
def test_measurement_check_rejects_invalid_latency(pr_snapshot, value):
    bench, path, _ = pr_snapshot
    data = json.loads(path.read_text(encoding="utf-8"))
    data["level4"]["allto_all_matmul"]["21"]["baseline_perf_us"] = value
    _write(path, data)
    with pytest.raises(ValueError, match="finite and positive"):
        measurements.verify(bench)


def test_measurement_check_rejects_missing_latency(pr_snapshot):
    bench, path, _ = pr_snapshot
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["level4"]["allto_all_matmul"]["21"]["baseline_perf_us"]
    _write(path, data)
    with pytest.raises(ValueError, match="Missing MC2 950PR measurement"):
        measurements.verify(bench)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "derived_proxy"),
        ("source_hardware", "950dt"),
        ("target_hardware", "950dt"),
    ],
)
def test_measurement_check_rejects_other_provenance(pr_snapshot, field, value):
    bench, path, _ = pr_snapshot
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_metadata"]["mc2_baseline_source"][field] = value
    _write(path, data)
    with pytest.raises(ValueError, match="must declare measurements on 950PR"):
        measurements.verify(bench)


@pytest.mark.parametrize("field", sorted(measurements.PROXY_FIELDS))
def test_measurement_check_rejects_obsolete_proxy_fields(pr_snapshot, field):
    bench, path, _ = pr_snapshot
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_metadata"]["mc2_baseline_source"][field] = "obsolete proxy provenance"
    _write(path, data)
    with pytest.raises(ValueError, match="must not contain proxy fields"):
        measurements.verify(bench)


@pytest.mark.parametrize("field,value", [("kind", "model_fit"), ("statement", "")])
def test_measurement_check_requires_measurement_declaration(pr_snapshot, field, value):
    bench, path, _ = pr_snapshot
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_metadata"]["mc2_baseline_source"]["provenance_evidence"][field] = value
    _write(path, data)
    with pytest.raises(ValueError, match="requires a measurement declaration"):
        measurements.verify(bench)


@pytest.mark.parametrize(
    "field,value", [("case_count", 179), ("case_ids", {}), ("case_sha256", "stale")]
)
def test_measurement_check_rejects_stale_scope(pr_snapshot, field, value):
    bench, path, _ = pr_snapshot
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_metadata"]["mc2_baseline_source"][field] = value
    _write(path, data)
    with pytest.raises(ValueError, match=f"Stale MC2 950PR {field}"):
        measurements.verify(bench)


def test_measurement_check_rejects_changed_case_spec(pr_snapshot):
    bench, _, cases = pr_snapshot
    case = next(
        case for op, case in cases if op == "allto_all_matmul" and case["case_id"] == 21
    )
    case["input_shape"][0][0] *= 2
    with pytest.raises(ValueError, match="Stale MC2 950PR case_sha256"):
        measurements.verify(bench)


def test_measurement_check_rejects_duplicate_case(pr_snapshot):
    bench, _, cases = pr_snapshot
    entry = next(entry for entry in cases if entry[0] == "allto_all_matmul")
    cases.append(copy.deepcopy(entry))
    with pytest.raises(ValueError, match="Duplicate MC2 950PR case"):
        measurements.verify(bench)


def test_measurement_check_rejects_empty_scope(pr_snapshot):
    bench, _, cases = pr_snapshot
    cases.clear()
    with pytest.raises(ValueError, match="No supported MC2 950PR cases"):
        measurements.verify(bench)


def test_measurement_cli_rejects_old_write_option():
    paths = [anchors.BENCH / "metadata" / f"{hw}.json" for hw in ("950pr", "950dt")]
    original = [path.read_bytes() for path in paths]
    result = subprocess.run(
        [sys.executable, str(Path(measurements.__file__)), "--write"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --write" in result.stderr
    assert [path.read_bytes() for path in paths] == original


def test_historical_unknown_pr_sampling_fields_remain_unknown():
    source = _metadata("950pr")["_metadata"]["mc2_baseline_source"]
    for field in (
        "implementation",
        "source_revision",
        "device",
        "world_size",
        "cann_version",
        "measured_at",
        "method",
        "warmup",
        "repeat",
        "aggregation",
        "reports",
    ):
        assert source[field] is None


def test_historical_unknown_dt_sampling_fields_remain_unknown():
    source = _metadata("950dt")["_metadata"]["baseline_source"]
    assert source["kind"] == "measured"
    for field in ("source_revision", "warmup", "repeat", "aggregation"):
        assert source[field] is None
    assert source["cann_version"]["markdown_reports"] is None


def test_pr_delivery_declares_measured_values_without_personal_attribution():
    source = _metadata("950pr")["_metadata"]["mc2_baseline_source"]
    assert source["kind"] == "measured"
    assert source["source_hardware"] == source["target_hardware"] == "950pr"
    assert "用户" not in json.dumps(source, ensure_ascii=False)
    readme = (anchors.ROOT / "README.md").read_text(encoding="utf-8")
    assert "950PR 有 180 条固定实测基线" in readme
