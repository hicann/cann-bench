# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import json
import math
from copy import deepcopy

import pytest

from kernel_eval.paired_statistics import main, summarize_pairs


def reports(ratios):
    return [{"eval_code": f"pair_{i}", "operator": "expand", "device": "npu:7",
             "source_diff": {"candidate_sha256": "source"},
             "baseline_source_sha256": "baseline",
             "baseline_status": {"success": True}, "candidate_status": {"success": True},
             "cases": [{"case_id": "case_01", "baseline_us": ratio * 10, "candidate_us": 10,
                        "baseline_call_us": 200, "candidate_call_us": 200,
                        "timing_scope": "kernel_total", "timing_comparable": True, "candidate_passed": True,
                        "measurement_protocol": {"warmup": 10, "repeat": 30}}]} for i, ratio in enumerate(ratios)]


def test_interval_counts_process_pairs_and_preserves_outlier():
    data = reports([1, 1, 1, 1, 1, 1.2])
    summary = summarize_pairs(data)
    result = summary["cases"][0]["kernel_speedup"]
    assert result["pair_count"] == 6
    assert result["raw_speedups"] == [1, 1, 1, 1, 1, 1.2]
    assert result["geometric_mean"] == pytest.approx(1.2 ** (1 / 6))
    assert result["ci95"][0] < 1 < result["ci95"][1]
    assert summary["pairs"][-1]["max_case_error_percent"] == pytest.approx(20)
    assert data == reports([1, 1, 1, 1, 1, 1.2])


def test_exact_student_t_interval_and_reciprocal_symmetry():
    data = reports([math.exp(-.1), 1, math.exp(.1)])
    result = summarize_pairs(data)["cases"][0]["kernel_speedup"]
    half_width = 4.302652729696142 * .1 / math.sqrt(3)
    assert result["ci95"] == pytest.approx([math.exp(-half_width), math.exp(half_width)])
    inverse = summarize_pairs(reports([2, 3, 4]))["cases"][0]["kernel_speedup"]
    reverse = summarize_pairs(reports([.5, 1/3, .25]))["cases"][0]["kernel_speedup"]
    assert reverse["ci95"] == pytest.approx([1/inverse["ci95"][1], 1/inverse["ci95"][0]])


def test_one_pair_has_no_interval_and_zero_variance_is_explicit():
    assert summarize_pairs(reports([1]))["cases"][0]["kernel_speedup"]["ci95"] is None
    assert summarize_pairs(reports([1, 1]))["cases"][0]["kernel_speedup"]["ci95"] == [1, 1]


@pytest.mark.parametrize("kind", ["duplicate", "failed", "device", "source", "baseline", "protocol", "missing", "nan", "uncomparable"])
def test_invalid_pairs_are_not_silently_dropped(kind):
    data = reports([1, 1])
    if kind == "duplicate": data[1]["eval_code"] = data[0]["eval_code"]
    elif kind == "failed": data[1]["candidate_status"]["success"] = False
    elif kind == "device": data[1]["device"] = "npu:6"
    elif kind == "source": data[1]["source_diff"]["candidate_sha256"] = "other"
    elif kind == "baseline": data[1]["baseline_source_sha256"] = "other"
    elif kind == "protocol": data[1]["cases"][0]["measurement_protocol"]["repeat"] = 31
    elif kind == "missing": data[1]["cases"] = []
    elif kind == "nan": data[1]["cases"][0]["candidate_us"] = float("nan")
    elif kind == "uncomparable": data[1]["cases"][0]["timing_comparable"] = False
    with pytest.raises(ValueError): summarize_pairs(data)


def test_case_vectors_keep_pair_count_for_overall_interval():
    data = reports([.8, 1.2, .9, 1.1, .95, 1.05])
    for report in data:
        other = deepcopy(report["cases"][0]); other["case_id"] = "case_02"
        other["baseline_us"] = 20 - other["baseline_us"]
        report["cases"].append(other)
    summary = summarize_pairs(data)
    assert summary["average_speedup"]["pair_count"] == 6
    assert summary["average_speedup"]["ci95"] == [1, 1]
    assert len(summary["cases"]) == 2


def test_legacy_same_source_identity_is_distinct_from_recorded_hash():
    data = reports([1, 1])
    for report in data:
        report.pop("baseline_source_sha256")
        report["source_diff"]["changed_file_count"] = 0
    summary = summarize_pairs(data)
    assert summary["baseline_source_identity"] == "same_source_diff:source"
    data[1]["source_diff"]["baseline_sha256"] = "source"
    with pytest.raises(ValueError): summarize_pairs(data)


def test_real_source_diff_detects_baseline_change_with_fixed_candidate(tmp_path):
    from kernel_eval.optimize import _collect_source_diff
    roots = [tmp_path / name for name in ["baseline_a", "baseline_b", "candidate"]]
    for root, text in zip(roots, ["baseline A", "baseline B", "candidate"]):
        root.mkdir(); (root / "op.cpp").write_text(text)
    data = reports([1, 1])
    for report, baseline in zip(data, roots[:2]):
        report.pop("baseline_source_sha256")
        report["source_diff"] = _collect_source_diff(baseline_dir=baseline, candidate_dir=roots[2])
    assert summarize_pairs(data[:1])["status"] == "insufficient_pairs"
    assert data[0]["source_diff"]["candidate_sha256"] == data[1]["source_diff"]["candidate_sha256"]
    with pytest.raises(ValueError, match="source fingerprints"):
        summarize_pairs(data)


def test_cli_produces_readable_files_without_changing_single_run(tmp_path):
    runs = []
    for report in reports([.99, 1, 1.01]):
        path = tmp_path / report["eval_code"]; path.mkdir()
        (path / f'{path.name}.json').write_text(json.dumps(report)); runs.append(str(path))
        for side in ("baseline_results", "results"):
            (path / side).mkdir()
            (path / side / "status.json").write_text(json.dumps({"all_stages_passed": True}))
    output = tmp_path / "summary"
    assert main(["--runs", *runs, "--output", str(output)]) == 0
    value = json.loads((output / "paired_statistics.json").read_text())
    assert value["pair_count"] == 3
    assert "95% CI" in (output / "paired_statistics.md").read_text()
    assert "0.9900–1.0100" in (output / "paired_statistics.md").read_text()


def test_cli_rejects_shared_historical_baseline(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--runs", str(tmp_path), "--output", str(tmp_path / "summary")])
    assert error.value.code != 0


def test_cli_reports_invalid_json_as_input_error(tmp_path):
    (tmp_path / "baseline_results").mkdir()
    (tmp_path / "baseline_results/status.json").write_text("{")
    with pytest.raises(SystemExit) as error:
        main(["--runs", str(tmp_path), "--output", str(tmp_path / "summary")])
    assert error.value.code != 0
