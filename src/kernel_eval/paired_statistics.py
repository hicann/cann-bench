# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Uncertainty across independent process pairs of operator evaluations."""

import argparse
import json
import math
from pathlib import Path
import statistics

from scipy.stats import t


def _ratio_interval(values):
    logs = [math.log(value) for value in values]
    center = statistics.mean(logs)
    interval = None
    if len(values) >= 2:
        half_width = float(t.ppf(0.975, len(values) - 1)) * statistics.stdev(logs) / math.sqrt(len(values))
        interval = [math.exp(center - half_width), math.exp(center + half_width)]
    return {
        "pair_count": len(values), "geometric_mean": math.exp(center), "ci95": interval,
        "min": min(values), "max": max(values), "raw_speedups": values,
    }


def summarize_pairs(reports):
    """Each report contributes one paired observation per case, never 30 observations."""
    if not reports:
        raise ValueError("at least one independent pair is required")
    first = reports[0]
    def identity(report):
        diff = report["source_diff"]
        baseline = report.get("baseline_source_sha256") or diff.get("baseline_sha256")
        if baseline is None and diff.get("changed_file_count") == 0:
            baseline = "same_source_diff:" + diff["candidate_sha256"]
        if baseline is None:
            raise ValueError("baseline source fingerprint missing; rerun with the current evaluator")
        return report["operator"], report["device"], diff["candidate_sha256"], baseline
    reference = {case["case_id"]: case for case in first["cases"]}
    if not reference or len(reference) != len(first["cases"]):
        raise ValueError("case IDs must be nonempty and unique")
    identities = set()
    case_values = {case: {"kernel": [], "call": []} for case in reference}
    pairs = []
    for report in reports:
        pair_id = report["eval_code"]
        if pair_id in identities:
            raise ValueError("independent pair IDs must be unique")
        identities.add(pair_id)
        if identity(report) != identity(first):
            raise ValueError("operator, device and source fingerprints must match across pairs")
        if not all(report[side].get("success") is True for side in ("baseline_status", "candidate_status")):
            raise ValueError(f"pair {pair_id} failed; no paired statistics published")
        rows = {case["case_id"]: case for case in report["cases"]}
        if set(rows) != set(reference) or len(rows) != len(report["cases"]):
            raise ValueError("all pairs must contain the same unique case IDs")
        for case_id, row in rows.items():
            protocol = row.get("measurement_protocol")
            if (row.get("timing_scope") != "kernel_total" or row.get("timing_comparable") is not True
                    or row.get("candidate_passed") is not True or not protocol
                    or protocol != reference[case_id].get("measurement_protocol")):
                raise ValueError(f"incompatible or failed measurement for {case_id}")
            for scope, before, after in (("kernel", "baseline_us", "candidate_us"),
                                         ("call", "baseline_call_us", "candidate_call_us")):
                values = [row.get(before), row.get(after)]
                if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                       for v in values):
                    raise ValueError(f"finite positive {scope} durations required for {case_id}")
                ratio = values[0] / values[1]
                if not math.isfinite(ratio) or ratio <= 0:
                    raise ValueError(f"finite positive {scope} ratio required for {case_id}")
                case_values[case_id][scope].append(ratio)
        raw = [case_values[case]["kernel"][-1] for case in reference]
        pairs.append({"pair_id": pair_id, "order": report.get("pair_order"),
                      "average_speedup": statistics.mean(raw),
                      "max_case_error_percent": 100 * max(abs(value - 1) for value in raw)})
    return {
        "schema_version": 1, "operator": first["operator"], "device": first["device"],
        "candidate_source_sha256": identity(first)[2], "pair_count": len(reports),
        "baseline_source_identity": identity(first)[3],
        "status": "estimated" if len(reports) >= 2 else "insufficient_pairs",
        "method": "student_t_on_independent_pair_log_ratios",
        "confidence_level": 0.95, "interval_scope": "pointwise",
        "deviation_reference": "max_abs_speedup_minus_one; measurement_error_only_for_same_source",
        "sampling_unit": "independent_baseline_candidate_process_pair",
        "assumptions": ["independent process pairs", "approximately normal pair log ratios", "fixed benchmark definitions and inputs"],
        "interpretation": "CI estimates the geometric mean ratio; it does not bound individual or future runs, or remove systematic bias",
        "pairs": pairs,
        "average_speedup": _ratio_interval([pair["average_speedup"] for pair in pairs]),
        "cases": [{"case_id": case, "measurement_protocol": reference[case]["measurement_protocol"],
                   "kernel_speedup": _ratio_interval(values["kernel"]),
                   "call_speedup": _ratio_interval(values["call"])} for case, values in sorted(case_values.items())],
    }


def write_summary(summary, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_statistics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    def text_interval(value):
        return "样本不足" if value["ci95"] is None else f'{value["ci95"][0]:.4f}–{value["ci95"][1]:.4f}'
    lines = ["# 独立配对性能统计", "", f'算子：{summary["operator"]}；设备：{summary["device"]}；独立配对：{summary["pair_count"]}。', "",
             "每次评测仍以逐轮 kernel 总时间中位数评分。下表汇总独立配对比值的几何均值，95% 区间采用配对 log 比值的 Student t 区间。统计单位为进程配对；每个进程内的重复采样数不作为独立配对数。", "",
             f'各配对平均 speedup 的几何均值：{summary["average_speedup"]["geometric_mean"]:.4f}；95% CI：{text_interval(summary["average_speedup"])}。', "",
             "| Case | Kernel speedup 几何均值 | 95% CI | 原始配对范围 | 调用 speedup 几何均值 | 95% CI |", "|---|---:|---:|---:|---:|---:|"]
    for case in summary["cases"]:
        kernel, call = case["kernel_speedup"], case["call_speedup"]
        lines.append(f'| {case["case_id"]} | {kernel["geometric_mean"]:.4f} | {text_interval(kernel)} | {kernel["min"]:.4f}–{kernel["max"]:.4f} | {call["geometric_mean"]:.4f} | {text_interval(call)} |')
    lines += ["", "| 配对 | 执行顺序 | 平均 speedup | 最大 speedup 偏离 1 的幅度 |", "|---|---|---:|---:|"]
    for pair in summary["pairs"]:
        order = " → ".join(pair["order"]) if pair["order"] else "未记录"
        lines.append(f'| {pair["pair_id"]} | {order} | {pair["average_speedup"]:.4f} | {pair["max_case_error_percent"]:.2f}% |')
    lines += ["", "区间为逐项区间，依赖进程配对独立、log 比值近似正态、用例定义与输入固定的假设；它估计几何均值，不约束单次或未来运行的波动，也不能消除系统偏移。同源稳定性验收继续检查原始逐对偏差；不同实现的比值变化还包含优化收益。JSON 保留全部配对比值及测量协议。", "",
              "方法参考：[NIST 配对观测](https://www.itl.nist.gov/div898/handbook/prc/section3/prc311.htm) 与 [均值置信区间](https://itl.nist.gov/div898/software/dataplot/refman1/auxillar/conflimi.htm)。"]
    (output / "paired_statistics.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="independent kernel_eval.optimize run directories")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reports = []
        for run in args.runs:
            for side in ("baseline_results", "results"):
                status = run / side / "status.json"
                if not status.is_file() or json.loads(status.read_text()).get("all_stages_passed") is not True:
                    raise ValueError(f"{run}: independent local baseline and candidate must pass all five stages; shared baselines cannot estimate pair uncertainty")
            reports.append(json.loads((run / f"{run.name}.json").read_text()))
        summary = summarize_pairs(reports)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))
    write_summary(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
