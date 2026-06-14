"""Read & judge kernel_eval results (the "解读结果" part): report parsing, the report-schema
contract, known_issues policy, the trimmed task tree, and artifact collection."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml


# ── report parsing ────────────────────────────────────────────────────────────
def latest_report_json(reports_dir) -> Path:
    files = sorted(Path(reports_dir).glob("*eval_*.json"))  # 匹配 eval_<code>.json 或 <prefix>_eval_<code>.json
    if not files:
        raise FileNotFoundError(f"no eval report json under {reports_dir}")
    return files[-1]  # eval_code 'eval_YYYYMMDD_HHMMSS' → 字典序即时间序


def load_report(json_path) -> dict:
    return json.loads(Path(json_path).read_text(encoding="utf-8"))


def iter_cases(report: dict):
    """Yield (operator, case_id, case_dict) for every case."""
    for op in report.get("operators", []):
        for case in op.get("cases", []):
            yield op["operator"], case["case_id"], case


def case_num(case: dict) -> int:
    """Integer case number. kernel_eval 的 case_id 是复合串 '<rel_path>_<n>'(如 'level2/cummin_3');
    也容忍裸 int。known_issues 用 (op, n) 整数键匹配。"""
    cid = case.get("case_id")
    if cid is None:
        raise ValueError(f"case has no case_id: {case!r}")
    if isinstance(cid, int):
        return cid
    return int(str(cid).rsplit("_", 1)[-1])


def case_has_accuracy(case: dict) -> bool:
    return case.get("accuracy") is not None


def case_has_perf(case: dict) -> bool:
    return (case.get("elapsed_us") or 0) > 0


def case_acc_passed(case: dict) -> bool:
    acc = case.get("accuracy")
    return isinstance(acc, dict) and acc.get("passed") is True


# ── report schema contract (soft drift warn) ──────────────────────────────────
# 当前 in-repo report_generator 的三层字段集合(2026-06-04 刷新:含 setup_info /
# cascade_cases / genuine_pass_rate / failure_type)。report 由 dataclass asdict() 序列化。
TOP_FIELDS = {
    "version", "eval_code", "timestamp", "device",
    "total_operators", "total_cases", "passed_cases", "failed_cases",
    "overall_score", "summary", "operators", "setup_info",
}
SUMMARY_FIELDS = {
    "total_operators", "total_cases", "passed_cases", "failed_cases",
    "pass_rate", "overall_score", "cascade_cases", "genuine_pass_rate",
}
OPERATOR_FIELDS = {
    "rel_path", "operator", "total_cases", "passed_cases", "failed_cases",
    "pass_rate", "avg_speedup", "score", "cases",
    "compilation_error", "subprocess_failure_reason",
}
CASE_FIELDS = {
    "rel_path", "operator", "case_id", "status", "elapsed_us", "op_times",
    "error_msg", "device", "timestamp", "accuracy", "speedup",
    "baseline_perf_us", "t_hw_us", "perf_score", "_perf_result", "failure_type",
}
ACCURACY_FIELDS = {"passed", "threshold", "error_msg", "output_results", "metadata"}


def _diff(expected: set, actual: set) -> dict:
    return {"missing": sorted(expected - actual), "unexpected": sorted(actual - expected)}


def schema_diff(report: dict) -> dict:
    """Per-level {missing, unexpected}(采样第一个 operator/case)。空 = 一致。"""
    out = {"top": _diff(TOP_FIELDS, set(report))}
    summary = report.get("summary")
    if isinstance(summary, dict):
        out["summary"] = _diff(SUMMARY_FIELDS, set(summary))
    ops = report.get("operators") or []
    if ops:
        out["operator"] = _diff(OPERATOR_FIELDS, set(ops[0]))
        cases = ops[0].get("cases") or []
        if cases:
            out["case"] = _diff(CASE_FIELDS, set(cases[0]))
            acc = cases[0].get("accuracy")
            if isinstance(acc, dict):
                out["accuracy"] = _diff(ACCURACY_FIELDS, set(acc))
    return out


def has_drift(diff: dict) -> bool:
    return any(level["missing"] or level["unexpected"] for level in diff.values())


# ── known_issues.yaml ─────────────────────────────────────────────────────────
VALID_ACTIONS = {"skip-hang", "xfail-accuracy", "xfail-perf", "skip"}
VALID_TARGETS = {None, "baseline", "golden"}  # which candidate an entry applies to


def load_known_issues(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or []
    for i in data:
        assert i.get("action") in VALID_ACTIONS, f"bad action: {i}"
        assert i.get("target") in VALID_TARGETS, f"bad target: {i}"
    return data


def for_target(issues, target: str) -> list[dict]:
    """Filter to entries for candidate `target` ('golden'|'baseline'); target=None 对两者生效。"""
    return [i for i in issues if i.get("target") in (None, target)]


def _is_all(i: dict) -> bool:
    return str(i.get("case")).lower() == "all"


def _case_key(i: dict) -> tuple[str, int]:
    return (str(i["op"]).lower(), int(i["case"]))


def hang_cases(issues) -> set[tuple[str, int]]:
    return {_case_key(i) for i in issues if i.get("action") == "skip-hang" and not _is_all(i)}


def xfail_set(issues, action: str) -> set[tuple[str, int]]:
    return {_case_key(i) for i in issues if i.get("action") == action and not _is_all(i)}


def xfail_all_ops(issues, action: str) -> set[str]:
    return {str(i["op"]).lower() for i in issues if i.get("action") == action and _is_all(i)}


def skip_ops(issues) -> dict[str, str]:
    return {str(i["op"]).lower(): str(i.get("reason", "")) for i in issues
            if i.get("action") == "skip" and _is_all(i)}


# ── trimmed task tree ─────────────────────────────────────────────────────────
def build_trimmed_tasktree(src_tasks, dst, hang: set[tuple[str, int]],
                           keep_ops: set[str] | None = None) -> Path:
    """Copy a tasks/ tree to a temp dir, keeping only `keep_ops` and dropping known-hang cases.

    keep_ops(算子名,大小写不敏感;None = 全留)是 single-run 集成口径的关键:pytest 的
    -k/-m 选择 → 这里裁成"只含选中算子"的子树 → cli 一次跑完整棵 → 单一报告。删整个 op 目录
    (不在 keep_ops 里的),cli discover 时自然看不到。hang 用例仍按 (op, case) 从 cases.yaml 删。
    """
    src_tasks, dst = Path(src_tasks), Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src_tasks, dst)
    if keep_ops is not None:
        keep = {o.lower() for o in keep_ops}
        for proto in list(dst.rglob("proto.yaml")):
            name = yaml.safe_load(proto.read_text(encoding="utf-8"))["operator"]["name"]
            if name.lower() not in keep:
                shutil.rmtree(proto.parent)  # 丢掉未选中算子的整个 task 目录
    if not hang:
        return dst
    for cases_yaml in dst.rglob("cases.yaml"):
        data = yaml.safe_load(cases_yaml.read_text(encoding="utf-8")) or {}
        cases = data.get("cases") or []
        kept = [c for c in cases
                if (str(c["operator"]).lower(), int(c["case_id"])) not in hang]
        if len(kept) != len(cases):
            data["cases"] = kept
            cases_yaml.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                                  encoding="utf-8")
    return dst


# ── artifact collection ───────────────────────────────────────────────────────
def collect_artifacts(tmp_root, out_dir) -> int:
    """Collect the single-run kernel_eval report from pytest tmp into out_dir.

    Single-run 口径:整个选中子集只有**一份** eval_*.json(含全部 operator)+ 配套 .md/.html。
    把这套矩阵级报告三件套拷到 out_dir 根,bulky 的 prof_data(trace_view/msprof)丢弃。
    注:op 级子进程隔离下,每个算子的 profiler prof_data 落在各自子进程的临时目录、不在父
    --reports-dir 下,故 per-case kernel_details.csv 不在此收集(性能数值已写进报告 json 的
    elapsed_us/op_times)。返回收集到的报告份数(正常为 1)。
    """
    tmp_root, out_dir = Path(tmp_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for jf in sorted(tmp_root.rglob("*eval_*.json")):
        try:
            report = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not (report.get("operators") or []):
            continue
        for ext in (".json", ".md", ".html"):  # 矩阵级报告三件套 → 根
            f = jf.with_suffix(ext)
            if f.exists():
                shutil.copy(f, out_dir / f.name)
        n += 1
    return n
