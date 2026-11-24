"""Run the bundled, trusted local CPU examples through a common adapter."""

import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import yaml

from ..catalog import Catalog
from ..contracts import ContractError, read_document, within


def plan(root, task, submission, reports_dir, *, case_id=None, device_id=None, no_perf=False):
    if submission.kind != "source" or not (submission.path / "solution.py").is_file():
        raise ContractError("The example adapter expects a source directory containing solution.py")
    if device_id is not None:
        raise ContractError("The CPU example adapter does not use a device ID")
    command = [
        sys.executable, "-m", "cannbench.adapters.python_example",
        "--root", str(root), "--task", task.id,
        "--candidate", str(submission.path.resolve() / "solution.py"),
        "--reports-dir", str(reports_dir.resolve()),
    ]
    if case_id is not None:
        command.extend(["--case-id", str(case_id)])
    if no_perf:
        command.append("--no-perf")
    return command


def load_predict(path, name):
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ContractError(f"Cannot load Python implementation: {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    predict = getattr(module, "predict", None)
    if not callable(predict):
        raise ContractError(f"{path}: expected predict(batch)")
    return predict


def matches(actual, expected, tolerance):
    if isinstance(expected, (tuple, list)):
        return (
            isinstance(actual, (tuple, list))
            and len(actual) == len(expected)
            and all(matches(x, y, tolerance) for x, y in zip(actual, expected))
        )
    return type(actual) in (int, float) and math.isclose(
        actual, expected, abs_tol=tolerance["absolute"], rel_tol=tolerance["relative"],
    )


def measure(predict, batch, repetitions):
    started = perf_counter()
    for _ in range(repetitions):
        predict(batch)
    return (perf_counter() - started) / repetitions


def evaluate(root, task_id, candidate_path, *, case_id=None, no_perf=False):
    catalog = Catalog(root)
    task = catalog.task(task_id)
    assets = task.data["assets"]
    workloads = read_document(within(root, assets["workloads"]))["cases"]
    selected = [row for row in workloads if case_id is None or row["id"] == case_id]
    if not selected:
        raise ContractError(f"No workloads match case {case_id!r}")
    policy = read_document(within(root, task.data["evaluation"]["scoring_rule"]))
    contract = read_document(within(root, assets["contract"]))
    oracle = load_predict(within(root, assets["oracle"]), "example_oracle")
    result = {
        "task_id": task.id, "task_version": task.version,
        "scoring_rule": {"id": policy["id"], "version": policy["version"]},
    }
    try:
        candidate = load_predict(candidate_path, "example_candidate")
    except (Exception, SystemExit) as exc:
        return {
            **result,
            "correct": False, "complete": False,
            "cases": [], "measurements": [], "score": None,
            "error": {
                "stage": "candidate_import",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    # Freeze the input records before either implementation runs.
    batches = [tuple(tuple(row) for row in case["batch"]) for case in selected]
    checks = []
    for case, batch in zip(selected, batches):
        expected = oracle(batch)
        try:
            correct = matches(candidate(batch), expected, contract["tolerance"])
            error = None
        except Exception as exc:
            correct, error = False, str(exc)
        checks.append({"id": case["id"], "correct": correct, "error": error})

    passed = all(check["correct"] for check in checks)
    full = len(selected) == len(workloads)
    score = None
    measurements = []
    if passed and full and policy["metric"] == "all-cases-correct":
        score = 1
    if passed and full and not no_perf and policy["metric"] == "latency-ratio":
        baseline = load_predict(within(root, assets["performance_baseline"]), "example_baseline")
        settings = read_document(within(root, contract["measurement_profile"]))
        for case, batch in zip(selected, batches):
            if not matches(baseline(batch), oracle(batch), contract["tolerance"]):
                raise ContractError("The official baseline disagrees with the oracle")
            for _ in range(settings["warmup"]):
                baseline(batch)
                candidate(batch)
            samples = [[], []]
            functions = (baseline, candidate)
            for round_index in range(settings["rounds"]):
                first = round_index % 2
                for index in (first, 1 - first):
                    samples[index].append(measure(functions[index], batch, settings["calls_per_round"]))
            baseline_seconds, candidate_seconds = (median(values) for values in samples)
            if min(baseline_seconds, candidate_seconds) <= 0:
                raise ContractError("Could not measure a positive duration")
            measurements.append({
                "id": case["id"], "baseline_seconds": baseline_seconds,
                "candidate_seconds": candidate_seconds,
                "speedup": baseline_seconds / candidate_seconds,
            })
        score = math.exp(sum(math.log(row["speedup"]) for row in measurements) / len(measurements))

    return {
        **result,
        "correct": passed, "complete": full,
        "cases": checks, "measurements": measurements, "score": score,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reports-dir", required=True, type=Path)
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--no-perf", action="store_true")
    args = parser.parse_args()
    try:
        result = evaluate(
            args.root.resolve(), args.task, args.candidate,
            case_id=args.case_id, no_perf=args.no_perf,
        )
    except (ContractError, OSError) as exc:
        parser.exit(2, f"example evaluator: {exc}\n")
    (args.reports_dir / "example_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
