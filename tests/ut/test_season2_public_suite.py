"""Release invariants for the public Season2 tree, without NPU execution."""

import ast
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from kernel_eval.benches.cann_loader import CannTaskLoader, GoldenLoader
from kernel_eval.eval.index_check import validate_index_gather_outputs
from kernel_eval.utils.baseline_store import BaselineStore

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"
MANIFEST = json.loads((TASKS / "suite_manifest.json").read_text(encoding="utf-8"))


def test_manifest_version_matches_runtime_task_version():
    from kernel_eval._version import read_tasks_version

    assert MANIFEST["tasks_version"] == read_tasks_version(TASKS)


def _text_sha256(content):
    """Hash public UTF-8 text with LF endings, independent of Git checkout OS."""
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def test_manifest_digest_policy_handles_line_endings_but_detects_content_changes():
    assert MANIFEST["files_sha256_policy"] == "utf8_text_crlf_to_lf"
    assert _text_sha256(b"alpha\r\nbeta\r\n") == _text_sha256(b"alpha\nbeta\n")
    assert _text_sha256(b"alpha\nbeta\n") != _text_sha256(b"alpha\ngamma\n")
    assert _text_sha256(b"alpha \nbeta\n") != _text_sha256(b"alpha\nbeta\n")


def test_public_manifest_and_archive_counts():
    declared = {r["path"] for r in MANIFEST["operators"]}
    observed = {
        p.parent.relative_to(ROOT).as_posix() for p in TASKS.glob("level*/*/proto.yaml")
    }
    assert declared == observed
    assert len(declared) == 60
    assert Counter(r["level"] for r in MANIFEST["operators"]) == {
        "L1": 2,
        "L2": 13,
        "L3": 27,
        "L4": 13,
        "L5": 5,
    }
    assert (
        len(list((ROOT / "archives/season1/tasks").glob("level*/*/proto.yaml"))) == 53
    )
    assert not (TASKS / "inner_tasks").exists()


@pytest.mark.parametrize("operator", MANIFEST["operators"], ids=lambda op: op["id"])
def test_public_task_contract(operator):
    directory = ROOT / operator["path"]
    for name, digest in operator["files_sha256"].items():
        assert _text_sha256((directory / name).read_bytes()) == digest, (
            f"{operator['path']}/{name}"
        )
    proto = yaml.safe_load((directory / "proto.yaml").read_text(encoding="utf-8"))[
        "operator"
    ]
    cases = yaml.safe_load((directory / "cases.yaml").read_text(encoding="utf-8"))[
        "cases"
    ]
    assert len(cases) == 20
    assert {c["case_id"] for c in cases} == set(range(1, 21))
    assert all(c["operator"] == proto["name"] for c in cases)
    assert proto["difficulty"] == operator["level"]
    assert proto["schema"].split("(")[0].strip() == operator["id"]
    tree = ast.parse((directory / "golden.py").read_text(encoding="utf-8"))
    compile(tree, str(directory / "golden.py"), "exec")
    assert operator["id"] in {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_loader_discovers_only_current_suite():
    specs = CannTaskLoader(str(TASKS)).list_tasks()
    assert len(specs) == 60
    assert Counter(s.difficulty.value for s in specs) == {
        "L1": 2,
        "L2": 13,
        "L3": 27,
        "L4": 13,
        "L5": 5,
    }


@pytest.mark.parametrize("operator", MANIFEST["operators"], ids=lambda op: op["id"])
def test_golden_module_import_and_schema_entrypoint(operator):
    rel_path = Path(operator["path"]).relative_to("tasks").as_posix()
    function = GoldenLoader(str(TASKS)).get_golden_function(rel_path)
    assert callable(function)
    assert function.__name__ == operator["id"]


@pytest.mark.parametrize(
    "platform,missing", [("910b2", 0), ("910c", 0), ("950pr", 0), ("950dt", 0)]
)
def test_metadata_has_only_selected_public_records(platform, missing):
    data = json.loads(
        (TASKS / "metadata" / f"{platform}.json").read_text(encoding="utf-8")
    )
    expected = {(f"level{r['level'][1:]}", r["id"]) for r in MANIFEST["operators"]}
    actual = {
        (level, op)
        for level, ops in data.items()
        if level.startswith("level")
        for op in ops
    }
    assert actual <= expected
    excluded = set()
    if platform in {"910b2", "910c"}:
        excluded.add(("level3", "embedding_hash_table_lookup_or_insert"))
        excluded.add(("level4", "dynamic_mx_quant"))
    assert len(actual) == 60 - len(excluded)
    assert expected - actual == excluded
    entries = [entry for level, op in actual for entry in data[level][op].values()]
    assert len(entries) == 20 * (60 - len(excluded))
    assert sum(e["baseline_perf_us"] is None for e in entries) == missing
    for level, op in actual:
        assert set(data[level][op]) == {str(i) for i in range(1, 21)}
    assert data["_metadata"]["release_status"] == "pending_npu_revalidation"
    assert {p.stem for p in (TASKS / "metadata").glob("*.json")} == {
        "910b2",
        "910c",
        "950pr",
        "950dt",
    }
    assert set(MANIFEST["published_metadata_platforms"]) == {
        "910b2",
        "910c",
        "950pr",
        "950dt",
    }
    assert set(MANIFEST["unsupported_operators_by_platform"]["910c"]) == {
        "dynamic_mx_quant",
        "embedding_hash_table_lookup_or_insert",
    }
    assert set(MANIFEST["unsupported_operators_by_platform"]["910b2"]) == {
        "dynamic_mx_quant",
        "embedding_hash_table_lookup_or_insert"
    }


def test_moe_topk_membership_contract():
    spec = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                name="idx",
                index_gather={
                    "input": "x",
                    "dim": -1,
                    "value_topk": True,
                    "k_attr": "k",
                },
            )
        ]
    )
    x = torch.tensor([[1.0, 9.0, 9.0, 8.0]])
    assert validate_index_gather_outputs(
        spec, {"x": x}, {"k": 2}, torch.tensor([[2, 1]])
    )[0]
    assert not validate_index_gather_outputs(
        spec, {"x": x}, {"k": 2}, torch.tensor([[1, 0]])
    )[0]
    assert not validate_index_gather_outputs(
        spec, {"x": x}, {"k": 2}, torch.tensor([[1, 1]])
    )[0]
    assert not validate_index_gather_outputs(
        spec, {"x": x}, {}, torch.tensor([[2, 1]])
    )[0]


def test_topk_distinct_output_contract():
    spec = SimpleNamespace(
        outputs=[
            SimpleNamespace(name="values", index_gather=None),
            SimpleNamespace(
                name="indices",
                index_gather={
                    "input": "x",
                    "dim_attr": "dim",
                    "value_output": "values",
                    "distinct": True,
                },
            ),
        ]
    )
    x = torch.tensor([[9.0, 9.0, 1.0]])
    assert not validate_index_gather_outputs(
        spec,
        {"x": x},
        {"dim": -1},
        [torch.tensor([[9.0, 9.0]]), torch.tensor([[0, 0]])],
    )[0]
    assert validate_index_gather_outputs(
        spec,
        {"x": x},
        {"dim": -1},
        [torch.tensor([[9.0, 9.0]]), torch.tensor([[1, 0]])],
    )[0]


def test_unpublished_platform_does_not_borrow_910b2_metadata():
    store = BaselineStore(TASKS, hardware="910b1")
    store.load()
    assert store.get_perf("level1/exp", 1) == 0
    assert store.get_t_hw("level1/exp", 1) == 0
    assert not store.has_baseline("level1/exp", 1)


@pytest.mark.parametrize(
    "platform,hardware", [("950pr", "Ascend950PR"), ("950dt", "Ascend950DT")]
)
def test_950_metadata_loads_own_platform(platform, hardware):
    data = json.loads(
        (TASKS / "metadata" / f"{platform}.json").read_text(encoding="utf-8")
    )
    store = BaselineStore(TASKS, hardware=hardware)
    store.load()
    for level, ops in data.items():
        if not level.startswith("level"):
            continue
        for op, records in ops.items():
            for cid, entry in records.items():
                assert entry["baseline_perf_us"] > 0
                assert entry["t_hw_us"] > 0
                assert (
                    store.get_perf(f"{level}/{op}", int(cid))
                    == entry["baseline_perf_us"]
                )
                assert store.get_t_hw(f"{level}/{op}", int(cid)) == entry["t_hw_us"]
                assert store.has_baseline(f"{level}/{op}", int(cid))


@pytest.mark.parametrize("platform", ["910b2", "910c"])
def test_missing_baselines_are_explicit_theory_estimates(platform):
    data = json.loads(
        (TASKS / "metadata" / f"{platform}.json").read_text(encoding="utf-8")
    )
    declared = data["_metadata"]["imputed_baseline_cases"]
    assert len(declared) == len(set(declared))
    assert data["_metadata"]["imputed_baseline_count"] == len(declared)
    assert set(declared).isdisjoint(data["_metadata"]["task_revision_cases"])
    assert data["_metadata"]["imputed_baseline_formula"] == "3 * t_hw_us"
    assert MANIFEST["missing_baseline_policy"]["formula"] == "3 * t_hw_us"
    store = BaselineStore(TASKS, hardware=platform)
    store.load()
    for key in declared:
        level, op, cid = key.split("/")
        entry = data[level][op][cid]
        assert entry["t_hw_us"] > 0
        expected = (3 * Decimal(str(entry["t_hw_us"]))).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )
        assert Decimal(str(entry["baseline_perf_us"])) == expected
        assert store.get_perf(f"{level}/{op}", int(cid)) == entry["baseline_perf_us"]
        assert store.has_baseline(f"{level}/{op}", int(cid))
    for op in MANIFEST["unsupported_operators_by_platform"].get(platform, []):
        assert not any(key.split("/")[1] == op for key in declared)


def test_moe_finished_and_nan_follow_golden_semantics():
    import importlib.util

    path = TASKS / "level3/moe_gating_top_k_softmax/golden.py"
    module_spec = importlib.util.spec_from_file_location("season2_moe_golden", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    op = CannTaskLoader(str(TASKS)).get_task("level3/moe_gating_top_k_softmax")
    x = torch.tensor([[1.0, 9.0, 9.0], [5.0, 4.0, 3.0], [float("inf"), 0.0, 1.0]])
    finished = torch.tensor([False, True, False])
    outputs = module.moe_gating_top_k_softmax(x, finished, k=2)
    assert validate_index_gather_outputs(
        op, {"x": x, "finished": finished}, {"k": 2}, outputs
    )[0]
    wrong = list(outputs)
    wrong[1] = outputs[1].clone()
    wrong[1][1] = 0
    assert not validate_index_gather_outputs(
        op, {"x": x, "finished": finished}, {"k": 2}, wrong
    )[0]
    assert op.outputs[2].compare is True


@pytest.mark.parametrize(
    "platform,baseline_floor_us",
    [("910b2", 1.0), ("910c", 1.0), ("950pr", 0.5), ("950dt", 0.5)],
)
def test_revised_public_records_are_explicit_estimates(platform, baseline_floor_us):
    data = json.loads(
        (TASKS / "metadata" / f"{platform}.json").read_text(encoding="utf-8")
    )
    revised = data["_metadata"]["task_revision_cases"]
    assert len(revised) == len(set(revised)) == 182
    for key in revised:
        level, op, cid = key.split("/")
        assert 1 <= int(cid) <= 20
        entry = data[level][op][cid]
        assert entry["baseline_status"] == "model_estimate"
        assert entry["baseline_validation"] == "pending_npu_revalidation"
        assert entry["baseline_perf_us"] >= entry["t_hw_us"] > 0
        assert entry["baseline_perf_us"] >= baseline_floor_us
