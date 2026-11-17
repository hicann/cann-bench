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



from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import auto_pipeline.cli as pipeline_cli
import auto_pipeline.generator.ascend_arena.workspace as workspace_module
import ascend_arena_e2e.agent_base as agent_base
from auto_pipeline.converter.registry import available_converters, create_converter
from auto_pipeline.core import (
    AGENT_SUCCESS,
    Artifact,
    BenchmarkPipeline,
    CannBenchClient,
    CannBenchEvalResult,
    RunnerPrompt,
)
from auto_pipeline.generator.ascend_arena import AscendArenaAgent
from auto_pipeline.generator.opencode import OpenCodeAgent
from auto_pipeline.generator.registry import available_generators, available_runners, create_generator
from auto_pipeline.prompt.registry import build_case_material

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_SELECTOR = "bench_lab/ascend_arena_bench/expand_into_jagged_permute"
ARENA_OPERATOR_COUNT = 7


@pytest.fixture(autouse=True)
def _skip_hardware_workspace_initialization(monkeypatch, arena_workspace_source):
    monkeypatch.setattr(workspace_module, "_run_initial_evaluation", lambda workspace: None)


class FakeRunner:
    type = "fake"

    def run(self, prompt: RunnerPrompt) -> Artifact:
        prompt.output_dir.mkdir(parents=True, exist_ok=True)
        log_file = prompt.output_dir / "fake.log"
        log_file.write_text("fake runner completed\n", encoding="utf-8")
        marker = prompt.cwd / "expand_into_jagged_permute" / "op_kernel" / "auto_pipeline_marker.txt"
        marker.write_text("generated\n", encoding="utf-8")
        return Artifact(status=AGENT_SUCCESS, workdir=prompt.output_dir, log_file=log_file)


class FakeEvalClient(CannBenchClient):
    def eval_submission(
        self,
        *,
        bench_name,
        source_dir,
        task_selector,
        reports_dir,
        device_id=None,
        extra_args=None,
    ):
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        report = reports_dir / "fake_report.json"
        report.write_text("{}\n", encoding="utf-8")
        return CannBenchEvalResult(
            returncode=0,
            command=["fake-eval", str(source_dir)],
            reports_dir=reports_dir,
            report_files=[report],
        )


def test_ascend_arena_case_material_and_eval_command(tmp_path):
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("ascend_arena", TASK_SELECTOR)
    material = build_case_material(case)

    assert case.operator == "expand_into_jagged_permute"
    assert case.files["proto"].name == "proto.yaml"
    assert case.files["profile_cases"].name == "profile_cases.json"
    assert len(case.metadata["case_preview"]) == 3
    assert material.op_name == "expand_into_jagged_permute"
    assert {item.target_name for item in material.task_files} >= {
        "api_desc.md",
        "reference.py",
        "cases/proto.yaml",
        "cases/profile_cases.json",
        "binding/pybind.cpp",
    }

    command = client.build_eval_command(
        bench_name="ascend_arena",
        source_dir=tmp_path / "submission",
        task_selector=TASK_SELECTOR,
        reports_dir=tmp_path / "reports",
        device_id=2,
    )
    assert command[1:3] == ["-m", "kernel_eval.optimize"]
    assert command[command.index("--operator") + 1] == "expand_into_jagged_permute"
    assert command[command.index("--device-id") + 1] == "2"
    assert "--bench-name" not in command

    env = client._build_eval_env(
        source_dir=tmp_path / "submission",
        reports_dir=tmp_path / "reports",
        isolate_python=False,
    )
    assert "PIP_TARGET" not in env
    assert not (tmp_path / "reports").exists()


def test_ascend_arena_pipeline_uses_cannbench_workspace_layout(tmp_path):
    client = FakeEvalClient(REPO_ROOT)
    case = client.load_case("ascend_arena", TASK_SELECTOR)
    generator = AscendArenaAgent(
        cann_bench_root=REPO_ROOT,
        runner=FakeRunner(),
        device_id=0,
    )
    converter = create_converter("ascend-arena", "ascend_arena", {})
    task_root = tmp_path / "run" / "expand_into_jagged_permute"

    result = BenchmarkPipeline(bench_name="ascend_arena", client=client).run_case(
        selector=TASK_SELECTOR,
        generator=generator,
        converter=converter,
        workdir=task_root / "work",
        submission_dir=task_root / "submission",
        reports_dir=task_root / "work" / "kernel_eval",
        timeout_sec=10,
        device_id=0,
    )

    assert result.ok
    assert (task_root / "work" / "expand_into_jagged_permute" / "op_host").is_dir()
    assert (task_root / "work" / "artifact" / "fake.log").is_file()
    assert (task_root / "work" / "validate.sh").is_file()
    assert not (task_root / "work" / "agent_logs").exists()
    assert (task_root / "custom" / "expand_into_jagged_permute" / "op_kernel" / "auto_pipeline_marker.txt").is_file()
    assert (task_root / "submission" / "expand_into_jagged_permute" / "op_host").is_dir()
    assert (task_root / "work" / "kernel_eval" / "fake_report.json").is_file()
    assert result.generated_artifact.files["source_dir"] == task_root / "custom" / "expand_into_jagged_permute"
    assert result.submission.source_dir == task_root / "submission"


def test_ascend_arena_registries_are_available():
    assert "ascend-arena" in available_generators()
    assert available_runners() == ["opencode"]
    assert "ascend-arena -> ascend-arena" in available_converters()
    generator = create_generator(
        "ascend_arena",
        {"repo_root": REPO_ROOT, "runner": "opencode", "model": "test-model"},
    )
    assert isinstance(generator, AscendArenaAgent)
    assert isinstance(generator.runner, OpenCodeAgent)
    assert generator.runner.model == "test-model"


def test_prepare_command_creates_agent_independent_workspace(tmp_path, capsys):
    output = tmp_path / "manual_workspace"
    argv = [
        "prepare",
        "--operator",
        "expand_into_jagged_permute",
        "--output",
        str(output),
        "--device-id",
        "3",
    ]

    assert pipeline_cli.main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    task_root = output / "expand_into_jagged_permute"
    workdir = task_root / "work"

    assert payload["status"] == "prepared"
    assert payload["workspace"] == str(workdir)
    assert payload["source_dir"] == str(workdir / "expand_into_jagged_permute")
    assert (workdir / "expand_into_jagged_permute" / "op_host").is_dir()
    assert (workdir / "expand_into_jagged_permute" / "op_kernel").is_dir()
    assert (workdir / "artifact").is_dir()
    assert (workdir / "workspace.json").is_file()
    assert not (workdir / "AGENTS.md").exists()
    assert not (workdir / "REQUIRE.md").exists()
    assert not (workdir / "cases" / "complete_cases.json").exists()
    assert not (workdir / "cases" / "profile_cases.json").exists()
    validate = (workdir / "validate.sh").read_text(encoding="utf-8")
    assert 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"' in validate
    assert 'CANN_BENCH_ROOT="${CANN_BENCH_ROOT:-' in validate
    assert '--source-dir "$SCRIPT_DIR/expand_into_jagged_permute"' in validate
    assert '--task-dir "$CANN_BENCH_ROOT/bench_lab/ascend_arena_bench/expand_into_jagged_permute"' in validate
    assert '--reports-dir "$SCRIPT_DIR/kernel_eval"' in validate
    assert '--baseline-dir "$SCRIPT_DIR/kernel_eval/init_results"' in validate
    assert 'if [[ "${1:-}" == "init" ]]; then MODE_ARGS+=(--init-only); shift; fi' in validate
    assert validate.endswith(' "$@"\n')
    assert 'ASCEND_ARENA_DEVICE_ID="${ASCEND_ARENA_DEVICE_ID:-3}"' in validate
    assert '--device-id "$ASCEND_ARENA_DEVICE_ID"' in validate
    assert 'if [[ "${1:-}" == "only-html" ]]' in validate
    assert 'python -m kernel_eval.optimization_trajectory "$SCRIPT_DIR/kernel_eval"' in validate
    assert not (task_root / "custom").exists()
    assert not (task_root / "submission").exists()
    assert not (workdir / "kernel_eval").exists()

    marker = workdir / "expand_into_jagged_permute" / "manual_edit.txt"
    marker.write_text("keep unless forced\n", encoding="utf-8")
    assert pipeline_cli.main(argv) == 1
    assert marker.is_file()
    assert "workspace already exists" in capsys.readouterr().err

    assert pipeline_cli.main([*argv, "--force"]) == 0
    capsys.readouterr()
    assert not marker.exists()


@pytest.mark.parametrize("timed_out,exitcode", [(False, 1), (True, 124)])
def test_prepare_command_runs_selected_agent_and_returns_its_status(tmp_path, monkeypatch, capsys, timed_out, exitcode):
    output = tmp_path / "agent_workspace"
    calls = {}

    class FakeAgent:
        def preflight(self, *, docker=False):
            pass

        name = "fake-agent"

        def build_docker_command(self, workspace, command, *, image, device_id, environment):
            calls["initial_workspace"] = workspace
            calls["initial_command"] = command
            calls["initial_image"] = image
            calls["initial_device_id"] = device_id
            calls["initial_environment"] = environment
            return ["docker", "init"]

        def run(self, workspace, *, prompt, device_id, docker_image):
            calls["workspace"] = workspace
            calls["prompt"] = prompt
            calls["device_id"] = device_id
            calls["docker_image"] = docker_image
            return SimpleNamespace(
                ok=False,
                timed_out=timed_out,
                to_dict=lambda: {"agent": self.name, "returncode": 124 if timed_out else 9, "ok": False},
            )

    def fake_agent_from_args(args):
        calls["agent"] = args.agent
        return FakeAgent()

    monkeypatch.setattr(agent_base, "agent_from_args", fake_agent_from_args)
    monkeypatch.setattr(
        workspace_module,
        "_run_initial_evaluation",
        lambda workspace, command=None: calls.update(evaluation_command=command),
    )

    assert (
        pipeline_cli.main(
            [
                "prepare",
                "--operator",
                "expand_into_jagged_permute",
                "--output",
                str(output),
                "--agent",
                "fake-agent",
                "--agent-prompt",
                "Optimize this operator.",
                "--docker",
                "--docker-image",
                "test-image",
            ]
        )
        == exitcode
    )

    payload = json.loads(capsys.readouterr().out)
    workdir = output / "expand_into_jagged_permute" / "work"
    assert calls == {
        "agent": "fake-agent",
        "initial_workspace": workdir,
        "initial_command": ["./validate.sh", "init"],
        "initial_image": "test-image",
        "initial_device_id": 0,
        "initial_environment": {},
        "evaluation_command": ["docker", "init"],
        "workspace": workdir,
        "prompt": "Optimize this operator.",
        "device_id": 0,
        "docker_image": "test-image",
    }
    assert payload["agent_run"] == {"agent": "fake-agent", "returncode": 124 if timed_out else 9, "ok": False}
    assert not (workdir / "AGENTS.md").exists()


def test_generated_validate_script_executes_and_writes_inside_workspace(tmp_path, capsys):
    output = tmp_path / "manual workspace"
    assert (
        pipeline_cli.main(
            [
                "prepare",
                "--operator",
                "expand_into_jagged_permute",
                "--output",
                str(output),
                "--device-id",
                "4",
            ]
        )
        == 0
    )
    capsys.readouterr()
    workdir = output / "expand_into_jagged_permute" / "work"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    capture = tmp_path / "python-args.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"${1:-}\" == --version ]]; then echo 'Python 3.10.20'; exit 0; fi\n"
        'printf \'%s\\n\' "$@" > "${FAKE_PYTHON_ARGS}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_PYTHON_ARGS": str(capture),
    }

    completed = subprocess.run(
        [str(workdir / "validate.sh")],
        cwd=workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert completed.returncode == 0, completed.stdout
    args = capture.read_text(encoding="utf-8").splitlines()
    assert args[:3] == ["-m", "kernel_eval.optimize", "--source-dir"]
    assert args[3] == str(workdir / "expand_into_jagged_permute")
    assert args[args.index("--task-dir") + 1] == str(REPO_ROOT / "bench_lab/ascend_arena_bench/expand_into_jagged_permute")
    assert args[args.index("--reports-dir") + 1] == str(workdir / "kernel_eval")
    assert args[args.index("--baseline-dir") + 1] == str(workdir / "kernel_eval" / "init_results")
    assert args[args.index("--device-id") + 1] == "4"
    assert (workdir / "kernel_eval").is_dir()

    initialized = subprocess.run(
        [str(workdir / "validate.sh"), "init"],
        cwd=workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert initialized.returncode == 0, initialized.stdout
    assert "--init-only" in capture.read_text(encoding="utf-8").splitlines()


def test_generated_validate_script_only_html_skips_evaluation(tmp_path, capsys):
    output = tmp_path / "manual workspace"
    assert (
        pipeline_cli.main(
            [
                "prepare",
                "--operator",
                "expand_into_jagged_permute",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    capsys.readouterr()
    workdir = output / "expand_into_jagged_permute" / "work"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    capture = tmp_path / "python-args.txt"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n" 'printf \'%s\\n\' "$@" > "${FAKE_PYTHON_ARGS}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_PYTHON_ARGS": str(capture),
    }

    completed = subprocess.run(
        [str(workdir / "validate.sh"), "only-html"],
        cwd=workdir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert completed.returncode == 0, completed.stdout
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "kernel_eval.optimization_trajectory",
        str(workdir / "kernel_eval"),
    ]


def test_arena_assets_match_ops_registry():
    e2e_root = REPO_ROOT / "src/ascend_arena_e2e"
    ops = json.loads((e2e_root / "ops.json").read_text(encoding="utf-8"))
    assert isinstance(ops, dict)
    required_metadata = {
        "op",
        "compile-repo",
        "compile-family",
        "compile-op-name",
        "support-devices",
        "is_fewshot",
    }
    assert all(required_metadata <= set(metadata) for metadata in ops.values())
    assert all(
        metadata["support-devices"] == ["ascend910b", "ascend910_93"]
        for metadata in ops.values()
    )
    task_root = REPO_ROOT / "bench_lab/ascend_arena_bench"
    operator_dirs = [path for path in task_root.iterdir() if path.is_dir()]
    assert all((path / "cases/proto.yaml").is_file() for path in operator_dirs)
    manifest = json.loads((e2e_root / "dependency/baselines.json").read_text())
    assert set(manifest["operators"]) == set(ops)
    assert all((path / path.name).is_symlink() for path in operator_dirs)
    assert not list(task_root.glob("*/.source"))
    assert set(ops) <= {
        proto.parent.parent.name
        for proto in task_root.glob("*/cases/proto.yaml")
    }
    for operator in ops:
        assert {Path(name).parts[0] for name in manifest["operators"][operator]} >= {"op_host", "op_kernel"}
        assert (task_root / operator / operator).resolve() == (e2e_root / "dependency/baselines" / operator).resolve()


def test_arena_skeleton_uses_shared_pinned_opbase(tmp_path):
    from kernel_eval.optimize import _copy_dependency_tree
    bundles = REPO_ROOT / "src/ascend_arena_e2e/dependency"
    _copy_dependency_tree(bundles, tmp_path / "extracted", compile_repo="ops-skeleton")
    skeleton = tmp_path / "extracted/ops-skeleton"
    assert not (skeleton / ".git").exists()
    assert (skeleton / "cmake/third_party/opbase.cmake").is_file()
    opbase = tmp_path / "extracted/ops-base"
    assert subprocess.check_output(["git", "-C", str(opbase), "rev-parse", "HEAD"], text=True).strip() == (
        "0c5e25790f67e2c6f60c1df36d74d28f0e08ecfc"
    )
    assert (opbase / ".git").exists()
    required = (
        "LICENSE",
        "include/op_common/op_host/util/opbase_export.h",
        "pkg_inc/op_common/atvoss/reduce/reduce_tiling.h",
        "src/op_common/atvoss/broadcast/broadcast_tiling.cpp",
        "src/op_common/atvoss/broadcast/broadcast_tiling_base.cpp",
        "src/op_common/atvoss/broadcast/broadcast_tiling_noncontiguous.cpp",
        "src/op_common/atvoss/elewise/elewise_tiling.cpp",
        "src/op_common/atvoss/reduce/reduce_tiling.cpp",
        "src/op_common/log/log.cpp",
        "src/op_common/op_host/infershape_broadcast_util.cpp",
        "src/op_common/op_host/infershape_elewise_util.cpp",
        "src/op_common/op_host/infershape_reduce_util.cpp",
        "src/op_common/op_host/tiling_util.cpp",
        "src/op_common/op_host/util/fp16.cpp",
        "src/op_common/op_host/util/math_util.cpp",
        "src/op_common/op_host/util/shape_util.cpp",
    )
    assert not [path for path in required if not (opbase / path).is_file()]


def test_preflight_reports_reproducible_install_commands(monkeypatch):
    from kernel_eval import optimize

    monkeypatch.setattr(optimize.importlib.util, "find_spec", lambda name: None if name == "pybind11" else object())
    monkeypatch.setattr(optimize.shutil, "which", lambda name: None if name == "cmake" else "/usr/bin/" + name)

    with pytest.raises(RuntimeError) as exc_info:
        optimize._preflight_dependencies(
            project_root=REPO_ROOT,
            dependency_root=REPO_ROOT / "src/ascend_arena_e2e/dependency",
            compile_repo="ops-skeleton",
            env_script=Path("/opt/ascend/set_env.sh"),
        )

    message = str(exc_info.value)
    assert "pybind11" in message
    assert "cmake" in message
    assert "python -m pip install -r" in message
    assert "ops-skeleton/install_deps.sh" not in message


def _make_optimize_main_fixture(tmp_path):
    dependency_root = tmp_path / "dependency"
    (dependency_root / "ops-skeleton").mkdir(parents=True)
    (dependency_root / "ops-skeleton/CMakeLists.txt").write_text("# build entry\n")
    original_op = tmp_path / "bench" / "angle_v2" / "angle_v2"
    (original_op / "op_host").mkdir(parents=True)
    (original_op / "op_kernel").mkdir(parents=True)
    (original_op / "op_kernel" / "kernel.cpp").write_text("original\n", encoding="utf-8")

    task_root = tmp_path / "bench"
    benchmark_op = task_root / "angle_v2"
    (benchmark_op / "cases").mkdir(parents=True)
    (benchmark_op / "cases" / "proto.yaml").write_text("name: angle_v2\n", encoding="utf-8")
    (benchmark_op / "reference.py").write_text("def reference(): pass\n", encoding="utf-8")

    source_dir = tmp_path / "source"
    (source_dir / "op_host").mkdir(parents=True)
    (source_dir / "op_kernel").mkdir(parents=True)
    (source_dir / "op_kernel" / "kernel.cpp").write_text("candidate\n", encoding="utf-8")

    ops_json = tmp_path / "ops.json"
    ops_json.write_text(
        json.dumps(
            {
                "angle_v2": {
                    "compile-repo": "ops-skeleton",
                    "compile-family": "math",
                    "compile-op-name": "angle_v2",
                }
            }
        ),
        encoding="utf-8",
    )
    return dependency_root, task_root, source_dir, ops_json, original_op


def _write_initialized_baseline(reports_root: Path, original_op: Path) -> Path:
    baseline_dir = reports_root / "init_results"
    baseline_dir.mkdir(parents=True)
    (baseline_dir / "result.json").write_text(
        json.dumps({"case": [{"task_duration_us": 2.0}]}), encoding="utf-8"
    )
    (baseline_dir / "status.json").write_text(
        json.dumps({"success": True, "stage": "baseline", "source_dir": str(original_op.resolve())}),
        encoding="utf-8",
    )
    return baseline_dir


def test_optimize_reuses_baseline_when_candidate_correctness_fails(tmp_path, monkeypatch):
    from kernel_eval import optimize

    dependency_root, task_root, source_dir, ops_json, original_op = _make_optimize_main_fixture(tmp_path)
    calls = []
    report_args = []

    def fake_run_stage(**kwargs):
        calls.append(kwargs)
        stage_dir = kwargs["result_path"].parent
        failed = len(calls) == 1
        build = SimpleNamespace(success=not failed, log_path=stage_dir / "compile.log")
        profile = SimpleNamespace(
            success=not failed,
            log_path=stage_dir / "profile.log",
            message="failed" if failed else "ok",
        )
        return SimpleNamespace(success=not failed, build_result=build, profile_result=profile), {}

    monkeypatch.setattr(optimize, "_preflight_dependencies", lambda **kwargs: None)
    monkeypatch.setattr(optimize, "_resolve_env_script", lambda explicit: None)
    monkeypatch.setattr(optimize, "_run_stage", fake_run_stage)
    monkeypatch.setattr(optimize, "_build_report", lambda **kwargs: report_args.append(kwargs))

    reports_root = tmp_path / "reports"
    baseline_dir = _write_initialized_baseline(reports_root, original_op)
    assert (
        optimize.main(
            [
                "--source-dir",
                str(source_dir),
                "--task-dir",
                str(task_root),
                "--operator",
                "angle_v2",
                "--dependency-root",
                str(dependency_root),
                "--ops-json",
                str(ops_json),
                "--reports-dir",
                str(reports_root),
                "--baseline-dir",
                str(baseline_dir),
            ]
        )
        == 1
    )

    assert len(calls) == 1
    assert calls[0]["run_root"].name == "candidate"
    run_dir = next(path for path in reports_root.iterdir() if path.name.startswith("ascend_arena_"))
    assert not (run_dir / "init_results").exists()
    status = json.loads((run_dir / "results" / "status.json").read_text(encoding="utf-8"))
    assert status["success"] is False
    assert status["source_dir"] == str(source_dir.resolve())
    assert report_args[0]["baseline_status"]["success"] is True
    assert report_args[0]["baseline_status"]["source_dir"] == str(original_op.resolve())


def test_optimize_explicit_baseline_requires_initialization(tmp_path):
    from kernel_eval import optimize

    dependency_root, task_root, source_dir, ops_json, _ = _make_optimize_main_fixture(tmp_path)
    with pytest.raises(SystemExit, match="run validate.sh init"):
        optimize.main(
            [
                "--source-dir", str(source_dir),
                "--task-dir", str(task_root),
                "--operator", "angle_v2",
                "--dependency-root", str(dependency_root),
                "--ops-json", str(ops_json),
                "--reports-dir", str(tmp_path / "reports"),
                "--baseline-dir", str(tmp_path / "fixed_baseline"),
            ]
        )


def test_optimize_parser_supports_no_clean_sandbox():
    from kernel_eval.optimize import build_parser

    args = build_parser().parse_args(
        [
            "--source-dir",
            "/tmp/source",
            "--task-dir",
            "/tmp/task",
            "--operator",
            "angle_v2",
            "--no-clean-sandbox",
        ]
    )

    assert args.no_clean_sandbox is True


def test_optimize_parser_supports_no_perf():
    from kernel_eval.optimize import build_parser

    args = build_parser().parse_args(
        [
            "--source-dir",
            "/tmp/source",
            "--task-dir",
            "/tmp/task",
            "--operator",
            "angle_v2",
            "--no-perf",
        ]
    )

    assert args.no_perf is True


def test_optimize_parser_supports_initialized_baseline():
    from kernel_eval.optimize import build_parser

    args = build_parser().parse_args(["--init-only", "--baseline-dir", "/tmp/init_results"])

    assert args.init_only is True
    assert args.baseline_dir == "/tmp/init_results"


@pytest.mark.parametrize('build_success,stage_success', [(True, True), (False, False), (True, False)])
def test_optimize_no_perf_skips_baseline(tmp_path, monkeypatch, build_success, stage_success):
    from kernel_eval import optimize

    dependency_root, task_root, source_dir, ops_json, original_op = _make_optimize_main_fixture(tmp_path)
    calls = []
    report_args = []

    def fake_run_stage(**kwargs):
        calls.append(kwargs)
        stage_dir = kwargs["result_path"].parent
        build = SimpleNamespace(success=build_success, log_path=stage_dir / "compile.log")
        profile = SimpleNamespace(
            success=stage_success,
            log_path=stage_dir / "profile.log",
            message="skipped_performance_disabled",
        )
        return SimpleNamespace(success=stage_success, build_result=build, profile_result=profile), {}

    monkeypatch.setattr(optimize, "_preflight_dependencies", lambda **kwargs: None)
    monkeypatch.setattr(optimize, "_resolve_env_script", lambda explicit: None)
    monkeypatch.setattr(optimize, "_run_stage", fake_run_stage)
    monkeypatch.setattr(optimize, "_build_report", lambda **kwargs: report_args.append(kwargs))

    reports_root = tmp_path / "reports"
    assert (
        optimize.main(
            [
                "--source-dir",
                str(source_dir),
                "--task-dir",
                str(task_root),
                "--operator",
                "angle_v2",
                "--dependency-root",
                str(dependency_root),
                "--ops-json",
                str(ops_json),
                "--reports-dir",
                str(reports_root),
                "--no-perf",
            ]
        )
        == (0 if stage_success else 1)
    )

    assert len(calls) == 1
    assert calls[0]["profile_performance"] is False
    assert report_args[0]["candidate_status"]["success"] is stage_success
    run_dir = next(reports_root.iterdir())
    assert not (run_dir / "init_results").exists()
    assert report_args[0]["baseline_status"] == {
        "success": True,
        "stage": "baseline",
        "skipped": True,
        "reason": "performance_disabled",
        "source_dir": str(original_op.resolve()),
    }


def test_optimize_no_clean_sandbox_retains_debug_workspace(tmp_path, monkeypatch):
    from kernel_eval import optimize

    dependency_root, task_root, source_dir, ops_json, original_op = _make_optimize_main_fixture(tmp_path)

    def fake_run_stage(**kwargs):
        stage_dir = kwargs["result_path"].parent
        build = SimpleNamespace(success=True, log_path=stage_dir / "compile.log")
        profile = SimpleNamespace(success=True, log_path=stage_dir / "profile.log", message="ok")
        return SimpleNamespace(success=True, build_result=build, profile_result=profile), {
            "case": [{"task_duration_us": 1.0}]
        }

    monkeypatch.setattr(optimize, "_preflight_dependencies", lambda **kwargs: None)
    monkeypatch.setattr(optimize, "_resolve_env_script", lambda explicit: None)
    monkeypatch.setattr(optimize, "_run_stage", fake_run_stage)
    monkeypatch.setattr(optimize, "_build_report", lambda **kwargs: None)

    reports_root = tmp_path / "reports"
    baseline_dir = _write_initialized_baseline(reports_root, original_op)
    assert (
        optimize.main(
            [
                "--source-dir",
                str(source_dir),
                "--task-dir",
                str(task_root),
                "--operator",
                "angle_v2",
                "--dependency-root",
                str(dependency_root),
                "--ops-json",
                str(ops_json),
                "--reports-dir",
                str(reports_root),
                "--baseline-dir",
                str(baseline_dir),
                "--no-clean-sandbox",
            ]
        )
        == 0
    )

    sandboxes = list((reports_root / ".sandboxes").iterdir())
    assert len(sandboxes) == 1
    assert (sandboxes[0] / "dependency" / "ops-skeleton").is_dir()


def test_optimize_initializes_baseline_once_then_runs_only_candidate(tmp_path, monkeypatch):
    from kernel_eval import optimize

    dependency_root, task_root, source_dir, ops_json, original_op = _make_optimize_main_fixture(tmp_path)
    calls = []

    def fake_run_stage(**kwargs):
        calls.append(kwargs)
        stage_dir = kwargs["result_path"].parent
        build = SimpleNamespace(success=True, log_path=stage_dir / "compile.log")
        profile = SimpleNamespace(success=True, log_path=stage_dir / "profile.log", message="ok")
        return SimpleNamespace(success=True, build_result=build, profile_result=profile), {
            "case": [{"task_duration_us": 1.0}]
        }

    monkeypatch.setattr(optimize, "_preflight_dependencies", lambda **kwargs: None)
    monkeypatch.setattr(optimize, "_resolve_env_script", lambda explicit: None)
    monkeypatch.setattr(optimize, "_run_stage", fake_run_stage)
    monkeypatch.setattr(optimize, "_build_report", lambda **kwargs: None)

    reports_root = tmp_path / "reports"
    baseline_dir = reports_root / "init_results"
    common_args = [
        "--source-dir", str(source_dir),
        "--task-dir", str(task_root),
        "--operator", "angle_v2",
        "--dependency-root", str(dependency_root),
        "--ops-json", str(ops_json),
        "--reports-dir", str(reports_root),
        "--baseline-dir", str(baseline_dir),
    ]
    assert (
        optimize.main([*common_args, "--init-only"])
        == 0
    )
    assert (
        optimize.main(common_args)
        == 0
    )

    assert [call["run_root"].name for call in calls] == ["init", "candidate"]
    assert all(call["stages"] == frozenset({"smoke", "complete", "profile"}) for call in calls)
    run_dir = next(path for path in reports_root.iterdir() if path.name.startswith("ascend_arena_"))
    candidate_status = json.loads((run_dir / "results" / "status.json").read_text(encoding="utf-8"))
    baseline_status = json.loads((baseline_dir / "status.json").read_text(encoding="utf-8"))
    assert candidate_status["source_dir"] == str(source_dir.resolve())
    assert baseline_status["source_dir"] == str(original_op.resolve())
    assert not (run_dir / "init_results").exists()
    assert not (reports_root / ".sandboxes").exists()


def test_optimize_writes_candidate_status_when_interrupted_by_sigterm(tmp_path, monkeypatch):
    from kernel_eval import optimize

    dependency_root, task_root, source_dir, ops_json, original_op = _make_optimize_main_fixture(tmp_path)
    reports_root = tmp_path / "reports"
    baseline_dir = _write_initialized_baseline(reports_root, original_op)

    def interrupt_stage(**kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM handler did not interrupt evaluation")

    monkeypatch.setattr(optimize, "_preflight_dependencies", lambda **kwargs: None)
    monkeypatch.setattr(optimize, "_resolve_env_script", lambda explicit: None)
    monkeypatch.setattr(optimize, "_run_stage", interrupt_stage)

    result = optimize.main([
        "--source-dir", str(source_dir),
        "--task-dir", str(task_root),
        "--operator", "angle_v2",
        "--dependency-root", str(dependency_root),
        "--ops-json", str(ops_json),
        "--reports-dir", str(reports_root),
        "--baseline-dir", str(baseline_dir),
    ])

    run_dir = next(path for path in reports_root.iterdir() if path.name.startswith("ascend_arena_"))
    status = json.loads((run_dir / "results" / "status.json").read_text(encoding="utf-8"))
    assert result == 143
    assert status["success"] is False
    assert status["stage"] == "candidate"
    assert status["interrupted"] is True
    assert status["signal"] == "SIGTERM"
    assert status["returncode"] == 143
    assert status["message"] == "evaluation interrupted by SIGTERM"


def test_stage_artifacts_keep_profile_and_correctness_results(tmp_path):
    from kernel_eval.optimize import _copy_profile_artifacts, _stage_artifact_paths

    stage_dir = tmp_path / "results"
    result_path = stage_dir / "result.json"
    artifacts = _stage_artifact_paths(stage_dir, result_path)

    assert artifacts["BENCHMARK_SMOKE_CORRECTNESS_RESULT_JSON_PATH"] == stage_dir / "smoke-correctness_result.json"
    assert (
        artifacts["BENCHMARK_COMPLETE_CORRECTNESS_RESULT_JSON_PATH"] == stage_dir / "complete-correctness_result.json"
    )
    assert artifacts["BENCHMARK_PROFILE_CORRECTNESS_RESULT_JSON_PATH"] == stage_dir / "profile-correctness_result.json"

    profile_source = tmp_path / "sandbox" / "profile"
    (profile_source / "case-0").mkdir(parents=True)
    (profile_source / "case-0" / "trace.json").write_text("trace\n", encoding="utf-8")
    stale_profile = stage_dir / "profile"
    stale_profile.mkdir(parents=True)
    (stale_profile / "stale.txt").write_text("stale\n", encoding="utf-8")

    _copy_profile_artifacts(profile_source, stage_dir)

    assert (stage_dir / "profile" / "case-0" / "trace.json").read_text(encoding="utf-8") == "trace\n"
    assert not (stage_dir / "profile" / "stale.txt").exists()


def test_success_stage_cleanup_removes_only_known_logs(tmp_path):
    from kernel_eval.optimize import _STAGE_LOG_NAMES, _cleanup_success_logs

    stage_dir = tmp_path / "results"
    stage_dir.mkdir()
    for name in (*_STAGE_LOG_NAMES, "keep.log"):
        (stage_dir / name).write_text("log\n", encoding="utf-8")

    _cleanup_success_logs(stage_dir, {"success": True})

    assert all(not (stage_dir / name).exists() for name in _STAGE_LOG_NAMES)
    assert (stage_dir / "keep.log").is_file()

    for name in _STAGE_LOG_NAMES:
        (stage_dir / name).write_text("failed log\n", encoding="utf-8")
    _cleanup_success_logs(stage_dir, {"success": False})
    assert all((stage_dir / name).is_file() for name in _STAGE_LOG_NAMES)


def test_arena_report_omits_sol_score_note(tmp_path):
    from kernel_eval.optimize import _build_report

    run_name = "ascend_arena_angle_v2_test"
    _build_report(
        output_dir=tmp_path,
        run_name=run_name,
        operator="angle_v2",
        baseline={"case-0": [{"task_duration_us": 20.0}]},
        candidate={"case-0": [{"task_duration_us": 10.0}]},
        baseline_status={"success": True},
        candidate_status={"success": True},
        device_id=0,
    )

    report_text = "\n".join(
        (tmp_path / f"{run_name}{suffix}").read_text(encoding="utf-8") for suffix in (".json", ".md", ".html")
    )
    assert "Ascend Arena 未提供 T_HW" not in report_text
    assert "score_note" not in report_text


def test_failed_candidate_report_uses_na_for_missing_performance(tmp_path):
    from kernel_eval.optimize import _build_report

    run_name = "ascend_arena_angle_v2_failed"
    _build_report(
        output_dir=tmp_path,
        run_name=run_name,
        operator="angle_v2",
        baseline={"case-0": [{"task_duration_us": 20.0}]},
        candidate={},
        baseline_status={"success": True},
        candidate_status={"success": False},
        device_id=0,
    )

    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))
    report = json.loads((tmp_path / f"{run_name}.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / f"{run_name}.md").read_text(encoding="utf-8")
    html_report = (tmp_path / f"{run_name}.html").read_text(encoding="utf-8")

    assert comparison["cases"][0]["candidate_us"] is None
    assert comparison["cases"][0]["speedup"] is None
    assert comparison["performance_status"] == "failed"
    assert report["summary"]["average_speedup"] is None
    assert "| N/A | N/A |" in markdown
    assert ">N/A</td><td>N/A</td>" in html_report


def test_no_perf_report_summarizes_correctness_without_fake_performance(tmp_path):
    from kernel_eval.optimize import _build_report

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    stage_cases = {"smoke": [True, True], "complete": [True, True, True], "profile": [True]}
    for stage, passed_values in stage_cases.items():
        payload = {
            "success": all(passed_values),
            "case_results": [
                {"case_id": f"{stage}-{index}", "passed": passed} for index, passed in enumerate(passed_values)
            ],
        }
        (results_dir / f"{stage}-correctness_result.json").write_text(json.dumps(payload), encoding="utf-8")

    run_name = "ascend_arena_angle_v2_correctness_only"
    _build_report(
        output_dir=tmp_path,
        run_name=run_name,
        operator="angle_v2",
        baseline={},
        candidate={},
        baseline_status={"success": True, "skipped": True, "reason": "performance_disabled"},
        candidate_status={"success": True},
        device_id=0,
    )

    report = json.loads((tmp_path / f"{run_name}.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / f"{run_name}.md").read_text(encoding="utf-8")
    html_report = (tmp_path / f"{run_name}.html").read_text(encoding="utf-8")
    comparison = json.loads((tmp_path / "comparison.json").read_text(encoding="utf-8"))

    assert report["evaluation_mode"] == "correctness_only"
    assert report["summary"] == {
        "total_cases": 6,
        "passed_cases": 6,
        "failed_cases": 0,
        "pass_rate": 1.0,
        "compared_cases": None,
        "average_speedup": None,
    }
    assert [item["stage"] for item in report["correctness_stages"]] == ["smoke", "complete", "profile"]
    assert comparison["performance_status"] == "not_run"
    assert comparison["metric"] is None
    assert "| 评测模式 | correctness_only |" in markdown
    assert "| 平均加速比 | 未执行 |" in markdown
    assert "| 通过率 | 0.00% |" not in markdown
    assert "性能对比：未执行" in html_report


def test_source_diff_is_embedded_and_rendered_in_reports(tmp_path):
    from kernel_eval.optimize import _build_report, _collect_source_diff

    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    (baseline / "op_kernel").mkdir(parents=True)
    (candidate / "op_kernel").mkdir(parents=True)
    (baseline / "op_kernel" / "kernel.cpp").write_text("before\n", encoding="utf-8")
    (baseline / "op_kernel" / "removed.cpp").write_text("removed\n", encoding="utf-8")
    (baseline / ".source").write_text("internal metadata\n", encoding="utf-8")
    (candidate / "op_kernel" / "kernel.cpp").write_text("after\n", encoding="utf-8")
    (candidate / "op_kernel" / "added.cpp").write_text("added\n", encoding="utf-8")

    source_diff = _collect_source_diff(baseline_dir=baseline, candidate_dir=candidate)

    assert source_diff["changed_file_count"] == 3
    assert source_diff["added_files"] == ["op_kernel/added.cpp"]
    assert source_diff["deleted_files"] == ["op_kernel/removed.cpp"]
    assert source_diff["modified_files"] == ["op_kernel/kernel.cpp"]
    assert "-before" in source_diff["unified_diff"]
    assert "+after" in source_diff["unified_diff"]
    assert not (tmp_path / "source_diff.patch").exists()

    run_name = "ascend_arena_angle_v2_diff_test"
    _build_report(
        output_dir=tmp_path,
        run_name=run_name,
        operator="angle_v2",
        baseline={"case-0": [{"task_duration_us": 20.0}]},
        candidate={"case-0": [{"task_duration_us": 10.0}]},
        baseline_status={"success": True},
        candidate_status={"success": True},
        device_id=0,
        source_diff=source_diff,
    )

    report = json.loads((tmp_path / f"{run_name}.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / f"{run_name}.md").read_text(encoding="utf-8")
    html = (tmp_path / f"{run_name}.html").read_text(encoding="utf-8")
    assert report["source_diff"] == source_diff
    assert "## 优化前后源码 Diff" in markdown
    assert "```diff" in markdown
    assert "-before" in markdown
    assert "source_diff.patch" not in markdown
    assert "优化前后源码 Diff" in html
    assert "查看完整 Diff" in html
    assert "source_diff.patch" not in html
    trajectory = tmp_path.parent / "optimization_trajectory.html"
    assert trajectory.is_file()
    assert "Kernel Optimization Trajectory" in trajectory.read_text(encoding="utf-8")


def test_optimize_sandbox_copies_only_compile_repo(tmp_path):
    from kernel_eval.optimize import _copy_dependency_tree

    dependency_root = tmp_path / "dependency"
    for name in ("ops-skeleton", "unrelated"):
        (dependency_root / name).mkdir(parents=True)
        (dependency_root / name / "marker.txt").write_text(name, encoding="utf-8")
    (dependency_root / "ops-skeleton/CMakeLists.txt").write_text("# build entry\n")

    sandbox_root = tmp_path / "sandbox-dependency"
    _copy_dependency_tree(dependency_root, sandbox_root, compile_repo="ops-skeleton")

    assert (sandbox_root / "ops-skeleton/marker.txt").read_text(encoding="utf-8") == "ops-skeleton"
    assert not (sandbox_root / "unrelated").exists()


def test_arena_compile_uses_upstream_dependency_interface(tmp_path):

    from ascend_arena_e2e.ascend_arena.runners.build import BuildRunner, CompileRepoTarget

    runner = BuildRunner(tmp_path, compile_jobs=7)
    target = CompileRepoTarget(
        repo_root=tmp_path / "ops-skeleton",
        repo_name="ops-skeleton",
        family="math",
        compile_op_name="angle_v2",
        relative_op_dir=Path("angle_v2"),
        vendor_name="benchmark",
        vendor_package_name="benchmark_math",
    )

    command = runner._build_compile_repo_command(chip_name="ascend910b", compile_target=target)

    assert command == (
        "bash build.sh --pkg --no_force --soc=ascend910b "
        "--ops=angle_v2 --vendor_name=benchmark --vendor_suffix=math -j7"
    )
    assert "--fast_pybind_op" not in command


def test_arena_compile_workspace_injects_operator_into_source_free_skeleton(tmp_path):
    from ascend_arena_e2e.ascend_arena.runners.build import BuildRunner, CompileRepoTarget

    skeleton = tmp_path / "dependency" / "ops-skeleton"
    skeleton.mkdir(parents=True)
    (skeleton / "CMakeLists.txt").write_text("# skeleton\n", encoding="utf-8")
    operator = tmp_path / "bench" / "demo" / "demo"
    (operator / "op_host").mkdir(parents=True)
    (operator / "op_kernel").mkdir()
    (operator / "_deps").mkdir()
    (operator / "CMakeLists.txt").write_text("# operator\n", encoding="utf-8")
    (operator / "_deps" / "marker.txt").write_text("dependency\n", encoding="utf-8")

    target = CompileRepoTarget(
        repo_root=skeleton,
        repo_name="ops-skeleton",
        family="nn",
        compile_op_name="demo",
        relative_op_dir=Path("demo"),
        vendor_name="benchmark",
        vendor_package_name="benchmark_nn",
    )
    workspace = tmp_path / "workspace"
    BuildRunner(tmp_path / "dependency")._prepare_run_local_compile_repo_workspace(
        src_dir=workspace,
        compile_target=target,
        source_root=operator,
    )

    assert (workspace / "CMakeLists.txt").read_text(encoding="utf-8") == "# skeleton\n"
    assert (workspace / "demo" / "CMakeLists.txt").read_text(encoding="utf-8") == "# operator\n"
    assert (workspace / "demo" / "_deps" / "marker.txt").is_file()


def test_arena_multi_operator_target_counts_native_tasks():
    target = CannBenchClient(REPO_ROOT).load_eval_target("ascend_arena", "bench_lab/ascend_arena_bench")
    assert target.metadata["multi_operator"] is True
    assert target.metadata["task_count"] == ARENA_OPERATOR_COUNT


def test_arena_profile_only_stage_skips_profile_correctness(tmp_path):

    from ascend_arena_e2e.ascend_arena.runners.benchmark import BenchmarkRunner
    from ascend_arena_e2e.ascend_arena.runners.build import BuildResult
    from ascend_arena_e2e.ascend_arena.runners.profile import ProfileResult

    runner = BenchmarkRunner(
        dependency_root=tmp_path,
        profile_requires_correctness=False,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    build = BuildResult(
        success=True,
        op_name="angle_v2",
        standard_op_root=tmp_path / "standard",
        work_dir=run_dir / "work",
        build_dir=run_dir / "build",
        log_path=run_dir / "compile.log",
        extension_path=run_dir / "module.so",
        custom_opp_root=None,
        message="ok",
    )
    called = []

    def fake_profile(**kwargs):
        called.append(kwargs["correctness_result"])
        return ProfileResult(True, "angle_v2", run_dir / "profile.log", run_dir / "profile", {}, "ok")

    runner.profile_runner.run = fake_profile
    correctness, profile = runner._run_profile_stage(build, tmp_path / "standard", 0, frozenset({"profile"}))

    assert correctness.success
    assert correctness.case_results == []
    assert correctness.log_path.read_text(encoding="utf-8").strip() == "skipped_profile_correctness_not_requested"
    assert profile.success
    assert called == [correctness]


def test_arena_correctness_only_stage_skips_performance(tmp_path):

    from ascend_arena_e2e.ascend_arena.runners.benchmark import BenchmarkRunner
    from ascend_arena_e2e.ascend_arena.runners.build import BuildResult
    from ascend_arena_e2e.ascend_arena.runners.correctness import CorrectnessResult

    runner = BenchmarkRunner(
        dependency_root=tmp_path,
        profile_performance=False,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    build = BuildResult(
        success=True,
        op_name="angle_v2",
        standard_op_root=tmp_path / "standard",
        work_dir=run_dir / "work",
        build_dir=run_dir / "build",
        log_path=run_dir / "compile.log",
        extension_path=run_dir / "module.so",
        custom_opp_root=None,
        message="ok",
    )
    correctness = CorrectnessResult(True, "angle_v2", run_dir / "profile-correctness.log", [])
    runner.profile_correctness_runner.run = lambda **kwargs: correctness
    runner.profile_runner.run = lambda **kwargs: pytest.fail("profile should be skipped")

    actual_correctness, profile = runner._run_profile_stage(build, tmp_path / "standard", 0, frozenset({"profile"}))

    assert actual_correctness is correctness
    assert profile.success
    assert profile.message == "skipped_performance_disabled"


def test_arena_benchmark_run_restores_runtime_opp_between_stages(tmp_path, monkeypatch):

    from ascend_arena_e2e.ascend_arena.runners.benchmark import BenchmarkRunner
    from ascend_arena_e2e.ascend_arena.runners.build import BuildResult

    base_opp = tmp_path / "base-opp"
    (base_opp / "built-in").mkdir(parents=True)
    (base_opp / "built-in" / "marker").write_text("base\n", encoding="utf-8")
    original_custom = tmp_path / "original-custom"
    original_custom.mkdir()
    original_cust_opapi = tmp_path / "original-libcust_opapi.so"
    original_cust_opapi.touch()
    monkeypatch.setenv("ASCEND_OPP_PATH", str(base_opp))
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", str(original_custom))
    monkeypatch.setenv("KERNEL_GEN_OPS_CUST_OPAPI_PATH", str(original_cust_opapi))

    runner = BenchmarkRunner(dependency_root=tmp_path)
    observed = []

    def fake_build(*, run_root, **kwargs):
        stage_root = run_root / "angle_v2"
        work_dir = stage_root / "src"
        work_dir.mkdir(parents=True, exist_ok=True)
        custom_opp = stage_root / "custom-opp"
        custom_opapi = custom_opp / "op_api" / "lib" / "libcust_opapi.so"
        custom_opapi.parent.mkdir(parents=True)
        custom_opapi.touch()
        return BuildResult(
            success=True,
            op_name="angle_v2",
            standard_op_root=tmp_path / "standard",
            work_dir=work_dir,
            build_dir=stage_root / "build",
            log_path=stage_root / "compile.log",
            extension_path=stage_root / "module.so",
            custom_opp_root=custom_opp,
            message="ok",
        )

    def fake_smoke(build_result, standard_dir, device_id, run_stages):
        observed.append(
            (
                os.environ.get("ASCEND_OPP_PATH"),
                os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
                os.environ.get("KERNEL_GEN_OPS_CUST_OPAPI_PATH"),
                "kernel_gen_ops" in sys.modules,
            )
        )
        monkeypatch.setitem(sys.modules, "kernel_gen_ops", SimpleNamespace())
        return SimpleNamespace(success=True)

    def fake_complete(*args):
        return SimpleNamespace(success=True)

    def fake_profile(*args):
        return SimpleNamespace(success=True), SimpleNamespace()

    monkeypatch.setattr(runner, "_run_build_stage", fake_build)
    monkeypatch.setattr(runner, "_run_smoke_stage", fake_smoke)
    monkeypatch.setattr(runner, "_run_complete_stage", fake_complete)
    monkeypatch.setattr(runner, "_run_profile_stage", fake_profile)
    monkeypatch.setattr(runner, "_assemble_result", lambda *args: SimpleNamespace(success=True))

    for index in range(2):
        runner.run(
            standard_dir=tmp_path / "standard",
            op_name="angle_v2",
            chip_name="ascend910b",
            run_root=tmp_path / f"run-{index}",
        )

    assert len(observed) == 2
    assert observed[0][0].endswith("run-0/angle_v2/runtime_opp")
    assert observed[1][0].endswith("run-1/angle_v2/runtime_opp")
    assert observed[0][1] == str(tmp_path / "run-0/angle_v2/custom-opp")
    assert observed[1][1] == str(tmp_path / "run-1/angle_v2/custom-opp")
    assert observed[0][2].endswith("run-0/angle_v2/custom-opp/op_api/lib/libcust_opapi.so")
    assert observed[1][2].endswith("run-1/angle_v2/custom-opp/op_api/lib/libcust_opapi.so")
    assert observed[0][3] is False
    assert observed[1][3] is False
    assert "kernel_gen_ops" not in sys.modules
    assert os.environ["ASCEND_OPP_PATH"] == str(base_opp)
    assert os.environ["ASCEND_CUSTOM_OPP_PATH"] == str(original_custom)
    assert os.environ["KERNEL_GEN_OPS_CUST_OPAPI_PATH"] == str(original_cust_opapi)


def test_workspace_only_and_resume_preserve_source(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(workspace_module, '_run_initial_evaluation', lambda workspace: calls.append(workspace))
    argv = ['prepare', '--operator', 'expand_into_jagged_permute', '--output', str(tmp_path), '--device-id', '3']
    assert pipeline_cli.main([*argv, '--workspace-only']) == 0
    assert not calls
    source = tmp_path / 'expand_into_jagged_permute/work/expand_into_jagged_permute/op_kernel/manual.txt'
    source.write_text('custom implementation')
    assert pipeline_cli.main([*argv, '--resume']) == 0
    assert len(calls) == 1
    assert source.read_text() == 'custom implementation'
    assert pipeline_cli.main([*argv, '--resume', '--force']) == 1
    assert pipeline_cli.main([*argv, '--resume', '--device-id', '4']) == 1
    assert source.read_text() == 'custom implementation'


@pytest.mark.parametrize('baseline_ok', [True, False])
def test_direct_evaluation_collects_both_implementations(tmp_path, monkeypatch, baseline_ok):
    from kernel_eval import optimize
    dependency, task, source, ops_json, original = _make_optimize_main_fixture(tmp_path)
    calls, reports = [], []

    def run_stage(**kwargs):
        calls.append(kwargs)
        directory = kwargs['result_path'].parent
        ok = baseline_ok or kwargs['run_root'].name == 'candidate'
        return SimpleNamespace(
            success=ok,
            build_result=SimpleNamespace(success=ok, log_path=directory / 'compile.log'),
            profile_result=SimpleNamespace(success=ok, log_path=directory / 'profile.log', message='test'),
        ), {'case': [{'task_duration_us': 2.0 if len(calls) == 1 else 1.0}]}

    monkeypatch.setattr(optimize, '_preflight_dependencies', lambda **kwargs: None)
    monkeypatch.setattr(optimize, '_resolve_env_script', lambda explicit: None)
    monkeypatch.setattr(optimize, '_run_stage', run_stage)
    monkeypatch.setattr(optimize, '_build_report', lambda **kwargs: reports.append(kwargs))
    result = optimize.main([
        '--source-dir', str(source), '--task-dir', str(task), '--operator', 'angle_v2',
        '--dependency-root', str(dependency), '--ops-json', str(ops_json),
        '--reports-dir', str(tmp_path / 'reports'),
    ])
    assert result == (0 if baseline_ok else 1)
    assert [call['run_root'].name for call in calls] == (['init', 'candidate'] if baseline_ok else ['init'])
    assert all(call['stages'] == frozenset({'smoke', 'complete', 'profile'}) for call in calls)
    assert calls[0]['result_path'].parent.name == 'baseline_results'
    if baseline_ok:
        assert reports[0]['baseline']['case'][0]['task_duration_us'] == 2.0
        assert reports[0]['candidate']['case'][0]['task_duration_us'] == 1.0
    else:
        assert not reports


def test_progress_summary_filters_noise_and_reports_case_latency(capsys):
    from kernel_eval.optimize import _print_stage_progress
    context = 'expand_into_jagged_permute/candidate/NPU:4'
    _print_stage_progress({'label': 'build_start'}, context)
    _print_stage_progress({'label': 'profile_case_finished', 'case_id': 'fp16',
                           'success': True, 'durations_us': [12.34567]}, context)
    _print_stage_progress({'label': 'profile_case_retry', 'case_id': 'bf16',
                           'reason': 'missing_profile_samples'}, context)
    _print_stage_progress({'label': 'complete_correctness_finished', 'success': False,
                           'case_count': 36, 'log_path': '/results/complete.log'}, context)
    _print_stage_progress({'label': 'unknown', 'message': 'raw compiler output'}, context)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert '编译 开始' in lines[0]
    assert 'fp16 12.3457 μs' in lines[1]
    assert 'bf16' in lines[2] and '重试' in lines[2]
    assert '失败' in lines[3] and '用例数=36' in lines[3] and '/results/complete.log' in lines[3]


def test_torch_profiler_retries_once_when_samples_are_missing(tmp_path, monkeypatch):
    from ascend_arena_e2e.ascend_arena.runners.profile import ProfileRunner

    runner = ProfileRunner(repeat=2, profiler='torch-profiler')
    attempts = []

    def execute(*, output_dir, case, runtime):
        attempts.append((output_dir / 'stale').exists())
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'stale').write_text('first attempt')

    monkeypatch.setattr(runner, '_execute_torch_profiler_case', execute)
    monkeypatch.setattr(runner, '_extract_metrics', lambda output_dir: [])
    proc = runner._run_torch_profiler_case_in_process(
        standard_dir=tmp_path,
        build_result=SimpleNamespace(),
        output_dir=tmp_path / 'profile-case',
        device_id=0,
        case={'case_id': 'case-1'},
        runtime={},
    )

    assert proc.returncode == 0
    assert attempts == [False, False]
