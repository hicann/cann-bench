import json
from pathlib import Path

import auto_pipeline.cli as cli
import auto_pipeline.core as config_runner
from auto_pipeline.converter.base import ConversionResult
from auto_pipeline.core import AGENT_SUCCESS, Artifact, CannBenchEvalResult, Submission
from auto_pipeline.generator.opencode.runner import OpenCodeAgent, OpenCodeRunResult
from auto_pipeline.generator.pypto import PyptoOrchestratorAgent


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_config(path: Path, *, agent_type: str, selectors: list[str]) -> None:
    tasks = "\n".join(f"    - {selector}" for selector in selectors)
    path.write_text(
        f"""
agent:
  type: {agent_type}
benchmark:
  root: {REPO_ROOT}
  name: cann
  tasks:
{tasks}
""".lstrip(),
        encoding="utf-8",
    )


def _register_source_run(
    *,
    run_id: str,
    config_path: Path,
    source_output: Path,
    workspace: Path,
    tasks: list[tuple[str, str]],
) -> None:
    cli.pipeline_state.upsert_run(
        run_id,
        {
            "status": "failed",
            "output": str(source_output),
            "workspace": str(workspace),
            "config_path": str(config_path),
            "model": "fake/model",
            "devices": [0, 1],
            "parallel": 2,
            "gen_timeout_sec": 30,
            "eval_timeout_sec": 40,
            "tasks_declared": [
                {
                    "task_id": name,
                    "task_index": index,
                    "name": name,
                    "selector": selector,
                    "result_file": str(source_output / name / "benchmark_result.json"),
                }
                for index, (name, selector) in enumerate(tasks)
            ],
        },
    )
    for index, (name, selector) in enumerate(tasks):
        cli.pipeline_state.update_task(
            run_id,
            name,
            {
                "task_index": index,
                "name": name,
                "selector": selector,
                "status": "failed",
                "output": str(source_output / name),
                "result_file": str(source_output / name / "benchmark_result.json"),
                "device_id": index,
            },
        )


def _patch_success_eval(monkeypatch) -> None:
    def fake_eval(self, *, bench_name, source_dir, task_selector, reports_dir, device_id=None, extra_args=None):
        return CannBenchEvalResult(
            returncode=0,
            command=["fake-eval", bench_name, task_selector],
            reports_dir=Path(reports_dir),
        )

    monkeypatch.setattr(config_runner.CannBenchClient, "eval_submission", fake_eval)


class RecordingConverter:
    name = "recording-converter"
    source_generator = "fake"
    target_benchmark = "cann"

    def __init__(self, artifacts: list[Artifact]) -> None:
        self.artifacts = artifacts

    def convert(self, bench_name, case, artifact, *, output_dir, runner=None, workdir=None):
        self.artifacts.append(artifact)
        submission_dir = Path(output_dir)
        package_dir = submission_dir / "cann_bench"
        package_dir.mkdir(parents=True, exist_ok=True)
        (submission_dir / "build.sh").write_text("#!/usr/bin/env bash\nset -e\n", encoding="utf-8")
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
        (package_dir / f"{case.operator.lower()}.py").write_text("def op(*args):\n    return args\n", encoding="utf-8")
        return ConversionResult(
            artifact=Artifact(status=AGENT_SUCCESS, workdir=submission_dir, files={"source_dir": submission_dir}),
            submission=Submission(bench_name, case.operator, submission_dir),
        )


def _patch_recording_converter(monkeypatch, artifacts: list[Artifact]) -> None:
    monkeypatch.setattr(config_runner, "create_converter", lambda *_args, **_kwargs: RecordingConverter(artifacts))


def test_retry_akg_stages_and_reuses_existing_model(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.pipeline_state, "DEFAULT_CANN_BENCH_ROOT", tmp_path)
    config_path = tmp_path / "akg.yaml"
    source_output = tmp_path / "source"
    workspace = tmp_path / "akg-workspace"
    retry_output = tmp_path / "retry-akg"
    _write_config(config_path, agent_type="akg-agent", selectors=["tasks/level1/exp", "tasks/level1/sigmoid"])

    artifact_dir = source_output / "exp" / "work" / "artifact"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "akg_model.py").write_text("# reused akg code\n", encoding="utf-8")
    (artifact_dir / "akg-agent.log").write_text("old log\n", encoding="utf-8")
    (source_output / "exp" / "benchmark_result.json").write_text("{}\n", encoding="utf-8")
    _register_source_run(
        run_id="source-akg",
        config_path=config_path,
        source_output=source_output,
        workspace=workspace,
        tasks=[("exp", "tasks/level1/exp"), ("sigmoid", "tasks/level1/sigmoid")],
    )

    class FailingAkgGenerator:
        type = "akg-agent"

        def generate(self, task):
            raise AssertionError("retry should reuse staged akg_model.py")

    converted_artifacts: list[Artifact] = []
    monkeypatch.setattr(config_runner, "create_generator", lambda *_args, **_kwargs: FailingAkgGenerator())
    _patch_recording_converter(monkeypatch, converted_artifacts)
    _patch_success_eval(monkeypatch)

    assert cli.main([
        "retry",
        "--run-id",
        "source-akg",
        "--task",
        "exp",
        "--foreground",
        "--retry-run-id",
        "retry-akg",
        "--retry-output",
        str(retry_output),
    ]) == 0

    staged_model = retry_output / "exp" / "work" / "artifact" / "akg_model.py"
    assert staged_model.read_text(encoding="utf-8") == "# reused akg code\n"
    assert converted_artifacts[0].metadata["akg_reused"] is True
    assert converted_artifacts[0].metadata["akg_reuse_source"] == str(staged_model)
    task_state = cli.pipeline_state.read_json(cli.pipeline_state.task_file("retry-akg", "exp"))
    run_state = cli.pipeline_state.read_json(cli.pipeline_state.run_file("retry-akg"))
    assert task_state["status"] == "success"
    assert run_state["status"] == "success"
    assert [task["selector"] for task in run_state["tasks_declared"]] == ["tasks/level1/exp"]


def test_retry_akg_generates_when_no_reusable_model_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.pipeline_state, "DEFAULT_CANN_BENCH_ROOT", tmp_path)
    config_path = tmp_path / "akg.yaml"
    source_output = tmp_path / "source"
    workspace = tmp_path / "akg-workspace"
    retry_output = tmp_path / "retry-akg-fallback"
    _write_config(config_path, agent_type="akg-agent", selectors=["tasks/level1/exp"])
    (source_output / "exp").mkdir(parents=True)
    (source_output / "exp" / "benchmark_result.json").write_text("{}\n", encoding="utf-8")
    _register_source_run(
        run_id="source-akg-fallback",
        config_path=config_path,
        source_output=source_output,
        workspace=workspace,
        tasks=[("exp", "tasks/level1/exp")],
    )

    calls = {"generate": 0}

    class FakeAkgGenerator:
        type = "akg-agent"

        def generate(self, task):
            calls["generate"] += 1
            task.output_dir.mkdir(parents=True, exist_ok=True)
            code_file = task.output_dir / "akg_model.py"
            code_file.write_text("# freshly generated\n", encoding="utf-8")
            return Artifact(
                status=AGENT_SUCCESS,
                workdir=task.output_dir,
                files={"generated_code": code_file},
                output_text="# freshly generated\n",
            )

    converted_artifacts: list[Artifact] = []
    monkeypatch.setattr(config_runner, "create_generator", lambda *_args, **_kwargs: FakeAkgGenerator())
    _patch_recording_converter(monkeypatch, converted_artifacts)
    _patch_success_eval(monkeypatch)

    assert cli.main([
        "retry",
        "--run-id",
        "source-akg-fallback",
        "--task",
        "exp",
        "--foreground",
        "--retry-run-id",
        "retry-akg-fallback",
        "--retry-output",
        str(retry_output),
    ]) == 0

    assert calls["generate"] == 1
    assert converted_artifacts[0].output_text == "# freshly generated\n"
    assert "akg_reused" not in converted_artifacts[0].metadata


def _write_completed_pypto_custom(op_dir: Path, *, op_name: str = "gelu") -> None:
    op_dir.mkdir(parents=True, exist_ok=True)
    for name in [f"{op_name}_impl.py", f"{op_name}_golden.py", f"test_{op_name}.py", "SPEC.md"]:
        (op_dir / name).write_text("# pypto artifact\n", encoding="utf-8")
    (op_dir / ".orchestrator_state.json").write_text(
        json.dumps({"stage_status": {str(index): "completed" for index in range(1, 8)}}),
        encoding="utf-8",
    )


def _patch_pypto_generator(monkeypatch) -> None:
    def fake_create_generator(generator_type, cfg):
        assert generator_type == "pypto"
        return PyptoOrchestratorAgent(pypto_repo_root=Path(cfg["repo_root"]))

    monkeypatch.setattr(config_runner, "create_generator", fake_create_generator)


def _patch_fake_runner(monkeypatch) -> None:
    class FakeRunner:
        type = "fake-runner"

        def run(self, prompt):
            raise AssertionError("recording converter should ignore conversion runner")

    monkeypatch.setattr(config_runner, "create_runner", lambda *_args, **_kwargs: FakeRunner())


def test_retry_pypto_skips_completed_workspace_custom(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.pipeline_state, "DEFAULT_CANN_BENCH_ROOT", tmp_path)
    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", str(tmp_path / "pto-isa"))
    config_path = tmp_path / "pypto.yaml"
    source_output = tmp_path / "source"
    workspace = tmp_path / "pypto-workspace"
    retry_output = tmp_path / "retry-pypto-skip"
    _write_config(config_path, agent_type="pypto", selectors=["tasks/level1/gelu"])
    _write_completed_pypto_custom(workspace / "custom" / "gelu")
    (workspace / "custom" / "gelu" / "prof").mkdir()
    (workspace / "custom" / "gelu" / "prof" / "trace.json").write_text("{}\n", encoding="utf-8")
    (source_output / "gelu").mkdir(parents=True)
    (source_output / "gelu" / "benchmark_result.json").write_text("{}\n", encoding="utf-8")
    _register_source_run(
        run_id="source-pypto-skip",
        config_path=config_path,
        source_output=source_output,
        workspace=workspace,
        tasks=[("gelu", "tasks/level1/gelu")],
    )

    opencode_calls = {"count": 0}

    def fail_run_opencode(self, *args, **kwargs):
        opencode_calls["count"] += 1
        raise AssertionError("completed PyPTO custom should skip opencode")

    converted_artifacts: list[Artifact] = []
    monkeypatch.setattr(OpenCodeAgent, "run_opencode", fail_run_opencode)
    _patch_pypto_generator(monkeypatch)
    _patch_fake_runner(monkeypatch)
    _patch_recording_converter(monkeypatch, converted_artifacts)
    _patch_success_eval(monkeypatch)

    assert cli.main([
        "retry",
        "--run-id",
        "source-pypto-skip",
        "--task",
        "gelu",
        "--foreground",
        "--retry-run-id",
        "retry-pypto-skip",
        "--retry-output",
        str(retry_output),
    ]) == 0

    snapshot = retry_output / "gelu" / "custom" / "gelu"
    assert opencode_calls["count"] == 0
    assert converted_artifacts[0].metadata["pypto_status"] == "skipped"
    assert converted_artifacts[0].files["source_dir"] == snapshot.resolve()
    assert (snapshot / "gelu_impl.py").is_file()
    assert not (snapshot / "prof").exists()
    task_state = cli.pipeline_state.read_json(cli.pipeline_state.task_file("retry-pypto-skip", "gelu"))
    assert task_state["workspace_custom_dir"] == str((workspace / "custom" / "gelu").resolve())
    assert task_state["output_custom_dir"] == str(snapshot.resolve())


def test_retry_pypto_resumes_incomplete_workspace_custom(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.pipeline_state, "DEFAULT_CANN_BENCH_ROOT", tmp_path)
    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", str(tmp_path / "pto-isa"))
    config_path = tmp_path / "pypto.yaml"
    source_output = tmp_path / "source"
    workspace = tmp_path / "pypto-workspace"
    retry_output = tmp_path / "retry-pypto-resume"
    _write_config(config_path, agent_type="pypto", selectors=["tasks/level1/gelu"])
    op_dir = workspace / "custom" / "gelu"
    op_dir.mkdir(parents=True)
    (op_dir / "KEEP.txt").write_text("keep me\n", encoding="utf-8")
    (source_output / "gelu").mkdir(parents=True)
    (source_output / "gelu" / "benchmark_result.json").write_text("{}\n", encoding="utf-8")
    _register_source_run(
        run_id="source-pypto-resume",
        config_path=config_path,
        source_output=source_output,
        workspace=workspace,
        tasks=[("gelu", "tasks/level1/gelu")],
    )

    opencode_calls = {"count": 0}

    def fake_run_opencode(self, prompt, *, cwd=None, prompt_text=None, log_name=None, **kwargs):
        opencode_calls["count"] += 1
        assert Path(cwd).resolve() == workspace.resolve()
        _write_completed_pypto_custom(Path(cwd) / "custom" / "gelu")
        prompt.output_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt.output_dir / "PROMPT.md"
        prompt_file.write_text(prompt_text or prompt.text, encoding="utf-8")
        log_file = prompt.output_dir / (log_name or "pypto.log")
        log_file.write_text("fake opencode completed\n", encoding="utf-8")
        return OpenCodeRunResult(
            status=AGENT_SUCCESS,
            returncode=0,
            timed_out=False,
            started=True,
            message="fake opencode success",
            log_file=log_file,
            prompt_file=prompt_file,
            live_bridge={},
            session_export={},
        )

    converted_artifacts: list[Artifact] = []
    monkeypatch.setattr(OpenCodeAgent, "run_opencode", fake_run_opencode)
    _patch_pypto_generator(monkeypatch)
    _patch_fake_runner(monkeypatch)
    _patch_recording_converter(monkeypatch, converted_artifacts)
    _patch_success_eval(monkeypatch)

    assert cli.main([
        "retry",
        "--run-id",
        "source-pypto-resume",
        "--task",
        "gelu",
        "--foreground",
        "--retry-run-id",
        "retry-pypto-resume",
        "--retry-output",
        str(retry_output),
    ]) == 0

    snapshot = retry_output / "gelu" / "custom" / "gelu"
    assert opencode_calls["count"] == 1
    assert (op_dir / "KEEP.txt").read_text(encoding="utf-8") == "keep me\n"
    assert converted_artifacts[0].metadata["pypto_status"] == "success"
    assert converted_artifacts[0].files["source_dir"] == snapshot.resolve()
    assert (snapshot / "KEEP.txt").is_file()
    assert (snapshot / "SPEC.md").is_file()
