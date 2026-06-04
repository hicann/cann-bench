#!/usr/bin/python3
# coding=utf-8

import csv
import json
import sqlite3
import subprocess
from concurrent.futures import Future
from pathlib import Path

import pytest

import auto_pipeline.core as config_runner
from auto_pipeline.generator.opencode.exporter import export_session_to_markdown
from auto_pipeline.generator.opencode.live_bridge import OpencodeLiveBridge, _opencode_config_with_plugin
from auto_pipeline.generator.opencode import opencode_permission_without_external_asks
from auto_pipeline.generator.pypto import PyptoOrchestratorAgent
from auto_pipeline.generator.registry import available_generators, create_generator
from auto_pipeline.prompt.registry import build_case_material
from auto_pipeline.core import build_eval_args, run_from_mapping
from auto_pipeline.core import CannBenchClient, CannBenchEvalResult
from auto_pipeline.core import GeneratorInput, PromptGenerator
from auto_pipeline.core import AGENT_SUCCESS, Artifact, RunnerPrompt, Submission
from auto_pipeline.core import BenchmarkPipeline, PipelineRunResult
from auto_pipeline.generator.pypto.converter import PyptoToCannConverter, PyptoToStanfordConverter
from auto_pipeline.converter.registry import available_converters, create_converter
from auto_pipeline.core import CannBenchCase
from kernel_eval.benches.cann_loader import CannCaseLoader
from kernel_eval.benches.stanford_loader import StanfordTaskLoader
from kernel_eval.benches.stanford_matcher import StanfordMatcher
from kernel_eval.utils.path_resolver import resolve_task_dir


REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_submission(adapter, case: CannBenchCase, artifact: Artifact, *, output_dir: Path) -> Submission:
    return adapter.build_submission(case.bench_name, case, artifact, output_dir=output_dir)


def _generation_task(case: CannBenchCase, workdir: Path, *, timeout_sec: int = 30) -> GeneratorInput:
    material = build_case_material(case)
    workdir = Path(workdir)
    return GeneratorInput(
        case=case,
        material=material,
        workdir=workdir,
        output_dir=workdir / "artifact",
        timeout_sec=timeout_sec,
        title=f"pypto:{case.operator}",
        metadata={
            "bench_name": case.bench_name,
            "operator": case.operator,
            "task_dir": str(case.task_dir),
            "schema": case.metadata.get("schema") or "",
            "case_preview": case.metadata.get("case_preview"),
        },
    )


def test_loads_cann_bench_gelu_case():
    client = CannBenchClient(REPO_ROOT)

    case = client.load_case("cann", "tasks/level1/gelu")

    assert case.operator == "Gelu"
    assert case.files["proto"].name == "proto.yaml"
    assert case.files["golden"].name == "golden.py"
    assert case.files["cases"].name in {"cases.yaml", "cases.csv"}
    assert case.metadata["schema"].startswith("gelu(")
    assert case.metadata["case_preview"]


def test_exp_general_case_remains_full_task():
    yaml_cases = config_runner.read_yaml_mapping(REPO_ROOT / "tasks/level1/exp/cases.yaml")["cases"]
    assert len(yaml_cases) == 20
    assert [case["case_id"] for case in yaml_cases] == list(range(1, 21))
    assert {dtype for case in yaml_cases for dtype in case["dtype"]} == {"float16", "float32", "bfloat16"}


def test_pypto_cann_bench_exp_task_is_static_2d_float32_subset():
    selector = "bench_lab/pypto_cann_bench/exp"
    task_dir = REPO_ROOT / selector

    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", selector)
    assert case.operator == "Exp"
    assert case.rel_path == selector
    assert case.metadata["schema"].startswith("exp(")
    bench_root, filter_prefix = resolve_task_dir(selector, REPO_ROOT)
    assert bench_root == str(REPO_ROOT / "bench_lab/pypto_cann_bench")
    assert filter_prefix == "exp"

    yaml_cases = config_runner.read_yaml_mapping(task_dir / "cases.yaml")["cases"]
    assert [case["case_id"] for case in yaml_cases] == [2, 8, 15]
    assert {tuple(case["dtype"]) for case in yaml_cases} == {("float32",)}
    assert all(len(case["input_shape"]) == 1 and len(case["input_shape"][0]) == 2 for case in yaml_cases)
    assert all(float(case["baseline_perf_us"]) > 0 and float(case["t_hw_us"]) > 0 for case in yaml_cases)

    with (task_dir / "cases.csv").open(encoding="utf-8", newline="") as handle:
        csv_cases = list(csv.DictReader(handle))
    assert [int(case["case_id"]) for case in csv_cases] == [2, 8, 15]
    assert {tuple(json.loads(case["dtype"])) for case in csv_cases} == {("float32",)}
    assert all(len(json.loads(case["input_shape"])[0]) == 2 for case in csv_cases)
    assert all(float(case["baseline_perf_us"]) > 0 and float(case["t_hw_us"]) > 0 for case in csv_cases)

    proto = config_runner.read_yaml_mapping(task_dir / "proto.yaml")["operator"]
    assert proto["name"] == "Exp"
    assert proto["schema"] == "exp(Tensor x, float base, float scale, float shift) -> Tensor y"
    assert proto["inputs"][0]["dtype"] == ["float32"]
    assert proto["outputs"][0]["dtype"] == ["float32"]

    scoped_text = "\n".join(
        [
            (task_dir / "proto.yaml").read_text(encoding="utf-8"),
            (task_dir / "desc.md").read_text(encoding="utf-8"),
            (task_dir / "golden.py").read_text(encoding="utf-8"),
        ]
    )
    assert "float16" not in scoped_text
    assert "bfloat16" not in scoped_text
    assert "任意维度" not in scoped_text
    assert "1D" not in scoped_text
    assert "5D" not in scoped_text

    loaded_cases = CannCaseLoader(tasks_root=str(REPO_ROOT / "bench_lab/pypto_cann_bench")).scan_by_rel_path("exp")
    assert [case.case_id for case in loaded_cases] == [
        "exp_2",
        "exp_8",
        "exp_15",
    ]
    assert not any("_pypto_" in case.case_id for case in loaded_cases)


def test_pypto_cann_bench_sigmoid_task_is_static_2d_float32_subset():
    selector = "bench_lab/pypto_cann_bench/sigmoid"
    task_dir = REPO_ROOT / selector

    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", selector)
    assert case.operator == "Sigmoid"
    assert case.rel_path == selector
    assert case.metadata["schema"].startswith("sigmoid(")
    bench_root, filter_prefix = resolve_task_dir(selector, REPO_ROOT)
    assert bench_root == str(REPO_ROOT / "bench_lab/pypto_cann_bench")
    assert filter_prefix == "sigmoid"

    yaml_cases = config_runner.read_yaml_mapping(task_dir / "cases.yaml")["cases"]
    assert [case["case_id"] for case in yaml_cases] == [8, 15]
    assert {tuple(case["dtype"]) for case in yaml_cases} == {("float32",)}
    assert all(len(case["input_shape"]) == 1 and len(case["input_shape"][0]) == 2 for case in yaml_cases)
    assert all(float(case["baseline_perf_us"]) > 0 and float(case["t_hw_us"]) > 0 for case in yaml_cases)

    with (task_dir / "cases.csv").open(encoding="utf-8", newline="") as handle:
        csv_cases = list(csv.DictReader(handle))
    assert [int(case["case_id"]) for case in csv_cases] == [8, 15]
    assert {tuple(json.loads(case["dtype"])) for case in csv_cases} == {("float32",)}
    assert all(len(json.loads(case["input_shape"])[0]) == 2 for case in csv_cases)

    proto = config_runner.read_yaml_mapping(task_dir / "proto.yaml")["operator"]
    assert proto["name"] == "Sigmoid"
    assert proto["schema"] == "sigmoid(Tensor x) -> Tensor y"
    assert proto["inputs"][0]["dtype"] == ["float32"]
    assert proto["outputs"][0]["dtype"] == ["float32"]

    scoped_text = "\n".join(
        [
            (task_dir / "proto.yaml").read_text(encoding="utf-8"),
            (task_dir / "desc.md").read_text(encoding="utf-8"),
            (task_dir / "golden.py").read_text(encoding="utf-8"),
        ]
    )
    assert "float16" not in scoped_text
    assert "bfloat16" not in scoped_text
    assert "任意维度" not in scoped_text
    assert "1D" not in scoped_text
    assert "5D" not in scoped_text

    loaded_cases = CannCaseLoader(tasks_root=str(REPO_ROOT / "bench_lab/pypto_cann_bench")).scan_by_rel_path("sigmoid")
    assert [case.case_id for case in loaded_cases] == [
        "sigmoid_8",
        "sigmoid_15",
    ]


def test_generator_benchmark_converters_are_registered():
    names = available_converters()
    assert "pypto -> cann" in names
    assert "pypto -> stanford" in names
    assert "akg-agent -> cann" in names
    assert "akg-agent -> stanford" in names
    assert isinstance(create_converter("pypto", "cann", {}), PyptoToCannConverter)
    assert isinstance(create_converter("pypto", "stanford", {}), PyptoToStanfordConverter)
    assert create_converter("akg-agent", "cann", {}).name == "akg-agent-to-cann"
    assert create_converter("akg-agent", "stanford", {}).name == "akg-agent-to-stanford"
    assert create_converter("akg_agent", "cann", {}).name == "akg-agent-to-cann"
    for generator_type in ("tilelang-ascend", "tilelang", "ascendc", "ascend-c", "triton"):
        with pytest.raises(ValueError, match="unsupported converter"):
            create_converter(generator_type, "cann", {})


def test_pipeline_generates_akg_to_cann_submission_from_fake_agent(tmp_path):
    class FakeAgent:
        type = "akg-agent"

        def run(self, prompt: RunnerPrompt) -> Artifact:
            return Artifact(
                status=AGENT_SUCCESS,
                workdir=prompt.output_dir,
                output_text="class ModelNew:\n    pass\n",
            )

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
            source_dir = Path(source_dir)
            assert source_dir.joinpath("build.sh").is_file()
            assert source_dir.joinpath("cann_bench", "exp.py").is_file()
            return CannBenchEvalResult(returncode=0, command=["fake"], reports_dir=Path(reports_dir))

    pipeline = BenchmarkPipeline(bench_name="cann", client=FakeEvalClient(REPO_ROOT))

    result = pipeline.run_case(
        selector="tasks/level1/exp",
        generator=PromptGenerator(FakeAgent()),
        converter=create_converter("akg-agent", "cann", {}),
        workdir=tmp_path / "work",
        submission_dir=tmp_path / "submission",
        reports_dir=tmp_path / "reports",
    )

    assert result.ok
    assert result.eval_result.command == ["fake"]
    assert result.submission.kind == "cann"
    assert result.submission.source_dir.joinpath("build.sh").is_file()
    assert result.submission.source_dir.joinpath("cann_bench", "exp.py").is_file()
    assert result.generated_artifact.output_text == "class ModelNew:\n    pass\n"


def test_pypto_orchestrator_agent_is_registered(monkeypatch, tmp_path):
    monkeypatch.delenv("PYPTO_PERF_ROUND", raising=False)
    agent = create_generator("pypto", {"repo_root": str(tmp_path)})
    model_agent = create_generator(
        "pypto",
        {"repo_root": str(tmp_path), "model": "zai-coding-plan/glm-5.1"},
    )
    monkeypatch.setenv("PYPTO_PERF_ROUND", "0")
    env_perf_agent = create_generator("pypto", {"repo_root": str(tmp_path)})

    assert isinstance(agent, PyptoOrchestratorAgent)
    assert isinstance(env_perf_agent, PyptoOrchestratorAgent)
    assert isinstance(model_agent, PyptoOrchestratorAgent)
    assert agent.perf_round == 3
    assert env_perf_agent.perf_round == 0
    assert model_agent.opencode_model == "zai-coding-plan/glm-5.1"
    assert "pypto" in available_generators()
    assert available_generators().count("pypto") == 1


def test_pypto_opencode_permission_merge_preserves_existing_rules():
    merged = json.loads(
        opencode_permission_without_external_asks(
            '{"bash": {"rm -rf *": "deny"}, "question": "deny"}'
        )
    )

    assert merged["external_directory"] == "deny"
    assert merged["bash"] == {"rm -rf *": "deny"}
    assert merged["question"] == "deny"


def test_pypto_adapter_validates_backend_artifact(tmp_path):
    client = CannBenchClient(REPO_ROOT)
    case = client.load_case("cann", "tasks/level1/gelu")
    adapter = create_converter("pypto", "cann", {})
    assert isinstance(adapter, PyptoToCannConverter)

    material = build_case_material(case)
    assert material.op_name == "gelu"
    assert material.bench_name == "cann"
    assert "schema: gelu" in material.require_text
    assert "golden.py 参考实现" in material.require_text

    pypto_cann_bench_case = client.load_case("cann", "bench_lab/pypto_cann_bench/exp")
    exp_material = build_case_material(pypto_cann_bench_case)
    assert exp_material.op_name == "exp"

    source_dir = tmp_path / "pypto" / "artifact" / "submission"
    source_dir.joinpath("cann_bench").mkdir(parents=True)
    source_dir.joinpath("cann_bench", "__init__.py").write_text("", encoding="utf-8")
    source_dir.joinpath("build.sh").write_text("#!/usr/bin/env bash\nset -e\n", encoding="utf-8")
    source_dir.joinpath("cann_bench", "gelu.py").write_text("import pypto\n", encoding="utf-8")

    output = Artifact(status=AGENT_SUCCESS, workdir=source_dir.parent, files={"source_dir": source_dir})
    submission = _build_submission(adapter, case, output, output_dir=tmp_path / "pypto" / "submission")

    assert submission.kind == "cann"
    assert submission.source_dir.joinpath("build.sh").is_file()
    assert submission.source_dir.joinpath("cann_bench", "__init__.py").is_file()


def test_pypto_orchestrator_agent_runs_real_pypto_agent_contract(tmp_path):
    repo_root = tmp_path / "pypto_repo"
    repo_root.mkdir()
    opencode = tmp_path / "fake_opencode.py"
    opencode.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "root = pathlib.Path.cwd()\n"
        "def emit_bridge(record):\n"
        "    bridge = os.environ.get('OPENCODE_SUBAGENT_BRIDGE_LOG')\n"
        "    if not bridge:\n"
        "        return\n"
        "    path = pathlib.Path(bridge)\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    with path.open('a', encoding='utf-8') as handle:\n"
        "        handle.write(json.dumps(record) + '\\n')\n"
        "if sys.argv[1:2] == ['run']:\n"
        "    (root / 'argv.json').write_text(json.dumps(sys.argv), encoding='utf-8')\n"
        "    (root / 'env.json').write_text(json.dumps({\n"
        "        'TMPDIR': os.environ.get('TMPDIR'),\n"
        "        'OPENCODE_PERMISSION': os.environ.get('OPENCODE_PERMISSION'),\n"
        "        'OPENCODE_SUBAGENT_BRIDGE_LOG': os.environ.get('OPENCODE_SUBAGENT_BRIDGE_LOG'),\n"
        "        'OPENCODE_CONFIG_CONTENT': os.environ.get('OPENCODE_CONFIG_CONTENT'),\n"
        "    }), encoding='utf-8')\n"
        "    emit_bridge({'kind': 'plugin_loaded', 'time': 1, 'pid': 123})\n"
        "    emit_bridge({'kind': 'event', 'type': 'session.created', 'sessionID': 'ses_root', 'session': {'id': 'ses_root', 'title': 'fake root'}})\n"
        "    emit_bridge({'kind': 'event', 'type': 'session.created', 'sessionID': 'ses_child', 'parentID': 'ses_root', 'session': {'id': 'ses_child', 'title': 'fake subagent', 'parentID': 'ses_root'}})\n"
        "    emit_bridge({'kind': 'event', 'type': 'message.part.updated', 'sessionID': 'ses_child', 'part': {'id': 'prt_tool', 'type': 'tool', 'sessionID': 'ses_child', 'tool': 'bash', 'callID': 'call_bash', 'status': 'running', 'input': '{\"command\":\"echo live\"}', 'outputDelta': 'live child line\\n'}})\n"
        "    emit_bridge({'kind': 'event', 'type': 'message.part.updated', 'sessionID': 'ses_child', 'part': {'id': 'prt_text', 'type': 'text', 'sessionID': 'ses_child', 'text': ''}})\n"
        "    emit_bridge({'kind': 'event', 'type': 'message.part.delta', 'sessionID': 'ses_child', 'delta': {'sessionID': 'ses_child', 'messageID': 'msg_child', 'partID': 'prt_text', 'field': 'text', 'text': 'child '}})\n"
        "    emit_bridge({'kind': 'event', 'type': 'message.part.delta', 'sessionID': 'ses_child', 'delta': {'sessionID': 'ses_child', 'messageID': 'msg_child', 'partID': 'prt_text', 'field': 'text', 'text': 'done'}})\n"
        "    emit_bridge({'kind': 'event', 'type': 'message.part.updated', 'sessionID': 'ses_child', 'part': {'id': 'prt_text', 'type': 'text', 'sessionID': 'ses_child', 'text': 'updated text should not render'}})\n"
        "if sys.argv[1:3] == ['session', 'list']:\n"
        "    title = (root / 'last_title.txt').read_text(encoding='utf-8')\n"
        "    print(f'{title} ses_fake')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:3] == ['export', 'ses_fake']:\n"
        "    print(json.dumps({\n"
        "        'info': {'id': 'ses_fake', 'title': 'fake root', 'time': {'created': 1, 'updated': 4}},\n"
        "        'messages': [{'info': {'role': 'assistant', 'time': {}}, 'parts': [{'type': 'text', 'text': 'root transcript'}]}],\n"
        "        'children': [{\n"
        "            'info': {'id': 'ses_child', 'parent_id': 'ses_fake', 'title': 'fake subagent', 'time': {'created': 2, 'updated': 5}},\n"
        "            'messages': [{'info': {'role': 'assistant', 'time': {}}, 'parts': [{'type': 'text', 'text': 'child transcript'}]}]\n"
        "        }],\n"
        "    }))\n"
        "    raise SystemExit(0)\n"
        "if '--title' in sys.argv:\n"
        "    (root / 'last_title.txt').write_text(sys.argv[sys.argv.index('--title') + 1], encoding='utf-8')\n"
        "op_dir = root / 'custom' / 'ReLU'\n"
        "op_dir.mkdir(parents=True, exist_ok=True)\n"
        "for name in ['SPEC.md', 'ReLU_impl.py', 'ReLU_golden.py', 'test_ReLU.py']:\n"
        "    (op_dir / name).write_text('# pypto\\n', encoding='utf-8')\n"
        "(op_dir / '.orchestrator_state.json').write_text(json.dumps({\n"
        "    'stage_status': {str(i): 'completed' for i in range(1, 8)}\n"
        "}), encoding='utf-8')\n"
        "print(' '.join(sys.argv[1:6]))\n",
        encoding="utf-8",
    )
    opencode.chmod(0o755)

    task_dir = tmp_path / "case" / "ReLU"
    task_dir.mkdir(parents=True)
    task_path = task_dir / "task_desc.py"
    task_path.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return torch.relu(x)\n"
        "def get_init_inputs():\n"
        "    return []\n"
        "def get_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )
    task_dir.joinpath("REQUIRE.md").write_text("# ReLU\n", encoding="utf-8")
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=task_dir,
        operator="TaskDesc",
        rel_path="level1/ReLU",
        files={"task": task_path},
    )
    task = _generation_task(case, tmp_path / "work")
    agent = PyptoOrchestratorAgent(
        pypto_repo_root=repo_root,
        opencode_bin=str(opencode),
        opencode_model="zai-coding-plan/glm-5.1",
    )

    output = agent.generate(task)

    assert output.ok
    assert output.workdir == repo_root / "custom" / "ReLU"
    assert output.files["source_dir"] == repo_root / "custom" / "ReLU"
    assert output.files["ReLU_impl.py"].is_file()
    assert output.metadata["pypto_status"] == "success"
    assert not (task.output_dir / "PYPTO_PROMPT.md").exists()
    assert not (task.output_dir / "INPUT_MATERIAL.md").exists()
    pypto_prompt = (task.output_dir / "PROMPT.md").read_text(encoding="utf-8")
    assert "pypto-op-orchestrator" in pypto_prompt
    assert "工作目录: `custom/ReLU/`" in pypto_prompt
    assert "不要为 cann-bench/Stanford submission 做输出格式对齐" in pypto_prompt
    assert "所有测试日志、临时文件和可再生产物也必须落在 `custom/ReLU/`" in pypto_prompt
    assert "不要写到 `/tmp`" in pypto_prompt
    assert "不要使用 `nohup`、`disown`、后台 `&`" in pypto_prompt
    assert "不要触发需要人工确认的 OpenCode permission ask" in pypto_prompt
    assert "ai_op.py" not in pypto_prompt
    assert "ModelNew" not in pypto_prompt
    assert "ReLU_pypto_impl.py" not in pypto_prompt
    assert "pypto_impl" not in pypto_prompt
    assert "下游 KernelBench 调用约定" not in pypto_prompt
    assert "state_dict" not in pypto_prompt
    assert (repo_root / "custom" / "ReLU" / "task_desc.py").is_file()
    assert (repo_root / "custom" / "ReLU" / "REQUIRE.md").read_text(encoding="utf-8") == "# ReLU\n"
    log_text = output.log_file.read_text(encoding="utf-8")
    assert "--agent pypto-op-orchestrator" in log_text
    assert "-m zai-coding-plan/glm-5.1" in log_text
    assert "<prompt>" in log_text
    argv = json.loads((repo_root / "argv.json").read_text(encoding="utf-8"))
    assert argv[argv.index("-m") + 1] == "zai-coding-plan/glm-5.1"
    env = json.loads((repo_root / "env.json").read_text(encoding="utf-8"))
    assert env["TMPDIR"] == str(repo_root / "custom" / "ReLU" / ".tmp")
    assert json.loads(env["OPENCODE_PERMISSION"])["external_directory"] == "deny"
    assert env["OPENCODE_SUBAGENT_BRIDGE_LOG"] == str(task.output_dir / "opencode-live" / "events.jsonl")
    opencode_config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert any(str(plugin).endswith("opencode_live_bridge_plugin.js") for plugin in opencode_config["plugin"])
    assert (repo_root / "custom" / "ReLU" / ".tmp").is_dir()
    bridge = output.metadata["opencode_live_bridge"]
    assert bridge["status"] == "captured"
    assert bridge["node_session_count"] == 2
    assert bridge["subagent_session_count"] == 1
    assert bridge["subagent_session_ids"] == ["ses_child"]
    assert Path(bridge["events_file"]).is_file()
    session_tree = Path(bridge["session_tree_file"])
    assert "ses_child" in session_tree.read_text(encoding="utf-8")
    child_live = next((task.output_dir / "opencode-live" / "nodes").glob("subagent__ses_child__*.live.md"))
    child_live_text = child_live.read_text(encoding="utf-8")
    assert "live child line" in child_live_text
    assert "child done" in child_live_text
    assert "updated text should not render" not in child_live_text
    assert output.metadata["opencode_session"]["status"] == "exported"
    assert output.metadata["prompt_file"] == str(task.output_dir / "PROMPT.md")
    assert output.metadata["opencode_session"]["node_session_count"] == 2
    assert Path(output.metadata["opencode_session"]["markdown_file"]).is_file()
    assert Path(output.metadata["opencode_session"]["json_file"]).is_file()
    assert Path(output.metadata["opencode_session"]["session_tree_file"]).is_file()
    assert "child transcript" in (task.output_dir / "opencode-session.md").read_text(encoding="utf-8")


def test_pypto_orchestrator_agent_uses_isolated_git_worktree(tmp_path):
    repo_root = tmp_path / "pypto_repo"
    repo_root.mkdir()
    repo_root.joinpath("README.md").write_text("# PyPTO\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )

    opencode = tmp_path / "fake_opencode.py"
    opencode.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "root = pathlib.Path.cwd()\n"
        "if sys.argv[1:2] == ['run']:\n"
        "    (root / 'env.json').write_text(json.dumps({\n"
        "        'PWD': os.environ.get('PWD'),\n"
        "        'TMPDIR': os.environ.get('TMPDIR'),\n"
        "    }), encoding='utf-8')\n"
        "if sys.argv[1:3] == ['session', 'list']:\n"
        "    title = (root / 'last_title.txt').read_text(encoding='utf-8')\n"
        "    print(f'{title} ses_fake')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:3] == ['export', 'ses_fake']:\n"
        "    print(json.dumps({'info': {'id': 'ses_fake', 'title': 'fake'}, 'messages': [], 'children': []}))\n"
        "    raise SystemExit(0)\n"
        "if '--title' in sys.argv:\n"
        "    (root / 'last_title.txt').write_text(sys.argv[sys.argv.index('--title') + 1], encoding='utf-8')\n"
        "op_dir = root / 'custom_iso' / 'ReLU'\n"
        "op_dir.mkdir(parents=True, exist_ok=True)\n"
        "for name in ['SPEC.md', 'ReLU_impl.py', 'ReLU_golden.py', 'test_ReLU.py']:\n"
        "    (op_dir / name).write_text('# pypto\\n', encoding='utf-8')\n"
        "(op_dir / '.orchestrator_state.json').write_text(json.dumps({\n"
        "    'stage_status': {str(i): 'completed' for i in range(1, 8)}\n"
        "}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    opencode.chmod(0o755)

    task_dir = tmp_path / "case" / "ReLU"
    task_dir.mkdir(parents=True)
    task_path = task_dir / "task_desc.py"
    task_path.write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return torch.relu(x)\n"
        "def get_init_inputs():\n"
        "    return []\n"
        "def get_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )
    task_dir.joinpath("REQUIRE.md").write_text("# ReLU\n", encoding="utf-8")
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=task_dir,
        operator="TaskDesc",
        rel_path="level1/ReLU",
        files={"task": task_path},
    )
    task = _generation_task(case, tmp_path / "work")
    worktree_root = tmp_path / "pypto_worktrees"
    agent = PyptoOrchestratorAgent(
        pypto_repo_root=repo_root,
        workdir_root="custom_iso",
        worktree_root=worktree_root,
        opencode_bin=str(opencode),
    )

    output = agent.generate(task)

    assert output.ok
    run_repo_root = Path(output.metadata["pypto_run_repo_root"])
    assert output.metadata["pypto_isolated_worktree"] is True
    assert output.metadata["pypto_repo_root"] == str(repo_root)
    assert output.metadata["pypto_worktree_root"] == str(worktree_root)
    assert run_repo_root.parent == worktree_root
    assert run_repo_root.joinpath(".auto_pipeline_pypto_worktree.json").is_file()
    assert output.workdir == run_repo_root / "custom_iso" / "ReLU"
    assert output.files["source_dir"] == run_repo_root / "custom_iso" / "ReLU"
    assert not (repo_root / "custom_iso").exists()
    env = json.loads(run_repo_root.joinpath("env.json").read_text(encoding="utf-8"))
    assert env["PWD"] == str(run_repo_root)
    assert env["TMPDIR"] == str(run_repo_root / "custom_iso" / "ReLU" / ".tmp")


def test_opencode_live_bridge_merges_config_plugin_once(tmp_path):
    plugin = tmp_path / "bridge.js"
    plugin.write_text("export default {}\n", encoding="utf-8")

    merged = json.loads(
        _opencode_config_with_plugin(
            '{"plugin": ["file:///already.js"], "provider": {"x": {}}}',
            plugin,
        )
    )
    merged_again = json.loads(_opencode_config_with_plugin(json.dumps(merged), plugin))

    assert merged["provider"] == {"x": {}}
    assert merged["plugin"][0] == "file:///already.js"
    assert len(merged["plugin"]) == 2
    assert merged_again["plugin"] == merged["plugin"]


def test_opencode_live_bridge_appends_node_markdown_without_replacing(tmp_path):
    bridge = OpencodeLiveBridge(output_dir=tmp_path)
    bridge.configure_env({})

    session_event = {
        "kind": "event",
        "type": "session.created",
        "sessionID": "ses_child",
        "parentID": "ses_root",
        "session": {"id": "ses_child", "title": "fake subagent", "parentID": "ses_root"},
    }
    bridge._process_jsonl_line(json.dumps(session_event), render=True)
    child_live = next((tmp_path / "opencode-live" / "nodes").glob("subagent__ses_child__*.live.md"))
    inode = child_live.stat().st_ino

    for text in ["hello ", "world"]:
        bridge._process_jsonl_line(
            json.dumps(
                {
                    "kind": "event",
                    "type": "message.part.delta",
                    "sessionID": "ses_child",
                    "delta": {
                        "sessionID": "ses_child",
                        "messageID": "msg_child",
                        "partID": "prt_text",
                        "field": "text",
                        "text": text,
                    },
                }
            ),
            render=True,
        )
        assert child_live.stat().st_ino == inode

    bridge._process_jsonl_line(
        json.dumps(
            {
                "kind": "event",
                "type": "message.part.updated",
                "sessionID": "ses_child",
                "part": {
                    "id": "prt_tool",
                    "type": "tool",
                    "sessionID": "ses_child",
                    "tool": "bash",
                    "callID": "call_bash",
                    "status": "running",
                    "input": '{"command":"echo live"}',
                    "outputDelta": "line 1\n",
                },
            }
        ),
        render=True,
    )
    assert child_live.stat().st_ino == inode

    bridge._process_jsonl_line(
        json.dumps(
            {
                "kind": "event",
                "type": "message.part.updated",
                "sessionID": "ses_child",
                "part": {
                    "id": "prt_tool",
                    "type": "tool",
                    "sessionID": "ses_child",
                    "tool": "bash",
                    "callID": "call_bash",
                    "status": "completed",
                    "outputDelta": "line 2\n",
                },
            }
        ),
        render=True,
    )

    text = child_live.read_text(encoding="utf-8")
    assert child_live.stat().st_ino == inode
    assert "hello world" in text
    assert "line 1\nline 2" in text
    assert "Status: completed" in text


def test_pypto_orchestrator_agent_renders_cann_input_branch(tmp_path):
    repo_root = tmp_path / "pypto_repo"
    repo_root.mkdir()
    opencode = tmp_path / "fake_opencode.py"
    opencode.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import pathlib\n"
        "root = pathlib.Path.cwd()\n"
        "op_dir = root / 'custom' / 'gelu'\n"
        "op_dir.mkdir(parents=True, exist_ok=True)\n"
        "for name in ['SPEC.md', 'gelu_impl.py', 'gelu_golden.py', 'test_gelu.py']:\n"
        "    (op_dir / name).write_text('# pypto\\n', encoding='utf-8')\n"
        "(op_dir / '.orchestrator_state.json').write_text(json.dumps({\n"
        "    'stage_status': {str(i): 'completed' for i in range(1, 8)}\n"
        "}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    opencode.chmod(0o755)

    task_dir = tmp_path / "tasks" / "level1" / "gelu"
    task_dir.mkdir(parents=True)
    proto = task_dir / "proto.yaml"
    proto.write_text("operator:\n  name: Gelu\n  schema: gelu(x) -> y\n", encoding="utf-8")
    cases = task_dir / "cases.yaml"
    cases.write_text("cases:\n  - shape: [16]\n", encoding="utf-8")
    golden = task_dir / "golden.py"
    golden.write_text("def golden(x):\n    return x\n", encoding="utf-8")
    desc = task_dir / "desc.md"
    desc.write_text("# Gelu\n", encoding="utf-8")

    case = CannBenchCase(
        bench_name="cann",
        task_dir=task_dir,
        operator="Gelu",
        rel_path="tasks/level1/gelu",
        files={"proto": proto, "cases": cases, "golden": golden, "desc": desc},
        metadata={
            "schema": "gelu(x) -> y",
            "case_preview": [{"shape": [16]}],
        },
    )
    task = _generation_task(case, tmp_path / "work")
    agent = PyptoOrchestratorAgent(pypto_repo_root=repo_root, opencode_bin=str(opencode))

    output = agent.generate(task)

    assert output.ok
    assert output.workdir == repo_root / "custom" / "gelu"
    assert (repo_root / "custom" / "gelu" / "proto.yaml").is_file()
    assert (repo_root / "custom" / "gelu" / "cases.yaml").is_file()
    assert (repo_root / "custom" / "gelu" / "golden.py").is_file()
    assert (repo_root / "custom" / "gelu" / "desc.md").is_file()
    require_text = (repo_root / "custom" / "gelu" / "REQUIRE.md").read_text(encoding="utf-8")
    assert "cann-bench 输入材料" in require_text
    assert "gelu(x) -> y" in require_text
    assert not (task.output_dir / "PYPTO_PROMPT.md").exists()
    pypto_prompt = (task.output_dir / "PROMPT.md").read_text(encoding="utf-8")
    assert "cann-bench proto" in pypto_prompt
    assert "custom/gelu/proto.yaml" in pypto_prompt
    assert "custom/gelu/cases.yaml" in pypto_prompt
    assert "custom/gelu/golden.py" in pypto_prompt
    assert "CANN selected-case 需求" in pypto_prompt
    assert "唯一权威 case 列表" in pypto_prompt
    assert "若 benchmark 已离线筛选 case" in pypto_prompt
    assert "input_shape、dtype、attrs 和 value_range" in pypto_prompt
    assert "不扩大本次 benchmark 的 dtype、rank 或 shape 范围" in pypto_prompt
    assert "schema 描述的算子语义" in pypto_prompt
    assert "convert 阶段只做提交格式归一，不作为需求修正或 kernel 语义修复环节" in pypto_prompt
    assert "实验可复现" in pypto_prompt
    assert "历史 run、历史 submission、历史 eval 或其它 artifact" in pypto_prompt
    assert "性能调优轮次" in pypto_prompt
    assert "运行边界" in pypto_prompt
    assert "所有测试日志、临时文件和可再生产物也必须落在 `custom/gelu/`" in pypto_prompt
    assert "不要写到 `/tmp`" in pypto_prompt
    assert "前台、有界执行" in pypto_prompt
    assert "OpenCode permission ask" in pypto_prompt
    assert ".orchestrator_state.json" in pypto_prompt
    assert "SPEC.md front matter" not in pypto_prompt
    assert "dtype-nested 的 mere/mare" not in pypto_prompt
    assert "tile-local 或 slice-local 写法" not in pypto_prompt
    assert "fail-fast" not in pypto_prompt
    assert "kernel_eval 的公开入口调用方式" not in pypto_prompt
    assert "输出不是 `None`" not in pypto_prompt
    assert "不要标注 `-> Tensor`" not in pypto_prompt
    assert "PyPTO 交付 lint" not in pypto_prompt
    assert "以 `_wrapper` 结尾" not in pypto_prompt
    assert "pypto.Tensor" not in pypto_prompt
    assert "README.md" not in pypto_prompt
    assert "level0/level1" not in pypto_prompt
    assert "Python module-level 全局常量" not in pypto_prompt
    assert "TILE_SIZE" not in pypto_prompt
    assert "configure_tiling" not in pypto_prompt
    assert "打包后入口形态" not in pypto_prompt
    assert "[PACKAGE_ENTRY_PASS]" not in pypto_prompt
    assert "Stage 5" not in pypto_prompt
    assert "timeout 1800s python -u test_gelu.py" not in pypto_prompt
    assert "stage5_precision.log" not in pypto_prompt
    assert "[PRECISION_PASS]" not in pypto_prompt
    assert "/tmp/opencode" not in pypto_prompt
    assert "CANN clean-room 约束" not in pypto_prompt
    assert "所有输入 rank" not in pypto_prompt
    assert "rank 适配" not in pypto_prompt
    assert "TensorErr::TENSOR_MEMORY_ALLOCATION" not in pypto_prompt
    assert "`pypto.loop` + `pypto.view`" not in pypto_prompt
    assert "task_desc.py" not in pypto_prompt
    assert "ai_op.py" not in pypto_prompt
    assert "ModelNew" not in pypto_prompt


def test_opencode_exporter_rejects_truncated_cli_json(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    fake_opencode = tmp_path / "fake_opencode.py"
    fake_opencode.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "if sys.argv[1:3] == ['export', 'ses_truncated']:\n"
        "    sys.stdout.write('{\"info\":{\"id\":\"ses_truncated\"},\"messages\":[{\"parts\":[{\"text\":\"cut')\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    fake_opencode.chmod(0o755)

    result = export_session_to_markdown(
        session_id="ses_truncated",
        output_file=tmp_path / "session.md",
        opencode_bin=str(fake_opencode),
        cwd=tmp_path,
    )

    assert result.status == "error"
    assert "JSON" in result.message
    assert not (tmp_path / "session.md").exists()


def test_opencode_exporter_exports_sqlite_child_sessions(monkeypatch, tmp_path):
    db_path = tmp_path / "opencode.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "create table session ("
            "id text primary key, parent_id text, title text, directory text, version text, "
            "time_created integer, time_updated integer)"
        )
        conn.execute(
            "create table message ("
            "id text primary key, session_id text, time_created integer, "
            "time_updated integer, data text)"
        )
        conn.execute(
            "create table part ("
            "id text primary key, message_id text, session_id text, "
            "time_created integer, time_updated integer, data text)"
        )
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?, ?, ?)",
            ("ses_parent", None, "parent transcript", str(tmp_path), "test", 1, 4),
        )
        conn.execute(
            "insert into session values (?, ?, ?, ?, ?, ?, ?)",
            ("ses_child", "ses_parent", "child transcript", str(tmp_path), "test", 2, 6),
        )
        conn.execute(
            "insert into message values (?, ?, ?, ?, ?)",
            (
                "msg_parent",
                "ses_parent",
                1,
                2,
                json.dumps(
                    {
                        "role": "assistant",
                        "providerID": "p",
                        "modelID": "m",
                        "tokens": {"total": 3, "input": 1, "output": 2},
                        "cost": 0.1,
                        "time": {},
                    }
                ),
            ),
        )
        conn.execute(
            "insert into part values (?, ?, ?, ?, ?, ?)",
            (
                "prt_parent",
                "msg_parent",
                "ses_parent",
                1,
                2,
                json.dumps({"type": "text", "text": "parent text"}),
            ),
        )
        conn.execute(
            "insert into message values (?, ?, ?, ?, ?)",
            (
                "msg_child",
                "ses_child",
                2,
                3,
                json.dumps({"role": "assistant", "time": {}}),
            ),
        )
        conn.execute(
            "insert into part values (?, ?, ?, ?, ?, ?)",
            (
                "prt_child",
                "msg_child",
                "ses_child",
                2,
                3,
                json.dumps({"type": "tool", "tool": "read", "state": {"status": "completed", "output": "child tool output"}}),
            ),
        )

    monkeypatch.setenv("OPENCODE_DB", str(db_path))
    result = export_session_to_markdown(
        session_id="ses_parent",
        output_file=tmp_path / "session.md",
        output_dir=tmp_path / "session_export",
        raw_json_file=tmp_path / "session.json",
        opencode_bin="/missing/opencode",
        cwd=tmp_path,
    )

    assert result.ok, result.to_dict()
    assert result.node_session_count == 2
    assert result.tree_updated_at_ms == 6
    assert result.token_usage["total"] == 3
    assert result.token_usage["session_count"] == 2
    assert result.session_tree_file.is_file()
    assert result.nodes_dir.is_dir()
    assert "child tool output" in result.markdown_file.read_text(encoding="utf-8")
    exported_json = json.loads(result.json_file.read_text(encoding="utf-8"))
    assert exported_json["children"][0]["info"]["id"] == "ses_child"


def test_pypto_adapter_requires_standard_stanford_submission_or_converter(tmp_path):
    source_dir = tmp_path / "artifact"
    source_dir.mkdir()
    source_dir.joinpath("ReLU_impl.py").write_text(
        "import pypto\n\n"
        "def ReLU_wrapper(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    source_dir.joinpath("ReLU_pypto_impl.py").write_text(
        "from ReLU_impl import ReLU_wrapper\n\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return ReLU_wrapper(x)\n",
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Relu",
        rel_path="level1/19_ReLU",
        files={},
        metadata={},
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=source_dir, files={"source_dir": source_dir})

    with pytest.raises(FileNotFoundError, match="converter agent"):
        _build_submission(PyptoToStanfordConverter(), case, output, output_dir=tmp_path / "submission")


def test_pypto_adapter_accepts_explicit_stanford_ai_op_from_converter(tmp_path):
    source_dir = tmp_path / "conversion_artifact" / "submission"
    source_dir.mkdir(parents=True)
    source_dir.joinpath("ReLU_impl.py").write_text(
        "# pypto\n\n"
        "def ReLU_wrapper(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    source_dir.joinpath("ai_op.py").write_text(
        "from ReLU_impl import ReLU_wrapper\n\n"
        "class ModelNew:\n"
        "    def forward(self, x):\n"
        "        return ReLU_wrapper(x)\n",
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Relu",
        rel_path="level1/19_ReLU",
        files={},
        metadata={},
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=source_dir.parent, files={"source_dir": source_dir})

    submission = _build_submission(PyptoToStanfordConverter(), case, output, output_dir=tmp_path / "submission")

    ai_op = submission.source_dir / "ai_op.py"
    assert ai_op.is_file()
    ai_op_text = ai_op.read_text(encoding="utf-8")
    assert "_sys.path.insert(0, _op_dir)" in ai_op_text
    assert "from ReLU_impl import ReLU_wrapper" in ai_op_text
    assert "class ModelNew" in ai_op_text
    assert "get_init_inputs" not in ai_op_text
    assert "get_inputs" not in ai_op_text
    assert submission.source_dir.joinpath("ReLU_impl.py").is_file()


def test_pypto_adapter_materializes_stanford_ai_op_before_cannbench_shape_check(tmp_path):
    task_path = tmp_path / "task.py"
    task_path.write_text(
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "def get_init_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )
    source_dir = tmp_path / "conversion_artifact" / "submission"
    source_dir.joinpath("cann_bench").mkdir(parents=True)
    source_dir.joinpath("cann_bench", "__init__.py").write_text("", encoding="utf-8")
    source_dir.joinpath("build.sh").write_text("#!/usr/bin/env bash\nset -e\n", encoding="utf-8")
    source_dir.joinpath("Foo_impl.py").write_text("# pypto\ndef Foo_wrapper(x):\n    return x\n", encoding="utf-8")
    source_dir.joinpath("ai_op.py").write_text(
        "import torch.nn as nn\n"
        "from Foo_impl import Foo_wrapper\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "    def forward(self, x):\n"
        "        return Foo_wrapper(x)\n",
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="level1/Foo",
        files={"task": task_path},
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=source_dir.parent, files={"source_dir": source_dir})

    submission = _build_submission(PyptoToStanfordConverter(), case, output, output_dir=tmp_path / "submission")

    ai_op_text = submission.source_dir.joinpath("ai_op.py").read_text(encoding="utf-8")
    assert "_sys.path.insert(0, _op_dir)" in ai_op_text
    assert submission.source_dir.joinpath("build.sh").is_file()


def test_config_runner_uses_isolated_converter_agent_for_raw_pypto_output(monkeypatch, tmp_path):
    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", "/data/pto-isa")
    bench_root = tmp_path / "bench"
    task_path = bench_root / "thirdparty" / "KernelBench" / "KernelBench" / "level1" / "19_ReLU.py"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        "import torch.nn as nn\n"
        "\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "\n"
        "def get_init_inputs():\n"
        "    return []\n"
        "\n"
        "def get_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.mkdir()
    raw_dir.joinpath("ReLU_impl.py").write_text(
        "# pypto\n\n"
        "def ReLU_wrapper(x):\n"
        "    return x\n",
        encoding="utf-8",
    )
    raw_dir.joinpath("ReLU_pypto_impl.py").write_text("# legacy bridge\n", encoding="utf-8")
    raw_dir.joinpath("test_ReLU.py").write_text("# test helper\n", encoding="utf-8")
    raw_dir.joinpath("ReLU_golden.py").write_text("# golden helper\n", encoding="utf-8")
    raw_dir.joinpath("modules").mkdir()
    raw_dir.joinpath("modules", "helper.py").write_text("# helper\n", encoding="utf-8")

    def fake_eval(self, *, bench_name, source_dir, task_selector, reports_dir, device_id=None, extra_args=None):
        return CannBenchEvalResult(
            returncode=0,
            command=self.build_eval_command(
                bench_name=bench_name,
                source_dir=source_dir,
                task_selector=task_selector,
                reports_dir=reports_dir,
                device_id=device_id,
                extra_args=extra_args,
            ),
            reports_dir=Path(reports_dir),
        )

    monkeypatch.setattr(CannBenchClient, "eval_submission", fake_eval)

    class FakeConverterRunner:
        type = "fake-converter"

        def run(self, prompt: RunnerPrompt) -> Artifact:
            prompt.output_dir.mkdir(parents=True, exist_ok=True)
            prompt.output_dir.joinpath("PROMPT.md").write_text(prompt.text, encoding="utf-8")
            submission_dir = prompt.output_dir / "submission"
            submission_dir.mkdir(parents=True, exist_ok=True)
            raw_impl = prompt.output_dir / "input" / "raw" / "ReLU_impl.py"
            submission_dir.joinpath("ReLU_impl.py").write_text(raw_impl.read_text(encoding="utf-8"), encoding="utf-8")
            submission_dir.joinpath("ai_op.py").write_text(
                "import torch.nn as nn\n"
                "from ReLU_impl import ReLU_wrapper\n"
                "class ModelNew(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "    def forward(self, x):\n"
                "        return ReLU_wrapper(x)\n",
                encoding="utf-8",
            )
            return Artifact(status=AGENT_SUCCESS, workdir=prompt.output_dir, files={"source_dir": submission_dir})

    class FakeGenerator:
        type = "pypto"

        def generate(self, task: GeneratorInput) -> Artifact:
            return Artifact(status=AGENT_SUCCESS, workdir=raw_dir, files={"source_dir": raw_dir})

    monkeypatch.setattr(config_runner, "create_generator", lambda generator_type, cfg: FakeGenerator())
    monkeypatch.setattr(config_runner, "create_runner", lambda runner_type, cfg: FakeConverterRunner())
    cfg = {
        "agent": {"type": "pypto"},
        "benchmark": {
            "root": str(bench_root),
            "name": "stanford",
            "tasks": ["thirdparty/KernelBench/KernelBench/level1/19_ReLU.py"],
        },
    }

    result = run_from_mapping(
        cfg,
        runtime={
            "output": str(tmp_path / "run"),
            "workspace": str(tmp_path / "pypto"),
        },
    )

    assert result.ok
    assert result.generated_artifact is not None
    assert result.conversion_artifact is not None
    assert result.converter_prompt_file is not None
    converter_prompt = result.converter_prompt_file.read_text(encoding="utf-8")
    assert "转换" in converter_prompt
    assert str(raw_dir) not in converter_prompt
    assert "pypto_impl" not in converter_prompt
    assert "只读取 input_dir 下的文件" in converter_prompt
    assert "不要写到 /tmp" in converter_prompt
    assert "不要实例化 ModelNew" in converter_prompt
    assert "不要调用 ModelNew.forward" in converter_prompt
    assert "触发 PyPTO/NPU 执行" in converter_prompt
    assert "input/raw/ReLU_impl.py" in converter_prompt
    assert "state_dict 键、shape 和 dtype 必须完全一致" in converter_prompt
    assert "相对导入" in converter_prompt
    assert "build.sh" not in converter_prompt
    assert "cann_bench" not in converter_prompt
    isolated_input = result.converter_prompt_file.parent / "input"
    assert isolated_input.joinpath("task", task_path.name).is_file()
    assert isolated_input.joinpath("raw", "ReLU_impl.py").is_file()
    assert not isolated_input.joinpath("raw", "ReLU_pypto_impl.py").exists()
    assert not isolated_input.joinpath("raw", "test_ReLU.py").exists()
    assert not isolated_input.joinpath("raw", "ReLU_golden.py").exists()
    assert not isolated_input.joinpath("raw", "modules").exists()
    assert result.submission.source_dir.joinpath("ai_op.py").is_file()
    assert result.submission.source_dir.joinpath("ReLU_impl.py").is_file()


def test_pypto_conversion_input_includes_nested_module_impls(tmp_path):
    task_path = tmp_path / "task.py"
    task_path.write_text("class Model:\n    pass\n", encoding="utf-8")
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.joinpath("modules").mkdir(parents=True)
    raw_dir.joinpath("modules", "Block_impl.py").write_text("# pypto\n", encoding="utf-8")
    raw_dir.joinpath("modules", "Block_golden.py").write_text("# golden\n", encoding="utf-8")
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Nested",
        rel_path="level3/Nested",
        files={"task": task_path},
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=raw_dir, files={"source_dir": raw_dir})

    prompt = PyptoToStanfordConverter().build_conversion_prompt(
        "stanford",
        case,
        output,
        workdir=tmp_path / "convert",
        output_dir=tmp_path / "convert" / "artifact",
        submission_dir=tmp_path / "submission",
    )

    assert prompt.output_dir.joinpath("input", "raw", "modules", "Block_impl.py").is_file()
    assert not prompt.output_dir.joinpath("input", "raw", "modules", "Block_golden.py").exists()
    assert "input/raw/modules/Block_impl.py" in prompt.text
    assert "build.sh" not in prompt.text
    assert "cann_bench" not in prompt.text


def test_pypto_conversion_prompt_uses_bench_specific_contract(tmp_path):
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.mkdir()
    raw_dir.joinpath("Foo_impl.py").write_text("# pypto\n", encoding="utf-8")
    output = Artifact(status=AGENT_SUCCESS, workdir=raw_dir, files={"source_dir": raw_dir})

    stanford_task = tmp_path / "stanford_task.py"
    stanford_task.write_text("class Model:\n    pass\n", encoding="utf-8")
    stanford_case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="level1/Foo",
        files={"task": stanford_task},
    )
    stanford_prompt = PyptoToStanfordConverter().build_conversion_prompt(
        "stanford",
        stanford_case,
        output,
        workdir=tmp_path / "stanford_convert",
        output_dir=tmp_path / "stanford_convert" / "artifact",
        submission_dir=tmp_path / "stanford_submission",
    )

    assert "标准 Stanford/KernelBench 提交" in stanford_prompt.text
    assert "ai_op.py" in stanford_prompt.text
    assert "ModelNew" in stanford_prompt.text
    assert "build.sh" not in stanford_prompt.text
    assert "cann_bench" not in stanford_prompt.text

    proto = tmp_path / "proto.yaml"
    proto.write_text("schema: foo(x) -> y\n", encoding="utf-8")
    cann_case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="tasks/level1/foo",
        files={"proto": proto},
    )
    cann_prompt = PyptoToCannConverter().build_conversion_prompt(
        "cann",
        cann_case,
        output,
        workdir=tmp_path / "cann_convert",
        output_dir=tmp_path / "cann_convert" / "artifact",
        submission_dir=tmp_path / "cann_submission",
    )

    assert "标准 cann-bench source_dir 提交" in cann_prompt.text
    assert "build.sh" in cann_prompt.text
    assert "cann_bench" in cann_prompt.text
    assert "转换边界" in cann_prompt.text
    assert "不能修复 PyPTO kernel/wrapper" in cann_prompt.text
    assert "不能改变计算路径" in cann_prompt.text
    assert "不要在转换阶段修复或规避 PyPTO 的 UB/片上内存问题" in cann_prompt.text
    assert "TensorErr::TENSOR_MEMORY_ALLOCATION" in cann_prompt.text
    assert "不要检查或搜索生成阶段 proof" in cann_prompt.text
    assert "不要读取、glob 或复制 `.orchestrator_state.json`" in cann_prompt.text
    assert "历史 submission" in cann_prompt.text
    assert "不要把原始 wrapper 直接 alias" in cann_prompt.text
    assert "无类型标注的 thin forwarder" in cann_prompt.text
    assert "按原始 wrapper 的实际签名绑定" in cann_prompt.text
    assert "优先用 positional 转发" in cann_prompt.text
    assert "entry -> raw wrapper 的绑定关系" in cann_prompt.text
    assert "拥有相同 `inspect.signature(raw_wrapper)` 的 spy/stub" in cann_prompt.text
    assert "不会抛 `TypeError`" in cann_prompt.text
    assert "return_annotation is inspect.Signature.empty" in cann_prompt.text
    assert "build.sh 必须在提交目录下生成 `dist/cann_bench*.whl`" in cann_prompt.text
    assert "只做 Python syntax/import 检查的 build.sh 是无效提交" in cann_prompt.text
    assert "setup.py 或 pyproject.toml" in cann_prompt.text
    assert "ai_op.py" not in cann_prompt.text
    assert "ModelNew" not in cann_prompt.text
    assert "Stanford" not in cann_prompt.text


def test_generic_conversion_prompt_renders_fallback_contract(tmp_path):
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.mkdir()
    impl_path = raw_dir / "impl.py"
    impl_path.write_text("# generated\n", encoding="utf-8")
    output = Artifact(status=AGENT_SUCCESS, workdir=raw_dir, files={"impl": impl_path})
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="tasks/level1/foo",
        files={},
    )

    from auto_pipeline.converter.base import BaseConverter

    class GenericConverter(BaseConverter):
        name = "generic"
        source_generator = "generic"
        target_benchmark = "cann"
        timeout_sec = 7200
        env = {}

    prompt = GenericConverter().build_conversion_prompt(
        "cann",
        case,
        output,
        workdir=tmp_path / "convert",
        output_dir=tmp_path / "convert" / "artifact",
        submission_dir=tmp_path / "submission",
    )

    assert "将 Foo 的 generic 生成产物转换为 cann benchmark submission。" in prompt.text
    assert f"- workdir: {raw_dir}" in prompt.text
    assert f"- impl: {impl_path}" in prompt.text
    expected_submission_path = tmp_path / "convert" / "artifact" / "submission"
    assert f"- 将转换后的提交写到 {expected_submission_path}" in prompt.text


def test_pypto_adapter_rejects_relative_stanford_ai_op_import(tmp_path):
    task_path = tmp_path / "task.py"
    task_path.write_text(
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "def get_init_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )
    source_dir = tmp_path / "submission"
    source_dir.mkdir()
    source_dir.joinpath("Foo_impl.py").write_text("# pypto\ndef Foo_wrapper(x):\n    return x\n", encoding="utf-8")
    source_dir.joinpath("ai_op.py").write_text(
        "import torch.nn as nn\n"
        "from .Foo_impl import Foo_wrapper\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "    def forward(self, x):\n"
        "        return Foo_wrapper(x)\n",
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="level1/Foo",
        files={"task": task_path},
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=source_dir.parent, files={"source_dir": source_dir})

    with pytest.raises(ValueError, match="relative import"):
        _build_submission(PyptoToStanfordConverter(), case, output, output_dir=tmp_path / "converted")


def test_pypto_adapter_rejects_stanford_state_dict_mismatch(tmp_path):
    task_path = tmp_path / "task.py"
    task_path.write_text(
        "import torch.nn as nn\n"
        "class Model(nn.Module):\n"
        "    def __init__(self, in_features, out_features):\n"
        "        super().__init__()\n"
        "        self.gemm = nn.Linear(in_features, out_features)\n"
        "    def forward(self, x):\n"
        "        return self.gemm(x)\n"
        "def get_init_inputs():\n"
        "    return [4, 8]\n",
        encoding="utf-8",
    )
    source_dir = tmp_path / "submission"
    source_dir.mkdir()
    source_dir.joinpath("Foo_impl.py").write_text("# pypto\ndef Foo_wrapper(x, weight, bias):\n    return x\n", encoding="utf-8")
    source_dir.joinpath("ai_op.py").write_text(
        "import torch\n"
        "import torch.nn as nn\n"
        "from Foo_impl import Foo_wrapper\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self, in_features, out_features):\n"
        "        super().__init__()\n"
        "        self.weight = nn.Parameter(torch.empty(out_features, in_features))\n"
        "        self.bias = nn.Parameter(torch.empty(out_features))\n"
        "    def forward(self, x):\n"
        "        return Foo_wrapper(x, self.weight, self.bias)\n",
        encoding="utf-8",
    )
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="level2/Foo",
        files={"task": task_path},
    )
    output = Artifact(status=AGENT_SUCCESS, workdir=source_dir.parent, files={"source_dir": source_dir})

    with pytest.raises(ValueError, match="state_dict keys"):
        _build_submission(PyptoToStanfordConverter(), case, output, output_dir=tmp_path / "converted")


def test_stanford_matcher_uses_task_init_inputs_for_source_ai_op(tmp_path):
    bench_root = tmp_path / "KernelBench"
    level_dir = bench_root / "level1"
    level_dir.mkdir(parents=True)
    level_dir.joinpath("Argmax_over_a_dimension.py").write_text(
        "class Model:\n"
        "    def __init__(self, dim):\n"
        "        self.dim = dim\n"
        "    def forward(self, x):\n"
        "        return x\n"
        "\n"
        "def get_init_inputs():\n"
        "    return [1]\n"
        "\n"
        "def get_inputs():\n"
        "    return []\n",
        encoding="utf-8",
    )

    source_dir = tmp_path / "submission"
    source_dir.mkdir()
    source_dir.joinpath("ai_op.py").write_text(
        "class ModelNew:\n"
        "    def __init__(self, dim):\n"
        "        self.dim = dim\n"
        "    def forward(self, x):\n"
        "        return x\n",
        encoding="utf-8",
    )

    matcher = StanfordMatcher(
        operator_loader=StanfordTaskLoader(str(bench_root)),
        source_dir=str(source_dir),
    )

    ai_func = matcher.load_ai_operator("ArgmaxOverADimension")

    assert ai_func.model.dim == 1


def test_config_runner_pypto_eval_env_is_empty_perf_strategy_via_cli():
    # perf 策略不再通过环境变量传递，改为 CLI --perf-metric-strategy
    assert config_runner._eval_env(agent_type="pypto", bench_name="cann") == {}
    assert config_runner._eval_env(agent_type="akg-agent", bench_name="cann") == {}
    # _is_pypto_cann_eval 仍用于判断是否传递 trace_view 策略
    assert config_runner._is_pypto_cann_eval(agent_type="pypto", bench_name="cann") is True
    assert config_runner._is_pypto_cann_eval(agent_type="akg-agent", bench_name="cann") is False


def test_config_runner_requires_pypto_tile_lib_environment(monkeypatch):
    cfg = {
        "agent": {"type": "pypto"},
        "benchmark": {"name": "cann", "tasks": ["tasks/level1/exp"]},
    }
    monkeypatch.delenv("PTO_TILE_LIB_CODE_PATH", raising=False)

    with pytest.raises(ValueError, match="PTO_TILE_LIB_CODE_PATH"):
        config_runner._parse_config(cfg, config_path=None, runtime={"workspace": "/tmp/pypto"})

    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", "/data/pto-isa")
    spec = config_runner._parse_config(cfg, config_path=None, runtime={"workspace": "/tmp/pypto"})
    assert spec.agent_type == "pypto"


def test_config_runner_wires_simplified_pypto_config_and_derived_paths(monkeypatch, tmp_path):
    tile_path = tmp_path / "pto-isa"
    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", str(tile_path))
    pypto_repo = tmp_path / "pypto"
    generator_cfgs = []
    runner_types = []

    def fake_eval(self, *, bench_name, source_dir, task_selector, reports_dir, device_id=None, extra_args=None):
        # perf 策略不再通过环境变量传递，改为通过 extra_args 里的
        # --perf-metric-strategy CLI 参数
        assert self.extra_env == {}
        assert "--perf-metric-strategy" in list(extra_args)
        assert "trace_view" in list(extra_args)
        assert device_id == 5
        assert Path(reports_dir) == tmp_path / "run" / "gelu" / "kernel_eval"
        assert "--no-subprocess-isolation" in list(extra_args)
        assert "--op-timeout-sec" in list(extra_args)
        command = self.build_eval_command(
            bench_name=bench_name,
            source_dir=source_dir,
            task_selector=task_selector,
            reports_dir=reports_dir,
            device_id=device_id,
            extra_args=extra_args,
        )
        return CannBenchEvalResult(returncode=0, command=command, reports_dir=Path(reports_dir))

    monkeypatch.setattr(CannBenchClient, "eval_submission", fake_eval)

    class FakeGenerator:
        type = "pypto"

        def generate(self, task: GeneratorInput) -> Artifact:
            raw_dir = task.output_dir
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.joinpath("GeLU_impl.py").write_text("# pypto\n", encoding="utf-8")
            assert task.workdir == tmp_path / "run" / "gelu" / "work"
            assert task.env == {}
            return Artifact(status=AGENT_SUCCESS, workdir=raw_dir, files={"source_dir": raw_dir})

    class FakeConverterRunner:
        type = "fake-converter"

        def run(self, prompt: RunnerPrompt) -> Artifact:
            prompt.output_dir.mkdir(parents=True, exist_ok=True)
            prompt.output_dir.joinpath("PROMPT.md").write_text(prompt.text, encoding="utf-8")
            submission_dir = prompt.output_dir / "submission"
            package_dir = submission_dir / "cann_bench"
            package_dir.mkdir(parents=True, exist_ok=True)
            submission_dir.joinpath("build.sh").write_text("#!/usr/bin/env bash\nset -e\n", encoding="utf-8")
            package_dir.joinpath("__init__.py").write_text("", encoding="utf-8")
            package_dir.joinpath("gelu.py").write_text(
                "import pypto\n"
                "def gelu(x, approximate='none'):\n"
                "    return x\n",
                encoding="utf-8",
            )
            assert prompt.cwd == tmp_path / "run" / "gelu" / "convert"
            assert prompt.env == {}
            return Artifact(status=AGENT_SUCCESS, workdir=prompt.output_dir, files={"source_dir": submission_dir})

    def fake_create_generator(generator_type, cfg):
        generator_cfgs.append((generator_type, cfg))
        return FakeGenerator()

    def fake_create_runner(runner_type, cfg):
        runner_types.append((runner_type, cfg))
        return FakeConverterRunner()

    monkeypatch.setattr(config_runner, "create_generator", fake_create_generator)
    monkeypatch.setattr(config_runner, "create_runner", fake_create_runner)

    cfg = {
        "agent": {"type": "pypto"},
        "benchmark": {"root": str(REPO_ROOT), "name": "cann", "tasks": ["tasks/level1/gelu"]},
    }

    result = run_from_mapping(
        cfg,
        runtime={
            "output": str(tmp_path / "run"),
            "workspace": str(pypto_repo),
            "model": "deepseek/deepseek-v4-pro",
            "devices": [5],
            "parallel": 1,
        },
    )

    assert result.ok
    assert result.submission.source_dir == tmp_path / "run" / "gelu" / "submission"
    assert result.eval_result.reports_dir == tmp_path / "run" / "gelu" / "kernel_eval"
    assert (tmp_path / "run" / "gelu" / "benchmark_result.json").is_file()
    assert generator_cfgs[0][0] == "pypto"
    assert generator_cfgs[0][1]["repo_root"] == pypto_repo.resolve()
    assert generator_cfgs[0][1]["model"] == "deepseek/deepseek-v4-pro"
    assert generator_cfgs[0][1]["device_id"] == 5
    assert generator_cfgs[0][1]["device_mode"] == "pool"
    assert generator_cfgs[0][1]["worktree_root"] == tmp_path / "run" / "pypto_worktrees"
    assert "env" not in generator_cfgs[0][1]
    assert runner_types == [("opencode", {"model": "deepseek/deepseek-v4-pro"})]


def test_config_runner_device_pool_uses_tasks_without_case_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", "/data/pto-isa")
    calls = []
    executor_instances = []

    class ImmediateExecutor:
        def __init__(self, *, max_workers):
            self.max_workers = max_workers
            executor_instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args):
            future = Future()
            try:
                future.set_result(fn(*args))
            except BaseException as exc:
                future.set_exception(exc)
            return future

    def fake_run_task(spec, task, *, device_id):
        calls.append((spec, task, device_id))
        task = CannBenchCase(
            bench_name=spec.bench_name,
            task_dir=tmp_path,
            operator=task.selector,
            rel_path=task.selector,
            files={},
        )
        return PipelineRunResult(
            case=task,
            submission=Submission("cann", task.operator, tmp_path / "submission"),
            eval_result=CannBenchEvalResult(
                returncode=0, command=["eval", task.operator], reports_dir=tmp_path / "reports"
            ),
        )

    monkeypatch.setattr(config_runner, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(config_runner, "_run_task", fake_run_task)
    report_path = tmp_path / "batch.json"
    cfg = {
        "agent": {"type": "pypto"},
        "benchmark": {
            "name": "cann",
            "tasks": ["tasks/level1/exp", "tasks/level1/gelu", "tasks/level1/mish"],
        },
    }

    entries = config_runner.run_cases_from_mapping(
        cfg,
        report_path=report_path,
        runtime={
            "output": str(tmp_path / "run"),
            "workspace": str(tmp_path / "pypto"),
            "devices": "0,3-4",
            "parallel": 2,
        },
    )

    assert executor_instances[0].max_workers == 2
    assert [entry["name"] for entry in entries] == ["exp", "gelu", "mish"]
    assert len(calls) == 3
    assert {calls[0][2], calls[1][2]} == {0, 3}
    assert calls[2][2] == 4
    assert [call[1].selector for call in calls] == [
        "tasks/level1/exp",
        "tasks/level1/gelu",
        "tasks/level1/mish",
    ]
    assert [call[1].root_dir for call in calls] == [
        tmp_path / "run" / "exp",
        tmp_path / "run" / "gelu",
        tmp_path / "run" / "mish",
    ]
    batch = json.loads(report_path.read_text(encoding="utf-8"))
    assert batch["total_cases"] == 3
    assert batch["completed_cases"] == 3
    assert batch["running_cases"] == 0
    assert batch["pending_cases"] == 0
    assert batch["output"] == str(tmp_path / "run")


def test_config_runner_preserves_akg_triton_ascend_options_and_device(tmp_path):
    cfg = {
        "agent": {
            "type": "akg-agent",
            "backend": "ascend",
            "arch": "ascend910b4",
            "framework": "torch",
            "codegen_target": "triton_ascend",
            "workflow": "kernelgen_only_workflow",
            "verify_timeout": 1800,
        },
        "benchmark": {"name": "cann", "tasks": ["tasks/level1/exp", "tasks/level1/sigmoid"]},
    }

    spec = config_runner._parse_config(
        cfg,
        config_path=None,
        runtime={
            "output": str(tmp_path / "run"),
            "workspace": str(tmp_path / "akg"),
            "devices": [0, 2],
            "parallel": 2,
        },
    )
    generator_cfg = config_runner._generator_config(spec, device_id=2)

    assert spec.agent_type == "akg-agent"
    assert spec.workspace == (tmp_path / "akg").resolve()
    assert spec.devices == (0, 2)
    assert spec.parallel == 2
    assert [task.name for task in spec.tasks] == ["exp", "sigmoid"]
    assert generator_cfg == {
        "repo_root": (tmp_path / "akg").resolve(),
        "backend": "ascend",
        "arch": "ascend910b4",
        "framework": "torch",
        "codegen_target": "triton_ascend",
        "workflow": "kernelgen_only_workflow",
        "verify_timeout": 1800,
        "device_id": 2,
    }


def test_config_runner_rejects_legacy_sections():
    cfg = {
        "agent": {"type": "pypto"},
        "benchmark": {"name": "cann", "tasks": ["tasks/level1/exp"]},
        "generator": {"type": "pypto"},
    }

    with pytest.raises(ValueError, match="unsupported legacy config sections"):
        config_runner.run_cases_from_mapping(cfg, runtime={"workspace": "/tmp/pypto"})


def test_config_runner_rejects_runtime_values_in_yaml():
    cfg = {
        "output": "/tmp/run",
        "agent": {"type": "pypto", "repo_root": "/tmp/pypto"},
        "benchmark": {"name": "cann", "tasks": ["tasks/level1/exp"]},
    }

    with pytest.raises(ValueError, match="output"):
        config_runner.run_cases_from_mapping(cfg, runtime={"workspace": "/tmp/pypto"})

    del cfg["output"]
    with pytest.raises(ValueError, match="agent.repo_root"):
        config_runner.run_cases_from_mapping(cfg, runtime={"workspace": "/tmp/pypto"})


def test_config_runner_rejects_duplicate_devices(monkeypatch):
    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", "/data/pto-isa")
    cfg = {
        "agent": {"type": "pypto"},
        "benchmark": {"name": "cann", "tasks": ["tasks/level1/exp"]},
    }

    with pytest.raises(ValueError, match="duplicate device ids"):
        config_runner.run_cases_from_mapping(cfg, runtime={"workspace": "/tmp/pypto", "devices": [0, 0]})


def test_config_runner_rejects_parallel_without_devices(monkeypatch):
    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", "/data/pto-isa")
    cfg = {
        "agent": {"type": "pypto"},
        "benchmark": {"name": "cann", "tasks": ["tasks/level1/exp", "tasks/level1/gelu"]},
    }

    with pytest.raises(ValueError, match="devices is required"):
        config_runner.run_cases_from_mapping(cfg, runtime={"workspace": "/tmp/pypto", "parallel": 2})


def test_config_runner_generates_random_output_when_omitted(monkeypatch):
    monkeypatch.setenv("PTO_TILE_LIB_CODE_PATH", "/data/pto-isa")
    cfg = {
        "agent": {"type": "pypto"},
        "benchmark": {"name": "cann", "tasks": ["tasks/level1/exp"]},
    }

    spec = config_runner._parse_config(cfg, config_path=None, runtime={"workspace": "/tmp/pypto"})

    assert spec.output.parent == (REPO_ROOT / "benchmark_runs").resolve()
    assert spec.output.name.startswith("run_")
    assert spec.tasks[0].root_dir == spec.output / "exp"


def test_eval_args_pass_through_kernel_eval_options():
    args = build_eval_args(
        {
            "device": "cpu",
            "case_id": 1,
            "no_perf": True,
            "warmup": 0,
            "repeat": 1,
            "extra_args": ["--no-subprocess-isolation"],
        }
    )

    assert args == [
        "--case-id",
        "1",
        "--device",
        "cpu",
        "--warmup",
        "0",
        "--repeat",
        "1",
        "--no-perf",
        "--no-subprocess-isolation",
    ]
