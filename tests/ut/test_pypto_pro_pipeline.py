#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Unit tests for the PyPTO-Pro integration in auto_pipeline.

These tests mirror the PyPTO (non-Pro) tests in ``test_benchmark_pipeline.py``
but adapt expectations to PyPTO-Pro's four-stage workflow:

* Artifacts are ``test_{op}.py`` (kernel+test single file), ``{op}_golden.py``,
  ``SPEC.md`` — there is no ``{op}_impl.py``.
* No ``.orchestrator_state.json`` state machine; completion is judged by
  artifact existence and opencode return code.
* No ``perf_round`` concept.
* Metadata key is ``pypto_pro_status`` (not ``pypto_status``).
* Submission validation checks for ``pypto_pro`` in Python files.
"""

import json
import subprocess
from pathlib import Path

import pytest

from auto_pipeline.converter.registry import available_converters, create_converter
from auto_pipeline.core import (
    AGENT_SUCCESS,
    Artifact,
    CannBenchCase,
    GeneratorInput,
)
from auto_pipeline.generator.pypto_pro import PyptoProOrchestratorAgent
from auto_pipeline.generator.pypto_pro.converter import (
    PyptoProToCannConverter,
    PyptoProToStanfordConverter,
)
from auto_pipeline.generator.registry import available_generators, create_generator
from auto_pipeline.prompt.registry import build_case_material

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_submission(adapter, case, artifact, *, output_dir):
    return adapter.build_submission(
        case.bench_name, case, artifact, output_dir=output_dir
    )


def _generation_task(case, workdir, *, timeout_sec=30):
    material = build_case_material(case)
    workdir = Path(workdir)
    return GeneratorInput(
        case=case,
        material=material,
        workdir=workdir,
        output_dir=workdir / "artifact",
        timeout_sec=timeout_sec,
        title=f"pypto-pro:{case.operator}",
        metadata={
            "bench_name": case.bench_name,
            "operator": case.operator,
            "task_dir": str(case.task_dir),
            "schema": case.metadata.get("schema") or "",
            "case_preview": case.metadata.get("case_preview"),
        },
    )


def _write_fake_opencode_stanford(
    opencode, *, op="ReLU", root_dir="custom", with_bridge=False
):
    """Write a fake opencode script that produces PyPTO-Pro artifacts.

    Unlike PyPTO, there is no ``.orchestrator_state.json`` and no ``{op}_impl.py``.
    The kernel lives inside ``test_{op}.py``.
    """
    lines = [
        "#!/usr/bin/env python3",
        "import json",
        "import os",
        "import pathlib",
        "import sys",
        "root = pathlib.Path.cwd()",
    ]
    if with_bridge:
        lines += [
            "def emit_bridge(record):",
            "    bridge = os.environ.get('OPENCODE_SUBAGENT_BRIDGE_LOG')",
            "    if not bridge:",
            "        return",
            "    path = pathlib.Path(bridge)",
            "    path.parent.mkdir(parents=True, exist_ok=True)",
            "    with path.open('a', encoding='utf-8') as handle:",
            "        handle.write(json.dumps(record) + '\\n')",
        ]
    lines += [
        "if sys.argv[1:2] == ['run']:",
        "    (root / 'argv.json').write_text(json.dumps(sys.argv), encoding='utf-8')",
        "    (root / 'env.json').write_text(json.dumps({",
        "        'TMPDIR': os.environ.get('TMPDIR'),",
        "        'OPENCODE_PERMISSION': os.environ.get('OPENCODE_PERMISSION'),",
        "        'OPENCODE_SUBAGENT_BRIDGE_LOG': os.environ.get('OPENCODE_SUBAGENT_BRIDGE_LOG'),",
        "        'OPENCODE_CONFIG_CONTENT': os.environ.get('OPENCODE_CONFIG_CONTENT'),",
        "    }), encoding='utf-8')",
    ]
    if with_bridge:
        lines += [
            "    emit_bridge({'kind': 'plugin_loaded', 'time': 1, 'pid': 123})",
            "    emit_bridge({'kind': 'event', 'type': 'session.created', 'sessionID': 'ses_root', 'session': {'id': 'ses_root', 'title': 'fake root'}})",
            "    emit_bridge({'kind': 'event', 'type': 'session.created', 'sessionID': 'ses_child', 'parentID': 'ses_root', 'session': {'id': 'ses_child', 'title': 'fake subagent', 'parentID': 'ses_root'}})",
            "    emit_bridge({'kind': 'event', 'type': 'message.part.delta', 'sessionID': 'ses_child', 'delta': {'sessionID': 'ses_child', 'messageID': 'msg_child', 'partID': 'prt_text', 'field': 'text', 'text': 'child done'}})",
        ]
    lines += [
        "if sys.argv[1:3] == ['session', 'list']:",
        "    title = (root / 'last_title.txt').read_text(encoding='utf-8')",
        "    print(f'{title} ses_fake')",
        "    raise SystemExit(0)",
        "if sys.argv[1:3] == ['export', 'ses_fake']:",
        "    print(json.dumps({",
        "        'info': {'id': 'ses_fake', 'title': 'fake root', 'time': {'created': 1, 'updated': 4}},",
        "        'messages': [{'info': {'role': 'assistant', 'time': {}}, 'parts': [{'type': 'text', 'text': 'root transcript'}]}],",
        "        'children': [{",
        "            'info': {'id': 'ses_child', 'parent_id': 'ses_fake', 'title': 'fake subagent', 'time': {'created': 2, 'updated': 5}},",
        "            'messages': [{'info': {'role': 'assistant', 'time': {}}, 'parts': [{'type': 'text', 'text': 'child transcript'}]}]",
        "        }],",
        "    }))",
        "    raise SystemExit(0)",
        "if '--title' in sys.argv:",
        "    (root / 'last_title.txt').write_text(sys.argv[sys.argv.index('--title') + 1], encoding='utf-8')",
        f"op_dir = root / '{root_dir}' / '{op}'",
        "op_dir.mkdir(parents=True, exist_ok=True)",
        f"for name in ['SPEC.md', 'test_{op}.py', '{op}_golden.py']:",
        "    (op_dir / name).write_text('# pypto_pro\\n', encoding='utf-8')",
        "print(' '.join(sys.argv[1:6]))",
    ]
    opencode.write_text("\n".join(lines), encoding="utf-8")
    opencode.chmod(0o755)


def _make_stanford_case(tmp_path, op="ReLU"):
    task_dir = tmp_path / "case" / op
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
    task_dir.joinpath("REQUIRE.md").write_text(f"# {op}\n", encoding="utf-8")
    return CannBenchCase(
        bench_name="stanford",
        task_dir=task_dir,
        operator="TaskDesc",
        rel_path=f"level1/{op}",
        files={"task": task_path},
    )


def _make_cann_case(tmp_path, op="gelu"):
    task_dir = tmp_path / "tasks" / "level1" / op
    task_dir.mkdir(parents=True)
    proto = task_dir / "proto.yaml"
    proto.write_text(
        f"operator:\n  name: {op.capitalize()}\n  schema: {op}(x) -> y\n",
        encoding="utf-8",
    )
    cases = task_dir / "cases.yaml"
    cases.write_text("cases:\n  - shape: [16]\n", encoding="utf-8")
    golden = task_dir / "golden.py"
    golden.write_text("def golden(x):\n    return x\n", encoding="utf-8")
    desc = task_dir / "desc.md"
    desc.write_text(f"# {op}\n", encoding="utf-8")
    return CannBenchCase(
        bench_name="cann",
        task_dir=task_dir,
        operator=op.capitalize(),
        rel_path=f"tasks/level1/{op}",
        files={"proto": proto, "cases": cases, "golden": golden, "desc": desc},
        metadata={
            "schema": f"{op}(x) -> y",
            "case_preview": [{"shape": [16]}],
        },
    )


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_pypto_pro_generators_and_converters_are_registered():
    assert "pypto-pro" in available_generators()
    assert available_generators().count("pypto-pro") == 1

    names = available_converters()
    assert "pypto-pro -> cann" in names
    assert "pypto-pro -> stanford" in names

    assert isinstance(
        create_converter("pypto-pro", "cann", {}), PyptoProToCannConverter
    )
    assert isinstance(
        create_converter("pypto-pro", "stanford", {}), PyptoProToStanfordConverter
    )
    # Underscore normalisation
    assert create_converter("pypto_pro", "cann", {}).name == "pypto-pro-to-cann"
    assert create_converter("pypto_pro", "stanford", {}).name == "pypto-pro-to-stanford"


def test_pypto_pro_orchestrator_agent_is_registered(tmp_path):
    agent = create_generator("pypto-pro", {"repo_root": str(tmp_path)})
    model_agent = create_generator(
        "pypto-pro",
        {"repo_root": str(tmp_path), "model": "zai-coding-plan/glm-5.1"},
    )

    assert isinstance(agent, PyptoProOrchestratorAgent)
    assert isinstance(model_agent, PyptoProOrchestratorAgent)
    assert agent.agent == "pypto-pro-op-orchestrator"
    assert model_agent.opencode_model == "zai-coding-plan/glm-5.1"
    assert agent.type == "pypto-pro"
    # No perf_round attribute (unlike PyPTO)
    assert not hasattr(agent, "perf_round")


def test_pypto_pro_orchestrator_agent_requires_repo_root():
    with pytest.raises(ValueError, match="repo_root"):
        create_generator("pypto-pro", {})


# ---------------------------------------------------------------------------
# Orchestrator contract tests
# ---------------------------------------------------------------------------


def test_pypto_pro_orchestrator_agent_runs_real_agent_contract(tmp_path):
    repo_root = tmp_path / "pypto_pro_repo"
    repo_root.mkdir()
    opencode = tmp_path / "fake_opencode.py"
    _write_fake_opencode_stanford(opencode, op="ReLU", with_bridge=True)

    case = _make_stanford_case(tmp_path, op="ReLU")
    task = _generation_task(case, tmp_path / "work")
    agent = PyptoProOrchestratorAgent(
        pypto_repo_root=repo_root,
        opencode_bin=str(opencode),
        opencode_model="zai-coding-plan/glm-5.1",
    )

    output = agent.generate(task)

    assert output.ok
    assert output.workdir == repo_root / "custom" / "ReLU"
    assert output.files["source_dir"] == repo_root / "custom" / "ReLU"
    # PyPTO-Pro artifacts: test_{op}.py, {op}_golden.py, SPEC.md (NO _impl.py)
    assert output.files["test_ReLU.py"].is_file()
    assert output.files["ReLU_golden.py"].is_file()
    assert output.files["SPEC.md"].is_file()
    assert "ReLU_impl.py" not in output.files
    assert output.metadata["pypto_pro_status"] == "success"
    assert not (task.output_dir / "PROMPT.md").exists() or (
        task.output_dir / "PROMPT.md"
    ).read_text(encoding="utf-8")
    prompt_text = (task.output_dir / "PROMPT.md").read_text(encoding="utf-8")
    assert "pypto-pro-op-orchestrator" in prompt_text
    assert "工作目录: `custom/ReLU/`" in prompt_text
    assert "不要为 cann-bench/Stanford submission 做输出格式对齐" in prompt_text
    assert "不要写到 `/tmp`" in prompt_text
    assert "前台、有界执行" in prompt_text
    # Entry function convention
    assert "入口函数约定" in prompt_text
    assert "test_ReLU.py" in prompt_text
    # No state machine references
    assert ".orchestrator_state.json" not in prompt_text
    # No perf_round references
    assert "perf_round" not in prompt_text.lower()
    assert "性能调优轮次" not in prompt_text
    # No PyPTO (non-Pro) references
    assert "pypto-op-orchestrator" not in prompt_text
    assert "ai_op.py" not in prompt_text
    assert "ModelNew" not in prompt_text
    # Task input files copied to workspace
    assert (repo_root / "custom" / "ReLU" / "task_desc.py").is_file()
    assert (repo_root / "custom" / "ReLU" / "REQUIRE.md").read_text(
        encoding="utf-8"
    ) == "# ReLU\n"
    # opencode invoked without --agent (build auto-loads primary AGENTS.md)
    log_text = output.log_file.read_text(encoding="utf-8")
    assert "--agent" not in log_text
    assert "-m zai-coding-plan/glm-5.1" in log_text
    argv = json.loads((repo_root / "argv.json").read_text(encoding="utf-8"))
    assert argv[argv.index("-m") + 1] == "zai-coding-plan/glm-5.1"
    env = json.loads((repo_root / "env.json").read_text(encoding="utf-8"))
    assert env["TMPDIR"] == str(repo_root / "custom" / "ReLU" / ".tmp")
    assert json.loads(env["OPENCODE_PERMISSION"])["external_directory"] == "deny"
    assert env["OPENCODE_SUBAGENT_BRIDGE_LOG"] == str(
        task.output_dir / "opencode-live" / "events.jsonl"
    )
    assert (repo_root / "custom" / "ReLU" / ".tmp").is_dir()
    # No .orchestrator_state.json should exist (unlike PyPTO)
    assert not (repo_root / "custom" / "ReLU" / ".orchestrator_state.json").exists()
    # Live bridge
    bridge = output.metadata["opencode_live_bridge"]
    assert bridge["status"] == "captured"
    assert bridge["node_session_count"] == 2
    assert bridge["subagent_session_count"] == 1
    # Session export
    assert output.metadata["opencode_session"]["status"] == "exported"
    assert Path(output.metadata["opencode_session"]["markdown_file"]).is_file()


def test_pypto_pro_orchestrator_agent_does_not_write_state_machine(tmp_path):
    """PyPTO-Pro has no state machine; completion is judged by artifacts only."""
    repo_root = tmp_path / "pypto_pro_repo"
    repo_root.mkdir()
    opencode = tmp_path / "fake_opencode.py"
    # Fake opencode that produces artifacts but NO .orchestrator_state.json
    opencode.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "root = pathlib.Path.cwd()\n"
        "if sys.argv[1:3] == ['session', 'list']:\n"
        "    print('title ses_fake')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:3] == ['export', 'ses_fake']:\n"
        "    print(json.dumps({'info': {'id': 'ses_fake', 'title': 'fake'}, 'messages': [], 'children': []}))\n"
        "    raise SystemExit(0)\n"
        "if '--title' in sys.argv:\n"
        "    (root / 'last_title.txt').write_text(sys.argv[sys.argv.index('--title') + 1], encoding='utf-8')\n"
        "op_dir = root / 'custom' / 'ReLU'\n"
        "op_dir.mkdir(parents=True, exist_ok=True)\n"
        "for name in ['SPEC.md', 'test_ReLU.py', 'ReLU_golden.py']:\n"
        "    (op_dir / name).write_text('# pypto_pro\\n', encoding='utf-8')\n"
        "# Deliberately NO .orchestrator_state.json\n",
        encoding="utf-8",
    )
    opencode.chmod(0o755)

    case = _make_stanford_case(tmp_path, op="ReLU")
    task = _generation_task(case, tmp_path / "work")
    agent = PyptoProOrchestratorAgent(
        pypto_repo_root=repo_root, opencode_bin=str(opencode)
    )

    output = agent.generate(task)

    # Should succeed without .orchestrator_state.json (unlike PyPTO which requires it)
    assert output.ok
    assert output.metadata["pypto_pro_status"] == "success"


def test_pypto_pro_orchestrator_agent_fails_on_missing_artifacts(tmp_path):
    """When test_{op}.py is missing, the agent should report artifact_missing."""
    repo_root = tmp_path / "pypto_pro_repo"
    repo_root.mkdir()
    opencode = tmp_path / "fake_opencode.py"
    opencode.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "root = pathlib.Path.cwd()\n"
        "if sys.argv[1:3] == ['session', 'list']:\n"
        "    print('title ses_fake')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:3] == ['export', 'ses_fake']:\n"
        "    print(json.dumps({'info': {'id': 'ses_fake', 'title': 'fake'}, 'messages': [], 'children': []}))\n"
        "    raise SystemExit(0)\n"
        "if '--title' in sys.argv:\n"
        "    (root / 'last_title.txt').write_text(sys.argv[sys.argv.index('--title') + 1], encoding='utf-8')\n"
        "op_dir = root / 'custom' / 'ReLU'\n"
        "op_dir.mkdir(parents=True, exist_ok=True)\n"
        "# Only write SPEC.md — missing test_ReLU.py and ReLU_golden.py\n"
        "(op_dir / 'SPEC.md').write_text('# pypto_pro\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    opencode.chmod(0o755)

    case = _make_stanford_case(tmp_path, op="ReLU")
    task = _generation_task(case, tmp_path / "work")
    agent = PyptoProOrchestratorAgent(
        pypto_repo_root=repo_root, opencode_bin=str(opencode)
    )

    output = agent.generate(task)

    assert not output.ok
    assert output.metadata["pypto_pro_status"] == "artifact_missing"
    assert "test_ReLU.py" in output.metadata.get("missing_artifacts", [])


def test_pypto_pro_orchestrator_agent_uses_isolated_git_worktree(tmp_path):
    repo_root = tmp_path / "pypto_pro_repo"
    repo_root.mkdir()
    repo_root.joinpath("README.md").write_text("# PyPTO-Pro\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=repo_root, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "add", "README.md"], cwd=repo_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
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
        "for name in ['SPEC.md', 'test_ReLU.py', 'ReLU_golden.py']:\n"
        "    (op_dir / name).write_text('# pypto_pro\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    opencode.chmod(0o755)

    case = _make_stanford_case(tmp_path, op="ReLU")
    task = _generation_task(case, tmp_path / "work")
    worktree_root = tmp_path / "pypto_pro_worktrees"
    agent = PyptoProOrchestratorAgent(
        pypto_repo_root=repo_root,
        workdir_root="custom_iso",
        worktree_root=worktree_root,
        opencode_bin=str(opencode),
    )

    output = agent.generate(task)

    assert output.ok
    run_repo_root = Path(output.metadata["pypto_pro_run_repo_root"])
    assert output.metadata["pypto_pro_isolated_worktree"] is True
    assert output.metadata["pypto_pro_repo_root"] == str(repo_root)
    assert output.metadata["pypto_pro_worktree_root"] == str(worktree_root)
    assert run_repo_root.parent == worktree_root
    assert run_repo_root.joinpath(".auto_pipeline_pypto_pro_worktree.json").is_file()
    assert output.workdir == run_repo_root / "custom_iso" / "ReLU"
    assert output.files["source_dir"] == run_repo_root / "custom_iso" / "ReLU"
    assert not (repo_root / "custom_iso").exists()
    env = json.loads(run_repo_root.joinpath("env.json").read_text(encoding="utf-8"))
    assert env["PWD"] == str(run_repo_root)
    assert env["TMPDIR"] == str(run_repo_root / "custom_iso" / "ReLU" / ".tmp")


def test_pypto_pro_orchestrator_agent_renders_cann_input_branch(tmp_path):
    repo_root = tmp_path / "pypto_pro_repo"
    repo_root.mkdir()
    opencode = tmp_path / "fake_opencode.py"
    opencode.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "root = pathlib.Path.cwd()\n"
        "if sys.argv[1:3] == ['session', 'list']:\n"
        "    print('title ses_fake')\n"
        "    raise SystemExit(0)\n"
        "if sys.argv[1:3] == ['export', 'ses_fake']:\n"
        "    print(json.dumps({'info': {'id': 'ses_fake', 'title': 'fake'}, 'messages': [], 'children': []}))\n"
        "    raise SystemExit(0)\n"
        "if '--title' in sys.argv:\n"
        "    (root / 'last_title.txt').write_text(sys.argv[sys.argv.index('--title') + 1], encoding='utf-8')\n"
        "op_dir = root / 'custom' / 'gelu'\n"
        "op_dir.mkdir(parents=True, exist_ok=True)\n"
        "for name in ['SPEC.md', 'test_gelu.py', 'gelu_golden.py']:\n"
        "    (op_dir / name).write_text('# pypto_pro\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    opencode.chmod(0o755)

    case = _make_cann_case(tmp_path, op="gelu")
    task = _generation_task(case, tmp_path / "work")
    agent = PyptoProOrchestratorAgent(
        pypto_repo_root=repo_root, opencode_bin=str(opencode)
    )

    output = agent.generate(task)

    assert output.ok
    assert output.workdir == repo_root / "custom" / "gelu"
    assert (repo_root / "custom" / "gelu" / "proto.yaml").is_file()
    assert (repo_root / "custom" / "gelu" / "cases.yaml").is_file()
    assert (repo_root / "custom" / "gelu" / "golden.py").is_file()
    assert (repo_root / "custom" / "gelu" / "desc.md").is_file()
    require_text = (repo_root / "custom" / "gelu" / "REQUIRE.md").read_text(
        encoding="utf-8"
    )
    assert "cann-bench 输入材料" in require_text
    assert "gelu(x) -> y" in require_text
    prompt_text = (task.output_dir / "PROMPT.md").read_text(encoding="utf-8")
    assert "cann-bench proto" in prompt_text
    assert "custom/gelu/proto.yaml" in prompt_text
    assert "custom/gelu/cases.yaml" in prompt_text
    assert "CANN selected-case 需求" in prompt_text
    assert "唯一权威 case 列表" in prompt_text
    assert "运行边界" in prompt_text
    assert "所有测试日志、临时文件和可再生产物也必须落在 `custom/gelu/`" in prompt_text
    assert "入口函数约定" in prompt_text
    assert "test_gelu.py" in prompt_text
    # No state machine
    assert ".orchestrator_state.json" not in prompt_text
    # No perf_round
    assert "性能调优轮次" not in prompt_text
    # No PyPTO non-Pro references
    assert "pypto-op-orchestrator" not in prompt_text
    assert "Stage 5" not in prompt_text
    assert "stage5" not in prompt_text
    assert "ai_op.py" not in prompt_text
    assert "ModelNew" not in prompt_text


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------


def test_pypto_pro_adapter_validates_cann_artifact(tmp_path):
    case = _make_cann_case(tmp_path, op="gelu")
    adapter = create_converter("pypto-pro", "cann", {})
    assert isinstance(adapter, PyptoProToCannConverter)

    source_dir = tmp_path / "pypto_pro" / "artifact" / "submission"
    source_dir.joinpath("cann_bench").mkdir(parents=True)
    source_dir.joinpath("cann_bench", "__init__.py").write_text("", encoding="utf-8")
    source_dir.joinpath("build.sh").write_text(
        "#!/usr/bin/env bash\nset -e\n", encoding="utf-8"
    )
    source_dir.joinpath("cann_bench", "gelu.py").write_text(
        "import pypto_pro\n", encoding="utf-8"
    )

    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=source_dir.parent,
        files={"source_dir": source_dir},
    )
    submission = _build_submission(
        adapter, case, output, output_dir=tmp_path / "pypto_pro" / "submission"
    )

    assert submission.kind == "cann"
    assert submission.source_dir.joinpath("build.sh").is_file()
    assert submission.source_dir.joinpath("cann_bench", "__init__.py").is_file()


def test_pypto_pro_adapter_rejects_submission_without_pypto_pro(tmp_path):
    """A submission that doesn't contain 'pypto_pro' in any .py file should be rejected."""
    case = _make_cann_case(tmp_path, op="gelu")
    adapter = create_converter("pypto-pro", "cann", {})

    source_dir = tmp_path / "submission"
    source_dir.joinpath("cann_bench").mkdir(parents=True)
    source_dir.joinpath("cann_bench", "__init__.py").write_text("", encoding="utf-8")
    source_dir.joinpath("build.sh").write_text(
        "#!/usr/bin/env bash\nset -e\n", encoding="utf-8"
    )
    # No 'pypto_pro' token in any file
    source_dir.joinpath("cann_bench", "gelu.py").write_text(
        "import torch\n", encoding="utf-8"
    )

    output = Artifact(
        status=AGENT_SUCCESS,
        workdir=source_dir.parent,
        files={"source_dir": source_dir},
    )
    with pytest.raises(ValueError, match="pypto.pro"):
        _build_submission(adapter, case, output, output_dir=tmp_path / "out")


def test_pypto_pro_conversion_input_collects_test_and_golden(tmp_path):
    """The converter should collect test_{op}.py and {op}_golden.py (not _impl.py)."""
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.mkdir()
    # PyPTO-Pro artifacts
    raw_dir.joinpath("test_Foo.py").write_text("# pypto_pro\n", encoding="utf-8")
    raw_dir.joinpath("Foo_golden.py").write_text("# golden\n", encoding="utf-8")
    raw_dir.joinpath("SPEC.md").write_text("# spec\n", encoding="utf-8")
    # Should NOT be collected
    raw_dir.joinpath("Foo_impl.py").write_text(
        "# should be excluded\n", encoding="utf-8"
    )
    raw_dir.joinpath("__init__.py").write_text("", encoding="utf-8")

    case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="level1/Foo",
        files={"task": tmp_path / "task.py"},
    )
    (tmp_path / "task.py").write_text("class Model:\n    pass\n", encoding="utf-8")
    output = Artifact(
        status=AGENT_SUCCESS, workdir=raw_dir, files={"source_dir": raw_dir}
    )

    prompt = PyptoProToStanfordConverter().build_conversion_prompt(
        "stanford",
        case,
        output,
        workdir=tmp_path / "convert",
        output_dir=tmp_path / "convert" / "artifact",
        submission_dir=tmp_path / "submission",
    )

    raw_input = prompt.output_dir / "input" / "raw"
    assert raw_input.joinpath("test_Foo.py").is_file()
    assert raw_input.joinpath("Foo_golden.py").is_file()
    # _impl.py should NOT be collected
    assert not raw_input.joinpath("Foo_impl.py").exists()
    # __init__.py should NOT be collected
    assert not raw_input.joinpath("__init__.py").exists()
    # SPEC.md is not a runtime file
    assert not raw_input.joinpath("SPEC.md").exists()
    assert str(Path("input/raw/test_Foo.py")) in prompt.text
    assert str(Path("input/raw/Foo_golden.py")) in prompt.text
    assert "Foo_impl.py" not in prompt.text
    assert "build.sh" not in prompt.text
    assert "cann_bench" not in prompt.text


def test_pypto_pro_conversion_prompt_uses_bench_specific_contract(tmp_path):
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.mkdir()
    raw_dir.joinpath("test_Foo.py").write_text("# pypto_pro\n", encoding="utf-8")
    raw_dir.joinpath("Foo_golden.py").write_text("# golden\n", encoding="utf-8")
    output = Artifact(
        status=AGENT_SUCCESS, workdir=raw_dir, files={"source_dir": raw_dir}
    )

    stanford_task = tmp_path / "stanford_task.py"
    stanford_task.write_text("class Model:\n    pass\n", encoding="utf-8")
    stanford_case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="level1/Foo",
        files={"task": stanford_task},
    )
    stanford_prompt = PyptoProToStanfordConverter().build_conversion_prompt(
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
    assert "test_Foo.py" in stanford_prompt.text
    assert "Foo_golden.py" in stanford_prompt.text
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
    cann_prompt = PyptoProToCannConverter().build_conversion_prompt(
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
    assert "test_Foo.py" in cann_prompt.text
    assert "Foo_golden.py" in cann_prompt.text


def test_pypto_pro_conversion_excludes_state_machine_docs(tmp_path):
    """The converter prompt must forbid reading state machine / doc artifacts."""
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.mkdir()
    raw_dir.joinpath("test_Foo.py").write_text("# pypto_pro\n", encoding="utf-8")
    raw_dir.joinpath("Foo_golden.py").write_text("# golden\n", encoding="utf-8")
    output = Artifact(
        status=AGENT_SUCCESS, workdir=raw_dir, files={"source_dir": raw_dir}
    )

    proto = tmp_path / "proto.yaml"
    proto.write_text("schema: foo(x) -> y\n", encoding="utf-8")
    case = CannBenchCase(
        bench_name="cann",
        task_dir=tmp_path,
        operator="Foo",
        rel_path="tasks/level1/foo",
        files={"proto": proto},
    )
    prompt = PyptoProToCannConverter().build_conversion_prompt(
        "cann",
        case,
        output,
        workdir=tmp_path / "convert",
        output_dir=tmp_path / "convert" / "artifact",
        submission_dir=tmp_path / "submission",
    )

    # Must explicitly forbid reading these PyPTO-Pro internal artifacts
    assert ".orchestrator_state.json" in prompt.text
    assert "MEMORY.md" in prompt.text
    assert "DESIGN.md" in prompt.text
    assert "SPEC.md" in prompt.text
    assert "EXPLORE_REPORT.md" in prompt.text
    assert "PRO_MATERIAL_INDEX.md" in prompt.text
    assert "GOLDEN_PERF_REPORT.md" in prompt.text


def test_pypto_pro_conversion_input_includes_nested_runtime_files(tmp_path):
    """Runtime files in subdirectories (e.g. modules/) should be collected."""
    task_path = tmp_path / "task.py"
    task_path.write_text("class Model:\n    pass\n", encoding="utf-8")
    raw_dir = tmp_path / "raw_artifact"
    raw_dir.joinpath("modules").mkdir(parents=True)
    raw_dir.joinpath("modules", "test_Block.py").write_text(
        "# pypto_pro\n", encoding="utf-8"
    )
    raw_dir.joinpath("modules", "Block_golden.py").write_text(
        "# golden\n", encoding="utf-8"
    )
    case = CannBenchCase(
        bench_name="stanford",
        task_dir=tmp_path,
        operator="Nested",
        rel_path="level3/Nested",
        files={"task": task_path},
    )
    output = Artifact(
        status=AGENT_SUCCESS, workdir=raw_dir, files={"source_dir": raw_dir}
    )

    prompt = PyptoProToStanfordConverter().build_conversion_prompt(
        "stanford",
        case,
        output,
        workdir=tmp_path / "convert",
        output_dir=tmp_path / "convert" / "artifact",
        submission_dir=tmp_path / "submission",
    )

    assert prompt.output_dir.joinpath(
        "input", "raw", "modules", "test_Block.py"
    ).is_file()
    assert prompt.output_dir.joinpath(
        "input", "raw", "modules", "Block_golden.py"
    ).is_file()
    assert str(Path("input/raw/modules/test_Block.py")) in prompt.text
    assert "build.sh" not in prompt.text
    assert "cann_bench" not in prompt.text


# ---------------------------------------------------------------------------
# Config YAML test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dispatcher test
# ---------------------------------------------------------------------------


def test_pypto_pro_dispatcher_loads_test_file(tmp_path):
    """The dispatcher should import test_{op}.py (not {op}_impl.py)."""
    from auto_pipeline.generator.pypto_pro.dispatcher import write_dispatcher

    op_name = "softmax"
    parent_op_dir = tmp_path / "softmax"
    parent_op_dir.mkdir()
    # Create a class subdir with test_softmax.py
    c1_dir = parent_op_dir / "c1"
    c1_dir.mkdir()
    (c1_dir / "test_softmax.py").write_text(
        "def softmax(x):\n    return x\n",
        encoding="utf-8",
    )
    manifest = {
        "op_name": op_name,
        "classes": [
            {
                "class_id": "c1",
                "subdir": "c1",
                "signature": [[2, "float32"]],
            },
        ],
    }
    target = write_dispatcher(parent_op_dir, manifest)
    assert target == parent_op_dir / "softmax.py"
    text = target.read_text(encoding="utf-8")
    # Should reference test_{op}.py via f-string (not {op}_impl.py)
    assert "test_{_OP_NAME}.py" in text
    assert "_impl.py" not in text
    # Should look for {op} or {op}_wrapper entry (via f-string)
    assert "_OP_NAME" in text
    assert "_wrapper" in text
