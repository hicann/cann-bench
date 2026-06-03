import asyncio
import gc
import os
import sys
import threading
import textwrap
import types
import warnings
from pathlib import Path

from auto_pipeline.generator.registry import available_generators, create_generator
from auto_pipeline.core import AGENT_TIMEOUT, RunnerPrompt


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def _fake_akg_repo(
    tmp_path: Path,
    *,
    name: str = "akg",
    coder_code: str = "class ModelNew:\n    pass\n",
    success: bool = True,
    error: object | None = None,
    profile_res: object | None = None,
    task_desc: str | None = None,
    task_desc_sleep_sec: float = 0.0,
    write_config: bool = True,
    register_sleep_sec: float = 0.0,
    sleep_sec: float = 0.0,
    fail_in_run: bool = False,
    mutate_env_key: str | None = None,
    mutate_env_value: str = "mutated",
    critical_marker: Path | None = None,
    violation_marker: Path | None = None,
    expected_env: str | None = None,
) -> Path:
    repo = tmp_path / name
    python_root = repo / "akg_agents" / "python"
    if profile_res is None:
        profile_res = {"latency_us": 1.0}
    for package in [
        "akg_agents",
        "akg_agents/core",
        "akg_agents/core/worker",
        "akg_agents/op",
        "akg_agents/op/config",
        "akg_agents/op/langgraph_op",
        "akg_agents/op/utils",
        "akg_agents/utils",
    ]:
        _write(python_root / package / "__init__.py", "")
    _write(
        python_root / "akg_agents/op/config/config_validator.py",
        '''
        def load_config(config_path):
            return {
                "loaded_config_path": config_path,
                "secret_token": "do-not-expose",
                "nested": {"do": "not expose"},
            }
        ''',
    )
    _write(
        python_root / "akg_agents/core/worker/manager.py",
        '''
        import asyncio

        async def register_local_worker(device_ids, backend, arch):
            if __REGISTER_SLEEP_SEC__:
                await asyncio.sleep(__REGISTER_SLEEP_SEC__)
            return None
        '''.replace("__REGISTER_SLEEP_SEC__", repr(register_sleep_sec)),
    )
    _write(
        python_root / "akg_agents/op/utils/cann_utils.py",
        (
            f'''
            import time

            def get_cann_task_desc_for_prompt(problem_dir):
                if {task_desc_sleep_sec!r}:
                    time.sleep({task_desc_sleep_sec!r})
                return {task_desc!r}
            '''
            if task_desc is not None
            else '''
            import time

            def get_cann_task_desc_for_prompt(problem_dir):
                if __TASK_DESC_SLEEP_SEC__:
                    time.sleep(__TASK_DESC_SLEEP_SEC__)
                return "fake cann task desc from " + str(problem_dir)
            '''.replace("__TASK_DESC_SLEEP_SEC__", repr(task_desc_sleep_sec))
        ),
    )
    _write(
        python_root / "akg_agents/utils/environment_check.py",
        '''
        def check_env_for_task(framework, backend, dsl, config, is_remote=False):
            config["env_checked"] = True
        ''',
    )
    task_module = '''
        import asyncio
        import os
        from pathlib import Path

        class LangGraphTask:
            def __init__(self, op_name, task_desc, task_id, backend, arch, dsl, config, framework="torch", workflow="default", bench_type="kernelbench"):
                self.kwargs = {
                    "op_name": op_name,
                    "task_desc": task_desc,
                    "task_id": task_id,
                    "backend": backend,
                    "arch": arch,
                    "dsl": dsl,
                    "config": config,
                    "framework": framework,
                    "workflow": workflow,
                    "bench_type": bench_type,
                }

            async def run(self):
                critical_marker = __CRITICAL_MARKER__
                violation_marker = __VIOLATION_MARKER__
                marker_fd = None
                if violation_marker and os.environ.get("AKG_FAKE_REPO_NAME") != __EXPECTED_ENV__:
                    Path(violation_marker).write_text("env-pollution", encoding="utf-8")
                if violation_marker and __REPO_NAME__ not in __file__:
                    Path(violation_marker).write_text("module-pollution", encoding="utf-8")
                if critical_marker:
                    try:
                        marker_fd = os.open(critical_marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.write(marker_fd, __REPO_NAME__.encode("utf-8"))
                    except FileExistsError:
                        if violation_marker:
                            Path(violation_marker).write_text("overlap", encoding="utf-8")
                try:
                    if __MUTATE_ENV_KEY__:
                        os.environ[__MUTATE_ENV_KEY__] = __MUTATE_ENV_VALUE__
                    if __SLEEP_SEC__:
                        await asyncio.sleep(__SLEEP_SEC__)
                    if __FAIL_IN_RUN__:
                        raise RuntimeError("fake AKG task exploded")
                    return (
                        self.kwargs["op_name"],
                        __SUCCESS__,
                        {
                            "coder_code": "__CODER_CODE__",
                            "verifier_result": __SUCCESS__,
                            "verifier_error": "",
                            "error": __ERROR__,
                            "profile_res": __PROFILE_RES__,
                            "debug_secret": "do-not-expose",
                            "task_kwargs": self.kwargs,
                            "task_config": self.kwargs["config"],
                        },
                    )
                finally:
                    if marker_fd is not None:
                        os.close(marker_fd)
                        try:
                            os.unlink(critical_marker)
                        except FileNotFoundError:
                            pass
        '''
    _write(
        python_root / "akg_agents/op/langgraph_op/task.py",
        (
            task_module
            .replace('"__CODER_CODE__"', repr(coder_code))
            .replace("__SUCCESS__", repr(success))
            .replace("__ERROR__", repr(error))
            .replace("__PROFILE_RES__", repr(profile_res))
            .replace("__MUTATE_ENV_KEY__", repr(mutate_env_key))
            .replace("__MUTATE_ENV_VALUE__", repr(mutate_env_value))
            .replace("__SLEEP_SEC__", repr(sleep_sec))
            .replace("__FAIL_IN_RUN__", repr(fail_in_run))
            .replace("__CRITICAL_MARKER__", repr(str(critical_marker)) if critical_marker else "None")
            .replace("__VIOLATION_MARKER__", repr(str(violation_marker)) if violation_marker else "None")
            .replace("__EXPECTED_ENV__", repr(expected_env))
            .replace("__REPO_NAME__", repr(name))
        ),
    )
    config_path = python_root / "akg_agents/op/config/triton_ascend_kernelgen_config.yaml"
    if write_config:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("default_workflow: kernelgen_only_workflow\n", encoding="utf-8")
        (config_path.parent / "triton_ascend_evolve_config.yaml").write_text(
            "default_workflow: kernelgen_only_workflow\n",
            encoding="utf-8",
        )
    return repo


def _fake_prompt(
    tmp_path: Path,
    *,
    write_proto: bool = True,
    write_golden: bool = True,
    write_cases: bool = True,
    timeout_sec: float = 7200,
) -> RunnerPrompt:
    task_dir = tmp_path / "tasks" / "level1" / "exp"
    task_dir.mkdir(parents=True)
    if write_proto:
        task_dir.joinpath("proto.yaml").write_text("operator:\n  name: Exp\n", encoding="utf-8")
    if write_golden:
        task_dir.joinpath("golden.py").write_text("def exp(x):\n    return x\n", encoding="utf-8")
    if write_cases:
        task_dir.joinpath("cases.yaml").write_text("cases: []\n", encoding="utf-8")
    return RunnerPrompt(
        text="AKG KernelGen prompt",
        cwd=tmp_path / "work",
        output_dir=tmp_path / "work" / "artifact",
        timeout_sec=timeout_sec,
        files={
            "proto": task_dir / "proto.yaml",
            "golden": task_dir / "golden.py",
            "cases": task_dir / "cases.yaml",
        },
        metadata={
            "bench_name": "cann",
            "operator": "Exp",
            "op_name": "exp",
            "task_dir": str(task_dir),
        },
    )


def _fake_stanford_prompt(
    tmp_path: Path,
    *,
    write_task: bool = True,
    timeout_sec: float = 7200,
) -> RunnerPrompt:
    task_path = tmp_path / "KernelBench" / "level1" / "19_ReLU.py"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    if write_task:
        _write(
            task_path,
            '''
            import torch
            import torch.nn as nn


            class Model(nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return torch.relu(x)


            def get_init_inputs():
                return []


            def get_inputs():
                return [torch.randn(4, 4)]
            ''',
        )
    return RunnerPrompt(
        text="AKG KernelBench prompt",
        cwd=tmp_path / "work",
        output_dir=tmp_path / "work" / "artifact",
        timeout_sec=timeout_sec,
        files={"task": task_path},
        metadata={
            "benchmark": "stanford",
            "operator": "ReLU",
            "op_name": "ReLU",
            "task_dir": str(task_path.parent),
        },
    )


def test_akg_agent_registered(tmp_path):
    repo = _fake_akg_repo(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo), "device_id": 7})

    assert "akg-agent" in available_generators()
    assert agent.type == "akg-agent"


def test_akg_agent_uses_adaptive_workflow_config(tmp_path):
    repo = _fake_akg_repo(tmp_path)
    prompt = _fake_prompt(tmp_path, timeout_sec=4321)
    agent = create_generator(
        "akg-agent",
        {"repo_root": str(repo), "workflow": "adaptive_search_workflow"},
    )

    output = agent.run(prompt)

    assert output.ok
    assert "akg-agent" in available_generators()
    state = output.metadata["akg_final_state"]
    assert state["task_kwargs"]["workflow"] == "adaptive_search_workflow"
    assert state["task_config"]["default_workflow"] == "kernelgen_only_workflow"
    assert state["task_config"]["workflow_timeout"] == 4321
    assert state["task_config"]["loaded_config_path"].endswith("triton_ascend_evolve_config.yaml")


def test_akg_agent_returns_coder_code(tmp_path):
    repo = _fake_akg_repo(tmp_path)
    prompt = _fake_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo), "device_id": 7})

    output = agent.run(prompt)

    assert output.ok
    assert output.output_text == "class ModelNew:\n    pass\n"
    assert output.files["generated_code"].read_text(encoding="utf-8") == output.output_text
    assert output.metadata["akg_success"] is True
    assert output.metadata["akg_final_state"]["verifier_result"] is True
    assert output.metadata["akg_final_state"]["profile_res"] == {"latency_us": 1.0}
    state = output.metadata["akg_final_state"]
    assert state["op_name"] == "exp"
    task_kwargs = state["task_kwargs"]
    assert task_kwargs["op_name"] == "exp"
    assert task_kwargs["backend"] == "ascend"
    assert task_kwargs["arch"] == "ascend910b4"
    assert task_kwargs["codegen_target"] == "triton_ascend"
    assert task_kwargs["framework"] == "torch"
    assert task_kwargs["workflow"] == "kernelgen_only_workflow"
    assert task_kwargs["bench_type"] == "cann"
    assert "fake cann task desc" in task_kwargs["task_desc"]
    task_config = state["task_config"]
    assert task_config["bench_type"] == "cann"
    assert task_config["cann_problem_dir"] == str(prompt.files["proto"].parent)
    assert task_config["env_checked"] is True
    assert task_config["default_workflow"] == "kernelgen_only_workflow"
    assert task_config["loaded_config_path"].endswith("triton_ascend_kernelgen_config.yaml")
    assert "config" not in task_kwargs
    assert "secret_token" not in task_config
    assert "nested" not in task_config
    assert "debug_secret" not in state
    assert output.metadata["akg_task_config"]["backend"] == "ascend"
    assert output.metadata["akg_task_config"]["arch"] == "ascend910b4"
    assert output.metadata["akg_task_config"]["codegen_target"] == "triton_ascend"
    assert output.metadata["akg_task_config"]["framework"] == "torch"
    assert output.metadata["akg_task_config"]["workflow"] == "kernelgen_only_workflow"
    assert "AKG KernelGen completed" in output.message


def test_akg_agent_uses_kernelbench_for_stanford_task(tmp_path):
    repo = _fake_akg_repo(tmp_path)
    prompt = _fake_stanford_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo), "device_id": 7})

    output = agent.run(prompt)

    assert output.ok
    state = output.metadata["akg_final_state"]
    task_kwargs = state["task_kwargs"]
    assert state["op_name"] == "ReLU"
    assert task_kwargs["op_name"] == "ReLU"
    assert task_kwargs["bench_type"] == "kernelbench"
    assert "class Model" in task_kwargs["task_desc"]
    task_config = state["task_config"]
    assert task_config["bench_type"] == "kernelbench"
    assert "cann_problem_dir" not in task_config
    assert task_config["env_checked"] is True


def test_akg_agent_uses_adaptive_base_config_for_adaptive_workflow(tmp_path):
    repo = _fake_akg_repo(tmp_path)
    prompt = _fake_prompt(tmp_path)
    agent = create_generator(
        "akg-agent",
        {
            "repo_root": str(repo),
            "workflow": "adaptive_search_workflow",
        },
    )

    output = agent.run(prompt)

    assert output.ok
    state = output.metadata["akg_final_state"]
    assert state["task_kwargs"]["workflow"] == "adaptive_search_workflow"
    assert state["task_config"]["default_workflow"] == "kernelgen_only_workflow"
    assert state["task_config"]["loaded_config_path"].endswith("triton_ascend_evolve_config.yaml")
    assert output.metadata["akg_task_config"]["workflow"] == "adaptive_search_workflow"


def test_akg_agent_compacts_large_metadata(tmp_path):
    long_task_desc = "fake cann task desc " + ("x" * 2000)
    long_secret = "secret-" + ("y" * 2000)
    repo = _fake_akg_repo(
        tmp_path,
        task_desc=long_task_desc,
        profile_res={
            "latency_us": 1.0,
            "huge": long_secret,
            "nested": {"secret": long_secret, "ok": True},
            "items": list(range(50)),
        },
    )
    prompt = _fake_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo), "device_id": 7})

    output = agent.run(prompt)

    state = output.metadata["akg_final_state"]
    assert output.ok
    assert len(state["task_kwargs"]["task_desc"]) <= 520
    assert len(state["profile_res"]["huge"]) <= 520
    assert len(state["profile_res"]["nested"]["secret"]) <= 520
    assert len(state["profile_res"]["items"]) <= 8


def test_akg_agent_fails_without_repo(tmp_path):
    prompt = _fake_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(tmp_path / "missing")})

    output = agent.run(prompt)

    assert not output.ok
    assert "AKG repo root not found" in output.message
    assert output.metadata["akg_task_config"]["backend"] == "ascend"
    assert output.metadata["akg_task_config"]["device_id"] == 0


def test_akg_agent_returns_failed_output_when_akg_fails(tmp_path):
    repo = _fake_akg_repo(
        tmp_path,
        success=False,
        error="kernel verifier failed\nfull traceback hidden",
        coder_code="class ModelNew:\n    partial = True\n",
    )
    prompt = _fake_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    output = agent.run(prompt)

    assert not output.ok
    assert output.output_text == "class ModelNew:\n    partial = True\n"
    assert "kernel verifier failed" in output.message
    assert "kernel verifier failed" in output.log_file.read_text(encoding="utf-8")
    assert output.metadata["akg_success"] is False
    assert output.metadata["akg_final_state"]["error"] == "kernel verifier failed\nfull traceback hidden"
    assert output.metadata["akg_final_state"]["verifier_result"] is False


def test_akg_agent_fails_when_success_has_empty_coder_code(tmp_path):
    repo = _fake_akg_repo(tmp_path, coder_code="")
    prompt = _fake_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    output = agent.run(prompt)

    assert not output.ok
    assert "empty coder_code" in output.message
    assert output.metadata["akg_success"] is True


def test_akg_agent_fails_without_default_config(tmp_path):
    repo = _fake_akg_repo(tmp_path, write_config=False)
    prompt = _fake_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    output = agent.run(prompt)

    assert not output.ok
    assert "AKG config path not found" in output.message


def test_akg_agent_fails_without_task_files(tmp_path):
    repo = _fake_akg_repo(tmp_path)
    prompt = _fake_prompt(tmp_path, write_cases=False)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    output = agent.run(prompt)

    assert not output.ok
    assert "AKG task dir missing required files" in output.message
    assert "cases.yaml" in output.message


def test_akg_agent_accepts_cases_csv_task_file(tmp_path):
    repo = _fake_akg_repo(tmp_path)
    prompt = _fake_prompt(tmp_path, write_cases=False)
    prompt.files["cases"].with_suffix(".csv").write_text("case_id\n0\n", encoding="utf-8")
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    output = agent.run(prompt)

    assert output.ok


def test_akg_agent_runs_inside_existing_event_loop(tmp_path):
    repo = _fake_akg_repo(tmp_path)
    prompt = _fake_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo), "device_id": 7})

    async def main():
        return agent.run(prompt)

    output = asyncio.run(main())

    assert output.ok
    assert output.output_text == "class ModelNew:\n    pass\n"


def test_akg_agent_isolates_akg_agents_modules_between_repos(tmp_path):
    first_repo = _fake_akg_repo(
        tmp_path,
        name="akg_first",
        coder_code="class ModelNew:\n    first = True\n",
    )
    second_repo = _fake_akg_repo(
        tmp_path,
        name="akg_second",
        coder_code="class ModelNew:\n    second = True\n",
    )
    first_prompt = _fake_prompt(tmp_path / "first")
    second_prompt = _fake_prompt(tmp_path / "second")

    first = create_generator("akg-agent", {"repo_root": str(first_repo)})
    second = create_generator("akg-agent", {"repo_root": str(second_repo)})

    first_output = first.run(first_prompt)
    second_output = second.run(second_prompt)

    assert first_output.ok
    assert first_output.output_text == "class ModelNew:\n    first = True\n"
    assert second_output.ok
    assert second_output.output_text == "class ModelNew:\n    second = True\n"


def test_akg_agent_serializes_in_process_runtime(tmp_path):
    critical_marker = tmp_path / "akg-active"
    violation_marker = tmp_path / "akg-violation"
    first_repo = _fake_akg_repo(
        tmp_path,
        name="akg_parallel_first",
        coder_code="class ModelNew:\n    first = True\n",
        sleep_sec=0.2,
        critical_marker=critical_marker,
        violation_marker=violation_marker,
        expected_env="first",
    )
    second_repo = _fake_akg_repo(
        tmp_path,
        name="akg_parallel_second",
        coder_code="class ModelNew:\n    second = True\n",
        sleep_sec=0.2,
        critical_marker=critical_marker,
        violation_marker=violation_marker,
        expected_env="second",
    )
    first = create_generator(
        "akg-agent",
        {"repo_root": str(first_repo), "env": {"AKG_FAKE_REPO_NAME": "first"}},
    )
    second = create_generator(
        "akg-agent",
        {"repo_root": str(second_repo), "env": {"AKG_FAKE_REPO_NAME": "second"}},
    )
    prompts = [_fake_prompt(tmp_path / "parallel_first"), _fake_prompt(tmp_path / "parallel_second")]
    barrier = threading.Barrier(3)
    outputs = []

    def run_agent(agent, prompt):
        barrier.wait()
        outputs.append(agent.run(prompt))

    threads = [
        threading.Thread(target=run_agent, args=(first, prompts[0])),
        threading.Thread(target=run_agent, args=(second, prompts[1])),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert [output.ok for output in outputs] == [True, True]
    assert not violation_marker.exists()


def test_akg_agent_times_out_async_workflow(tmp_path):
    repo = _fake_akg_repo(tmp_path, sleep_sec=0.2)
    prompt = _fake_prompt(tmp_path, timeout_sec=0.05)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    output = agent.run(prompt)

    assert not output.ok
    assert output.status == AGENT_TIMEOUT
    assert "timeout" in output.message.lower()
    assert "0.05" in output.message


def test_akg_agent_uses_shared_async_timeout_budget(tmp_path):
    repo = _fake_akg_repo(tmp_path, register_sleep_sec=0.08, sleep_sec=0.08)
    prompt = _fake_prompt(tmp_path, timeout_sec=0.12)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    output = agent.run(prompt)

    assert not output.ok
    assert output.status == AGENT_TIMEOUT


def test_akg_agent_does_not_leak_unawaited_coroutine_when_deadline_expired(tmp_path):
    repo = _fake_akg_repo(tmp_path, task_desc_sleep_sec=0.08)
    prompt = _fake_prompt(tmp_path, timeout_sec=0.05)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        output = agent.run(prompt)
        gc.collect()

    assert not output.ok
    assert output.status == AGENT_TIMEOUT
    assert not [
        warning for warning in caught
        if "was never awaited" in str(warning.message)
    ]


def test_akg_agent_uses_unique_task_ids_with_fixed_time(tmp_path, monkeypatch):
    repo = _fake_akg_repo(tmp_path)
    monkeypatch.setattr("auto_pipeline.generator.akg.time.time_ns", lambda: 12345)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})

    first_output = agent.run(_fake_prompt(tmp_path / "task_id_first"))
    second_output = agent.run(_fake_prompt(tmp_path / "task_id_second"))

    first_task_id = first_output.metadata["akg_final_state"]["task_kwargs"]["task_id"]
    second_task_id = second_output.metadata["akg_final_state"]["task_kwargs"]["task_id"]
    assert first_output.ok
    assert second_output.ok
    assert first_task_id != second_task_id


def test_akg_agent_restores_process_state_after_exception(tmp_path, monkeypatch):
    repo = _fake_akg_repo(tmp_path, fail_in_run=True)
    prompt = _fake_prompt(tmp_path)
    agent = create_generator(
        "akg-agent",
        {"repo_root": str(repo), "env": {"AKG_RESTORE_ME": "temporary"}},
    )
    sentinel_module = types.ModuleType("akg_agents")
    monkeypatch.setitem(sys.modules, "akg_agents", sentinel_module)
    monkeypatch.setenv("AKG_RESTORE_ME", "original")
    original_sys_path = list(sys.path)

    output = agent.run(prompt)

    assert not output.ok
    assert "fake AKG task exploded" in output.message
    assert os.environ["AKG_RESTORE_ME"] == "original"
    assert sys.modules["akg_agents"] is sentinel_module
    assert sys.path == original_sys_path


def test_akg_agent_restores_untracked_env_mutation(tmp_path, monkeypatch):
    repo = _fake_akg_repo(
        tmp_path,
        mutate_env_key="AKG_UNTRACKED_MUTATION",
        mutate_env_value="leaked",
    )
    prompt = _fake_prompt(tmp_path)
    agent = create_generator("akg-agent", {"repo_root": str(repo)})
    monkeypatch.delenv("AKG_UNTRACKED_MUTATION", raising=False)

    output = agent.run(prompt)

    assert output.ok
    assert "AKG_UNTRACKED_MUTATION" not in os.environ
