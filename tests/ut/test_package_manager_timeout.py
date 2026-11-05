# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from kernel_eval import cli, staged_eval
from kernel_eval.data.package_manager import PackageManager, resolve_build_timeout_sec


def test_shell_keeps_existing_timeout_defaults():
    script = (
        Path(__file__).resolve().parents[2] / "scripts/run_evaluation.sh"
    ).read_text(encoding="utf-8")
    assert "\nTIMEOUT_PER_PROCESS=300\n" in script
    assert 'BUILD_TIMEOUT="${CANN_BENCH_BUILD_TIMEOUT_SEC-600}"' in script


def test_build_timeout_uses_default(monkeypatch):
    monkeypatch.delenv("CANN_BENCH_BUILD_TIMEOUT_SEC", raising=False)

    assert resolve_build_timeout_sec() == 600


def test_build_timeout_accepts_environment_override(monkeypatch):
    monkeypatch.setenv("CANN_BENCH_BUILD_TIMEOUT_SEC", "3600")

    assert resolve_build_timeout_sec() == 3600


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_build_timeout_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("CANN_BENCH_BUILD_TIMEOUT_SEC", value)

    with pytest.raises(ValueError):
        resolve_build_timeout_sec()


@pytest.mark.parametrize(
    "module, args, entry",
    [
        (cli, ["eval"], "cmd_eval"),
        (staged_eval, [], "run"),
    ],
)
@pytest.mark.parametrize("value", ["abc", "0", "-1"])
def test_python_entry_rejects_invalid_build_timeout(
    module, args, entry, value, monkeypatch, capsys
):
    monkeypatch.setenv("CANN_BENCH_BUILD_TIMEOUT_SEC", value)
    monkeypatch.setattr(
        sys, "argv", [module.__name__, *args, "--source-dir", "submission"]
    )
    monkeypatch.setattr(
        module, entry, lambda args: pytest.fail("evaluation must not start")
    )

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "CANN_BENCH_BUILD_TIMEOUT_SEC must be" in captured.err
    assert "Traceback" not in captured.out + captured.err


@pytest.mark.parametrize("value", ["abc", "0", "-1"])
def test_run_build_reports_invalid_timeout_without_starting_process(
    tmp_path, monkeypatch, value
):
    monkeypatch.setenv("CANN_BENCH_BUILD_TIMEOUT_SEC", value)
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("build must not start")
    )
    log_path = tmp_path / "_compile.log"

    rc, log_text = PackageManager()._run_build(tmp_path, log_path)

    assert rc == 1
    assert "CANN_BENCH_BUILD_TIMEOUT_SEC must be" in log_text
    assert "Traceback" not in log_text
    assert log_path.read_text() == log_text


def test_run_build_uses_override_and_reports_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("CANN_BENCH_BUILD_TIMEOUT_SEC", "3600")

    def timeout(cmd, **kwargs):
        assert cmd == ["bash", "build.sh", "--pkg"]
        assert kwargs["timeout"] == 3600
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timeout)
    pm = PackageManager(config=SimpleNamespace(source_build_args=["--pkg"]))

    rc, log_text = pm._run_build(tmp_path, tmp_path / "_compile.log")

    assert rc == 124
    assert "build.sh timed out after 3600s" in log_text


@pytest.mark.parametrize(
    "args, message",
    [
        (["--build-timeout", "abc"], "必须为正整数"),
        (["--build-timeout", "0"], "必须为正整数"),
        (["--build-timeout", "-1"], "必须为正整数"),
        (["--build-timeout"], "缺少参数"),
        (["--build-timeout", "--no-perf"], "缺少参数"),
        (["--build-timeout", ""], "缺少参数"),
    ],
)
def test_shell_rejects_invalid_timeout_before_setup(tmp_path, args, message):
    result = _run_shell_parser(tmp_path, args)

    assert result.returncode == 2
    assert "--build-timeout" in result.stderr
    assert message in result.stderr
    assert "项目根目录:" not in result.stdout


def test_shell_accepts_positive_timeout(tmp_path):
    result = _run_shell_parser(tmp_path, ["--build-timeout", "3600", "--help"])

    assert result.returncode == 0
    assert "--build-timeout" in result.stdout


@pytest.mark.parametrize(
    "module, prefix, config_builder",
    [
        (cli, ["eval"], lambda args, path: cli._create_config_from_args(args, path)),
        (
            staged_eval,
            [],
            lambda args, path: staged_eval._make_config(
                args, path, enable_profiler=False
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "environment, explicit, expected",
    [
        (None, None, 600),
        ("3600", None, 3600),
        ("3600", "7200", 7200),
        ("bad", "7200", 7200),
    ],
)
def test_timeout_precedence_reaches_build_config(
    module,
    prefix,
    config_builder,
    environment,
    explicit,
    expected,
    monkeypatch,
    tmp_path,
):
    if environment is None:
        monkeypatch.delenv("CANN_BENCH_BUILD_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("CANN_BENCH_BUILD_TIMEOUT_SEC", environment)
    argv = prefix + (["--build-timeout", explicit] if explicit is not None else [])
    args = module.create_parser().parse_args(argv)
    assert config_builder(args, str(tmp_path)).build_timeout_sec == expected


def test_build_config_timeout_overrides_invalid_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CANN_BENCH_BUILD_TIMEOUT_SEC", "bad")
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: calls.append(kwargs) or SimpleNamespace(returncode=0),
    )
    manager = PackageManager(
        config=SimpleNamespace(build_timeout_sec=1200, source_build_args=[])
    )
    rc, _ = manager._run_build(tmp_path, tmp_path / "_compile.log")
    assert rc == 0
    assert calls[0]["timeout"] == 1200


def test_shell_rejects_invalid_environment_before_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("CANN_BENCH_BUILD_TIMEOUT_SEC", "bad")
    result = _run_shell_parser(tmp_path, [])
    assert result.returncode == 2
    assert "CANN_BENCH_BUILD_TIMEOUT_SEC" in result.stderr
    assert "项目根目录:" not in result.stdout


def _run_shell_parser(tmp_path, args):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is required to exercise the shell argument parser")
    script_path = Path(__file__).resolve().parents[2] / "scripts/run_evaluation.sh"
    # Normalize checkout line endings for Git Bash on Windows. All tested paths
    # exit during parsing, before dependency setup or evaluation can run.
    script_copy = tmp_path / "run_evaluation.sh"
    script_copy.write_text(
        script_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    return subprocess.run(
        [bash, str(script_copy), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=10,
    )
