# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import shutil
import subprocess
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "run_evaluation.sh"


def _shell_path(path):
    value = path.as_posix()
    if os.name == "nt":
        return f"/{value[0].lower()}{value[2:]}"
    return value


def _run_with_python_stub(tmp_path, args):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required for shell entrypoint tests")
    project = tmp_path / "bench fixture"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / SCRIPT.name
    script.write_text(
        SCRIPT.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    (scripts / "ensure_cann_bench_utils.sh").write_text(
        "ensure_cann_bench_utils() { return 0; }\n", encoding="utf-8", newline="\n"
    )
    (project / "tasks").mkdir()
    (project / "bench_lab" / "kernel_bench").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "python-argv.txt"
    capture_env = tmp_path / "python-env.txt"
    python_stub = bin_dir / "python"
    python_stub.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" == --version ]]; then echo 'Python stub 3.12'; exit 0; fi\n"
        'if [[ "$1" == -c ]]; then exit 0; fi\n'
        'if [[ "$1" == -m && "$2" == pip ]]; then exit 1; fi\n'
        'printf \'%s\\n\' --CALL-- "$@" >> "$CAPTURE_FILE"\n'
        'printf \'%s\\n\' "$CANN_BENCH_BUILD_TIMEOUT_SEC" "$CANN_BENCH_SOURCE_BUILD_ARGS" '
        '> "$CAPTURE_ENV_FILE"\n',
        encoding="utf-8",
        newline="\n",
    )
    python_stub.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = (
        f"{_shell_path(bin_dir)}:/usr/bin:/bin"
        if os.name == "nt"
        else f"{bin_dir}{os.pathsep}{env['PATH']}"
    )
    env["LANG"] = "C.UTF-8"
    env["LC_ALL"] = "C.UTF-8"
    env["CAPTURE_FILE"] = _shell_path(capture)
    env["CAPTURE_ENV_FILE"] = _shell_path(capture_env)
    result = subprocess.run(
        [bash, _shell_path(script), *args],
        cwd=project,
        env=env,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    lines = capture.read_text(encoding="utf-8").splitlines()
    marker = lines.index("--CALL--")
    return (
        lines[marker + 1 :],
        result.stdout,
        capture_env.read_text(encoding="utf-8").splitlines(),
    )


def test_list_keeps_task_dir_as_one_argument(tmp_path):
    argv, _, _ = _run_with_python_stub(
        tmp_path, ["--action", "list", "--task-dir", "/tmp/task path"]
    )
    assert argv[0:2] == ["-m", "kernel_eval.cli"]
    assert argv[argv.index("--task-dir") + 1] == "/tmp/task path"


def test_eval_keeps_all_directory_arguments_as_one_argument(tmp_path):
    argv, _, _ = _run_with_python_stub(
        tmp_path,
        [
            "--action",
            "eval",
            "--device",
            "cpu",
            "--source-dir",
            "/tmp/source path",
            "--task-dir",
            "/tmp/task path",
            "--reports-dir",
            _shell_path(tmp_path / "report path"),
        ],
    )
    assert argv[0:2] == ["-m", "kernel_eval.cli"]
    for option, value in (
        ("--source-dir", "/tmp/source path"),
        ("--task-dir", "/tmp/task path"),
        ("--reports-dir", _shell_path(tmp_path / "report path")),
    ):
        assert argv[argv.index(option) + 1] == value


def test_staged_eval_keeps_task_dir_as_one_argument(tmp_path):
    argv, _, _ = _run_with_python_stub(
        tmp_path,
        [
            "--action",
            "eval",
            "--task-dir",
            "/tmp/task path",
            "--reports-dir",
            _shell_path(tmp_path / "report path"),
        ],
    )
    assert argv[0:2] == ["-m", "kernel_eval.staged_eval"]
    assert argv[argv.index("--task-dir") + 1] == "/tmp/task path"


@pytest.mark.parametrize(
    ("device_args", "module"),
    [([], "kernel_eval.staged_eval"), (["--device", "cpu"], "kernel_eval.cli")],
)
def test_eval_preserves_paths_range_and_build_environment(
    tmp_path, device_args, module
):
    build_args = '--pkg --soc=ascend950 --output-dir="/tmp/build output" -j4'
    options = {
        "--source-dir": "/tmp/source path",
        "--task-dir": "/tmp/task path",
        "--reports-dir": _shell_path(tmp_path / "report path"),
        "--case-id-range": "21-40",
    }
    args = ["--action", "eval", *device_args]
    for option, value in options.items():
        args.extend([option, value])
    args.extend(["--source-build-args", build_args, "--build-timeout", "7200"])

    argv, _, environment = _run_with_python_stub(tmp_path, args)

    assert argv[:2] == ["-m", module]
    for option, value in options.items():
        assert argv.count(option) == 1
        assert argv[argv.index(option) + 1] == value
    assert environment == ["7200", build_args]
