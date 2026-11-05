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


from types import SimpleNamespace

from kernel_eval import staged_eval
from kernel_eval.cli import _create_config_from_args, create_parser as create_cli_parser
from kernel_eval.config import Config, get_project_root
from kernel_eval.data.package_manager import PackageManager
from kernel_eval.registry.bench_registry import BenchConfig, BenchRegistry


BUILD_ARGS = "--pkg --soc=ascend950 --ops=matmul_all_reduce,allto_all_matmul -j16"
EXPECTED_ARGS = [
    "--pkg",
    "--soc=ascend950",
    "--ops=matmul_all_reduce,allto_all_matmul",
    "-j16",
]


def test_staged_eval_parses_source_build_args_into_config(tmp_path):
    args = staged_eval.create_parser().parse_args(
        [
            "--source-dir",
            str(tmp_path),
            "--source-build-args",
            BUILD_ARGS,
        ]
    )

    config = staged_eval._make_config(args, str(tmp_path), enable_profiler=False)

    assert config.source_build_args == EXPECTED_ARGS


def test_cli_parses_source_build_args_into_config(tmp_path, monkeypatch):
    # This test checks CLI argument forwarding, independent of global bench registration.
    monkeypatch.setitem(BenchRegistry._items, "cann", BenchConfig(name="cann"))

    args = create_cli_parser().parse_args(
        [
            "eval",
            "--source-dir",
            str(tmp_path),
            "--source-build-args",
            BUILD_ARGS,
        ]
    )

    config = _create_config_from_args(args, str(tmp_path))

    assert config.source_build_args == EXPECTED_ARGS


def test_package_manager_passes_build_args_without_shell_reparse(tmp_path, monkeypatch):
    source_dir = tmp_path / "submission"
    source_dir.mkdir()
    log_path = source_dir / "_compile.log"
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("kernel_eval.data.package_manager.subprocess.run", fake_run)
    config = Config(source_build_args=EXPECTED_ARGS)

    rc, _ = PackageManager(config=config)._run_build(source_dir, log_path)

    assert rc == 0
    assert calls[0][0] == ["bash", "build.sh", *EXPECTED_ARGS]
    assert "shell" not in calls[0][1]


def test_run_evaluation_exposes_source_build_args():
    script = (get_project_root() / "scripts" / "run_evaluation.sh").read_text()

    assert "--source-build-args" in script
    assert 'CANN_BENCH_SOURCE_BUILD_ARGS="${SOURCE_BUILD_ARGS}"' in script
