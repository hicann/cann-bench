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

import os
import subprocess
import sys
import types

from src.kernel_eval.data.package_manager import PackageManager, _detect_opp_path


def test_private_opp_install_root_does_not_replace_system_opp(monkeypatch, tmp_path):
    system_opp = tmp_path / "system_opp"
    system_opp.mkdir()
    private_opp = tmp_path / "lane" / "opp"
    monkeypatch.setenv("ASCEND_OPP_PATH", str(system_opp))
    monkeypatch.setenv("MC2_OPP_INSTALL_ROOT", str(private_opp))

    assert _detect_opp_path() == str(private_opp.resolve())
    assert private_opp.is_dir()
    assert os.environ["ASCEND_OPP_PATH"] == str(system_opp)


def test_private_opp_install_root_requires_absolute_path(monkeypatch):
    monkeypatch.setenv("MC2_OPP_INSTALL_ROOT", "relative/opp")
    try:
        _detect_opp_path()
    except ValueError as exc:
        assert "absolute" in str(exc)
    else:
        raise AssertionError("relative MC2_OPP_INSTALL_ROOT unexpectedly accepted")


def test_install_run_package_uses_supported_makeself_flags(monkeypatch, tmp_path):
    """The generated .run installer supports --quiet and --install-path."""
    run_file = tmp_path / "custom.run"
    run_file.write_text("#!/bin/sh\nexit 0\n")
    opp_path = tmp_path / "opp"
    opp_path.mkdir()

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("ASCEND_OPP_PATH", str(opp_path))
    monkeypatch.setattr(subprocess, "run", fake_run)

    PackageManager().install_run_package(str(run_file))

    assert calls
    cmd = calls[0]
    assert "--quiet" in cmd
    assert f"--install-path={opp_path}" in cmd
    assert "--force" not in cmd


def test_install_run_package_sources_vendor_created_by_installer(monkeypatch, tmp_path):
    """The evaluator must source the actual vendor instead of assuming custom_ops."""
    run_file = tmp_path / "custom.run"
    run_file.write_text("#!/bin/sh\nexit 0\n")
    opp_path = tmp_path / "opp"
    stale_vendor = opp_path / "vendors" / "custom_ops" / "bin"
    stale_vendor.mkdir(parents=True)
    stale_vendor.joinpath("set_env.bash").write_text(
        "export ASCEND_CUSTOM_OPP_PATH=/stale/custom_ops:${ASCEND_CUSTOM_OPP_PATH}\n"
    )

    actual_vendor = opp_path / "vendors" / "custom_transformer"

    def fake_run(cmd, **kwargs):
        env_dir = actual_vendor / "bin"
        env_dir.mkdir(parents=True, exist_ok=True)
        env_dir.joinpath("set_env.bash").write_text(
            f"export ASCEND_CUSTOM_OPP_PATH={actual_vendor}:${{ASCEND_CUSTOM_OPP_PATH}}\n"
            f"export LD_LIBRARY_PATH={actual_vendor}/op_api/lib/:${{LD_LIBRARY_PATH}}\n"
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setenv("ASCEND_OPP_PATH", str(opp_path))
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(subprocess, "run", fake_run)

    PackageManager().install_run_package(str(run_file))

    assert os.environ["ASCEND_CUSTOM_OPP_PATH"].startswith(str(actual_vendor))
    normalized_ld_path = os.environ["LD_LIBRARY_PATH"].replace("\\", "/")
    assert (
        str(actual_vendor / "op_api" / "lib").replace("\\", "/") in normalized_ld_path
    )


def test_install_whl_force_reinstalls_same_version_without_dependencies(
    monkeypatch, tmp_path
):
    """A new submission must replace an installed wheel with the same version."""
    wheel = tmp_path / "cann_bench-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"placeholder")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert PackageManager().install_whl_package(str(wheel)) is True

    install_cmd = calls[0]
    assert install_cmd[1:4] == ["-m", "pip", "install"]
    assert "--force-reinstall" in install_cmd
    assert "--no-deps" in install_cmd
    assert str(wheel) in install_cmd


def test_scan_interfaces_reloads_cann_bench_submodules(monkeypatch, tmp_path):
    """A replacement submission must not reuse child modules from the old wheel."""
    package_dir = tmp_path / "cann_bench"
    package_dir.mkdir()
    package_dir.joinpath("__init__.py").write_text(
        "from .fresh_op import fresh_op\n", encoding="utf-8"
    )
    package_dir.joinpath("fresh_op.py").write_text(
        "def fresh_op(x):\n    return x\n", encoding="utf-8"
    )

    stale_package = types.ModuleType("cann_bench")
    stale_child = types.ModuleType("cann_bench.fresh_op")
    stale_package.fresh_op = lambda x: "stale"
    monkeypatch.setitem(sys.modules, "cann_bench", stale_package)
    monkeypatch.setitem(sys.modules, "cann_bench.fresh_op", stale_child)
    monkeypatch.syspath_prepend(str(tmp_path))

    interfaces = PackageManager().scan_interfaces()
    fresh_op = next(item.callable for item in interfaces if item.name == "fresh_op")

    assert fresh_op("fresh") == "fresh"
    assert sys.modules["cann_bench.fresh_op"] is not stale_child


def test_detect_opp_ignores_working_directory_when_home_is_empty(monkeypatch, tmp_path):
    from src.kernel_eval.data.package_manager import _detect_opp_path

    monkeypatch.chdir(tmp_path)
    tmp_path.joinpath("opp").mkdir()
    monkeypatch.delenv("ASCEND_OPP_PATH", raising=False)
    for home in (None, ""):
        if home is None:
            monkeypatch.delenv("ASCEND_HOME_PATH", raising=False)
        else:
            monkeypatch.setenv("ASCEND_HOME_PATH", home)
        assert _detect_opp_path() == "/usr/local/Ascend/opp"


def test_detect_opp_preserves_explicit_environment_priority(monkeypatch, tmp_path):
    from src.kernel_eval.data.package_manager import _detect_opp_path

    explicit = tmp_path / "explicit_opp"
    home = tmp_path / "toolkit"
    explicit.mkdir()
    (home / "opp").mkdir(parents=True)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(home))
    monkeypatch.setenv("ASCEND_OPP_PATH", str(explicit))
    assert _detect_opp_path() == str(explicit)
    monkeypatch.delenv("ASCEND_OPP_PATH")
    assert _detect_opp_path() == str(home / "opp")


def test_detect_opp_preserves_uninstalled_explicit_path(monkeypatch, tmp_path):
    from src.kernel_eval.data.package_manager import _detect_opp_path

    explicit = tmp_path / "new_opp"
    monkeypatch.setenv("ASCEND_OPP_PATH", str(explicit))
    monkeypatch.setenv("ASCEND_HOME_PATH", "")
    monkeypatch.setattr("os.path.isdir", lambda path: False)
    assert _detect_opp_path() == str(explicit)
