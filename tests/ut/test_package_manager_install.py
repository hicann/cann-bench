#!/usr/bin/python3
# coding=utf-8

import subprocess

from src.kernel_eval.data.package_manager import PackageManager


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
