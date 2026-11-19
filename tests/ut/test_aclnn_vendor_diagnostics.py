#!/usr/bin/python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# See the repository LICENSE for the full text.

"""Exercise ACLNN loader diagnostics with real host shared libraries, without an NPU."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch


REPOSITORY = Path(__file__).resolve().parents[2]
HEADER = REPOSITORY / "examples/aclnn_launch_example/csrc/ops/_common/aclnn_common.h"


@pytest.fixture(scope="module")
def loader(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("ACLNN loader diagnostics require a host C++ compiler")
    directory = tmp_path_factory.mktemp("aclnn_loader")
    source = directory / "loader.cpp"
    header = HEADER.read_text(encoding="utf-8")
    # Compile the actual loader helpers with c10's real TORCH_CHECK. The rest of
    # aclnn_common.h needs the NPU SDK; only the logging macros are stubbed here.
    start = header.index("inline const char")
    end = header.index("inline c10::Scalar ConvertTensorToScalar")
    source.write_text(
        """
#include <c10/util/Exception.h>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <string>
#define ASCEND_LOGW(...) ((void)0)
"""
        + header[start:end]
        + """
int main(int argc, char **argv)
{
    try {
        if (std::string(argv[1]) == "library") {
            GetOpApiLibHandler(argv[2], RTLD_LAZY, true);
        } else {
            bool required = std::string(argv[1]) == "required";
            auto address = GetOpApiFuncAddr(argv[2], required);
            if (address) {
                std::cout << reinterpret_cast<int (*)()>(address)();
            } else {
                std::cout << "optional symbol absent";
            }
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what();
        return 1;
    }
}
""",
        encoding="utf-8",
    )
    torch_root = Path(torch.__file__).resolve().parent
    executable = directory / "loader"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            str(source),
            "-I",
            str(torch_root / "include"),
            "-L",
            str(torch_root / "lib"),
            f"-Wl,-rpath,{torch_root / 'lib'}",
            "-lc10",
            "-ldl",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return compiler, executable


def _library(compiler, directory, name, source, *link_args):
    source_file = directory / f"{name}.cpp"
    source_file.write_text(source, encoding="utf-8")
    output = directory / name
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(source_file), *link_args, "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def _run(loader, directory, mode, argument):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(directory)
    env["ASCEND_CUSTOM_OPP_PATH"] = str(directory / "vendor")
    return subprocess.run(
        [str(loader[1]), mode, str(argument)],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )


def test_missing_library_reports_environment_and_original_error(loader, tmp_path):
    missing_library = tmp_path / "libcust_opapi.so"
    result = _run(loader, tmp_path, "library", missing_library)

    assert result.returncode == 1
    assert f"Failed to load {missing_library}" in result.stderr
    assert "No such file or directory" in result.stderr
    assert "<vendor>/bin/set_env.bash" in result.stderr
    assert "before starting Python" in result.stderr
    assert f"LD_LIBRARY_PATH={tmp_path}" in result.stderr
    assert f"ASCEND_CUSTOM_OPP_PATH={tmp_path / 'vendor'}" in result.stderr


def test_missing_dependency_is_not_reported_as_missing_operator(loader, tmp_path):
    compiler, _ = loader
    dependency_name = "libaclnn_test_dependency.so"
    dependency = _library(
        compiler,
        tmp_path,
        dependency_name,
        'extern "C" int dependency() { return 17; }',
        f"-Wl,-soname,{dependency_name}",
    )
    _library(
        compiler,
        tmp_path,
        "libcust_opapi.so",
        'extern "C" int dependency(); extern "C" int aclnnAdd() { return dependency(); }',
        "-L",
        str(tmp_path),
        "-laclnn_test_dependency",
    )
    dependency.unlink()

    result = _run(loader, tmp_path, "required", "aclnnAdd")

    assert result.returncode == 1
    assert "Failed to load libcust_opapi.so" in result.stderr
    assert dependency_name in result.stderr
    assert "ldd <vendor>/op_api/lib/libcust_opapi.so" in result.stderr
    assert "Required custom operator symbol" not in result.stderr


@pytest.mark.parametrize("symbol", ["aclnnAdd", "aclnnAddGetWorkspaceSize"])
def test_missing_custom_symbol_does_not_fall_back_to_builtin(loader, tmp_path, symbol):
    compiler, _ = loader
    _library(
        compiler,
        tmp_path,
        "libcust_opapi.so",
        'extern "C" int unrelated() { return 17; }',
    )
    _library(
        compiler, tmp_path, "libopapi.so", f'extern "C" int {symbol}() {{ return 99; }}'
    )

    result = _run(loader, tmp_path, "required", symbol)

    assert result.returncode == 1
    assert f"Required custom operator symbol '{symbol}' was not found" in result.stderr
    assert str(tmp_path / "libcust_opapi.so") in result.stderr
    assert "matching vendor" in result.stderr
    assert "Failed to load" not in result.stderr


def test_custom_symbol_loads_after_vendor_environment_is_set(loader, tmp_path):
    _library(
        loader[0],
        tmp_path,
        "libcust_opapi.so",
        'extern "C" int aclnnAdd() { return 17; }',
    )

    result = _run(loader, tmp_path, "required", "aclnnAdd")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "17"


def test_optional_runtime_symbol_remains_optional(loader, tmp_path):
    _library(
        loader[0],
        tmp_path,
        "libcust_opapi.so",
        'extern "C" int aclnnAdd() { return 17; }',
    )

    result = _run(loader, tmp_path, "optional", "InitHugeMemThreadLocal")

    assert result.returncode == 0, result.stderr
    assert result.stdout == "optional symbol absent"
