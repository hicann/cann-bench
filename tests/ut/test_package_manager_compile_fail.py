#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""编译失败 = 整批 0 分（不隔离/不补救/不改源码）测试。

验证去掉迭代隔离后：build.sh 任一算子编译失败 → build_packages 返回 build_failed=True，
compile_errors 覆盖本次提交的所有算子，且**不修改用户源码、不产生 _quarantine 目录**。
"""

from pathlib import Path

from kernel_eval.data.package_manager import PackageManager


# build.sh：bad_op 报一条可被 _OP_ERROR_LINE_RE 命中的编译错误并失败（不产出 whl）。
_FAIL_BUILD_SH = """#!/bin/bash
echo "[ 50%] Building CXX object csrc/ops/bad_op/op_kernel/bad.cpp:5:1: error: boom"
echo "make: *** [csrc/ops/bad_op/op_kernel/bad.cpp.o] Error 1"
exit 1
"""

# build.sh：成功，产出 dist 下的 whl。
_OK_BUILD_SH = """#!/bin/bash
mkdir -p dist
: > dist/cann_bench-0.0.0-py3-none-any.whl
exit 0
"""


def _make_submission(root: Path, build_sh: str) -> Path:
    sub = root / "submission"
    for op in ("good_op", "bad_op"):
        (sub / "csrc" / "ops" / op / "op_kernel").mkdir(parents=True)
        (sub / "csrc" / "ops" / op / "op_kernel" / f"{op}.cpp").write_text("// x\n")
    (sub / "build.sh").write_text(build_sh, encoding="utf-8", newline="\n")
    return sub


def test_compile_failure_zeros_all_ops_without_touching_source(tmp_path):
    sub = _make_submission(tmp_path, _FAIL_BUILD_SH)
    pm = PackageManager()

    pkg = pm.build_packages(str(sub))

    # 1) 整体编译失败标志 + 无 whl
    assert pkg.build_failed is True
    assert not pkg.whl_path

    # 2) 本次提交的所有算子都被标记为编译失败（→ 全部计 0 分）
    assert set(pkg.compile_errors.keys()) == {"good_op", "bad_op"}
    # 能定位到的 bad_op 带具体错误片段；good_op 带整体日志摘要
    assert (
        "bad_op" in pkg.compile_errors["bad_op"]
        or "error" in pkg.compile_errors["bad_op"].lower()
    )

    # 3) 关键：用户源码零改动，且不产生 _quarantine 目录（不隔离/不挪源码）
    assert (sub / "csrc" / "ops" / "good_op" / "op_kernel" / "good_op.cpp").is_file()
    assert (sub / "csrc" / "ops" / "bad_op" / "op_kernel" / "bad_op.cpp").is_file()
    assert not (sub / "_quarantine").exists()


def test_compile_success_no_errors(tmp_path):
    sub = _make_submission(tmp_path, _OK_BUILD_SH)
    pm = PackageManager()

    pkg = pm.build_packages(str(sub))

    assert pkg.build_failed is False
    assert pkg.compile_errors == {}
    assert pkg.whl_path and Path(pkg.whl_path).is_file()


# build.sh：编译算子内核之前就失败（cmake/依赖错误），日志不含可识别的 csrc/ops/<op> 错误模式。
_EARLY_FAIL_BUILD_SH = """#!/bin/bash
echo "CMake Error: Could NOT find Python (missing: Python_INCLUDE_DIRS)"
echo "make: *** No rule to make target. Stop."
exit 1
"""


def test_compile_failure_no_ops_falls_back_to_build_sentinel(tmp_path):
    """提交无 csrc/ops/ 且日志不含算子错误 → compile_errors 退化为 {"<build>": ...}。

    这是 evaluator/cli 兜底逻辑（提交级编译失败记录）的触发前提。
    """
    sub = tmp_path / "submission"
    sub.mkdir()
    (sub / "build.sh").write_text(_EARLY_FAIL_BUILD_SH)

    pm = PackageManager()
    pkg = pm.build_packages(str(sub))

    assert pkg.build_failed is True
    assert not pkg.whl_path
    # 没有 csrc/ops/* 也没有可解析的算子错误 → 兜底键 <build>
    assert list(pkg.compile_errors.keys()) == ["<build>"]
    assert "编译失败" in pkg.compile_errors["<build>"]


def test_parse_failing_ops_ignores_crossref_note_lines():
    """note/交叉引用行不应把无辜算子误判为编译失败算子。

    形如 `good.cpp:10:5: note: previous error from .../bad.cpp` 的行同时出现
    good_op 路径与 'error' 字样，旧正则会误识别 good_op。修复后只认编译器诊断
    规范位置（file.cpp:line:col: error）与 make 的 `Error N` 行。
    """
    pm = PackageManager()
    log = (
        "[ 30%] Building CXX object csrc/ops/good_op/op_kernel/good.cpp.o\n"
        "csrc/ops/good_op/op_kernel/good.cpp:10:5: note: previous error from "
        "csrc/ops/bad_op/op_kernel/bad.cpp\n"
        "csrc/ops/good_op/op_kernel/good.cpp:7:2: warning: unused variable\n"
        "csrc/ops/bad_op/op_kernel/bad.cpp:5:1: error: 'X' was not declared\n"
        "make[2]: *** [csrc/ops/bad_op/op_kernel/bad.cpp.o] Error 1\n"
    )
    ops = pm._parse_failing_ops(log)

    # 只识别真正报 error 的 bad_op，不把 good_op（note/warning/building 行）算进去
    assert set(ops.keys()) == {"bad_op"}


def test_parse_failing_ops_recognizes_fatal_error():
    """clang/bisheng 的 `fatal error:`（如缺头文件）也应被识别。"""
    pm = PackageManager()
    log = "csrc/ops/foo_op/op_host/foo.cpp:3:10: fatal error: 'foo.h' file not found\n"
    assert set(pm._parse_failing_ops(log).keys()) == {"foo_op"}


class _FakeOp:
    def __init__(self, name, rel_path=""):
        self.name = name
        self.rel_path = rel_path


class _FakeMatcher:
    """snake 名 -> _FakeOp 的反查表替身（None 表示未注册）。"""

    def __init__(self, mapping):
        self.mapping = mapping

    def find_operator_info_by_snake(self, snake):
        return self.mapping.get(snake)

    def find_operator_info(self, name):
        return next((op for op in self.mapping.values() if op.name == name), None)


def _pkg(compile_errors):
    class _P:
        pass

    p = _P()
    p.compile_errors = compile_errors
    p.build_failed = True
    return p


def _synth():
    from kernel_eval.eval.failure_synthesizer import FailureSynthesizer

    # 提交级/编译失败合成不依赖 case_loader（_synthesize_failure 对异常兜底为空用例集）。
    return FailureSynthesizer(case_loader=None)


def test_synth_all_maps_every_registered_op():
    m = _FakeMatcher(
        {"exp": _FakeOp("Exp", "level1/exp"), "foo": _FakeOp("Foo", "level1/foo")}
    )
    res = _synth().synthesize_all_compile_failures(m, _pkg({"exp": "e1", "foo": "e2"}))
    assert sorted(r.operator for r in res) == ["Exp", "Foo"]
    assert all(r.compilation_error for r in res)


def test_synth_all_respects_operator_filter():
    m = _FakeMatcher({"exp": _FakeOp("Exp"), "foo": _FakeOp("Foo")})
    res = _synth().synthesize_all_compile_failures(
        m, _pkg({"exp": "e1", "foo": "e2"}), operator_filter=["Exp"]
    )
    assert [r.operator for r in res] == ["Exp"]


def test_synth_all_build_sentinel_falls_back_to_submission_level():
    m = _FakeMatcher({})  # 没有任何算子能反查到 spec
    res = _synth().synthesize_all_compile_failures(
        m, _pkg({"<build>": "boom cmake error"})
    )
    assert len(res) == 1 and res[0].operator == "<submission>"
    assert "boom cmake error" in (res[0].compilation_error or "")


def test_synth_all_build_sentinel_uses_selected_real_operator():
    """Issue #128: 单算子早期编译失败不得新增 <submission> 伪算子。"""
    op = _FakeOp("Conv3DBackpropFilter", "level3/conv_3d_backprop_filter")
    m = _FakeMatcher({"conv_3d_backprop_filter": op})

    res = _synth().synthesize_all_compile_failures(
        m,
        _pkg({"<build>": "boom cmake error"}),
        operator_filter=["conv_3d_backprop_filter"],
    )

    assert [r.operator for r in res] == ["Conv3DBackpropFilter"]
    assert res[0].rel_path == "level3/conv_3d_backprop_filter"
    assert "boom cmake error" in (res[0].compilation_error or "")


def test_synth_all_operator_filter_accepts_snake_case_for_mapped_error():
    op = _FakeOp("Conv3DBackpropFilter", "level3/conv_3d_backprop_filter")
    m = _FakeMatcher({"conv_3d_backprop_filter": op})

    res = _synth().synthesize_all_compile_failures(
        m,
        _pkg({"conv_3d_backprop_filter": "compiler error"}),
        operator_filter=["conv_3d_backprop_filter"],
    )

    assert [r.operator for r in res] == ["Conv3DBackpropFilter"]


def test_synth_all_filtered_out_does_not_trigger_submission_fallback():
    """算子能映射但被 operator_filter 过滤掉时，不应虚报提交级失败（潜在 bug 回归）。"""
    m = _FakeMatcher({"foo": _FakeOp("Foo")})
    res = _synth().synthesize_all_compile_failures(
        m, _pkg({"foo": "e"}), operator_filter=["Exp"]
    )
    assert res == []


def test_synth_all_mixed_mapped_and_unregistered():
    """部分算子能映射、部分是未注册/兜底键时：只合成能映射的，不再额外兜底提交级。"""
    m = _FakeMatcher({"foo": _FakeOp("Foo")})  # "bar" 反查不到
    res = _synth().synthesize_all_compile_failures(m, _pkg({"foo": "e1", "bar": "e2"}))
    assert [r.operator for r in res] == ["Foo"]


def _fake_report(failed_cases, compile_errors):
    """构造最小 report 替身，仅含 _compute_exit_code 关心的字段。"""

    class _Op:
        def __init__(self, ce):
            self.compilation_error = ce

    class _Report:
        def __init__(self):
            self.failed_cases = failed_cases
            self.operators = [_Op(ce) for ce in compile_errors]

    return _Report()


def test_exit_code_zero_when_all_pass():
    from kernel_eval.cli import _compute_exit_code

    assert _compute_exit_code(_fake_report(0, [None, None])) == 0


def test_exit_code_nonzero_on_genuine_failures():
    from kernel_eval.cli import _compute_exit_code

    assert _compute_exit_code(_fake_report(5, [None])) == 5


def test_exit_code_nonzero_on_compile_failure():
    """编译失败（compilation_error 非空）即使 failed_cases==0 也必须退出非零。"""
    from kernel_eval.cli import _compute_exit_code

    # 1 个算子编译失败、无真实失败用例 → 退出码 1（而非 0）
    assert _compute_exit_code(_fake_report(0, ["compile failed: ..."])) == 1
    # 多算子编译失败累加
    assert _compute_exit_code(_fake_report(0, ["e1", "e2", None])) == 2


def test_exit_code_capped_at_255():
    from kernel_eval.cli import _compute_exit_code

    assert _compute_exit_code(_fake_report(1000, ["e"])) == 255


def test_submission_level_compile_failure_is_visible():
    """synthesize_submission_compile_failure 生成一条提交级 all-FAIL 记录（避免空报告静默失败）。"""
    from kernel_eval.eval.failure_synthesizer import FailureSynthesizer

    # 提交级合成不依赖 case_loader（不按算子展开用例），传 None 即可。
    synth = FailureSynthesizer(case_loader=None)
    res = synth.synthesize_submission_compile_failure(
        "[<build>] build.sh 编译失败：CMake Error"
    )

    # 报告可见：非空、计为失败、0 分、带编译错误诊断
    assert res.total_cases == 1
    assert res.failed_cases == 1
    assert res.passed_cases == 0
    assert res.compilation_error and "CMake Error" in res.compilation_error
    # rel_path 留空，避免被错误归入某个 level / 触发 rel_path 解析崩溃
    assert res.rel_path == ""
    assert res.results and res.results[0].success is False
