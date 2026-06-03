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

"""
回归测试 (gitcode issue #11): 子进程隔离路径必须把父进程的性能/profiler 配置
透传给真正执行算子的子进程，否则 --no-perf / --warmup / --repeat /
--profiler-level 在默认隔离路径下被静默忽略。
"""

from unittest.mock import MagicMock

from kernel_eval.config import Config
from kernel_eval.eval.subprocess_runner import SubprocessRunner


def _runner(config):
    return SubprocessRunner(
        failure_synthesizer=MagicMock(),
        device_id=config.device_id,
        kernel_eval_root="/tmp/src",
        config=config,
    )


def _pairs(cmd, flag):
    """返回紧跟在 flag 后面的值（按出现顺序）。"""
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag and i + 1 < len(cmd)]


def test_no_perf_forwarded_when_profiler_disabled():
    cfg = Config()
    cfg.enable_profiler = False
    cmd = _runner(cfg)._build_child_cmd("Cummin", "/tmp/frag.json", source_dir="cand")
    assert "--no-perf" in cmd


def test_no_perf_absent_when_profiler_enabled():
    cfg = Config()
    cfg.enable_profiler = True
    cmd = _runner(cfg)._build_child_cmd("Cummin", "/tmp/frag.json", source_dir="cand")
    assert "--no-perf" not in cmd


def test_warmup_repeat_profiler_level_device_forwarded():
    cfg = Config()
    cfg.warmup = 7
    cfg.repeat = 11
    cfg.profiler_level = "Level2"
    cfg.device_id = 3
    cmd = _runner(cfg)._build_child_cmd("Cummin", "/tmp/frag.json", source_dir="cand")
    assert _pairs(cmd, "--warmup") == ["7"]
    assert _pairs(cmd, "--repeat") == ["11"]
    assert _pairs(cmd, "--profiler-level") == ["Level2"]
    assert _pairs(cmd, "--device-id") == ["3"]


def test_base_invariants_preserved():
    cfg = Config()
    cmd = _runner(cfg)._build_child_cmd("Cummin", "/tmp/frag.json", source_dir="cand")
    # 隔离子进程必须避免无限递归与重复安装
    assert "--no-subprocess-isolation" in cmd
    assert "--skip-install" in cmd
    assert _pairs(cmd, "--operator") == ["Cummin"]
    assert _pairs(cmd, "--child-json-output") == ["/tmp/frag.json"]
    assert _pairs(cmd, "--source-dir") == ["cand"]


def test_unbuffered_adds_dash_u():
    cfg = Config()
    cmd = _runner(cfg)._build_child_cmd(
        "Cummin", "/tmp/frag.json", source_dir="cand", unbuffered=True
    )
    assert "-u" in cmd


def test_case_id_forwarded():
    cfg = Config()
    cmd = _runner(cfg)._build_child_cmd(
        "Cummin", "/tmp/frag.json", source_dir="cand", case_filter={"case_id": 5}
    )
    assert _pairs(cmd, "--case-id") == ["5"]


def test_task_dir_forwarded_from_config():
    cfg = Config()
    cfg.tasks_root = "/repo/bench_lab/pypto_cann_bench"

    cmd = _runner(cfg)._build_child_cmd("Sigmoid", "/tmp/frag.json", source_dir="cand")

    assert _pairs(cmd, "--task-dir") == ["/repo/bench_lab/pypto_cann_bench"]


def test_task_dir_prefers_operator_dir_when_rel_path_exists(tmp_path):
    bench_root = tmp_path / "bench_lab" / "pypto_cann_bench"
    task_dir = bench_root / "sigmoid"
    task_dir.mkdir(parents=True)
    cfg = Config()
    cfg.tasks_root = str(bench_root)
    runner = _runner(cfg)

    assert runner._task_dir_arg("sigmoid") == str(task_dir)


def test_no_config_is_safe():
    """config 缺省时不应崩溃，也不应透传任何 perf flag。"""
    runner = SubprocessRunner(failure_synthesizer=MagicMock())
    cmd = runner._build_child_cmd("Cummin", "/tmp/frag.json", source_dir="cand")
    assert "--no-perf" not in cmd
    assert "--warmup" not in cmd
    assert "--task-dir" not in cmd
