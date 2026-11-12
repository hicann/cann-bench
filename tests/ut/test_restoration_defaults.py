# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Default protection and explicit opt-out survive every CLI/child boundary."""

import pytest

from kernel_eval.cli import _create_config_from_args_for_child, create_parser
from kernel_eval.config import Config
from kernel_eval.eval.process_pool import (
    ProcessConfig,
    ProcessPoolCoordinator,
    TaskUnit,
)
from kernel_eval.staged_eval import _make_config
from kernel_eval.staged_eval import create_parser as staged_parser
from tests.ut.test_process_pool import make_case


def test_config_restores_by_default_with_programmatic_opt_out():
    assert Config().perf_restore_inputs is True
    assert Config(perf_restore_inputs=False).perf_restore_inputs is False


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], True),
        (["--perf-restore-inputs"], True),
        (["--no-perf-restore-inputs"], False),
    ],
)
def test_cli_and_staged_defaults_and_opt_out(tmp_path, flags, expected):
    parser = create_parser()
    assert parser.parse_args(["eval", *flags]).perf_restore_inputs is expected
    child = parser.parse_args(
        [
            "eval-child",
            "--device-id",
            "0",
            "--cases-file",
            "cases.json",
            "--output",
            "out.json",
            *flags,
        ]
    )
    assert (
        _create_config_from_args_for_child(child, str(tmp_path)).perf_restore_inputs
        is expected
    )
    staged = staged_parser().parse_args(flags)
    for profiler in (True, False):
        cfg = _make_config(staged, str(tmp_path), enable_profiler=profiler)
        assert cfg.perf_restore_inputs is expected
        assert cfg.enable_profiler is profiler


@pytest.mark.parametrize("enabled", [True, False])
def test_child_command_roundtrip_preserves_explicit_selection(tmp_path, enabled):
    config = Config(device_type="cpu", perf_restore_inputs=enabled)
    coordinator = ProcessPoolCoordinator(
        base_config=config, process_config=ProcessConfig(enable_profiler=True)
    )
    task = TaskUnit(
        operator="Exp", rel_path="level1/test", cases=[make_case("Exp", 1)], device_id=0
    )
    command = coordinator._build_child_cmd(
        task, str(tmp_path / "cases.json"), str(tmp_path / "out.json")
    )
    args = create_parser().parse_args(command[command.index("eval-child") :])
    restored = _create_config_from_args_for_child(args, str(tmp_path))
    assert restored.perf_restore_inputs is enabled
    assert args.perf_batch_cases is (not enabled)


def test_missing_namespace_option_keeps_new_default(tmp_path):
    args = staged_parser().parse_args([])
    del args.perf_restore_inputs
    assert _make_config(args, str(tmp_path), enable_profiler=True).perf_restore_inputs
