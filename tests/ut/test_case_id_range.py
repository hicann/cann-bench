# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import argparse

import pytest

from kernel_eval import staged_eval
from kernel_eval.cli import create_parser
from kernel_eval.utils.case_selector import case_id_in_range, parse_case_id_range


def test_parse_case_id_range_is_inclusive():
    bounds = parse_case_id_range("21-40")

    assert bounds == (21, 40)
    assert case_id_in_range(21, bounds)
    assert case_id_in_range(40, bounds)
    assert not case_id_in_range(20, bounds)
    assert not case_id_in_range(41, bounds)


@pytest.mark.parametrize(
    "value, message",
    [
        ("21", "case range must use START-END"),
        ("40-21", "case range start must not exceed its end"),
        ("0-20", "case range bounds must be positive"),
        ("a-b", "case range bounds must be integers"),
    ],
)
def test_parse_case_id_range_rejects_invalid_values(value, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        parse_case_id_range(value)


@pytest.mark.parametrize(
    "parser_factory, prefix",
    [
        (create_parser, ["eval"]),
        (staged_eval.create_parser, []),
    ],
)
@pytest.mark.parametrize(
    "value, message",
    [
        ("21", "case range must use START-END"),
        ("40-21", "case range start must not exceed its end"),
        ("0-20", "case range bounds must be positive"),
        ("a-b", "case range bounds must be integers"),
    ],
)
def test_case_range_cli_preserves_error_detail(
    parser_factory, prefix, value, message, capsys
):
    with pytest.raises(SystemExit) as exc:
        parser_factory().parse_args([*prefix, "--case-id-range", value])

    assert exc.value.code == 2
    error = capsys.readouterr().err
    assert "--case-id-range" in error
    assert message in error
    assert "Traceback" not in error


def test_eval_parser_accepts_case_id_range():
    args = create_parser().parse_args(["eval", "--case-id-range", "21-40"])

    assert args.case_id_range == (21, 40)


def test_staged_parser_accepts_case_id_range():
    args = staged_eval.create_parser().parse_args(["--case-id-range", "21-40"])

    assert args.case_id_range == (21, 40)


@pytest.mark.parametrize(
    "parser_args",
    [
        ["eval", "--case-id", "21", "--case-id-range", "21-40"],
    ],
)
def test_eval_parser_rejects_single_id_with_range(parser_args):
    with pytest.raises(SystemExit):
        create_parser().parse_args(parser_args)
