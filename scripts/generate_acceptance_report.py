#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Create a reviewer-friendly acceptance report from an Ascend Arena run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    from kernel_eval.acceptance_report import write_acceptance_report

    parser = argparse.ArgumentParser(
        description="Generate a portable repository feature-completeness report and report_data.json."
    )
    parser.add_argument(
        "run_root",
        type=Path,
        help=(
            "batch, single-operator, workspace, or candidate result root "
            "(e2e_summary.json/config.json/kernel_eval/results/status.json)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Markdown output; defaults to <run_root>/acceptance_report/acceptance_report.md",
    )
    parser.add_argument(
        "--screenshots-dir",
        type=Path,
        help="directory containing PNG/JPG/WebP screenshots to copy into the report",
    )
    parser.add_argument(
        "--screenshot",
        action="append",
        default=[],
        type=Path,
        help="individual screenshot to copy; may be repeated",
    )
    parser.add_argument(
        "--command",
        default="",
        help="exact command used for the test, recorded verbatim in the report",
    )
    parser.add_argument(
        "--feature-results",
        type=Path,
        help="JSON file containing README feature checks; defaults to <run_root>/feature_results.json",
    )
    args = parser.parse_args(argv)
    output = write_acceptance_report(
        args.run_root,
        args.output,
        command=args.command,
        screenshots_dir=args.screenshots_dir,
        screenshot_files=args.screenshot,
        feature_results=args.feature_results,
    )
    print(output)
    print(output.with_name("report_data.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
