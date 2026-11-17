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


"""Generate the self-contained report for a prepared batch run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main(argv: list[str] | None = None) -> int:
    from kernel_eval.batch_report import write_batch_report

    parser = argparse.ArgumentParser(description="Generate one self-contained batch optimization report.")
    parser.add_argument("run_root", type=Path, help="batch run directory containing operator workspaces")
    parser.add_argument("--output", type=Path, help="output HTML; defaults to <run_root>/report.html")
    args = parser.parse_args(argv)
    print(write_batch_report(args.run_root, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
