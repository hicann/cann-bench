# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Helpers for selecting one or more numeric benchmark case IDs."""

import argparse
from typing import Tuple


def parse_case_id_range(value: str) -> Tuple[int, int]:
    """Parse an inclusive ``START-END`` case range for argparse."""
    text = str(value).strip()
    parts = text.split("-", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "case range must use START-END, for example 21-40"
        )
    try:
        start, end = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("case range bounds must be integers") from exc
    if start <= 0 or end <= 0:
        raise argparse.ArgumentTypeError("case range bounds must be positive")
    if start > end:
        raise argparse.ArgumentTypeError("case range start must not exceed its end")
    return start, end


def case_id_in_range(case_id: int, bounds: Tuple[int, int]) -> bool:
    """Return whether *case_id* is inside an inclusive range."""
    start, end = bounds
    return start <= int(case_id) <= end
