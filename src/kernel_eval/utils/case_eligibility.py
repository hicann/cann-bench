# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Execution hardware eligibility, separate from baseline fallback lookup."""

import logging
from typing import Iterable, List, Mapping, Optional, TypeVar

from .baseline_resolver import resolve_hardware


_logger = logging.getLogger(__name__)
_Case = TypeVar("_Case")


class HardwareDetectionError(RuntimeError):
    """The execution environment cannot establish usable target hardware."""


def detect_execution_hardware(
    device_type: str, device_id: Optional[int] = None
) -> Optional[str]:
    """Resolve visible NPU hardware without substituting a baseline default.

    MC2 uses the parent's visible ranks, so mixed hardware is rejected before
    scheduling rather than choosing an arbitrary card's platform.
    """
    if device_type != "npu":
        return None
    try:
        import torch
        import torch_npu  # noqa: F401

        count = torch.npu.device_count()
        if count <= 0:
            raise HardwareDetectionError("No visible NPU hardware for evaluation")
        if device_id is not None and not 0 <= device_id < count:
            raise HardwareDetectionError(
                f"NPU device index {device_id} is outside the visible set"
            )
        names = [torch.npu.get_device_name(index) for index in range(count)]
    except HardwareDetectionError:
        raise
    except Exception as exc:
        raise HardwareDetectionError(
            f"Cannot detect execution NPU hardware: {exc}"
        ) from exc
    if any(
        not isinstance(name, str)
        or not name.strip()
        or name.strip().lower() == "unknown"
        for name in names
    ):
        raise HardwareDetectionError(
            f"Cannot identify execution NPU hardware: {names!r}"
        )
    hardware = {resolve_hardware(name) for name in names}
    # A real device name need not already have a baseline alias. Preserve the
    # resolver's pass-through behavior for existing/custom hardware platforms.
    if len(hardware) != 1:
        raise HardwareDetectionError(
            f"Mixed execution NPU hardware is unsupported: {sorted(hardware)}"
        )
    return hardware.pop()


def supports_case(attrs: Optional[Mapping], target_hardware: Optional[str]) -> bool:
    """Keep unmarked cases and offline catalogs; normalize declared aliases."""
    supported = (attrs or {}).get("supported_devices")
    if target_hardware is None or not supported:
        return True
    if not isinstance(supported, (list, tuple)) or not all(
        isinstance(item, str) for item in supported
    ):
        raise ValueError("supported_devices must be a list of hardware names")
    return resolve_hardware(target_hardware) in {
        resolve_hardware(item) for item in supported
    }


def filter_cases_for_hardware(
    cases: Iterable[_Case], target_hardware: Optional[str]
) -> List[_Case]:
    """Filter case specs or raw YAML dictionaries before counting or scheduling."""
    selected = []
    excluded = 0
    for case in cases:
        attrs = (
            case.get("attrs")
            if isinstance(case, dict)
            else getattr(case, "attrs", None)
        )
        if supports_case(attrs, target_hardware):
            selected.append(case)
        else:
            excluded += 1
    if excluded:
        _logger.info(
            "Excluded %d cases unsupported on %s before evaluation",
            excluded,
            target_hardware,
        )
    return selected
