# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Benchmark artifact path resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path


def artifact_env_var_name(filename: str) -> str:
    """Derive the environment variable name for a benchmark artifact file."""
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", filename).strip("_").upper()
    return f"BENCHMARK_{normalized}_PATH"


def resolve_benchmark_artifact_path(default_path: Path) -> Path:
    """Return the artifact path, respecting env-var overrides."""
    override = os.environ.get(artifact_env_var_name(default_path.name), "").strip()
    if override:
        return Path(override).resolve()
    return default_path
