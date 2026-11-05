# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""The documented submission contract includes the complete executable Golden."""

import ast
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = ROOT / "bench_lab" / "kernel_bench" / "level4"
MC2_TASKS = (
    "all_gather_matmul",
    "all_gather_matmul_v2",
    "allto_all_matmul",
    "allto_allv_grouped_mat_mul",
    "allto_allv_quant_grouped_mat_mul",
    "grouped_mat_mul_allto_allv",
    "matmul_all_reduce",
    "matmul_allto_all",
    "matmul_reduce_scatter_v2",
    "quant_all_reduce",
    "quant_grouped_mat_mul_allto_allv",
    "quant_reduce_scatter",
)


@pytest.mark.parametrize("task", MC2_TASKS)
def test_documented_golden_is_the_complete_submission_contract(task):
    directory = TASK_ROOT / task
    document = (directory / "desc.md").read_text(encoding="utf-8")
    section = re.search(r"(?ms)^## 5\. 标准 Golden 代码\n(.*?)(?=^## |\Z)", document)
    assert section is not None
    blocks = re.findall(r"(?ms)^```python\n(.*?)^```\s*$", section.group(1))
    assert len(blocks) == 1
    assert (
        blocks[0].strip()
        == (directory / "golden.py").read_text(encoding="utf-8").strip()
    )


def test_quant_grouped_golden_only_requires_torch():
    golden = TASK_ROOT / "quant_grouped_mat_mul_allto_allv" / "golden.py"
    tree = ast.parse(golden.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert imports == ["torch"]
