# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Package an optimized Ascend C operator for the Arena evaluator."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from auto_pipeline.converter.base import BaseConverter
from auto_pipeline.core import Artifact, CannBenchCase, Submission


class AscendArenaConverter(BaseConverter):
    name = "ascend-arena-to-ascend-arena"
    source_generator = "ascend-arena"
    target_benchmark = "ascend_arena"

    def __init__(self, *, timeout_sec: int = 7200, env: dict[str, str] | None = None) -> None:
        self.timeout_sec = timeout_sec
        self.env = dict(env or {})

    def build_submission(
        self,
        bench_name: str,
        case: CannBenchCase,
        artifact: Artifact,
        *,
        output_dir: Path,
    ) -> Submission:
        if bench_name.strip().lower().replace("-", "_") != self.target_benchmark:
            raise ValueError(f"{self.name} targets {self.target_benchmark}, got {bench_name}")
        raw_source = artifact.files.get("source_dir")
        if raw_source is None:
            raise ValueError("Ascend Arena artifact has no source_dir")
        source = Path(raw_source).expanduser().resolve()
        if not (source / "op_host").is_dir() or not (source / "op_kernel").is_dir():
            raise ValueError(f"invalid Ascend Arena operator source: {source}")

        output = Path(output_dir).expanduser().resolve()
        target = output / case.operator
        if source == output or source == target:
            raise ValueError(f"refusing to overwrite Ascend Arena artifact: {source}")
        if output.exists():
            shutil.rmtree(output)
        output.mkdir(parents=True)
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns(".tmp", "__pycache__", "*.pyc", "prof"),
        )
        metadata = {
            "converter": self.name,
            "source_generator": self.source_generator,
            "target_benchmark": self.target_benchmark,
            "operator": case.operator,
            "source_dir": str(source),
        }
        (output / "benchmark_submission.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return Submission(self.target_benchmark, case.operator, output, metadata)
