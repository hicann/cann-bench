"""AKG agent artifact to Stanford/KernelBench submission converter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping, Optional

from auto_pipeline.generator.akg.converter.to_cann import (
    _generated_code_from_output,
    _implementation_source,
    _safe_submission_dir,
)
from auto_pipeline.converter.base import BaseConverter
from auto_pipeline.converter.submission import validate_submission
from auto_pipeline.core import Artifact, Submission
from auto_pipeline.core import CannBenchCase


class AkgToStanfordConverter(BaseConverter):
    """Packages AKG generated Triton Ascend code as Stanford submissions."""

    name = "akg-agent-to-stanford"
    source_generator = "akg-agent"
    target_benchmark = "stanford"

    def __init__(self, *, timeout_sec: int = 7200, env: Optional[Mapping[str, str]] = None) -> None:
        self.timeout_sec = timeout_sec
        self.env = {str(key): str(value) for key, value in dict(env or {}).items()}

    def build_submission(
        self,
        bench_name: str,
        case: CannBenchCase,
        artifact: Artifact,
        *,
        output_dir: Path,
    ) -> Submission:
        if bench_name != self.target_benchmark:
            raise ValueError(f"{self.name} targets benchmark {self.target_benchmark}, got {bench_name}")
        code, source = _generated_code_from_output(artifact)
        if not code.strip():
            raise ValueError("generated AKG Triton Ascend code is empty")

        source_dir = _safe_submission_dir(output_dir, artifact)
        if source_dir.exists():
            shutil.rmtree(source_dir)
        source_dir.mkdir(parents=True, exist_ok=True)
        source_dir.joinpath("ai_op.py").write_text(_implementation_source(code), encoding="utf-8")

        metadata = {
            "converter": self.name,
            "source_generator": self.source_generator,
            "target_benchmark": self.target_benchmark,
            "operator": case.operator,
            "artifact_status": artifact.status,
            "artifact_message": artifact.message,
            "source": source,
        }
        source_dir.joinpath("benchmark_submission.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        validate_submission(self.target_benchmark, case, source_dir, label="akg-agent")
        return Submission(self.target_benchmark, case.operator, source_dir, metadata)
