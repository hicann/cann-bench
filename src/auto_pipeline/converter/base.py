"""Artifact conversion contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from auto_pipeline.generator.base import Runner
from auto_pipeline.core import Artifact, RunnerPrompt, Submission
from auto_pipeline.core import render_prompt_file
from auto_pipeline.core import CannBenchCase


_DEFAULT_CONVERSION_TEMPLATE = Path(__file__).with_name("templates") / "default.j2"


@dataclass(frozen=True)
class ConversionResult:
    """Result of converting a generated artifact into a benchmark submission."""

    artifact: Artifact
    submission: Submission | None = None
    prompt_file: Path | None = None

    @property
    def ok(self) -> bool:
        return self.artifact.ok and self.submission is not None


class Converter(Protocol):
    """Converts a generated artifact into a benchmark submission."""

    name: str
    source_generator: str
    target_benchmark: str

    def convert(
        self,
        bench_name: str,
        case: CannBenchCase,
        artifact: Artifact,
        *,
        output_dir: Path,
        runner: Runner | None = None,
        workdir: Path | None = None,
    ) -> ConversionResult:
        ...


class BaseConverter:
    """Shared converter flow; concrete converters own submission packaging."""

    name: str
    source_generator: str
    target_benchmark: str
    timeout_sec: int
    env: dict[str, str]

    def convert(
        self,
        bench_name: str,
        case: CannBenchCase,
        artifact: Artifact,
        *,
        output_dir: Path,
        runner: Runner | None = None,
        workdir: Path | None = None,
    ) -> ConversionResult:
        conversion_artifact = artifact
        prompt_file = None
        if runner is not None:
            convert_workdir = Path(workdir or Path(output_dir).parent / "convert").expanduser().resolve()
            convert_output_dir = convert_workdir / "artifact"
            prompt = self.build_conversion_prompt(
                bench_name,
                case,
                artifact,
                workdir=convert_workdir,
                output_dir=convert_output_dir,
                submission_dir=output_dir,
            )
            conversion_artifact = runner.run(prompt)
            prompt_file = prompt.output_dir / "PROMPT.md"
            if not prompt_file.is_file():
                prompt_file = None

        if not conversion_artifact.ok:
            return ConversionResult(artifact=conversion_artifact, prompt_file=prompt_file)

        submission = self.build_submission(bench_name, case, conversion_artifact, output_dir=output_dir)
        return ConversionResult(artifact=conversion_artifact, submission=submission, prompt_file=prompt_file)

    def build_conversion_prompt(
        self,
        bench_name: str,
        case: CannBenchCase,
        artifact: Artifact,
        *,
        workdir: Path,
        output_dir: Path,
        submission_dir: Path,
    ) -> RunnerPrompt:
        files = "\n".join(f"- {key}: {path}" for key, path in artifact.files.items()) or "- <none>"
        text = render_prompt_file(
            _DEFAULT_CONVERSION_TEMPLATE,
            source_generator=self.source_generator,
            target_benchmark=self.target_benchmark,
            operator=case.operator,
            raw_workdir=artifact.workdir,
            files=files,
            submission_path=output_dir / "submission",
        )
        return RunnerPrompt(
            text=text,
            cwd=workdir,
            output_dir=output_dir,
            timeout_sec=self.timeout_sec,
            env=dict(self.env),
            title=f"convert:{self.name}:{case.operator}",
            files=dict(artifact.files),
            metadata={
                "source_generator": self.source_generator,
                "target_benchmark": self.target_benchmark,
                "converter": self.name,
                "operator": case.operator,
                "benchmark": bench_name,
                "raw_workdir": str(artifact.workdir),
                "submission_dir": str(submission_dir),
            },
        )

    def build_submission(
        self,
        bench_name: str,
        case: CannBenchCase,
        artifact: Artifact,
        *,
        output_dir: Path,
    ) -> Submission:
        raise NotImplementedError
