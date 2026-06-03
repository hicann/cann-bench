"""Shared PyPTO artifact converters."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Optional

from auto_pipeline.converter.base import BaseConverter
from auto_pipeline.converter.submission import validate_submission
from auto_pipeline.generator.pypto.converter.submission_utils import (
    copy_submission_tree,
    python_files_contain,
)
from auto_pipeline.core import Artifact, RunnerPrompt, Submission
from auto_pipeline.core import render_prompt_file
from auto_pipeline.core import CannBenchCase


class PyptoToBenchmarkConverter(BaseConverter):
    """Converts PyPTO artifacts into a benchmark submission."""

    source_generator = "pypto"
    target_benchmark: str
    conversion_template: Path

    def __init__(
        self,
        *,
        timeout_sec: int = 7200,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.timeout_sec = timeout_sec
        self.env = {str(key): str(value) for key, value in dict(env or {}).items()}

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
        workdir = Path(workdir).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        submission_path = output_dir / "submission"
        isolated_files = _prepare_conversion_input(case, artifact, output_dir)
        task_files = _render_file_list(isolated_files, "task:")
        raw_files = _render_file_list(isolated_files, "raw:")
        input_dir = output_dir / "input"
        raw_dir = input_dir / "raw"
        text = _render_conversion_prompt_text(
            template_name=self.conversion_template,
            bench_name=bench_name,
            case=case,
            input_dir=input_dir,
            raw_dir=raw_dir,
            output_dir=output_dir,
            submission_path=submission_path,
            task_files=task_files,
            raw_files=raw_files,
        )
        return RunnerPrompt(
            text=text,
            cwd=workdir,
            output_dir=output_dir,
            timeout_sec=self.timeout_sec,
            env=dict(self.env),
            title=f"convert:{self.name}:{case.operator}",
            files=isolated_files,
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
        source_dir = self._find_submission(artifact)
        target_dir = Path(output_dir).expanduser().resolve()
        if source_dir.resolve() != target_dir:
            copy_submission_tree(source_dir, target_dir)
            source_dir = target_dir
        self._validate_submission(bench_name, case, source_dir)
        metadata = self._write_metadata(case, artifact, source_dir)
        return Submission(self.target_benchmark, case.operator, source_dir, metadata)

    def _find_submission(self, output: Artifact) -> Path:
        candidates = []
        if output.files.get("source_dir"):
            candidates.append(Path(output.files["source_dir"]))
        candidates.append(output.workdir / "submission")
        for candidate in candidates:
            path = candidate.expanduser().resolve()
            if path.is_dir():
                return path
        display = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"pypto agent output must include submission source_dir; tried: {display}")

    def _validate_submission(self, bench_name: str, case: CannBenchCase, source_dir: Path) -> None:
        if bench_name != self.target_benchmark:
            raise ValueError(f"{self.name} targets benchmark {self.target_benchmark}, got {bench_name}")
        validate_submission(self.target_benchmark, case, source_dir, label="pypto")
        if not python_files_contain(source_dir, {"pypto"}):
            raise ValueError("pypto submission must contain PyPTO implementation code")

    def _write_metadata(self, case: CannBenchCase, output: Artifact, source_dir: Path) -> dict:
        metadata = {
            "converter": self.name,
            "source_generator": self.source_generator,
            "target_benchmark": self.target_benchmark,
            "operator": case.operator,
            "artifact_status": output.status,
            "artifact_message": output.message,
            "source": str(source_dir),
        }
        (source_dir / "benchmark_submission.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadata


def _render_conversion_prompt_text(
    *,
    template_name: str,
    bench_name: str,
    case: CannBenchCase,
    input_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    submission_path: Path,
    task_files: str,
    raw_files: str,
) -> str:
    return render_prompt_file(
        template_name,
        operator=case.operator,
        bench_name=bench_name,
        rel_path=case.rel_path,
        input_dir=input_dir,
        raw_dir=raw_dir,
        output_dir=output_dir,
        submission_path=submission_path,
        task_files=task_files,
        raw_files=raw_files,
    )


def _prepare_conversion_input(case: CannBenchCase, output: Artifact, output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    input_dir = output_dir / "input"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    task_dir = input_dir / "task"
    raw_dir = input_dir / "raw"
    task_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    for key, source in _iter_conversion_task_files(case.files):
        target = _copy_unique_file(source, task_dir)
        files[f"task:{key}"] = target

    for source, source_root in _iter_raw_impl_files(output):
        target = _copy_raw_impl_file(source, raw_dir, source_root=source_root)
        files[f"raw:{target.relative_to(raw_dir).as_posix()}"] = target

    files["input_dir"] = input_dir
    files["raw_dir"] = raw_dir
    files["submission_dir"] = output_dir / "submission"
    return files


def _iter_conversion_task_files(case_files: Mapping[str, Path]) -> Iterable[tuple[str, Path]]:
    skip_tokens = {"golden", "test", "report", "readme", "doc"}
    for key, raw_path in sorted(case_files.items()):
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            continue
        lower_key = key.lower()
        lower_name = path.name.lower()
        if any(token in lower_key or token in lower_name for token in skip_tokens):
            continue
        yield key, path


def _iter_raw_impl_files(output: Artifact) -> list[tuple[Path, Optional[Path]]]:
    candidates: list[tuple[Path, Optional[Path]]] = []
    source_dir = output.files.get("source_dir")
    source_roots: list[Path] = []
    if source_dir is not None:
        source_root = Path(source_dir).expanduser().resolve()
        source_roots.append(source_root)
        candidates.extend((path, source_root) for path in _impl_files(source_root))
    workdir = Path(output.workdir).expanduser().resolve()
    source_roots.append(workdir)

    for key, raw_path in sorted(output.files.items()):
        if key == "source_dir":
            continue
        path = Path(raw_path).expanduser().resolve()
        if path.is_file() and _is_standard_impl_file(path):
            candidates.append((path, _infer_source_root(path, source_roots)))

    candidates.extend((path, workdir) for path in _impl_files(workdir))

    seen: set[Path] = set()
    unique: list[tuple[Path, Optional[Path]]] = []
    for path, source_root in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append((resolved, source_root))
    return unique


def _impl_files(path: Path) -> list[Path]:
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        return []
    return sorted(candidate for candidate in directory.rglob("*_impl.py") if _is_standard_impl_file(candidate))


def _is_standard_impl_file(path: Path) -> bool:
    name = path.name
    return name.endswith("_impl.py") and not name.endswith("_pypto_impl.py")


def _infer_source_root(path: Path, source_roots: Iterable[Path]) -> Optional[Path]:
    for source_root in source_roots:
        try:
            path.relative_to(source_root)
        except ValueError:
            continue
        return source_root
    return None


def _copy_unique_file(source: Path, target_dir: Path) -> Path:
    target = target_dir / source.name
    if not target.exists():
        shutil.copy2(source, target)
        return target

    stem = source.stem
    suffix = source.suffix
    index = 2
    while True:
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            shutil.copy2(source, candidate)
            return candidate
        index += 1


def _copy_raw_impl_file(source: Path, raw_dir: Path, *, source_root: Optional[Path]) -> Path:
    if source_root is not None:
        try:
            relative = source.relative_to(source_root)
        except ValueError:
            relative = Path(source.name)
    else:
        relative = Path(source.name)
    target = raw_dir / relative
    if target.exists():
        return _copy_unique_file(source, target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def _render_file_list(files: Mapping[str, Path], prefix: str) -> str:
    lines = [f"- {key.removeprefix(prefix)}: {path}" for key, path in files.items() if key.startswith(prefix)]
    return "\n".join(lines) if lines else "- <none>"
