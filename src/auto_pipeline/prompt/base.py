"""Prompt material contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional



@dataclass(frozen=True)
class CaseFile:
    key: str
    source_path: Path
    target_name: str


@dataclass(frozen=True)
class CaseMaterial:
    bench_name: str
    op_name: str
    task_files: tuple[CaseFile, ...]
    require_path: Optional[Path]
    require_text: str
    prompt_context: Mapping[str, Any] = field(default_factory=dict)
