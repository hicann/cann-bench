"""Helpers for validating and copying cann-bench submission trees."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


def is_cannbench_source_dir(path: Path) -> bool:
    has_build = (path / "build.sh").is_file()
    has_package = (path / "cann_bench").is_dir()
    has_dist = any((path / "dist").glob("cann_bench*.whl")) if (path / "dist").is_dir() else False
    return has_build and (has_package or has_dist)


def is_stanford_source_dir(path: Path) -> bool:
    return (path / "ai_op.py").is_file()


def copy_submission_tree(source_dir: Path, target_dir: Path) -> None:
    source_dir = Path(source_dir).expanduser().resolve()
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        target = target_dir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def python_files_contain(path: Path, tokens: Iterable[str]) -> bool:
    required = tuple(tokens)
    for py_file in Path(path).rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        if all(token in text for token in required):
            return True
    return False


def has_any_file_with_suffix(path: Path, suffixes: Iterable[str]) -> bool:
    allowed = {suffix.lower() for suffix in suffixes}
    return any(file.is_file() and file.suffix.lower() in allowed for file in Path(path).rglob("*"))
