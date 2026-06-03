"""Prompt material builder registry."""

from __future__ import annotations

from auto_pipeline.prompt.base import CaseMaterial
from auto_pipeline.prompt.builders import (
    CannPromptBuilder,
    StanfordPromptBuilder,
)
from auto_pipeline.core import CannBenchCase


_PROMPT_BUILDERS = {
    "cann": CannPromptBuilder,
    "stanford": StanfordPromptBuilder,
}


def _prompt_builder_type(bench_name: str):
    key = str(bench_name or "cann").strip().lower()
    return _PROMPT_BUILDERS.get(key, CannPromptBuilder)


def build_case_material(case: CannBenchCase) -> CaseMaterial:
    return _prompt_builder_type(case.bench_name)().build_case_material(case)
