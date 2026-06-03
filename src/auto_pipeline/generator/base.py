"""Low-level runner contract."""

from __future__ import annotations

from typing import Protocol

from auto_pipeline.core import Artifact, RunnerPrompt


class Runner(Protocol):
    """Prompt runner used inside generators or converters."""

    type: str

    def run(self, prompt: RunnerPrompt) -> Artifact:
        ...
