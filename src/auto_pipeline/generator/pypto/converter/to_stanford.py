"""PyPTO artifact to Stanford/KernelBench submission converter."""

from __future__ import annotations

from pathlib import Path

from auto_pipeline.generator.pypto.converter.base import PyptoToBenchmarkConverter


_TEMPLATE_DIR = Path(__file__).with_name("templates")


class PyptoToStanfordConverter(PyptoToBenchmarkConverter):
    """Converts PyPTO artifacts into Stanford/KernelBench submissions."""

    name = "pypto-to-stanford"
    target_benchmark = "stanford"
    conversion_template = _TEMPLATE_DIR / "to_stanford.j2"
