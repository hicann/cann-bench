"""PyPTO artifact to CANN benchmark submission converter."""

from __future__ import annotations

from pathlib import Path

from auto_pipeline.generator.pypto.converter.base import PyptoToBenchmarkConverter


_TEMPLATE_DIR = Path(__file__).with_name("templates")


class PyptoToCannConverter(PyptoToBenchmarkConverter):
    """Converts PyPTO artifacts into CANN benchmark submissions."""

    name = "pypto-to-cann"
    target_benchmark = "cann"
    conversion_template = _TEMPLATE_DIR / "to_cann.j2"
