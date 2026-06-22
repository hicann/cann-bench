"""Classify benchmark cases by per-input (ndim, dtype) signature.

PyPTO generation runs one OpenCode orchestrator per class so each class can
specialize on its own shape rank and dtype. A single class degenerates to the
original whole-operator workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import yaml


@dataclass(frozen=True)
class CaseClass:
    """One generation unit covering all cases sharing an input signature."""

    class_id: int
    subdir: str
    signature: tuple[tuple[int, str], ...]
    cases: tuple[Mapping[str, Any], ...]

    @property
    def signature_label(self) -> str:
        return "; ".join(f"{ndim}D-{dtype}" for ndim, dtype in self.signature) or "scalar"


def classify_cases(cases_path: Path) -> List[CaseClass]:
    """Group cases into classes keyed by each input's (ndim, dtype).

    Returns a single class when cases cannot be parsed or are uniform, so the
    multi-class flow stays equivalent to the original single-operator run.
    """

    cases = _load_cases(cases_path)
    if not cases:
        return [CaseClass(class_id=1, subdir=".", signature=(), cases=())]

    order: List[tuple[tuple[int, str], ...]] = []
    grouped: dict[tuple[tuple[int, str], ...], List[Mapping[str, Any]]] = {}
    for case in cases:
        signature = _case_signature(case)
        if signature not in grouped:
            grouped[signature] = []
            order.append(signature)
        grouped[signature].append(case)

    if len(order) <= 1:
        return [CaseClass(class_id=1, subdir=".", signature=order[0] if order else (), cases=tuple(cases))]

    classes: List[CaseClass] = []
    for index, signature in enumerate(order, start=1):
        classes.append(
            CaseClass(
                class_id=index,
                subdir=f"c{index}",
                signature=signature,
                cases=tuple(grouped[signature]),
            )
        )
    return classes


def _load_cases(cases_path: Path) -> List[Mapping[str, Any]]:
    path = Path(cases_path).expanduser()
    if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = data.get("cases") if isinstance(data, dict) else None
    return [case for case in cases if isinstance(case, Mapping)] if isinstance(cases, list) else []


def _case_signature(case: Mapping[str, Any]) -> tuple[tuple[int, str], ...]:
    shapes = case.get("input_shape")
    dtypes = case.get("dtype")
    shape_list = shapes if isinstance(shapes, Sequence) and not isinstance(shapes, str) else []
    dtype_list = dtypes if isinstance(dtypes, Sequence) and not isinstance(dtypes, str) else []
    signature: List[tuple[int, str]] = []
    for index, shape in enumerate(shape_list):
        ndim = len(shape) if isinstance(shape, Sequence) and not isinstance(shape, str) else 0
        signature.append((ndim, _dtype_at(dtype_list, index)))
    return tuple(signature)


def _dtype_at(dtype_list: Sequence[Any], index: int) -> str:
    if index < len(dtype_list):
        return str(dtype_list[index])
    return str(dtype_list[0]) if dtype_list else ""


def write_class_cases(cases_path: Path, case_class: CaseClass, target: Path) -> Optional[Path]:
    """Write a class-specific cases.yaml; return None when nothing to specialize."""

    cases = _load_cases(cases_path)
    if not cases or case_class.subdir == ".":
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump({"cases": list(case_class.cases)}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return target
