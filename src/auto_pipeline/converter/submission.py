"""Submission format checks owned by converters."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Iterable

from auto_pipeline.core import CannBenchCase


def prepare_submission(target_benchmark: str, case: CannBenchCase, source_dir: Path) -> None:
    if _normalize_name(target_benchmark) == "stanford":
        _prepare_stanford_submission(case, source_dir)


def is_submission_dir(target_benchmark: str, source_dir: Path) -> bool:
    key = _normalize_name(target_benchmark)
    if key == "stanford":
        return _is_stanford_source_dir(source_dir)
    return _is_cannbench_source_dir(source_dir)


def validate_submission(target_benchmark: str, case: CannBenchCase, source_dir: Path, *, label: str) -> None:
    prepare_submission(target_benchmark, case, source_dir)
    if not is_submission_dir(target_benchmark, source_dir):
        raise ValueError(f"{label} submission is not valid for benchmark {target_benchmark}")


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _is_cannbench_source_dir(path: Path) -> bool:
    has_build = (path / "build.sh").is_file()
    has_package = (path / "cann_bench").is_dir()
    has_dist = any((path / "dist").glob("cann_bench*.whl")) if (path / "dist").is_dir() else False
    return has_build and (has_package or has_dist)


def _is_stanford_source_dir(path: Path) -> bool:
    return (path / "ai_op.py").is_file()


def _prepare_stanford_submission(case: CannBenchCase, source_dir: Path) -> None:
    ai_op_path = source_dir / "ai_op.py"
    if not ai_op_path.is_file():
        raise FileNotFoundError(
            "Stanford submission must include standard ai_op.py; "
            "run a separate converter agent for raw outputs"
        )
    _ensure_stanford_ai_op_prelude(ai_op_path)
    _validate_stanford_model_contract(case, source_dir)


def _validate_no_relative_imports(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            module = "." * node.level + (node.module or "")
            raise ValueError(
                "Stanford ai_op.py must use flat/local absolute imports, "
                f"not relative import {module!r}"
            )


def _validate_stanford_model_contract(case: CannBenchCase, source_dir: Path) -> None:
    task_path = case.files.get("task")
    if task_path is None or not Path(task_path).is_file():
        return

    task_module = _load_module_from_path("stanford_task_contract", Path(task_path))
    ai_module = _load_module_from_path("stanford_ai_contract", source_dir / "ai_op.py", prepend_paths=[source_dir])
    task_model = getattr(task_module, "Model", None)
    ai_model = getattr(ai_module, "ModelNew", None)
    if task_model is None or ai_model is None:
        raise ValueError("Stanford submission must define ModelNew and task must define Model")

    _compare_method_signature(task_model, ai_model, "__init__")
    _compare_method_signature(task_model, ai_model, "forward")
    _compare_state_dict_contract(task_module, task_model, ai_model)


def _load_module_from_path(name: str, path: Path, *, prepend_paths: Iterable[Path] = ()):
    old_path = list(sys.path)
    for entry in reversed([str(Path(item)) for item in prepend_paths]):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    try:
        module_name = f"{name}_{abs(hash(str(path.resolve())))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"failed to load module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path


def _compare_method_signature(task_model: object, ai_model: object, method_name: str) -> None:
    task_signature = _normalized_signature(getattr(task_model, method_name))
    ai_signature = _normalized_signature(getattr(ai_model, method_name))
    if task_signature != ai_signature:
        raise ValueError(
            f"Stanford ModelNew.{method_name} signature must match task Model.{method_name}; "
            f"expected={task_signature}, actual={ai_signature}"
        )


def _normalized_signature(method: object) -> list[tuple[str, str, bool, str]]:
    signature = inspect.signature(method)
    normalized = []
    for param in signature.parameters.values():
        has_default = param.default is not inspect._empty
        default = repr(param.default) if has_default else ""
        normalized.append((param.name, str(param.kind), has_default, default))
    return normalized


def _compare_state_dict_contract(task_module: object, task_model_cls: object, ai_model_cls: object) -> None:
    get_init_inputs = getattr(task_module, "get_init_inputs", None)
    init_inputs = list(get_init_inputs() if callable(get_init_inputs) else [])
    task_model = task_model_cls(*init_inputs)
    ai_model = ai_model_cls(*init_inputs)
    task_state = task_model.state_dict()
    ai_state = ai_model.state_dict()
    task_keys = list(task_state.keys())
    ai_keys = list(ai_state.keys())
    if task_keys != ai_keys:
        raise ValueError(
            "Stanford ModelNew state_dict keys must match task Model; "
            f"expected={task_keys}, actual={ai_keys}"
        )
    for key in task_keys:
        task_tensor = task_state[key]
        ai_tensor = ai_state[key]
        if tuple(task_tensor.shape) != tuple(ai_tensor.shape):
            raise ValueError(
                "Stanford ModelNew state_dict tensor shapes must match task Model; "
                f"key={key}, expected={tuple(task_tensor.shape)}, actual={tuple(ai_tensor.shape)}"
            )
        if task_tensor.dtype != ai_tensor.dtype:
            raise ValueError(
                "Stanford ModelNew state_dict tensor dtypes must match task Model; "
                f"key={key}, expected={task_tensor.dtype}, actual={ai_tensor.dtype}"
            )


def _render_stanford_ai_op(source: str) -> str:
    prelude = (
        "from pathlib import Path as _Path\n"
        "import sys as _sys\n"
        "_op_dir = str(_Path(__file__).resolve().parent)\n"
        "if _op_dir not in _sys.path:\n"
        "    _sys.path.insert(0, _op_dir)\n"
        "del _Path, _sys, _op_dir\n"
        "\n"
    )
    return prelude + source


def _ensure_stanford_ai_op_prelude(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if "_sys.path.insert(0, _op_dir)" in source:
        _validate_no_relative_imports(path)
        return
    path.write_text(_render_stanford_ai_op(source), encoding="utf-8")
    _validate_no_relative_imports(path)
