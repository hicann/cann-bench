"""AKG agent artifact to CANN benchmark submission converter."""

from __future__ import annotations

import ast
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from auto_pipeline.converter.base import BaseConverter
from auto_pipeline.converter.submission import validate_submission
from auto_pipeline.core import Artifact, Submission
from auto_pipeline.core import CannBenchCase


@dataclass(frozen=True)
class SchemaParameter:
    name: str
    type_name: str
    default: Any = None
    has_default: bool = False


@dataclass(frozen=True)
class AttributeDefault:
    name: str
    default: Any = None
    has_default: bool = False


@dataclass(frozen=True)
class OperatorSchema:
    operator_name: str
    function_name: str
    schema: str
    parameters: tuple[SchemaParameter, ...]
    tensor_inputs: tuple[str, ...]
    attrs: tuple[tuple[str, Any], ...]
    attr_defaults: tuple[AttributeDefault, ...]


class AkgToCannConverter(BaseConverter):
    """Packages AKG generated Triton Ascend code as CANN submissions."""

    name = "akg-agent-to-cann"
    source_generator = "akg-agent"
    target_benchmark = "cann"

    def __init__(self, *, timeout_sec: int = 7200, env: Optional[Mapping[str, str]] = None) -> None:
        self.timeout_sec = timeout_sec
        self.env = {str(key): str(value) for key, value in dict(env or {}).items()}

    def build_submission(
        self,
        bench_name: str,
        case: CannBenchCase,
        artifact: Artifact,
        *,
        output_dir: Path,
    ) -> Submission:
        if bench_name != self.target_benchmark:
            raise ValueError(f"{self.name} targets benchmark {self.target_benchmark}, got {bench_name}")
        schema = load_operator_schema(case)
        code, source = _generated_code_from_output(artifact)
        if not code.strip():
            raise ValueError("generated AKG Triton Ascend code is empty")

        source_dir = _safe_submission_dir(output_dir, artifact)
        if source_dir.exists():
            shutil.rmtree(source_dir)
        package_dir = source_dir / "cann_bench"
        package_dir.mkdir(parents=True, exist_ok=True)

        function_name = schema.function_name
        impl_name = f"{function_name}_triton_ascend_impl"
        package_dir.joinpath(f"{impl_name}.py").write_text(_implementation_source(code), encoding="utf-8")
        package_dir.joinpath(f"{function_name}.py").write_text(_wrapper_source(schema), encoding="utf-8")
        package_dir.joinpath("__init__.py").write_text(_init_source(function_name), encoding="utf-8")
        source_dir.joinpath("setup.py").write_text(_setup_source(), encoding="utf-8")
        build_script = source_dir / "build.sh"
        build_script.write_text(_build_script_source(), encoding="utf-8")
        build_script.chmod(0o755)

        metadata = {
            "converter": self.name,
            "source_generator": self.source_generator,
            "target_benchmark": self.target_benchmark,
            "operator": case.operator,
            "function_name": function_name,
            "artifact_status": artifact.status,
            "artifact_message": artifact.message,
            "source": source,
        }
        source_dir.joinpath("benchmark_submission.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        validate_submission(self.target_benchmark, case, source_dir, label="akg-agent")
        return Submission(self.target_benchmark, case.operator, source_dir, metadata)


def load_operator_schema(case: CannBenchCase) -> OperatorSchema:
    proto_path = Path(case.files["proto"]).expanduser().resolve()
    proto = yaml.safe_load(proto_path.read_text(encoding="utf-8")) or {}
    op = proto.get("operator") if isinstance(proto, dict) else {}
    if not isinstance(op, dict):
        raise ValueError(f"invalid proto.yaml operator section: {proto_path}")
    operator_name = str(op.get("name") or case.operator or proto_path.parent.name)
    schema = str(op.get("schema") or case.metadata.get("schema") or "")
    function_name = _function_name_from_schema(schema) or _to_snake_case(operator_name)
    parameters = _parameters_from_schema(schema)
    tensor_inputs = tuple(_sanitize_identifier(str(item.get("name") or "")) for item in op.get("inputs", []) or [])
    tensor_inputs = tuple(name for name in tensor_inputs if name)
    attr_defaults = tuple(
        AttributeDefault(
            _sanitize_identifier(str(item.get("name") or "")),
            item.get("default"),
            "default" in item,
        )
        for item in op.get("attrs", []) or []
        if item.get("name")
    )
    attrs = tuple((item.name, item.default) for item in attr_defaults)
    if not tensor_inputs:
        raise ValueError(f"operator {operator_name} has no tensor inputs in {proto_path}")
    return OperatorSchema(
        operator_name=operator_name,
        function_name=_sanitize_identifier(function_name),
        schema=schema,
        parameters=parameters,
        tensor_inputs=tensor_inputs,
        attrs=attrs,
        attr_defaults=attr_defaults,
    )


def _function_name_from_schema(schema: str) -> str:
    match = re.match(r"^\s*([A-Za-z_]\w*)\s*\(", schema or "")
    return match.group(1) if match else ""


def _parameters_from_schema(schema: str) -> tuple[SchemaParameter, ...]:
    match = re.match(r"^\s*[A-Za-z_]\w*\s*\((.*)\)\s*->", schema or "")
    if not match:
        return ()
    parameters = []
    for token in _split_schema_parameters(match.group(1)):
        parameter = _schema_parameter_from_token(token)
        if parameter is not None:
            parameters.append(parameter)
    return tuple(parameters)


def _split_schema_parameters(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _schema_parameter_from_token(token: str) -> Optional[SchemaParameter]:
    head, sep, default = token.partition("=")
    parts = head.strip().split()
    if len(parts) < 2:
        return None
    type_name = parts[0]
    name = _sanitize_identifier(parts[-1])
    if not name:
        return None
    default_value: Any = None
    if sep:
        default_value = _parse_default_literal(default)
    return SchemaParameter(name=name, type_name=type_name, default=default_value, has_default=bool(sep))


def _to_snake_case(value: str) -> str:
    value = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", value)
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return _sanitize_identifier(value.lower())


def _sanitize_identifier(value: str) -> str:
    name = re.sub(r"\W+", "_", str(value).strip()).strip("_")
    if not name:
        return ""
    if name[0].isdigit():
        name = f"_{name}"
    return name


def _generated_code_from_output(output: Artifact) -> tuple[str, str]:
    if output.output_text.strip():
        return output.output_text, "output_text"

    for key in ("generated_code", "code", "model_code", "impl"):
        raw_path = output.files.get(key)
        if raw_path is None:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = output.workdir / path
        path = path.resolve()
        if path.is_file():
            return path.read_text(encoding="utf-8"), str(path)
    return "", ""


def _parse_default_literal(value: str) -> Any:
    text = value.strip()
    if text.lower() in {"false", "true"}:
        return text.lower() == "true"
    if text == "None" or text.lower() == "null":
        return None
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def _safe_submission_dir(output_dir: Path, output: Artifact) -> Path:
    source_dir = Path(output_dir).expanduser().resolve()
    workdir = Path(output.workdir).expanduser().resolve()
    if source_dir == workdir or _path_contains(source_dir, workdir):
        raise ValueError(f"refusing to overwrite agent workdir: {workdir}")
    if source_dir in {Path("/").resolve(), Path.home().resolve()}:
        raise ValueError(f"refusing to overwrite unsafe submission dir: {source_dir}")
    return source_dir


def _path_contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _implementation_source(code: str) -> str:
    return "\n".join(
        [
            "try:",
            "    from akg_agents.op.utils.triton_autotune_patch import apply_triton_patches",
            "    apply_triton_patches()",
            "except Exception:",
            "    pass",
            "",
            "import torch",
            "import triton",
            "import triton.language as tl",
            "",
            code.strip(),
            "",
        ]
    )


def _wrapper_source(schema: OperatorSchema) -> str:
    function_name = schema.function_name
    impl_name = f"{function_name}_triton_ascend_impl"
    parameter_names = _schema_parameter_names(schema)
    signature = ", ".join(_signature_parts(schema, parameter_names))
    call_args = ", ".join(f"{name}={name}" for name in parameter_names)
    tensor_args = ", ".join(name for name in parameter_names if name in set(schema.tensor_inputs))
    device_args = tensor_args or ", ".join(parameter_names[:1])
    if not device_args:
        device_args = "None"

    return "\n".join(
        [
            f"from .{impl_name} import ModelNew",
            "",
            "",
            "_MODEL = None",
            "",
            "",
            "def _first_tensor_device(*values):",
            "    for value in values:",
            "        device = getattr(value, \"device\", None)",
            "        if device is not None:",
            "            return device",
            "    return None",
            "",
            "",
            "def _get_model(device):",
            "    global _MODEL",
            "    if _MODEL is None:",
            "        _MODEL = ModelNew()",
            "    if device is not None and hasattr(_MODEL, \"to\"):",
            "        moved = _MODEL.to(device)",
            "        if moved is not None:",
            "            _MODEL = moved",
            "    return _MODEL",
            "",
            "",
            f"def {function_name}({signature}):",
            f"    model = _get_model(_first_tensor_device({device_args}))",
            f"    return model({call_args})",
            "",
        ]
    )


def _schema_parameter_names(schema: OperatorSchema) -> tuple[str, ...]:
    if schema.parameters:
        return _unique_names(parameter.name for parameter in schema.parameters)
    return _unique_names(
        [
            *schema.tensor_inputs,
            *(name for name, _default in schema.attrs),
        ]
    )


def _unique_names(names: Any) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        clean_name = _sanitize_identifier(str(name))
        if clean_name and clean_name not in seen:
            seen.add(clean_name)
            result.append(clean_name)
    return tuple(result)


def _signature_parts(schema: OperatorSchema, parameter_names: tuple[str, ...]) -> list[str]:
    schema_default_map = {parameter.name: parameter.default for parameter in schema.parameters if parameter.has_default}
    attr_default_map = {
        item.name: item.default
        for item in schema.attr_defaults
        if item.has_default and item.name not in schema_default_map
    }
    default_map = {**schema_default_map, **attr_default_map}
    parts = []
    seen_default = False
    inserted_kw_marker = False
    for index, name in enumerate(parameter_names):
        has_default = name in default_map
        can_emit_default = has_default and (
            name in schema_default_map or not _has_later_required_parameter(parameter_names, default_map, index)
        )
        if can_emit_default:
            parts.append(f"{name}={_default_literal(default_map[name])}")
            seen_default = True
            continue
        if seen_default and not inserted_kw_marker:
            parts.append("*")
            inserted_kw_marker = True
        parts.append(name)
    return parts


def _has_later_required_parameter(parameter_names: tuple[str, ...], default_map: Mapping[str, Any], index: int) -> bool:
    return any(name not in default_map for name in parameter_names[index + 1 :])


def _default_literal(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return repr(value)


def _init_source(function_name: str) -> str:
    return "\n".join(
        [
            f"from .{function_name} import {function_name}",
            "",
            f"__all__ = [\"{function_name}\"]",
            "",
        ]
    )


def _setup_source() -> str:
    return "\n".join(
        [
            "from setuptools import find_packages, setup",
            "",
            "",
            "setup(",
            "    name=\"cann-bench\",",
            "    version=\"0.0.0\",",
            "    packages=find_packages(),",
            ")",
            "",
        ]
    )


def _build_script_source() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "python -m pip wheel . -w dist",
            "",
        ]
    )
