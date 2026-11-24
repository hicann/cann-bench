"""Versioned task, suite, submission and result contracts."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re

import yaml


SCHEMA_VERSION = 1
IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9_.-]*\Z")
OBJECTIVES = {"implement", "optimize", "diagnose", "repair"}
SCOPES = {"operator", "subgraph", "model"}
SUBMISSIONS = {"source", "wheel", "patch", "report"}
TASK_CLASSES = {"published", "archived", "experimental", "example", "draft"}


class ContractError(ValueError):
    """An invalid or unsupported catalog declaration."""


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ContractError(f"Invalid identifier: {value!r}")
    return value


def within(root: Path, value: str, *, required=True) -> Path:
    if not isinstance(value, str) or "\\" in value or ":" in value:
        raise ContractError(f"Invalid repository-relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ContractError(f"Invalid repository-relative path: {value!r}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ContractError(f"Path escapes repository: {value!r}")
    if required and not resolved.exists():
        raise ContractError(f"Missing repository asset: {value}")
    return resolved


def read_document(path):
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise ContractError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
        raise ContractError(f"{path}: expected a versioned mapping")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"{path}: unsupported schema version {data['schema_version']}")
    return data


@dataclass(frozen=True)
class TaskSpec:
    path: Path
    data: dict

    @property
    def id(self):
        return self.data["id"]

    @property
    def version(self):
        return self.data["version"]

    @property
    def objective(self):
        return self.data["objective"]

    @property
    def adapter(self):
        return self.data["evaluation"]["adapter"]


@dataclass(frozen=True)
class Submission:
    kind: str
    path: Path | None

    def __post_init__(self):
        if self.kind not in SUBMISSIONS:
            raise ContractError(f"Unsupported submission kind: {self.kind}")
        if self.path is None or not self.path.exists():
            raise ContractError(f"Submission path does not exist: {self.path}")


@dataclass(frozen=True)
class RunResult:
    task_id: str
    task_version: str
    suite_id: str | None
    suite_version: str | None
    adapter: str
    exit_code: int
    reports: tuple[str, ...]

    def as_dict(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": self.task_id,
            "task_version": self.task_version,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "adapter": self.adapter,
            "execution": {"exit_code": self.exit_code, "reports": list(self.reports)},
        }
