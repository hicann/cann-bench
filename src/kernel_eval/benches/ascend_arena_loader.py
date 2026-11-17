#!/usr/bin/python3
# coding=utf-8
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------



"""Load the native Ascend Arena benchmark layout.

Arena assets intentionally remain in their native layout.  This module only
adapts the metadata to kernel-eval's registry interfaces used by ``list`` and
``info``; execution is owned by the Arena BenchmarkRunner.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from ..base.enums import DifficultyLevel
from ..base.loaders import CaseLoader, GoldenLoaderBase, TaskLoader
from ..base.models import AttrSpec
from ..config import get_project_root
from .cann_spec import CannCaseSpec, CannInputSpec, CannOutputSpec, CannTaskSpec


class AscendArenaCaseSpec(CannCaseSpec):
    """Arena case IDs are opaque strings supplied by the native JSON asset."""

    def get_case_id_str(self) -> str:
        return self.case_id


class AscendArenaTaskLoader(TaskLoader):
    """Discover Arena operator directories and parse ``cases/proto.yaml``."""

    REQUIRED_FILES = ("api_desc.md", "reference.py", "cases/proto.yaml")

    def __init__(self, bench_root: str | None = None, tasks_root: str | None = None):
        root = tasks_root or bench_root
        self.bench_root = Path(root or get_project_root() / "bench_lab" / "ascend_arena_bench").resolve()
        self._cache: Dict[str, CannTaskSpec] = {}

    def _is_operator_dir(self, path: Path) -> bool:
        return path.is_dir() and all((path / item).exists() for item in self.REQUIRED_FILES)

    def _find_operator_dirs(self) -> List[Path]:
        if self._is_operator_dir(self.bench_root):
            return [self.bench_root]
        return sorted(
            op_dir
            for path in self.bench_root.rglob("proto.yaml")
            for op_dir in (path.parent.parent if path.parent.name == "cases" else path.parent,)
            if self._is_operator_dir(op_dir)
        )

    def _parse(self, op_dir: Path) -> CannTaskSpec:
        rel_path = "." if op_dir == self.bench_root else op_dir.relative_to(self.bench_root).as_posix()
        payload = yaml.safe_load((op_dir / "cases/proto.yaml").read_text(encoding="utf-8")) or {}
        operator = payload.get("operator") or {}
        name = str(operator.get("name") or op_dir.name)

        def difficulty(value: Any) -> DifficultyLevel:
            text = str(value or "basic").upper()
            return getattr(DifficultyLevel, text, DifficultyLevel.L1)

        attrs = [
            AttrSpec(
                name=str(item.get("name", "")),
                type=str(item.get("type", "")),
                default=item.get("default"),
                description=str(item.get("description", "")),
            )
            for item in operator.get("attrs", []) or []
        ]
        inputs = []
        for item in operator.get("inputs", []) or []:
            dtypes = item.get("dtype", [])
            inputs.append(CannInputSpec(name=str(item.get("name", "")), dtypes=[dtypes] if isinstance(dtypes, str) else list(dtypes or []), shape=None))
        outputs = []
        for item in operator.get("outputs", []) or []:
            dtypes = item.get("dtype", [])
            outputs.append(CannOutputSpec(name=str(item.get("name", "")), dtypes=[dtypes] if isinstance(dtypes, str) else list(dtypes or []), shape=None, compare=item.get("compare", True)))
        return CannTaskSpec(
            task_id=rel_path,
            name=name,
            rel_path=rel_path,
            dir_name=op_dir.name,
            difficulty=difficulty(operator.get("difficulty")),
            description=(op_dir / "api_desc.md").read_text(encoding="utf-8"),
            category=str(operator.get("category", "")),
            formula=str(operator.get("formula", "")),
            schema=str(operator.get("schema", "")),
            attrs=attrs,
            inputs=inputs,
            outputs=outputs,
            metadata={"arena_root": str(op_dir), "proto": payload},
        )

    def list_tasks(self) -> List[CannTaskSpec]:
        tasks = []
        for op_dir in self._find_operator_dirs():
            rel_path = "." if op_dir == self.bench_root else op_dir.relative_to(self.bench_root).as_posix()
            if rel_path not in self._cache:
                self._cache[rel_path] = self._parse(op_dir)
            tasks.append(self._cache[rel_path])
        return tasks

    def get_task(self, task_id: str) -> Optional[CannTaskSpec]:
        if task_id in {"", "."} and self._is_operator_dir(self.bench_root):
            return self._parse(self.bench_root)
        candidate = self.bench_root / task_id
        if self._is_operator_dir(candidate):
            return self._parse(candidate)
        return next((task for task in self.list_tasks() if task.task_id == task_id), None)

    def get_task_by_name(self, name: str) -> Optional[CannTaskSpec]:
        return next((task for task in self.list_tasks() if task.name.lower() == name.lower()), None)

    def get_operator(self, rel_path: str) -> CannTaskSpec:
        task = self.get_task(rel_path)
        if task is None:
            raise FileNotFoundError(f"Arena task not found: {rel_path}")
        return task

    def list_operators(self) -> List[CannTaskSpec]:
        return self.list_tasks()

    def get_operator_dir(self, task_id: str) -> Path:
        return self.bench_root if task_id in {"", "."} and self._is_operator_dir(self.bench_root) else self.bench_root / task_id

    def get_statistics(self) -> Dict[str, Any]:
        tasks = self.list_tasks()
        return {"total": len(tasks), "operators": [task.name for task in tasks]}


class AscendArenaCaseLoader(CaseLoader):
    """Normalize Arena JSON cases into ``CannCaseSpec`` without rewriting assets."""

    CASE_FILES = ("smoke_cases.json", "complete_cases.json", "profile_cases.json")

    def __init__(self, bench_root: str | None = None, tasks_root: str | None = None):
        root = tasks_root or bench_root
        self.bench_root = Path(root or get_project_root() / "bench_lab" / "ascend_arena_bench").resolve()

    def _operator_dirs(self) -> List[Path]:
        return AscendArenaTaskLoader(str(self.bench_root))._find_operator_dirs()

    @staticmethod
    def _parse_case(raw: Dict[str, Any], op_dir: Path, source: Path, index: int) -> AscendArenaCaseSpec:
        case_id = str(raw.get("case_id") or f"{op_dir.name}_{index + 1}")
        inputs = raw.get("inputs") or {}
        input_shapes: List[List[int]] = []
        dtypes: List[str] = []
        value_ranges: List[Dict[str, float]] = []
        for value in inputs.values() if isinstance(inputs, dict) else []:
            if not isinstance(value, dict):
                continue
            shape = value.get("shape", [])
            input_shapes.append(list(shape) if isinstance(shape, list) else [])
            dtypes.append(str(value.get("dtype", "")))
            value_ranges.append({key: value[key] for key in ("min", "max") if key in value})
        return AscendArenaCaseSpec(
            case_id=case_id,
            operator=op_dir.name,
            rel_path=op_dir.name,
            case_num=index + 1,
            input_shapes=input_shapes,
            dtypes=dtypes,
            value_ranges=value_ranges,
            attrs=raw.get("attrs", {}) or {},
            tolerance={"rtol": 1e-3, "atol": 1e-3},
            yaml_path=str(source),
            metadata={"arena_case": raw, "stage": source.stem},
        )

    def _load_operator(self, op_dir: Path) -> List[CannCaseSpec]:
        cases: List[CannCaseSpec] = []
        for filename in self.CASE_FILES:
            source = op_dir / "cases" / filename
            if not source.exists():
                continue
            payload = __import__("json").loads(source.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload = payload.get("cases", payload.get("test_cases", []))
            for index, raw in enumerate(payload if isinstance(payload, list) else []):
                cases.append(self._parse_case(raw, op_dir, source, index))
        return cases

    def scan_all(self) -> List[CannCaseSpec]:
        if AscendArenaTaskLoader(str(self.bench_root))._is_operator_dir(self.bench_root):
            return self._load_operator(self.bench_root)
        cases: List[CannCaseSpec] = []
        for op_dir in self._operator_dirs():
            cases.extend(self._load_operator(op_dir))
        return cases

    def scan_by_task(self, task_name: str) -> List[CannCaseSpec]:
        return [case for case in self.scan_all() if case.operator.lower() == task_name.lower()]

    def get_statistics(self) -> Dict[str, Any]:
        cases = self.scan_all()
        return {"total": len(cases), "operators": {name: sum(case.operator == name for case in cases) for name in sorted({case.operator for case in cases})}}

    scan_all_cases = scan_all
    scan_by_operator = scan_by_task

    def scan_by_rel_path(self, rel_path: str) -> List[CannCaseSpec]:
        return [case for case in self.scan_all() if case.rel_path == rel_path]


class AscendArenaGoldenLoader(GoldenLoaderBase):
    """Load ``Model.forward`` from an Arena ``reference.py``."""

    def __init__(self, bench_root: str | None = None, tasks_root: str | None = None):
        root = tasks_root or bench_root
        self.bench_root = Path(root or get_project_root() / "bench_lab" / "ascend_arena_bench").resolve()
        self._modules: Dict[str, Any] = {}

    def _module(self, task_id: str):
        op_dir = self.bench_root if task_id in {"", "."} and (self.bench_root / "reference.py").exists() else self.bench_root / task_id
        path = op_dir / "reference.py"
        key = str(path)
        if key not in self._modules:
            spec = importlib.util.spec_from_file_location(f"ascend_arena_reference_{op_dir.name}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load Arena reference: {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._modules[key] = module
        return self._modules[key]

    def get_golden_function(self, task_id: str) -> Callable:
        module = self._module(task_id)
        model = module.Model()
        return model.forward

    def get_input_function(self, task_id: str) -> Optional[Callable]:
        return None

    def get_output_function(self, task_id: str) -> Optional[Callable]:
        return None

    def get_operator_dir(self, task_id: str) -> str:
        return str(self.bench_root if task_id in {"", "."} else self.bench_root / task_id)
