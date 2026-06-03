"""Prompt material builders for supported benchmark case formats."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Iterable, Optional

from auto_pipeline.prompt.base import CaseFile, CaseMaterial
from auto_pipeline.core import CannBenchCase


class CannPromptBuilder:
    name = "cann"

    def case_op_name(self, case: CannBenchCase) -> str:
        return _sanitize_op_name(str(case.task_dir.name or case.operator))

    def build_case_material(self, case: CannBenchCase) -> CaseMaterial:
        required = ("proto", "cases", "golden")
        paths: dict[str, Path] = {}
        for key in required:
            raw_path = case.files.get(key)
            if raw_path is None:
                raise FileNotFoundError(f"cann task missing required file: {key}")
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"cann task file not found: {path}")
            paths[key] = path

        desc_path: Optional[Path] = None
        raw_desc = case.files.get("desc")
        if raw_desc is not None:
            candidate = Path(raw_desc).expanduser().resolve()
            if candidate.is_file():
                desc_path = candidate

        schema = str(case.metadata.get("schema") or "")
        op_name = _case_op_name_from_schema(schema, fallback=self.case_op_name(case))
        task_files = [
            CaseFile("proto", paths["proto"], "proto.yaml"),
            CaseFile("cases", paths["cases"], paths["cases"].name),
            CaseFile("golden", paths["golden"], "golden.py"),
        ]
        if desc_path is not None:
            task_files.append(CaseFile("desc", desc_path, "desc.md"))

        case_preview_json = json.dumps(case.metadata.get("case_preview") or [], ensure_ascii=False, indent=2)
        proto_source = _read_text_for_prompt(paths["proto"])
        golden_source = _read_text_for_prompt(paths["golden"])
        desc_source = _read_text_for_prompt(desc_path) if desc_path else ""
        return CaseMaterial(
            bench_name=case.bench_name,
            op_name=op_name,
            task_files=tuple(task_files),
            require_path=_find_require_path(paths["proto"]),
            require_text=_render_cann_require(
                op_name=op_name,
                schema=schema,
                case_preview_json=case_preview_json,
                proto_source=proto_source,
                golden_source=golden_source,
                desc_source=desc_source,
            ),
            prompt_context={
                "case_task_files_text": "\n".join(
                    [
                        f"- cann-bench proto: `{_target_rel(task_files[0])}`",
                        f"- cann-bench cases: `{_target_rel(task_files[1])}`",
                        f"- cann-bench golden: `{_target_rel(task_files[2])}`",
                        *([f"- cann-bench desc: `{_target_rel(task_files[3])}`"] if len(task_files) > 3 else []),
                    ]
                ),
                "case_detail_sections": _join_sections(
                    [
                        (
                            "本 case 的 cann-bench 关键信息",
                            "\n".join(
                                [
                                    f"- proto 文件: `{_target_rel(task_files[0])}`",
                                    f"- cases 文件: `{_target_rel(task_files[1])}`",
                                    f"- golden 文件: `{_target_rel(task_files[2])}`",
                                    *([f"- desc 文件: `{_target_rel(task_files[3])}`"] if len(task_files) > 3 else []),
                                    f"- schema: {schema}",
                                    "",
                                    "用例预览（来自 cases，仅用于理解任务输入材料）：",
                                    case_preview_json,
                                ]
                            ),
                        ),
                        ("proto.yaml 内容", proto_source),
                        ("golden.py 参考源码", golden_source),
                        *([("desc.md 内容", desc_source)] if desc_source else []),
                    ]
                ),
            },
        )


class StanfordPromptBuilder:
    name = "stanford"

    def case_op_name(self, case: CannBenchCase) -> str:
        task_path = case.files.get("task")
        if task_path is not None:
            task_path = Path(task_path)
            if task_path.name == "task_desc.py" and task_path.parent.name:
                return _sanitize_op_name(task_path.parent.name)
            return _sanitize_op_name(_strip_numeric_prefix(task_path.stem))
        return _sanitize_op_name(str(case.operator or case.task_dir.name or "op"))

    def build_case_material(self, case: CannBenchCase) -> CaseMaterial:
        task_path = case.files.get("task")
        if task_path is None:
            raise FileNotFoundError("stanford task missing required file: task")
        task_path = Path(task_path).expanduser().resolve()
        if not task_path.is_file():
            raise FileNotFoundError(f"stanford task file not found: {task_path}")

        source = task_path.read_text(encoding="utf-8")
        op_name = _derive_op_name_from_task(task_path, str(case.operator or ""))
        tree = ast.parse(source)
        task_file = CaseFile("task", task_path, "task_desc.py")
        init_args_repr = _extract_init_args_repr(tree)
        init_source = _extract_model_method_source(tree, source, "__init__")
        forward_source = _extract_model_method_source(tree, source, "__call__", "forward")
        return CaseMaterial(
            bench_name=case.bench_name,
            op_name=op_name,
            task_files=(task_file,),
            require_path=_find_require_path(task_path),
            require_text=_render_stanford_require(
                op_name=op_name,
                init_args_repr=init_args_repr,
                init_source=init_source,
                forward_source=forward_source,
                task_source=source.strip(),
            ),
            prompt_context={
                "case_task_files_text": f"- KernelBench task_desc: `{_target_rel(task_file)}`",
                "case_detail_sections": _join_sections(
                    [
                        (
                            "本 case 的 task_desc 关键信息",
                            "\n".join(
                                [
                                    f"- task_desc 文件: `{_target_rel(task_file)}`",
                                    f"- get_init_inputs() 探针 repr: `{init_args_repr}`",
                                ]
                            ),
                        ),
                        ("Model.__init__ 参考源码", init_source),
                        ("Model.forward / __call__ 参考源码", forward_source),
                    ]
                ),
            },
        )


def _target_rel(task_file: CaseFile) -> str:
    return f"{{op_dir_rel}}/{task_file.target_name}"


def case_material_prompt_context(material: CaseMaterial, op_dir_rel: str) -> dict:
    rendered = {}
    for key, value in material.prompt_context.items():
        rendered[key] = str(value).replace("{op_dir_rel}", op_dir_rel)
    return rendered


def _join_sections(sections: Iterable[tuple[str, str]]) -> str:
    chunks = []
    for title, body in sections:
        if not body:
            continue
        chunks.append(f"{title}:\n----------------------------------------------------------------------\n{body}\n----------------------------------------------------------------------")
    return "\n\n".join(chunks)


def _strip_numeric_prefix(value: str) -> str:
    return re.sub(r"^\d+_", "", value)


def _derive_op_name_from_task(task_path: Path, fallback: str) -> str:
    if task_path.name == "task_desc.py" and task_path.parent.name:
        return _sanitize_op_name(task_path.parent.name)
    stem = re.sub(r"^\d+_", "", task_path.stem)
    return _sanitize_op_name(stem or fallback or "op")


def _sanitize_op_name(value: str) -> str:
    name = re.sub(r"\W+", "_", str(value).strip()).strip("_")
    if not name:
        name = "op"
    if name[0].isdigit():
        name = f"op_{name}"
    return name


def _case_op_name_from_schema(schema: str, *, fallback: str) -> str:
    match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", schema or "")
    if match:
        return _sanitize_op_name(match.group(1))
    return _sanitize_op_name(fallback)


def _find_require_path(task_path: Path) -> Optional[Path]:
    for name in ("REQUIRE.md", "require.md"):
        candidate = task_path.parent / name
        if candidate.is_file():
            return candidate
    return None


def _read_text_for_prompt(path: Path, *, limit: int = 20000) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n# ... truncated ..."


def _extract_model_method_source(tree: ast.Module, source: str, *method_names: str) -> str:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in method_names:
                    return ast.get_source_segment(source, item) or ""
    return "# (method source unavailable)"


def _extract_init_args_repr(tree: ast.Module) -> str:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_init_inputs":
            for item in node.body:
                if isinstance(item, ast.Return):
                    try:
                        return repr(ast.literal_eval(item.value))
                    except (ValueError, TypeError):
                        return "<dynamic get_init_inputs()>"
    return "[]"


def _render_cann_require(
    *,
    op_name: str,
    schema: str,
    case_preview_json: str,
    proto_source: str,
    golden_source: str,
    desc_source: str,
) -> str:
    schema_json = json.dumps(schema, ensure_ascii=False)
    desc_section = ""
    if desc_source:
        desc_section = f"""
## desc.md 补充说明

```markdown
{desc_source}
```
"""
    return f"""\
---
schema_version: 1
op_name: {op_name}
schema: {schema_json}
---

# {op_name} 算子需求规格

本 REQUIRE 由 auto_pipeline 根据 cann-bench 输入材料自动生成，用作
pypto-op-orchestrator Stage 1 的用户需求输入。

## proto.yaml

```yaml
{proto_source}
```

## cases 预览

```json
{case_preview_json}
```

## golden.py 参考实现

```python
{golden_source}
```
{desc_section}
"""


def _render_stanford_require(
    *,
    op_name: str,
    init_args_repr: str,
    init_source: str,
    forward_source: str,
    task_source: str,
) -> str:
    return f"""\
---
schema_version: 1
op_name: {op_name}
supported_dtypes: ["float32"]
p0_shapes: []
tolerance: {{"rtol": 0.001, "atol": 0.001}}
---

# {op_name} 算子需求规格

本 REQUIRE 由 auto_pipeline 根据 KernelBench task_desc.py 自动生成，用作
pypto-op-orchestrator Stage 1 的用户需求输入。

## 初始化参数

```python
init_args = {init_args_repr}
```

## Model.__init__ 参考实现

```python
{init_source}
```

## Model.forward 参考实现

```python
{forward_source}
```

## 原始 KernelBench 任务描述

```python
{task_source}
```
"""
