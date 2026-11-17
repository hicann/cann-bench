# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ascend_arena_e2e.ascend_arena.runners._proc_utils import DEFAULT_BUILD_TIMEOUT_SEC, run_shell_with_timeout
from ascend_arena_e2e.ascend_arena.utils.paths import resolve_compile_repo


DEFAULT_COMPILE_VENDOR_NAME = "benchmark"
SUPPORTED_COMPILE_REPOS = {"ops-skeleton"}
BUILD_MANIFEST_FILENAME = "build_manifest.json"
BUILD_OUT_MANIFEST_FILENAME = "build_out_manifest.json"
MANIFEST_SCHEMA_VERSION = 1
# The unified skeleton supports the no-force package build used by Arena.
REPOS_SUPPORTING_NO_FORCE = {"ops-skeleton"}
GENERATED_PROJECT_REQUIRED_PATHS = ("build.sh", "CMakeLists.txt", "op_host", "op_kernel")


@dataclass
class ResolvedStandardDir:
    op_name: str
    op_root: Path
    source_root: Path
    layout_mode: str


@dataclass
class CompileRepoTarget:
    repo_root: Path
    repo_name: str
    family: str
    compile_op_name: str
    relative_op_dir: Path
    vendor_name: str
    vendor_package_name: str


@dataclass
class BuildResult:
    success: bool
    op_name: str
    standard_op_root: Path
    work_dir: Path
    build_dir: Path
    log_path: Path
    extension_path: Path | None
    custom_opp_root: Path | None
    message: str
    reload_mode: str = "full_recompile"


@dataclass
class ReloadBuildPaths:
    root: Path | None
    build_path: Path | None
    build_out_path: Path | None


def _json_dumps_stable(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _strip_unstable_manifest_fields(payload: object) -> object:
    if isinstance(payload, dict):
        return {
            key: _strip_unstable_manifest_fields(value)
            for key, value in payload.items()
            if key != "path"
        }
    if isinstance(payload, list):
        return [_strip_unstable_manifest_fields(item) for item in payload]
    return payload


def _manifest_digest(payload: dict[str, object]) -> str:
    stable_payload = _strip_unstable_manifest_fields(payload)
    return hashlib.sha256(_json_dumps_stable(stable_payload).encode("utf-8")).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path, *, ignored_names: set[str] | None = None) -> str:
    digest = hashlib.sha256()
    ignored = ignored_names or set()
    if not root.exists():
        digest.update(b"missing")
        return digest.hexdigest()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(part in ignored for part in path.relative_to(root).parts):
            continue
        digest.update(relative.encode("utf-8"))
        if path.is_symlink():
            digest.update(b"symlink")
            digest.update(os.readlink(path).encode("utf-8"))
            continue
        if path.is_dir():
            digest.update(b"dir")
            continue
        digest.update(b"file")
        digest.update(_hash_file(path).encode("utf-8"))
    return digest.hexdigest()


def _path_fingerprint(path: Path, *, ignored_names: set[str] | None = None) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.exists():
        return {"path": str(resolved), "kind": "missing"}
    if resolved.is_dir():
        return {
            "path": str(resolved),
            "kind": "dir",
            "hash": _hash_tree(resolved, ignored_names=ignored_names),
        }
    if resolved.is_symlink():
        return {
            "path": str(resolved),
            "kind": "symlink",
            "target": os.readlink(resolved),
        }
    return {
        "path": str(resolved),
        "kind": "file",
        "hash": _hash_file(resolved),
    }


def _multi_path_fingerprint(paths: list[Path], *, ignored_names: set[str] | None = None) -> dict[str, object]:
    resolved_paths = [path.resolve() for path in paths]
    digest = hashlib.sha256()
    hashes: list[str] = []
    for path in resolved_paths:
        fingerprint = _path_fingerprint(path, ignored_names=ignored_names)
        path_hash = str(fingerprint.get("hash") or fingerprint.get("target") or fingerprint.get("kind"))
        hashes.append(path_hash)
        digest.update(path_hash.encode("utf-8"))
    return {
        "kind": "multi_dir",
        "paths": [str(path) for path in resolved_paths],
        "hashes": hashes,
        "hash": digest.hexdigest(),
    }


def _read_manifest_file(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_manifest_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_standard_dir(standard_dir: Path, chip_name: str, op_name: str) -> ResolvedStandardDir:
    def _required_cases_exist(metadata_root: Path) -> bool:
        # 中文说明：
        # Build 阶段只需要确认“至少有可用于 profile/correctness 的 case 文件”，
        # 不能在这里判断 case 是否完备；完整性由 runner 阶段和只读审计工具检查。
        # 因历史资产仍使用 test_cases.json，所以这里同时接受新旧文件名。
        cases_dir = metadata_root / "cases"
        return (cases_dir / "profile_cases.json").exists() or (cases_dir / "test_cases.json").exists()

    def _required_paths(*, metadata_root: Path, source_root: Path) -> list[Path]:
        return [
            source_root / "op_host",
            source_root / "op_kernel",
            metadata_root / "binding" / "module.py",
            metadata_root / "binding" / "pybind.cpp",
            metadata_root / "cases" / "prepare_inputs.py",
            metadata_root / "reference.py",
        ]

    flat_required = _required_paths(metadata_root=standard_dir, source_root=standard_dir)
    if all(path.exists() for path in flat_required) and _required_cases_exist(standard_dir):
        return ResolvedStandardDir(
            op_name=op_name,
            op_root=standard_dir,
            source_root=standard_dir,
            layout_mode="flat",
        )

    nested_op_root = standard_dir / op_name
    nested_required = _required_paths(metadata_root=standard_dir, source_root=nested_op_root)
    if all(path.exists() for path in nested_required):
        return ResolvedStandardDir(
            op_name=op_name,
            op_root=standard_dir,
            source_root=nested_op_root,
            layout_mode="flat_with_nested_op_dir",
        )

    chip_root = standard_dir / chip_name
    chip_required = _required_paths(metadata_root=chip_root, source_root=chip_root)
    if all(path.exists() for path in chip_required):
        return ResolvedStandardDir(
            op_name=op_name,
            op_root=chip_root,
            source_root=chip_root,
            layout_mode="chip_subdir",
        )

    missing_flat = [str(path.relative_to(standard_dir)) for path in flat_required if not path.exists()]
    missing_nested = [str(path.relative_to(standard_dir)) for path in nested_required if not path.exists()]
    missing_chip = [str(path.relative_to(standard_dir)) for path in chip_required if not path.exists()]
    raise FileNotFoundError(
        "Unsupported standard_dir layout. "
        f"Checked flat layout under {standard_dir} (missing: {missing_flat}), "
        f"nested-op layout under {nested_op_root} (missing: {missing_nested}), "
        f"and chip layout under {chip_root} (missing: {missing_chip})."
    )


def _load_op_info_binding(op_root: Path) -> dict:
    path = op_root / "op_info.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    binding = payload.get("binding")
    return binding if isinstance(binding, dict) else {}


_ACLNN_SYMBOL_RE = None


def _extract_first_aclnn_symbol(text: str) -> str | None:
    global _ACLNN_SYMBOL_RE
    if _ACLNN_SYMBOL_RE is None:
        import re
        _ACLNN_SYMBOL_RE = re.compile(r"\baclnn[A-Za-z0-9_]+")
    match = _ACLNN_SYMBOL_RE.search(text)
    if match is None:
        return None
    symbol = match.group(0)
    if symbol.endswith("GetWorkspaceSize"):
        symbol = symbol[: -len("GetWorkspaceSize")]
    return symbol


def _to_acl_symbol_name(op_name: str) -> str:
    if not op_name:
        raise ValueError("op_name must not be empty")
    parts = [part for part in op_name.replace("-", "_").split("_") if part]
    if parts:
        return "".join(part[:1].upper() + part[1:] for part in parts)
    return op_name[:1].upper() + op_name[1:]


def _to_snake_case(name: str) -> str:
    normalized = name.replace("-", "_")
    chars: list[str] = []
    for index, ch in enumerate(normalized):
        if ch.isupper():
            if index > 0 and chars and chars[-1] != "_":
                chars.append("_")
            chars.append(ch.lower())
            continue
        chars.append(ch)
    return "".join(chars).strip("_")


def _find_mdef_statement(text: str, python_symbol_name: str) -> str | None:
    symbol_literal = f'"{python_symbol_name}"'
    search_from = 0
    while True:
        symbol_index = text.find(symbol_literal, search_from)
        if symbol_index < 0:
            return None
        start = text.rfind("m.def", 0, symbol_index)
        if start < 0:
            search_from = symbol_index + len(symbol_literal)
            continue
        line_start = text.rfind("\n", 0, start) + 1
        end = text.find(");", symbol_index)
        if end < 0:
            return None
        statement = text[line_start:end + 2]
        if "m.def" in statement:
            return statement
        search_from = symbol_index + len(symbol_literal)


def _add_custom_pybind_api_alias(
    *,
    pybind_text: str,
    original_custom_mdef: str | None,
    python_symbol_name: str,
) -> str:
    if original_custom_mdef is None or python_symbol_name == "custom_pybind_api":
        return pybind_text
    if '"custom_pybind_api"' in pybind_text:
        return pybind_text

    renamed_mdef = original_custom_mdef.replace("custom_pybind_api", python_symbol_name)
    if renamed_mdef not in pybind_text:
        return pybind_text

    # 中文说明：
    # 历史 benchmark 资产里有一批 module.py 仍调用 kernel_gen_ops.custom_pybind_api。
    # BuildRunner 为了统一导出名会把 pybind 的 custom_pybind_api 重命名成算子名；
    # 如果不保留这个别名，编译和 symbol check 都会成功，但 smoke 阶段会在
    # Python 侧 AttributeError。这里复用同一个 C++ 函数指针导出兼容入口。
    compat_mdef = original_custom_mdef.replace("&custom_pybind_api", f"&{python_symbol_name}")
    return pybind_text.replace(renamed_mdef, renamed_mdef + "\n" + compat_mdef, 1)


def _repo_vendor_suffix(repo_name: str) -> str:
    if repo_name.startswith("ops-"):
        return repo_name.removeprefix("ops-").replace("-", "_")
    return repo_name.replace("-", "_")


def _handle_remove_readonly(func, path, exc_info) -> None:
    current = Path(path)
    for candidate in [current, *current.parents]:
        try:
            if candidate.is_symlink():
                continue
            if candidate.exists():
                os.chmod(candidate, 0o700)
        except OSError:
            continue
    func(path)


def _ensure_tree_removable(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    for root, dirnames, filenames in os.walk(path, topdown=False):
        root_path = Path(root)
        for filename in filenames:
            try:
                candidate = root_path / filename
                if candidate.is_symlink():
                    continue
                os.chmod(candidate, 0o700)
            except OSError:
                continue
        for dirname in dirnames:
            try:
                candidate = root_path / dirname
                if candidate.is_symlink():
                    continue
                os.chmod(candidate, 0o700)
            except OSError:
                continue
    try:
        if not path.is_symlink():
            os.chmod(path, 0o700)
    except OSError:
        pass


def _remove_tree(path: Path) -> None:
    if path.exists():
        _ensure_tree_removable(path)
        try:
            shutil.rmtree(path, onerror=_handle_remove_readonly)
        except PermissionError:
            _ensure_tree_removable(path)
            shutil.rmtree(path, onerror=_handle_remove_readonly)


def _copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        _remove_path(dst)
    shutil.copytree(src, dst)


def _copy_build_tree(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        _remove_path(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("runtime"))


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    _remove_tree(path)


def _restore_template_cmakelists(*, template_dir: Path, target_dir: Path) -> None:
    template_cmake = template_dir / "CMakeLists.txt"
    target_cmake = target_dir / "CMakeLists.txt"
    if template_cmake.exists() and not target_cmake.exists():
        shutil.copy2(template_cmake, target_cmake)


def _make_compile_repo_ignore(repo_root: Path):
    root = repo_root.resolve()
    any_depth_ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    root_only_ignored = {"build", "build_out", "output"}

    def _ignore(path: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in any_depth_ignored}
        if Path(path).resolve() == root:
            ignored.update(name for name in names if name in root_only_ignored)
        return ignored

    return _ignore


def _shell_quote(value: Path | str) -> str:
    return shlex.quote(str(value))


def _is_generated_project_root(path: Path) -> bool:
    return all((path / name).exists() for name in GENERATED_PROJECT_REQUIRED_PATHS)


def _missing_generated_project_paths(path: Path) -> list[str]:
    return [name for name in GENERATED_PROJECT_REQUIRED_PATHS if not (path / name).exists()]


def _patch_compile_repo_ops_info_generation(repo_root: Path) -> bool:
    gen_ops_info_path = repo_root / "cmake" / "gen_ops_info.cmake"
    if not gen_ops_info_path.is_file():
        return False

    old = "foreach(compute_unit ${ASCEND_ALL_COMPUTE_UNIT})"
    new = "foreach(compute_unit ${ASCEND_COMPUTE_UNIT})"
    text = gen_ops_info_path.read_text(encoding="utf-8")
    if old not in text:
        return False

    gen_ops_info_path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


class BuildRunner:
    def __init__(
        self,
        dependency_root: Path,
        env_script: Path | None = None,
        compile_jobs: int = 16,
        *,
        isolated_build: bool = False,
    ) -> None:
        self.dependency_root = dependency_root
        self.env_script = env_script
        self.compile_jobs = max(1, compile_jobs)
        self.isolated_build = isolated_build

    @property
    def common_root(self) -> Path:
        nested_common = self.dependency_root / "common"
        return nested_common if nested_common.exists() else self.dependency_root

    @property
    def validation_root(self) -> Path:
        return Path(__file__).with_name("validation")

    def run(
        self,
        standard_dir: Path,
        chip_name: str,
        run_root: Path,
        op_name: str,
        *,
        compile_repo: str | None = None,
        compile_family: str | None = None,
        compile_op_name: str | None = None,
        reload_build: bool = False,
        reload_build_root: Path | None = None,
        reload_build_build_path: Path | None = None,
        reload_build_build_out_path: Path | None = None,
    ) -> BuildResult:
        try:
            resolved = resolve_standard_dir(standard_dir, chip_name, op_name)
        except FileNotFoundError as exc:
            op_run_root = run_root / op_name
            return self._failure_result(
                op_name=op_name,
                standard_op_root=standard_dir,
                op_run_root=op_run_root,
                log_label="layout_error",
                message=str(exc),
            )

        op_root = resolved.op_root
        source_root = resolved.source_root
        op_run_root = run_root / resolved.op_name
        src_dir = op_run_root / "src"
        build_dir = op_run_root / "build"
        log_path = self._resolve_compile_log_path(op_run_root)
        if op_run_root.exists():
            _remove_tree(op_run_root)

        try:
            compile_target = self._resolve_compile_repo_target(
                compile_repo="common" if self.isolated_build else compile_repo,
                compile_family=None if self.isolated_build else compile_family,
                compile_op_name=compile_op_name,
            )
        except (FileNotFoundError, ValueError) as exc:
            return self._failure_result(
                op_name=resolved.op_name,
                standard_op_root=op_root,
                op_run_root=op_run_root,
                log_label="compile_repo_error",
                message=str(exc),
            )

        try:
            if compile_target is None:
                self._prepare_common_scaffold_workspace(
                    source_root=source_root,
                    src_dir=src_dir,
                    chip_name=chip_name,
                )
            else:
                self._prepare_run_local_compile_repo_workspace(
                    src_dir=src_dir,
                    compile_target=compile_target,
                    source_root=source_root,
                )
        except (FileNotFoundError, ValueError) as exc:
            return self._failure_result(
                op_name=resolved.op_name,
                standard_op_root=op_root,
                op_run_root=op_run_root,
                log_label="workspace_error",
                message=str(exc),
            )

        reloaded_result = self._try_reload_build(
            op_root=op_root,
            op_run_root=op_run_root,
            build_dir=build_dir,
            src_dir=src_dir,
            source_root=source_root,
            op_name=resolved.op_name,
            chip_name=chip_name,
            compile_target=compile_target,
            reload_build=reload_build,
            reload_build_root=reload_build_root,
            reload_build_build_path=reload_build_build_path,
            reload_build_build_out_path=reload_build_build_out_path,
            log_path=log_path,
        )
        if reloaded_result is not None:
            return reloaded_result

        if compile_target is None:
            return self._run_common_scaffold_build(
                op_root=op_root,
                source_root=source_root,
                op_name=resolved.op_name,
                chip_name=chip_name,
                op_run_root=op_run_root,
                src_dir=src_dir,
                build_dir=build_dir,
                log_path=log_path,
            )

        return self._run_compile_repo_build(
            op_root=op_root,
            source_root=source_root,
            op_name=resolved.op_name,
            chip_name=chip_name,
            op_run_root=op_run_root,
            src_dir=src_dir,
            build_dir=build_dir,
            log_path=log_path,
            compile_target=compile_target,
        )

    def _resolve_compile_repo_target(
        self,
        *,
        compile_repo: str | None,
        compile_family: str | None,
        compile_op_name: str | None,
    ) -> CompileRepoTarget | None:
        explicit_repo_name = (compile_repo or "").strip() or "common"
        explicit_family = (compile_family or "").strip() or None
        explicit_op_name = (compile_op_name or "").strip() or None

        if explicit_repo_name == "common":
            return None

        if explicit_repo_name not in SUPPORTED_COMPILE_REPOS:
            raise ValueError(f"unsupported compile repo: {explicit_repo_name}")
        if explicit_family is None:
            raise ValueError("compile family is required for non-common compile-repo mode")
        if explicit_op_name is None:
            raise ValueError("compile op name is required for non-common compile-repo mode")

        repo_name = explicit_repo_name
        family = explicit_family
        resolved_compile_op_name = explicit_op_name
        relative_op_dir = Path(resolved_compile_op_name)
        repo_root = resolve_compile_repo(self.dependency_root, repo_name).resolve()
        if not repo_root.is_dir():
            raise FileNotFoundError(f"compile repository not found under dependency root: {repo_root}")

        # The source family remains package metadata when the benchmark source
        # is injected under the skeleton root.
        vendor_suffix = family if repo_name == "ops-skeleton" else _repo_vendor_suffix(repo_name)
        vendor_name = DEFAULT_COMPILE_VENDOR_NAME
        return CompileRepoTarget(
            repo_root=repo_root,
            repo_name=repo_name,
            family=family,
            compile_op_name=resolved_compile_op_name,
            relative_op_dir=relative_op_dir,
            vendor_name=vendor_name,
            vendor_package_name=f"{vendor_name}_{vendor_suffix}",
        )

    def _prepare_common_scaffold_workspace(
        self,
        *,
        source_root: Path,
        src_dir: Path,
        chip_name: str,
    ) -> None:
        if src_dir.exists() or src_dir.is_symlink():
            _remove_path(src_dir)
        if self.isolated_build:
            missing = _missing_generated_project_paths(source_root)
            if missing:
                raise FileNotFoundError(
                    "generated project missing required path(s): "
                    + ", ".join(missing)
                    + f" under {source_root}"
                )
            shutil.copytree(source_root, src_dir)
            preset_path = src_dir / "CMakePresets.json"
            if preset_path.is_file():
                self._patch_cmake_preset(preset_path, chip_name)
            return
        shutil.copytree(self.common_root, src_dir, dirs_exist_ok=True)
        _copytree_clean(source_root / "op_host", src_dir / "op_host")
        _copytree_clean(source_root / "op_kernel", src_dir / "op_kernel")
        _restore_template_cmakelists(template_dir=self.common_root / "op_host", target_dir=src_dir / "op_host")
        _restore_template_cmakelists(template_dir=self.common_root / "op_kernel", target_dir=src_dir / "op_kernel")
        self._sync_op_common(scaffold_root=src_dir, source_root=source_root, target_root=src_dir)
        self._patch_cmake_preset(src_dir / "CMakePresets.json", chip_name)

    def _build_compile_manifest(
        self,
        *,
        source_root: Path,
        chip_name: str,
        op_name: str,
        compile_target: CompileRepoTarget | None,
    ) -> dict[str, object]:
        op_host_dir = source_root / "op_host"
        op_kernel_dir = source_root / "op_kernel"
        if op_host_dir.is_dir() and op_kernel_dir.is_dir() and source_root.name != op_name:
            workspace_fingerprint = _multi_path_fingerprint([op_host_dir, op_kernel_dir])
        else:
            workspace_fingerprint = _path_fingerprint(source_root)
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact": "build_out",
            "chip_name": chip_name,
            "op_name": op_name,
            "compile_repo": compile_target.repo_name if compile_target is not None else "common",
            "compile_family": compile_target.family if compile_target is not None else "",
            "compile_op_name": compile_target.compile_op_name if compile_target is not None else op_name,
            "workspace": workspace_fingerprint,
        }

    def _build_binding_manifest(
        self,
        *,
        op_root: Path,
        op_name: str,
        acl_op_name: str,
        compile_manifest: dict[str, object],
    ) -> dict[str, object]:
        op_info_path = op_root / "op_info.json"
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "artifact": "build",
            "op_name": op_name,
            "acl_op_name": acl_op_name,
            "python_executable": str(Path(sys.executable).absolute()),
            "python_version": sys.version,
            "compile_manifest_digest": _manifest_digest(compile_manifest),
            "binding_inputs": {
                "module_py": _path_fingerprint(op_root / "binding" / "module.py"),
                "pybind_cpp": _path_fingerprint(op_root / "binding" / "pybind.cpp"),
                "op_info_json": _path_fingerprint(op_info_path) if op_info_path.exists() else {"kind": "missing"},
                "validation_cmake": _path_fingerprint(self.validation_root / "CMakeLists.txt"),
                "validation_helper": _path_fingerprint(self.validation_root / "pytorch_npu_helper.hpp"),
            },
        }

    def _load_build_manifest(self, build_path: Path) -> dict[str, object] | None:
        return _read_manifest_file(build_path / BUILD_MANIFEST_FILENAME)

    def _load_build_out_manifest(self, build_out_path: Path) -> dict[str, object] | None:
        return _read_manifest_file(build_out_path / BUILD_OUT_MANIFEST_FILENAME)

    def _write_build_manifest(self, build_dir: Path, payload: dict[str, object]) -> None:
        _write_manifest_file(build_dir / BUILD_MANIFEST_FILENAME, payload)

    def _write_build_out_manifest(self, build_out_dir: Path, payload: dict[str, object]) -> None:
        _write_manifest_file(build_out_dir / BUILD_OUT_MANIFEST_FILENAME, payload)

    def _resolve_reload_build_paths(
        self,
        *,
        reload_build_root: Path | None,
        reload_build_build_path: Path | None,
        reload_build_build_out_path: Path | None,
    ) -> ReloadBuildPaths:
        build_path = reload_build_build_path.resolve() if reload_build_build_path is not None else None
        build_out_path = reload_build_build_out_path.resolve() if reload_build_build_out_path is not None else None
        root = reload_build_root.resolve() if reload_build_root is not None else None
        if root is not None:
            if build_path is None:
                candidate = root / "build"
                if candidate.exists():
                    build_path = candidate
            if build_out_path is None:
                for candidate in (root / "build_out", root / "src" / "build_out"):
                    if candidate.exists():
                        build_out_path = candidate
                        break
        return ReloadBuildPaths(root=root, build_path=build_path, build_out_path=build_out_path)

    def _expected_vendor_dir_name(self, compile_target: CompileRepoTarget | None) -> str:
        if compile_target is None:
            return "customize"
        return compile_target.vendor_package_name

    def _extract_single_run_package(self, build_out_dir: Path) -> bool:
        run_packages = sorted(build_out_dir.glob("*.run"))
        if len(run_packages) != 1:
            return False
        proc = subprocess.run(
            ["bash", str(run_packages[0]), "--noexec", f"--extract={build_out_dir}"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return proc.returncode == 0

    def _try_reload_build(
        self,
        *,
        op_root: Path,
        op_run_root: Path,
        build_dir: Path,
        src_dir: Path,
        source_root: Path,
        op_name: str,
        chip_name: str,
        compile_target: CompileRepoTarget | None,
        reload_build: bool,
        reload_build_root: Path | None,
        reload_build_build_path: Path | None,
        reload_build_build_out_path: Path | None,
        log_path: Path,
    ) -> BuildResult | None:
        if not reload_build:
            return None
        # 中文说明：
        # reload build 只在 manifest digest 完全匹配时复用旧 build/build_out。
        # digest 会覆盖源码、binding、pybind、Python 版本和 helper 等关键输入，
        # 防止候选代码变化后错误复用旧的编译产物。
        reload_paths = self._resolve_reload_build_paths(
            reload_build_root=reload_build_root,
            reload_build_build_path=reload_build_build_path,
            reload_build_build_out_path=reload_build_build_out_path,
        )
        if reload_paths.build_out_path is None or not reload_paths.build_out_path.is_dir():
            return None
        current_compile_manifest = self._build_compile_manifest(
            source_root=source_root,
            chip_name=chip_name,
            op_name=op_name,
            compile_target=compile_target,
        )
        reused_compile_manifest = self._load_build_out_manifest(reload_paths.build_out_path)
        if reused_compile_manifest is None or _manifest_digest(reused_compile_manifest) != _manifest_digest(current_compile_manifest):
            return None
        copied_build_out = src_dir / "build_out"
        _copytree_clean(reload_paths.build_out_path, copied_build_out)
        custom_opp_root = self._find_custom_opp_root(
            build_out_dir=copied_build_out,
            expected_vendor_dir_name=self._expected_vendor_dir_name(compile_target),
        )
        kernel_name = compile_target.compile_op_name if compile_target is not None else op_name
        if (
            not self._kernel_objects_exist(custom_opp_root, expected_name=kernel_name)
            or not self._custom_opp_install_ready(custom_opp_root)
        ) and self._extract_single_run_package(copied_build_out):
            custom_opp_root = self._find_custom_opp_root(
                build_out_dir=copied_build_out,
                expected_vendor_dir_name=self._expected_vendor_dir_name(compile_target),
            )
        if not self._kernel_objects_exist(custom_opp_root, expected_name=kernel_name) or not self._custom_opp_install_ready(custom_opp_root):
            _remove_path(copied_build_out)
            return None
        acl_op_name = compile_target.compile_op_name if compile_target is not None else op_name
        current_binding_manifest = self._build_binding_manifest(
            op_root=op_root,
            op_name=op_name,
            acl_op_name=acl_op_name,
            compile_manifest=current_compile_manifest,
        )
        if reload_paths.build_path is not None and reload_paths.build_path.is_dir():
            reused_build_manifest = self._load_build_manifest(reload_paths.build_path)
            if reused_build_manifest is not None and _manifest_digest(reused_build_manifest) == _manifest_digest(current_binding_manifest):
                _copy_build_tree(reload_paths.build_path, build_dir)
                extension_path = next(build_dir.glob("kernel_gen_ops*.so"), None)
                if extension_path is not None:
                    self._stage_validation_runtime_root(build_dir=build_dir, custom_opp_root=custom_opp_root)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(
                        "\n".join(
                            [
                                "## reload_build",
                                "status=reused_full",
                                f"reload_build_root={reload_paths.root}",
                                f"reload_build_build_path={reload_paths.build_path}",
                                f"reload_build_build_out_path={reload_paths.build_out_path}",
                            ]
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    return BuildResult(
                        success=True,
                        op_name=op_name,
                        standard_op_root=op_root,
                        work_dir=src_dir,
                        build_dir=build_dir,
                        log_path=log_path,
                        extension_path=extension_path,
                        custom_opp_root=custom_opp_root,
                        message="reload_build_reused",
                        reload_mode="full_reuse",
                    )
        return self._rebuild_binding_from_reused_build_out(
            op_root=op_root,
            op_name=op_name,
            acl_op_name=acl_op_name,
            op_run_root=op_run_root,
            build_dir=build_dir,
            src_dir=src_dir,
            custom_opp_root=custom_opp_root,
            log_path=log_path,
            current_compile_manifest=current_compile_manifest,
            current_binding_manifest=current_binding_manifest,
            reload_paths=reload_paths,
        )

    def _rebuild_binding_from_reused_build_out(
        self,
        *,
        op_root: Path,
        op_name: str,
        acl_op_name: str,
        op_run_root: Path,
        build_dir: Path,
        src_dir: Path,
        custom_opp_root: Path,
        log_path: Path,
        current_compile_manifest: dict[str, object],
        current_binding_manifest: dict[str, object],
        reload_paths: ReloadBuildPaths,
    ) -> BuildResult:
        validation_dir = op_run_root / "validation"
        acl_symbol_name, aclnn_symbol_source = self._prepare_validation_sources(
            op_root=op_root,
            validation_dir=validation_dir,
            acl_op_name=acl_op_name,
            exported_api_name=op_name,
        )
        shared_lib_dir = custom_opp_root / "op_api" / "lib"
        if build_dir.exists():
            _remove_path(build_dir)
        validation_proc = self._run_shell_command(
            command_parts=[
                f"export PATH={_shell_quote(Path(sys.executable).absolute().parent)}:$PATH",
                (
                    "cmake --no-warn-unused-cli "
                    f"-S {_shell_quote(validation_dir)} "
                    f"-B {_shell_quote(build_dir)} "
                    f"-DSHARED_LIB_DIR={_shell_quote(shared_lib_dir)} "
                    f"-DPython3_EXECUTABLE={_shell_quote(sys.executable)}"
                ),
                f"cmake --build {_shell_quote(build_dir)} -j{self.compile_jobs}",
            ],
            live_log_path=log_path,
            live_log_header="## live_validation_build_reload",
        )
        extension_path = next(build_dir.glob("kernel_gen_ops*.so"), None)
        success = validation_proc.returncode == 0 and extension_path is not None
        symbol_check_ok = True
        symbol_check_detail = "skipped: build_failed_before_symbol_check"
        if success:
            symbol_check_ok, symbol_check_detail = self._verify_aclnn_symbol(
                custom_opp_root=custom_opp_root,
                aclnn_symbol=acl_symbol_name,
            )
            success = symbol_check_ok
        if success:
            self._write_build_manifest(build_dir, current_binding_manifest)
            self._write_build_out_manifest(src_dir / "build_out", current_compile_manifest)
            self._stage_validation_runtime_root(build_dir=build_dir, custom_opp_root=custom_opp_root)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "\n".join(
                [
                    "## reload_build",
                    "status=rebound",
                    f"reload_build_root={reload_paths.root}",
                    f"reload_build_build_path={reload_paths.build_path}",
                    f"reload_build_build_out_path={reload_paths.build_out_path}",
                    "## aclnn_symbol",
                    f"aclnn_symbol={acl_symbol_name}",
                    f"aclnn_symbol_source={aclnn_symbol_source}",
                    "## validation_build_reload",
                    f"validation_returncode={validation_proc.returncode}",
                    f"extension_path={extension_path}",
                    (validation_proc.stdout or "") + "\n" + (validation_proc.stderr or ""),
                    "## symbol_check",
                    symbol_check_detail,
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return BuildResult(
            success=success,
            op_name=op_name,
            standard_op_root=op_root,
            work_dir=src_dir,
            build_dir=build_dir,
            log_path=log_path,
            extension_path=extension_path,
            custom_opp_root=custom_opp_root,
            message="reload_build_rebound" if success else f"reload_build_rebound_failed:{validation_proc.returncode}",
            reload_mode="rebind_only",
        )

    def _run_common_scaffold_build(
        self,
        *,
        op_root: Path,
        source_root: Path,
        op_name: str,
        chip_name: str,
        op_run_root: Path,
        src_dir: Path,
        build_dir: Path,
        log_path: Path,
    ) -> BuildResult:
        if _is_generated_project_root(src_dir):
            return self._run_generated_project_build(
                op_root=op_root,
                source_root=source_root,
                op_name=op_name,
                chip_name=chip_name,
                op_run_root=op_run_root,
                src_dir=src_dir,
                build_dir=build_dir,
                log_path=log_path,
            )

        validation_dir = src_dir / "validation"
        validation_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(op_root / "binding" / "module.py", validation_dir / "module.py")
        shutil.copy2(op_root / "binding" / "pybind.cpp", validation_dir / "pybind.cpp")
        shutil.copy2(op_root / "cases" / "prepare_inputs.py", validation_dir / "prepare_inputs.py")
        # Copy whichever case files exist. The three-stage layout may ship up
        # to 3 files (smoke/complete/profile); the legacy layout only has
        # test_cases.json. We copy every one that is present so downstream
        # runners can load each stage.
        case_filenames = (
            "smoke_cases.json",
            "complete_cases.json",
            "profile_cases.json",
            "test_cases.json",
        )
        for filename in case_filenames:
            src = op_root / "cases" / filename
            if src.exists():
                shutil.copy2(src, validation_dir / filename)
        binding_info = _load_op_info_binding(op_root)
        declared_symbol = str(binding_info.get("aclnn_symbol") or "").strip() or None
        acl_symbol_name, aclnn_symbol_source = self._patch_pybind(
            pybind_path=validation_dir / "pybind.cpp",
            acl_op_name=op_name,
            exported_api_name=op_name,
            explicit_aclnn_symbol=declared_symbol,
        )

        proc = self._run_shell_command(
            command_parts=[
                f"export PATH={_shell_quote(Path(sys.executable).absolute().parent)}:$PATH",
                f"cd {_shell_quote(src_dir)}",
                "bash build.sh install",
            ],
            live_log_path=log_path,
            live_log_header="## live_common_scaffold_build",
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        symbol_header = (
            "## aclnn_symbol\n"
            f"aclnn_symbol={acl_symbol_name}\n"
            f"aclnn_symbol_source={aclnn_symbol_source}\n"
        )
        log_path.write_text(
            symbol_header + "\n" + (proc.stdout or "") + "\n" + (proc.stderr or ""),
            encoding="utf-8",
        )

        compile_manifest = self._build_compile_manifest(
            source_root=source_root,
            chip_name=chip_name,
            op_name=op_name,
            compile_target=None,
        )
        extension = next(build_dir.glob("kernel_gen_ops*.so"), None)
        custom_opp_root = self._find_custom_opp_root(
            build_out_dir=src_dir / "build_out",
            expected_vendor_dir_name="customize",
        )
        kernel_exists = self._kernel_objects_exist(custom_opp_root, expected_name=op_name)
        install_ready = self._custom_opp_install_ready(custom_opp_root)
        if custom_opp_root is not None and kernel_exists and install_ready:
            self._write_build_out_manifest(src_dir / "build_out", compile_manifest)
        success = proc.returncode == 0 and extension is not None and custom_opp_root is not None and kernel_exists
        if success:
            symbol_check_ok, symbol_check_detail = self._verify_aclnn_symbol(
                custom_opp_root=custom_opp_root,
                aclnn_symbol=acl_symbol_name,
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n## symbol_check\n" + symbol_check_detail + "\n")
            if not symbol_check_ok:
                success = False
            else:
                self._write_build_manifest(
                    build_dir,
                    self._build_binding_manifest(
                        op_root=op_root,
                        op_name=op_name,
                        acl_op_name=op_name,
                        compile_manifest=compile_manifest,
                    ),
                )
                self._stage_validation_runtime_root(build_dir=build_dir, custom_opp_root=custom_opp_root)
        return BuildResult(
            success=success,
            op_name=op_name,
            standard_op_root=op_root,
            work_dir=src_dir,
            build_dir=build_dir,
            log_path=log_path,
            extension_path=extension,
            custom_opp_root=custom_opp_root,
            message=f"exit_code={proc.returncode}",
        )

    def _run_generated_project_build(
        self,
        *,
        op_root: Path,
        source_root: Path,
        op_name: str,
        chip_name: str,
        op_run_root: Path,
        src_dir: Path,
        build_dir: Path,
        log_path: Path,
    ) -> BuildResult:
        log_sections = [
            "## compile_workspace",
            "\n".join(
                [
                    "compile_mode=msopgen_project",
                    f"compile_workspace_root={src_dir}",
                ]
            ),
        ]
        compile_proc = self._run_shell_command(
            command_parts=[
                f"export PATH={_shell_quote(Path(sys.executable).absolute().parent)}:$PATH",
                f"cd {_shell_quote(src_dir)}",
                "bash build.sh",
            ],
            live_log_path=log_path,
            live_log_header="## live_msopgen_project_build",
        )
        compile_output = (compile_proc.stdout or "") + "\n" + (compile_proc.stderr or "")

        custom_opp_root = self._find_custom_opp_root(
            build_out_dir=src_dir / "build_out",
            expected_vendor_dir_name="customize",
        )
        kernel_exists = self._kernel_objects_exist(custom_opp_root, expected_name=op_name)
        install_ready = self._custom_opp_install_ready(custom_opp_root)
        if (not kernel_exists or not install_ready) and self._extract_single_run_package(src_dir / "build_out"):
            custom_opp_root = self._find_custom_opp_root(
                build_out_dir=src_dir / "build_out",
                expected_vendor_dir_name="customize",
            )
            kernel_exists = self._kernel_objects_exist(custom_opp_root, expected_name=op_name)
            install_ready = self._custom_opp_install_ready(custom_opp_root)

        compile_metadata = [
            "source_mode=msopgen_project",
            f"compile_returncode={compile_proc.returncode}",
            f"custom_opp_root={custom_opp_root}",
            f"kernel_exists={kernel_exists}",
            f"install_ready={install_ready}",
        ]
        log_sections.extend(
            [
                "## msopgen_project_build",
                "\n".join(compile_metadata) + "\n" + compile_output,
            ]
        )
        compile_manifest = self._build_compile_manifest(
            source_root=source_root,
            chip_name=chip_name,
            op_name=op_name,
            compile_target=None,
        )
        if custom_opp_root is not None and kernel_exists and install_ready:
            self._write_build_out_manifest(src_dir / "build_out", compile_manifest)

        if compile_proc.returncode != 0 or custom_opp_root is None or not kernel_exists or not install_ready:
            log_sections.extend(["## validation_build", "validation_skipped\n"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_sections), encoding="utf-8")
            return BuildResult(
                success=False,
                op_name=op_name,
                standard_op_root=op_root,
                work_dir=src_dir,
                build_dir=build_dir,
                log_path=log_path,
                extension_path=None,
                custom_opp_root=custom_opp_root,
                message=f"compile_mode=msopgen_project; compile_exit_code={compile_proc.returncode}; validation_exit_code=skipped",
            )

        validation_dir = op_run_root / "validation"
        acl_symbol_name, aclnn_symbol_source = self._prepare_validation_sources(
            op_root=op_root,
            validation_dir=validation_dir,
            acl_op_name=op_name,
            exported_api_name=op_name,
        )
        shared_lib_dir = custom_opp_root / "op_api" / "lib"
        if build_dir.exists():
            _remove_path(build_dir)
        validation_proc = self._run_shell_command(
            command_parts=[
                f"export PATH={_shell_quote(Path(sys.executable).absolute().parent)}:$PATH",
                (
                    "cmake --no-warn-unused-cli "
                    f"-S {_shell_quote(validation_dir)} "
                    f"-B {_shell_quote(build_dir)} "
                    f"-DSHARED_LIB_DIR={_shell_quote(shared_lib_dir)} "
                    f"-DPython3_EXECUTABLE={_shell_quote(sys.executable)}"
                ),
                f"cmake --build {_shell_quote(build_dir)} -j{self.compile_jobs}",
            ],
            live_log_path=log_path,
            live_log_header="## live_validation_build_msopgen_project",
        )
        validation_output = (validation_proc.stdout or "") + "\n" + (validation_proc.stderr or "")
        extension_path = next(build_dir.glob("kernel_gen_ops*.so"), None)
        log_sections.extend(
            [
                "## aclnn_symbol",
                f"aclnn_symbol={acl_symbol_name}\naclnn_symbol_source={aclnn_symbol_source}",
                "## validation_build_msopgen_project",
                "\n".join(
                    [
                        f"validation_returncode={validation_proc.returncode}",
                        f"extension_path={extension_path}",
                    ]
                )
                + "\n"
                + validation_output,
            ]
        )

        success = validation_proc.returncode == 0 and extension_path is not None
        symbol_check_ok = True
        symbol_check_detail = "skipped: build_failed_before_symbol_check"
        if success:
            symbol_check_ok, symbol_check_detail = self._verify_aclnn_symbol(
                custom_opp_root=custom_opp_root,
                aclnn_symbol=acl_symbol_name,
            )
            success = symbol_check_ok
        log_sections.extend(["## symbol_check", symbol_check_detail])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_sections), encoding="utf-8")

        if success:
            self._write_build_manifest(
                build_dir,
                self._build_binding_manifest(
                    op_root=op_root,
                    op_name=op_name,
                    acl_op_name=op_name,
                    compile_manifest=compile_manifest,
                ),
            )
            self._stage_validation_runtime_root(build_dir=build_dir, custom_opp_root=custom_opp_root)
        message = (
            f"compile_mode=msopgen_project; compile_exit_code={compile_proc.returncode}; "
            f"validation_exit_code={validation_proc.returncode}; "
            f"aclnn_symbol={acl_symbol_name}; aclnn_symbol_source={aclnn_symbol_source}"
        )
        if not symbol_check_ok:
            message += "; symbol_check=failed"
        return BuildResult(
            success=success,
            op_name=op_name,
            standard_op_root=op_root,
            work_dir=src_dir,
            build_dir=build_dir,
            log_path=log_path,
            extension_path=extension_path,
            custom_opp_root=custom_opp_root,
            message=message,
        )

    def _run_compile_repo_build(
        self,
        *,
        op_root: Path,
        source_root: Path,
        op_name: str,
        chip_name: str,
        op_run_root: Path,
        src_dir: Path,
        build_dir: Path,
        log_path: Path,
        compile_target: CompileRepoTarget,
    ) -> BuildResult:
        log_sections = [
            "## compile_workspace",
            "\n".join(
                [
                    "compile_mode=direct",
                    f"compile_repo_dir={compile_target.repo_root}",
                    f"compile_workspace_root={src_dir}",
                ]
            ),
        ]
        compile_proc = self._run_run_local_compile_repo_build(
            repo_root=src_dir,
            compile_target=compile_target,
            chip_name=chip_name,
            log_path=log_path,
        )
        compile_output = (compile_proc.stdout or "") + "\n" + (compile_proc.stderr or "")

        custom_opp_root = self._find_custom_opp_root(
            build_out_dir=src_dir / "build_out",
            expected_vendor_dir_name=compile_target.vendor_package_name,
        )
        staged_custom_opp_root: Path | None = None
        kernel_exists = self._kernel_objects_exist(
            custom_opp_root,
            expected_name=compile_target.compile_op_name,
        )
        install_ready = self._custom_opp_install_ready(custom_opp_root)
        compile_metadata = [
            "source_mode=direct",
            f"compile_repo_dir={compile_target.repo_root}",
            f"compile_workspace_root={src_dir}",
            f"compile_returncode={compile_proc.returncode}",
            f"custom_opp_root={custom_opp_root}",
            f"kernel_exists={kernel_exists}",
            f"install_ready={install_ready}",
        ]
        log_sections.extend(
            [
                "## compile_repo_build_direct",
                "\n".join(compile_metadata) + "\n" + compile_output,
            ]
        )
        compile_manifest = self._build_compile_manifest(
            source_root=source_root,
            chip_name=chip_name,
            op_name=op_name,
            compile_target=compile_target,
        )
        if custom_opp_root is not None and kernel_exists and install_ready:
            self._write_build_out_manifest(src_dir / "build_out", compile_manifest)

        if compile_proc.returncode != 0 or custom_opp_root is None or not kernel_exists or not install_ready:
            log_sections.extend(["## validation_build", "validation_skipped\n"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("\n".join(log_sections), encoding="utf-8")
            return BuildResult(
                success=False,
                op_name=op_name,
                standard_op_root=op_root,
                work_dir=src_dir,
                build_dir=build_dir,
                log_path=log_path,
                extension_path=None,
                custom_opp_root=custom_opp_root,
                message=f"compile_mode=direct; compile_exit_code={compile_proc.returncode}; validation_exit_code=skipped",
            )

        staged_custom_opp_root = self._stage_run_local_custom_opp_root(
            source_custom_opp_root=custom_opp_root,
            op_run_root=op_run_root,
            vendor_dir_name=compile_target.vendor_package_name,
        )
        validation_dir = op_run_root / "validation"
        acl_symbol_name, aclnn_symbol_source = self._prepare_validation_sources(
            op_root=op_root,
            validation_dir=validation_dir,
            acl_op_name=compile_target.compile_op_name,
            exported_api_name=op_name,
        )
        log_sections.extend(
            [
                "## aclnn_symbol",
                f"aclnn_symbol={acl_symbol_name}\naclnn_symbol_source={aclnn_symbol_source}",
            ]
        )
        shared_lib_dir = staged_custom_opp_root / "op_api" / "lib"
        if build_dir.exists():
            _remove_path(build_dir)
        validation_proc = self._run_shell_command(
            command_parts=[
                f"export PATH={_shell_quote(Path(sys.executable).absolute().parent)}:$PATH",
                (
                    "cmake --no-warn-unused-cli "
                    f"-S {_shell_quote(validation_dir)} "
                    f"-B {_shell_quote(build_dir)} "
                    f"-DSHARED_LIB_DIR={_shell_quote(shared_lib_dir)} "
                    f"-DPython3_EXECUTABLE={_shell_quote(sys.executable)}"
                ),
                f"cmake --build {_shell_quote(build_dir)} -j{self.compile_jobs}",
            ],
            live_log_path=log_path,
            live_log_header="## live_validation_build_direct",
        )
        validation_returncode = validation_proc.returncode
        validation_output = (validation_proc.stdout or "") + "\n" + (validation_proc.stderr or "")
        extension_path = next(build_dir.glob("kernel_gen_ops*.so"), None)
        validation_metadata = [
            "source_mode=direct",
            f"validation_returncode={validation_returncode}",
            f"extension_path={extension_path}",
        ]
        log_sections.extend(
            [
                "## validation_build_direct",
                "\n".join(validation_metadata) + "\n" + validation_output,
            ]
        )

        success = validation_returncode == 0 and extension_path is not None
        symbol_check_ok = True
        symbol_check_detail = "skipped: build_failed_before_symbol_check"
        if success:
            symbol_check_ok, symbol_check_detail = self._verify_aclnn_symbol(
                custom_opp_root=staged_custom_opp_root,
                aclnn_symbol=acl_symbol_name,
            )
            if not symbol_check_ok:
                success = False
        log_sections.extend(["## symbol_check", symbol_check_detail])

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(log_sections), encoding="utf-8")

        if success:
            self._stage_validation_runtime_root(build_dir=build_dir, custom_opp_root=staged_custom_opp_root)
            self._write_build_manifest(
                build_dir,
                self._build_binding_manifest(
                    op_root=op_root,
                    op_name=op_name,
                    acl_op_name=compile_target.compile_op_name,
                    compile_manifest=compile_manifest,
                ),
            )
        message = (
            f"compile_mode=direct; compile_exit_code={compile_proc.returncode}; "
            f"validation_exit_code={validation_returncode}; "
            f"aclnn_symbol={acl_symbol_name}; aclnn_symbol_source={aclnn_symbol_source}"
        )
        if not symbol_check_ok:
            message += "; symbol_check=failed"
        return BuildResult(
            success=success,
            op_name=op_name,
            standard_op_root=op_root,
            work_dir=src_dir,
            build_dir=build_dir,
            log_path=log_path,
            extension_path=extension_path,
            custom_opp_root=staged_custom_opp_root,
            message=message,
        )

    def _stage_run_local_custom_opp_root(
        self,
        *,
        source_custom_opp_root: Path,
        op_run_root: Path,
        vendor_dir_name: str,
    ) -> Path:
        staged_root = op_run_root / "custom_opp" / vendor_dir_name
        staged_root.parent.mkdir(parents=True, exist_ok=True)
        _copytree_clean(source_custom_opp_root, staged_root)
        return staged_root

    def _prepare_run_local_compile_repo_workspace(
        self,
        *,
        src_dir: Path,
        compile_target: CompileRepoTarget,
        source_root: Path,
    ) -> None:
        if src_dir.exists() or src_dir.is_symlink():
            _remove_path(src_dir)
        shutil.copytree(
            compile_target.repo_root,
            src_dir,
            ignore=_make_compile_repo_ignore(compile_target.repo_root),
            dirs_exist_ok=True,
        )
        _patch_compile_repo_ops_info_generation(src_dir)
        if compile_target.repo_name == "ops-skeleton" and (src_dir / "scripts/build_cmake.sh").is_file():
            subprocess.run(
                ["git", "apply", "--no-index", str(Path(__file__).resolve().parents[2] / "dependency/ops-skeleton.patch")],
                cwd=src_dir, check=True,
            )
        target_op_dir = src_dir / compile_target.relative_op_dir
        _copytree_clean(source_root, target_op_dir)
        self._sync_op_common(scaffold_root=src_dir, source_root=source_root, target_root=target_op_dir)

    def _run_run_local_compile_repo_build(
        self,
        *,
        repo_root: Path,
        compile_target: CompileRepoTarget,
        chip_name: str,
        log_path: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        opbase = (self.dependency_root / "ops-base").resolve()
        opbase_env = []
        if (opbase / ".git").exists():
            opbase_env = [
                'if [ -z "${OPBASE_SOURCE_PATH:-}" ]; then '
                f"export OPBASE_SOURCE_PATH={_shell_quote(opbase)}; fi"
            ]
        return self._run_shell_command(
            command_parts=[
                *opbase_env,
                f"export PATH={_shell_quote(Path(sys.executable).absolute().parent)}:$PATH",
                "export SKIP_ONNX_PLUGIN_FOR_CUSTOM=TRUE",
                "export PIPELINE_INCREMENTAL_BUILD=TRUE",
                f"cd {_shell_quote(repo_root)}",
                self._build_compile_repo_command(chip_name=chip_name, compile_target=compile_target),
            ],
            live_log_path=log_path,
            live_log_header="## live_compile_repo_build_direct",
        )

    def _verify_aclnn_symbol(
        self,
        *,
        custom_opp_root: Path | None,
        aclnn_symbol: str,
    ) -> tuple[bool, str]:
        nm_path = shutil.which("nm")
        if nm_path is None:
            return True, f"skipped: nm_not_on_PATH; aclnn_symbol={aclnn_symbol}"

        wanted = [aclnn_symbol, f"{aclnn_symbol}GetWorkspaceSize"]
        candidates: list[Path] = []
        if custom_opp_root is not None:
            candidates.append(custom_opp_root / "op_api" / "lib" / "libcust_opapi.so")
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        candidates.append(Path(ascend_home) / "lib64" / "libopapi.so")

        existing = [lib for lib in candidates if lib.is_file()]
        if not existing:
            detail_lines = [f"aclnn_symbol={aclnn_symbol}", "skipped: no_candidate_libs_found"]
            detail_lines.extend(f"  missing: {lib}" for lib in candidates)
            return True, "\n".join(detail_lines)

        remaining = set(wanted)
        hits: dict[str, Path] = {}
        per_lib_lines: list[str] = []
        for lib in existing:
            try:
                proc = subprocess.run(
                    [nm_path, "-D", "--defined-only", str(lib)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                per_lib_lines.append(f"  {lib}: nm_error={exc}")
                continue
            if proc.returncode != 0:
                per_lib_lines.append(f"  {lib}: nm_returncode={proc.returncode}")
                continue
            lib_symbols = {
                line.rsplit(" ", 1)[-1]
                for line in proc.stdout.splitlines()
                if line
            }
            matched_in_lib = [sym for sym in wanted if sym in lib_symbols]
            for sym in matched_in_lib:
                hits.setdefault(sym, lib)
                remaining.discard(sym)
            if matched_in_lib:
                per_lib_lines.append(f"  {lib}: FOUND={','.join(matched_in_lib)}")
            else:
                per_lib_lines.append(f"  {lib}: NOT_FOUND")

        header = [f"aclnn_symbol={aclnn_symbol}", f"wanted={','.join(wanted)}", "checked:"]
        if not remaining:
            detail = "\n".join(header + per_lib_lines + ["status=ok"])
            return True, detail

        hint_lines = [
            "status=failed",
            "hint: pybind.cpp is calling an aclnn symbol that is not exported by the",
            "      vendor's libcust_opapi.so nor by CANN's libopapi.so. Verify the",
            "      real symbol in the operator source tree (op_host/op_api/*.h) and",
            "      record it in op_info.json binding.aclnn_symbol, then freeze",
            "      binding/pybind.cpp to call that symbol directly.",
            f"missing_symbols={','.join(sorted(remaining))}",
        ]
        detail = "\n".join(header + per_lib_lines + hint_lines)
        return False, detail

    def _custom_opp_install_ready(self, custom_opp_root: Path | None) -> bool:
        if custom_opp_root is None or not custom_opp_root.exists():
            return False
        required_paths = [
            custom_opp_root / "op_api" / "lib" / "libcust_opapi.so",
            custom_opp_root / "op_impl",
        ]
        return all(path.exists() for path in required_paths)

    def _sync_op_common(self, *, scaffold_root: Path, source_root: Path, target_root: Path) -> None:
        op_common_src = source_root / "op_common"
        op_common_dst = target_root / "op_common"
        if not op_common_src.exists():
            return
        _copytree_clean(op_common_src, op_common_dst)
        self._ensure_op_common_includes(scaffold_root / "CMakeLists.txt")

    def _ensure_op_common_includes(self, cmake_path: Path) -> None:
        cmake_text = cmake_path.read_text(encoding="utf-8")
        if "op_common" in cmake_text:
            return
        op_common_snippet = (
            "\n# op_common: shared headers copied from the operator directory\n"
            "if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/op_common)\n"
            "    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/op_common)\n"
            "endif()\n"
        )
        cmake_path.write_text(cmake_text + op_common_snippet, encoding="utf-8")

    def _patch_pybind(
        self,
        *,
        pybind_path: Path,
        acl_op_name: str,
        exported_api_name: str,
        explicit_aclnn_symbol: str | None = None,
        explicit_aclnn_symbol_source: str | None = None,
    ) -> tuple[str, str]:
        pybind_text = pybind_path.read_text(encoding="utf-8")
        original_custom_mdef = _find_mdef_statement(pybind_text, "custom_pybind_api")
        if explicit_aclnn_symbol:
            acl_symbol_name = explicit_aclnn_symbol
            aclnn_symbol_source = explicit_aclnn_symbol_source or "op_info.json"
        else:
            acl_symbol_name = f"aclnn{_to_acl_symbol_name(acl_op_name)}"
            aclnn_symbol_source = "derived"
        python_symbol_name = _to_snake_case(exported_api_name)
        if "aclnnCustomOp" in pybind_text:
            pybind_text = pybind_text.replace("aclnnCustomOp", acl_symbol_name)
        elif "aclnn" not in pybind_text:
            raise ValueError(f"expected placeholder aclnnCustomOp or any aclnn symbol in {pybind_path}")
        else:
            frozen_symbol = _extract_first_aclnn_symbol(pybind_text)
            if frozen_symbol and explicit_aclnn_symbol:
                pybind_text = pybind_text.replace(
                    f"{frozen_symbol}GetWorkspaceSize",
                    f"{acl_symbol_name}GetWorkspaceSize",
                )
                pybind_text = pybind_text.replace(frozen_symbol, acl_symbol_name)
            elif frozen_symbol:
                acl_symbol_name = frozen_symbol
                aclnn_symbol_source = "frozen_pybind"
        if "custom_pybind_api" in pybind_text:
            pybind_text = pybind_text.replace("custom_pybind_api", python_symbol_name)
            pybind_text = _add_custom_pybind_api_alias(
                pybind_text=pybind_text,
                original_custom_mdef=original_custom_mdef,
                python_symbol_name=python_symbol_name,
            )
        elif python_symbol_name not in pybind_text:
            raise ValueError(
                f"expected placeholder custom_pybind_api or final symbol {python_symbol_name} in {pybind_path}"
            )
        pybind_path.write_text(pybind_text, encoding="utf-8")
        return acl_symbol_name, aclnn_symbol_source

    def _patch_cmake_preset(self, preset_path: Path, chip_name: str) -> None:
        ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
        preset_data = json.loads(
            preset_path.read_text(encoding="utf-8").replace("/usr/local/Ascend/ascend-toolkit/latest", ascend_home)
        )
        for preset in preset_data.get("configurePresets", []):
            cache_variables = preset.get("cacheVariables", {})
            compute_unit = cache_variables.get("ASCEND_COMPUTE_UNIT")
            if isinstance(compute_unit, dict):
                compute_unit["value"] = chip_name
        preset_path.write_text(json.dumps(preset_data, indent=4), encoding="utf-8")

    def _build_compile_repo_command(self, *, chip_name: str, compile_target: CompileRepoTarget) -> str:
        command = ["bash build.sh", "--pkg"]
        if compile_target.repo_name in REPOS_SUPPORTING_NO_FORCE:
            command.append("--no_force")
        command.extend(
            [
                f"--soc={chip_name}",
                f"--ops={compile_target.compile_op_name}",
                f"--vendor_name={compile_target.vendor_name}",
                f"--vendor_suffix={compile_target.family}",
                f"-j{self.compile_jobs}",
            ]
        )
        if compile_target.relative_op_dir.parts and compile_target.relative_op_dir.parts[0] == "experimental":
            command.append("--experimental")
        return " ".join(command)

    def _prepare_validation_sources(
        self,
        *,
        op_root: Path,
        validation_dir: Path,
        acl_op_name: str,
        exported_api_name: str,
    ) -> tuple[str, str]:
        if validation_dir.exists() or validation_dir.is_symlink():
            _remove_path(validation_dir)
        validation_dir.mkdir(parents=True, exist_ok=True)
        template_dir = self.validation_root
        shutil.copy2(template_dir / "CMakeLists.txt", validation_dir / "CMakeLists.txt")
        shutil.copy2(template_dir / "pytorch_npu_helper.hpp", validation_dir / "pytorch_npu_helper.hpp")
        shutil.copy2(op_root / "binding" / "pybind.cpp", validation_dir / "pybind.cpp")
        binding_info = _load_op_info_binding(op_root)
        declared_symbol = str(binding_info.get("aclnn_symbol") or "").strip() or None
        return self._patch_pybind(
            pybind_path=validation_dir / "pybind.cpp",
            acl_op_name=acl_op_name,
            exported_api_name=exported_api_name,
            explicit_aclnn_symbol=declared_symbol,
        )

    def _stage_validation_runtime_root(self, *, build_dir: Path, custom_opp_root: Path | None) -> None:
        if custom_opp_root is None:
            return
        runtime_root = build_dir / "runtime"
        if runtime_root.exists() or runtime_root.is_symlink():
            _remove_path(runtime_root)
        runtime_root.symlink_to(custom_opp_root.resolve(), target_is_directory=True)

    def _find_custom_opp_root(self, *, build_out_dir: Path, expected_vendor_dir_name: str | None) -> Path | None:
        vendor_roots: list[Path] = []
        # Search both supported CPack layouts. The unified skeleton currently
        # stages packages under build_out/, while build/ remains a compatibility
        # fallback for previously generated workspaces.
        search_roots = [build_out_dir, build_out_dir.parent / "build"]
        for search_root in search_roots:
            direct_vendor_root = search_root / "packages" / "vendors"
            if direct_vendor_root.exists() and direct_vendor_root not in vendor_roots:
                vendor_roots.append(direct_vendor_root)
            cpack_root = search_root / "_CPack_Packages"
            if cpack_root.exists():
                vendor_roots.extend(
                    candidate
                    for candidate in sorted(cpack_root.rglob("packages/vendors"))
                    if candidate.is_dir() and candidate not in vendor_roots
                )
        if not vendor_roots:
            return None
        if expected_vendor_dir_name is not None:
            for vendor_root in vendor_roots:
                candidate = vendor_root / expected_vendor_dir_name
                if candidate.exists():
                    return candidate
        for vendor_root in vendor_roots:
            libraries = sorted(vendor_root.rglob("libcust_opapi.so"))
            for library in libraries:
                if library.parent.name == "lib" and library.parent.parent.name == "op_api":
                    return library.parent.parent.parent
        for vendor_root in vendor_roots:
            vendor_dirs = sorted(path for path in vendor_root.iterdir() if path.is_dir())
            if vendor_dirs:
                return vendor_dirs[0]
        return None

    def _kernel_objects_exist(self, custom_opp_root: Path | None, *, expected_name: str) -> bool:
        if custom_opp_root is None or not custom_opp_root.exists():
            return False
        object_paths = list(custom_opp_root.rglob("*.o"))
        if not object_paths:
            return False
        normalized_name = expected_name.lower().replace("-", "_")
        return any(
            normalized_name in path.as_posix().lower().replace("-", "_") or path.stem.lower().startswith(normalized_name)
            for path in object_paths
        )

    def _run_shell_command(
        self,
        *,
        command_parts: list[str],
        live_log_path: Path | None = None,
        live_log_header: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        full_command = []
        if self.env_script is not None:
            full_command.append(f"source {_shell_quote(self.env_script)}")
        full_command.extend(command_parts)
        shell_command = " && ".join(full_command)
        if live_log_path is not None:
            live_log_path.parent.mkdir(parents=True, exist_ok=True)
            header_lines = [part for part in (live_log_header, f"$ {shell_command}") if part]
            with live_log_path.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(header_lines) + "\n")
            shell_command = (
                "set -o pipefail && "
                f"({shell_command}) 2>&1 | tee -a {_shell_quote(live_log_path)}"
            )
        return run_shell_with_timeout(
            shell_command,
            timeout=DEFAULT_BUILD_TIMEOUT_SEC,
        )

    def _resolve_compile_log_path(self, op_run_root: Path) -> Path:
        override = os.environ.get("BENCHMARK_COMPILE_LOG_PATH", "").strip()
        if override:
            return Path(override).resolve()
        return op_run_root / "compile.log"

    def _failure_result(
        self,
        *,
        op_name: str,
        standard_op_root: Path,
        op_run_root: Path,
        log_label: str,
        message: str,
    ) -> BuildResult:
        log_path = self._resolve_compile_log_path(op_run_root)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(f"{log_label}\n{message}\n", encoding="utf-8")
        return BuildResult(
            success=False,
            op_name=op_name,
            standard_op_root=standard_op_root,
            work_dir=op_run_root / "src",
            build_dir=op_run_root / "build",
            log_path=log_path,
            extension_path=None,
            custom_opp_root=None,
            message=message,
        )
