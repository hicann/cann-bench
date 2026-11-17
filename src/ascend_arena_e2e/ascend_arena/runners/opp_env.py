# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""OPP runtime environment preparation for benchmark execution."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping

from ascend_arena_e2e.ascend_arena.utils.paths import ensure_tree_writable, remove_path


def _link_or_copy_file(src: Path, dst: Path) -> None:
    """Hard-link a file if possible, otherwise copy."""
    try:
        os.link(src, dst, follow_symlinks=True)
    except OSError:
        shutil.copy2(src, dst, follow_symlinks=True)


def _materialize_opp_child(src: Path, dst: Path) -> None:
    """Copy a child entry from the base OPP root into the runtime root."""
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            copy_function=_link_or_copy_file,
            symlinks=False,
        )
        ensure_tree_writable(dst)
        return
    _link_or_copy_file(src, dst)


def _refresh_runtime_opp_root(
    *,
    base_opp_root: Path,
    runtime_opp_root: Path,
    custom_opp_root: Path,
) -> None:
    """Reuse the completed OPP tree across stages of one evaluation run."""
    stamp = runtime_opp_root.with_suffix(".source.json")
    fingerprint = {"base": str(base_opp_root.resolve()), "custom": str(custom_opp_root.resolve())}
    vendor = runtime_opp_root / "vendors" / custom_opp_root.name
    try:
        if (
            json.loads(stamp.read_text()) == fingerprint
            and vendor.is_symlink() and vendor.resolve() == custom_opp_root.resolve()
            and (runtime_opp_root / "vendors/config.ini").read_text() == f"load_priority={custom_opp_root.name}\n"
            and all((runtime_opp_root / child.name).exists()
                    for child in base_opp_root.iterdir() if child.name != "vendors")
        ):
            return
    except (OSError, ValueError):
        pass
    stamp.unlink(missing_ok=True)
    remove_path(runtime_opp_root)
    runtime_opp_root.mkdir(parents=True, exist_ok=True)

    for child in base_opp_root.iterdir():
        if child.name == "vendors":
            continue
        target = runtime_opp_root / child.name
        _materialize_opp_child(child, target)

    vendors_dir = runtime_opp_root / "vendors"
    vendors_dir.mkdir(parents=True, exist_ok=True)
    vendor_name = custom_opp_root.name
    (vendors_dir / "config.ini").write_text(f"load_priority={vendor_name}\n", encoding="utf-8")
    (vendors_dir / vendor_name).symlink_to(custom_opp_root.resolve(), target_is_directory=True)
    stamp.write_text(json.dumps(fingerprint), encoding="utf-8")


def prepare_runtime_opp_env(
    *,
    op_run_root: Path,
    custom_opp_root: Path | None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Prepare a runtime OPP environment dict with custom vendor overlay.

    Returns a copy of *env* (or empty dict) with ASCEND_OPP_PATH pointing
    to the assembled runtime OPP directory.
    """
    prepared_env = dict(env or {})
    if custom_opp_root is None:
        return prepared_env

    prepared_env.pop("ASCEND_CUSTOM_OPP_PATH", None)
    base_opp_path = prepared_env.get("ASCEND_OPP_PATH", "").strip()
    if not base_opp_path:
        return prepared_env

    base_opp_root = Path(base_opp_path).resolve()
    if not base_opp_root.exists():
        return prepared_env

    # CANN also uses this vendor path to select the candidate's host tiling.
    prepared_env["ASCEND_CUSTOM_OPP_PATH"] = str(custom_opp_root.resolve())
    runtime_opp_root = op_run_root.resolve() / "runtime_opp"
    if base_opp_root == runtime_opp_root.resolve() and runtime_opp_root.exists():
        prepared_env["ASCEND_OPP_PATH"] = str(runtime_opp_root)
        return prepared_env

    _refresh_runtime_opp_root(
        base_opp_root=base_opp_root,
        runtime_opp_root=runtime_opp_root,
        custom_opp_root=custom_opp_root.resolve(),
    )
    prepared_env["ASCEND_OPP_PATH"] = str(runtime_opp_root)
    prepared_env["KERNEL_GEN_OPS_CUST_OPAPI_PATH"] = str(
        custom_opp_root.resolve() / "op_api" / "lib" / "libcust_opapi.so"
    )
    return prepared_env


@contextmanager
def isolated_runtime_opp_env(*, op_run_root: Path, custom_opp_root: Path | None):
    """Context manager that temporarily sets ASCEND_OPP_PATH in os.environ."""
    original_env = dict(os.environ)
    prepared_env = prepare_runtime_opp_env(
        op_run_root=op_run_root,
        custom_opp_root=custom_opp_root,
        env=original_env,
    )
    managed_keys = (
        "ASCEND_OPP_PATH",
        "ASCEND_CUSTOM_OPP_PATH",
        "KERNEL_GEN_OPS_CUST_OPAPI_PATH",
    )
    changed_keys = [
        key
        for key in managed_keys
        if prepared_env.get(key) != original_env.get(key) or ((key in prepared_env) != (key in original_env))
    ]

    for key in changed_keys:
        if key in prepared_env:
            os.environ[key] = prepared_env[key]
        else:
            os.environ.pop(key, None)

    try:
        yield prepared_env
    finally:
        for key in changed_keys:
            if key in original_env:
                os.environ[key] = original_env[key]
            else:
                os.environ.pop(key, None)
