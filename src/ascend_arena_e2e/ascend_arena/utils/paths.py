# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Path manipulation utilities — consolidated from multiple modules."""

from __future__ import annotations

import errno
import time
import shutil
import stat
from pathlib import Path


def ensure_writable(path: Path) -> None:
    """Make a single file or directory user-writable."""
    if not path.exists():
        return
    try:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    except OSError:
        pass


def ensure_tree_writable(path: Path) -> None:
    """Recursively make all files and directories under *path* user-writable."""
    if not path.exists():
        return
    targets = [path, *path.rglob("*")] if path.is_dir() else [path]
    for item in targets:
        try:
            item.chmod(item.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass


def ensure_parent_writable(path: Path) -> None:
    """Ensure the parent directory of *path* is writable, then make *path* writable."""
    if path.parent.exists():
        try:
            path.parent.chmod(path.parent.stat().st_mode | stat.S_IWUSR)
        except OSError:
            pass
    ensure_writable(path)


def _handle_remove_readonly(func, path, _exc_info) -> None:  # type: ignore[no-untyped-def]
    """Error handler for shutil.rmtree on read-only files."""
    if _exc_info[1].errno not in (errno.EACCES, errno.EPERM):
        raise _exc_info[1]
    p = Path(path)
    p.chmod(p.stat().st_mode | stat.S_IWUSR)
    func(path)


def remove_path(path: Path) -> None:
    """Remove a file or symlink. Does nothing if *path* doesn't exist."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        remove_tree(path)
    else:
        path.unlink()


def remove_tree(path: Path) -> None:
    """Recursively remove a directory tree, handling read-only files."""
    if not path.exists():
        return
    ensure_tree_writable(path)
    delays = (0.2, 0.5, 1, 2, 4)
    for attempt in range(len(delays) + 1):
        try:
            shutil.rmtree(path, onerror=_handle_remove_readonly)
            return
        except FileNotFoundError:
            if not path.exists():
                return
            raise
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EBUSY) or attempt == len(delays):
                raise
            time.sleep(delays[attempt])


def copy_tree(src: Path, dst: Path) -> None:
    """Copy *src* directory to *dst*, removing *dst* first if it exists."""
    if dst.exists():
        remove_tree(dst)
    shutil.copytree(src, dst)


def copy_path(source: Path, destination: Path) -> None:
    """Copy a file or directory from *source* to *destination*."""
    if destination.exists() or destination.is_symlink():
        remove_path(destination)
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def resolve_compile_repo(dependency_root: Path, name: str) -> Path:
    """Locate the build root in a checkout or a prepared skeleton directory."""
    repo = dependency_root / name
    nested = repo / "ops_optimization_skeleton"
    if name == "ops-skeleton" and (nested / "CMakeLists.txt").is_file():
        return nested
    return repo
