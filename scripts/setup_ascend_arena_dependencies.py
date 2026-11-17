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

"""Fetch pinned Arena build sources and materialize the benchmark baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


DEPENDENCY = "src/ascend_arena_e2e/dependency"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def sparse_dependency(root: Path, relative: str, patterns: list[str], expected: str | None = None) -> None:
    url = git(root, "config", "-f", ".gitmodules", f"submodule.{relative}.url")
    entry = git(root, "ls-files", "--stage", "--", relative).split()
    if len(entry) != 4 or entry[0] != "160000" or entry[2] != "0":
        raise RuntimeError(f"Dependency must have an unambiguous pinned submodule entry: {relative}")
    revision = entry[1]
    if expected is not None and revision != expected:
        raise RuntimeError(f"Submodule and source manifest revisions differ: {relative}")
    checkout = root / relative
    if not (checkout / ".git").exists():
        checkout.mkdir(parents=True, exist_ok=True)
        git(checkout, "init", "--quiet")
        git(checkout, "remote", "add", "origin", url)
        git(checkout, "config", "remote.origin.promisor", "true")
        git(checkout, "config", "remote.origin.partialclonefilter", "blob:none")
    git(checkout, "remote", "set-url", "origin", url)
    # Resolve only the ref: probing a missing object can start an unbounded promisor fetch.
    current = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "--verify", "--quiet", "HEAD"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    ).stdout.strip()
    if current != revision:
        git(checkout, "fetch", "--filter=blob:none", "--depth=1", "origin", revision)
    git(checkout, "sparse-checkout", "set", "--no-cone", *patterns)
    git(checkout, "checkout", "--detach", revision)
    git(root, "submodule", "absorbgitdirs", "--", relative)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_baselines(directory: Path, manifest: dict) -> None:
    expected = {f"{op}/{relative}": item["sha256"]
                for op, files in manifest["operators"].items() for relative, item in files.items()}
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if (actual != set(expected) or any(path.is_symlink() for path in directory.rglob("*"))
            or any(digest(directory / path) != value for path, value in expected.items())):
        raise RuntimeError(f"Baseline sources differ from the manifest; move local changes aside first: {directory}")


def materialize_baselines(root: Path, manifest: dict) -> None:
    dependency = root / DEPENDENCY
    destination = dependency / "baselines"
    if destination.exists():
        verify_baselines(destination, manifest)
        return
    with tempfile.TemporaryDirectory(prefix=".baselines-", dir=dependency) as temporary:
        staging = Path(temporary) / "sources"
        staging.mkdir()
        for op, files in manifest["operators"].items():
            for relative, item in files.items():
                if "repository" in item:
                    source = dependency / item["repository"] / item["source"]
                    if digest(source) != item["source_sha256"]:
                        raise RuntimeError(f"Official source checksum mismatch: {source}")
                else:
                    source = dependency / "baseline_overrides" / item["override"]
                target = staging / op / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if "repository" in item:
                    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
                else:
                    shutil.copy2(source, target)
        subprocess.run(
            ["git", "apply", "--no-index", "--unidiff-zero", "--whitespace=nowarn",
             str((dependency / "baselines.patch").resolve())], cwd=staging, check=True,
        )
        verify_baselines(staging, manifest)
        staging.rename(destination)


def setup(root: Path) -> None:
    dependency = root / DEPENDENCY
    manifest = json.loads((dependency / "baselines.json").read_text())
    paths = [f"{DEPENDENCY}/{name}" for name in ("ops-skeleton", "ops-base", "ops-nn", "ops-transformer")]
    # Check every checkout and existing baseline before modifying any dependency.
    for relative in paths:
        path = root / relative
        if (path / ".git").exists():
            if git(path, "status", "--porcelain", "--untracked-files=all"):
                raise RuntimeError(f"Dependency has local changes; commit or move them first: {path}")
        elif path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Dependency directory is not an empty Git checkout: {path}")
    if (dependency / "baselines").exists():
        verify_baselines(dependency / "baselines", manifest)

    git(root, "submodule", "init", "--", *paths)
    git(root, "submodule", "sync", "--", *paths)
    sparse_dependency(root, paths[0], ["/ops_optimization_skeleton/"])
    opbase = paths[1]
    git(root, "submodule", "update", "--init", "--depth=1", "--", opbase)
    for name, entry in manifest["repositories"].items():
        sparse_dependency(root, f"{DEPENDENCY}/{name}", [f"/{path}" for path in entry["paths"]], entry["revision"])
    if not (dependency / "ops-skeleton/ops_optimization_skeleton/CMakeLists.txt").is_file():
        raise RuntimeError("Pinned checkout does not contain the build skeleton")
    materialize_baselines(root, manifest)
    print(git(root, "submodule", "status", "--", *paths))
    print("Arena build sources and baselines are ready; evaluation can run offline.")


if __name__ == "__main__":
    setup(Path(__file__).resolve().parents[1])
