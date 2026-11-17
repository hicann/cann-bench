# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Build compatibility for the native measurement helpers."""

import hashlib
import json
import os
import sys
from pathlib import Path


def current_environment():
    import torch
    import torch_npu

    root = os.environ.get("ASCEND_HOME_PATH")
    if not root:
        raise RuntimeError("Load CANN set_env.sh before checking measurement helpers")
    version = next(line.split("=", 1)[1].strip()
                   for line in (Path(root) / "compiler/version.info").read_text().splitlines()
                   if line.lower().startswith("version="))
    return {"cann": version, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}"}


def source_fingerprint(source_root):
    root = Path(source_root)
    files = [root / name for name in ("CMakeLists.txt", "setup.py", "build.sh")]
    for directory, suffixes in (("cmake", {".cmake"}), ("csrc", {".cpp", ".h"}),
                                ("cann_bench_utils", {".py"})):
        files.extend(path for path in (root / directory).rglob("*") if path.suffix in suffixes)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda path: path.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def check_build_compatibility(source_root=None):
    package = Path(__file__).parent
    path = package / "_build_info.json"
    if not path.is_file():
        raise RuntimeError("Measurement helper build metadata is missing; rebuild cann_bench_utils in the active CANN environment")
    info = json.loads(path.read_text())
    active = current_environment()
    mismatched = [name for name, value in active.items() if info.get(name) != value]
    if mismatched:
        raise RuntimeError(f"Measurement helper build mismatch: {', '.join(mismatched)}; rebuild cann_bench_utils in the active environment")
    if info.get("binary_sha256") != hashlib.sha256((package / "_C.so").read_bytes()).hexdigest():
        raise RuntimeError("Measurement helper binary does not match its build metadata; rebuild cann_bench_utils")
    if source_root is not None and info.get("source_sha256") != source_fingerprint(source_root):
        raise RuntimeError("Measurement helper source has changed; rebuild cann_bench_utils")
    return info
