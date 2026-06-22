"""Deterministically render the public dispatch entry for a multi-class op.

After every class is generated, the orchestrator writes ``<op>.py`` next to the
class subdirs. The public ``<op>`` function reads each input's (ndim, dtype) and
forwards to the matching class impl wrapper. No convert-time agent is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def write_dispatcher(parent_op_dir: Path, manifest: Mapping[str, object]) -> Path:
    op_name = str(manifest["op_name"])
    target = parent_op_dir / f"{op_name}.py"
    target.write_text(_render(manifest), encoding="utf-8")
    return target


def _render(manifest: Mapping[str, object]) -> str:
    op_name = str(manifest["op_name"])
    classes = [
        {"subdir": str(c["subdir"]), "signature": [[int(n), str(d)] for n, d in c["signature"]]}
        for c in manifest["classes"]
    ]
    return f'''"""Auto-generated dim/dtype dispatcher for `{op_name}`. Do not edit by hand."""

import importlib.util
from pathlib import Path

_OP_NAME = {op_name!r}
_CLASSES = {json.dumps(classes)}
_BASE = Path(__file__).resolve().parent
_impl_cache = {{}}


def _dtype_name(value):
    return str(value).rsplit(".", 1)[-1]


def _signature(args):
    sig = []
    for a in args:
        if hasattr(a, "dim") and hasattr(a, "dtype"):
            sig.append([int(a.dim()), _dtype_name(a.dtype)])
    return sig


def _load(subdir):
    if subdir not in _impl_cache:
        path = _BASE / subdir / f"{{_OP_NAME}}_impl.py"
        spec = importlib.util.spec_from_file_location(f"{{_OP_NAME}}_{{subdir}}_impl", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = next((getattr(module, n) for n in (_OP_NAME, f"{{_OP_NAME}}_wrapper") if callable(getattr(module, n, None))), None)
        if fn is None:
            raise AttributeError(f"{{path}} exposes no '{{_OP_NAME}}' or '{{_OP_NAME}}_wrapper' entry")
        _impl_cache[subdir] = fn
    return _impl_cache[subdir]


def {op_name}(*args, **kwargs):
    sig = _signature(args)
    for entry in _CLASSES:
        n = len(entry["signature"])
        if sig[:n] == entry["signature"]:
            return _load(entry["subdir"])(*args, **kwargs)
    raise ValueError(f"no {{_OP_NAME}} class for signature {{sig}}; classes={{_CLASSES}}")
'''
