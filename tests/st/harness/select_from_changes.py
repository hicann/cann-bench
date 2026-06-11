"""把 PR 改动文件列表映射成 pytest -k 表达式,供 run_st.sh 按改动范围选 op。

把改动分成两类:tasks/levelN/<op>/ 下的(→ 该 op,读 proto.yaml 取真名)与其余「tasks 外」
改动(框架 / 文档 / tasks/metadata / CI 脚本等,可能影响任意算子)。据此:
  - 仅 tasks 算子改动        → 相应算子(-k "levelN:Name or ...");
  - 仅 tasks 外改动          → 默认组(返回空 → run_st.sh 退回 conftest 的默认冒烟 Cummin);
  - tasks 与 tasks 外都有    → 相应算子 ∪ 默认组(在 -k 里追加 DEFAULT_GROUP);
  - 无任何可识别算子         → 空(默认组)。
(清单不存在的情形由 run_st.sh 处理:WARN + 默认组,与上面区分。)

用 level 前缀 + 真名,避免跨 level / 大小写子串误命中。

用法:PYTHONPATH=tests/st python -m harness.select_from_changes [filelist]  # 省略则读 stdin
输出:一行 -k 表达式(如 "level1:Exp or level2:Cummin"),或空行。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

from .eval_run import TASKS

# search(非 match):兼容裸路径与 git --name-status 的 "M\ttasks/..." 前缀。
_TASK_RE = re.compile(r"tasks/(level[1-4])/([^/]+)/")
# 默认冒烟组 —— 须与 conftest.py 无参时的默认选择一致(目前是 Cummin)。改默认组时两处同步。
DEFAULT_GROUP = "level2:Cummin"


def _read_lines(argv: list[str]) -> list[str]:
    src = argv[1] if len(argv) > 1 and argv[1] not in ("", "-") else None
    text = Path(src).read_text(encoding="utf-8") if src else sys.stdin.read()
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _op_token(level: str, op_dir: str) -> str | None:
    """levelN/<dir> → 'levelN:OperatorName'(读 proto.yaml 取真名);缺 proto / 已删 → None。"""
    try:
        name = yaml.safe_load((TASKS / level / op_dir / "proto.yaml").read_text(encoding="utf-8"))[
            "operator"
        ]["name"]
    except Exception:
        return None
    return f"{level}:{name}"


def selector(changed_paths: list[str]) -> str:
    """改动路径 → pytest -k 表达式。命中算子去重保序;若同时有 tasks 外改动则并入默认组;
    无命中算子则返回空(由 run_st.sh 退回默认组)。"""
    ops: list[str] = []
    has_nontask = False
    for path in changed_paths:
        m = _TASK_RE.search(path.replace("\\", "/"))
        if not m:
            has_nontask = True          # 框架/文档/metadata 等:可能影响任意算子
            continue
        tok = _op_token(m.group(1), m.group(2))
        if tok and tok not in ops:
            ops.append(tok)
    if not ops:
        return ""                       # 仅 tasks 外改动(或无可识别算子)→ 默认组
    if has_nontask and DEFAULT_GROUP not in ops:
        ops.append(DEFAULT_GROUP)       # tasks ∪ tasks 外 → 相应算子 + 默认组
    return " or ".join(ops)


if __name__ == "__main__":
    print(selector(_read_lines(sys.argv)))
