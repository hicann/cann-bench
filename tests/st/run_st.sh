#!/bin/bash
# Golden-candidate ST entry: pytest over the golden ops on the NPU. CI ST stage runs this
# directly inside its NPU image(`bash tests/st/run_st.sh`)。Args pass to pytest;
# default selection (unless --full / -k / -m) lives in tests/st/conftest.py.
set -uo pipefail   # not -e: run every case + still emit junit/reports when some fail

# make sure ST_OUT is clean
ST_OUT="${ST_OUT:-tests/st/_artifacts}"
rm -rf "$ST_OUT"
mkdir -p "$ST_OUT"
export PYTHONUNBUFFERED=1

PYTEST_ARGS=(tests/st/test_golden_npu_mock.py)
# 无参 `bash tests/st/run_st.sh` 是 CI 入口, 应提供改动清单 $PR_FILELIST
PR_FILELIST="${PR_FILELIST:-pr_filelist.txt}"
if [ "$#" -eq 0 ]; then
  if [ ! -r "$PR_FILELIST" ]; then
    echo "[run_st] ERROR: 未找到改动清单 $PR_FILELIST, CI 应在仓库根产出该清单." >&2
    echo "[run_st]        本地手动跑请显式传 -k/-m/--full (如 bash tests/st/run_st.sh -k Cummin)." >&2
    exit 2
  fi
  expr=$(PYTHONPATH=tests/st python -m harness.select_from_changes "$PR_FILELIST" 2>/dev/null || true)
  if [ -n "$expr" ]; then
    echo "[run_st] $PR_FILELIST → -k '$expr'"
    PYTEST_ARGS+=(-k "$expr")
  else
    echo "[run_st] $PR_FILELIST 无 tasks/ 算子改动 → 默认组 (见 tests/st/conftest.py)"
  fi
fi
# basetemp under $ST_OUT (bind mount), not the container's /tmp: if the container dies
# mid-run, the candidate/trimmed-tree/report tmp survives for post-mortem (cleaned on normal exit).
PYTEST_ARGS+=("$@" -v -ra -p no:cacheprovider \
              --junitxml="$ST_OUT/matrix_junit.xml" --basetemp="$ST_OUT/tmp")

echo ""
echo "################################  CANN-BENCH ST  ################################"
python -m pytest "${PYTEST_ARGS[@]}"
rc=$?
# single-run 集成口径: 整个选中子集只产一份 eval_*.{json,md,html}(含全部算子)。
# 路径经环境变量传入,不内插进 python -c 字符串 (否则恶意 ST_OUT 可注入任意代码)。
n=$(PYTHONPATH=tests/st ST_TMP="$ST_OUT/tmp" ST_OUT_DIR="$ST_OUT" python3 -c \
  "import os; from harness.report import collect_artifacts as c; print(c(os.environ['ST_TMP'], os.environ['ST_OUT_DIR']))" \
  2>/dev/null || echo 0)
rm -rf "$ST_OUT/tmp"
echo "################################################################################"
if [ "$rc" -eq 0 ]; then
  echo "##  CANN-BENCH ST PASSED (rc=0) -- artifacts: $ST_OUT (${n} report)"
else
  echo "##  CANN-BENCH ST FAILED (rc=${rc}) -- 见上方 pytest 输出 -- artifacts: $ST_OUT (${n} report)"
fi
echo "PYTEST_RC=${rc}  artifacts: $ST_OUT (${n} report)"   # 机器可解析行,保持不变
echo "################################################################################"
exit "${rc}"
