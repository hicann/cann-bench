#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Benchmark-integrity: disable the built-in AiCore *operator kernel binaries* for every benchmarked op,
# so a submission cannot "cheat" by launching the stock kernel (via aclnn<Op> / ADD_TO_LAUNCHER_LIST_AICORE)
# instead of providing its own kernel.
#
# Only the prebuilt kernel binary dir (kernel/<soc>/<cat>/<op>/  -> *.o + *.json) is moved out of OPP.
# The TBE/AscendC impl sources, op_proto and registration are LEFT INTACT, and torch_npu is unaffected.
# Device-side AscendC intrinsics (AscendC::Add/Mul/Exp/Mmad/...) compile into the candidate kernel and are
# NOT affected by removing operator binaries.
#
# PROTECTED (never listed): MatMul*/ReduceMax* (used by perf_eval freq-boost / L2-flush) + generic primitives.
#
# Implemented with `mv` (NOT `rm`): each kernel dir is MOVED into the backup dir — that single step both
# backs it up and removes it from OPP, so no data is ever deleted. Run restore_builtin_kernels.sh to undo.
#
# Usage:
#   bash scripts/anti_cheat/disable_builtin_kernels.sh [--soc=<dir>] [--list=<file>] [--backup-dir=<dir>] [--dry-run]
#
# --soc selects the kernel tree subdir under built-in/op_impl/ai_core/tbe/kernel/
# (e.g. ascend910b, ascend910_93, ascend910_95). When omitted, it is auto-detected
# from the live chip via `acl.get_soc_name()`; falling back fails loud rather than
# silently disabling kernels for the wrong SOC.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOC=""
LIST="${SCRIPT_DIR}/benchmarked_kernels.txt"
BACKUP_DIR="${CANN_BENCH_KERNEL_BACKUP:-$HOME/.cann_bench_kernel_backup}"
DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --soc=*)        SOC="${arg#*=}";;
    --list=*)       LIST="${arg#*=}";;
    --backup-dir=*) BACKUP_DIR="${arg#*=}";;
    --dry-run)      DRY_RUN=1;;
    --yes|-y)       ASSUME_YES=1;;
    *) echo "unknown arg: $arg"; exit 1;;
  esac
done

# soc_name_to_kernel_dir <Ascend chip name> → kernel subdir under kernel/<soc>/.
soc_name_to_kernel_dir() {
  case "$1" in
    Ascend910_93*) echo ascend910_93 ;;
    Ascend910_95*) echo ascend910_95 ;;
    Ascend910B*)   echo ascend910b ;;
    Ascend310P*)   echo ascend310p ;;
    *)             return 1 ;;
  esac
}

if [[ -z "$SOC" ]]; then
  # Detect from the live chip. Fail loud — operating with the wrong default SOC
  # silently leaves the actual runtime kernel tree untouched (the failure mode
  # users hit when ascend910b is hardcoded on a 910C / Ascend910_9362 host).
  soc_name="$(python3 -c 'import acl,sys; n=acl.get_soc_name(); sys.stdout.write(n or "")' 2>/dev/null || true)"
  if [[ -n "$soc_name" ]]; then
    if SOC="$(soc_name_to_kernel_dir "$soc_name")"; then
      echo "[INFO] auto-detected SOC=${SOC} from chip ${soc_name}"
    else
      echo "[ERROR] could not map chip name '${soc_name}' to a kernel SOC dir." >&2
      echo "        Pass --soc=<dir> explicitly (e.g. --soc=ascend910b)." >&2
      exit 1
    fi
  else
    echo "[ERROR] --soc not given and acl.get_soc_name() failed." >&2
    echo "        Source CANN set_env.sh (so pyACL is importable) or pass --soc explicitly." >&2
    exit 1
  fi
fi

: "${ASCEND_OPP_PATH:?ASCEND_OPP_PATH not set (source the CANN set_env.sh)}"
KROOT="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel/${SOC}"
[[ -d "$KROOT" ]] || { echo "kernel root not found: $KROOT"; exit 1; }
[[ -f "$LIST" ]]  || { echo "list not found: $LIST"; exit 1; }

# Sanity: if the SOC kernel tree is essentially empty (no ops_legacy / ops_nn),
# the CANN install probably doesn't ship binaries for this SOC (e.g. a 910b-only
# ops package on a 910C host). Disabling here is a no-op — but it silently
# "succeeds" and masks the real install gap. Refuse instead.
have_core_op_dirs=0
for sub in ops_legacy ops_nn ops_math ops_cv ops_transformer; do
  [[ -d "$KROOT/$sub" ]] && have_core_op_dirs=$((have_core_op_dirs+1))
done
if [[ $have_core_op_dirs -eq 0 ]]; then
  echo "[ERROR] ${KROOT} has no ops_legacy/ops_nn/... subdirs — this CANN install" >&2
  echo "        ships no built-in kernels for SOC=${SOC}. Install the matching" >&2
  echo "        ops package (e.g. Ascend-cann-A3-ops for ascend910_93) before" >&2
  echo "        running this script." >&2
  exit 1
fi

BK="${BACKUP_DIR}/disabled_kernels/${SOC}"
mkdir -p "$BK"
MANIFEST="${BK}/.manifest.txt"

echo "SOC=${SOC}"
echo "KROOT=${KROOT}"
echo "BACKUP=${BK}"
echo "DRY_RUN=${DRY_RUN}"
echo "------------------------------------------------------------"

# 风险确认：本脚本会修改系统 CANN 安装，绝不应在无人值守 / 共享机器上误触发。
if [[ $DRY_RUN -eq 0 ]]; then
  cat <<'WARN'
=============================== ⚠️  风险提示（务必阅读） ===============================
本操作会把系统 CANN 安装目录 (ASCEND_OPP_PATH) 下【被评测算子】的内置 kernel 二进制
用 mv 移动到 BACKUP 目录（即从 OPP 移除，不使用 rm，数据不会被删除）：
  * 修改的是【全局共享】的 CANN 安装，会影响本机所有进程 / 用户 / 其它项目；
  * 改动持续存在（不随进程退出自动恢复），仅能通过 restore_builtin_kernels.sh 还原；
  * 移除后，torch_npu 等在 NPU 上调用这些算子会直接报错（找不到 kernel）。
强烈建议：仅在【一次性 docker 容器 / 专用评测机】中执行，不要在共享开发机上直接运行。
本操作可逆：每个目录都是 mv 到 BACKUP 目录，可用 restore 脚本一键移回。
=====================================================================================
WARN
  if [[ $ASSUME_YES -eq 1 ]]; then
    echo "[--yes] 已确认，继续执行。"
  elif [[ -t 0 ]]; then
    read -r -p '确认删除以上内置 kernel？请输入大写 DELETE 以继续： ' _ans
    [[ "${_ans:-}" == "DELETE" ]] || { echo "未确认（输入非 DELETE），已取消。"; exit 1; }
  else
    echo "[ERROR] 非交互环境（无 TTY）且未指定 --yes：为防止误删已中止。" >&2
    echo "        如确需在脚本/容器中自动执行，请显式追加 --yes。" >&2
    exit 1
  fi
fi

removed=0; missing=0; already=0
while IFS= read -r rel; do
  rel="${rel%%#*}"; rel="$(echo "$rel" | xargs)"   # strip comments / whitespace
  [[ -z "$rel" ]] && continue
  src="${KROOT}/${rel}"
  dst="${BK}/${rel}"
  if [[ ! -e "$src" ]]; then
    if [[ -e "$dst" ]]; then already=$((already+1));   # already disabled (backup present)
    else echo "  MISSING (not present): ${rel}"; missing=$((missing+1)); fi
    continue
  fi
  if [[ $DRY_RUN -eq 1 ]]; then echo "  would disable: ${rel}"; removed=$((removed+1)); continue; fi
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    # 备份已存在：保护原始备份，不覆盖、不动当前 src（避免任何数据丢失）。
    echo "  [WARN] 备份已存在，跳过(保护原始备份；如需重新禁用请先 restore): ${rel}"
    already=$((already+1))
    continue
  fi
  mv "$src" "$dst"      # 用 mv：把内置 kernel 移到备份目录 = 备份 + 从 OPP 移除（不使用 rm）
  echo "${rel}" >> "$MANIFEST"
  echo "  disabled (moved to backup): ${rel}"
  removed=$((removed+1))
done < "$LIST"

echo "------------------------------------------------------------"
echo "disabled=${removed}  already-disabled=${already}  missing=${missing}"
[[ $DRY_RUN -eq 0 ]] && echo "manifest: ${MANIFEST}"
echo "restore with: bash ${SCRIPT_DIR}/restore_builtin_kernels.sh --soc=${SOC} --backup-dir=${BACKUP_DIR}"
