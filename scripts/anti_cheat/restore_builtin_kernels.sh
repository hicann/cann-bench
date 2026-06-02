#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Undo disable_builtin_kernels.sh: restore every backed-up built-in kernel binary dir.
#
# Usage:
#   bash scripts/anti_cheat/restore_builtin_kernels.sh [--soc=<dir>] [--backup-dir=<dir>] [--dry-run]
#
# --soc selects the kernel tree subdir under built-in/op_impl/ai_core/tbe/kernel/.
# When omitted, it is auto-detected from the live chip via `acl.get_soc_name()`
# to stay symmetric with disable_builtin_kernels.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOC=""
BACKUP_DIR="${CANN_BENCH_KERNEL_BACKUP:-$HOME/.cann_bench_kernel_backup}"
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --soc=*)        SOC="${arg#*=}";;
    --backup-dir=*) BACKUP_DIR="${arg#*=}";;
    --dry-run)      DRY_RUN=1;;
    *) echo "unknown arg: $arg"; exit 1;;
  esac
done

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
    echo "        Source CANN set_env.sh or pass --soc explicitly." >&2
    exit 1
  fi
fi

: "${ASCEND_OPP_PATH:?ASCEND_OPP_PATH not set}"
KROOT="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel/${SOC}"
BK="${BACKUP_DIR}/disabled_kernels/${SOC}"
MANIFEST="${BK}/.manifest.txt"
[[ -f "$MANIFEST" ]] || { echo "no manifest at ${MANIFEST}; nothing to restore"; exit 0; }

echo "restoring from ${BK} -> ${KROOT} (dry_run=${DRY_RUN})"
echo "------------------------------------------------------------"
restored=0
while IFS= read -r rel; do
  [[ -z "$rel" ]] && continue
  src="${BK}/${rel}"; dst="${KROOT}/${rel}"
  [[ -e "$src" ]] || { echo "  backup missing: ${rel}"; continue; }
  if [[ $DRY_RUN -eq 1 ]]; then echo "  would restore: ${rel}"; restored=$((restored+1)); continue; fi
  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" ]]; then
    # 原位置已存在(可能 CANN 已重装/已恢复)，跳过以免覆盖现有内容（不使用 rm）。
    echo "  [WARN] 原位置已存在，跳过以免覆盖: ${rel}"
    continue
  fi
  mv "$src" "$dst"      # 用 mv：把备份移回原位（不使用 rm）
  echo "  restored (moved back): ${rel}"
  restored=$((restored+1))
done < "$MANIFEST"
echo "------------------------------------------------------------"
echo "restored=${restored}"
[[ $DRY_RUN -eq 0 ]] && { : > "${MANIFEST}"; echo "manifest cleared"; }
