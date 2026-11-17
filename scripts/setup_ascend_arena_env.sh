#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE="$REPO_ROOT/.ascend_arena_env"
SAVED_CONDA_BASE=""
SAVED_ENV_NAME=""
SAVED_CANN_SCRIPT=""
if [[ -r "$STATE_FILE" ]]; then
    while IFS='=' read -r key value; do
        case "$key" in
            CONDA_BASE) SAVED_CONDA_BASE="$value" ;;
            ENV_NAME) SAVED_ENV_NAME="$value" ;;
            CANN_ENV_SCRIPT) SAVED_CANN_SCRIPT="$value" ;;
        esac
    done < "$STATE_FILE"
fi

PREFERRED_CONDA_BASE="${ASCEND_ARENA_CONDA_BASE:-$SAVED_CONDA_BASE}"
ENV_NAME="${ASCEND_ARENA_ENV_NAME:-${SAVED_ENV_NAME:-ascend-arena}}"
CONDA_DIR="${PREFERRED_CONDA_BASE:-$HOME/miniconda3}"
PYPI_INDEX="${ASCEND_ARENA_PYPI_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
if [[ -z "${CANN_ENV_SCRIPT:-}" && -n "$SAVED_CANN_SCRIPT" ]]; then
    export CANN_ENV_SCRIPT="$SAVED_CANN_SCRIPT"
fi

trap 'echo "[ERROR] 环境安装失败，请查看上方第一条错误信息。" >&2' ERR

if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "请在完整的 cann-bench-optimize 仓库中运行本脚本。" >&2
    exit 1
fi

echo "[1/6] 检查服务器架构"
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|aarch64) ;;
    *) echo "不支持的服务器架构：$ARCH" >&2; exit 1 ;;
esac

echo "[2/6] 准备 Miniconda"
NEED_CONDA_INSTALL=0
if [[ -n "$PREFERRED_CONDA_BASE" ]]; then
    [[ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]] || NEED_CONDA_INSTALL=1
elif command -v conda >/dev/null 2>&1; then
    CONDA_DIR="$(conda info --base)"
elif [[ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
    :
else
    NEED_CONDA_INSTALL=1
fi

if (( NEED_CONDA_INSTALL )); then
    if command -v curl >/dev/null 2>&1; then
        INSTALLER="$(mktemp "/tmp/miniconda-${ARCH}.XXXXXX.sh")"
        trap 'rm -f "${INSTALLER:-}"' EXIT
        curl -fL --progress-bar \
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh" \
            -o "$INSTALLER"
    elif command -v wget >/dev/null 2>&1; then
        INSTALLER="$(mktemp "/tmp/miniconda-${ARCH}.XXXXXX.sh")"
        trap 'rm -f "${INSTALLER:-}"' EXIT
        wget --show-progress \
            "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-${ARCH}.sh" \
            -O "$INSTALLER"
    else
        echo "安装 Miniconda 需要 curl 或 wget，请联系管理员安装其中一个。" >&2
        exit 1
    fi
    bash "$INSTALLER" -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "[3/6] 创建 Conda 环境：$ENV_NAME"
CONDA_PACKAGES=(python=3.10 pip git curl 'cmake<4' ninja make 'libstdcxx-ng>=14')
if [[ ! -x /usr/bin/gcc || ! -x /usr/bin/g++ ]]; then
    echo "未检测到系统 GCC/G++，将安装 Conda 编译器"
    CONDA_PACKAGES+=(c-compiler cxx-compiler)
fi
if conda env list | awk -v name="$ENV_NAME" '$1 == name { found=1 } END { exit !found }'; then
    conda install -n "$ENV_NAME" --override-channels -c conda-forge -y "${CONDA_PACKAGES[@]}"
else
    conda create -n "$ENV_NAME" --override-channels -c conda-forge -y "${CONDA_PACKAGES[@]}"
fi

echo "[4/6] 加载 Conda 与 CANN"
export ASCEND_ARENA_CONDA_BASE="$CONDA_DIR"
export ASCEND_ARENA_ENV_NAME="$ENV_NAME"
source "$REPO_ROOT/scripts/activate_ascend_arena_env.sh"

echo "[5/6] 安装项目运行依赖"
env -u PYTHONPATH PYTHONNOUSERSITE=1 python -m pip install \
    --upgrade --index-url "$PYPI_INDEX" pip
env -u PYTHONPATH PYTHONNOUSERSITE=1 python -m pip install \
    --index-url "$PYPI_INDEX" "$REPO_ROOT"
env -u PYTHONPATH PYTHONNOUSERSITE=1 python -m pip check

echo "[6/6] 检查 NPU"
command -v npu-smi >/dev/null 2>&1 || {
    echo "未找到 npu-smi，请先安装 Ascend 驱动。" >&2
    exit 1
}
npu-smi info >/dev/null
python - <<'PY'
import torch
import torch_npu
import sqlite3

count = torch.npu.device_count()
assert count > 0, "没有发现可用 NPU"
x = torch.tensor([1.0, 2.0], device="npu:0")
torch.testing.assert_close((x + 1).cpu(), torch.tensor([2.0, 3.0]))
torch.npu.synchronize()
print(
    f"NPU 环境正常，共发现 {count} 张卡；"
    f"torch={torch.__version__}，torch_npu={torch_npu.__version__}"
)
PY

(
    umask 077
    printf 'CONDA_BASE=%s\nENV_NAME=%s\nCANN_ENV_SCRIPT=%s\n' \
        "$CONDA_DIR" "$ENV_NAME" "$CANN_ENV_SCRIPT" > "$STATE_FILE"
)

echo
echo "安装完成。新终端中运行："
echo "  cd $REPO_ROOT"
echo "  source scripts/activate_ascend_arena_env.sh"
