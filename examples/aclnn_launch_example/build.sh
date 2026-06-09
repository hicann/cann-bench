#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------
# Build both run package and wheel package

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# Auto-detect SoC version from npu-smi if not specified
detect_soc_version() {
    # 优先通过 torch_npu runtime 获取完整 SoC 名称（含子型号，如 Ascend910_9362）
    # npu-smi info 只报告基类名（如 Ascend910），无法区分 910B 和 910_93 子型号
    local torch_soc=$(python3 -c "
import torch, torch_npu
print(torch.npu.get_device_name(0))
" 2>/dev/null)

    if [ -n "${torch_soc}" ]; then
        # 注意: Ascend910_93* 模式会误匹配 Ascend910_9361/9362（它们是 910B 而非 910_93）
        # 必须将 910B 的产品 ID（936x 系列）放在 910_93 通配之前
        case "${torch_soc}" in
            Ascend910B*)     echo "ascend910b" ; return ;;
            Ascend910_936*)  echo "ascend910b" ; return ;;  # 9361=B1, 9362=B2, 等 910B 产品 ID
            Ascend910_93*)   echo "ascend910_93" ; return ;;
            Ascend950*)      echo "ascend950" ; return ;;
        esac
    fi

    # 兜底: npu-smi info（仅基类名，无法区分子型号时返回空）
    local npu_name=$(npu-smi info 2>/dev/null | grep -oP 'Ascend\S+' | head -1)
    case "${npu_name}" in
        Ascend910B1|Ascend910B2|Ascend910B3|Ascend910B4) echo "ascend910b" ;;
        Ascend910_93*)  echo "ascend910_93" ;;
        Ascend950*)     echo "ascend950" ;;
        *)              echo "" ;;
    esac
}

SOC_VERSION=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --soc=*)
            SOC_VERSION="${1#*=}"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ -z "${SOC_VERSION}" ]; then
    SOC_VERSION=$(detect_soc_version)
    if [ -z "${SOC_VERSION}" ]; then
        echo "[ERROR] Cannot detect SoC version. Use --soc=<soc_version> to specify."
        exit 1
    fi
    echo "[INFO] Auto-detected SoC: ${SOC_VERSION}"
fi

echo "=== Building cann_bench packages ==="
echo "SOC: ${SOC_VERSION}"

# Clean dist directory
DIST_DIR="${PROJECT_DIR}/dist"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# Build run package
bash "${PROJECT_DIR}/scripts/build_run.sh" --soc=${SOC_VERSION}

# Build wheel package
bash "${PROJECT_DIR}/scripts/build_wheel.sh"

echo ""
echo "=== Build complete ==="
echo "Output directory: ${DIST_DIR}"
ls -la "${DIST_DIR}"