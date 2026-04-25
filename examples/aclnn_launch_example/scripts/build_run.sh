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
# Build ACLNN run package

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DIST_DIR="${PROJECT_DIR}/dist"

# Parse arguments
SOC_VERSION="ascend910b"
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

echo "=== Building ACLNN run package for ${SOC_VERSION} ==="

# Clean build directory
BUILD_DIR="${PROJECT_DIR}/build"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

# Ensure dist directory exists
mkdir -p "${DIST_DIR}"

# Configure CMake
cmake -S "${PROJECT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_COMPUTE_UNIT=${SOC_VERSION}

# Build libraries
cmake --build "${BUILD_DIR}" --parallel "$(nproc)"

# Build kernel binary (required for run package)
echo "Building kernel binaries..."
cmake --build "${BUILD_DIR}" --target binary --parallel "$(nproc)" || {
    echo "WARNING: Kernel binary build may have errors, continuing with package creation..."
}

# Build package
cmake --build "${BUILD_DIR}" --target package || {
    echo "WARNING: Package build may have errors due to missing kernel binaries..."
}

# Find run package (check both build and dist directories)
RUN_PACKAGE=$(find "${DIST_DIR}" -name "*.run" -type f 2>/dev/null | head -1)
if [[ -z "${RUN_PACKAGE}" ]]; then
    RUN_PACKAGE=$(find "${BUILD_DIR}" -name "*.run" -type f 2>/dev/null | head -1)
fi

if [[ -z "${RUN_PACKAGE}" ]]; then
    echo "ERROR: No run package found"
    echo "Please check build errors above"
    exit 1
fi

# Generate versioned filename if not already named correctly
RUN_BASENAME=$(basename "${RUN_PACKAGE}")
VERSION=$(date +%Y%m%d)
RUN_FILENAME="cann_bench_${VERSION}_linux_aarch64.run"

if [[ ! "${RUN_BASENAME}" =~ ^cann_bench.*\.run$ ]]; then
    cp "${RUN_PACKAGE}" "${DIST_DIR}/${RUN_FILENAME}"
    # Remove original run package to avoid duplicates
    rm -f "${RUN_PACKAGE}"
else
    # Already correctly named, rename to include platform info
    if [[ "${RUN_PACKAGE}" != "${DIST_DIR}/${RUN_FILENAME}" ]]; then
        mv "${RUN_PACKAGE}" "${DIST_DIR}/${RUN_FILENAME}"
    fi
fi

# Clean up any other run packages in dist
find "${DIST_DIR}" -name "*.run" -type f ! -name "${RUN_FILENAME}" -delete
find "${DIST_DIR}" -name "*.run.json" -type f -delete
rm -rf "${DIST_DIR}/_CPack_Packages"

echo "=== Run package built successfully ==="
echo "Output: ${DIST_DIR}/${RUN_FILENAME}"