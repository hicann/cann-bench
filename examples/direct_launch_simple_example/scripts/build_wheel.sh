#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Build Python wheel package for direct_launch_simple_example

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR}")"
DIST_DIR="${PROJECT_DIR}/dist"

echo "=== Building wheel package ==="

# Clean previous builds
rm -rf "${PROJECT_DIR}/build" "${PROJECT_DIR}/*.egg-info"

# Ensure dist directory exists
mkdir -p "${DIST_DIR}"

# Build wheel
cd "${PROJECT_DIR}"
python3 -m build --wheel --no-isolation --outdir "${DIST_DIR}"

# Find and display wheel
WHEEL_FILE=$(find "${DIST_DIR}" -name "*.whl" -type f | head -1)
if [[ -z "${WHEEL_FILE}" ]]; then
    echo "ERROR: No wheel package found"
    exit 1
fi

echo "=== Wheel package built successfully ==="
echo "Output: ${WHEEL_FILE}"