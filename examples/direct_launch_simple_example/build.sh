#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------
# Build wheel package and optionally install

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

INSTALL=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --install)
            INSTALL=true
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=== Building cann_bench_ops wheel package ==="

# Clean dist directory
DIST_DIR="${SCRIPT_DIR}/dist"
rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# Build wheel
bash "${SCRIPT_DIR}/scripts/build_wheel.sh"

if [[ "${INSTALL}" == "true" ]]; then
    echo "=== Installing wheel package ==="
    pip install ${DIST_DIR}/cann_bench_ops*.whl --force-reinstall --no-deps
fi

echo ""
echo "=== Build complete ==="
ls -la "${DIST_DIR}"