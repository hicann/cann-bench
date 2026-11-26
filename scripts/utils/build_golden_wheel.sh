#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# Golden whl 打包脚本
#
# 将 tasks/ 下所有 golden.py 收集到纯 Python cann_bench whl 包中，
# 注册到 torch.ops.cann_bench 命名空间，
# 使 run_evaluation.sh 能将 golden 作为"AI算子"加载评测，
# 验证 golden(NPU) 与 golden(CPU fp64) 的精度一致性。
#
# 用法:
#   OP_NAME 为所选目录 proto.yaml 的 operator.name；TASK_DIR 为该算子的实际目录。
#   ./scripts/utils/build_golden_wheel.sh                     # 构建所有算子
#   ./scripts/utils/build_golden_wheel.sh --operator "$OP_NAME" # 只打包指定算子
#   ./scripts/utils/build_golden_wheel.sh --level 1           # 只打包 level1
#   ./scripts/utils/build_golden_wheel.sh --install           # 构建后自动安装
#   ./scripts/utils/build_golden_wheel.sh --clean             # 清理构建临时目录
#
# 评测流程:
#   1. ./scripts/utils/build_golden_wheel.sh --install
#   2. ./scripts/run_evaluation.sh --task-dir "$TASK_DIR" --no-perf

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
UTILS_DIR="${SCRIPT_DIR}"

# 默认配置
OPERATORS=()
LEVEL=""
TASK_DIR=""
OUTPUT_DIR=""
INSTALL=false
CLEAN=false
VERBOSE=false

show_help() {
    cat << 'EOF'
Golden whl 打包脚本

将 tasks/ 下 golden.py 函数收集打包成纯 Python cann_bench whl 包，
注册到 torch.ops.cann_bench 命名空间，使 run_evaluation.sh 能作为"AI算子"评测。

用法: ./scripts/utils/build_golden_wheel.sh [选项]

选项:
  --operator <op_name>    只打包指定算子（proto.yaml 的 operator.name），可多次指定
  --level <N>             只打包指定级别 (取值取决于 --task-dir 下实际存在的 levelN 目录)
  --task-dir <path>       指定 tasks 目录（默认: 项目根目录下的 tasks）
  --output-dir <path>     输出目录（默认: dist/golden_wheel）
  --install               构建后自动安装（卸载旧版本再安装）
  --clean                 清理构建临时目录
  -v, --verbose           详细输出
  -h, --help              显示此帮助信息

示例:
  # 先查看当前题集，以下占位值须替换为实际名称和目录后再执行
  ./scripts/run_evaluation.sh -a list
  OP_NAME="<op_name>"                # 所选目录 proto.yaml 的 operator.name
  TASK_DIR="tasks/level<N>/<op_dir>"  # 与 OP_NAME 对应的实际算子目录

  # 构建所有算子的 golden whl
  ./scripts/utils/build_golden_wheel.sh

  # 只打包所选算子并安装
  ./scripts/utils/build_golden_wheel.sh --operator "$OP_NAME" --install

  # 只打包 level1 算子
  ./scripts/utils/build_golden_wheel.sh --level 1

  # 构建后通过 run_evaluation 评测
  ./scripts/utils/build_golden_wheel.sh --install
  ./scripts/run_evaluation.sh --task-dir "$TASK_DIR" --no-perf

EOF
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --operator)
            OPERATORS+=("$2")
            shift 2
            ;;
        --level)
            LEVEL="$2"
            shift 2
            ;;
        --task-dir)
            TASK_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --install)
            INSTALL=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "========================================"
echo "Golden whl 打包"
echo "========================================"

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 python 命令"
    exit 1
fi

# 构建参数
ARGS=()

if [[ -n "${TASK_DIR}" ]]; then
    ARGS+=("--task-dir" "${TASK_DIR}")
fi

if [[ "${#OPERATORS[@]}" -gt 0 ]]; then
    ARGS+=("--operator" "${OPERATORS[@]}")
fi

if [[ -n "${LEVEL}" ]]; then
    ARGS+=("--level" "${LEVEL}")
fi

if [[ -n "${OUTPUT_DIR}" ]]; then
    ARGS+=("--output-dir" "${OUTPUT_DIR}")
fi

if [[ "${INSTALL}" == "true" ]]; then
    ARGS+=("--install")
fi

if [[ "${CLEAN}" == "true" ]]; then
    ARGS+=("--clean")
fi

if [[ "${VERBOSE}" == "true" ]]; then
    ARGS+=("-v")
fi

# 执行 Python 脚本
PYTHON_SCRIPT="${UTILS_DIR}/build_golden_wheel.py"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "错误: Python 脚本不存在: ${PYTHON_SCRIPT}"
    exit 1
fi

# 确保 PyYAML 可用
python -c "import yaml" 2>/dev/null || {
    echo "[INFO] 安装 PyYAML..."
    pip install pyyaml --quiet
}

# 确保 python-build 可用
python -c "import build" 2>/dev/null || {
    echo "[INFO] 安装 python-build..."
    pip install build --quiet
}

echo "[INFO] 执行: python ${PYTHON_SCRIPT} ${ARGS[*]}"
python "${PYTHON_SCRIPT}" "${ARGS[@]}"
