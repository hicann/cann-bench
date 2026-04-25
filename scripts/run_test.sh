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
# 统一的测试运行脚本（简化版）
#
# 使用 run_simple.py 直接执行测试，避免 pytest 的参数化收集开销
#
# 用法:
#   ./run_test.sh                          # 运行所有测试
#   ./run_test.sh --cpu                    # CPU 设备测试
#   ./run_test.sh --npu                    # NPU 设备测试
#   ./run_test.sh --level 1                # Level 1 测试
#   ./run_test.sh --operator gelu          # gelu 算子测试
#   ./run_test.sh --cpu --level 2 -v       # CPU Level 2 测试，详细输出
#
# 详细帮助: ./run_test.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
TEST_DIR="${PROJECT_DIR}/tests"

# 默认配置
DEVICE=""
LEVEL=""
OPERATOR=""
CASE_ID=""
PROFILE=false
VERBOSE=false
OUTPUT=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            python "${TEST_DIR}/run_simple.py" --help
            exit 0
            ;;
        --cpu|--npu)
            if [[ -n "${DEVICE}" ]]; then
                echo "错误: --cpu 和 --npu 不能同时使用"
                exit 1
            fi
            DEVICE="$1"
            shift
            ;;
        --level)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "错误: --level 需要参数 (1, 2, 3, 4)"
                exit 1
            fi
            LEVEL="$2"
            shift 2
            ;;
        --operator)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "错误: --operator 需要算子名称参数"
                exit 1
            fi
            OPERATOR="$2"
            shift 2
            ;;
        --case-id)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "错误: --case-id 需要用例编号参数"
                exit 1
            fi
            CASE_ID="$2"
            shift 2
            ;;
        --prof)
            PROFILE=true
            shift
            ;;
        --output|-o)
            if [[ -z "$2" || "$2" =~ ^- ]]; then
                echo "错误: --output 需要文件路径参数"
                exit 1
            fi
            OUTPUT="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "未知参数: $1"
            python "${TEST_DIR}/run_simple.py" --help
            exit 1
            ;;
    esac
done

# 依赖检查
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 python 命令，请先安装 Python 3"
    exit 1
fi

if [[ ! -f "${TEST_DIR}/run_simple.py" ]]; then
    echo "错误: run_simple.py 不存在: ${TEST_DIR}/run_simple.py"
    exit 1
fi

if [[ ! -d "${PROJECT_DIR}/kernel_bench" ]]; then
    echo "错误: kernel_bench 目录不存在: ${PROJECT_DIR}/kernel_bench"
    exit 1
fi

# 打印配置
echo "========================================"
echo "设备: ${DEVICE:-cpu (默认)}"
if [[ -n "${LEVEL}" ]]; then
    echo "级别: Level ${LEVEL}"
fi
if [[ -n "${OPERATOR}" ]]; then
    echo "算子: ${OPERATOR}"
fi
if [[ -n "${CASE_ID}" ]]; then
    echo "用例ID: ${CASE_ID}"
fi
if [[ "${PROFILE}" == "true" ]]; then
    echo "性能采集: 已启用"
fi
if [[ "${VERBOSE}" == "true" ]]; then
    echo "输出模式: 详细"
fi
echo "========================================"

# 使用数组构建参数，避免 word splitting 和 glob 展开
args=()
if [[ -n "${DEVICE}" ]]; then
    args+=("${DEVICE}")
fi
if [[ -n "${LEVEL}" ]]; then
    args+=("--level" "${LEVEL}")
fi
if [[ -n "${OPERATOR}" ]]; then
    args+=("--operator" "${OPERATOR}")
fi
if [[ -n "${CASE_ID}" ]]; then
    args+=("--case-id" "${CASE_ID}")
fi
if [[ "${PROFILE}" == "true" ]]; then
    args+=(--prof)
fi
if [[ "${VERBOSE}" == "true" ]]; then
    args+=(-v)
fi
if [[ -n "${OUTPUT}" ]]; then
    args+=("--output" "${OUTPUT}")
fi

# 执行测试
cd "${TEST_DIR}" || exit 1
python run_simple.py "${args[@]}"
