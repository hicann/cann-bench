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
# 测试运行脚本
#
# 用法:
#   ./run_test.sh --cpu --operator Sigmoid          # CPU 简单验证
#   ./run_test.sh --npu --operator Scatter          # NPU 模拟评测（标准流程，采集性能）
#   ./run_test.sh --npu --task-dir kernel_bench/level2/scatter  # 指定算子目录
#   ./run_test.sh --npu --device-id 1 --export-baseline baseline.json
#
# 详细帮助: ./run_test.sh --help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
TEST_DIR="${PROJECT_DIR}/tests"

# 默认配置
DEVICE=""
DEVICE_ID=""  # 空表示多卡模式，指定值表示单卡模式
BENCH_DIR=""   # 替代原来的 LEVEL
OPERATOR=""
CASE_ID=""
CASE_TIMEOUT_SEC=""
VERBOSE=false
OUTPUT=""
EXPORT_BASELINE=""
NO_PERF=false

# 多进程并行配置（统一架构）
PROCESSES_PER_CARD=2
TIMEOUT_PER_OPERATOR=300

show_help() {
    cat << 'EOF'
测试运行脚本

用法: ./run_test.sh --cpu|--npu [选项]

模式选项（必选其一）:
    --cpu                   CPU 简单验证模式
    --npu                   NPU 进程池评测模式

设备配置（仅 NPU 模式）:
    --device-id <id>        指定 NPU 设备 ID（单卡模式）
                            不指定则自动使用全部可用卡（多卡并行模式）

目录配置:
    --task-dir <path>            指定评测目录（bench根目录或算子目录）
                            默认: kernel_bench
                            支持: kernel_bench, kernel_bench/level2/scatter 等

多进程并行配置（统一架构）:
    --processes-per-card <n> 每卡进程数（默认: 2）
    --timeout-per-operator <n> 单算子超时（秒，默认: 300）。进程总超时 = 算子数 × timeout_per_operator

用例筛选:
    --operator <name>       按算子名称筛选
    --case-id <id>          按用例编号筛选
    --case-timeout-sec <n>  用例超时时间（秒），超时则标记失败继续下一用例

性能配置（仅 NPU 模式）:
    --warmup <n>            预热次数 (默认: 3)
    --repeat <n>            采集次数 (默认: 5)
    --no-perf               关闭性能采集，仅做精度验证
    --export-baseline <p>   导出性能基线到 JSON

输出选项:
    --output <path>         结果输出文件路径
    -v, --verbose           详细输出模式（显示第三方库日志）
    --help, -h              显示此帮助信息

示例:
    # CPU 验证（默认 kernel_bench）
    ./run_test.sh --cpu --operator Sigmoid

    # NPU 单卡评测（指定设备）
    ./run_test.sh --npu --device-id 0 --operator Scatter

    # NPU 多卡并行评测（自动检测全部卡）
    ./run_test.sh --npu --operator Scatter

    # 指定算子目录评测
    ./run_test.sh --npu --task-dir kernel_bench/level2/scatter

    # 指定自定义 bench 目录
    ./run_test.sh --npu --task-dir my_custom_bench

    # NPU 评测并导出基线
    ./run_test.sh --npu --export-baseline reports/baseline.json

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --cpu)
            if [[ -n "${DEVICE}" ]]; then
                echo "错误: --cpu 和 --npu 不能同时使用"
                exit 1
            fi
            DEVICE="--cpu"
            shift
            ;;
        --npu)
            if [[ -n "${DEVICE}" ]]; then
                echo "错误: --cpu 和 --npu 不能同时使用"
                exit 1
            fi
            DEVICE="--npu"
            shift
            ;;
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        --processes-per-card)
            PROCESSES_PER_CARD="$2"
            shift 2
            ;;
        --timeout-per-operator)
            TIMEOUT_PER_OPERATOR="$2"
            shift 2
            ;;
        --task-dir)
            BENCH_DIR="$2"
            shift 2
            ;;
        --operator)
            OPERATOR="$2"
            shift 2
            ;;
        --case-id)
            CASE_ID="$2"
            shift 2
            ;;
        --case-timeout-sec)
            CASE_TIMEOUT_SEC="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --repeat)
            REPEAT="$2"
            shift 2
            ;;
        --export-baseline)
            EXPORT_BASELINE="$2"
            shift 2
            ;;
        --no-perf)
            NO_PERF=true
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
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

if [[ -z "${DEVICE}" ]]; then
    echo "错误: 必须指定 --cpu 或 --npu"
    show_help
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "错误: 未找到 python 命令"
    exit 1
fi

if [[ ! -f "${TEST_DIR}/run_simple.py" ]]; then
    echo "错误: run_simple.py 不存在"
    exit 1
fi

echo "========================================"
if [[ "${DEVICE}" == "--cpu" ]]; then
    echo "模式: CPU 简单验证"
else
    if [[ -n "${DEVICE_ID}" ]]; then
        echo "模式: NPU 单卡进程池评测 (NPU:${DEVICE_ID})"
    else
        echo "模式: NPU 多卡进程池评测"
    fi
    echo "进程配置: ${PROCESSES_PER_CARD} 进程/卡"
    echo "单算子超时: ${TIMEOUT_PER_OPERATOR}s"
fi
if [[ -n "${BENCH_DIR}" ]]; then
    echo "目录: ${BENCH_DIR}"
else
    echo "目录: kernel_bench (默认)"
fi
if [[ -n "${OPERATOR}" ]]; then
    echo "算子: ${OPERATOR}"
fi
if [[ -n "${EXPORT_BASELINE}" && "${DEVICE}" == "--npu" ]]; then
    echo "基线导出: ${EXPORT_BASELINE}"
fi
echo "========================================"

# 构建参数
args=()
args+=("${DEVICE}")

# 单卡模式：指定 device-id；多卡模式：不指定
if [[ -n "${DEVICE_ID}" ]]; then
    args+=("--device-id" "${DEVICE_ID}")
fi

# 多进程并行参数
args+=("--processes-per-card" "${PROCESSES_PER_CARD}")
args+=("--timeout-per-operator" "${TIMEOUT_PER_OPERATOR}")

# 目录参数（替代 --level）
if [[ -n "${BENCH_DIR}" ]]; then
    args+=("--task-dir" "${BENCH_DIR}")
fi
if [[ -n "${OPERATOR}" ]]; then
    args+=("--operator" "${OPERATOR}")
fi
if [[ -n "${CASE_ID}" ]]; then
    args+=("--case-id" "${CASE_ID}")
fi
if [[ -n "${CASE_TIMEOUT_SEC}" ]]; then
    args+=("--case-timeout-sec" "${CASE_TIMEOUT_SEC}")
fi
if [[ -n "${WARMUP}" ]]; then
    args+=("--warmup" "${WARMUP}")
fi
if [[ -n "${REPEAT}" ]]; then
    args+=("--repeat" "${REPEAT}")
fi
if [[ "${VERBOSE}" == "true" ]]; then
    args+=(-v)
fi
if [[ "${NO_PERF}" == "true" ]]; then
    args+=("--no-perf")
fi
if [[ -n "${OUTPUT}" ]]; then
    args+=("--output" "${OUTPUT}")
fi
if [[ -n "${EXPORT_BASELINE}" ]]; then
    args+=("--export-baseline" "${EXPORT_BASELINE}")
fi

# 执行
cd "${TEST_DIR}" || exit 1
python run_simple.py "${args[@]}"