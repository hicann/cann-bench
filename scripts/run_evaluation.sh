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
# Kernel Bench 评测脚本
#
# 用法:
#   ./scripts/run_evaluation.sh /path/to/ai_ops                    # 从源码目录评测
#   ./scripts/run_evaluation.sh --source-dir /path/to/ai_ops       # 同上（显式指定）
#   ./scripts/run_evaluation.sh --task-dir tasks/level1/exp      # 指定评测目录
#   ./scripts/run_evaluation.sh --operator Exp                     # 按算子名称筛选
#   ./scripts/run_evaluation.sh --device-id 0                      # 单卡模式
#   ./scripts/run_evaluation.sh                                    # 多卡并行模式（自动检测）
#   ./scripts/run_evaluation.sh --no-perf                          # 仅精度验证
#
# 详细帮助: ./scripts/run_evaluation.sh --help

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
TASKS_ROOT="${PROJECT_ROOT}/tasks"
# REPORTS_DIR 默认值在 main() 中根据 BENCH_NAME 动态设置

# 默认配置
SOURCE_DIR=""
TASK_DIR=""
BENCH_NAME="cann"  # 评测集名称：cann 或 stanford
OPERATOR=""
CASE_ID=""
DEVICE_ID=""  # 空表示多卡模式，指定值表示单卡模式
VERBOSE=false
ACTION="eval"
STAGED_EVAL=true

# 多进程并行配置（统一架构）
PROCESSES_PER_CARD=2
TIMEOUT_PER_PROCESS=300

# 性能配置
WARMUP=3
REPEAT=5
NO_PERF=false
PROFILER_LEVEL="Level1"
EVAL_SEED=""  # 评测种子（默认: 0 = 确定性，-1 = 纯随机）

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_help() {
    echo "Kernel Bench 评测脚本"
    echo ""
    echo "用法: $0 [源码目录] [选项]"
    echo ""
    echo "参数:"
    echo "  源码目录                  AI生成的算子源码目录（可选，自动扫描编译安装）"
    echo "                            如果不指定 --source-dir，第一个位置参数将作为源码目录"
    echo ""
    echo "选项:"
    echo "  -a, --action <action>     操作类型: eval(评测), list(列表), info(详情), config(配置)"
    echo "                            默认: eval"
    echo "  --source-dir <dir>        AI生成的算子源码目录（显式指定，与位置参数等效）"
    echo ""
    echo "评测集配置:"
    echo "  --bench-name <name>       评测集名称: cann, stanford（默认: cann）"
    echo "                            stanford 会自动使用 bench_lab/stanford_bench/KernelBench 目录"
    echo ""
    echo "目录配置:"
    echo "  --task-dir <path>         指定评测目录（bench根目录或算子目录）"
    echo "                            默认: tasks (cann) 或 bench_lab/stanford_bench/KernelBench/KernelBench (stanford)"
    echo "                            支持: tasks, tasks/level1, tasks/level1/exp 等"
    echo ""
    echo "设备配置:"
    echo "  --device <type>           设备类型: cpu, npu（默认: npu）"
    echo "  --device-id <id>          指定 NPU 设备 ID（单卡模式）"
    echo "                            不指定则自动使用全部可用卡（多卡并行模式）"
    echo ""
    echo "多进程并行配置:"
    echo "  --processes-per-card <n>  每卡进程数（默认: 2）"
    echo "  --timeout-per-process <n> 单进程超时（秒，默认: 300）"
    echo ""
    echo "用例筛选:"
    echo "  --operator <name>         按算子名称筛选"
    echo "  --case-id <id>            按用例编号筛选"
    echo ""
    echo "性能配置:"
    echo "  --warmup <n>              预热次数（默认: 3）"
    echo "  --repeat <n>              采集次数（默认: 5）"
    echo "  --no-perf                 关闭性能采集，仅做精度验证"
    echo "  --profiler-level <level>  Profiler 级别: Level1, Level2（默认: Level1）"
    echo "  --eval-seed <n>           输入生成确定性种子（默认: 0 = 自动确定性）。"
    echo "                            改变种子可获得不同但可复现的输入。-1 表示纯随机。"
    echo "  --single-pass             使用旧的一体化 eval 流程（默认使用编译/精度/性能三阶段）"
    echo ""
    echo "其他选项:"
    echo "  -v, --verbose             详细输出"
    echo "  -h, --help                显示帮助信息"
    echo ""
    echo "示例:"
    echo "  # CANN 评测（默认）"
    echo "  $0 --operator Exp"
    echo ""
    echo "  # StanfordBench 评测"
    echo "  $0 --bench-name stanford --operator Softmax"
    echo ""
    echo "  # StanfordBench 从源码目录评测"
    echo "  $0 --bench-name stanford --source-dir examples/stanfordbench/Softmax --operator Softmax"
    echo ""
    echo "  # 从源码目录评测（推荐）"
    echo "  $0 /path/to/ai_ops"
    echo ""
    echo "  # 指定评测目录"
    echo "  $0 --task-dir tasks/level1"
    echo ""
    echo "  # 评测单个算子目录"
    echo "  $0 --task-dir tasks/level1/exp"
    echo ""
    echo "  # 按算子名称筛选"
    echo "  $0 --operator Exp"
    echo ""
    echo "  # 单卡评测"
    echo "  $0 --device-id 0 --operator Exp"
    echo ""
    echo "  # 多卡并行评测（自动检测全部卡）"
    echo "  $0 --operator Exp"
    echo ""
    echo "  # 仅精度验证（关闭性能采集）"
    echo "  $0 --no-perf --operator Exp"
    echo ""
    echo "  # 列出算子"
    echo "  $0 -a list"
    echo ""
    echo "  # 查看算子详情"
    echo "  $0 -a info --operator Exp"
    echo ""
    echo "输出:"
    echo "  评测报告保存到: ${REPORTS_DIR}/"
    echo ""
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 解析参数
# 支持位置参数作为源码目录（第一个不以 - 开头的参数）
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--action)
            ACTION="$2"
            shift 2
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --task-dir)
            TASK_DIR="$2"
            shift 2
            ;;
        --operator)
            OPERATOR="$2"
            shift 2
            ;;
        --bench-name)
            BENCH_NAME="$2"
            shift 2
            ;;
        --case-id)
            CASE_ID="$2"
            shift 2
            ;;
        --device)
            DEVICE_TYPE="$2"
            shift 2
            ;;
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        --processes-per-card)
            PROCESSES_PER_CARD="$2"
            shift 2
            ;;
        --timeout-per-process)
            TIMEOUT_PER_PROCESS="$2"
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
        --no-perf)
            NO_PERF=true
            shift
            ;;
        --profiler-level)
            PROFILER_LEVEL="$2"
            shift 2
            ;;
        --eval-seed)
            EVAL_SEED="$2"
            shift 2
            ;;
        --single-pass)
            STAGED_EVAL=false
            shift
            ;;
        --staged)
            STAGED_EVAL=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        -*)
            log_error "未知参数: $1"
            print_help
            exit 1
            ;;
        *)
            # 位置参数，收集起来
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 处理位置参数：第一个位置参数作为源码目录（如果 --source-dir 未显式指定）
if [[ -z "${SOURCE_DIR}" && ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    SOURCE_DIR="${POSITIONAL_ARGS[0]}"
fi

# 检查Python环境
check_python() {
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    log_info "Python版本: $(python --version)"
}

# 卸载已安装的cann_bench包（避免算子重复注册冲突）
uninstall_packages() {
    for pkg in cann_bench cann_bench_golden; do
        if pip show "${pkg}" &> /dev/null; then
            log_info "卸载已安装的包: ${pkg}"
            pip uninstall "${pkg}" -y &> /dev/null || true
        fi
    done
}

# 检查tasks 数据目录
check_tasks() {
    if [[ ! -d "${TASKS_ROOT}" ]]; then
        log_error "tasks 目录不存在: ${TASKS_ROOT}"
        exit 1
    fi
    log_info "tasks 目录: ${TASKS_ROOT}"
}

# 检查 StanfordBench 数据目录
check_stanford_data() {
    KERNELBENCH_DIR="${PROJECT_ROOT}/bench_lab/stanford_bench/KernelBench"
    if [[ ! -d "${KERNELBENCH_DIR}" ]]; then
        log_info "StanfordBench 数据目录不存在，正在下载..."
        bash "${PROJECT_ROOT}/bench_lab/stanford_bench/download.sh"
        if [[ ! -d "${KERNELBENCH_DIR}" ]]; then
            log_error "下载失败: ${KERNELBENCH_DIR}"
            exit 1
        fi
    fi
    log_info "StanfordBench 数据目录: ${KERNELBENCH_DIR}"
}

# 创建报告目录
ensure_reports_dir() {
    if [[ ! -d "${REPORTS_DIR}" ]]; then
        mkdir -p "${REPORTS_DIR}"
        log_info "创建报告目录: ${REPORTS_DIR}"
    fi
}

# 构建命令参数
build_cmd_args() {
    CMD_ARGS=""

    case "${ACTION}" in
        eval)
            if [[ "${STAGED_EVAL}" == true && "${BENCH_NAME}" == "cann" && "${DEVICE_TYPE:-npu}" == "npu" ]]; then
                CMD_ARGS="staged-eval"
            else
                CMD_ARGS="eval"
            fi
            # 评测集名称（必须传递）
            CMD_ARGS="${CMD_ARGS} --bench-name ${BENCH_NAME}"
            # 报告目录（按 bench_name 分目录）
            CMD_ARGS="${CMD_ARGS} --reports-dir ${REPORTS_DIR}"
            if [[ -n "${SOURCE_DIR}" ]]; then
                CMD_ARGS="${CMD_ARGS} --source-dir ${SOURCE_DIR}"
            fi
            if [[ -n "${TASK_DIR}" ]]; then
                CMD_ARGS="${CMD_ARGS} --task-dir ${TASK_DIR}"
            fi
            if [[ -n "${OPERATOR}" ]]; then
                CMD_ARGS="${CMD_ARGS} --operator ${OPERATOR}"
            fi
            if [[ -n "${CASE_ID}" ]]; then
                CMD_ARGS="${CMD_ARGS} --case-id ${CASE_ID}"
            fi

            # 设备配置
            if [[ -n "${DEVICE_TYPE}" ]]; then
                CMD_ARGS="${CMD_ARGS} --device ${DEVICE_TYPE}"
            else
                CMD_ARGS="${CMD_ARGS} --device npu"
            fi

            # 单卡模式：指定 device-id；多卡模式：不指定
            if [[ -n "${DEVICE_ID}" ]]; then
                CMD_ARGS="${CMD_ARGS} --device-id ${DEVICE_ID}"
            fi

            # 性能配置
            CMD_ARGS="${CMD_ARGS} --processes-per-card ${PROCESSES_PER_CARD}"
            CMD_ARGS="${CMD_ARGS} --timeout-per-operator ${TIMEOUT_PER_PROCESS}"
            CMD_ARGS="${CMD_ARGS} --warmup ${WARMUP}"
            CMD_ARGS="${CMD_ARGS} --repeat ${REPEAT}"
            CMD_ARGS="${CMD_ARGS} --profiler-level ${PROFILER_LEVEL}"

            # 评测种子（确保可复现）
            if [[ -n "${EVAL_SEED}" ]]; then
                CMD_ARGS="${CMD_ARGS} --eval-seed ${EVAL_SEED}"
            fi

            if [[ "${NO_PERF}" == true ]]; then
                CMD_ARGS="${CMD_ARGS} --no-perf"
            fi

            # 多进程并行参数（通过环境变量传递给底层）
            export TASKS_PROCESSES_PER_CARD="${PROCESSES_PER_CARD}"
            export TASKS_TIMEOUT_PER_PROCESS="${TIMEOUT_PER_PROCESS}"

            if [[ "${VERBOSE}" == true ]]; then
                if [[ "${CMD_ARGS}" == eval* ]]; then
                    CMD_ARGS="${CMD_ARGS} -v"
                fi
            fi
            ;;
        list)
            CMD_ARGS="list"
            CMD_ARGS="${CMD_ARGS} --bench-name ${BENCH_NAME}"
            if [[ -n "${TASK_DIR}" ]]; then
                CMD_ARGS="${CMD_ARGS} --task-dir ${TASK_DIR}"
            fi
            if [[ -n "${OPERATOR}" ]]; then
                CMD_ARGS="${CMD_ARGS} --operator ${OPERATOR}"
            fi
            ;;
        info)
            CMD_ARGS="info"
            if [[ -n "${OPERATOR}" ]]; then
                CMD_ARGS="${CMD_ARGS} --operator ${OPERATOR}"
            fi
            ;;
        config)
            CMD_ARGS="config --show"
            ;;
        *)
            log_error "未知操作: ${ACTION}"
            print_help
            exit 1
            ;;
    esac

    echo "${CMD_ARGS}"
}

# 执行命令
run_cmd() {
    CMD_ARGS="$1"
    if [[ "${CMD_ARGS}" == staged-eval* ]]; then
        PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python -m kernel_eval.staged_eval ${CMD_ARGS#staged-eval}
    else
        PYTHONPATH="${SRC_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python -m kernel_eval.cli ${CMD_ARGS}
    fi
}

# 主函数
main() {
    log_info "Kernel Bench 评测脚本"
    log_info "项目根目录: ${PROJECT_ROOT}"
    log_info "操作: ${ACTION}"

    # 推断 bench_name 和 task_dir（支持双向推断）
    # 反向推断：task_dir → bench_name
    if [[ -n "${TASK_DIR}" ]] && [[ "${TASK_DIR}" == *thirdparty/KernelBench* ]]; then
        BENCH_NAME="stanford"
    fi

    # 正向推断：bench_name → task_dir 默认值
    if [[ "${BENCH_NAME}" == "stanford" ]] && [[ -z "${TASK_DIR}" ]]; then
        TASK_DIR="bench_lab/stanford_bench/KernelBench/KernelBench"
    elif [[ "${BENCH_NAME}" == "cann" ]] && [[ -z "${TASK_DIR}" ]]; then
        TASK_DIR="tasks"
    fi

    log_info "评测集: ${BENCH_NAME}"

    # 根据 bench_name 设置 reports_dir
    # CANN（默认评测集）保持历史路径 reports/，与 Python 入口(config.py)默认值一致；
    # 其它评测集（stanford 等）使用 reports/<bench_name>/ 子目录避免互相覆盖。
    if [[ "${BENCH_NAME}" == "cann" ]]; then
        REPORTS_DIR="${PROJECT_ROOT}/reports"
    else
        REPORTS_DIR="${PROJECT_ROOT}/reports/${BENCH_NAME}"
    fi

    check_python

    # 仅在有 source-dir 时卸载 cann_bench（避免与即将编译安装的包冲突）
    # 无 source-dir 时保留已安装的包（golden whl 或 submission whl）
    if [[ -n "${SOURCE_DIR}" ]]; then
        uninstall_packages
    fi

    # 检查数据目录
    if [[ "${BENCH_NAME}" == "stanford" ]]; then
        check_stanford_data
    else
        check_tasks
    fi

    if [[ "${ACTION}" == "eval" ]]; then
        ensure_reports_dir

        if [[ -n "${SOURCE_DIR}" ]]; then
            log_info "源码目录: ${SOURCE_DIR}"
        fi
        if [[ -n "${TASK_DIR}" ]]; then
            log_info "评测目录: ${TASK_DIR}"
        else
            log_info "评测目录: tasks (默认)"
        fi
        if [[ -n "${OPERATOR}" ]]; then
            log_info "算子: ${OPERATOR}"
        fi

        # 显示设备配置
        if [[ -n "${DEVICE_ID}" ]]; then
            log_info "设备模式: 单卡 (NPU:${DEVICE_ID})"
        else
            log_info "设备模式: 多卡并行（自动检测）"
            log_info "进程配置: ${PROCESSES_PER_CARD} 进程/卡"
            log_info "进程超时: ${TIMEOUT_PER_PROCESS}s"
        fi

        # 显示性能配置
        log_info "预热次数: ${WARMUP}"
        log_info "采集次数: ${REPEAT}"
        log_info "Profiler: ${PROFILER_LEVEL}"
        if [[ "${NO_PERF}" == true ]]; then
            log_info "性能采集: 关闭（仅精度验证）"
        fi
        if [[ "${STAGED_EVAL}" == true && "${BENCH_NAME}" == "cann" && "${DEVICE_TYPE:-npu}" == "npu" ]]; then
            log_info "评测模式: 编译/精度/性能三阶段"
        else
            log_info "评测模式: 单次 eval"
        fi
    fi

    log_info "开始执行..."
    CMD_ARGS=$(build_cmd_args)
    if [[ "${CMD_ARGS}" == staged-eval* ]]; then
        log_info "命令: python -m kernel_eval.staged_eval ${CMD_ARGS#staged-eval}"
    else
        log_info "命令: python -m kernel_eval.cli ${CMD_ARGS}"
    fi

    if run_cmd "${CMD_ARGS}"; then
        log_success "执行完成"

        if [[ "${ACTION}" == "eval" ]]; then
            # 显示最新报告
            latest_report=$(ls -t "${REPORTS_DIR}"/*.md 2>/dev/null | head -1)
            if [[ -n "${latest_report}" ]]; then
                log_success "报告已生成: ${latest_report}"
                echo ""
                echo "========================================"
                echo "报告摘要:"
                echo "========================================"
                head -30 "${latest_report}"
            fi
        fi
    else
        log_error "执行失败"
        exit 1
    fi
}

main
