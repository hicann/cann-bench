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
# 用法: ./script/run_evaluation.sh [选项]

set -e

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="${PROJECT_ROOT}/src"
KERNEL_BENCH_ROOT="${PROJECT_ROOT}/kernel_bench"
REPORTS_DIR="${PROJECT_ROOT}/reports"

# 默认配置
LEVEL=""
OPERATOR=""
CASE_ID=""
SOURCE_DIR=""
VERBOSE=false
ACTION="eval"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_help() {
    echo "Kernel Bench 评测脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -a, --action <action>     操作类型: eval(评测), list(列表), info(详情), config(配置)"
    echo "                            默认: eval"
    echo "  --source-dir <dir>        AI生成的算子源码目录（自动扫描编译安装）"
    echo "  -l, --level <level>       算子难度级别 (1/2/3/4)"
    echo "  -o, --operator <name>     算子名称 (如 Exp, Softmax)"
    echo "  -c, --case-id <id>        用例编号"
    echo "  -v, --verbose             详细输出"
    echo "  -h, --help                显示帮助信息"
    echo ""
    echo "示例:"
    echo "  # 查看帮助"
    echo "  $0 --help"
    echo ""
    echo "  # 列出L1级所有算子"
    echo "  $0 -a list -l 1"
    echo ""
    echo "  # 查看算子详情"
    echo "  $0 -a info -o Exp"
    echo ""
    echo "  # 从源码目录评测（自动扫描编译安装）"
    echo "  $0 --source-dir /path/to/ai_ops"
    echo ""
    echo "  # 仅执行Golden验证（不安装whl）"
    echo "  $0 -l 1 -o Exp"
    echo ""
    echo "  # 评测单个用例"
    echo "  $0 -l 1 -o Exp -c 1"
    echo ""
    echo "  # 查看配置"
    echo "  $0 -a config"
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
        -l|--level)
            LEVEL="$2"
            shift 2
            ;;
        -o|--operator)
            OPERATOR="$2"
            shift 2
            ;;
        -c|--case-id)
            CASE_ID="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            print_help
            exit 1
            ;;
    esac
done

# 检查Python环境
check_python() {
    if ! command -v python &> /dev/null; then
        log_error "Python未安装"
        exit 1
    fi
    log_info "Python版本: $(python --version)"
}

# 检查kernel_bench数据目录
check_kernel_bench() {
    if [[ ! -d "${KERNEL_BENCH_ROOT}" ]]; then
        log_error "kernel_bench目录不存在: ${KERNEL_BENCH_ROOT}"
        exit 1
    fi
    log_info "kernel_bench目录: ${KERNEL_BENCH_ROOT}"
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
            CMD_ARGS="eval"
            if [[ -n "${SOURCE_DIR}" ]]; then
                CMD_ARGS="${CMD_ARGS} --source-dir ${SOURCE_DIR}"
            fi
            if [[ -n "${OPERATOR}" ]]; then
                CMD_ARGS="${CMD_ARGS} --operator ${OPERATOR}"
            fi
            if [[ -n "${LEVEL}" ]]; then
                CMD_ARGS="${CMD_ARGS} --level ${LEVEL}"
            fi
            if [[ -n "${CASE_ID}" ]]; then
                CMD_ARGS="${CMD_ARGS} --case-id ${CASE_ID}"
            fi
            if [[ "${VERBOSE}" == true ]]; then
                CMD_ARGS="${CMD_ARGS} -v"
            fi
            ;;
        list)
            CMD_ARGS="list"
            if [[ -n "${LEVEL}" ]]; then
                CMD_ARGS="${CMD_ARGS} --level ${LEVEL}"
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
            if [[ -n "${LEVEL}" ]]; then
                CMD_ARGS="${CMD_ARGS} --level ${LEVEL}"
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
    PYTHONPATH="${SRC_DIR}" python -m kernel_eval.cli ${CMD_ARGS}
}

# 主函数
main() {
    log_info "Kernel Bench 评测脚本"
    log_info "项目根目录: ${PROJECT_ROOT}"
    log_info "操作: ${ACTION}"

    check_python

    if [[ "${ACTION}" == "eval" ]]; then
        check_kernel_bench
        ensure_reports_dir

        if [[ -n "${SOURCE_DIR}" ]]; then
            log_info "源码目录: ${SOURCE_DIR}"
        fi
        if [[ -n "${OPERATOR}" ]]; then
            log_info "算子: ${OPERATOR}"
        fi
        if [[ -n "${LEVEL}" ]]; then
            log_info "级别: L${LEVEL}"
        fi
    fi

    log_info "开始执行..."
    CMD_ARGS=$(build_cmd_args)
    log_info "命令: python -m kernel_bench.cli ${CMD_ARGS}"

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