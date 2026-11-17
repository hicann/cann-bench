#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms of the
# CANN Open Software License Agreement Version 2.0 (the "License").
# ----------------------------------------------------------------------------------------------------------
# 共享：确保 cann_bench_utils 已编译安装（V3 Anti-Cheat 强制依赖）
#
# 由 scripts/run_evaluation.sh 与 scripts/run_auto_pipeline.sh 共同 source。
# 幂等：已安装时检查导入及构建环境兼容性，未安装则自动编译 + 安装 + 验证。
# ----------------------------------------------------------------------------------------------------------

# 自带最小日志函数（加 _cbu_ 前缀，避免与调用方的 log_info/log_error 命名冲突）
_cbu_log_info()  { echo -e "\033[0;34m[INFO]\033[0m $1"; }
_cbu_log_error() { echo -e "\033[0;31m[ERROR]\033[0m $1"; }

_cbu_check_build() {
    "$1" - "$2" <<'PY_CHECK'
import sys
from pathlib import Path
try:
    from cann_bench_utils import cann_bench_warmup, cann_bench_cache_clean
    from cann_bench_utils.build_info import check_build_compatibility
    source_root = Path(sys.argv[1])
    check_build_compatibility(source_root=source_root if (source_root / "CMakeLists.txt").is_file() else None)
except Exception as error:
    print(str(error), file=sys.stderr)
    sys.exit(1)
PY_CHECK
}

ensure_cann_bench_utils() {
    local py="${PYTHON:-python}"
    # PROJECT_ROOT 由调用方（run_evaluation.sh / run_auto_pipeline.sh）预先设置；
    # 若未设置则按本文件位置回退推导。
    local project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    local utils_dir="${project_root}/src/cann_bench_utils"
    if _cbu_check_build "$py" "$utils_dir" 2>/dev/null; then
        _cbu_log_info "cann_bench_utils 已安装且构建环境匹配"
        return 0
    fi

    _cbu_log_info "cann_bench_utils 缺失或构建环境不匹配，开始重新编译安装..."

    if [[ ! -d "${utils_dir}" ]]; then
        _cbu_log_error "cann_bench_utils 源码目录不存在: ${utils_dir}"
        _cbu_log_error "V3 Anti-Cheat 需要 cann_bench_utils，请检查代码库完整性"
        return 1
    fi

    _cbu_log_info "编译 cann_bench_utils..."
    ( cd "${utils_dir}" && PYTHON="$py" bash build.sh --clean ) &> /tmp/cann_bench_utils_build.log || {
        _cbu_log_error "cann_bench_utils 编译失败，查看日志: /tmp/cann_bench_utils_build.log"
        tail -15 /tmp/cann_bench_utils_build.log
        return 1
    }

    local wheel
    wheel=$(ls -t "${utils_dir}"/dist/cann_bench_utils-*.whl 2>/dev/null | head -1)
    if [[ -z "${wheel}" ]]; then
        _cbu_log_error "未找到编译的 wheel 包"
        return 1
    fi

    _cbu_log_info "安装 cann_bench_utils..."
    "$py" -m pip install "${wheel}" --force-reinstall --no-deps &> /tmp/cann_bench_utils_install.log || {
        _cbu_log_error "cann_bench_utils 安装失败，查看日志: /tmp/cann_bench_utils_install.log"
        return 1
    }

    if _cbu_check_build "$py" "$utils_dir" 2>/dev/null; then
        _cbu_log_info "cann_bench_utils 安装成功"
    else
        _cbu_log_error "cann_bench_utils 安装验证失败"
        return 1
    fi
}
