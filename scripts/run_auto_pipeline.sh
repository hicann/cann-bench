#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."
export PROJECT_ROOT="${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

case "${1:-}" in
    ""|-*|prepare|run|list|monitor|kill|retry) ;;
    *) echo "未知子命令: $1；可用命令: prepare, run, list, monitor, kill, retry" >&2; exit 2 ;;
esac

# prepare 使用独立的 Ascend Arena 评测入口。
if [[ "${1:-}" != "prepare" ]]; then
    source "${SCRIPT_DIR}/ensure_cann_bench_utils.sh"
    ensure_cann_bench_utils || exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- --help
fi

# 自动注册 pypto-gym（cannbot-skills 架构）的 agent 到 opencode
# 从 CLI 参数中自动检测 --workspace，若存在 cannbot-skills 结构则建立
# .opencode/agents/ + .opencode/skills/ 软链接，无感完成 agent 注册
_prev=""
for _i in "$@"; do
    if [[ "${_prev}" == "--workspace" ]]; then
        _PTO_WS="${_i}"
        break
    fi
    _prev="${_i}"
done
if [[ -n "${_PTO_WS:-}" && -d "${_PTO_WS}/cannbot-skills/plugins-official/pypto-op-orchestrator/agents" ]]; then
    _PLUGIN="$(realpath "${_PTO_WS}/cannbot-skills/plugins-official/pypto-op-orchestrator" 2>/dev/null || true)"
    _OPS="$(realpath "${_PTO_WS}/cannbot-skills/ops" 2>/dev/null || true)"
    if [[ -n "${_PLUGIN}" && -n "${_OPS}" && -f "${_PLUGIN}/AGENTS.md" ]]; then
        _OC_DIR="$(realpath "${_PTO_WS}" 2>/dev/null || echo "${_PTO_WS}")/.opencode"
        mkdir -p "${_OC_DIR}/agents" "${_OC_DIR}/skills"
        ln -sfn "${_PLUGIN}/AGENTS.md" "${_OC_DIR}/agents/pypto-op-orchestrator.md"
        for _a in "${_PLUGIN}/agents/"*.md; do
            [[ -f "${_a}" ]] && ln -sfn "$(realpath "${_a}")" "${_OC_DIR}/agents/$(basename "${_a}")"
        done
        for _s in "${_OPS}/"*/; do
            [[ -d "${_s}" ]] && ln -sfn "$(realpath "${_s}")" "${_OC_DIR}/skills/$(basename "${_s}")"
        done
        if [[ -d "${_PLUGIN}/hooks/opencode" ]]; then
            mkdir -p "${_OC_DIR}/plugins"
            cp -a "${_PLUGIN}/hooks/opencode/." "${_OC_DIR}/plugins/" || true
        fi
        if [[ -d "${_PLUGIN}/hooks/pypto-op-lint" ]]; then
            mkdir -p "${_OC_DIR}/hooks/pypto-op-lint"
            cp -a "${_PLUGIN}/hooks/pypto-op-lint/." "${_OC_DIR}/hooks/pypto-op-lint/" || true
        fi
        echo "[agent] 已注册 .opencode/agents + .opencode/skills"
    fi
fi

exec python -m auto_pipeline.cli "$@"
