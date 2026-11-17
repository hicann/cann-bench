#!/usr/bin/env bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------



if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请使用 source scripts/activate_ascend_arena_env.sh" >&2
    exit 2
fi

_ascend_arena_activate() {
    local repo_root state_file key value
    local saved_conda_base="" saved_env_name="" saved_cann_script=""
    local conda_base="" conda_script cann_script="" conda_libstdcpp search_root candidate
    local env_name source_status=0 nounset_was_enabled=0
    local clean_path="" path_entry
    local -a path_entries

    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || return 1
    state_file="$repo_root/.ascend_arena_env"
    if [[ -r "$state_file" ]]; then
        while IFS='=' read -r key value; do
            case "$key" in
                CONDA_BASE) saved_conda_base="$value" ;;
                ENV_NAME) saved_env_name="$value" ;;
                CANN_ENV_SCRIPT) saved_cann_script="$value" ;;
            esac
        done < "$state_file"
    fi

    env_name="${ASCEND_ARENA_ENV_NAME:-${saved_env_name:-ascend-arena}}"
    if [[ -n "${ASCEND_ARENA_CONDA_BASE:-}" ]]; then
        conda_base="$ASCEND_ARENA_CONDA_BASE"
    elif [[ -n "$saved_conda_base" ]]; then
        conda_base="$saved_conda_base"
    elif command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base 2>/dev/null)" || return 1
    else
        for candidate in "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/anaconda3"; do
            if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
                conda_base="$candidate"
                break
            fi
        done
    fi

    conda_script="${conda_base:-}/etc/profile.d/conda.sh"
    if [[ ! -f "$conda_script" ]]; then
        echo "未找到 Conda。请先运行 bash scripts/setup_ascend_arena_env.sh" >&2
        return 1
    fi
    source "$conda_script" || return 1
    conda activate "$env_name" || {
        echo "未找到 Conda 环境 '$env_name'。请先运行安装脚本。" >&2
        return 1
    }
    export ASCEND_ARENA_CONDA_BASE="$conda_base"
    export ASCEND_ARENA_ENV_NAME="$env_name"
    if [[ -x /usr/bin/gcc && -x /usr/bin/g++ ]]; then
        export CC=/usr/bin/gcc
        export CXX=/usr/bin/g++
    fi

    cann_script="${CANN_ENV_SCRIPT:-$saved_cann_script}"
    if [[ -z "$cann_script" && -n "${ASCEND_HOME_PATH:-}" \
        && -r "$ASCEND_HOME_PATH/set_env.sh" ]]; then
        cann_script="$ASCEND_HOME_PATH/set_env.sh"
    fi
    if [[ -z "$cann_script" ]]; then
        for search_root in "$HOME/Ascend" /usr/local/Ascend; do
            [[ -d "$search_root" ]] || continue

            cann_script="$(find "$search_root" -mindepth 2 -maxdepth 2 \
                \( -type f -o -type l \) \
                -path "$search_root/cann-*/set_env.sh" \
                2>/dev/null | sort -V | tail -n 1)"
            [[ -n "$cann_script" ]] && break

            for candidate in \
                "$search_root/ascend-toolkit/set_env.sh" \
                "$search_root/ascend-toolkit/latest/set_env.sh"; do
                if [[ -r "$candidate" ]]; then
                    cann_script="$candidate"
                    break 2
                fi
            done

            if [[ -d "$search_root/ascend-toolkit" ]]; then
                cann_script="$(find "$search_root/ascend-toolkit" \
                    -mindepth 2 -maxdepth 2 \( -type f -o -type l \) \
                    -name set_env.sh 2>/dev/null | sort -V | tail -n 1)"
                [[ -n "$cann_script" ]] && break
            fi
        done
    fi
    if [[ -z "$cann_script" || ! -r "$cann_script" ]]; then
        echo "未找到 CANN set_env.sh。请设置 CANN_ENV_SCRIPT 后重试。" >&2
        return 1
    fi

    if [[ "${ASCEND_ARENA_CANN_LOADED:-}" != "$cann_script" ]] \
        || ! command -v bisheng >/dev/null 2>&1; then
        case "$-" in
            *u*) nounset_was_enabled=1; set +u ;;
        esac
        if source "$cann_script"; then
            source_status=0
        else
            source_status=$?
        fi
        (( nounset_was_enabled == 0 )) || set -u
        if (( source_status != 0 )); then
            echo "加载 CANN 失败：$cann_script" >&2
            return "$source_status"
        fi
        export ASCEND_ARENA_CANN_LOADED="$cann_script"
    fi
    export CANN_ENV_SCRIPT="$cann_script"

    if ! command -v bisheng >/dev/null 2>&1; then
        echo "CANN 已加载，但未找到 bisheng；请检查 Toolkit 安装。" >&2
        return 1
    fi

    conda_libstdcpp="$CONDA_PREFIX/lib/libstdc++.so.6"
    if [[ ! -e "$conda_libstdcpp" ]]; then
        echo "缺少 $conda_libstdcpp，请重新运行安装脚本。" >&2
        return 1
    fi
    IFS=': ' read -r -a path_entries <<< "${LD_PRELOAD:-}"
    clean_path=""
    for path_entry in "${path_entries[@]}"; do
        [[ -z "$path_entry" || "$path_entry" == "$conda_libstdcpp" ]] && continue
        clean_path="${clean_path:+$clean_path:}$path_entry"
    done
    export LD_PRELOAD="$conda_libstdcpp${clean_path:+:$clean_path}"
    export PYTHONNOUSERSITE=1

    IFS=':' read -r -a path_entries <<< "${PYTHONPATH:-}"
    clean_path=""
    for path_entry in "${path_entries[@]}"; do
        [[ -z "$path_entry" || "$path_entry" == "$repo_root/src" ]] && continue
        clean_path="${clean_path:+$clean_path:}$path_entry"
    done
    export PYTHONPATH="$repo_root/src${clean_path:+:$clean_path}"
    export ASCEND_ARENA_REPO_ROOT="$repo_root"

    echo "已激活 $env_name"
    echo "CANN: $CANN_ENV_SCRIPT"
    echo "Python: $(command -v python)"
}

if _ascend_arena_activate; then
    unset -f _ascend_arena_activate
    return 0
else
    unset -f _ascend_arena_activate
    return 1
fi
