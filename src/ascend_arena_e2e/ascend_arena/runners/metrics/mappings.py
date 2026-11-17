# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""Metric constants and column mappings for profiling.

Extracted from ProfileRunner to centralize configuration.
"""

from __future__ import annotations

DROPPED_NULL_KEYS: frozenset[str] = frozenset({
    "task_type",
    "hf32_eligible",
    "input_shapes",
    "input_data_types",
    "input_formats",
    "output_shapes",
    "output_data_types",
    "output_formats",
})

CACHE_CLEAR_OP_SPECS: tuple[dict[str, str], ...] = (
    {
        "op_type": "DSARandomUniform",
        "op_name_prefix": "aclnnInplaceUniform_DSARandomUniform",
    },
    {
        "op_type": "MatMulV2",
        "op_name_prefix": "aclnnMatmul_MatMulCommon_MatMulV2",
        "input_shapes": "10240,10240;10240,10240",
        "input_data_types": "FLOAT16;FLOAT16",
    },
    {
        "op_type": "MatMulV3",
        "op_name_prefix": "aclnnMatmul_MatMulV3",
        "input_shapes": "10240,10240;10240,10240",
        "input_data_types": "FLOAT16;FLOAT16",
    },
    {
        "op_type": "ReduceMax",
        "op_name_prefix": "aclnnMax_ReduceMax",
        "input_shapes": "96,1024,1024;3",
        "input_data_types": "FLOAT16;INT64",
    },
)

GROUP_KEY_COLUMNS: tuple[str, ...] = (
    "Op Name",
    "OP Type",
    "Input Shapes",
    "Input Data Types",
    "Output Shapes",
)

TOP_LEVEL_BASIC_KEYS: tuple[str, ...] = (
    "op_name",
    "sample_count",
    "op_type",
    "task_type",
    "hf32_eligible",
    "input_shapes",
    "input_data_types",
    "input_formats",
    "output_shapes",
    "output_data_types",
    "output_formats",
    "block_dim",
    "mix_block_dim",
    "task_duration_us",
    "task_wait_time_us",
    "device_id",
    "pid",
    "current_freq",
    "rated_freq",
    "aic_frequency_mhz",
    "aiv_frequency_mhz",
)

NON_PERF_COLUMNS: frozenset[str] = frozenset({"block_id", "sub_block_id"})

PRESERVED_FIELDS: dict[str, str] = {
    "op_type": "OP Type",
    "task_type": "Task Type",
    "hf32_eligible": "HF32 Eligible",
    "input_shapes": "Input Shapes",
    "input_data_types": "Input Data Types",
    "input_formats": "Input Formats",
    "output_shapes": "Output Shapes",
    "output_data_types": "Output Data Types",
    "output_formats": "Output Formats",
}

PRESERVED_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "block_dim": ("Block Dim", "Block Num"),
    "mix_block_dim": ("Mix Block Dim", "Mix Block Num"),
}

AVERAGED_FIELDS: dict[str, str] = {
    "task_duration_us": "Task Duration(us)",
    "task_wait_time_us": "Task Wait Time(us)",
    "aicore_time_us": "aicore_time(us)",
    "aic_total_cycles": "aic_total_cycles",
    "aic_mac_time_us": "aic_mac_time(us)",
    "aic_mac_ratio": "aic_mac_ratio",
    "aic_scalar_time_us": "aic_scalar_time(us)",
    "aic_scalar_ratio": "aic_scalar_ratio",
    "aic_mte1_time_us": "aic_mte1_time(us)",
    "aic_mte1_ratio": "aic_mte1_ratio",
    "aic_mte2_time_us": "aic_mte2_time(us)",
    "aic_mte2_ratio": "aic_mte2_ratio",
    "aic_fixpipe_time_us": "aic_fixpipe_time(us)",
    "aic_fixpipe_ratio": "aic_fixpipe_ratio",
    "aic_icache_miss_rate": "aic_icache_miss_rate",
    "aiv_time_us": "aiv_time(us)",
    "aiv_total_cycles": "aiv_total_cycles",
    "aiv_vec_time_us": "aiv_vec_time(us)",
    "aiv_vec_ratio": "aiv_vec_ratio",
    "aiv_scalar_time_us": "aiv_scalar_time(us)",
    "aiv_scalar_ratio": "aiv_scalar_ratio",
    "aiv_mte2_time_us": "aiv_mte2_time(us)",
    "aiv_mte2_ratio": "aiv_mte2_ratio",
    "aiv_mte3_time_us": "aiv_mte3_time(us)",
    "aiv_mte3_ratio": "aiv_mte3_ratio",
    "aiv_icache_miss_rate": "aiv_icache_miss_rate",
    "cube_utilization_pct": "cube_utilization(%)",
}

GROUPED_KEYS: dict[str, tuple[str, ...]] = {
    "OpBasicInfo": (
        "op_name",
        "op_type",
        "block_dim",
        "mix_block_dim",
        "task_duration_us",
        "task_wait_time_us",
    ),
    "PipeUtilization": (
        "aic_total_cycles",
        "aic_mac_time_us",
        "aic_mac_ratio",
        "aic_scalar_time_us",
        "aic_scalar_ratio",
        "aic_mte1_time_us",
        "aic_mte1_ratio",
        "aic_mte2_time_us",
        "aic_mte2_ratio",
        "aic_fixpipe_time_us",
        "aic_fixpipe_ratio",
        "aic_icache_miss_rate",
        "aiv_time_us",
        "aiv_total_cycles",
        "aiv_vec_time_us",
        "aiv_vec_ratio",
        "aiv_scalar_time_us",
        "aiv_scalar_ratio",
        "aiv_mte2_time_us",
        "aiv_mte2_ratio",
        "aiv_mte3_time_us",
        "aiv_mte3_ratio",
        "aiv_icache_miss_rate",
        "core_prefix",
        "core_time_us",
        "vec_ratio_pct",
        "scalar_ratio_pct",
        "mte2_ratio_pct",
        "mte3_ratio_pct",
        "icache_miss_pct",
        "mte2_active_bw_gbps",
        "mte3_active_bw_gbps",
        "head_overhead_us",
        "head_overhead_pct",
    ),
    "Memory": (
        "gm_to_ub_kb_total",
        "ub_to_gm_kb_total",
        "read_main_memory_kb_total",
        "write_main_memory_kb_total",
        "gm_to_ub_bw_usage_pct",
        "ub_to_gm_bw_usage_pct",
        "gm_to_ub_avg_bw_gbps",
        "ub_to_gm_avg_bw_gbps",
        "avg_mte2_transfer_kb",
    ),
    "MemoryUB": (
        "ub_read_bw_vector_gbps",
        "ub_write_bw_vector_gbps",
        "ub_read_bw_scalar_gbps",
        "ub_write_bw_scalar_gbps",
    ),
    "MemoryL0": (),
    "L2Cache": (
        "l2_total_hit_rate_pct",
        "l2_read_hit_rate_pct",
        "l2_write_hit_rate_pct",
        "write_cache_hit_total",
        "write_cache_miss_allocate_total",
    ),
    "ResourceConflictRatio": (
        "vec_total_cflt_pct",
        "bankgroup_cflt_pct",
        "bank_cflt_pct",
        "resc_cflt_pct",
        "mte_cflt_pct",
        "vec_wait_pct",
        "mte2_wait_pct",
        "mte3_wait_pct",
    ),
    "ArithmeticUtilization": (
        "vec_fp32_pct",
        "vec_fp16_pct",
        "vec_int32_pct",
        "vec_int16_pct",
        "vec_misc_pct",
        "vec_fops_per_core",
        "cube_utilization_pct",
    ),
}
