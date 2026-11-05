#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
评测工程配置管理

职责：
1. 定义评测工程全局配置
2. 提供配置获取接口
3. 管理路径配置（tasks 数据目录、报告输出目录等）
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


def _default_precision_thresholds() -> dict:
    """返回 utils.precision.PRECISION_THRESHOLDS 的浅拷贝。

    在函数体内延迟 import 是为了打破 utils <-> config 的潜在循环依赖。
    """
    from .utils.thresholds import PRECISION_THRESHOLDS

    return dict(PRECISION_THRESHOLDS)


@dataclass
class Config:
    """评测工程配置"""

    # 路径配置
    tasks_root: str = ""  # tasks 数据目录路径
    reports_dir: str = ""  # 报告输出目录
    bench_name: str = "cann"  # 评测集名称

    # AI 模型信息（由用户配置，用于报告摘要；为空时摘要中不体现）
    agent_skill: str = ""
    base_model: str = ""
    harness: str = ""

    # 源码目录（AI生成的算子源码，通过参数传入）
    source_dir: str = ""  # AI生成的算子源码目录
    source_build_args: List[str] = field(
        default_factory=list
    )  # 传给 source_dir/build.sh 的额外参数
    build_timeout_sec: Optional[int] = None  # CLI > environment > 600 seconds

    # 设备配置
    device_type: str = "npu"  # cpu / npu
    device_id: int = 0

    # 性能配置
    # NPU 模式下默认启用 profiler 以获取 kernel-only 时间
    enable_profiler: bool = True
    # Profiler 级别：Level1（默认，47列CSV）或 Level2（更详细AICPU采集）
    profiler_level: str = "Level1"
    # NPU 升频清 cache（10240x10240 大矩阵 warmup），部分卡大矩阵会 hang 时可关闭
    perf_freq_boost: bool = True
    # Profiler 采集失败后的自动重试次数（0=不重试，2=最多重试2次共3次尝试）
    # 当 kernel_details.csv 缺失或 elapsed_us<=0 但算子已成功执行（Host 侧有 kernel launch
    # 记录，即 last_outputs is not None）时自动重试，避免 flaky capture 导致误判 0 分
    profiler_max_retries: int = 2
    # 重试间隔（秒），用于等待 profiler session 状态清理
    profiler_retry_delay: float = 1.0

    # 评测配置
    warmup: int = 3  # 性能评测预热次数
    repeat: int = 5  # 性能评测采集次数
    timeout_per_operator: int = 300  # 单算子超时（秒）

    # PyPTO Pro 评测进程隔离标记。由外层调度器在确认每个
    # eval-child 只包含一个 case 时设置；子进程据此直接执行 AI op，
    # 避免再启动一层子进程并序列化大张量。
    pypto_pro_outer_case_isolation: bool = False

    # 多进程并行配置（统一架构）
    processes_per_card: int = 2  # 每卡进程数
    # 单个 eval-child 最多处理的 case 数。通过缩短子进程生命周期限制
    # torch_npu/ACL/JIT 等 native host 内存池的累积，同时不降低并发度。
    max_cases_per_task_unit: int = 64

    # 防作弊：用新鲜输入二次验证（Evaluator._retry_with_fresh_inputs，
    # 见 eval/evaluator.py；accuracy_eval.evaluate_with_retry 是早期独立 API，
    # 已 DeprecationWarning）。
    # 防止 submission 缓存第一次输出 / 翻转 "computed-once" 标志骗过单次评测。
    # 启用后每个 case 用新鲜输入再跑一次 golden + AI，两次都通过才记 pass。
    enable_accuracy_retry: bool = False

    # 防作弊：仅在**性能测量阶段**轮换输入地址（对实际调用的 args/kwargs 张量做 clone 池），
    # 使按 data_ptr() 命中的缓存 / 预设结果在 repeat 各步 cache miss 而失效。
    # 只作用于 profiler 采集路径（性能阶段）；精度阶段（golden / AI 一次执行）不启用。
    # 为 True 时 op_runner 以 use_input_pool=True 调 run_profiled。内存占用按 InputPool 上限裁剪。
    perf_rotate_inputs: bool = True

    # 同一 TaskUnit 内多个 case 共用 profiler 会话。正确性阶段会记录每个
    # case 的 host RSS 峰值；超过阈值的 case 在性能阶段自动改走逐 case profiler。
    perf_batch_cases: bool = True
    perf_batch_case_memory_limit_mb: float = 32 * 1024
    monitor_case_memory: bool = False

    # 防作弊：监听 AI 算子在执行时是否直接调用了 torch.matmul / conv / softmax
    # 等内置数学 API（=把计算甩给 PyTorch，跳过自己写的 AscendC kernel）。
    # off=不监听；warn=记日志不阻断（便于排查）；block=直接抛错（默认）。
    torch_op_guard_mode: str = "block"

    # 精度配置（采用生态算子开源精度标准）
    # 通过条件: MERE < threshold, MARE < 10 * threshold
    # MERE = avg(|actual - golden| / (|golden| + 1e-7))
    # MARE = max(|actual - golden| / (|golden| + 1e-7))
    # 单一事实来源：utils.precision.PRECISION_THRESHOLDS。本字段持有它的拷贝，
    # 允许 caller 在不污染模块级常量的前提下自定义阈值（例如 op_info.precision_thresholds
    # 通过 dict.update 覆盖单 dtype）。
    precision_thresholds: dict = field(
        default_factory=lambda: _default_precision_thresholds()
    )

    # 精度判断器名称
    # 支持选择不同的精度判断标准：
    # - "relative_error": MERE/MARE + 小值域 + 相消处理（默认，完整精度标准）
    # - "allclose": torch.allclose 简化对比（快速验证/调试）
    # 可通过注册机制添加自定义判断器
    checker_name: str = "relative_error"

    # 输入生成确定性种子
    # 0（默认）: 基于case_id自动确定种子，确保可复现
    # N（正整数）: 用 (N + case_id_hash) % 2^31 作为种子，不同N给出不同但可复现的输入
    # None: 纯随机模式（不推荐，会导致 flaky 测试）。CLI中通过 --eval-seed -1 启用。
    eval_seed: Optional[int] = 0

    # 输入数据分布（本次评测全局生效；评测哪些用例由 --task-dir / --operator /
    # --case-id 决定，与本项正交）
    # "uniform"（默认）: 在 value_range 内均匀采样 —— 与历史行为逐位一致
    # "normal": 正态分布，参数由 value_range 推出，mu=(min+max)/2、sigma=(max-min)/6
    #           （即 ±3σ 覆盖声明区间），采样后 clamp 回 [min, max]
    # 仅作用于浮点输入；整数 / bool 始终均匀（其取值分布常带语义约束，如索引需
    # 覆盖整个维度）。CLI 通过 --input-dist 设置。
    input_dist: str = "uniform"

    # Read-only, explicitly selected task fixtures; build with kernel_eval.cache_cli.
    tensor_cache_dir: str = ""
    tensor_cache_mode: str = "off"  # off / prefer (compute on miss) / require
    tensor_cache_tasks: list[str] = field(default_factory=list)

    # 性能指标策略覆盖
    # None（默认）: 使用 BenchConfig 中注册的策略（通常为 kernel_details）
    # "trace_view": PYPTO 口径 — tilefwk/PYPTO aicore_e2e（由 CLI --perf-metric-strategy 设置）
    # "msprof_summary": 基准采集专用 — 优先 kernel_details.csv，fallback msprof op_summary
    perf_metric_strategy_override: Optional[str] = None

    # ACL 逐算子模式（默认 True）
    # 单算子 benchmark 场景下 ACL 是正确模式：
    # - Ge 图优化不适用（只有一个算子，无融合对象）
    # - ACL 确保 profiler 产出完整 kernel_details.csv
    enable_acl_launch_mode: bool = True

    # msprof export 开关（默认 True）
    # 仅在 MsProfSummaryStrategy 下生效
    enable_msprof_export: bool = True

    def __post_init__(self):
        """初始化后自动设置默认路径"""
        if not self.tasks_root:
            self.tasks_root = str(get_project_root() / "tasks")

        if not self.reports_dir:
            self.reports_dir = str(get_project_root() / "reports")

    def get_tasks_path(self) -> Path:
        """获取tasks 数据目录路径"""
        return Path(self.tasks_root)

    def get_reports_path(self) -> Path:
        """获取报告输出目录路径"""
        return Path(self.reports_dir)

    def get_source_path(self) -> Path:
        """获取源码目录路径"""
        return Path(self.source_dir) if self.source_dir else None


# 全局配置实例
_global_config: Optional[Config] = None

# 项目根目录缓存
_project_root: Optional[Path] = None


def get_project_root() -> Path:
    """返回项目根目录（向上查找 tasks 或 .git 标记）"""
    global _project_root
    if _project_root is not None:
        return _project_root
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "tasks").is_dir() or (current / ".git").is_dir():
            _project_root = current
            return current
        current = current.parent
    raise RuntimeError("Cannot determine project root")


def get_config() -> Config:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """设置全局配置"""
    global _global_config
    _global_config = config
