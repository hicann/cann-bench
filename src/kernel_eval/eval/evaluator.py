#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
综合评测调度器

职责：
1. 协调精度评测和性能评测执行顺序
2. 支持源码目录扫描、编译、安装
3. 实现评测任务筛选（level/operator/case_id）
4. 生成评测结果

重构说明：
- 数据类移至 results.py
- 失败结果合成移至 failure_synthesizer.py
- 算子匹配移至 operator_matcher.py
"""

import ctypes
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from inspect import Parameter, signature

import torch

from ..config import Config, get_config
from ..registry.loader_registry import get_task_loader, get_case_loader
from ..utils.case_eligibility import (
    detect_execution_hardware,
    filter_cases_for_hardware,
)
from ..registry.golden_registry import get_golden_loader
from ..registry.bench_registry import get_bench_config
from ..data.data_generator import DataGenerator
from ..data.tensor_cache import (
    TensorCache,
    TensorCacheError,
    TensorCacheUnsupportedError,
    case_seed as cache_case_seed,
    identity_for,
    input_hashes,
    selected,
    validate_inputs,
)
from ..data.package_manager import PackageManager
from ..base.models import CaseSpec
from ..utils.device_manager import DeviceManager, DeviceConfig
from ..utils.param_builder import ParamBuilder
from ..utils.tensor_utils import tensors_to_cpu, tensors_to_fp64_cpu
from ..base.result import (
    FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
    FAILURE_TYPE_REFERENCE_INVALID,
    get_accuracy_failure_type,
)
from .op_runner import OpRunner, OpRunResult
from .accuracy_eval import AccuracyEvaluator
from .restored_inputs import clone_reference_inputs
from .perf_eval import PerfEvaluator
from .results import EvalCaseResult, EvalOperatorResult, EvalSessionResult
from .failure_synthesizer import FailureSynthesizer
from .index_check import validate_index_gather_outputs
from .subprocess_utils import detect_pypto_pro_submission, _terminate_process_group
from ..registry.matcher_registry import get_operator_matcher

# 导入 benches 模块，确保 Registry 已注册
from .. import benches as _benches  # noqa: F401 -- register benchmark components


# RSS 主动清理阈值环境变量（单位 MB）；<=0 表示禁用
_RSS_TRIM_ENV_VAR = "KERNEL_EVAL_RSS_TRIM_MB"
# 默认 48 GiB：runner 容器 128 GiB 限额下，为单 case 的 fp64 golden 峰值预留余量
_RSS_TRIM_DEFAULT_MB = 48 * 1024


def _write_json_atomic(path: str, payload: dict) -> None:
    """Write JSON without exposing a truncated destination to the parent."""
    output_path = Path(path)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, ensure_ascii=False, indent=2)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temp_path, output_path)
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


def _read_rss_mb() -> Optional[float]:
    """读取 /proc/self/status 的 VmRSS（MB）。任何失败静默返回 None。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


class _CaseMemoryMonitor:
    """在 case 执行期间低频采样当前进程 VmRSS。"""

    def __init__(self, enabled: bool, interval_sec: float = 0.05):
        self.enabled = enabled
        self.interval_sec = interval_sec
        self.baseline_mb: Optional[float] = None
        self.peak_mb: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        rss_mb = _read_rss_mb()
        if rss_mb is not None:
            self.peak_mb = rss_mb if self.peak_mb is None else max(self.peak_mb, rss_mb)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._sample()

    def __enter__(self):
        if not self.enabled:
            return self
        self.baseline_mb = _read_rss_mb()
        self.peak_mb = self.baseline_mb
        self._thread = threading.Thread(
            target=self._run,
            name="case-rss-monitor",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.enabled:
            return False
        self._sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec * 2, 0.1))
        return False


def _rss_trim_threshold_mb() -> Optional[float]:
    """RSS 主动清理阈值（MB）。

    默认 48 GiB；环境变量 KERNEL_EVAL_RSS_TRIM_MB 覆盖；
    <=0 表示禁用（返回 None）；非法值（含 nan/inf）回退默认。
    """
    raw = os.environ.get(_RSS_TRIM_ENV_VAR)
    if raw is None:
        return float(_RSS_TRIM_DEFAULT_MB)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float(_RSS_TRIM_DEFAULT_MB)
    # nan 的所有比较都为 False，不做显式判断的话会走成"静默禁用"，
    # 与文档承诺的"非法值回退默认"相反。
    if math.isnan(value):
        return float(_RSS_TRIM_DEFAULT_MB)
    return value if value > 0 else None


@lru_cache(maxsize=1)
def _libc() -> Optional["ctypes.CDLL"]:
    """glibc handle（已声明 malloc_trim 签名）；非 glibc 平台返回 None。

    _cleanup_memory 每个 case 都会走到，故只解析一次。
    """
    try:
        libc = ctypes.CDLL("libc.so.6")
        # 不声明签名时 ctypes 按 c_int 传参/取返回值；malloc_trim(size_t) 在
        # 64 位下靠零扩展侥幸可用，显式声明才是有定义的行为。
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        return libc
    except Exception:
        return None


def _malloc_trim() -> bool:
    """调用 glibc malloc_trim(0) 归还空闲堆页，返回是否真的归还了内存。

    malloc_trim 返回 1 表示确实释放了页，0 表示无事可做——后者对调用方而言
    与"没执行"等价，所以一并返回 False，避免日志谎报归还。
    非 glibc 平台或任何调用失败静默降级（返回 False），不影响评测主流程。
    """
    libc = _libc()
    if libc is None:
        return False
    try:
        return bool(libc.malloc_trim(0))
    except Exception:
        return False


class Evaluator:
    """综合评测调度器"""

    def __init__(
        self,
        config: Config = None,
        bench_name: str = "cann",
        incremental_output_path: str = None,
    ):
        self.config = config or get_config()
        self.bench_name = bench_name
        self.bench_config = get_bench_config(self.bench_name)
        # 增量输出路径：子进程模式下，每个用例完成后写入部分结果，
        # 使 OOM Kill 时已完成的用例结果可被主进程恢复
        self.incremental_output_path = incremental_output_path

        # 初始化设备管理器
        device_config = DeviceConfig(
            type=self.config.device_type,
            device_id=self.config.device_id,
        )
        self.device_manager = DeviceManager(device_config)
        self.target_hardware = detect_execution_hardware(
            self.config.device_type, self.config.device_id
        )

        # 初始化性能评测器
        # PerfEvaluator 从 Config.perf_metric_strategy_override 自取策略，
        # 无需外部传入策略实例
        self.perf_evaluator = PerfEvaluator(
            config=self.config,
            device_manager=self.device_manager,
            warmup=self.config.warmup,
            repeat=self.config.repeat,
            archive_prof=True,
            freq_boost=getattr(self.config, "perf_freq_boost", True),
        )

        # 初始化算子执行器
        self.op_runner = OpRunner(self.device_manager, self.perf_evaluator)

        # 初始化精度评测器
        self.accuracy_evaluator = AccuracyEvaluator(
            custom_thresholds=self.config.precision_thresholds,
            checker_name=self.config.checker_name,
        )

        # 初始化数据层组件（通过 Registry 获取）
        loader_kwargs = (
            {"target_hardware": self.target_hardware}
            if self.bench_name == "cann"
            else {}
        )
        self.case_loader = get_case_loader(
            self.bench_name, tasks_root=self.config.tasks_root, **loader_kwargs
        )
        self.golden_loader = get_golden_loader(
            eval_system=self.bench_name,
            bench_root=self.config.tasks_root,
        )
        self.operator_loader = get_task_loader(
            self.bench_name, tasks_root=self.config.tasks_root
        )
        self.data_generator = DataGenerator(
            input_dist=getattr(self.config, "input_dist", "uniform")
        )
        self.param_builder = ParamBuilder(self.golden_loader)

        # 初始化包管理器
        self.package_manager = PackageManager(config=self.config)

        # 初始化拆分模块
        self.operator_matcher = get_operator_matcher(
            eval_system=self.bench_name,
            operator_loader=self.operator_loader,
        )
        self.failure_synthesizer = FailureSynthesizer(self.case_loader)

        self._pypto_pro_jit_isolation = (
            not getattr(self.config, "pypto_pro_outer_case_isolation", False)
            and self._detect_pypto_pro_dispatch()
        )

    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载AI生成的算子函数（委托给 OperatorMatcher）"""
        return self.operator_matcher.load_ai_operator(operator_name)

    def build_tensor_cache(self, case: CaseSpec):
        """Build trusted fixtures without importing/running a submitted candidate.

        V1 excludes get_input hooks, non-contiguous/aliased input layouts and
        device/distributed references. Raw outputs are saved before get_output.
        """
        identity = identity_for(self, case)
        cache = TensorCache(self.config.tensor_cache_dir)
        if not self.config.tensor_cache_dir:
            raise TensorCacheError("Cache build requires a directory")
        existing = cache.load(identity, case)
        if existing is not None:
            return cache.root / existing.key
        inputs = self.data_generator.generate_input_tensors_from_case(
            input_shapes=case.input_shapes,
            dtypes=case.dtypes,
            value_ranges=case.value_ranges,
            seed=cache_case_seed(case, self.config.eval_seed),
        )
        validate_inputs(inputs, case)
        original_hashes = input_hashes(inputs)
        golden_func = self.golden_loader.get_golden_function(case.rel_path)
        oracle_func = self._oracle_reference_func(case.rel_path, golden_func)
        golden_inputs = self._apply_golden_precision(inputs)
        golden_params = self.param_builder.build_call_params(
            oracle_func, case, golden_inputs
        )
        oracle = self.op_runner.run(
            oracle_func,
            golden_params,
            case.get_case_id_str(),
            golden_inputs,
            to_device=False,
            enable_profiler=False,
        )
        if not oracle.success:
            raise TensorCacheError(
                self._format_run_failure("Cache oracle failed", oracle)
            )
        del golden_params, golden_inputs
        if identity["golden_precision"] == "native_cpu":
            native = oracle
        else:
            bench_func = self._bench_reference_func(case.rel_path, golden_func)
            native_inputs = tensors_to_cpu(inputs)
            native_params = self.param_builder.build_call_params(
                bench_func, case, native_inputs
            )
            native = self.op_runner.run(
                bench_func,
                native_params,
                case.get_case_id_str(),
                native_inputs,
                to_device=False,
                enable_profiler=False,
            )
            if not native.success:
                raise TensorCacheError(
                    self._format_run_failure("Cache native reference failed", native)
                )
        if input_hashes(inputs) != original_hashes:
            raise TensorCacheError(
                "Cache v1 does not support reference functions that mutate inputs"
            )
        return cache.write(identity, case, inputs, oracle.outputs, native.outputs)

    def evaluate_case(
        self, case: CaseSpec, *, ai_op_func: Callable = None
    ) -> EvalCaseResult:
        """评测单个用例。

        评测流程：
        1. Golden 参考执行（精度策略由 bench_config.golden_precision 控制）：
           - fp64_cpu: 升精度到 fp64 + CPU 计算，避免 NPU 溢出污染
           - native_cpu: 原始精度在 CPU 上计算
           - native_npu: 原始精度在 NPU 上计算
        2. AI 算子执行（NPU + profiler 采集性能）
        3. 精度对比（checker 三输入：AI 输出、golden 输出、同精度参考输出）
           - fp64_cpu 时同精度参考需单独执行（golden 是 fp64，精度不同）
           - native_cpu/native_npu 时同精度参考直接复用 golden（精度相同）
        """
        case_id_str = case.get_case_id_str()

        if (getattr(self.config, "perf_restore_inputs", True)
                and self.config.enable_profiler) and (
            (case.attrs or {}).get("mc2_distributed", False)
            or self._pypto_pro_jit_isolation
            or getattr(self.config, "pypto_pro_outer_case_isolation", False)
        ):
            return EvalCaseResult(
                case_id=case_id_str, rel_path=case.rel_path, operator=case.operator,
                case_num=case.case_num, success=False,
                error_msg="perf_restore_inputs does not yet support MC2/PyPTO isolated backends",
                failure_type=FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
            )

        # MC2 多卡通信类算子（如 MatmulReduceScatter / AllGatherMatmul 等）需要
        # 多 rank HCCL 进程组，无法走单进程 golden 路径，转交专用分布式 runner。
        if (case.attrs or {}).get("mc2_distributed", False):
            if selected(self.config, case):
                reason = "Tensor cache v1 does not support distributed cases"
                if self.config.tensor_cache_mode == "require":
                    return EvalCaseResult(
                        case_id=case_id_str,
                        rel_path=case.rel_path,
                        operator=case.operator,
                        case_num=case.case_num,
                        success=False,
                        error_msg=reason,
                        failure_type=FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
                    )
                print(
                    f"[tensor-cache] WARNING {case_id_str}: {reason}; computing without cache",
                    flush=True,
                )
            from .mc2_distributed_runner import MC2DistributedEvaluator

            merged_thresholds = self._get_merged_thresholds(case.rel_path)
            return MC2DistributedEvaluator(self.config).evaluate_case(
                case, custom_thresholds=merged_thresholds
            )

        # 本 case 的大对象局部名先声明：下方 finally 统一释放它们，而异常可能发生在
        # 任何一次赋值之前，未声明的名字会让 finally 自己抛 UnboundLocalError。
        input_tensors = params = None
        golden_inputs = golden_params = golden_result = golden_outs = None
        native_inputs = native_params = native_result = native_out = None
        ai_result = ai_outs = None
        cached = None

        try:
            # 1. 获取golden函数
            golden_func = self.golden_loader.get_golden_function(case.rel_path)

            # 1.5 计算确定性种子（默认始终开启，确保评测可复现）
            # eval_seed=0: 基于 case_id hash；eval_seed=N: 用 N+case_id hash
            # eval_seed=None: 纯随机模式（不推荐，会导致 flaky 测试）
            # 注意：使用 hashlib 而非 Python hash()，因为 Python 3 的 hash()
            # 在跨进程时不可复现（PYTHONHASHSEED 随机化）。
            if self.config.eval_seed is not None:
                import hashlib

                # SHA256 hash of case_id_str, 取前 8 字节作为 int，确保跨进程可复现
                digest = hashlib.sha256(case_id_str.encode("utf-8")).digest()
                deterministic_hash = int.from_bytes(digest[:8], byteorder="big") % (
                    2**31
                )
                case_seed = (self.config.eval_seed + deterministic_hash) % (2**31)
            else:
                case_seed = None  # 纯随机模式

            # Selected cache hits bypass generation and both CPU references.
            # Missing/unsupported fixtures can fall back; corrupt fixtures and
            # invalid configuration/source identities always fail.
            if selected(self.config, case):
                try:
                    identity = identity_for(self, case)
                except TensorCacheUnsupportedError as exc:
                    if self.config.tensor_cache_mode == "require":
                        raise
                    print(
                        f"[tensor-cache] WARNING {case_id_str}: {exc}; computing without cache",
                        flush=True,
                    )
                else:
                    cached = TensorCache(self.config.tensor_cache_dir).load(
                        identity, case
                    )
                    if cached is None and self.config.tensor_cache_mode == "require":
                        raise TensorCacheError(
                            f"Required tensor cache missing for {case_id_str}"
                        )
                    print(
                        f"[tensor-cache] {case_id_str}: {'hit ' + cached.key if cached else 'miss; computing'}",
                        flush=True,
                    )
            if cached is not None:
                input_tensors = cached.inputs
            else:
                input_tensors = self.data_generator.generate_input_tensors_from_case(
                    input_shapes=case.input_shapes,
                    dtypes=case.dtypes,
                    value_ranges=case.value_ranges,
                    seed=case_seed,
                )

            # case attrs 无条件先定义，供后续 get_input / 索引校验等多处使用
            # （避免无 get_input 的算子在后续引用 case_attrs 时 NameError）
            case_attrs = getattr(case, "attrs", None) or {}

            # 2.5 调用 get_input 预处理（如果存在）
            get_input_func = self.golden_loader.get_input_function(case.rel_path)
            if get_input_func is not None:
                # cases.yaml 已规范为使用 null 占位符表示省略的 optional 参数
                # input_shapes 与 proto.inputs 长度一致，直接按顺序映射
                op_info = self.operator_loader.get_operator(case.rel_path)

                params_for_get_input = {}
                for i, input_info in enumerate(op_info.inputs):
                    if i < len(input_tensors):
                        params_for_get_input[input_info.name] = input_tensors[i]
                    else:
                        params_for_get_input[input_info.name] = None

                # 添加 attrs
                for attr_key, attr_val in case_attrs.items():
                    if attr_key not in params_for_get_input:
                        params_for_get_input[attr_key] = attr_val
                if "skip2_exist" not in params_for_get_input:
                    params_for_get_input["skip2_exist"] = case_attrs.get(
                        "skip2_exist", True
                    )

                input_tensors = get_input_func(**params_for_get_input)
                if isinstance(input_tensors, tuple):
                    input_tensors = list(input_tensors)

            # 3. 构建调用参数
            golden_sig = signature(golden_func)
            if get_input_func is not None:
                # get_input 已重新排序，input_tensors 现在按 golden 签名顺序排列
                # 直接按位置构建参数
                params = {}
                tensor_idx = 0
                for param_name, param in golden_sig.parameters.items():
                    annotation = (
                        str(param.annotation)
                        if param.annotation != Parameter.empty
                        else ""
                    )
                    # 检查是否是 tensor 参数
                    if "Tensor" in annotation:
                        if tensor_idx < len(input_tensors):
                            val = input_tensors[tensor_idx]
                            params[param_name] = val
                            tensor_idx += 1
                        else:
                            params[param_name] = None
                    # attrs 参数从 case_attrs 获取
                    elif param_name in case_attrs:
                        params[param_name] = case_attrs[param_name]
                    elif param.default != Parameter.empty:
                        params[param_name] = param.default
            else:
                params = self.param_builder.build_call_params(
                    golden_func, case, input_tensors
                )

            # 4. 执行Golden函数获取参考结果（精度策略由 golden_precision 控制）
            # oracle (g)：优先 op 的 <op>_oracle 钩子（dtype-agnostic 真值），否则回退 golden。
            if cached is not None:
                golden_result = OpRunResult(
                    success=True, outputs=cached.oracle, device="cpu"
                )
            else:
                oracle_func = self._oracle_reference_func(case.rel_path, golden_func)
                if (getattr(self.config, "perf_restore_inputs", True)
                        and self.config.enable_profiler):
                    # Clone before golden can mutate integers/native CPU tensors.
                    # Convert unique tensors once so reference aliases survive.
                    golden_inputs, reference_params = clone_reference_inputs(
                        input_tensors, params,
                        lambda tensor: self._apply_golden_precision([tensor])[0])
                else:
                    golden_inputs = self._apply_golden_precision(input_tensors)
                    reference_params = params
                if get_input_func is not None:
                    golden_params = reference_params
                else:
                    golden_params = self.param_builder.build_call_params(
                        oracle_func, case, golden_inputs
                    )
                golden_result = self.op_runner.run(
                    oracle_func,
                    golden_params,
                    case_id_str,
                    golden_inputs,
                    to_device=self._get_golden_to_device(),
                    enable_profiler=False,
                )  # Golden 不启用 profiler
            if not golden_result.success:
                return EvalCaseResult(
                    case_id=case_id_str,
                    rel_path=case.rel_path,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=golden_result,
                    error_msg=self._format_run_failure("Golden执行失败", golden_result),
                    baseline_perf_us=case.baseline_perf_us,
                    baseline_source=(getattr(case, "metadata", None) or {}).get(
                        "baseline_source", {}
                    ),
                    t_hw_us=case.t_hw_us,
                    failure_type=FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
                )

            # 5. 确定使用的 AI 算子函数
            # 优先级：传入参数 > 加载 AI 算子
            actual_ai_func = ai_op_func
            if actual_ai_func is None:
                try:
                    actual_ai_func = self.operator_matcher.load_ai_operator(
                        case.operator
                    )
                except Exception as load_err:
                    return EvalCaseResult(
                        case_id=case_id_str,
                        rel_path=case.rel_path,
                        operator=case.operator,
                        case_num=case.case_num,
                        success=False,
                        golden_run_result=golden_result,
                        error_msg=f"AI算子加载失败: {load_err}",
                        baseline_perf_us=case.baseline_perf_us,
                        baseline_source=(getattr(case, "metadata", None) or {}).get(
                            "baseline_source", {}
                        ),
                        t_hw_us=case.t_hw_us,
                        failure_type=FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
                    )

            # 6. 执行AI算子（profiler 一次运行同时提供输出和性能数据，避免跑两遍）
            use_profiler = (
                self.perf_evaluator is not None
                and self.perf_evaluator.config.enable_profiler
            )

            ai_result = self._run_ai_op_isolated(
                actual_ai_func,
                params,
                case_id_str,
                input_tensors,
                enable_perf=use_profiler,
                operator_name=case.operator,
            )
            if not ai_result.success:
                return EvalCaseResult(
                    case_id=case_id_str,
                    rel_path=case.rel_path,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=golden_result,
                    ai_run_result=ai_result,
                    error_msg=self._format_run_failure("AI算子执行失败", ai_result),
                    baseline_perf_us=case.baseline_perf_us,
                    baseline_source=(getattr(case, "metadata", None) or {}).get(
                        "baseline_source", {}
                    ),
                    t_hw_us=case.t_hw_us,
                    failure_type=FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
                )

            # 7. 精度对比（使用与性能采集同一次运行的输出）
            dtype = self._determine_dtype(ai_result, case)
            merged_thresholds = self._get_merged_thresholds(case.rel_path, dtype)

            # 同精度参考输出（用于 checker 小值域判断）
            # native_cpu/native_npu 时 golden 已是同精度，直接复用以避免重复计算
            golden_strategy = getattr(self.bench_config, "golden_precision", "fp64_cpu")
            if cached is not None:
                native_out = cached.native
            elif golden_strategy in ("native_cpu", "native_npu"):
                native_out = golden_result.outputs
            else:
                # 同精度参考 (b)：优先 <op>_bench（硬件忠实），否则回退 golden（现状行为）。
                bench_func = self._bench_reference_func(case.rel_path, golden_func)
                if (getattr(self.config, "perf_restore_inputs", True)
                        and self.config.enable_profiler):
                    native_inputs, reference_params = clone_reference_inputs(
                        input_tensors, params, lambda tensor: tensor.cpu())
                else:
                    native_inputs = tensors_to_cpu(input_tensors)
                    reference_params = params
                if get_input_func is not None:
                    native_params = reference_params
                else:
                    native_params = self.param_builder.build_call_params(
                        bench_func, case, native_inputs
                    )
                native_result = self.op_runner.run(
                    bench_func,
                    native_params,
                    case_id_str,
                    native_inputs,
                    to_device=False,
                    enable_profiler=False,
                )  # 同精度参考不启用 profiler
                native_out = native_result.outputs if native_result.success else None

            ignore_output_indices = self._get_ignore_output_indices(case.rel_path)

            # 获取算子输出名称（用于填充 SingleOutputResult.name）
            op_info = self.operator_loader.get_operator(case.rel_path)
            output_names = (
                [out.name for out in op_info.outputs]
                if op_info and op_info.outputs
                else []
            )
            output_dtype_contracts = self._get_output_dtype_contracts(op_info)
            output_optional_flags = (
                [bool(getattr(out, "optional", False)) for out in op_info.outputs]
                if op_info and op_info.outputs
                else None
            )

            # 7.5 调用 get_output 后处理（如果存在）
            # 与 get_input 对称：对 golden / AI / native 三路输出统一变换后再对比，
            # 确保变换一致、对比公平。不修改 ai_result.outputs 原始引用，
            # 使 validate_index_gather_outputs 仍能对原始输出做索引自洽校验。
            golden_outs = golden_result.outputs
            ai_outs = ai_result.outputs
            get_output_func = self.golden_loader.get_output_function(case.rel_path)
            if get_output_func is not None:
                golden_outs = self._apply_get_output(
                    get_output_func, golden_outs, op_info, case_attrs
                )
                ai_outs = self._apply_get_output(
                    get_output_func, ai_outs, op_info, case_attrs
                )
                if native_out is not None:
                    native_out = self._apply_get_output(
                        get_output_func, native_out, op_info, case_attrs
                    )

            accuracy_result = self.accuracy_evaluator.evaluate(
                ai_output=ai_outs,
                golden_output=golden_outs,
                dtype=dtype,
                custom_thresholds=merged_thresholds,
                native_output=native_out,
                ignore_output_indices=ignore_output_indices,
                diagnostic_context=self._format_output_diagnostic(ai_result),
                expected_output_dtypes=output_dtype_contracts,
                output_optional_flags=output_optional_flags,
                output_names=output_names,
                output_dtype_source=ai_result.outputs,
            )

            # 防作弊二次验证：用新鲜输入再跑一遍 golden + AI，两次都过才算 pass
            # 只在 config.enable_accuracy_retry=True 且第一次已通过时触发，避免开销
            if accuracy_result.passed and getattr(
                self.config, "enable_accuracy_retry", False
            ):
                accuracy_result = self._retry_with_fresh_inputs(
                    case=case,
                    golden_func=golden_func,
                    ai_op_func=ai_op_func,
                    dtype=dtype,
                    merged_thresholds=merged_thresholds,
                    ignore_output_indices=ignore_output_indices,
                    first_result=accuracy_result,
                    case_seed=case_seed,
                )

            # 索引类输出（TopK/ArgSort 等）的"指向值"校验（tie 顺序无关，见 issue #40）：
            # 候选索引须指向其值输出对应的元素（x.gather(dim, idx) == values）。
            # 仅当 proto 为某输出声明了 index_gather 时才生效，对其他算子零影响。
            if accuracy_result.passed:
                idx_ok, idx_msg = validate_index_gather_outputs(
                    op_info,
                    params,
                    case_attrs,
                    ai_result.outputs,
                )
                if not idx_ok:
                    accuracy_result.passed = False
                    accuracy_result.error_msg = (
                        f"{accuracy_result.error_msg} | {idx_msg}"
                        if accuracy_result.error_msg
                        else idx_msg
                    )

            # 填充 output_results 中的输出名称
            if hasattr(accuracy_result, "output_results") and output_names:
                for i, sr in enumerate(accuracy_result.output_results):
                    if i < len(output_names):
                        sr.name = output_names[i]

            # 8. 性能数据已在上面的 profiler 运行中采集，直接提取
            if accuracy_result.is_passed():
                # perf 仅来自 profiler 路径;非 profiler(--no-perf/CPU)时 perf_result 为 None,
                # 评分侧会把该 case 的 perf 分按 0 计入(不再回退到墙钟)。
                perf_result = ai_result.perf_result
                error_msg = None
            else:
                perf_result = None
                # 使用新的多输出格式显示失败原因
                if (
                    hasattr(accuracy_result, "format_all_outputs")
                    and accuracy_result.output_results
                ):
                    output_details = accuracy_result.format_all_outputs()
                    # 用 " | " 连接多输出摘要，保持单行便于 console 显示
                    output_details_single_line = output_details.replace("\n", " | ")
                    error_msg = f"精度不达标: {output_details_single_line}"
                else:
                    # 兼容旧格式（从 metadata 获取 mere/mare）
                    metadata = accuracy_result.get_metadata()
                    mere = metadata.get("mere", 0.0)
                    mare = metadata.get("mare", 0.0)
                    mare_threshold = (
                        10 * accuracy_result.threshold
                        if accuracy_result.threshold and accuracy_result.threshold > 0
                        else 0
                    )
                    fail_reasons = []
                    # 只有当 mare 或 threshold 非零时才判断相对误差超标
                    if mare > 0 and mare_threshold > 0 and mare >= mare_threshold:
                        fail_reasons.append(
                            f"MARE({mare:.6e}) >= mare_threshold({mare_threshold:.6e})"
                        )
                    if (
                        accuracy_result.threshold
                        and accuracy_result.threshold > 0
                        and mere >= accuracy_result.threshold
                    ):
                        fail_reasons.append(
                            f"MERE({mere:.6e}) >= threshold({accuracy_result.threshold:.6e})"
                        )
                    if accuracy_result.error_msg:
                        fail_reasons.append(accuracy_result.error_msg)
                    error_msg = (
                        f"精度不达标: {', '.join(fail_reasons)}"
                        if fail_reasons
                        else "精度不达标"
                    )

            return EvalCaseResult(
                case_id=case_id_str,
                rel_path=case.rel_path,
                operator=case.operator,
                case_num=case.case_id,
                success=accuracy_result.is_passed(),
                accuracy_result=accuracy_result,
                perf_result=perf_result,
                golden_run_result=golden_result,
                ai_run_result=ai_result,
                error_msg=error_msg,
                baseline_perf_us=case.baseline_perf_us,
                baseline_source=(getattr(case, "metadata", None) or {}).get(
                    "baseline_source", {}
                ),
                t_hw_us=case.t_hw_us,
                failure_type=get_accuracy_failure_type(accuracy_result),
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            return EvalCaseResult(
                case_id=case_id_str,
                rel_path=case.rel_path,
                operator=case.operator,
                case_num=case.case_id,
                success=False,
                error_msg=f"评测异常: {type(e).__name__}: {e}\n{tb_str.rstrip()}",
                failure_type=FAILURE_TYPE_COMPILE_RUNTIME_ERROR,
            )

        finally:
            # 本 case 的大对象（fp64 golden 输入副本、golden/native/AI 输出）在这里统一
            # 释放，而不是等函数返回随帧回收。放在 finally 而不是各个 return 之前，是为了
            # 覆盖全部四条出口：golden 执行失败、AI 算子加载失败、AI 执行失败、评测异常。
            # 异常出口尤其需要——traceback 会一直引用本帧，不置 None 的话这些对象要活到
            # 调用方丢弃 EvalCaseResult 为止，而这正是巨型 case 最可能走的那条路径。
            #
            # 安全性：返回值只消费标量指标（序列化仅取 elapsed_us 等，见 results.py），
            # AccuracyResult 不持有张量；EvalCaseResult 自己持有 golden/ai run result 的
            # 引用，这里置空的只是局部名。
            self._release_outputs(golden_result)
            self._release_outputs(ai_result)
            cached = None
            input_tensors = params = None
            golden_inputs = golden_params = golden_result = golden_outs = None
            reference_params = None
            native_inputs = native_params = native_result = native_out = None
            ai_result = ai_outs = None
            # 紧随的 gc.collect() 让上面的释放立即生效
            self._cleanup_memory()

    def evaluate_operator(
        self, operator: str, rel_path: str, case_filter: Dict = None
    ) -> EvalOperatorResult:
        """评测单个算子

        从磁盘扫描加载用例，然后委托给 run_cases() 执行评测循环。
        """
        cases = self.case_loader.scan_by_operator(operator)
        if case_filter:
            cases = self._filter_cases(cases, case_filter)

        if not cases:
            return EvalOperatorResult(
                rel_path=rel_path,
                operator=operator,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                skipped_cases=0,
                results=[],
                pass_rate=0.0,
                avg_speedup=0.0,
            )

        return self.run_cases(cases, operator, rel_path)

    def _should_batch_perf(self, cases: list) -> bool:
        if getattr(self.config, "perf_restore_inputs", True):
            return False
        if not getattr(self.config, "perf_batch_cases", False):
            return False
        # Pro worker reuse is experimental process/JIT reuse only. The shared
        # profiler currently discards sampled outputs and changes conditioning
        # boundaries; its unprofiled precheck cannot certify measured results.
        if getattr(self.config, "pypto_pro_outer_case_isolation", False):
            return False
        if len(cases) < 2 or self._pypto_pro_jit_isolation:
            return False
        if self.perf_evaluator is None or not self.config.enable_profiler:
            return False
        if not self.device_manager.is_npu_mode():
            return False
        strategy = self.perf_evaluator.perf_metric_strategy
        if strategy is None or strategy.get_strategy_name() != "kernel_details":
            return False
        return not any(
            (getattr(case, "attrs", None) or {}).get("mc2_distributed", False)
            for case in cases
        )

    def run_cases(
        self, cases: list, operator: str, rel_path: str
    ) -> EvalOperatorResult:
        """评测一组已加载的用例

        包含渐进式设备恢复机制：
        - 连续失败后检测 NPU 设备健康状态
        - 轻量恢复（empty_cache + set_device）→ 重量恢复（aclrtResetDevice）
        - 恢复无效时跳过剩余 case 并标记 failure_type="cascade_device"

        供 evaluate_operator()（磁盘加载）和 eval-child（JSON 加载）共用。
        """
        import gc

        cases = filter_cases_for_hardware(cases, getattr(self, "target_hardware", None))
        self.operator_matcher.clear_cache()
        results = []
        batch_perf = self._should_batch_perf(cases)
        if batch_perf:
            self.op_runner.begin_perf_batch()
        reused_pro_worker = (
            getattr(self.config, "pypto_pro_outer_case_isolation", False)
            and len(cases) > 1
        )
        consecutive_failures = 0
        # 设备状态: healthy → recovering → unrecoverable
        device_state = "healthy"
        skipped_count = 0
        print(f"[INFO] 评测算子 {operator} ({rel_path}), 用例数: {len(cases)}")

        for i, case in enumerate(cases, 1):
            case_id_str = case.get_case_id_str()

            if device_state == "unrecoverable":
                # 设备不可恢复，剩余 case 直接标记跳过
                result = EvalCaseResult(
                    case_id=case_id_str,
                    rel_path=case.rel_path,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    error_msg="设备不可恢复，用例跳过",
                    failure_type="cascade_device",
                    skipped_before_execution=reused_pro_worker,
                )
                results.append(result)
                skipped_count += 1
                print(f"[{i}/{len(cases)}] {case_id_str}: ⏭️ 设备不可恢复，跳过")
                continue

            with _CaseMemoryMonitor(
                bool(getattr(self.config, "monitor_case_memory", False))
            ) as memory_monitor:
                result = self.evaluate_case(case)
            result.memory_peak_mb = memory_monitor.peak_mb
            result.memory_baseline_mb = memory_monitor.baseline_mb
            results.append(result)
            if batch_perf and not result.success:
                self.op_runner.cancel_perf_batch_case(case_id_str)

            # 增量输出：子进程模式下，每个用例完成后刷新写入部分结果
            # 使 OOM Kill 时已完成的用例结果可被主进程恢复
            if self.incremental_output_path:
                self._write_incremental_output(operator, rel_path, results, len(cases))

            perf_failed_in_pro = (
                reused_pro_worker
                and result.perf_result is not None
                and not result.perf_result.is_success()
            )
            if (
                not result.success
                and result.failure_type != FAILURE_TYPE_REFERENCE_INVALID
            ) or perf_failed_in_pro:
                consecutive_failures += 1

                # 增强 cleanup：empty_cache + gc.collect
                self._cleanup_memory()
                gc.collect()

                # 检查设备健康状态
                if not self.device_manager.health_check():
                    if reused_pro_worker:
                        # A poisoned Pro context must not produce verdicts for
                        # later cases. The parent retries unexecuted cases in
                        # fresh workers while retaining the actual failure.
                        device_state = "unrecoverable"
                        print(
                            "[WARN] Pro worker 设备不健康，停止后续执行并交给父进程隔离恢复"
                        )
                    # 仅在连续 ≥3 失败且设备不健康时触发恢复尝试
                    # 避免偶发的单个 case 失败触发不必要的设备 reset
                    elif consecutive_failures >= 3 and device_state == "healthy":
                        print(
                            f"[WARN] 连续 {consecutive_failures} 个 case 失败，"
                            f"NPU 设备可能异常，尝试恢复..."
                        )

                        # 渐进恢复：先 light，失败再 full
                        # profiler 活跃时不尝试 recover_full（aclrtResetDevice
                        # 会破坏 profiler 的 ACL profiling 资源）
                        profiler_active = (
                            self.perf_evaluator is not None
                            and self.perf_evaluator.config.enable_profiler
                            and self.device_manager.is_npu_mode()
                        )

                        if self.device_manager.recover_light():
                            if self.device_manager.health_check():
                                device_state = "recovering"
                                print("[INFO] 轻量恢复成功，继续评测")
                            else:
                                # light 恢复了但设备仍不健康 → 尝试 full（仅 profiler 未活跃时）
                                if (
                                    not profiler_active
                                    and self.device_manager.recover_full()
                                ):
                                    if self.device_manager.health_check():
                                        device_state = "recovering"
                                        print(
                                            "[INFO] 重量恢复成功（aclrtResetDevice），继续评测"
                                        )
                                    else:
                                        device_state = "unrecoverable"
                                else:
                                    if profiler_active:
                                        print(
                                            "[WARN] profiler 活跃，跳过 aclrtResetDevice"
                                        )
                                    device_state = "unrecoverable"
                        else:
                            # light 恢复失败 → 尝试 full（仅 profiler 未活跃时）
                            if (
                                not profiler_active
                                and self.device_manager.recover_full()
                            ):
                                if self.device_manager.health_check():
                                    device_state = "recovering"
                                    print(
                                        "[INFO] 重量恢复成功（aclrtResetDevice），继续评测"
                                    )
                                else:
                                    device_state = "unrecoverable"
                            else:
                                device_state = "unrecoverable"

                        if device_state == "unrecoverable":
                            remaining = len(cases) - i
                            print(
                                f"[WARN] NPU 设备不可恢复，剩余 {remaining} 个用例跳过"
                            )
                        else:
                            # 恢复成功：重置 consecutive_failures，让下一个 case 有全新起点。
                            # 不在本次迭代中检查 recovering 状态——本次 case 的失败
                            # 是触发恢复的原因，不是恢复后仍失败的证据。
                            consecutive_failures = 0

                # recovering 状态下如果**下一个** case 继续失败 → 恢复无效
                # 注意：此检查仅在 consecutive_failures > 0 时触发（即恢复后有新失败），
                # 恢复成功时 consecutive_failures 已被重置为 0，所以不会在同一次迭代误判。
                if device_state == "recovering" and consecutive_failures > 0:
                    device_state = "unrecoverable"
                    remaining = len(cases) - i
                    print(
                        f"[WARN] 恢复后仍失败，设备不可恢复，剩余 "
                        f"{remaining} 个用例跳过"
                    )

            else:
                consecutive_failures = 0
                if device_state == "recovering":
                    # 恢复后第一个 case 成功 → 设备恢复正常
                    device_state = "healthy"
                    print("[INFO] 恢复后评测成功，设备状态恢复正常")

            # 打印进度
            status_icon = "✅" if result.success else "❌"
            elapsed_str = self._format_elapsed(result)
            speedup_str = (
                f"{result.get_speedup():.2f}x" if result.get_speedup() > 0 else "N/A"
            )

            # 添加精度信息
            if result.success and result.accuracy_result:
                acc = result.accuracy_result
                if hasattr(acc, "output_results") and acc.output_results:
                    output_summaries = [
                        sr.format_summary() for sr in acc.output_results
                    ]
                    acc_str = ", " + ", ".join(output_summaries)
                else:
                    metadata = acc.get_metadata()
                    mare = metadata.get("mare")
                    mere = metadata.get("mere")
                    mare_str = f"MARE={mare:.6f}" if mare is not None else ""
                    mere_str = f"MERE={mere:.6f}" if mere is not None else ""
                    acc_str = (
                        f", {mare_str}, {mere_str}" if mare_str or mere_str else ""
                    )
                print(
                    f"[{i}/{len(cases)}] {case_id_str}: {status_icon} "
                    f"(耗时: {elapsed_str}, 加速比: {speedup_str}{acc_str})"
                )
            elif result.success:
                print(
                    f"[{i}/{len(cases)}] {case_id_str}: {status_icon} (耗时: {elapsed_str}, 加速比: {speedup_str})"
                )
            else:
                # 错误信息可能包含多行，只显示第一行（通常包含关键错误原因）
                error_hint = result.error_msg.split("\n")[0] if result.error_msg else ""
                failure_tag = ""
                if result.failure_type == "cascade_device":
                    failure_tag = " [级联失败-设备]"
                elif result.failure_type == "oom_killed":
                    failure_tag = " [OOM Kill]"
                elif result.failure_type == "skipped":
                    failure_tag = " [跳过]"
                elif result.failure_type == FAILURE_TYPE_REFERENCE_INVALID:
                    failure_tag = " [参考失效，评分不可用]"
                print(
                    f"[{i}/{len(cases)}] {case_id_str}: {status_icon}{failure_tag} {error_hint}"
                )

        if batch_perf:
            try:
                perf_results = self.op_runner.finalize_perf_batch()
            except Exception as exc:
                perf_results = {}
                print(
                    f"[WARN] 批量性能采集异常，结果按未采集处理: "
                    f"{type(exc).__name__}: {exc}"
                )
            for result in results:
                perf_result = perf_results.get(result.case_id)
                if not result.success or perf_result is None:
                    continue
                result.perf_result = perf_result
                if result.ai_run_result is not None:
                    result.ai_run_result.perf_result = perf_result
                    result.ai_run_result.elapsed_us = perf_result.elapsed_us
            if self.incremental_output_path:
                self._write_incremental_output(
                    operator,
                    rel_path,
                    results,
                    len(cases),
                )
            measured = sum(
                1
                for result in results
                if result.perf_result is not None and result.perf_result.elapsed_us > 0
            )
            fallback = sum(
                1
                for result in results
                if result.perf_result is not None
                and result.perf_result.metadata.get("perf_batch_fallback")
            )
            print(
                f"[INFO] 批量性能采集完成: {measured}/{len(results)} 有效, "
                f"{fallback} 个 case 回退逐 case"
            )

        passed = sum(1 for r in results if r.success)
        failed = sum(
            1
            for r in results
            if not r.success
            and r.failure_type
            not in ("cascade_device", "skipped", FAILURE_TYPE_REFERENCE_INVALID)
        )
        speedups = [
            r.get_speedup() for r in results if r.success and r.get_speedup() > 0
        ]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0
        reference_invalid = sum(
            1
            for result in results
            if not result.success
            and result.failure_type == FAILURE_TYPE_REFERENCE_INVALID
        )
        valid_cases = len(cases) - reference_invalid

        return EvalOperatorResult(
            rel_path=rel_path,
            operator=operator,
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=failed,
            skipped_cases=skipped_count,
            results=results,
            pass_rate=passed / valid_cases if valid_cases > 0 else 0.0,
            avg_speedup=avg_speedup,
        )

    def _write_incremental_output(
        self,
        operator: str,
        rel_path: str,
        results: list,
        total_cases: int,
    ) -> None:
        """增量写入部分结果到 incremental_output_path。

        eval-child 子进程模式下，每个用例完成后调用此方法刷新写入当前已完成的结果。
        当子进程被 OOM Kill (SIGKILL) 时，已完成用例的结果已写入文件，
        主进程 (ProcessPoolCoordinator) 可从中恢复部分结果，而非全部标记为失败。

        输出格式为 {"case_results": [...]} — 与 ProcessPoolCoordinator 的
        eval-child 结果读取格式一致，便于 OOM 恢复时直接解析。
        """
        if not self.incremental_output_path:
            return

        payload = {"case_results": [r.to_dict() for r in results]}
        _write_json_atomic(self.incremental_output_path, payload)

    def _has_selected_cases(self, operator_filter=None, case_filter=None) -> bool:
        cases = self.case_loader.scan_all()
        if operator_filter:
            names = {name.lower() for name in operator_filter}
            cases = [case for case in cases if case.operator.lower() in names]
        if case_filter:
            cases = self._filter_cases(cases, case_filter)
        return bool(cases)

    def evaluate_from_source(
        self,
        source_dir: str,
        operator_filter: List[str] = None,
        case_filter: Dict = None,
        verbose: bool = False,
        op_timeout_sec: int = 240,
        case_timeout_sec: int = None,
        iterative_compile: bool = True,
    ) -> EvalSessionResult:
        """从源码目录执行完整评测"""
        print("")
        print("=" * 60)
        print("开始评测")
        print("=" * 60)

        if not self._has_selected_cases(operator_filter, case_filter):
            return EvalSessionResult(operators=[])

        # 1. 准备环境（编译安装）
        matched_operators, package_info = self.package_manager.prepare_from_source(
            source_dir,
            verbose=verbose,
            iterative_compile=iterative_compile,
        )

        # 1.5 整体编译失败：不隔离/不补救，本次提交相关算子全部按编译失败计 0 分
        # （与 cli 共用 synthesize_all_compile_failures，逻辑单一真源）。
        if getattr(package_info, "build_failed", False):
            print(
                "[ERROR] 编译失败，本次提交相关算子按编译失败计 0 分（错误汇总见 _compile.log）"
            )
            results = self.failure_synthesizer.synthesize_all_compile_failures(
                self.operator_matcher,
                package_info,
                operator_filter=operator_filter,
                case_filter=case_filter,
                filter_func=self._filter_cases,
            )
            return EvalSessionResult(
                operators=[result for result in results if result is not None],
                package_info=package_info,
            )

        # 2. APIGuard 验证
        from ..security.api_guard import APIGuard

        guard = APIGuard()
        try:
            guard.verify()
        except RuntimeError as e:
            print(f"[ERROR] APIGuard 检测到 Timing API 篡改: {e}")
            results = []
            for operator_name in matched_operators:
                op_info = self.operator_matcher.find_operator_info(operator_name)
                if op_info:
                    result = self.failure_synthesizer.synthesize_security_failure(
                        op_info,
                        str(e),
                        case_filter,
                        self._filter_cases,
                    )
                    results.append(result)
            for snake_op_name, err in (package_info.compile_errors or {}).items():
                op_info = self.operator_matcher.find_operator_info_by_snake(
                    snake_op_name
                )
                if op_info and (not operator_filter or op_info.name in operator_filter):
                    results.append(
                        self.failure_synthesizer.synthesize_compile_failure(
                            op_info,
                            err,
                            case_filter,
                            self._filter_cases,
                        )
                    )
            return EvalSessionResult(
                operators=[result for result in results if result is not None],
                package_info=package_info,
            )

        # 3. 合成编译失败结果
        compile_failed_results: List[EvalOperatorResult] = []
        for snake_op_name, err in (package_info.compile_errors or {}).items():
            op_info = self.operator_matcher.find_operator_info_by_snake(snake_op_name)
            if op_info is None:
                # F006: 编译失败的 snake_name 在 OperatorMatcher 找不到对应 CannTaskSpec
                # 时旧代码静默 continue，导致这条编译失败不出现在最终报告，看起来
                # 像"没编译就没编译" 实际是失踪。加 WARN log 让运维 / Agent 能注意到。
                # 可能原因：tasks/ 目录有新算子但 spec 未注册 / 命名约定不一致
                # （新 op 用 PascalCase 但 build 输出 snake_case 的 .so 时未 lookup）。
                print(
                    f"[WARN] evaluator: 编译失败算子 {snake_op_name!r} 未在 OperatorMatcher 中找到 "
                    f"对应 CannTaskSpec，已跳过合成失败结果——该算子不会出现在最终报告中。"
                    f"请核查 tasks/<level>/<op>/proto.yaml 是否已注册。",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if operator_filter and op_info.name not in operator_filter:
                continue
            compile_failed_results.append(
                self.failure_synthesizer.synthesize_compile_failure(
                    op_info, err, case_filter, self._filter_cases
                ),
            )

        compile_failed_results = [
            result for result in compile_failed_results if result is not None
        ]
        if not matched_operators and not compile_failed_results:
            return EvalSessionResult(operators=[], package_info=package_info)

        # 4. 应用算子筛选
        if operator_filter:
            matched_operators = [
                op for op in matched_operators if op in operator_filter
            ]
            print(f"[INFO] 筛选后算子: {matched_operators}")

        # 5. 逐个评测算子（in-process 执行，不再 fork 子进程）
        results: List[EvalOperatorResult] = list(compile_failed_results)
        print(
            f"[INFO] 编译失败: {len(compile_failed_results)} 个算子 | 可运行: "
            f"{len(matched_operators)} 个算子"
        )

        for operator_name in matched_operators:
            op_info = self.operator_matcher.find_operator_info(operator_name)
            if not op_info:
                print(f"[WARN] 算子 {operator_name} 未找到定义，跳过")
                continue

            result = self.evaluate_operator(
                operator=operator_name,
                rel_path=op_info.rel_path,
                case_filter=case_filter,
            )
            if result.total_cases:
                results.append(result)

        print("")
        print("=" * 60)
        print("评测完成")
        print("=" * 60)

        return EvalSessionResult(operators=results, package_info=package_info)

    def evaluate_skip_build(
        self,
        operator_filter: List[str] = None,
        case_filter: Dict = None,
    ) -> EvalSessionResult:
        """跳过编译安装，直接评测已安装的cann_bench（in-process 执行）"""
        print("")
        print("=" * 60)
        print("开始评测（跳过编译安装）")
        print("=" * 60)

        if not self._has_selected_cases(operator_filter, case_filter):
            return EvalSessionResult(operators=[])

        matched_operators = self.package_manager.prepare_skip_build()
        if not matched_operators:
            return EvalSessionResult(operators=[])

        if operator_filter:
            matched_operators = [
                op for op in matched_operators if op in operator_filter
            ]
            print(f"[INFO] 筛选后算子: {matched_operators}")

        results = []
        print(f"[INFO] 可运行算子: {len(matched_operators)}")

        for operator_name in matched_operators:
            op_info = self.operator_matcher.find_operator_info(operator_name)
            if not op_info:
                print(f"[WARN] 算子 {operator_name} 未找到定义，跳过")
                continue

            result = self.evaluate_operator(
                operator=operator_name,
                rel_path=op_info.rel_path,
                case_filter=case_filter,
            )
            if result.total_cases:
                results.append(result)

        print("")
        print("=" * 60)
        print("评测完成")
        print("=" * 60)

        return EvalSessionResult(operators=results)

    # ---- 辅助方法 ----

    def _oracle_reference_func(self, rel_path: str, golden_func):
        """fp64 oracle (g) 用的函数:优先 op 的 <op>_oracle 钩子(dtype-agnostic 真值),
        否则回退 golden。回退分支与历史行为一致。"""
        return self.golden_loader.get_oracle_function(rel_path) or golden_func

    def _bench_reference_func(self, rel_path: str, golden_func):
        """同精度参考 (b) 用的函数:优先 op 的 <op>_bench 钩子(硬件忠实精度),否则回退 golden。
        回退分支与历史行为一致 —— 未提供 bench 的算子 b 仍 = golden(case dtype)。"""
        return self.golden_loader.get_bench_function(rel_path) or golden_func

    def _apply_golden_precision(self, input_tensors: List) -> List:
        """根据 bench 配置的 golden_precision 策略转换输入张量。

        取值：
          - fp64_cpu（默认）: 升精度到 float64 + CPU 计算，避免 NPU 溢出污染
          - native_cpu: 保持原始精度在 CPU 上计算
          - native_npu: 保持原始精度在 NPU 上计算
        """
        strategy = getattr(self.bench_config, "golden_precision", "fp64_cpu")
        if strategy == "fp64_cpu":
            return tensors_to_fp64_cpu(input_tensors)
        elif strategy == "native_cpu":
            return tensors_to_cpu(input_tensors)
        elif strategy == "native_npu":
            return list(input_tensors)
        return list(input_tensors)

    def _get_golden_to_device(self) -> bool:
        """golden 执行时 to_device 参数：仅 native_npu 在 NPU 上计算时为 True"""
        strategy = getattr(self.bench_config, "golden_precision", "fp64_cpu")
        return strategy == "native_npu"

    def _determine_dtype(self, ai_result, case) -> str:
        """确定 dtype"""
        dtype = None
        if isinstance(ai_result.outputs, torch.Tensor):
            dtype = str(ai_result.outputs.dtype).replace("torch.", "")
        elif isinstance(ai_result.outputs, (list, tuple)) and ai_result.outputs:
            first_out = ai_result.outputs[0]
            if isinstance(first_out, torch.Tensor):
                dtype = str(first_out.dtype).replace("torch.", "")

        if dtype is None:
            try:
                op_info = self.operator_loader.get_operator(case.rel_path)
                if op_info and op_info.outputs and op_info.outputs[0].dtype:
                    output_dtype_list = op_info.outputs[0].dtype
                    dtype = (
                        output_dtype_list[0]
                        if isinstance(output_dtype_list, list)
                        else output_dtype_list
                    )
            except Exception as e:
                # 静默吞掉会让 proto.yaml 缺失/损坏的算子退回到 case.dtypes[0]，
                # 可能选错精度阈值。至少记录可见的告警。
                print(
                    f"[WARN] _determine_dtype: 读取 {case.rel_path} 算子定义失败({type(e).__name__}: {e})，回退到 case.dtypes"
                )

        return dtype or (case.dtypes[0] if case.dtypes else "float32")

    def _format_elapsed(self, result) -> str:
        """格式化耗时"""
        if result.success and result.perf_result and result.perf_result.elapsed_us > 0:
            return f"{result.perf_result.elapsed_us:.2f}μs"
        elif result.ai_run_result and result.ai_run_result.elapsed_us > 0:
            return f"{result.ai_run_result.elapsed_us:.2f}μs"
        return "N/A"

    @staticmethod
    def _format_run_failure(prefix: str, run_result: OpRunResult) -> str:
        error = run_result.error or "未知错误"
        message = f"{prefix}: {error}"
        tb_str = (run_result.traceback or "").rstrip()
        if tb_str and tb_str not in message:
            message = f"{message}\n{tb_str}"
        return message

    @classmethod
    def _format_output_diagnostic(cls, run_result: OpRunResult) -> str:
        parts = [
            f"AI算子输出类型: {type(run_result.outputs).__name__}",
            f"AI算子输出结构: {cls._structure_summary(run_result.outputs)}",
        ]
        if run_result.error:
            parts.append(f"AI算子执行错误: {run_result.error}")
        if run_result.traceback:
            parts.append(f"AI算子执行 traceback:\n{run_result.traceback.rstrip()}")
        elif run_result.outputs is None:
            parts.append(
                "AI算子未抛异常但返回 None；请检查是否漏写 return 或使用了不匹配的 out-param 接口。"
            )
        return "\n".join(parts)

    @staticmethod
    def _structure_summary(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, torch.Tensor):
            return f"Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})"
        if isinstance(value, (list, tuple)):
            type_name = type(value).__name__
            items = [Evaluator._structure_summary(v) for v in value]
            return f"{type_name}({', '.join(items)})"
        return f"{type(value).__name__}({value!r})"

    def _filter_cases(self, cases: List[CaseSpec], filter_dict: Dict) -> List[CaseSpec]:
        """筛选用例"""
        result = cases
        if "case_id" in filter_dict:
            # case_num 是 CannCaseSpec 的特化字段，通过 hasattr 兼容
            result = [
                c
                for c in result
                if hasattr(c, "case_num") and c.case_num == filter_dict["case_id"]
            ]
        if "case_id_range" in filter_dict:
            start, end = filter_dict["case_id_range"]
            result = [
                c
                for c in result
                if hasattr(c, "case_num") and start <= c.case_num <= end
            ]
        if "dtype" in filter_dict:
            result = [
                c
                for c in result
                if filter_dict["dtype"].lower() in [d.lower() for d in c.dtypes]
            ]
        return result

    def _get_merged_thresholds(
        self, rel_path: str, dtype: str = None
    ) -> Dict[str, float]:
        """获取合并后的精度阈值

        合并优先级（低 → 高）：
          1. config.precision_thresholds（评测集默认 atol/rtol）
          2. bench_config.dtype_tolerance_map（按 dtype 注入 atol=rtol，
             对齐原生 KernelBench 精度分档：fp32=1e-4, fp16/bf16=1e-2）
          3. op_info.precision_thresholds（算子级显式覆盖，最高）
        """
        merged = dict(self.config.precision_thresholds)
        # 按 dtype 注入容差（仅当 bench 配置了 dtype_tolerance_map 且已知 dtype）
        dtype_map = getattr(self.bench_config, "dtype_tolerance_map", None)
        if dtype_map and dtype:
            tol = dtype_map.get(dtype) or dtype_map.get(dtype.lower())
            if tol is not None:
                merged["atol"] = tol
                merged["rtol"] = tol
        op_info = self.operator_loader.get_operator(rel_path)
        if op_info and op_info.precision_thresholds:
            merged.update(op_info.precision_thresholds)
        return merged

    def _get_ignore_output_indices(self, rel_path: str) -> List[int]:
        """获取需要忽略对比的输出索引"""
        ignore_indices = []
        op_info = self.operator_loader.get_operator(rel_path)
        if op_info and op_info.outputs:
            for idx, output in enumerate(op_info.outputs):
                if not output.compare:
                    ignore_indices.append(idx)
        return ignore_indices

    @staticmethod
    def _get_output_dtype_contracts(op_info) -> Optional[List[List[str]]]:
        """读取 proto 输出允许的 dtype；支持 CANN spec 的 dtypes 和通用 dtype 字段。"""
        if not op_info or not op_info.outputs:
            return None

        contracts = []
        for output in op_info.outputs:
            dtypes = getattr(output, "dtypes", None) or getattr(output, "dtype", None)
            if isinstance(dtypes, str):
                dtypes = [dtypes] if dtypes else []
            contracts.append(list(dtypes or []))
        return contracts

    @staticmethod
    def _apply_get_output(get_output_func, outputs, op_info, case_attrs):
        """调用 get_output 对算子输出进行后处理，返回变换后的输出列表。

        与 get_input 对称：按 proto.yaml 声明的输出名 + attrs 作为 kwargs 传入，
        返回值规整为 list。对 golden / AI / native 三路输出分别调用同一函数，
        确保变换一致、对比公平。
        """
        if outputs is None:
            return None
        outs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

        kwargs = {}
        if op_info and op_info.outputs:
            for i, output_info in enumerate(op_info.outputs):
                kwargs[output_info.name] = outs[i] if i < len(outs) else None
        if case_attrs:
            for attr_key, attr_val in case_attrs.items():
                if attr_key not in kwargs:
                    kwargs[attr_key] = attr_val

        transformed = get_output_func(**kwargs)
        if isinstance(transformed, tuple):
            return list(transformed)
        if isinstance(transformed, list):
            return transformed
        return [transformed]

    def _retry_with_fresh_inputs(
        self,
        case,
        golden_func,
        ai_op_func,
        dtype,
        merged_thresholds,
        ignore_output_indices,
        first_result,
        case_seed=None,
    ):
        """防作弊二次验证：用一组新鲜（微扰过的）输入再跑一遍 golden + AI，
        两次都过才记为 pass；任何一次失败把 first_result 替换成失败 result。

        启用条件：Config.enable_accuracy_retry=True 且第一轮已通过。
        参考 AccuracyEvaluator.evaluate_with_retry 的设计思路。

        Args:
            case_seed: 第一轮使用的确定性种子。二次验证使用偏移后的种子。
        """
        try:
            import torch

            # 二次验证使用偏移种子（+1），确保输入不同但可复现
            retry_seed = (case_seed + 1) if case_seed is not None else None
            fresh_inputs = self.data_generator.generate_input_tensors_from_case(
                input_shapes=case.input_shapes,
                dtypes=case.dtypes,
                value_ranges=case.value_ranges,
                seed=retry_seed,
            )

            # 微扰：浮点输入加 0.01，防止 seed 偶然重合导致两次 inputs 相同
            # F011: was perturbing only the first floating tensor (break after
            # the first match), so a cheater that inspects later inputs to
            # detect a re-run could still slip through. Perturb every
            # floating-point tensor at every nesting level.
            for item in fresh_inputs:
                if isinstance(item, torch.Tensor) and item.is_floating_point():
                    item.add_(0.01)
                elif isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, torch.Tensor) and sub.is_floating_point():
                            sub.add_(0.01)

            # 重建 params + 跑 golden + 跑 AI
            case_id_str = case.get_case_id_str()
            params = self.param_builder.build_call_params(
                golden_func, case, fresh_inputs
            )
            oracle_func = self._oracle_reference_func(case.rel_path, golden_func)
            golden_inputs = self._apply_golden_precision(fresh_inputs)
            golden_params = self.param_builder.build_call_params(
                oracle_func, case, golden_inputs
            )
            golden_result2 = self.op_runner.run(
                oracle_func,
                golden_params,
                case_id_str,
                golden_inputs,
                to_device=self._get_golden_to_device(),
                enable_profiler=False,
            )  # 二次验证 Golden 不启用 profiler
            if not golden_result2.success:
                return first_result  # golden 自己挂了，第二轮无意义，保持第一轮结果

            # 二次验证 AI 算子也不启用 profiler，仅验证精度
            ai_result2 = self.op_runner.run(
                ai_op_func, params, case_id_str, fresh_inputs, enable_profiler=False
            )
            if not ai_result2.success:
                from ..base.result import AccuracyResult

                return AccuracyResult(
                    passed=False,
                    threshold=first_result.threshold or 0,
                    error_msg=f"二次验证失败：AI 算子崩溃 ({ai_result2.error})",
                )

            # 与主流程一致：若存在 get_output 则对 golden / AI 输出统一变换后再对比
            golden_outs2 = golden_result2.outputs
            ai_outs2 = ai_result2.outputs
            op_info = self.operator_loader.get_operator(case.rel_path)
            output_dtype_contracts = self._get_output_dtype_contracts(op_info)
            output_optional_flags = (
                [bool(getattr(out, "optional", False)) for out in op_info.outputs]
                if op_info and op_info.outputs
                else None
            )
            output_names = (
                [out.name for out in op_info.outputs]
                if op_info and op_info.outputs
                else []
            )

            # 同精度参考 (b)：与主流程同构——优先 <op>_bench，否则回退 golden。
            # issue #311：此前复验硬传 native_output=None，compare 的小值域/相消
            # 兜底分母恒 0，依赖兜底营救的 case 会"首验过、复验必挂"。
            golden_strategy = getattr(self.bench_config, "golden_precision", "fp64_cpu")
            if golden_strategy in ("native_cpu", "native_npu"):
                native_out2 = golden_outs2
            else:
                bench_func = self._bench_reference_func(case.rel_path, golden_func)
                native_inputs2 = tensors_to_cpu(fresh_inputs)
                if self.golden_loader.get_input_function(case.rel_path) is not None:
                    native_params2 = params
                else:
                    native_params2 = self.param_builder.build_call_params(
                        bench_func, case, native_inputs2
                    )
                native_result2 = self.op_runner.run(
                    bench_func,
                    native_params2,
                    case_id_str,
                    native_inputs2,
                    to_device=False,
                    enable_profiler=False,
                )
                native_out2 = native_result2.outputs if native_result2.success else None

            get_output_func = self.golden_loader.get_output_function(case.rel_path)
            if get_output_func is not None:
                retry_attrs = getattr(case, "attrs", None) or {}
                golden_outs2 = self._apply_get_output(
                    get_output_func, golden_outs2, op_info, retry_attrs
                )
                ai_outs2 = self._apply_get_output(
                    get_output_func, ai_outs2, op_info, retry_attrs
                )
                if native_out2 is not None:
                    native_out2 = self._apply_get_output(
                        get_output_func, native_out2, op_info, retry_attrs
                    )

            return self.accuracy_evaluator.evaluate(
                ai_output=ai_outs2,
                golden_output=golden_outs2,
                dtype=dtype,
                custom_thresholds=merged_thresholds,
                native_output=native_out2,
                ignore_output_indices=ignore_output_indices,
                expected_output_dtypes=output_dtype_contracts,
                output_optional_flags=output_optional_flags,
                output_names=output_names,
                output_dtype_source=ai_result2.outputs,
            )
        except Exception as e:
            # 二次验证基础设施异常不应整体阻断评测；记 warn 并返回第一轮
            print(
                f"[WARN] enable_accuracy_retry 二次验证基础设施异常 ({case.rel_path}): {e}",
                flush=True,
            )
            return first_result

    def _run_ai_op_isolated(
        self, ai_op_func, params, case_id_str, input_tensors, enable_perf, operator_name
    ):
        """执行 AI 算子。

        pypto_pro 模式下用独立子进程执行每个 case，确保 pypto_pro runtime
        的进程内全局状态完全隔离，避免不同 class kernel 连续执行时的
        AICore 异常（error 507035）。

        非 pypto_pro 模式回退到 op_runner 直接执行。
        profiler 模式下子进程内启动 torch_npu.profiler，trace 写入 prof_dir，
        父进程解析 prof 文件构造 PerfResult。
        """
        if not self._pypto_pro_jit_isolation:
            return self.op_runner.run_ai_op(
                ai_op_func, params, case_id_str, input_tensors, enable_perf=enable_perf
            )

        import json
        import pickle
        import subprocess

        work_dir = tempfile.mkdtemp(prefix="pypto_sub_")
        prof_dir = None
        try:
            func_name = getattr(ai_op_func, "__name__", None) or operator_name.lower()
            with open(os.path.join(work_dir, "input.pkl"), "wb") as f:
                pickle.dump(input_tensors, f)
            with open(os.path.join(work_dir, "params.pkl"), "wb") as f:
                pickle.dump(params, f)

            child_script = os.path.join(os.path.dirname(__file__), "pypto_pro_child.py")
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(self.config.device_id)
            env["PYTHONUNBUFFERED"] = "1"

            # profiler 模式: 创建 prof_dir 并传给子进程
            warmup = 3
            repeat = 5
            profiler_level = getattr(self.config, "profiler_level", "Level1")
            freq_boost = getattr(self.config, "perf_freq_boost", True)
            if enable_perf and self.perf_evaluator is not None:
                warmup = self.perf_evaluator.warmup
                repeat = self.perf_evaluator.repeat
                rel_path, caseid = self.perf_evaluator._parse_case_id(case_id_str)
                if self.perf_evaluator.archive_prof:
                    prof_dir = os.path.join(
                        self.perf_evaluator.prof_data_dir, rel_path, caseid
                    )
                else:
                    prof_dir = tempfile.mkdtemp(prefix="cann_prof_")
                os.makedirs(prof_dir, exist_ok=True)
                # 清理上次评测遗留的时间戳子目录
                for entry in os.listdir(prof_dir):
                    entry_path = os.path.join(prof_dir, entry)
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path, ignore_errors=True)

            cmd = [
                sys.executable,
                "-u",
                child_script,
                work_dir,
                func_name,
                str(self.config.device_id),
            ]
            if prof_dir:
                cmd.extend(
                    [
                        prof_dir,
                        str(warmup),
                        str(repeat),
                        profiler_level,
                        "1" if freq_boost else "0",
                    ]
                )
            else:
                cmd.extend(["", "", "", profiler_level, "1" if freq_boost else "0"])

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            timeout = max(1, getattr(self.config, "timeout_per_operator", 300) - 10)
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _terminate_process_group(proc)
                stdout, stderr = proc.communicate()
                captured = (
                    stdout.decode(errors="replace") + stderr.decode(errors="replace")
                ).strip()
                return OpRunResult(
                    success=False,
                    error=f"PyPTO Pro 子进程超时 ({timeout}s)，已终止进程组",
                    elapsed_us=0,
                    device=f"npu:{self.config.device_id}",
                    captured_output=captured if captured else None,
                )
            rc = proc.returncode

            result_path = os.path.join(work_dir, "result.json")
            if os.path.exists(result_path):
                with open(result_path) as f:
                    result = json.load(f)
            else:
                result = {
                    "success": False,
                    "error": f"子进程退出 code={rc}\n{stderr.decode()[:2000]}",
                    "traceback": None,
                }

            device_str = f"npu:{self.config.device_id}"
            captured = (stdout.decode() + stderr.decode()).strip()

            if result["success"]:
                with open(os.path.join(work_dir, "output.pkl"), "rb") as f:
                    outputs = pickle.load(f)

                # profiler 模式: 解析 prof 文件构造 PerfResult
                perf_result = None
                if prof_dir and self.perf_evaluator is not None:
                    perf_result = self._parse_prof_result(prof_dir, case_id_str)

                return OpRunResult(
                    success=True,
                    outputs=outputs,
                    elapsed_us=perf_result.elapsed_us if perf_result else 0,
                    perf_result=perf_result,
                    device=device_str,
                    captured_output=captured if captured else None,
                )
            else:
                return OpRunResult(
                    success=False,
                    error=result.get("error", "unknown error"),
                    elapsed_us=0,
                    device=device_str,
                    traceback=result.get("traceback"),
                    captured_output=captured if captured else None,
                )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            # 非 archive 模式清理临时 prof_dir
            if (
                prof_dir
                and self.perf_evaluator
                and not self.perf_evaluator.archive_prof
            ):
                shutil.rmtree(prof_dir, ignore_errors=True)

    def _parse_prof_result(self, prof_dir, case_id_str):
        """从 prof_dir 解析 profiler 产出文件，构造 PerfResult。

        复用 perf_evaluator 的 _locate_prof_files + perf_metric_strategy.parse。
        """
        from ..base.result import PerfResult
        import logging

        _logger = logging.getLogger(__name__)

        result = PerfResult(
            metadata={
                "case_id": case_id_str,
                "_repeat": self.perf_evaluator.repeat,
                "warmup_used": self.perf_evaluator.freq_boost,
                "freq_boost": self.perf_evaluator.freq_boost,
            }
        )

        try:
            prof_files = self.perf_evaluator._locate_prof_files(prof_dir)
            _logger.info(
                "pypto_pro_child: case %s — prof_files: csv=%s, trace_view=%s",
                case_id_str,
                prof_files.csv_path,
                prof_files.trace_view_path,
            )
            if self.perf_evaluator.perf_metric_strategy:
                result = self.perf_evaluator.perf_metric_strategy.parse(
                    prof_files, result
                )
            else:
                result.elapsed_us = 0.0
                result.error_msg = "no perf_metric_strategy configured"
        except Exception as e:
            result.error_msg = f"prof 文件解析失败: {e}"

        return result

    def _detect_pypto_pro_dispatch(self) -> bool:
        """检测 cann_bench 包是否为 pypto_pro 提交。

        pypto_pro 提交的运行时实现文件（test_<op>.py / <op>_impl.py）会
        ``import pypto_pro``；非 pypto_pro 提交（akg、手写 CANN 等）不会。
        AICore 507035 崩溃的根因是 pypto_pro JIT kernel 的进程内全局状态，
        因此直接检测 pypto_pro runtime 依赖最可靠，不依赖 dispatcher 文件名
        （converter agent 会不可预测地重命名 dispatcher 文件）。
        """
        try:
            return detect_pypto_pro_submission()
        except Exception:
            return False

    def _cleanup_memory(self):
        """清理 NPU cache 并强制 GC 回收

        注意：当 NPU 设备因 AICPU 异常进入错误状态后，
        torch_npu.npu.empty_cache() 会因设备同步失败而抛出 RuntimeError。
        必须静默处理，避免掩盖原始算子错误。

        RSS 主动清理：VmRSS 超过阈值（默认 48 GiB，可用环境变量
        KERNEL_EVAL_RSS_TRIM_MB 覆盖，<=0 禁用）时追加 malloc_trim(0)，
        把 glibc 持有的空闲堆页归还内核，缓解 eval-child 的 RSS 单调累积。
        /proc 读取与 malloc_trim 调用失败均静默降级，不影响评测主流程。
        """
        try:
            import torch_npu

            if hasattr(torch_npu, "npu") and torch.npu.is_available():
                torch_npu.npu.empty_cache()
        except Exception:
            pass
        import gc

        gc.collect()
        threshold_mb = _rss_trim_threshold_mb()
        if threshold_mb is None:
            return
        rss_mb = _read_rss_mb()
        if rss_mb is None or rss_mb <= threshold_mb:
            return
        if _malloc_trim():
            after_mb = _read_rss_mb()
            now = f"{after_mb:.0f}MB" if after_mb is not None else "未知"
            print(
                f"[INFO] VmRSS {rss_mb:.0f}MB 超过阈值 {threshold_mb:.0f}MB，"
                f"malloc_trim 已归还空闲堆页（现 {now}）"
            )

    def _release_outputs(self, op_run_result: OpRunResult) -> OpRunResult:
        """释放 outputs tensor，保留元数据

        outputs 从未被使用（to_dict 只取 elapsed_us），清除避免批跑时内存累积导致 OOM。
        """
        if op_run_result is None:
            return None
        op_run_result.outputs = None
        return op_run_result

    def shutdown(self):
        """关闭评测器"""
        self.perf_evaluator.shutdown()
