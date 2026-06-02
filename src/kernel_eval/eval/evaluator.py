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
- 子进程执行移至 subprocess_runner.py
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from inspect import Parameter, signature

import torch

from ..config import Config, get_config, get_project_root
from ..registry.loader_registry import get_task_loader, get_case_loader
from ..registry.golden_registry import get_golden_loader
from ..registry.bench_registry import get_bench_config
from ..data.data_generator import DataGenerator
from ..data.package_manager import PackageManager, PackageInfo
from ..base.models import TaskSpec, CaseSpec
from ..utils.device_manager import DeviceManager, DeviceConfig
from ..utils.param_builder import ParamBuilder
from ..utils.tensor_utils import tensors_to_cpu, tensors_to_fp64_cpu
from .op_runner import OpRunner, OpRunResult
from .accuracy_eval import AccuracyEvaluator, AccuracyResult
from .perf_eval import PerfEvaluator, PerfResult
from .results import EvalCaseResult, EvalOperatorResult, EvalSessionResult
from .failure_synthesizer import FailureSynthesizer
from ..registry.matcher_registry import get_operator_matcher
from .subprocess_runner import SubprocessRunner

# 导入 benches 模块，确保 Registry 已注册
from .. import benches as _benches


class Evaluator:
    """综合评测调度器"""

    def __init__(self, config: Config = None, bench_name: str = 'cann'):
        self.config = config or get_config()
        self.bench_name = bench_name
        self.bench_config = get_bench_config(self.bench_name)

        # 初始化设备管理器
        device_config = DeviceConfig(
            type=self.config.device_type,
            device_id=self.config.device_id,
            auto_fallback=self.config.auto_fallback,
        )
        self.device_manager = DeviceManager(device_config)

        # 初始化性能评测器
        self.perf_evaluator = PerfEvaluator(
            config=self.config,
            device_manager=self.device_manager,
            warmup=self.config.warmup,
            repeat=self.config.repeat,
            archive_prof=True,
        )

        # 初始化算子执行器
        self.op_runner = OpRunner(self.device_manager, self.perf_evaluator)

        # 初始化精度评测器
        self.accuracy_evaluator = AccuracyEvaluator(
            custom_thresholds=self.config.precision_thresholds,
            checker_name=self.bench_config.checker,
        )

        # 初始化数据层组件（通过 Registry 获取）
        self.case_loader = get_case_loader(self.bench_name, tasks_root=self.config.tasks_root)
        self.golden_loader = get_golden_loader(
            eval_system=self.bench_name,
            bench_root=self.config.tasks_root,
        )
        self.operator_loader = get_task_loader(self.bench_name, tasks_root=self.config.tasks_root)
        self.data_generator = DataGenerator()
        self.param_builder = ParamBuilder(self.golden_loader)

        # 初始化包管理器
        self.package_manager = PackageManager(config=self.config)

        # 初始化拆分模块
        self.operator_matcher = get_operator_matcher(
            eval_system=self.bench_name,
            operator_loader=self.operator_loader,
        )
        self.failure_synthesizer = FailureSynthesizer(self.case_loader)
        kernel_eval_root = str(get_project_root() / "src")
        self.subprocess_runner = SubprocessRunner(
            failure_synthesizer=self.failure_synthesizer,
            device_id=self.config.device_id,
            kernel_eval_root=kernel_eval_root,
            config=self.config,
        )

    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载AI生成的算子函数（委托给 OperatorMatcher）"""
        return self.operator_matcher.load_ai_operator(operator_name)

    def evaluate_case(self, case: CaseSpec, *, ai_op_func: Callable = None) -> EvalCaseResult:
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

        try:
            # 1. 获取golden函数
            golden_func = self.golden_loader.get_golden_function(case.rel_path)

            # 2. 生成输入数据
            input_tensors = self.data_generator.generate_input_tensors_from_case(
                input_shapes=case.input_shapes,
                dtypes=case.dtypes,
                value_ranges=case.value_ranges,
            )

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
                case_attrs = getattr(case, 'attrs', None) or {}
                for attr_key, attr_val in case_attrs.items():
                    if attr_key not in params_for_get_input:
                        params_for_get_input[attr_key] = attr_val
                if 'skip2_exist' not in params_for_get_input:
                    params_for_get_input['skip2_exist'] = case_attrs.get('skip2_exist', True)

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
                    annotation = str(param.annotation) if param.annotation != Parameter.empty else ""
                    # 检查是否是 tensor 参数
                    if 'Tensor' in annotation:
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
                params = self.param_builder.build_call_params(golden_func, case, input_tensors)

            # 4. 执行Golden函数获取参考结果（精度策略由 golden_precision 控制）
            golden_inputs = self._apply_golden_precision(input_tensors)
            if get_input_func is not None:
                golden_params = params
            else:
                golden_params = self.param_builder.build_call_params(golden_func, case, golden_inputs)
            golden_result = self.op_runner.run(golden_func, golden_params, case_id_str,
                                               golden_inputs, to_device=self._get_golden_to_device(),
                                               enable_profiler=False)  # Golden 不启用 profiler
            if not golden_result.success:
                return EvalCaseResult(
                    case_id=case_id_str,
                    rel_path=case.rel_path,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=self._release_outputs(golden_result),
                    error_msg=f"Golden执行失败: {golden_result.error}",
                    baseline_perf_us=case.baseline_perf_us,
                    t_hw_us=case.t_hw_us,
                )

            # 5. 确定使用的 AI 算子函数
            # 优先级：传入参数 > 加载 AI 算子
            actual_ai_func = ai_op_func
            if actual_ai_func is None:
                try:
                    actual_ai_func = self.operator_matcher.load_ai_operator(case.operator)
                except Exception as load_err:
                    return EvalCaseResult(
                        case_id=case_id_str,
                        rel_path=case.rel_path,
                        operator=case.operator,
                        case_num=case.case_num,
                        success=False,
                        golden_run_result=self._release_outputs(golden_result),
                        error_msg=f"AI算子加载失败: {load_err}",
                        baseline_perf_us=case.baseline_perf_us,
                        t_hw_us=case.t_hw_us,
                    )

            # 6. 执行AI算子（profiler 一次运行同时提供输出和性能数据，避免跑两遍）
            use_profiler = (self.perf_evaluator is not None
                        and self.perf_evaluator.config.enable_profiler)
            ai_result = self.op_runner.run_ai_op(actual_ai_func, params, case_id_str,
                                                  input_tensors, enable_perf=use_profiler)
            if not ai_result.success:
                self._cleanup_memory()
                return EvalCaseResult(
                    case_id=case_id_str,
                    rel_path=case.rel_path,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=self._release_outputs(golden_result),
                    ai_run_result=self._release_outputs(ai_result),
                    error_msg=f"AI算子执行失败: {ai_result.error}",
                    baseline_perf_us=case.baseline_perf_us,
                    t_hw_us=case.t_hw_us,
                )

            # 7. 精度对比（使用与性能采集同一次运行的输出）
            dtype = self._determine_dtype(ai_result, case)
            merged_thresholds = self._get_merged_thresholds(case.rel_path)

            # 同精度参考输出（用于 checker 小值域判断）
            # native_cpu/native_npu 时 golden 已是同精度，直接复用以避免重复计算
            golden_strategy = getattr(self.bench_config, 'golden_precision', 'fp64_cpu')
            if golden_strategy in ('native_cpu', 'native_npu'):
                native_out = golden_result.outputs
            else:
                native_inputs = tensors_to_cpu(input_tensors)
                if get_input_func is not None:
                    native_params = params
                else:
                    native_params = self.param_builder.build_call_params(golden_func, case, native_inputs)
                native_result = self.op_runner.run(golden_func, native_params, case_id_str,
                                                   native_inputs, to_device=False,
                                                   enable_profiler=False)  # 同精度参考不启用 profiler
                native_out = native_result.outputs if native_result.success else None

            ignore_output_indices = self._get_ignore_output_indices(case.rel_path)

            # 获取算子输出名称（用于填充 SingleOutputResult.name）
            op_info = self.operator_loader.get_operator(case.rel_path)
            output_names = [out.name for out in op_info.outputs] if op_info and op_info.outputs else []

            accuracy_result = self.accuracy_evaluator.evaluate(
                ai_output=ai_result.outputs,
                golden_output=golden_result.outputs,
                dtype=dtype,
                custom_thresholds=merged_thresholds,
                native_output=native_out,
                ignore_output_indices=ignore_output_indices,
            )

            # 防作弊二次验证：用新鲜输入再跑一遍 golden + AI，两次都过才算 pass
            # 只在 config.enable_accuracy_retry=True 且第一次已通过时触发，避免开销
            if accuracy_result.passed and getattr(self.config, 'enable_accuracy_retry', False):
                accuracy_result = self._retry_with_fresh_inputs(
                    case=case,
                    golden_func=golden_func,
                    ai_op_func=ai_op_func,
                    dtype=dtype,
                    merged_thresholds=merged_thresholds,
                    ignore_output_indices=ignore_output_indices,
                    first_result=accuracy_result,
                )

            # 填充 output_results 中的输出名称
            if hasattr(accuracy_result, 'output_results') and output_names:
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
                if hasattr(accuracy_result, 'format_all_outputs') and accuracy_result.output_results:
                    output_details = accuracy_result.format_all_outputs()
                    error_msg = f"精度不达标:\n{output_details}"
                else:
                    # 兼容旧格式（从 metadata 获取 mere/mare）
                    metadata = accuracy_result.get_metadata()
                    mere = metadata.get('mere', 0.0)
                    mare = metadata.get('mare', 0.0)
                    mare_threshold = 10 * accuracy_result.threshold if accuracy_result.threshold and accuracy_result.threshold > 0 else 0
                    fail_reasons = []
                    if mare >= mare_threshold:
                        fail_reasons.append(f"MARE({mare:.6f}) >= mare_threshold({mare_threshold:.6f})")
                    if accuracy_result.threshold and mere >= accuracy_result.threshold:
                        fail_reasons.append(f"MERE({mere:.6f}) >= threshold({accuracy_result.threshold:.6f})")
                    if accuracy_result.error_msg:
                        fail_reasons.append(accuracy_result.error_msg)
                    error_msg = f"精度不达标: {', '.join(fail_reasons)}"

            self._cleanup_memory()

            return EvalCaseResult(
                case_id=case_id_str,
                rel_path=case.rel_path,
                operator=case.operator,
                case_num=case.case_id,
                success=accuracy_result.is_passed(),
                accuracy_result=accuracy_result,
                perf_result=perf_result,
                golden_run_result=self._release_outputs(golden_result),
                ai_run_result=self._release_outputs(ai_result),
                error_msg=error_msg,
                baseline_perf_us=case.baseline_perf_us,
                t_hw_us=case.t_hw_us,
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            self._cleanup_memory()
            return EvalCaseResult(
                case_id=case_id_str,
                rel_path=case.rel_path,
                operator=case.operator,
                case_num=case.case_id,
                success=False,
                error_msg=f"评测异常: {e}",
            )

    def evaluate_operator(self, operator: str, rel_path: str, case_filter: Dict = None) -> EvalOperatorResult:
        """评测单个算子"""
        cases = self.case_loader.scan_by_operator(operator)
        if case_filter:
            cases = self._filter_cases(cases, case_filter)

        if not cases:
            return EvalOperatorResult(
                rel_path=rel_path, operator=operator, total_cases=0,
                passed_cases=0, failed_cases=0, skipped_cases=0,
                results=[], pass_rate=0.0, avg_speedup=0.0,
            )

        self.operator_matcher.clear_cache()
        results = []
        print(f"[INFO] 评测算子 {operator} ({rel_path}), 用例数: {len(cases)}")

        for i, case in enumerate(cases, 1):
            case_id_str = case.get_case_id_str()
            result = self.evaluate_case(case)
            results.append(result)

            status_icon = "✅" if result.success else "❌"
            elapsed_str = self._format_elapsed(result)
            speedup_str = f"{result.get_speedup():.2f}x" if result.get_speedup() > 0 else "N/A"

            # 添加精度信息
            if result.success and result.accuracy_result:
                acc = result.accuracy_result
                if hasattr(acc, 'output_results') and acc.output_results:
                    output_summaries = [sr.format_summary() for sr in acc.output_results]
                    acc_str = ", " + ", ".join(output_summaries)
                else:
                    metadata = acc.get_metadata()
                    mare = metadata.get('mare')
                    mere = metadata.get('mere')
                    mare_str = f"MARE={mare:.6f}" if mare is not None else ""
                    mere_str = f"MERE={mere:.6f}" if mere is not None else ""
                    acc_str = f", {mare_str}, {mere_str}" if mare_str or mere_str else ""
                    print(
                        f"[{i}/{len(cases)}] {case_id_str}: {status_icon} "
                        f"(耗时: {elapsed_str}, 加速比: {speedup_str}{acc_str})"
                    )
            elif result.success:
                print(f"[{i}/{len(cases)}] {case_id_str}: {status_icon} (耗时: {elapsed_str}, 加速比: {speedup_str})")
            else:
                error_hint = result.error_msg[:50] if result.error_msg else ""
                print(f"[{i}/{len(cases)}] {case_id_str}: {status_icon} {error_hint}")

        passed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success)
        speedups = [r.get_speedup() for r in results if r.success and r.get_speedup() > 0]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

        return EvalOperatorResult(
            rel_path=rel_path, operator=operator,
            total_cases=len(cases), passed_cases=passed, failed_cases=failed,
            skipped_cases=0, results=results,
            pass_rate=passed / len(cases) if len(cases) > 0 else 0.0,
            avg_speedup=avg_speedup,
        )

    def evaluate_from_source(
        self,
        source_dir: str,
        operator_filter: List[str] = None,
        case_filter: Dict = None,
        verbose: bool = False,
        subprocess_isolation: bool = True,
        op_timeout_sec: int = 240,
        case_timeout_sec: int = None,
        case_subprocess_isolation: bool = True,
        iterative_compile: bool = True,
    ) -> EvalSessionResult:
        """从源码目录执行完整评测"""
        print("")
        print("=" * 60)
        print("开始评测")
        print("=" * 60)

        # 1. 准备环境（编译安装）
        matched_operators, package_info = self.package_manager.prepare_from_source(
            source_dir, verbose=verbose, iterative_compile=iterative_compile,
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
                        op_info, str(e), case_filter, self._filter_cases,
                    )
                    results.append(result)
            for snake_op_name, err in (package_info.compile_errors or {}).items():
                op_info = self.operator_matcher.find_operator_info_by_snake(snake_op_name)
                if op_info and (not operator_filter or op_info.name in operator_filter):
                    results.append(self.failure_synthesizer.synthesize_compile_failure(
                        op_info, err, case_filter, self._filter_cases,
                    ))
            return EvalSessionResult(operators=results, package_info=package_info)

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
                self.failure_synthesizer.synthesize_compile_failure(op_info, err, case_filter, self._filter_cases),
            )

        if not matched_operators and not compile_failed_results:
            return EvalSessionResult(operators=[], package_info=package_info)

        # 4. 应用算子筛选
        if operator_filter:
            matched_operators = [op for op in matched_operators if op in operator_filter]
            print(f"[INFO] 筛选后算子: {matched_operators}")

        # 5. 逐个评测算子
        results: List[EvalOperatorResult] = list(compile_failed_results)
        print(f"[INFO] 编译失败: {len(compile_failed_results)} 个算子 | 可运行: "
              f"{len(matched_operators)} 个算子 "
              f"({'subprocess-per-op' if subprocess_isolation else 'in-process'})")

        for operator_name in matched_operators:
            op_info = self.operator_matcher.find_operator_info(operator_name)
            if not op_info:
                print(f"[WARN] 算子 {operator_name} 未找到定义，跳过")
                continue

            if subprocess_isolation:
                result = self.subprocess_runner.run_operator_subprocess(
                    operator_name, rel_path=op_info.rel_path,
                    source_dir=source_dir, case_filter=case_filter,
                    timeout_sec=op_timeout_sec, filter_func=self._filter_cases,
                )
                results.append(result)
                continue

            result = self.evaluate_operator(operator=operator_name, rel_path=op_info.rel_path, case_filter=case_filter)
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
        operator_subprocess_isolation: bool = True,
    ) -> EvalSessionResult:
        """跳过编译安装，直接评测已安装的cann_bench"""
        print("")
        print("=" * 60)
        print("开始评测（跳过编译安装）")
        print("=" * 60)

        matched_operators = self.package_manager.prepare_skip_build()
        if not matched_operators:
            return EvalSessionResult(operators=[])

        if operator_filter:
            matched_operators = [op for op in matched_operators if op in operator_filter]
            print(f"[INFO] 筛选后算子: {matched_operators}")

        results = []
        print(f"[INFO] 可运行算子: {len(matched_operators)} "
              f"({'subprocess-per-op' if operator_subprocess_isolation else 'in-process'})")

        for operator_name in matched_operators:
            op_info = self.operator_matcher.find_operator_info(operator_name)
            if not op_info:
                print(f"[WARN] 算子 {operator_name} 未找到定义，跳过")
                continue

            if operator_subprocess_isolation:
                result = self.subprocess_runner.run_operator_subprocess_simple(
                    operator_name, rel_path=op_info.rel_path,
                    case_filter=case_filter, timeout_sec=240,
                    filter_func=self._filter_cases,
                )
                results.append(result)
                continue

            result = self.evaluate_operator(operator=operator_name, rel_path=op_info.rel_path, case_filter=case_filter)
            results.append(result)

        print("")
        print("=" * 60)
        print("评测完成")
        print("=" * 60)

        return EvalSessionResult(operators=results)

    def evaluate_golden_only(self, operator: str, rel_path: str, case_filter: Dict = None) -> EvalOperatorResult:
        """仅执行Golden验证"""
        return self.evaluate_operator(operator=operator, rel_path=rel_path, case_filter=case_filter)

    # ---- 辅助方法 ----

    def _apply_golden_precision(self, input_tensors: List) -> List:
        """根据 bench 配置的 golden_precision 策略转换输入张量。

        取值：
          - fp64_cpu（默认）: 升精度到 float64 + CPU 计算，避免 NPU 溢出污染
          - native_cpu: 保持原始精度在 CPU 上计算
          - native_npu: 保持原始精度在 NPU 上计算
        """
        strategy = getattr(self.bench_config, 'golden_precision', 'fp64_cpu')
        if strategy == 'fp64_cpu':
            return tensors_to_fp64_cpu(input_tensors)
        elif strategy == 'native_cpu':
            return tensors_to_cpu(input_tensors)
        elif strategy == 'native_npu':
            return list(input_tensors)
        return list(input_tensors)

    def _get_golden_to_device(self) -> bool:
        """golden 执行时 to_device 参数：仅 native_npu 在 NPU 上计算时为 True"""
        strategy = getattr(self.bench_config, 'golden_precision', 'fp64_cpu')
        return strategy == 'native_npu'

    def _determine_dtype(self, ai_result, case) -> str:
        """确定 dtype"""
        dtype = None
        if isinstance(ai_result.outputs, torch.Tensor):
            dtype = str(ai_result.outputs.dtype).replace('torch.', '')
        elif isinstance(ai_result.outputs, (list, tuple)) and ai_result.outputs:
            first_out = ai_result.outputs[0]
            if isinstance(first_out, torch.Tensor):
                dtype = str(first_out.dtype).replace('torch.', '')

        if dtype is None:
            try:
                op_info = self.operator_loader.get_operator(case.rel_path)
                if op_info and op_info.outputs and op_info.outputs[0].dtype:
                    output_dtype_list = op_info.outputs[0].dtype
                    dtype = output_dtype_list[0] if isinstance(output_dtype_list, list) else output_dtype_list
            except Exception as e:
                # 静默吞掉会让 proto.yaml 缺失/损坏的算子退回到 case.dtypes[0]，
                # 可能选错精度阈值。至少记录可见的告警。
                print(f"[WARN] _determine_dtype: 读取 {case.rel_path} 算子定义失败({type(e).__name__}: {e})，回退到 case.dtypes")

        return dtype or (case.dtypes[0] if case.dtypes else 'float32')

    def _format_elapsed(self, result) -> str:
        """格式化耗时"""
        if result.success and result.perf_result and result.perf_result.elapsed_us > 0:
            return f"{result.perf_result.elapsed_us:.2f}μs"
        elif result.ai_run_result and result.ai_run_result.elapsed_us > 0:
            return f"{result.ai_run_result.elapsed_us:.2f}μs"
        return "N/A"

    def _filter_cases(self, cases: List[CaseSpec], filter_dict: Dict) -> List[CaseSpec]:
        """筛选用例"""
        result = cases
        if 'case_id' in filter_dict:
            # case_num 是 CannCaseSpec 的特化字段，通过 hasattr 兼容
            result = [c for c in result if hasattr(c, 'case_num') and c.case_num == filter_dict['case_id']]
        if 'dtype' in filter_dict:
            result = [c for c in result if filter_dict['dtype'].lower() in [d.lower() for d in c.dtypes]]
        return result

    def _get_merged_thresholds(self, rel_path: str) -> Dict[str, float]:
        """获取合并后的精度阈值"""
        op_info = self.operator_loader.get_operator(rel_path)
        if op_info and op_info.precision_thresholds:
            merged = dict(self.config.precision_thresholds)
            merged.update(op_info.precision_thresholds)
            return merged
        return self.config.precision_thresholds

    def _get_ignore_output_indices(self, rel_path: str) -> List[int]:
        """获取需要忽略对比的输出索引"""
        ignore_indices = []
        op_info = self.operator_loader.get_operator(rel_path)
        if op_info and op_info.outputs:
            for idx, output in enumerate(op_info.outputs):
                if not output.compare:
                    ignore_indices.append(idx)
        return ignore_indices

    def _retry_with_fresh_inputs(self, case, golden_func, ai_op_func, dtype,
                                  merged_thresholds, ignore_output_indices,
                                  first_result):
        """防作弊二次验证：用一组新鲜（微扰过的）输入再跑一遍 golden + AI，
        两次都过才记为 pass；任何一次失败把 first_result 替换成失败 result。

        启用条件：Config.enable_accuracy_retry=True 且第一轮已通过。
        参考 AccuracyEvaluator.evaluate_with_retry 的设计思路。
        """
        try:
            import torch

            # 生成新鲜输入（DataGenerator 每次调用 seed 不同）
            fresh_inputs = self.data_generator.generate_input_tensors_from_case(
                input_shapes=case.input_shapes,
                dtypes=case.dtypes,
                value_ranges=case.value_ranges,
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
            params = self.param_builder.build_call_params(golden_func, case, fresh_inputs)
            golden_inputs = self._apply_golden_precision(fresh_inputs)
            golden_params = self.param_builder.build_call_params(golden_func, case, golden_inputs)
            golden_result2 = self.op_runner.run(golden_func, golden_params, case_id_str,
                                                golden_inputs, to_device=self._get_golden_to_device(),
                                                enable_profiler=False)  # 二次验证 Golden 不启用 profiler
            if not golden_result2.success:
                return first_result   # golden 自己挂了，第二轮无意义，保持第一轮结果

            # 二次验证 AI 算子也不启用 profiler，仅验证精度
            ai_result2 = self.op_runner.run(ai_op_func, params, case_id_str, fresh_inputs,
                                             enable_profiler=False)
            if not ai_result2.success:
                from ..base.result import AccuracyResult
                return AccuracyResult(
                    passed=False, threshold=first_result.threshold or 0,
                    error_msg=f"二次验证失败：AI 算子崩溃 ({ai_result2.error})",
                )

            return self.accuracy_evaluator.evaluate(
                ai_output=ai_result2.outputs,
                golden_output=golden_result2.outputs,
                dtype=dtype,
                custom_thresholds=merged_thresholds,
                native_output=None,
                ignore_output_indices=ignore_output_indices,
            )
        except Exception as e:
            # 二次验证基础设施异常不应整体阻断评测；记 warn 并返回第一轮
            print(f"[WARN] enable_accuracy_retry 二次验证基础设施异常 ({case.rel_path}): {e}", flush=True)
            return first_result

    def _cleanup_memory(self):
        """清理 NPU cache（不触发完整 GC，引用计数足以处理大多数情况）

        注意：当 NPU 设备因 AICPU 异常进入错误状态后，
        torch_npu.npu.empty_cache() 会因设备同步失败而抛出 RuntimeError。
        必须静默处理，避免掩盖原始算子错误。
        """
        try:
            import torch_npu
            if hasattr(torch_npu, 'npu') and torch.npu.is_available():
                torch_npu.npu.empty_cache()
        except Exception:
            pass

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