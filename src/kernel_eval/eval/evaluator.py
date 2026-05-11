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
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from inspect import Parameter, signature

import torch

from ..config import Config, get_config, get_project_root
from ..data.case_loader import CaseLoader, CaseInfo
from ..data.golden_loader import GoldenLoader
from ..data.data_generator import DataGenerator
from ..data.operator_loader import OperatorLoader, OperatorInfo
from ..data.package_manager import PackageManager, PackageInfo
from ..utils.device_manager import DeviceManager, DeviceConfig
from ..utils.param_builder import ParamBuilder
from ..utils.tensor_utils import tensors_to_cpu
from .op_runner import OpRunner, OpRunResult
from .accuracy_eval import AccuracyEvaluator, AccuracyResult
from .perf_eval import PerfEvaluator, PerfResult
from .results import EvalCaseResult, EvalOperatorResult, EvalSessionResult
from .failure_synthesizer import FailureSynthesizer
from .operator_matcher import OperatorMatcher
from .subprocess_runner import SubprocessRunner


class Evaluator:
    """综合评测调度器"""

    def __init__(self, config: Config = None):
        self.config = config or get_config()

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
        self.accuracy_evaluator = AccuracyEvaluator(self.config.precision_thresholds)

        # 初始化数据层组件
        self.case_loader = CaseLoader(self.config.kernel_bench_root)
        self.golden_loader = GoldenLoader(self.config.kernel_bench_root)
        self.operator_loader = OperatorLoader(self.config.kernel_bench_root)
        self.data_generator = DataGenerator()
        self.param_builder = ParamBuilder(self.golden_loader)

        # 初始化包管理器
        self.package_manager = PackageManager(config=self.config)

        # 初始化拆分模块
        self.operator_matcher = OperatorMatcher(self.operator_loader)
        self.failure_synthesizer = FailureSynthesizer(self.case_loader)
        kernel_eval_root = str(get_project_root() / "src")
        self.subprocess_runner = SubprocessRunner(
            failure_synthesizer=self.failure_synthesizer,
            device_id=self.config.device_id,
            kernel_eval_root=kernel_eval_root,
        )

    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载AI生成的算子函数（委托给 OperatorMatcher）"""
        return self.operator_matcher.load_ai_operator(operator_name)

    def evaluate_case(self, case: CaseInfo, ai_op_func: Callable = None) -> EvalCaseResult:
        """评测单个用例"""
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
                # get_input 期望 proto.yaml 顺序：x, h0, weights, biases
                # 但 cases.yaml 可能省略 optional 参数，顺序为：x, weights, biases (或 x, weights, biases, h0)

                # 获取 proto.yaml inputs 信息
                op_info = self.operator_loader.get_operator(case.rel_path)

                # 统计 proto.yaml tensor inputs 数量
                proto_tensor_count = len([i for i in op_info.inputs if 'Tensor' in str(i.dtype) or isinstance(i.dtype, list)])
                actual_tensor_count = len(input_tensors)

                # 如果实际 tensor 数量少于 proto tensor 数量，说明有 optional 参数被省略
                # 检测情况：cases.yaml 省略了 h0 (单 tensor)，但提供了所有 TensorList (weights/biases)

                params_for_get_input = {}

                # 检查是否有 optional h0/c0 参数
                has_optional_h0 = any(i.name in ['h0', 'c0'] for i in op_info.inputs)

                # 情况1: input_tensors 数量少于 proto tensor 数量
                # 根据格式匹配来判断哪些 proto inputs 对应哪些 input_tensors
                # 策略：按顺序匹配，单 tensor → 单 tensor input，TensorList → TensorList input
                if actual_tensor_count < proto_tensor_count:
                    params_for_get_input = {}
                    tensor_idx = 0

                    # 先处理 x (总是第一个)
                    params_for_get_input['x'] = input_tensors[0] if len(input_tensors) > 0 else None
                    tensor_idx = 1 if len(input_tensors) > 0 else 0

                    # proto.yaml 第二个是 h0/c0 (optional)，如果 cases.yaml 没有单独的 h0 tensor，设为 None
                    # 检查 input_tensors 中是否有单独的 h0 tensor（最后一个 tensor）
                    last_val = input_tensors[-1] if input_tensors else None
                    has_separate_h0 = isinstance(last_val, torch.Tensor) and actual_tensor_count > 3

                    if has_separate_h0:
                        params_for_get_input['h0'] = last_val
                        # 剩余 tensors: input_tensors[1:-1]
                        remaining_tensors = input_tensors[1:-1]
                    else:
                        params_for_get_input['h0'] = None
                        remaining_tensors = input_tensors[1:]

                    # 映射剩余 proto inputs (weight_ih, weight_hh, bias_ih, bias_hh)
                    remaining_proto_inputs = [i for i in op_info.inputs if i.name not in ['x', 'h0', 'c0']]
                    for proto_input in remaining_proto_inputs:
                        if remaining_tensors:
                            params_for_get_input[proto_input.name] = remaining_tensors[0]
                            remaining_tensors = remaining_tensors[1:]
                        else:
                            params_for_get_input[proto_input.name] = None

                # 情况2: input_tensors 数量 = proto tensor 数量 - 1，且 proto 有 optional h0/c0
                # 此时 cases.yaml 顺序可能是：x, weights, biases (省略 h0)
                elif has_optional_h0 and actual_tensor_count == proto_tensor_count - 1:
                    # cases.yaml 省略了 h0/c0
                    # 按 cases.yaml 实际顺序映射：x, weights, biases
                    tensor_idx = 0
                    for input_info in op_info.inputs:
                        if input_info.name in ['h0', 'c0']:
                            # optional tensor 未提供
                            params_for_get_input[input_info.name] = None
                        elif tensor_idx < len(input_tensors):
                            params_for_get_input[input_info.name] = input_tensors[tensor_idx]
                            tensor_idx += 1
                        else:
                            params_for_get_input[input_info.name] = None

                # 情况2: input_tensors 数量 = proto tensor 数量
                # 此时 cases.yaml 可能包含 h0/c0，但顺序可能不同
                elif actual_tensor_count == proto_tensor_count:
                    # 检查 input_tensors 最后一个是否是单 tensor (h0/c0 candidate)
                    last_val = input_tensors[-1] if input_tensors else None
                    last_is_single_tensor = isinstance(last_val, torch.Tensor)

                    if last_is_single_tensor and has_optional_h0:
                        # cases.yaml 顺序：x, weights, biases, h0
                        # proto.yaml 顺序：x, h0, weights, biases
                        tensor_idx = 0
                        for input_info in op_info.inputs:
                            if input_info.name in ['h0', 'c0']:
                                # h0 在 cases.yaml 最后
                                params_for_get_input[input_info.name] = input_tensors[-1]
                            elif tensor_idx < len(input_tensors) - 1:  # 排除最后一个是 h0
                                params_for_get_input[input_info.name] = input_tensors[tensor_idx]
                                tensor_idx += 1
                            else:
                                params_for_get_input[input_info.name] = None
                    else:
                        # 完全按 proto.yaml 顺序
                        for i, input_info in enumerate(op_info.inputs):
                            if i < len(input_tensors):
                                params_for_get_input[input_info.name] = input_tensors[i]
                            else:
                                params_for_get_input[input_info.name] = None

                else:
                    # 其他情况，按 proto.yaml 顺序映射
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

            # 4. 执行Golden函数获取参考结果
            golden_result = self.op_runner.run_golden(golden_func, params, case_id_str, input_tensors)
            if not golden_result.success:
                return EvalCaseResult(
                    case_id=case_id_str,
                    rel_path=case.rel_path,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=golden_result,
                    error_msg=f"Golden执行失败: {golden_result.error}",
                    baseline_perf_us=case.baseline_perf_us,
                    t_hw_us=case.t_hw_us,
                )

            # 5. 如果没有传入AI算子函数，尝试加载
            actual_ai_func = ai_op_func
            if actual_ai_func is None:
                try:
                    actual_ai_func = self.operator_matcher.load_ai_operator(case.operator)
                except Exception:
                    return EvalCaseResult(
                        case_id=case_id_str,
                        rel_path=case.rel_path,
                        operator=case.operator,
                        case_num=case.case_id,
                        success=True,
                        golden_run_result=golden_result,
                        accuracy_result=AccuracyResult(
                            passed=True,
                            dtype=case.dtypes[0] if case.dtypes else 'float32',
                            threshold=0, mere=0, mare=0,
                        ),
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
                    golden_run_result=golden_result,
                    ai_run_result=ai_result,
                    error_msg=f"AI算子执行失败: {ai_result.error}",
                    baseline_perf_us=case.baseline_perf_us,
                    t_hw_us=case.t_hw_us,
                )

            # 7. 精度对比（使用与性能采集同一次运行的输出）
            dtype = self._determine_dtype(ai_result, case)
            merged_thresholds = self._get_merged_thresholds(case.rel_path)

            # CPU 同精度输出
            cpu_inputs = tensors_to_cpu(input_tensors)
            if get_input_func is not None:
                # get_input 已重新排序，直接按位置构建参数
                golden_sig = signature(golden_func)
                cpu_params = {}
                tensor_idx = 0
                for param_name, param in golden_sig.parameters.items():
                    annotation = str(param.annotation) if param.annotation != Parameter.empty else ""
                    if 'Tensor' in annotation:
                        if tensor_idx < len(cpu_inputs):
                            cpu_params[param_name] = cpu_inputs[tensor_idx]
                            tensor_idx += 1
                    elif param_name in case_attrs:
                        cpu_params[param_name] = case_attrs[param_name]
                    elif param.default != Parameter.empty:
                        cpu_params[param_name] = param.default
            else:
                cpu_params = self.param_builder.build_call_params(golden_func, case, cpu_inputs)
            with torch.no_grad():
                cpu_out = golden_func(**cpu_params)

            ignore_output_indices = self._get_ignore_output_indices(case.rel_path)

            accuracy_result = self.accuracy_evaluator.evaluate(
                ai_output=ai_result.outputs,
                golden_output=golden_result.outputs,
                dtype=dtype,
                custom_thresholds=merged_thresholds,
                cpu_output=cpu_out,
                ignore_output_indices=ignore_output_indices,
            )

            # 8. 性能数据已在上面的 profiler 运行中采集，直接提取
            if accuracy_result.passed:
                if ai_result.perf_result is not None:
                    perf_result = ai_result.perf_result
                elif ai_result.elapsed_us > 0:
                    perf_result = PerfResult(case_id=case_id_str, elapsed_us=ai_result.elapsed_us)
                else:
                    perf_result = None
                error_msg = None
            else:
                perf_result = None
                error_msg = (
                    f"精度不达标: MARE={accuracy_result.mare:.6f}, "
                    f"MERE={accuracy_result.mere:.6f}, "
                    f"阈值={accuracy_result.threshold}, "
                    f"最大差值={accuracy_result.max_diff:.6f}, "
                    f"不匹配比例={accuracy_result.mismatch_ratio:.4f}"
                )

            self._cleanup_memory()

            return EvalCaseResult(
                case_id=case_id_str,
                rel_path=case.rel_path,
                operator=case.operator,
                case_num=case.case_id,
                success=accuracy_result.passed,
                accuracy_result=accuracy_result,
                perf_result=perf_result,
                golden_run_result=golden_result,
                ai_run_result=ai_result,
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
                mare_str = f"MARE={acc.mare:.6f}" if acc.mare is not None else ""
                mere_str = f"MERE={acc.mere:.6f}" if acc.mere is not None else ""
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
            except Exception:
                pass

        return dtype or (case.dtypes[0] if case.dtypes else 'float32')

    def _format_elapsed(self, result) -> str:
        """格式化耗时"""
        if result.success and result.perf_result and result.perf_result.elapsed_us > 0:
            return f"{result.perf_result.elapsed_us:.2f}μs"
        elif result.ai_run_result and result.ai_run_result.elapsed_us > 0:
            return f"{result.ai_run_result.elapsed_us:.2f}μs"
        return "N/A"

    def _filter_cases(self, cases: List[CaseInfo], filter_dict: Dict) -> List[CaseInfo]:
        """筛选用例"""
        result = cases
        if 'case_id' in filter_dict:
            result = [c for c in result if c.case_id == filter_dict['case_id']]
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

    def shutdown(self):
        """关闭评测器"""
        self.perf_evaluator.shutdown()