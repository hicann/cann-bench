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
综合评测调度器

职责：
1. 协调精度评测和性能评测执行顺序
2. 支持源码目录扫描、编译、安装
3. 实现评测任务筛选（level/operator/case_id）
4. 生成评测结果
"""

import gc
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass


def _snake_case_candidates(name: str) -> List[str]:
    """Generate plausible snake_case forms of a CamelCase operator name.

    The reference ``cann_bench`` module is inconsistent: some ops use
    ``pool3_d`` (digit glued to preceding letters, underscore before the
    trailing letter) while others use ``sampler_3d`` (underscore before
    the digit, digit glued to the trailing letter). Acronyms like
    ``ROIAlign`` → ``roi_align`` and ``NMS`` → ``nms`` also don't survive
    a naive "insert _ before every capital" pass. Emit the reasonable
    variants in order and let the caller ``hasattr`` the first that hits.
    """
    cands: List[str] = []
    # V1: naive — underscore before every uppercase letter.
    # Covers plain CamelCase like MaskedScale → masked_scale and weird
    # digit-suffix ops like AdaptiveAvgPool3D → adaptive_avg_pool3_d.
    v1 = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    v1 = re.sub(r"_{2,}", "_", v1)
    cands.append(v1)
    # V2: acronym-aware — keep runs of capitals together, then split
    # camelCase / letter-or-digit-then-capital boundaries. Covers
    # ROIAlign → roi_align, NMS → nms, TopK → top_k.
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    v2 = s.lower()
    if v2 not in cands:
        cands.append(v2)
    # V3: V2 plus an underscore between a lowercase letter and a digit —
    # grid_sampler3_d → grid_sampler_3_d.
    v3 = re.sub(r"([a-z])(\d)", r"\1_\2", s).lower()
    if v3 not in cands:
        cands.append(v3)
    # V4: V3 but re-joining digit_<letter> so trailing units stick
    # together — grid_sampler_3_d → grid_sampler_3d (the convention
    # GridSampler3D actually uses in cann_bench).
    v4 = re.sub(r"(\d)_([a-z])", r"\1\2", v3)
    if v4 not in cands:
        cands.append(v4)
    return cands

from ..config import Config, get_config
from ..data.case_loader import CaseLoader, CaseInfo
from ..data.golden_loader import GoldenLoader
from ..data.data_generator import DataGenerator
from ..data.operator_loader import OperatorLoader, OperatorInfo
from ..data.package_manager import PackageManager, PackageInfo
from ..utils.device_manager import DeviceManager, DeviceConfig
from ..utils.param_builder import ParamBuilder
from ..eval.op_runner import OpRunner, OpRunResult
from ..eval.accuracy_eval import AccuracyEvaluator, AccuracyResult
from ..eval.perf_eval import PerfEvaluator, PerfResult


@dataclass
class EvalCaseResult:
    """单用例评测结果"""
    case_id: str
    level: int
    operator: str
    case_num: int
    success: bool
    accuracy_result: Optional[AccuracyResult] = None
    perf_result: Optional[PerfResult] = None
    golden_run_result: Optional[OpRunResult] = None
    ai_run_result: Optional[OpRunResult] = None
    error_msg: Optional[str] = None
    baseline_perf_us: float = 0.0

    def get_speedup(self) -> float:
        """计算加速比"""
        if self.perf_result and self.baseline_perf_us > 0:
            return self.baseline_perf_us / self.perf_result.elapsed_us if self.perf_result.elapsed_us > 0 else 0.0
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'case_id': self.case_id,
            'level': self.level,
            'operator': self.operator,
            'case_num': self.case_num,
            'success': self.success,
            'accuracy': self.accuracy_result.to_dict() if self.accuracy_result else None,
            'perf': {
                'elapsed_us': self.perf_result.elapsed_us if self.perf_result else 0,
                'speedup': self.get_speedup(),
                'op_times': self.perf_result.op_times if self.perf_result else {},
            } if self.perf_result else None,
            'golden_elapsed_us': self.golden_run_result.elapsed_us if self.golden_run_result else 0,
            'ai_elapsed_us': self.ai_run_result.elapsed_us if self.ai_run_result else 0,
            'error_msg': self.error_msg,
            'baseline_perf_us': self.baseline_perf_us,
        }


@dataclass
class EvalOperatorResult:
    """算子评测结果"""
    operator: str
    level: int
    total_cases: int
    passed_cases: int
    failed_cases: int
    skipped_cases: int
    results: List[EvalCaseResult]
    pass_rate: float
    avg_speedup: float
    # 当算子跑不起来时附带的诊断信息：
    #   - compilation_error: build.sh 阶段隔离出来的算子，每条 case 都标记
    #     为 FAIL。PackageManager 的迭代编译 populate 到 package_info，再
    #     由 evaluate_from_source 合成。
    #   - subprocess_failure_reason: 开启子进程隔离后，该算子的 subprocess
    #     超时 / 崩溃 / 返回非零 / 输出异常。evaluator._run_operator_subprocess
    #     在异常路径下 populate。
    # 两个字段都是 Optional，常规路径下为 None 不写入 JSON。
    compilation_error: Optional[str] = None
    subprocess_failure_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'operator': self.operator,
            'level': self.level,
            'total_cases': self.total_cases,
            'passed_cases': self.passed_cases,
            'failed_cases': self.failed_cases,
            'skipped_cases': self.skipped_cases,
            'pass_rate': self.pass_rate,
            'avg_speedup': self.avg_speedup,
            'results': [r.to_dict() for r in self.results],
        }
        if self.compilation_error:
            d['compilation_error'] = self.compilation_error
        if self.subprocess_failure_reason:
            d['subprocess_failure_reason'] = self.subprocess_failure_reason
        return d


@dataclass
class EvalSessionResult:
    """评测会话结果"""
    operators: List[EvalOperatorResult]
    package_info: Optional[PackageInfo] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operators': [op.to_dict() for op in self.operators],
            'package_info': {
                'source_dir': self.package_info.source_dir if self.package_info else '',
                'whl_path': self.package_info.whl_path if self.package_info else '',
                'run_path': self.package_info.run_path if self.package_info else '',
            } if self.package_info else None,
        }


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
            enabled=self.config.enable_profiler and self.device_manager.is_npu_mode(),
            device_manager=self.device_manager,
            warmup=self.config.warmup,
            repeat=self.config.repeat,
            archive_prof=False,  # 使用临时目录避免 torch_npu 解析旧数据产生的 ERROR 日志
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
        self.package_manager = PackageManager()

        # AI算子模块缓存
        self._ai_op_cache: Dict[str, Callable] = {}

    def load_ai_operator(self, operator_name: str) -> Callable:
        """加载AI生成的算子函数"""
        cache_key = operator_name.lower()
        if cache_key in self._ai_op_cache:
            return self._ai_op_cache[cache_key]

        try:
            import cann_bench

            candidates = _snake_case_candidates(operator_name) + [
                operator_name.lower(),
                operator_name,
            ]
            for name in candidates:
                if hasattr(cann_bench, name):
                    func = getattr(cann_bench, name)
                    self._ai_op_cache[cache_key] = func
                    return func

            # 尝试 torch.ops.cann_bench
            try:
                import torch
                if hasattr(torch.ops, 'cann_bench'):
                    for name in candidates:
                        if hasattr(torch.ops.cann_bench, name):
                            func = getattr(torch.ops.cann_bench, name)
                            self._ai_op_cache[cache_key] = func
                            return func
            except Exception:
                pass

            raise AttributeError(f"无法找到算子 {operator_name} 在 cann_bench 模块中")

        except ImportError as e:
            raise ImportError(f"无法导入 cann_bench 模块: {e}")

    def evaluate_case(self, case: CaseInfo, ai_op_func: Callable = None) -> EvalCaseResult:
        """评测单个用例"""
        case_id_str = case.get_case_id_str()

        try:
            # 1. 获取golden函数
            golden_func = self.golden_loader.get_golden_function(case.level, case.operator)

            # 2. 生成输入数据
            input_tensors = self.data_generator.generate_input_tensors_from_case(
                input_shapes=case.input_shapes,
                dtypes=case.dtypes,
                value_ranges=case.value_ranges,
            )

            # 2.5 调用 get_input 预处理（如果存在）
            get_input_func = self.golden_loader.get_input_function(case.level, case.operator)
            if get_input_func is not None:
                params_for_get_input = self.param_builder.build_call_params(golden_func, case, input_tensors)
                # 将 case.attrs 中的 _exist 标志也传递给 get_input
                case_attrs = getattr(case, 'attrs', None) or {}
                for attr_key, attr_val in case_attrs.items():
                    if attr_key not in params_for_get_input:
                        params_for_get_input[attr_key] = attr_val
                # 确保 skip2_exist 被传递
                if 'skip2_exist' not in params_for_get_input:
                    params_for_get_input['skip2_exist'] = case_attrs.get('skip2_exist', True)
                input_tensors = get_input_func(**params_for_get_input)
                if isinstance(input_tensors, tuple):
                    input_tensors = list(input_tensors)

            # 3. 构建调用参数
            params = self.param_builder.build_call_params(golden_func, case, input_tensors)

            # 4. 执行Golden函数获取参考结果
            golden_result = self.op_runner.run_golden(golden_func, params, case_id_str, input_tensors)
            if not golden_result.success:
                return EvalCaseResult(
                    case_id=case_id_str,
                    level=case.level,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=golden_result,
                    error_msg=f"Golden执行失败: {golden_result.error}",
                    baseline_perf_us=case.baseline_perf_us,
                )

            # 5. 如果没有AI算子，只返回Golden结果（用于测试Golden正确性）
            if ai_op_func is None:
                return EvalCaseResult(
                    case_id=case_id_str,
                    level=case.level,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=True,
                    golden_run_result=golden_result,
                    accuracy_result=AccuracyResult(passed=True, dtype=case.dtypes[0] if case.dtypes else 'float32', threshold=0, mere=0, mare=0),
                    baseline_perf_us=case.baseline_perf_us,
                )

            # 6. 精度验证：先执行AI算子（不开profiler，只跑一次），确认精度通过后再采集性能
            ai_result = self.op_runner.run_ai_op(ai_op_func, params, case_id_str, input_tensors, enable_perf=False)
            if not ai_result.success:
                return EvalCaseResult(
                    case_id=case_id_str,
                    level=case.level,
                    operator=case.operator,
                    case_num=case.case_id,
                    success=False,
                    golden_run_result=golden_result,
                    ai_run_result=ai_result,
                    error_msg=f"AI算子执行失败: {ai_result.error}",
                    baseline_perf_us=case.baseline_perf_us,
                )

            # 7. 精度对比
            import torch
            # 优先使用AI算子输出的dtype
            dtype = None
            if isinstance(ai_result.outputs, torch.Tensor):
                dtype = str(ai_result.outputs.dtype).replace('torch.', '')
            elif isinstance(ai_result.outputs, (list, tuple)) and ai_result.outputs:
                first_out = ai_result.outputs[0]
                if isinstance(first_out, torch.Tensor):
                    dtype = str(first_out.dtype).replace('torch.', '')

            # 如果无法从AI output确定dtype，使用case中定义的输出dtype
            if dtype is None:
                try:
                    op_info = self.operator_loader.get_operator(case.operator, case.level)
                    if op_info and op_info.outputs and op_info.outputs[0].dtype:
                        output_dtype_list = op_info.outputs[0].dtype
                        if isinstance(output_dtype_list, list):
                            dtype = output_dtype_list[0]
                        else:
                            dtype = output_dtype_list
                except Exception:
                    pass

            # 最后使用case的输入dtype作为后备
            if dtype is None:
                dtype = case.dtypes[0] if case.dtypes else 'float32'

            # 获取算子自定义精度阈值（如果有）
            merged_thresholds = self._get_merged_thresholds(case.operator, case.level)

            # 计算 CPU 同精度输出（用于小值域比较）
            # 使用原始 dtype 的输入调用 golden_fn
            cpu_inputs = []
            for item in input_tensors:
                if isinstance(item, torch.Tensor):
                    cpu_inputs.append(item.cpu())
                elif isinstance(item, (list, tuple)):
                    cpu_inputs.append([sub.cpu() if isinstance(sub, torch.Tensor) else sub for sub in item])
                else:
                    cpu_inputs.append(item)

            cpu_params = self.param_builder.build_call_params(golden_func, case, cpu_inputs)
            with torch.no_grad():
                cpu_out = golden_func(**cpu_params)

            # 保留多输出格式，不做截断
            # cpu_out 可能是 tuple/list（多输出）或 Tensor（单输出）

            # 获取需要跳过的输出索引
            ignore_output_indices = self._get_ignore_output_indices(case.operator, case.level)

            accuracy_result = self.accuracy_evaluator.evaluate(
                ai_output=ai_result.outputs,
                golden_output=golden_result.outputs,
                dtype=dtype,
                custom_thresholds=merged_thresholds,
                cpu_output=cpu_out,
                ignore_output_indices=ignore_output_indices,
            )

            # 8. 精度通过后才采集性能（避免对精度不合格的算子浪费profiling时间）
            perf_result = None
            if accuracy_result.passed and self.perf_evaluator and self.perf_evaluator.enabled:
                ai_perf_result = self.op_runner.run_ai_op(
                    ai_op_func, params, case_id_str, input_tensors, enable_perf=True
                )
                if ai_perf_result.success:
                    perf_result = ai_perf_result.perf_result
                    if perf_result is None and ai_perf_result.elapsed_us > 0:
                        perf_result = PerfResult(
                            case_id=case_id_str,
                            elapsed_us=ai_perf_result.elapsed_us,
                        )
            elif accuracy_result.passed:
                # 关闭性能采集时用simple timing作为参考耗时
                if ai_result.elapsed_us > 0:
                    perf_result = PerfResult(
                        case_id=case_id_str,
                        elapsed_us=ai_result.elapsed_us,
                    )
            else:
                # 精度不通过，用simple timing兜底
                if ai_result.elapsed_us > 0:
                    perf_result = PerfResult(
                        case_id=case_id_str,
                        elapsed_us=ai_result.elapsed_us,
                    )

            # 清理内存
            self._cleanup_memory()

            return EvalCaseResult(
                case_id=case_id_str,
                level=case.level,
                operator=case.operator,
                case_num=case.case_id,
                success=accuracy_result.passed,
                accuracy_result=accuracy_result,
                perf_result=perf_result,
                golden_run_result=golden_result,
                ai_run_result=ai_result,
                baseline_perf_us=case.baseline_perf_us,
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            return EvalCaseResult(
                case_id=case_id_str,
                level=case.level,
                operator=case.operator,
                case_num=case.case_id,
                success=False,
                error_msg=f"评测异常: {e}",
            )

    def evaluate_operator(
        self,
        operator: str,
        level: int,
        ai_op_func: Callable = None,
        case_filter: Dict = None,
    ) -> EvalOperatorResult:
        """
        评测单个算子

        Args:
            operator: 算子名称
            level: 难度级别
            ai_op_func: AI算子函数（可选，如果不提供则只测试Golden）
            case_filter: 用例筛选条件（可选）

        Returns:
            EvalOperatorResult: 算子评测结果
        """
        # 加载用例
        cases = self.case_loader.scan_by_operator(level, operator)

        # 应用筛选条件
        if case_filter:
            cases = self._filter_cases(cases, case_filter)

        if not cases:
            return EvalOperatorResult(
                operator=operator,
                level=level,
                total_cases=0,
                passed_cases=0,
                failed_cases=0,
                skipped_cases=0,
                results=[],
                pass_rate=0.0,
                avg_speedup=0.0,
            )

        # 清空AI算子缓存（确保使用最新加载的函数）
        self._ai_op_cache.clear()

        # 逐个评测
        results = []
        print(f"[INFO] 评测算子 {operator} (L{level}), 用例数: {len(cases)}")
        for i, case in enumerate(cases, 1):
            case_id_str = case.get_case_id_str()
            result = self.evaluate_case(case, ai_op_func)
            results.append(result)

            # 打印进度
            status_icon = "✅" if result.success else "❌"
            if result.success and result.perf_result and result.perf_result.elapsed_us > 0:
                elapsed_str = f"{result.perf_result.elapsed_us:.2f}μs"
            elif result.ai_run_result and result.ai_run_result.elapsed_us > 0:
                elapsed_str = f"{result.ai_run_result.elapsed_us:.2f}μs"
            else:
                elapsed_str = "N/A"
            speedup_str = f"{result.get_speedup():.2f}x" if result.get_speedup() > 0 else "N/A"
            if result.success:
                print(f"[{i}/{len(cases)}] {case_id_str}: {status_icon} (耗时: {elapsed_str}, 加速比: {speedup_str})")
            else:
                print(f"[{i}/{len(cases)}] {case_id_str}: {status_icon}")

        # 性能解析已在 op_runner.run 中完成，无需额外等待
        # self.perf_evaluator.wait_all()

        # 计算统计
        passed = sum(1 for r in results if r.success)
        failed = sum(1 for r in results if not r.success and r.accuracy_result is not None)
        skipped = sum(1 for r in results if not r.success and r.accuracy_result is None)

        # 计算平均加速比（只考虑通过的用例）
        speedups = [r.get_speedup() for r in results if r.success and r.get_speedup() > 0]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0.0

        return EvalOperatorResult(
            operator=operator,
            level=level,
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=failed,
            skipped_cases=skipped,
            results=results,
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
        iterative_compile: bool = True,
    ) -> EvalSessionResult:
        """
        从源码目录执行完整评测

        流程：
        1. 扫描源码目录 + **迭代编译**（失败算子被隔离到 _quarantine/，
           错误摘要带到 package_info.compile_errors）
        2. 安装包
        3. 扫描接口
        4. 为每个"编译失败"的算子合成 FAIL 记录
        5. 逐个评测剩下的算子（默认每个算子跑在独立子进程里，一个算子
           挂死 AI Core 不会污染后面）
        6. 返回评测结果

        Args:
            source_dir: 源码目录路径
            operator_filter: 算子筛选列表
            case_filter: 用例筛选条件
            verbose: 详细输出
            subprocess_isolation: 每个算子 fork 子进程评测（默认 True）
            op_timeout_sec: 子进程隔离下的 per-op 超时（默认 240 秒）

        Returns:
            EvalSessionResult: 评测会话结果（包含运行正常的算子 +
            编译失败的算子 + 子进程异常的算子）
        """
        print("")
        print("=" * 60)
        print("开始评测")
        print("=" * 60)

        # 1. 准备环境（编译安装）—— 迭代模式下失败算子会被隔离，错误摘要
        #    落到 package_info.compile_errors；关闭迭代（老行为）则 build.sh
        #    首次失败即 raise，整个评测停在这一步。
        matched_operators, package_info = self.package_manager.prepare_from_source(
            source_dir,
            verbose=verbose,
            iterative_compile=iterative_compile,
        )

        # 给 kernel_bench 里有对应定义的"编译失败"算子合成 FAIL 记录，让它们
        # 不至于从报告里消失。matched_operators 不含这些（因为 cann_bench 里
        # 没有它们的函数），这里从 compile_errors 反向查找算子定义。
        compile_failed_results: List[EvalOperatorResult] = []
        for snake_op_name, err in (package_info.compile_errors or {}).items():
            op_info = self._find_operator_info_by_snake(snake_op_name)
            if op_info is None:
                # kernel_bench 里没这个算子，静默丢弃（和"submission 多出"情况对齐）
                continue
            if operator_filter and op_info.name not in operator_filter:
                continue
            compile_failed_results.append(
                self._synthesize_compile_failure(op_info, err, case_filter)
            )

        if not matched_operators and not compile_failed_results:
            return EvalSessionResult(
                operators=[],
                package_info=package_info,
            )

        # 2. 应用算子筛选
        if operator_filter:
            matched_operators = [op for op in matched_operators if op in operator_filter]
            print(f"[INFO] 筛选后算子: {matched_operators}")

        # 3. 逐个评测算子（先把编译失败合成结果合入，再跑可运行的算子）
        results: List[EvalOperatorResult] = list(compile_failed_results)
        print(f"[INFO] 编译失败: {len(compile_failed_results)} 个算子 | 可运行: "
              f"{len(matched_operators)} 个算子 "
              f"({'subprocess-per-op' if subprocess_isolation else 'in-process'})")
        for operator_name in matched_operators:
            # 获取算子信息（确定level）
            op_info = self._find_operator_info(operator_name)
            if not op_info:
                print(f"[WARN] 算子 {operator_name} 未找到定义，跳过")
                continue

            if subprocess_isolation:
                result = self._run_operator_subprocess(
                    operator_name, op_info.level, source_dir, case_filter, op_timeout_sec
                )
                results.append(result)
                continue

            # 加载AI算子函数
            try:
                ai_op_func = self.load_ai_operator(operator_name)
            except Exception as e:
                print(f"[ERROR] 加载算子 {operator_name} 失败: {e}")
                continue

            # 执行评测
            result = self.evaluate_operator(
                operator=operator_name,
                level=op_info.level,
                ai_op_func=ai_op_func,
                case_filter=case_filter,
            )
            results.append(result)

        print("")
        print("=" * 60)
        print("评测完成")
        print("=" * 60)

        return EvalSessionResult(
            operators=results,
            package_info=package_info,
        )

    def evaluate_skip_build(
        self,
        operator_filter: List[str] = None,
        case_filter: Dict = None,
    ) -> EvalSessionResult:
        """
        跳过编译安装，直接评测已安装的cann_bench

        Args:
            operator_filter: 算子筛选列表
            case_filter: 用例筛选条件

        Returns:
            EvalSessionResult: 评测会话结果
        """
        print("")
        print("=" * 60)
        print("开始评测（跳过编译安装）")
        print("=" * 60)

        # 扫描已安装的cann_bench接口
        matched_operators = self.package_manager.prepare_skip_build()

        if not matched_operators:
            return EvalSessionResult(operators=[])

        # 应用算子筛选
        if operator_filter:
            matched_operators = [op for op in matched_operators if op in operator_filter]
            print(f"[INFO] 筛选后算子: {matched_operators}")

        # 逐个评测算子
        results = []
        for operator_name in matched_operators:
            # 获取算子信息（确定level）
            op_info = self._find_operator_info(operator_name)
            if not op_info:
                print(f"[WARN] 算子 {operator_name} 未找到定义，跳过")
                continue

            # 加载AI算子函数
            try:
                ai_op_func = self.load_ai_operator(operator_name)
            except Exception as e:
                print(f"[ERROR] 加载算子 {operator_name} 失败: {e}")
                continue

            # 执行评测
            result = self.evaluate_operator(
                operator=operator_name,
                level=op_info.level,
                ai_op_func=ai_op_func,
                case_filter=case_filter,
            )
            results.append(result)

        print("")
        print("=" * 60)
        print("评测完成")
        print("=" * 60)

        return EvalSessionResult(operators=results)

    def evaluate_golden_only(
        self,
        operator: str,
        level: int,
        case_filter: Dict = None,
    ) -> EvalOperatorResult:
        """
        仅执行Golden验证（不安装whl包）

        Args:
            operator: 算子名称
            level: 难度级别
            case_filter: 用例筛选条件

        Returns:
            EvalOperatorResult: 算子评测结果
        """
        return self.evaluate_operator(
            operator=operator,
            level=level,
            ai_op_func=None,  # 不加载AI算子
            case_filter=case_filter,
        )

    def _find_operator_info(self, operator_name: str) -> Optional[OperatorInfo]:
        """查找算子定义信息"""
        operators = self.operator_loader.list_operators()
        for op_info in operators:
            if op_info.name == operator_name:
                return op_info
        return None

    def _find_operator_info_by_snake(self, snake_name: str) -> Optional[OperatorInfo]:
        """通过 snake_case 名称（build_submission 里的 op 目录名）反查 OperatorInfo。
        与 load_ai_operator 的 CamelCase→snake_case 规则保持一致。"""
        target = snake_name.lower()
        operators = self.operator_loader.list_operators()
        for op_info in operators:
            if target in _snake_case_candidates(op_info.name):
                return op_info
        return None

    def _synthesize_compile_failure(
        self,
        op_info: OperatorInfo,
        error_excerpt: str,
        case_filter: Optional[Dict] = None,
    ) -> EvalOperatorResult:
        """为编译失败的算子生成一条 all-FAIL 的 EvalOperatorResult，
        这样它仍然出现在 session 结果里，summary.md / 报告看得到原因。"""
        from ..data.case_loader import CaseInfo as _CI  # 避免循环导入
        try:
            cases = self.case_loader.scan_by_operator(op_info.level, op_info.name)
            if case_filter:
                cases = self._filter_cases(cases, case_filter)
        except Exception:
            cases = []

        # 取错误摘要的第一行做 case-level detail（大段错误单独放在 op-level 字段里）
        first_line = (error_excerpt.strip().splitlines() or ["(no detail)"])[0]
        reason_short = f"compile failed: {first_line[:180]}"

        case_results: List[EvalCaseResult] = []
        for c in cases:
            case_results.append(EvalCaseResult(
                case_id=str(getattr(c, "case_id", 0)),
                level=op_info.level,
                operator=op_info.name,
                case_num=int(getattr(c, "case_id", 0) or 0),
                success=False,
                error_msg=reason_short,
            ))

        return EvalOperatorResult(
            operator=op_info.name,
            level=op_info.level,
            total_cases=len(case_results),
            passed_cases=0,
            failed_cases=len(case_results),
            skipped_cases=0,
            results=case_results,
            pass_rate=0.0,
            avg_speedup=0.0,
            compilation_error=error_excerpt,
        )

    def _synthesize_subprocess_failure(
        self,
        operator_name: str,
        level: int,
        reason: str,
        case_filter: Optional[Dict] = None,
    ) -> EvalOperatorResult:
        """子进程超时 / 崩溃时合成 all-FAIL 的 EvalOperatorResult。"""
        try:
            cases = self.case_loader.scan_by_operator(level, operator_name)
            if case_filter:
                cases = self._filter_cases(cases, case_filter)
        except Exception:
            cases = []

        short = f"subprocess failed: {reason}"
        case_results: List[EvalCaseResult] = []
        for c in cases:
            case_results.append(EvalCaseResult(
                case_id=str(getattr(c, "case_id", 0)),
                level=level,
                operator=operator_name,
                case_num=int(getattr(c, "case_id", 0) or 0),
                success=False,
                error_msg=short,
            ))

        return EvalOperatorResult(
            operator=operator_name,
            level=level,
            total_cases=len(case_results),
            passed_cases=0,
            failed_cases=len(case_results),
            skipped_cases=0,
            results=case_results,
            pass_rate=0.0,
            avg_speedup=0.0,
            subprocess_failure_reason=reason,
        )

    def _run_operator_subprocess(
        self,
        operator_name: str,
        level: int,
        source_dir: str,
        case_filter: Optional[Dict],
        timeout_sec: int,
    ) -> EvalOperatorResult:
        """Fork 一个子进程运行单个算子的评测。子进程用 --skip-install
        +  --no-subprocess-isolation 避免重复安装和无限递归；超时先 SIGTERM
        给 finally 块做 NPU 清理的机会，10s 宽限后 SIGKILL。成功则 load 子
        进程写出的 JSON，lift 出这个算子的 EvalOperatorResult；异常则合成
        一条 subprocess_failure_reason 记录。"""
        import json as _json
        import subprocess as _sp
        import tempfile as _tf

        fd, frag_path = _tf.mkstemp(suffix=".json", prefix="cannbench_child_")
        os.close(fd)
        try:
            # 子进程通过 `python -m kernel_eval.cli eval --source-dir X
            # --operator <name> --skip-install --no-subprocess-isolation
            # --child-json-output <frag>` 的形式被调起。skip-install 走
            # prepare_skip_build 路径（wheel 已安装，不用再跑 build）。
            cmd = [
                sys.executable, "-m", "kernel_eval.cli", "eval",
                "--source-dir", str(source_dir),
                "--operator", operator_name,
                "--child-json-output", frag_path,
                "--no-subprocess-isolation",
                "--skip-install",
            ]
            if case_filter and "case_id" in case_filter:
                cmd += ["--case-id", str(case_filter["case_id"])]

            print(f"[INFO] {operator_name}: subprocess (timeout {timeout_sec}s)")
            # 确保子进程能找到 kernel_eval 模块：将当前模块所在目录（src/）添加到 PYTHONPATH
            env = os.environ.copy()
            kernel_eval_root = str(Path(__file__).parent.parent.parent)  # evaluator.py -> eval -> kernel_eval -> src
            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                # 避免重复添加
                paths = existing_pythonpath.split(":")
                if kernel_eval_root not in paths:
                    env["PYTHONPATH"] = f"{kernel_eval_root}:{existing_pythonpath}"
            else:
                env["PYTHONPATH"] = kernel_eval_root
            proc = _sp.Popen(cmd, start_new_session=True, env=env)
            try:
                rc = proc.wait(timeout=timeout_sec)
                if rc != 0:
                    return self._synthesize_subprocess_failure(
                        operator_name, level,
                        f"subprocess exited rc={rc}", case_filter,
                    )
            except _sp.TimeoutExpired:
                print(f"[WARN] {operator_name} 超过 {timeout_sec}s — SIGTERM")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except _sp.TimeoutExpired:
                    print(f"[WARN] {operator_name} 宽限后仍未退出 — SIGKILL")
                    proc.kill()
                    proc.wait()
                return self._synthesize_subprocess_failure(
                    operator_name, level,
                    f"exceeded {timeout_sec}s timeout — killed", case_filter,
                )

            if not os.path.exists(frag_path) or os.path.getsize(frag_path) == 0:
                return self._synthesize_subprocess_failure(
                    operator_name, level,
                    "subprocess produced no output", case_filter,
                )
            try:
                data = _json.loads(Path(frag_path).read_text())
            except Exception as e:
                return self._synthesize_subprocess_failure(
                    operator_name, level,
                    f"parse child JSON: {e}", case_filter,
                )
            ops = data.get("operators", [])
            if not ops:
                return self._synthesize_subprocess_failure(
                    operator_name, level,
                    "subprocess output had no operators", case_filter,
                )
            # rehydrate EvalOperatorResult from dict — 最小必要字段
            op_d = ops[0]
            case_results: List[EvalCaseResult] = []
            for r in op_d.get("results", []):
                case_results.append(EvalCaseResult(
                    case_id=str(r.get("case_id", "")),
                    level=r.get("level", level),
                    operator=r.get("operator", operator_name),
                    case_num=int(r.get("case_num", r.get("case_id", 0) or 0)),
                    success=bool(r.get("success", False)),
                    error_msg=r.get("error_msg"),
                    baseline_perf_us=r.get("baseline_perf_us", 0.0),
                ))
            return EvalOperatorResult(
                operator=op_d.get("operator", operator_name),
                level=op_d.get("level", level),
                total_cases=op_d.get("total_cases", len(case_results)),
                passed_cases=op_d.get("passed_cases", 0),
                failed_cases=op_d.get("failed_cases", len(case_results)),
                skipped_cases=op_d.get("skipped_cases", 0),
                results=case_results,
                pass_rate=op_d.get("pass_rate", 0.0),
                avg_speedup=op_d.get("avg_speedup", 0.0),
            )
        finally:
            try:
                os.unlink(frag_path)
            except OSError:
                pass

    def _filter_cases(self, cases: List[CaseInfo], filter_dict: Dict) -> List[CaseInfo]:
        """筛选用例"""
        result = cases
        if 'case_id' in filter_dict:
            result = [c for c in result if c.case_id == filter_dict['case_id']]
        if 'dtype' in filter_dict:
            result = [c for c in result if filter_dict['dtype'].lower() in [d.lower() for d in c.dtypes]]
        return result

    def _get_merged_thresholds(self, operator: str, level: int) -> Dict[str, float]:
        """获取合并后的精度阈值（全局 + 自定义，自定义优先）"""
        # 从proto.yaml获取自定义阈值
        op_info = self.operator_loader.get_operator(operator, level)
        if op_info and op_info.precision_thresholds:
            # 合并：全局阈值为基础，自定义阈值覆盖
            merged = dict(self.config.precision_thresholds)
            merged.update(op_info.precision_thresholds)
            return merged
        return self.config.precision_thresholds

    def _get_ignore_output_indices(self, operator: str, level: int) -> List[int]:
        """获取需要忽略对比的输出索引列表"""
        ignore_indices = []
        op_info = self.operator_loader.get_operator(operator, level)
        if op_info and op_info.outputs:
            for idx, output in enumerate(op_info.outputs):
                if not output.compare:
                    ignore_indices.append(idx)
        return ignore_indices

    def _cleanup_memory(self):
        """清理内存"""
        gc.collect()
        try:
            import torch
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                import torch_npu
                if hasattr(torch_npu, 'npu'):
                    torch_npu.npu.empty_cache()
            except ImportError:
                pass
        except Exception:
            pass

    def shutdown(self):
        """关闭评测器，释放资源"""
        self.perf_evaluator.shutdown()