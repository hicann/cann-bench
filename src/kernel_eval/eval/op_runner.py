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
算子执行器

职责：
1. 执行golden函数或AI算子
2. 采集性能数据
3. 返回执行结果
"""

import traceback
import io
import sys
from typing import Callable, Dict, Optional, Any, List
from dataclasses import dataclass
from contextlib import contextmanager

import torch

from ..utils.device_manager import DeviceManager
from .perf_eval import PerfEvaluator, PerfResult


@contextmanager
def capture_output():
    """捕获 stdout/stderr 输出，用于记录算子执行时的日志"""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    captured_out = io.StringIO()
    captured_err = io.StringIO()
    try:
        sys.stdout = captured_out
        sys.stderr = captured_err
        yield captured_out, captured_err
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@dataclass
class OpRunResult:
    """执行结果"""
    success: bool
    outputs: Optional[Any] = None
    error: Optional[str] = None
    elapsed_us: float = 0
    perf_result: Optional[PerfResult] = None
    device: str = ""
    traceback: Optional[str] = None
    captured_output: Optional[str] = None  # 捕获的 stdout/stderr


class OpRunner:
    """算子执行器"""

    def __init__(self, device_manager: DeviceManager, perf_evaluator: PerfEvaluator = None):
        self.device_manager = device_manager
        self.perf_evaluator = perf_evaluator

    def run(self, func: Callable, params: Dict, case_id: str, input_tensors: List,
            to_device: bool = True, enable_profiler: bool = False) -> OpRunResult:
        """执行函数

        Args:
            func: 要执行的函数
            params: 调用参数字典
            case_id: 用例标识
            input_tensors: 输入张量列表
            to_device: 是否将输入迁移到设备端
            enable_profiler: 是否启用 profiler 采集性能（仅 AI 算子启用，Golden 不启用）
        """
        captured_output = ""
        try:
            # 迁移输入到设备
            if to_device:
                device_tensors = self.device_manager.to_device_batch(input_tensors)
                updated_params = self._update_params(params, device_tensors)
            else:
                updated_params = self._update_params(params, input_tensors)

            # 执行
            use_profiler = (enable_profiler
                         and self.perf_evaluator is not None
                         and self.perf_evaluator.config.enable_profiler
                         and self.device_manager.is_npu_mode())
            if use_profiler:
                outputs, perf_result = self.perf_evaluator.run_profiled(case_id, func, **updated_params)
                self.perf_evaluator.wait_all()
                elapsed_us = perf_result.elapsed_us
                # profiler 路径：异常被 run_profiled 捕获存在 error_msg 中，outputs 为 None
                if outputs is None and perf_result.error_msg:
                    error_msg = f"算子执行失败: {perf_result.error_msg}"
                    # 如果有 traceback，追加到错误信息中
                    tb = perf_result.metadata.get('profile_exception_traceback')
                    if tb:
                        error_msg = f"{error_msg}\n{tb}"
                    raise RuntimeError(error_msg)
            else:
                # 非 profiler 路径(--no-perf / CPU / golden):只跑出 outputs 供精度比对,
                # 不再用墙钟计时(受环境影响大、与 profiler 设备时间不可比)。perf_result=None,
                # 由评分侧将该 case 的 perf 分按 0 计入,不影响 function/total。
                try:
                    with capture_output() as (cap_out, cap_err):
                        if to_device:
                            outputs = func(**updated_params)
                            self.device_manager.synchronize()
                        else:
                            with torch.no_grad():
                                outputs = func(**updated_params)
                    captured_output = cap_out.getvalue() + cap_err.getvalue()
                except Exception:
                    # 异常发生时也要保存捕获的输出
                    captured_output = cap_out.getvalue() + cap_err.getvalue()
                    raise
                elapsed_us = 0.0
                perf_result = None

            # 检查输出：函数返回 None 通常意味着算子不支持当前 dtype 或执行静默失败
            if outputs is None:
                raise RuntimeError(
                    "算子执行返回 None，可能不支持当前 dtype "
                    f"（请检查算子是否支持 {self._infer_dtype(input_tensors)}）"
                )

            return OpRunResult(
                success=True,
                outputs=outputs,
                elapsed_us=elapsed_us,
                perf_result=perf_result,
                device="cpu" if not to_device else self.device_manager.get_device()
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = str(e)
            if captured_output.strip():
                error_msg += f"\n算子执行期间输出:\n{captured_output.strip()}"
            return OpRunResult(
                success=False,
                error=error_msg,
                elapsed_us=0,
                device="cpu" if not to_device else self.device_manager.get_device(),
                traceback=tb_str,
                captured_output=captured_output if captured_output.strip() else None
            )

    @staticmethod
    def _format_perf_error(perf_result: PerfResult) -> str:
        error = perf_result.error_msg or "profiling failed"
        tb_str = perf_result.metadata.get("profile_exception_traceback")
        if tb_str and tb_str.strip() not in error:
            return f"{error}\n{tb_str.rstrip()}"
        return error

    def run_ai_op(self, ai_op_func: Callable, params: Dict, case_id: str, input_tensors: List,
                  enable_perf: bool = True) -> OpRunResult:
        """执行AI算子（受 TorchOpGuard 监视，检测对禁用 builtin 数学 API 的调用）"""
        # 防作弊监听：detect AI op calling torch.matmul / conv / softmax 等
        guard_mode = "block"
        if self.perf_evaluator is not None:
            guard_mode = getattr(self.perf_evaluator.config, 'torch_op_guard_mode', 'block')

        from ..security.torch_op_guard import TorchOpGuard
        from ..security.device_residency_guard import DeviceResidencyGuard
        # 设备驻留守卫：拦"在 CPU 上算完、再把结果拷回 NPU"的作弊（成块 NPU→CPU 外流）。
        # H2D（搬输入/权重上卡）方向相反、天然忽略，不会误伤正常加载。
        drg_mode = guard_mode
        if self.perf_evaluator is not None:
            drg_mode = getattr(self.perf_evaluator.config, 'device_residency_guard_mode', guard_mode)
        with TorchOpGuard(mode=guard_mode), DeviceResidencyGuard(mode=drg_mode):
            # 如果需要性能采集且evaluator可用，临时启用
            if enable_perf and self.perf_evaluator:
                return self.run(ai_op_func, params, case_id, input_tensors,
                                enable_profiler=enable_perf)
            else:
                # 不采集性能，简单执行
                return self._run_simple(ai_op_func, params, input_tensors)

    def _run_simple(self, func: Callable, params: Dict, input_tensors: List) -> OpRunResult:
        """简单执行（不采集性能）"""
        captured_output = ""
        try:
            device_tensors = self.device_manager.to_device_batch(input_tensors)
            updated_params = self._update_params(params, device_tensors)

            # 不采集性能:只跑出 outputs(墙钟已弃用,perf 由 profiler 路径专责)。
            try:
                with capture_output() as (cap_out, cap_err):
                    outputs = func(**updated_params)
                    self.device_manager.synchronize()
                captured_output = cap_out.getvalue() + cap_err.getvalue()
            except Exception:
                # 异常发生时也要保存捕获的输出
                captured_output = cap_out.getvalue() + cap_err.getvalue()
                raise

            # 检查输出
            if outputs is None:
                raise RuntimeError(
                    "算子执行返回 None，可能不支持当前 dtype "
                    f"（请检查算子是否支持 {self._infer_dtype(input_tensors)}）"
                )

            elapsed_us = 0.0

            return OpRunResult(
                success=True,
                outputs=outputs,
                elapsed_us=elapsed_us,
                device=self.device_manager.get_device()
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            error_msg = str(e)
            if captured_output.strip():
                error_msg += f"\n算子执行期间输出:\n{captured_output.strip()}"
            return OpRunResult(
                success=False,
                error=error_msg,
                elapsed_us=0,
                device=self.device_manager.get_device(),
                traceback=tb_str,
                captured_output=captured_output if captured_output.strip() else None
            )

    def _update_params(self, params: Dict, device_tensors: List) -> Dict:
        """更新参数中的张量引用"""
        import torch
        updated = {}
        tensor_idx = 0

        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                # 跳过None值
                while tensor_idx < len(device_tensors) and device_tensors[tensor_idx] is None:
                    tensor_idx += 1
                if tensor_idx < len(device_tensors):
                    item = device_tensors[tensor_idx]
                    updated[key] = item if isinstance(item, torch.Tensor) else item[0]
                    tensor_idx += 1
                else:
                    updated[key] = self.device_manager.to_device(value)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                while tensor_idx < len(device_tensors) and device_tensors[tensor_idx] is None:
                    tensor_idx += 1
                if tensor_idx < len(device_tensors):
                    updated[key] = (
                        device_tensors[tensor_idx]
                        if isinstance(device_tensors[tensor_idx], list)
                        else [device_tensors[tensor_idx]]
                    )
                    tensor_idx += 1
                else:
                    updated[key] = [
                        self.device_manager.to_device(t) if isinstance(t, torch.Tensor) else t
                        for t in value
                    ]
            else:
                updated[key] = value

        return updated

    def _infer_dtype(self, input_tensors: List) -> str:
        """从输入张量推断 dtype（用于错误信息）"""
        import torch
        for t in input_tensors:
            if isinstance(t, torch.Tensor):
                return str(t.dtype).replace('torch.', '')
            elif isinstance(t, (list, tuple)) and t:
                for sub in t:
                    if isinstance(sub, torch.Tensor):
                        return str(sub.dtype).replace('torch.', '')
        return "unknown"
