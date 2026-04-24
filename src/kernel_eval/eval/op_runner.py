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

import time
import traceback
from typing import Callable, Dict, Optional, Any, List
from dataclasses import dataclass

from ..utils.device_manager import DeviceManager
from .perf_eval import PerfEvaluator, PerfResult


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


class OpRunner:
    """算子执行器"""

    def __init__(self, device_manager: DeviceManager, perf_evaluator: PerfEvaluator = None):
        self.device_manager = device_manager
        self.perf_evaluator = perf_evaluator

    def run(self, func: Callable, params: Dict, case_id: str, input_tensors: List) -> OpRunResult:
        """执行函数"""
        try:
            # 迁移输入到设备
            device_tensors = self.device_manager.to_device_batch(input_tensors)
            updated_params = self._update_params(params, device_tensors)

            # 执行
            use_profiler = self.perf_evaluator is not None and self.perf_evaluator.enabled and self.device_manager.is_npu_mode()
            if use_profiler:
                outputs, perf_result = self.perf_evaluator.run_profiled(case_id, func, **updated_params)
                # 等待当前 case 的解析完成，获取性能数据
                self.perf_evaluator.wait_all()
                elapsed_us = perf_result.elapsed_us
            else:
                self.device_manager.synchronize()
                t0 = time.perf_counter()
                outputs = func(**updated_params)
                self.device_manager.synchronize()
                elapsed_us = (time.perf_counter() - t0) * 1_000_000
                perf_result = None

            return OpRunResult(
                success=True,
                outputs=outputs,
                elapsed_us=elapsed_us,
                perf_result=perf_result,
                device=self.device_manager.get_device()
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            return OpRunResult(
                success=False,
                error=str(e),
                elapsed_us=0,
                device=self.device_manager.get_device(),
                traceback=tb_str
            )

    def run_golden(self, golden_func: Callable, params: Dict, case_id: str, input_tensors: List) -> OpRunResult:
        """Execute the golden reference on CPU. Single shot, no profiler, no
        device transfer — golden is the precision reference, not the
        performance subject.

        Floats are promoted to fp64 so the reference is more precise than
        the device's native dtype; the accuracy checker casts both sides
        back to fp32 for MERE/MARE so thresholds stay keyed to the device
        dtype. Integer/bool tensors keep their dtype — some ops refuse a
        Double substitute.

        Running golden on the device instead of on CPU isn't safe in general:
        some device kernels can be orders of magnitude slower than CPU for
        certain dtype/shape combinations, and a subset return wrong values
        on edge cases, which would silently corrupt the reference and flip
        correct AI ops to FAIL. Golden is for correctness, not performance.
        """
        import torch
        try:
            def _to_fp64_cpu(t: torch.Tensor) -> torch.Tensor:
                t = t.cpu()
                return t.double() if t.is_floating_point() else t

            cpu_tensors: List[Any] = []
            for item in input_tensors:
                if isinstance(item, torch.Tensor):
                    cpu_tensors.append(_to_fp64_cpu(item))
                elif isinstance(item, (list, tuple)):
                    cpu_tensors.append([
                        _to_fp64_cpu(sub) if isinstance(sub, torch.Tensor) else sub
                        for sub in item
                    ])
                else:
                    cpu_tensors.append(item)

            updated_params = self._update_params(params, cpu_tensors)

            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = golden_func(**updated_params)
            elapsed_us = (time.perf_counter() - t0) * 1_000_000

            return OpRunResult(
                success=True,
                outputs=outputs,
                elapsed_us=elapsed_us,
                device="cpu",
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            return OpRunResult(
                success=False,
                error=str(e),
                elapsed_us=0,
                device="cpu",
                traceback=tb_str,
            )

    def run_ai_op(self, ai_op_func: Callable, params: Dict, case_id: str, input_tensors: List,
                  enable_perf: bool = True) -> OpRunResult:
        """执行AI算子"""
        # 如果需要性能采集且evaluator可用，临时启用
        if enable_perf and self.perf_evaluator:
            return self.run(ai_op_func, params, case_id, input_tensors)
        else:
            # 不采集性能，简单执行
            return self._run_simple(ai_op_func, params, input_tensors)

    def _run_simple(self, func: Callable, params: Dict, input_tensors: List) -> OpRunResult:
        """简单执行（不采集性能）"""
        try:
            device_tensors = self.device_manager.to_device_batch(input_tensors)
            updated_params = self._update_params(params, device_tensors)

            self.device_manager.synchronize()
            t0 = time.perf_counter()
            outputs = func(**updated_params)
            self.device_manager.synchronize()
            elapsed_us = (time.perf_counter() - t0) * 1_000_000

            return OpRunResult(
                success=True,
                outputs=outputs,
                elapsed_us=elapsed_us,
                device=self.device_manager.get_device()
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            return OpRunResult(
                success=False,
                error=str(e),
                elapsed_us=0,
                device=self.device_manager.get_device(),
                traceback=tb_str
            )

    def _update_params(self, params: Dict, device_tensors: List) -> Dict:
        """更新参数中的张量引用"""
        import torch
        updated = {}
        tensor_idx = 0

        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                if tensor_idx < len(device_tensors):
                    item = device_tensors[tensor_idx]
                    updated[key] = item if isinstance(item, torch.Tensor) else item[0]
                    tensor_idx += 1
                else:
                    updated[key] = self.device_manager.to_device(value)
            elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                if tensor_idx < len(device_tensors):
                    updated[key] = device_tensors[tensor_idx] if isinstance(device_tensors[tensor_idx], list) else [device_tensors[tensor_idx]]
                    tensor_idx += 1
                else:
                    updated[key] = [self.device_manager.to_device(t) if isinstance(t, torch.Tensor) else t for t in value]
            else:
                updated[key] = value

        return updated