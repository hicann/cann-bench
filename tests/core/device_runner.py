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
设备执行器

职责：
1. 执行golden函数
2. 采集性能数据
3. 返回执行结果
"""

import time
import traceback
from typing import Callable, Dict, Optional, Any, List
from dataclasses import dataclass

from ..utils.device_manager import DeviceManager
from .profiler_manager import ProfilerManager


@dataclass
class DeviceRunResult:
    """执行结果"""
    success: bool
    outputs: Optional[Any] = None
    error: Optional[str] = None
    elapsed_us: float = 0
    profiler_result: Any = None
    device: str = ""
    traceback: Optional[str] = None


class DeviceRunner:
    """设备执行器"""

    def __init__(self, device_manager: DeviceManager, profiler_manager: ProfilerManager):
        self.device_manager = device_manager
        self.profiler_manager = profiler_manager

    def run(self, golden_func: Callable, params: Dict, case_id: str, input_tensors: List) -> DeviceRunResult:
        """执行golden函数"""
        try:
            # 迁移输入到设备
            device_tensors = self.device_manager.to_device_batch(input_tensors)
            updated_params = self._update_params(params, device_tensors)

            # 执行
            use_profiler = self.profiler_manager.enabled and self.device_manager.is_npu_mode()
            if use_profiler:
                outputs, profiler_result = self.profiler_manager.run_profiled(case_id, golden_func, **updated_params)
                # 等待当前 case 的解析完成，获取性能数据
                self.profiler_manager.wait_all()
                elapsed_us = profiler_result.elapsed_us
            else:
                self.device_manager.synchronize()
                t0 = time.perf_counter()
                outputs = golden_func(**updated_params)
                self.device_manager.synchronize()
                elapsed_us = (time.perf_counter() - t0) * 1_000_000
                profiler_result = None

            return DeviceRunResult(
                success=True,
                outputs=outputs,
                elapsed_us=elapsed_us,
                profiler_result=profiler_result,
                device=self.device_manager.get_device()
            )

        except Exception as e:
            tb_str = traceback.format_exc()
            return DeviceRunResult(
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