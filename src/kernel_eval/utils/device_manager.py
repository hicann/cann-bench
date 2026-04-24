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
设备管理器

职责：
1. 检测NPU/CPU可用性
2. 提供设备切换接口
3. 处理张量设备迁移
"""

import torch
from typing import List
from dataclasses import dataclass


@dataclass
class DeviceConfig:
    """设备配置"""
    type: str = "cpu"              # cpu / npu
    device_id: int = 0
    auto_fallback: bool = True


class DeviceManager:
    """设备管理器 - 支持CPU和NPU设备切换"""

    def __init__(self, config: DeviceConfig):
        self.config = config
        self._npu_available = self._check_npu()
        self._device = self._resolve_device()

    def _check_npu(self) -> bool:
        """检测NPU是否可用"""
        if self.config.type != "npu":
            return False
        try:
            import torch_npu
            return torch.npu.is_available()
        except ImportError:
            return False

    def _resolve_device(self) -> str:
        """确定执行设备"""
        if self.config.type == "npu":
            if self._npu_available:
                device = f"npu:{self.config.device_id}"
                print(f"[INFO] 使用NPU设备: {device}")
                return device
            elif self.config.auto_fallback:
                print("[WARN] NPU不可用，回退到CPU")
                return "cpu"
            else:
                raise RuntimeError("NPU不可用且未启用自动回退")
        print("[INFO] 使用CPU设备")
        return "cpu"

    def get_device(self) -> str:
        return self._device

    def is_npu_mode(self) -> bool:
        return self._device.startswith("npu")

    def is_cpu_mode(self) -> bool:
        return self._device == "cpu"

    def to_device(self, tensor):
        return tensor.to(self._device)

    def to_device_batch(self, tensors: List) -> List:
        """批量迁移张量到当前设备"""
        result = []
        for item in tensors:
            if isinstance(item, torch.Tensor):
                result.append(item.to(self._device))
            elif isinstance(item, list):
                result.append([t.to(self._device) for t in item])
            else:
                result.append(item)
        return result

    def synchronize(self):
        """同步设备"""
        if self.is_npu_mode():
            torch.npu.synchronize()
        else:
            torch.cpu.synchronize()