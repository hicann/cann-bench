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

from typing import List
from dataclasses import dataclass

import torch

from ..config import get_config


@dataclass
class DeviceConfig:
    """设备配置"""
    type: str = "cpu"              # cpu / npu
    device_id: int = 0


class DeviceManager:
    """设备管理器 - 支持CPU和NPU设备切换"""

    def __init__(self, config: DeviceConfig):
        self.config = config
        self._npu_available = self._check_npu()
        self._device = self._resolve_device()
        if self._npu_available:
            torch.npu.set_device(self.config.device_id)

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
            raise RuntimeError(
                "NPU设备不可用。请检查：\n"
                "  1. NPU硬件是否已安装\n"
                "  2. torch_npu是否正确安装（pip install torch_npu）\n"
                "  3. CANN环境是否配置正确\n"
                "若当前环境无NPU，可使用 --device cpu 模式"
            )
        print("[INFO] 使用CPU设备")
        return "cpu"

    def get_device(self) -> str:
        return self._device

    def is_npu_mode(self) -> bool:
        return self._device.startswith("npu")

    def is_cpu_mode(self) -> bool:
        return self._device == "cpu"

    def get_device_name(self) -> str:
        """获取 NPU 设备原始名称（如 "Ascend950PR_xxx"）。

        返回 torch.npu.get_device_name() 的原始字符串，供上层
        （如 baseline_resolver.resolve_hardware）映射为逻辑名。
        DeviceManager 自身不做任何映射。

        Returns:
            设备名称字符串；NPU 不可用或检测失败时返回 "unknown"
        """
        if not self._npu_available:
            return "unknown"
        try:
            return torch.npu.get_device_name(self.config.device_id)
        except Exception:
            return "unknown"

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
        elif hasattr(torch.cpu, 'synchronize'):
            torch.cpu.synchronize()
        # PyTorch < 2.1 不支持 torch.cpu.synchronize，跳过即可

    # ---- 设备健康检查与恢复 ----

    def health_check(self) -> bool:
        """检测 NPU 设备是否健康

        通过 torch.npu.synchronize() 检测：若设备处于 device error 状态
        （如 AIC/AIV error），synchronize 会抛 RuntimeError。
        CPU 模式下始终返回 True。
        """
        if not self.is_npu_mode():
            return True
        try:
            torch.npu.synchronize()
            return True
        except RuntimeError:
            return False

    def recover_light(self) -> bool:
        """轻量级设备恢复

        尝试清理缓存 + 重新设置设备 + 验证同步。
        对内存污染类错误有效，对 AIC/AIV error 导致的 device error 状态无效
        （需要 aclrtResetDevice 才能重置）。

        Returns:
            True: 恢复成功且 synchronize 通过
            False: 恢复失败
        """
        if not self.is_npu_mode():
            return True
        try:
            import torch_npu
            torch_npu.npu.empty_cache()
            torch.npu.set_device(self.config.device_id)
            torch.npu.synchronize()
            return True
        except Exception:
            return False

    def recover_full(self) -> bool:
        """重量级设备恢复（last-resort）

        通过 ctypes 调用 aclrtResetDevice() 彻底重置 NPU 设备，
        然后重新初始化设备上下文并验证同步。
        仅在 recover_light() 失败后才尝试。

        注意：aclrtResetDevice 会释放当前进程在该设备上的所有 ACL 资源
        （stream、event、内存等）。torch_npu 内部缓存（stream pool /
        memory pool）可能指向已释放资源，因此恢复后需立即验证
        synchronize。后续首个 case 若仍失败则不再尝试恢复。

        Returns:
            True: 恢复成功且 synchronize 通过
            False: 恢复失败（包括 aclrtResetDevice 本身失败或同步验证失败）
        """
        if not self.is_npu_mode():
            return True
        try:
            import ctypes
            acl_lib = ctypes.CDLL("libascendcl.so")
            ret = acl_lib.aclrtResetDevice(self.config.device_id)
            if ret != 0:  # ACL_SUCCESS = 0
                return False
            torch.npu.set_device(self.config.device_id)
            torch.npu.synchronize()
            return True
        except Exception:
            return False