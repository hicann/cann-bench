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
返回值类型检查模块

职责：
1. 检查算子返回值是否为torch.Tensor
2. 拒绝FakeTensor、懒求值包装器等非真实Tensor
3. 验证Tensor的有效性（非空、正确的dtype和shape）

防护原理：
- 攻击者可能返回FakeTensor或懒求值包装器
- 这些对象可能通过eq-style检查但从未真正计算
- 使用type(obj) is torch.Tensor严格检查类型
"""

import torch
from typing import Any, Optional, Tuple, List, Union


def check_output_type(
    output: Any,
    expected_type: type = torch.Tensor,
    strict: bool = True
) -> bool:
    """
    检查返回值类型

    Args:
        output: 算子返回值
        expected_type: 期望的类型
        strict: 是否使用严格类型检查（type() is而非isinstance）

    Returns:
        类型检查是否通过

    Raises:
        RuntimeError: 如果类型不匹配且strict=True
    """
    if strict:
        # 严格检查：type(obj) is torch.Tensor
        # 这可以拒绝子类伪装攻击（如FakeTensor）
        if type(output) is not expected_type:
            raise RuntimeError(
                f"[SECURITY] 算子返回值类型不匹配: 期望 {expected_type.__name__}, "
                f"实际 {type(output).__name__}\n"
                "可能存在懒求值或子类伪装攻击"
            )
        return True
    else:
        #宽松检查：isinstance(obj, torch.Tensor)
        if not isinstance(output, expected_type):
            raise RuntimeError(
                f"算子返回值类型不匹配: 期望 {expected_type.__name__}, "
                f"实际 {type(output).__name__}"
            )
        return True


def check_tensor_validity(
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    check_nan_inf: bool = True
) -> bool:
    """
    验证Tensor的有效性

    Args:
        tensor: 待验证的Tensor
        expected_shape: 期望的形状（可选）
        expected_dtype: 期望的数据类型（可选）
        check_nan_inf: 是否检查NaN/Inf

    Returns:
        验证是否通过

    Raises:
        RuntimeError: 如果验证失败
    """
    # 类型检查
    check_output_type(tensor, torch.Tensor, strict=True)

    # 形状检查
    if expected_shape is not None:
        if tuple(tensor.shape) != tuple(expected_shape):
            raise RuntimeError(
                f"Tensor形状不匹配: 期望 {expected_shape}, "
                f"实际 {tuple(tensor.shape)}"
            )

    # dtype检查
    if expected_dtype is not None:
        if tensor.dtype != expected_dtype:
            raise RuntimeError(
                f"Tensor dtype不匹配: 期望 {expected_dtype}, "
                f"实际 {tensor.dtype}"
            )

    # NaN/Inf检查
    if check_nan_inf and tensor.is_floating_point():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if nan_count > 0:
            # NaN不一定是错误，但需要记录
            pass
        if inf_count > 0:
            # Inf也不一定是错误，但需要记录
            pass

    return True


def check_multi_output(
    outputs: Union[torch.Tensor, Tuple, List],
    expected_count: Optional[int] = None,
    strict: bool = True
) -> List[torch.Tensor]:
    """
    检查多输出情况

    Args:
        outputs: 算子输出（单个Tensor或多个）
        expected_count: 期望的输出数量（可选）
        strict: 是否使用严格类型检查

    Returns:
        Tensor列表

    Raises:
        RuntimeError: 如果检查失败
    """
    # 标准化为列表
    if isinstance(outputs, torch.Tensor):
        tensors = [outputs]
    elif isinstance(outputs, (tuple, list)):
        tensors = []
        for item in outputs:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, (tuple, list)):
                # 处理嵌套情况
                for sub_item in item:
                    if isinstance(sub_item, torch.Tensor):
                        tensors.append(sub_item)
    else:
        raise RuntimeError(f"输出类型不支持: {type(outputs).__name__}")

    # 数量检查
    if expected_count is not None:
        if len(tensors) != expected_count:
            raise RuntimeError(
                f"输出数量不匹配: 期望 {expected_count}, "
                f"实际 {len(tensors)}"
            )

    # 类型检查
    for tensor in tensors:
        check_output_type(tensor, torch.Tensor, strict=strict)

    return tensors


def is_real_tensor(obj: Any) -> bool:
    """
    判断是否为真实Tensor（非FakeTensor等）

    Args:
        obj: 待判断对象

    Returns:
        是否为真实Tensor
    """
    return type(obj) is torch.Tensor and isinstance(obj, torch.Tensor)