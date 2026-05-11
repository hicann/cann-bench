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

import torch
from typing import Optional, Tuple

"""
Unique 算子 Torch Golden 参考实现

去除张量中的重复元素
公式：y, inverse = unique(x, return_inverse)
"""
def unique(
    x: torch.Tensor,
    return_inverse: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    去除张量中的重复元素

    公式：y, inverse = unique(x, return_inverse)

    Args:
        x: 输入张量
        return_inverse: 是否返回逆索引，用于重建原始张量

    Returns:
        y: 唯一值张量
        inverse: 逆索引，满足 x = y[inverse] (当 return_inverse=True 时)
    """

    if return_inverse:
        y, inverse = torch.unique(x, return_inverse=True)
        return y, inverse
    else:
        y = torch.unique(x, return_inverse=False)
        return y, None
