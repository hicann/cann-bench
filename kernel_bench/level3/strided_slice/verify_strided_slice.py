#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""验证 PyTorch golden 与 TensorFlow strided_slice 对比"""

import torch
import numpy as np
import subprocess
import json

def strided_slice(x: torch.Tensor, begin: list, end: list, strides: list,
    begin_mask: int = 0, end_mask: int = 0, ellipsis_mask: int = 0,
    shrink_axis_mask: int = 0, new_axis_mask: int = 0
) -> torch.Tensor:
    """PyTorch strided_slice golden 实现"""
    ndim = x.dim()
    shape = x.shape

    # 处理 ellipsis_mask
    ellipsis_pos = None
    for i in range(32):
        if ellipsis_mask & (1 << i):
            ellipsis_pos = i
            break

    # 计算 new_axis 数量
    num_new_axis = 0
    for i in range(len(begin) if begin else 0):
        if new_axis_mask & (1 << i):
            num_new_axis += 1

    indices = []
    input_dim_idx = 0
    param_idx = 0

    if ellipsis_pos is not None:
        num_params = len(begin) if begin else 0
        num_ellipsis_dims = ndim - (num_params - num_new_axis - 1)
        if num_ellipsis_dims < 0:
            num_ellipsis_dims = 0

    while input_dim_idx < ndim or param_idx < (len(begin) if begin else 0):
        if param_idx < len(begin) and (new_axis_mask & (1 << param_idx)):
            indices.append(None)
            param_idx += 1
            continue

        if ellipsis_pos is not None and param_idx == ellipsis_pos:
            for _ in range(num_ellipsis_dims):
                indices.append(slice(None, None, None))
                input_dim_idx += 1
            param_idx += 1
            continue

        if input_dim_idx < ndim and param_idx < len(begin):
            dim_size = shape[input_dim_idx]
            b = begin[param_idx] if param_idx < len(begin) else 0
            e = end[param_idx] if param_idx < len(end) else dim_size
            s = strides[param_idx] if param_idx < len(strides) else 1

            if b < 0:
                b = b + dim_size
            if e < 0:
                e = e + dim_size

            if begin_mask & (1 << param_idx):
                b = 0 if s > 0 else dim_size - 1

            if end_mask & (1 << param_idx):
                e = dim_size if s > 0 else -1

            if shrink_axis_mask & (1 << param_idx):
                indices.append(b)
            else:
                indices.append(slice(b, e, s))

            input_dim_idx += 1
            param_idx += 1
        elif input_dim_idx < ndim:
            indices.append(slice(None, None, None))
            input_dim_idx += 1
        else:
            if param_idx < len(begin) and (new_axis_mask & (1 << param_idx)):
                indices.append(None)
            param_idx += 1

    return x[tuple(indices)]


def run_tf_strided_slice(data, begin, end, strides, begin_mask=0, end_mask=0,
                         ellipsis_mask=0, shrink_axis_mask=0, new_axis_mask=0):
    """通过 subprocess 运行 TF strided_slice 避免 segfault"""
    tf_code = '''
import tensorflow as tf
import numpy as np
import json
import sys

params = json.loads(sys.argv[1])
data = np.array(params['data'])
x = tf.constant(data)

y = tf.strided_slice(
    x,
    begin=params['begin'],
    end=params['end'],
    strides=params['strides'],
    begin_mask=params['begin_mask'],
    end_mask=params['end_mask'],
    ellipsis_mask=params['ellipsis_mask'],
    shrink_axis_mask=params['shrink_axis_mask'],
    new_axis_mask=params['new_axis_mask']
)

result = {
    'shape': list(y.shape),
    'data': y.numpy().tolist()
}
print(json.dumps(result))
'''

    params = {
        'data': data.tolist(),
        'begin': begin,
        'end': end,
        'strides': strides,
        'begin_mask': begin_mask,
        'end_mask': end_mask,
        'ellipsis_mask': ellipsis_mask,
        'shrink_axis_mask': shrink_axis_mask,
        'new_axis_mask': new_axis_mask
    }

    result = subprocess.run(
        ['python3', '-c', tf_code, json.dumps(params)],
        capture_output=True, text=True, timeout=30
    )

    if result.returncode != 0:
        raise RuntimeError(f"TF error: {result.stderr}")

    return json.loads(result.stdout)


def test_case(name, data, begin, end, strides, **masks):
    """单个测试用例"""
    print(f"\n--- {name} ---")

    # TF 结果
    tf_result = run_tf_strided_slice(data, begin, end, strides, **masks)

    # PT 结果
    pt_x = torch.tensor(data)
    pt_y = strided_slice(pt_x, begin, end, strides, **masks)

    # 对比
    tf_shape = tf_result['shape']
    pt_shape = list(pt_y.shape)
    tf_data = np.array(tf_result['data'])
    pt_data = pt_y.numpy()

    shape_match = tf_shape == pt_shape
    data_match = np.allclose(tf_data, pt_data) if tf_data.shape == pt_data.shape else False

    print(f"TF shape: {tf_shape}")
    print(f"PT shape: {pt_shape}")
    print(f"Shape match: {shape_match}")
    print(f"Data match: {data_match}")

    if not shape_match or not data_match:
        print("TF result:\n", tf_data)
        print("PT result:\n", pt_data)

    return shape_match and data_match


if __name__ == '__main__':
    print("=== PyTorch Golden vs TensorFlow strided_slice ===")

    data = np.arange(24).reshape(2, 3, 4)
    print(f"Input shape: {data.shape}")

    all_passed = True

    # Test 1: 基本切片
    all_passed &= test_case("Test 1: basic slice",
        data, [0, 0, 0], [2, 3, 4], [1, 1, 1])

    # Test 2: begin_mask
    all_passed &= test_case("Test 2: begin_mask=1",
        data, [1, 1, 0], [2, 3, 4], [1, 1, 1], begin_mask=1)

    # Test 3: end_mask
    all_passed &= test_case("Test 3: end_mask=1",
        data, [0, 0, 0], [1, 1, 2], [1, 1, 1], end_mask=1)

    # Test 4: shrink_axis_mask=1
    all_passed &= test_case("Test 4: shrink_axis_mask=1",
        data, [1, 0, 0], [2, 3, 4], [1, 1, 1], shrink_axis_mask=1)

    # Test 5: shrink_axis_mask=2
    all_passed &= test_case("Test 5: shrink_axis_mask=2",
        data, [0, 1, 0], [2, 2, 4], [1, 1, 1], shrink_axis_mask=2)

    # Test 6: shrink_axis_mask=3
    all_passed &= test_case("Test 6: shrink_axis_mask=3",
        data, [1, 1, 0], [2, 2, 4], [1, 1, 1], shrink_axis_mask=3)

    # Test 7: new_axis_mask=1
    all_passed &= test_case("Test 7: new_axis_mask=1",
        data, [0, 0, 0], [2, 3, 4], [1, 1, 1], new_axis_mask=1)

    # Test 8: new_axis_mask=2
    all_passed &= test_case("Test 8: new_axis_mask=2",
        data, [0, 0, 0], [2, 3, 4], [1, 1, 1], new_axis_mask=2)

    # Test 9: 组合 shrink + end_mask
    all_passed &= test_case("Test 9: shrink_axis_mask=1 + end_mask=4",
        data, [0, 0, 0], [1, 3, 2], [1, 1, 1],
        shrink_axis_mask=1, end_mask=4)

    print(f"\n=== Summary ===")
    print(f"All passed: {all_passed}")