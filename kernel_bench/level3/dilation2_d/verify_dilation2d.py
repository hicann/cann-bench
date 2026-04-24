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

"""
Dilation2D 对比验证脚本

通过 subprocess 在独立进程中运行 TensorFlow，避免与 PyTorch 的段错误冲突
"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import torch

# 添加 golden 模块路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import dilation2_d


def pytorch_dilation2d(x_np, filter_np, strides, rates, padding):
    """PyTorch Golden 实现"""
    x = torch.from_numpy(x_np)
    filter = torch.from_numpy(filter_np)

    y = dilation2_d(
        x, filter,
        strides=strides,
        rates=rates,
        padding_mode=padding,
        pads=[0, 0, 0, 0],
        ceil_mode=False,
        data_format='NHWC'
    )
    return y.numpy()


def tensorflow_dilation2d_subprocess(x_np, filter_np, strides, rates, padding, work_dir):
    """通过 subprocess 运行 TensorFlow 计算"""
    # 保存输入数据
    np.save(os.path.join(work_dir, 'x.npy'), x_np)
    np.save(os.path.join(work_dir, 'filter.npy'), filter_np)

    # 保存配置
    config = {
        'strides': strides,
        'rates': rates,
        'padding': padding
    }
    config_file = os.path.join(work_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f)

    # 运行 TF 脚本
    tf_script = os.path.join(os.path.dirname(__file__), 'tf_dilation2d.py')
    output_file = os.path.join(work_dir, 'tf_output.npy')

    result = subprocess.run(
        ['python3', tf_script, 'x.npy', output_file, config_file],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        print(f"TF subprocess failed: {result.stderr}")
        return None

    # 读取输出
    return np.load(output_file)


def compare_outputs(pytorch_out, tf_out, rtol=1e-3, atol=1e-3):
    """对比两个输出"""
    print(f"PyTorch output shape: {pytorch_out.shape}")
    print(f"TensorFlow output shape: {tf_out.shape}")

    if pytorch_out.shape != tf_out.shape:
        print("ERROR: Shape mismatch!")
        return False

    # 计算差异
    diff = np.abs(pytorch_out.astype(np.float32) - tf_out.astype(np.float32))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # 检查是否在阈值内
    is_close = np.allclose(pytorch_out, tf_out, rtol=rtol, atol=atol)

    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"Within tolerance (rtol={rtol}, atol={atol}): {is_close}")

    if max_diff > 0:
        # 找出差异最大的位置
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Max diff location: {idx}")
        print(f"  PyTorch value: {pytorch_out[idx]}")
        print(f"  TensorFlow value: {tf_out[idx]}")

    return is_close


def run_test_case(case_id, x_shape, filter_shape, strides, rates, padding):
    """运行单个测试用例"""
    print(f"\n{'='*60}")
    print(f"Test Case {case_id}")
    print(f"{'='*60}")
    print(f"Input shape: {x_shape} (NHWC)")
    print(f"Filter shape: {filter_shape} [H, W, C]")
    print(f"Strides: {strides}")
    print(f"Rates: {rates}")
    print(f"Padding: {padding}")

    # 生成随机数据
    np.random.seed(case_id * 42)
    x_np = np.random.randn(*x_shape).astype(np.float16)
    filter_np = np.random.randn(*filter_shape).astype(np.float16)

    # PyTorch 计算
    print("\n[PyTorch Golden]")
    pytorch_out = pytorch_dilation2d(x_np, filter_np, strides, rates, padding)
    print(f"Output shape: {pytorch_out.shape}")

    # TensorFlow 计算（独立进程）
    print("\n[TensorFlow dilation2d (subprocess)]")
    work_dir = tempfile.mkdtemp(prefix=f'dilation2d_test{case_id}_')
    try:
        tf_out = tensorflow_dilation2d_subprocess(x_np, filter_np, strides, rates, padding, work_dir)
        if tf_out is None:
            print("ERROR: TensorFlow computation failed")
            return False

        print(f"Output shape: {tf_out.shape}")

        # 对比
        print("\n[Comparison: PyTorch vs TensorFlow]")
        is_match = compare_outputs(pytorch_out, tf_out)
        return is_match
    finally:
        # 清理临时文件
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print("Dilation2D 验证脚本: PyTorch Golden vs TensorFlow")
    print("=" * 60)
    print("(TensorFlow 在独立进程中运行，避免与 PyTorch 冲突)")

    # 测试用例
    test_cases = [
        # Case 1: 基本场景
        {
            'id': 1,
            'x_shape': [2, 64, 64, 64],
            'filter_shape': [3, 3, 64],
            'strides': [1, 1, 1, 1],
            'rates': [1, 1, 1, 1],
            'padding': 'SAME'
        },
        # Case 2: stride=2
        {
            'id': 2,
            'x_shape': [2, 32, 32, 16],
            'filter_shape': [3, 3, 16],
            'strides': [1, 2, 2, 1],
            'rates': [1, 1, 1, 1],
            'padding': 'SAME'
        },
        # Case 3: dilation(rate)=2
        {
            'id': 3,
            'x_shape': [1, 16, 16, 8],
            'filter_shape': [3, 3, 8],
            'strides': [1, 1, 1, 1],
            'rates': [1, 2, 2, 1],
            'padding': 'SAME'
        },
        # Case 4: VALID padding
        {
            'id': 4,
            'x_shape': [1, 10, 10, 4],
            'filter_shape': [3, 3, 4],
            'strides': [1, 1, 1, 1],
            'rates': [1, 1, 1, 1],
            'padding': 'VALID'
        },
        # Case 5: 不同 filter 尺寸
        {
            'id': 5,
            'x_shape': [1, 8, 8, 4],
            'filter_shape': [5, 5, 4],
            'strides': [1, 1, 1, 1],
            'rates': [1, 1, 1, 1],
            'padding': 'SAME'
        },
        # Case 6: stride + rate 组合
        {
            'id': 6,
            'x_shape': [2, 20, 20, 8],
            'filter_shape': [3, 3, 8],
            'strides': [1, 2, 2, 1],
            'rates': [1, 2, 2, 1],
            'padding': 'SAME'
        },
    ]

    results = []
    for tc in test_cases:
        try:
            result = run_test_case(
                tc['id'],
                tc['x_shape'],
                tc['filter_shape'],
                tc['strides'],
                tc['rates'],
                tc['padding']
            )
            results.append((tc['id'], result))
        except Exception as e:
            print(f"ERROR in case {tc['id']}: {e}")
            results.append((tc['id'], False))

    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, r in results if r is True)
    failed = len(results) - passed
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")

    if failed > 0:
        print("\nFailed cases:")
        for case_id, result in results:
            if result is False:
                print(f"  - Case {case_id}")

    if passed == len(results):
        print("\n✓ All test cases passed! PyTorch Golden matches TensorFlow dilation2d.")

    return 0 if passed == len(results) else 1


if __name__ == '__main__':
    sys.exit(main())