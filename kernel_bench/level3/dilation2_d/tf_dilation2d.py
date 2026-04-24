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
TensorFlow dilation2d 计算脚本（独立进程运行）

读取 dump 的输入数据，计算 dilation2d 结果，dump 输出
"""

import os
import sys
import json
import numpy as np

def run_tf_dilation2d(input_file, output_file, config_file):
    """使用 TensorFlow 计算 dilation2d"""
    import tensorflow as tf

    # 读取配置
    with open(config_file, 'r') as f:
        config = json.load(f)

    # 读取输入数据
    x_np = np.load(os.path.join(os.path.dirname(input_file), 'x.npy'))
    filter_np = np.load(os.path.join(os.path.dirname(input_file), 'filter.npy'))

    # 转换为 TF tensor
    x = tf.constant(x_np, dtype=tf.float16)
    filters = tf.constant(filter_np, dtype=tf.float16)

    # TF 2.21 API: dilation2d(input, filters, strides, padding, data_format, dilations)
    # strides 格式: [batch_stride, h_stride, w_stride, channel_stride]
    # dilations 格式: [batch_dilation, h_dilation, w_dilation, channel_dilation]
    y = tf.nn.dilation2d(
        input=x,
        filters=filters,
        strides=config['strides'],
        padding=config['padding'].upper(),
        data_format='NHWC',
        dilations=config['rates']
    )

    # 保存输出
    np.save(output_file, y.numpy())
    print(f"TF output saved to {output_file}")
    print(f"Output shape: {y.shape}")


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python tf_dilation2d.py <input_file> <output_file> <config_file>")
        sys.exit(1)

    run_tf_dilation2d(sys.argv[1], sys.argv[2], sys.argv[3])