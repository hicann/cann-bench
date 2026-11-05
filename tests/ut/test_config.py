#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
配置管理模块单元测试

测试对象：kernel_eval.config
核心功能：
1. Config 数据类
2. get_config / set_config 全局配置
"""

import pytest
from pathlib import Path

from kernel_eval.config import Config, get_config, set_config


class TestConfig:
    """Config 数据类测试"""

    def test_default_config(self):
        """默认配置"""
        config = Config()
        assert config.device_type == "npu"
        assert config.device_id == 0
        assert config.warmup == 3
        assert config.repeat == 5
        assert config.perf_batch_cases is True

    def test_custom_config(self):
        """自定义配置"""
        config = Config(
            tasks_root="/custom/path",
            device_type="cpu",
            device_id=1,
            warmup=5,
            repeat=10,
        )
        assert config.tasks_root == "/custom/path"
        assert config.device_type == "cpu"
        assert config.device_id == 1
        assert config.warmup == 5
        assert config.repeat == 10

    def test_post_init_sets_default_paths(self):
        """初始化后设置默认路径"""
        config = Config()
        # 默认路径应在项目根目录下
        assert config.tasks_root != ""
        assert config.reports_dir != ""

    def test_get_tasks_path(self):
        """获取 tasks 路径"""
        config = Config(tasks_root="/test/path")
        path = config.get_tasks_path()
        assert isinstance(path, Path)
        assert str(path) == str(Path("/test/path"))

    def test_get_reports_path(self):
        """获取报告路径"""
        config = Config(reports_dir="/reports")
        path = config.get_reports_path()
        assert isinstance(path, Path)
        assert str(path) == str(Path("/reports"))

    def test_get_source_path(self):
        """获取源码路径"""
        config = Config(source_dir="/source")
        path = config.get_source_path()
        assert isinstance(path, Path)
        assert str(path) == str(Path("/source"))

    def test_get_source_path_empty(self):
        """空源码路径"""
        config = Config()
        path = config.get_source_path()
        assert path is None

    def test_precision_thresholds(self):
        """精度阈值配置"""
        config = Config()
        assert "float16" in config.precision_thresholds
        assert "float32" in config.precision_thresholds
        assert "int32" in config.precision_thresholds

    def test_precision_threshold_values(self):
        """精度阈值值"""
        config = Config()
        # float16 阈值约为 2^-10
        assert config.precision_thresholds["float16"] == pytest.approx(2**-10)
        # float32 阈值约为 2^-13
        assert config.precision_thresholds["float32"] == pytest.approx(2**-13)
        # int 类型阈值为 0
        assert config.precision_thresholds["int32"] == 0

    def test_enable_profiler_default(self):
        """默认启用 profiler"""
        config = Config()
        assert config.enable_profiler is True

    def test_profiler_level_default(self):
        """默认 profiler 级别"""
        config = Config()
        assert config.profiler_level == "Level1"


class TestGetSetConfig:
    """get_config / set_config 函数测试"""

    def test_get_config_returns_config_instance(self):
        """get_config 返回 Config 实例"""
        config = get_config()
        assert isinstance(config, Config)

    def test_set_config(self):
        """set_config 设置配置（conftest autouse fixture 自动恢复）"""
        custom_config = Config(
            device_type="cpu",
            device_id=2,
        )
        set_config(custom_config)
        config = get_config()
        assert config.device_type == "cpu"
        assert config.device_id == 2


class TestConfigPathResolution:
    """配置路径解析测试"""

    def test_tasks_root_relative(self):
        """相对路径 tasks_root"""
        config = Config(tasks_root="relative/path")
        assert config.tasks_root == "relative/path"

    def test_reports_dir_absolute(self):
        """绝对路径 reports_dir"""
        config = Config(reports_dir="/absolute/reports")
        assert config.reports_dir == "/absolute/reports"

    def test_empty_paths_use_defaults(self):
        """空路径使用默认"""
        config = Config(tasks_root="", reports_dir="")
        # __post_init__ 会设置默认值
        assert config.tasks_root != ""
        assert config.reports_dir != ""

    def test_source_dir_optional(self):
        """源码目录可选"""
        config = Config(source_dir="")
        assert config.source_dir == ""
        assert config.get_source_path() is None


class TestConfigDeviceSettings:
    """配置设备设置测试"""

    def test_npu_device(self):
        """NPU 设备"""
        config = Config(device_type="npu", device_id=0)
        assert config.device_type == "npu"

    def test_cpu_device(self):
        """CPU 设备"""
        config = Config(device_type="cpu")
        assert config.device_type == "cpu"

    def test_multiple_device_ids(self):
        """多设备 ID"""
        for device_id in range(4):
            config = Config(device_id=device_id)
            assert config.device_id == device_id


class TestConfigPerformanceSettings:
    """配置性能设置测试"""

    def test_warmup_repeat_values(self):
        """预热和重复次数"""
        config = Config(warmup=10, repeat=20)
        assert config.warmup == 10
        assert config.repeat == 20

    def test_profiler_disabled(self):
        """禁用 profiler"""
        config = Config(enable_profiler=False)
        assert config.enable_profiler is False

    def test_profiler_level_2(self):
        """Profiler Level2"""
        config = Config(profiler_level="Level2")
        assert config.profiler_level == "Level2"


class TestConfigPrecisionSettings:
    """配置精度设置测试"""

    def test_custom_precision_thresholds(self):
        """自定义精度阈值"""
        custom_thresholds = {
            "float16": 0.001,
            "float32": 0.0001,
        }
        config = Config(precision_thresholds=custom_thresholds)
        assert config.precision_thresholds["float16"] == 0.001
        assert config.precision_thresholds["float32"] == 0.0001

    def test_all_dtype_thresholds_present(self):
        """所有 dtype 阈值存在"""
        config = Config()
        expected_dtypes = [
            "float16",
            "bfloat16",
            "float32",
            "hifloat32",
            "float8_e4m3fn",
            "float8_e5m2",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ]
        for dtype in expected_dtypes:
            assert dtype in config.precision_thresholds
