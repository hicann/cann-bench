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
进程池协调器单元测试

测试覆盖：
1. ProcessConfig 配置解析
2. ProcessWorker 基本功能（mock subprocess）
3. ProcessPoolCoordinator 任务分配逻辑
4. 分布式任务分配策略
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 添加项目路径
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kernel_eval.eval.process_pool import (
    ProcessConfig,
    ProcessWorker,
    ProcessPoolCoordinator,
)
from src.kernel_eval.eval.results import EvalOperatorResult, EvalCaseResult
from src.kernel_eval.benches import CannCaseSpec
from src.kernel_eval.config import Config


def make_case(operator, case_id, input_shapes=None, dtypes=None, value_ranges=None, rel_path="level1/test"):
    """创建测试用例的辅助函数"""
    # value_ranges should be List[Dict[str, float]] format
    vr = value_ranges or [{"min": -1, "max": 1}]
    return CannCaseSpec(
        case_id=f"{rel_path}_{case_id}",
        rel_path=rel_path,
        operator=operator,
        case_num=case_id,
        input_shapes=input_shapes or [[1024, 1024]],
        dtypes=dtypes or ["float32"],
        attrs={},
        value_ranges=vr,
        metadata={},
    )


class TestProcessConfig(unittest.TestCase):
    """测试 ProcessConfig 配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = ProcessConfig()
        self.assertEqual(config.processes_per_card, 2)
        self.assertEqual(config.timeout_per_operator, 300)
        self.assertTrue(config.enable_profiler)

    def test_custom_config(self):
        """测试自定义配置"""
        config = ProcessConfig(
            processes_per_card=4,
            timeout_per_operator=600,
            enable_profiler=False,
        )
        self.assertEqual(config.processes_per_card, 4)
        self.assertEqual(config.timeout_per_operator, 600)
        self.assertFalse(config.enable_profiler)


class TestProcessWorker(unittest.TestCase):
    """测试 ProcessWorker"""

    def setUp(self):
        """测试前准备"""
        self.base_config = Config()
        self.base_config.tasks_root = str(project_root / "tasks")
        self.process_config = ProcessConfig()

    def test_worker_creation(self):
        """测试工作单元创建"""
        worker = ProcessWorker(
            process_id=0,
            card_id=0,
            base_config=self.base_config,
            process_config=self.process_config,
        )
        self.assertEqual(worker.process_id, 0)
        self.assertEqual(worker.card_id, 0)
        self.assertFalse(worker._started)

    def test_serialize_cases(self):
        """测试用例序列化"""
        cases = [
            make_case("Sigmoid", 1, [[1024, 1024]], ["float32"]),
            make_case("Sigmoid", 2, [[2048, 2048]], ["float16"]),
        ]

        worker = ProcessWorker(
            process_id=0,
            card_id=0,
            base_config=self.base_config,
            process_config=self.process_config,
        )

        serialized = worker._serialize_cases(cases)
        self.assertEqual(len(serialized), 2)
        self.assertEqual(serialized[0]['operator'], "Sigmoid")
        self.assertEqual(serialized[0]['case_id'], "level1/test_1")
        self.assertEqual(serialized[1]['input_shapes'], [[2048, 2048]])

    @patch('subprocess.Popen')
    def test_worker_start_mock(self, mock_popen):
        """测试启动子进程（mock）"""
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        worker = ProcessWorker(
            process_id=0,
            card_id=0,
            base_config=self.base_config,
            process_config=self.process_config,
        )

        cases = [make_case("Sigmoid", 1)]

        worker.start(cases=cases)

        self.assertTrue(worker._started)
        mock_popen.assert_called_once()

        # 验证命令参数
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        self.assertIn("eval-process", cmd)
        self.assertIn("--process-id", cmd)
        self.assertIn("0", cmd)

    def test_cleanup(self):
        """测试临时文件清理"""
        worker = ProcessWorker(
            process_id=0,
            card_id=0,
            base_config=self.base_config,
            process_config=self.process_config,
        )

        # 创建临时文件
        fd, temp_file = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        worker._output_file = temp_file

        worker._cleanup()

        self.assertFalse(os.path.exists(temp_file))
        self.assertIsNone(worker._output_file)


class TestProcessPoolCoordinator(unittest.TestCase):
    """测试 ProcessPoolCoordinator"""

    def setUp(self):
        """测试前准备"""
        self.base_config = Config()
        self.base_config.tasks_root = str(project_root / "tasks")
        self.base_config.device_type = "npu"

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_coordinator_creation(self, mock_detect):
        """测试协调器创建"""
        mock_detect.return_value = 2

        process_config = ProcessConfig(processes_per_card=2, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
        )

        self.assertEqual(coordinator.card_count, 2)
        self.assertEqual(coordinator.total_processes, 4)

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_distribute_rel_paths(self, mock_detect):
        """测试算子分配"""
        mock_detect.return_value = 2

        process_config = ProcessConfig(processes_per_card=2, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
        )

        rel_paths = ["level1/Exp", "level1/Sigmoid", "level1/Mish",
                     "level1/SwiGlu", "level1/Add", "level1/Mul",
                     "level1/Div", "level1/Sub"]
        distribution = coordinator.distribute_rel_paths(rel_paths)

        # 验证分配：8 rel_paths 分配到 4 processes
        self.assertEqual(len(distribution), 4)

        # 验证轮询分配
        # Process 0: level1/Exp, level1/Add
        # Process 1: level1/Sigmoid, level1/Mul
        # Process 2: level1/Mish, level1/Div
        # Process 3: level1/SwiGlu, level1/Sub
        self.assertIn("level1/Exp", distribution[0])
        self.assertIn("level1/Add", distribution[0])
        self.assertIn("level1/Sigmoid", distribution[1])
        self.assertIn("level1/Mish", distribution[2])
        self.assertIn("level1/SwiGlu", distribution[3])

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_distribute_cases(self, mock_detect):
        """测试用例分配"""
        mock_detect.return_value = 2

        process_config = ProcessConfig(processes_per_card=2, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
        )

        cases = [make_case("Sigmoid", i) for i in range(20)]

        distribution = coordinator.distribute_cases(cases)

        # 验证分配：20 cases 分配到 4 processes
        self.assertEqual(len(distribution), 4)

        # 验证每个进程的用例数
        total_assigned = sum(len(c) for c in distribution.values())
        self.assertEqual(total_assigned, 20)

        # 验证轮询分配（每个进程应该有 5 个用例）
        for proc_id in range(4):
            self.assertEqual(len(distribution[proc_id]), 5)

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_operator_parallel_card_mapping(self, mock_detect):
        """测试 operator_parallel 模式的卡映射"""
        mock_detect.return_value = 2

        process_config = ProcessConfig(processes_per_card=2, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
        )

        # 验证 process_id 到 card_id 的映射
        # Process 0, 1 -> Card 0
        # Process 2, 3 -> Card 1
        workers = coordinator._create_workers()

        self.assertEqual(len(workers), 4)
        self.assertEqual(workers[0].card_id, 0)
        self.assertEqual(workers[1].card_id, 0)
        self.assertEqual(workers[2].card_id, 1)
        self.assertEqual(workers[3].card_id, 1)

    def test_no_cards_fallback(self):
        """测试无 NPU 卡时的回退"""
        self.base_config.device_type = "cpu"

        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=ProcessConfig(),
        )

        self.assertEqual(coordinator.card_count, 0)
        self.assertEqual(coordinator.total_processes, 0)

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_single_card_mode(self, mock_detect):
        """测试单卡模式（指定 device_id）"""
        mock_detect.return_value = 2  # 环境有 2 卡

        process_config = ProcessConfig(processes_per_card=3, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
            device_id=0,  # 指定单卡
        )

        # 单卡模式：card_count=1，所有进程绑定到 device_id=0
        self.assertEqual(coordinator.card_count, 1)
        self.assertEqual(coordinator.device_id, 0)
        self.assertEqual(coordinator.total_processes, 3)  # 1 卡 × 3 进程

        # 验证 workers 都绑定到 card 0
        workers = coordinator._create_workers()
        for w in workers:
            self.assertEqual(w.card_id, 0)

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_multi_card_mode(self, mock_detect):
        """测试多卡模式（不指定 device_id）"""
        mock_detect.return_value = 2  # 环境有 2 卡

        process_config = ProcessConfig(processes_per_card=2, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
            # 不指定 device_id，自动检测
        )

        # 多卡模式：card_count=2，进程轮询分配
        self.assertEqual(coordinator.card_count, 2)
        self.assertIsNone(coordinator.device_id)
        self.assertEqual(coordinator.total_processes, 4)  # 2 卡 × 2 进程

        # 验证 workers 轮询分配到各卡
        workers = coordinator._create_workers()
        self.assertEqual(workers[0].card_id, 0)
        self.assertEqual(workers[1].card_id, 0)
        self.assertEqual(workers[2].card_id, 1)
        self.assertEqual(workers[3].card_id, 1)


class TestIntegration(unittest.TestCase):
    """集成测试"""

    def test_cli_eval_process_help(self):
        """测试 CLI eval-process 命令解析"""
        # 直接测试 CLI parser
        from src.kernel_eval.cli import create_parser

        parser = create_parser()

        # 测试 eval-process 命令参数解析
        args = parser.parse_args([
            'eval-process',
            '--process-id', '0',
            '--card-id', '0',
            '--output', '/tmp/test.json',
            '--rel-paths', 'level1/sigmoid,level1/exp',
        ])

        self.assertEqual(args.command, 'eval-process')
        self.assertEqual(args.process_id, 0)
        self.assertEqual(args.card_id, 0)
        self.assertEqual(args.output, '/tmp/test.json')
        self.assertEqual(args.rel_paths, 'level1/sigmoid,level1/exp')

    def test_coordinator_stats(self):
        """测试协调器统计信息"""
        base_config = Config()
        base_config.device_type = "cpu"  # 避免 NPU 检测

        coordinator = ProcessPoolCoordinator(
            base_config=base_config,
            process_config=ProcessConfig(processes_per_card=3, enable_profiler=False),
        )

        stats = coordinator.get_stats()
        self.assertIn('device_id', stats)
        self.assertIn('card_count', stats)
        self.assertIn('processes_per_card', stats)
        self.assertIn('total_processes', stats)
        self.assertEqual(stats['processes_per_card'], 3)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)