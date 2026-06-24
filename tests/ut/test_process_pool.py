#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
进程池协调器 + 子进程公共工具 单元测试

测试覆盖：
1. ProcessConfig 配置解析
2. TaskUnit 与 build_task_units 任务分配
3. aggregate_by_operator 结果聚合
4. ProcessPoolCoordinator 创建与配置
5. subprocess_utils 工具函数（OOM 保护、失败合成、部分结果恢复）
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kernel_eval.eval.process_pool import (
    ProcessConfig,
    TaskUnit,
    build_task_units,
    aggregate_by_operator,
    ProcessPoolCoordinator,
)
from src.kernel_eval.eval.subprocess_utils import (
    _CANN_ENV_VARS,
    _write_oom_score_adj,
    _is_oom_killed,
    _synthesize_failure_cases,
    _try_recover_partial_results,
)
from src.kernel_eval.eval.results import EvalCaseResult, summarize_case_results
from src.kernel_eval.benches import CannCaseSpec
from src.kernel_eval.config import Config


def make_case(operator, case_id, input_shapes=None, dtypes=None, value_ranges=None,
              rel_path="level1/test"):
    """创建测试用例的辅助函数"""
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

    def test_profiler_forces_single_process_per_card(self):
        """profiler 开启时每卡仅 1 进程"""
        base_config = Config()
        base_config.device_type = "npu"
        process_config = ProcessConfig(processes_per_card=4, enable_profiler=True)
        with patch.object(ProcessPoolCoordinator, '_detect_cards', return_value=2):
            coordinator = ProcessPoolCoordinator(
                base_config=base_config,
                process_config=process_config,
            )
        # profiler 开启强制 processes_per_card=1
        self.assertEqual(coordinator.process_config.processes_per_card, 1)
        self.assertEqual(coordinator.total_processes, 2)


class TestTaskUnit(unittest.TestCase):
    """测试 TaskUnit 与 build_task_units"""

    def test_task_unit_creation(self):
        """TaskUnit 基本属性"""
        cases = [make_case("Exp", 1), make_case("Exp", 2)]
        unit = TaskUnit(operator="Exp", rel_path="level1/Exp", cases=cases, device_id=0)
        self.assertEqual(unit.operator, "Exp")
        self.assertEqual(unit.rel_path, "level1/Exp")
        self.assertEqual(len(unit.cases), 2)
        self.assertEqual(unit.device_id, 0)

    def test_build_task_units_single_operator_single_card(self):
        """单算子单卡 → 1 个 TaskUnit"""
        cases = [make_case("Exp", i) for i in range(5)]
        cases_by_op = {"Exp": cases}
        units = build_task_units(cases_by_op, card_count=1)
        self.assertEqual(len(units), 1)
        self.assertEqual(units[0].operator, "Exp")
        self.assertEqual(units[0].device_id, 0)
        self.assertEqual(len(units[0].cases), 5)

    def test_build_task_units_single_operator_multi_card(self):
        """单算子多卡 → 用例均分到各卡"""
        cases = [make_case("Exp", i) for i in range(8)]
        cases_by_op = {"Exp": cases}
        units = build_task_units(cases_by_op, card_count=4)
        self.assertEqual(len(units), 4)
        # 每卡 2 个用例
        for unit in units:
            self.assertEqual(len(unit.cases), 2)

    def test_build_task_units_multi_operator_multi_card(self):
        """多算子多卡 → 每个算子均分到各卡"""
        cases_a = [make_case("Exp", i, rel_path="level1/Exp") for i in range(4)]
        cases_b = [make_case("Sigmoid", i, rel_path="level1/Sigmoid") for i in range(4)]
        cases_by_op = {"Exp": cases_a, "Sigmoid": cases_b}
        units = build_task_units(cases_by_op, card_count=2)
        # 2 算子 × 2 卡 = 4 TaskUnits
        self.assertEqual(len(units), 4)
        exp_units = [u for u in units if u.operator == "Exp"]
        sig_units = [u for u in units if u.operator == "Sigmoid"]
        self.assertEqual(len(exp_units), 2)
        self.assertEqual(len(sig_units), 2)


class TestAggregateByOperator(unittest.TestCase):
    """测试 aggregate_by_operator 结果聚合"""

    def test_aggregate_single_operator(self):
        """单算子结果聚合"""
        passed = EvalCaseResult(case_id="test_1", rel_path="level1/Exp",
                                operator="Exp", case_num=1, success=True)
        failed = EvalCaseResult(case_id="test_2", rel_path="level1/Exp",
                                operator="Exp", case_num=2, success=False, error_msg="err",
                                failure_type="oom_killed")
        results = aggregate_by_operator([passed, failed])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].operator, "Exp")
        self.assertEqual(results[0].passed_cases, 1)
        # summarize_case_results 区分 failed/skipped: accuracy_result=None → skipped
        # failure_type=oom_killed 的结果 accuracy_result=None → skipped
        self.assertEqual(results[0].skipped_cases, 1)

    def test_aggregate_multi_operator(self):
        """多算子结果聚合"""
        r1 = EvalCaseResult(case_id="a_1", rel_path="level1/Exp",
                             operator="Exp", case_num=1, success=True)
        r2 = EvalCaseResult(case_id="b_1", rel_path="level1/Sigmoid",
                             operator="Sigmoid", case_num=1, success=True)
        r3 = EvalCaseResult(case_id="b_2", rel_path="level1/Sigmoid",
                             operator="Sigmoid", case_num=2, success=False, error_msg="err",
                             failure_type="oom_killed")
        results = aggregate_by_operator([r1, r2, r3])
        self.assertEqual(len(results), 2)
        # 每个算子的 passed/skipped 正确
        for op_result in results:
            if op_result.operator == "Exp":
                self.assertEqual(op_result.passed_cases, 1)
            elif op_result.operator == "Sigmoid":
                self.assertEqual(op_result.passed_cases, 1)
                self.assertEqual(op_result.skipped_cases, 1)


class TestProcessPoolCoordinator(unittest.TestCase):
    """测试 ProcessPoolCoordinator"""

    def setUp(self):
        self.base_config = Config()
        self.base_config.tasks_root = str(project_root / "tasks")
        self.base_config.device_type = "npu"

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_coordinator_creation_multi_card(self, mock_detect):
        """多卡模式创建"""
        mock_detect.return_value = 2
        process_config = ProcessConfig(processes_per_card=2, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
        )
        self.assertEqual(coordinator.card_count, 2)
        self.assertEqual(coordinator.total_processes, 4)

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_coordinator_creation_single_card(self, mock_detect):
        """单卡模式（指定 device_id）"""
        mock_detect.return_value = 2
        process_config = ProcessConfig(processes_per_card=3, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
            device_id=0,
        )
        self.assertEqual(coordinator.card_count, 1)
        self.assertEqual(coordinator.device_id, 0)
        self.assertEqual(coordinator.total_processes, 3)

    def test_no_cards_cpu_mode(self):
        """CPU 模式下 card_count=0"""
        self.base_config.device_type = "cpu"
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=ProcessConfig(),
        )
        self.assertEqual(coordinator.card_count, 0)
        self.assertEqual(coordinator.total_processes, 0)

    def test_coordinator_stats(self):
        """统计信息"""
        self.base_config.device_type = "cpu"
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=ProcessConfig(processes_per_card=3, enable_profiler=False),
        )
        stats = coordinator.get_stats()
        self.assertIn('device_id', stats)
        self.assertIn('card_count', stats)
        self.assertIn('processes_per_card', stats)
        self.assertEqual(stats['processes_per_card'], 3)

    def test_build_env_includes_cann_vars(self):
        """环境变量构建包含 CANN 继承"""
        self.base_config.device_type = "cpu"
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=ProcessConfig(),
        )
        env = coordinator._build_env()
        self.assertIn("PYTHONPATH", env)
        self.assertIn("PYTHONUNBUFFERED", env)
        # 应包含 CANN 环境变量继承（如果系统有设置）
        for var in _CANN_ENV_VARS:
            if var in os.environ:
                self.assertIn(var, env)

    def test_build_child_cmd_propagates_reports_dir(self):
        """eval-child 子进程继承父进程 reports_dir"""
        self.base_config.device_type = "cpu"
        self.base_config.reports_dir = "/tmp/cann-bench-reports"
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=ProcessConfig(enable_profiler=True),
        )
        task = TaskUnit(
            operator="Exp",
            rel_path="level1/test",
            cases=[make_case("Exp", 1)],
            device_id=0,
        )
        cmd = coordinator._build_child_cmd(task, "/tmp/cases.json", "/tmp/out.json")
        self.assertIn("--reports-dir", cmd)
        idx = cmd.index("--reports-dir")
        self.assertEqual(cmd[idx + 1], "/tmp/cann-bench-reports")

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_multi_card_child_visibility_is_narrowed(self, mock_detect):
        """多卡 child 只暴露分配到的单张物理卡"""
        mock_detect.return_value = 4
        process_config = ProcessConfig(processes_per_card=1, enable_profiler=False)
        with patch.dict(os.environ, {
            "ASCEND_RT_VISIBLE_DEVICES": "4,5,6,7",
            "ASCEND_VISIBLE_DEVICES": "4,5,6,7",
            "NPU_VISIBLE_DEVICES": "4,5,6,7",
        }, clear=False):
            coordinator = ProcessPoolCoordinator(
                base_config=self.base_config,
                process_config=process_config,
            )
            task = TaskUnit(
                operator="Exp",
                rel_path="level1/Exp",
                cases=[make_case("Exp", 1)],
                device_id=2,
            )
            env = coordinator._build_env_for_task(coordinator._build_env(), task)
            self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "6")
            self.assertEqual(env["ASCEND_VISIBLE_DEVICES"], "6")
            self.assertEqual(env["NPU_VISIBLE_DEVICES"], "6")
            self.assertEqual(env["KERNEL_EVAL_PHYSICAL_DEVICE_ID"], "6")
            self.assertEqual(env["KERNEL_EVAL_LOGICAL_DEVICE_ID"], "0")

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_multi_card_child_uses_logical_device_zero(self, mock_detect):
        """多卡 child 收窄 visibility 后使用逻辑 device 0"""
        mock_detect.return_value = 2
        process_config = ProcessConfig(processes_per_card=1, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
        )
        task = TaskUnit(
            operator="Exp",
            rel_path="level1/Exp",
            cases=[make_case("Exp", 1)],
            device_id=1,
        )
        cmd = coordinator._build_child_cmd(task, "/tmp/cases.json", "/tmp/out.json")
        device_idx = cmd.index("--device-id") + 1
        self.assertEqual(cmd[device_idx], "0")

    @patch('src.kernel_eval.eval.process_pool.ProcessPoolCoordinator._detect_cards')
    def test_single_card_child_keeps_requested_device(self, mock_detect):
        """单卡显式 device_id 模式保持原有 device-id 语义"""
        mock_detect.return_value = 4
        process_config = ProcessConfig(processes_per_card=1, enable_profiler=False)
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=process_config,
            device_id=3,
        )
        task = TaskUnit(
            operator="Exp",
            rel_path="level1/Exp",
            cases=[make_case("Exp", 1)],
            device_id=3,
        )
        cmd = coordinator._build_child_cmd(task, "/tmp/cases.json", "/tmp/out.json")
        device_idx = cmd.index("--device-id") + 1
        self.assertEqual(cmd[device_idx], "3")

    def test_build_child_cmd_passes_tasks_root(self):
        """eval-child 应接收 tasks_root，避免 full rel_path 被重复拼接"""
        self.base_config.tasks_root = "/tmp/tasks"
        coordinator = ProcessPoolCoordinator(
            base_config=self.base_config,
            process_config=ProcessConfig(enable_profiler=False),
            device_id=0,
        )
        task = TaskUnit(
            operator="Exp",
            rel_path="level1/exp",
            cases=[make_case("Exp", 1, rel_path="level1/exp")],
            device_id=0,
        )
        cmd = coordinator._build_child_cmd(task, "/tmp/cases.json", "/tmp/out.json")

        task_dir_idx = cmd.index("--task-dir")
        self.assertEqual(cmd[task_dir_idx + 1], "/tmp/tasks")
        self.assertNotIn("/tmp/tasks/level1/exp", cmd)


class TestSubprocessUtils(unittest.TestCase):
    """测试 subprocess_utils 工具函数"""

    def test_write_oom_score_adj_current_process(self):
        """写入当前进程 oom_score_adj（通常能成功）"""
        # 当前进程 pid，写 0（恢复默认值，不改变行为）
        result = _write_oom_score_adj(os.getpid(), 0)
        # 不强制成功（可能没有权限），但不应抛异常
        # result 是 bool，确认类型正确
        self.assertIsInstance(result, bool)

    def test_write_oom_score_adj_invalid_pid(self):
        """无效 pid 应返回 False"""
        result = _write_oom_score_adj(999999, 1000)
        self.assertFalse(result)

    def test_is_oom_killed_negative_9(self):
        """退出码 -9 是 OOM Kill"""
        mock_proc = MagicMock()
        self.assertTrue(_is_oom_killed(mock_proc, -9))

    def test_is_oom_killed_137(self):
        """退出码 137 (bash) 是 OOM Kill"""
        mock_proc = MagicMock()
        self.assertTrue(_is_oom_killed(mock_proc, 137))

    def test_is_oom_killed_normal_exit(self):
        """正常退出码不是 OOM Kill"""
        mock_proc = MagicMock()
        self.assertFalse(_is_oom_killed(mock_proc, 0))
        self.assertFalse(_is_oom_killed(mock_proc, 1))

    def test_synthesize_failure_cases_oom(self):
        """OOM 失败结果合成"""
        cases = [make_case("Exp", 1), make_case("Exp", 2)]
        results = _synthesize_failure_cases(cases, "oom_killed",
            "子进程被 OOM Killer 杀死")
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertFalse(r.success)
            self.assertEqual(r.failure_type, "oom_killed")
            self.assertIn("OOM Killer", r.error_msg)

    def test_synthesize_failure_cases_timeout(self):
        """超时失败结果合成"""
        cases = [make_case("Exp", 1)]
        results = _synthesize_failure_cases(cases, "timeout",
            "子进程超时被杀")
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].failure_type, "timeout")

    def test_synthesize_failure_cases_preserves_case_attrs(self):
        """失败合成保留 baseline_perf_us 和 t_hw_us"""
        case = make_case("Exp", 1)
        case.baseline_perf_us = 100.0
        case.t_hw_us = 50.0
        results = _synthesize_failure_cases([case], "subprocess_failure", "rc=1")
        self.assertEqual(results[0].baseline_perf_us, 100.0)
        self.assertEqual(results[0].t_hw_us, 50.0)

    def test_try_recover_partial_results_empty_file(self):
        """空文件 → 无可恢复结果"""
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            results = _try_recover_partial_results(tmp)
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(tmp)

    def test_try_recover_partial_results_valid_json(self):
        """有效 JSON → 可恢复部分结果"""
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            payload = {"case_results": [
                {"case_id": "test_1", "rel_path": "level1/Exp",
                 "operator": "Exp", "case_num": 1, "success": True},
            ]}
            Path(tmp).write_text(json.dumps(payload))
            results = _try_recover_partial_results(tmp)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].case_id, "test_1")
        finally:
            os.unlink(tmp)

    def test_try_recover_partial_results_invalid_json(self):
        """无效 JSON → 无可恢复结果"""
        fd, tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            Path(tmp).write_text("{invalid json")
            results = _try_recover_partial_results(tmp)
            self.assertEqual(len(results), 0)
        finally:
            os.unlink(tmp)

    def test_cann_env_vars_list_complete(self):
        """CANN 环境变量列表包含关键变量"""
        essential = ["ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME",
                     "LD_LIBRARY_PATH", "PATH"]
        for var in essential:
            self.assertIn(var, _CANN_ENV_VARS)


class TestCLI(unittest.TestCase):
    """测试 CLI 命令解析"""

    def test_cli_eval_child_parse(self):
        """eval-child 命令参数解析"""
        from src.kernel_eval.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'eval-child',
            '--device-id', '0',
            '--cases-file', '/tmp/cases.json',
            '--output', '/tmp/output.json',
            '--reports-dir', '/tmp/reports',
            '--bench-name', 'cann',
            '--warmup', '3',
            '--repeat', '5',
            '--no-perf',
        ])
        self.assertEqual(args.command, 'eval-child')
        self.assertEqual(args.device_id, 0)
        self.assertEqual(args.cases_file, '/tmp/cases.json')
        self.assertEqual(args.output, '/tmp/output.json')
        self.assertEqual(args.reports_dir, '/tmp/reports')
        self.assertTrue(args.no_perf)

    def test_cli_eval_child_config_uses_reports_dir(self):
        """eval-child 配置使用命令行 reports_dir 而不是默认项目 reports"""
        from src.kernel_eval.cli import create_parser, _create_config_from_args_for_child
        parser = create_parser()
        args = parser.parse_args([
            'eval-child',
            '--device-id', '0',
            '--cases-file', '/tmp/cases.json',
            '--output', '/tmp/output.json',
            '--reports-dir', '/tmp/job-local-reports',
        ])
        config = _create_config_from_args_for_child(args, str(project_root / "tasks"))
        self.assertEqual(config.reports_dir, '/tmp/job-local-reports')

    def test_cli_eval_child_torch_op_guard(self):
        """eval-child 接收 torch-op-guard-mode"""
        from src.kernel_eval.cli import create_parser
        parser = create_parser()
        args = parser.parse_args([
            'eval-child',
            '--device-id', '0',
            '--cases-file', '/tmp/cases.json',
            '--output', '/tmp/output.json',
            '--torch-op-guard-mode', 'block',
        ])
        self.assertEqual(args.torch_op_guard_mode, 'block')

    def test_cli_eval_no_removed_flags(self):
        """eval 命令不再包含已删除的内部开关"""
        from src.kernel_eval.cli import create_parser
        parser = create_parser()
        # --no-subprocess-isolation, --child-json-output 已删除
        # 注：--skip-install 已恢复，供 ST harness 使用
        for flag in ['--no-subprocess-isolation', '--child-json-output']:
            try:
                parser.parse_args(['eval', flag])
                self.fail(f"已删除的参数 {flag} 不应被 parser 接受")
            except SystemExit:
                pass  # argparse 拒绝未知参数 → 正确行为

    def test_cli_eval_process_no_longer_exists(self):
        """eval-process 命令已删除"""
        from src.kernel_eval.cli import create_parser
        parser = create_parser()
        try:
            parser.parse_args(['eval-process', '--process-id', '0'])
            self.fail("eval-process 不应被 parser 接受")
        except SystemExit:
            pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
