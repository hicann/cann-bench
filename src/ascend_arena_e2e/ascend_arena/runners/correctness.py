# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import shlex
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from ascend_arena_e2e.ascend_arena.evaluator.compare import OutputCompareResult, TensorCompareResult, compare_tensors
from ascend_arena_e2e.ascend_arena.evaluator.policy import PrecisionPolicy, load_precision_policy
from ascend_arena_e2e.ascend_arena.evaluator.precision import PrecisionSummary, judge_precision
from ascend_arena_e2e.ascend_arena.runners.artifacts import resolve_benchmark_artifact_path
from ascend_arena_e2e.ascend_arena.runners.build import BuildResult
from ascend_arena_e2e.ascend_arena.runners.cases import load_test_cases, select_cases
from ascend_arena_e2e.ascend_arena.runners.opp_env import isolated_runtime_opp_env, prepare_runtime_opp_env
from ascend_arena_e2e.ascend_arena.runners._proc_utils import DEFAULT_CORRECTNESS_TIMEOUT_SEC, run_shell_with_timeout


def compare_outputs_for_case(
    row: dict[str, Any],
    expected: Any,
    actual: Any,
    *,
    max_abs_error: float,
    max_rel_error: float,
    native: Any | None = None,
    policy: PrecisionPolicy | None = None,
) -> TensorCompareResult:
    """Compare one case using only the reviewed operator-level precision policy."""

    del row
    resolved_policy = policy or PrecisionPolicy()
    return compare_tensors(
        expected,
        actual,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        native=native,
        ignore_output_indices=set(resolved_policy.ignore_output_indices),
        optional_output_indices=set(resolved_policy.optional_output_indices),
        precision_thresholds=resolved_policy.thresholds,
        allowed_output_dtypes=resolved_policy.output_dtypes,
        output_precision_standards=resolved_policy.output_precision_standards,
    )


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    compare: TensorCompareResult | None = None
    precision: PrecisionSummary | None = None
    seeds: list[int] = field(default_factory=list)
    passed_runs: int = 0
    total_runs: int = 0
    pass_rate: float = 0.0


@dataclass
class CorrectnessResult:
    success: bool
    op_name: str
    log_path: Path
    case_results: list[CaseResult]
    correctness_runs: int = 1


@dataclass
class PreparedCaseEvaluation:
    init_inputs_cpu: Any
    inputs_cpu: Any
    init_inputs_device: Any
    inputs_device: Any
    golden: Any
    native: Any


class CorrectnessRunner:
    """算子正确性验证器。

    对每个 test case，分别在 CPU（reference.Model）和 NPU（binding.ModelNew）
    上运行相同输入，通过 compare_tensors + judge_precision 比较输出精度。

    执行模式：
      - use_subprocess=False（默认）：在当前进程中运行，复用已加载的 NPU 上下文。
      - use_subprocess=True：在子进程中运行，隔离 NPU 内存状态（避免 CUDA OOM 污染）。

    correctness_runs > 1 时对同一 case 多次运行（不同随机 seed），
    全部通过才算该 case 通过，用于检测数值稳定性。
    """

    MAX_FULL_TENSOR_LOG_ELEMENTS = 1024
    TENSOR_LOG_SAMPLE_ELEMENTS = 16

    def __init__(
        self,
        max_abs_error: float = 1e-3,
        max_rel_error: float = 1e-3,
        *,
        correctness_runs: int = 1,
        env_script: Path | None = None,
        use_subprocess: bool = False,
        case_filter: str = "all",
        log_name: str = "correctness.log",
        dump_io: bool = False,
        require_cases: bool = False,
        min_cases: int | None = None,
        max_cases: int | None = None,
    ) -> None:
        if correctness_runs < 1:
            raise ValueError("correctness_runs must be >= 1")
        self.max_abs_error = max_abs_error
        self.max_rel_error = max_rel_error
        self.correctness_runs = correctness_runs
        self.env_script = env_script
        self.use_subprocess = use_subprocess
        self.case_filter = case_filter
        self.log_name = log_name
        self.dump_io = dump_io
        self.require_cases = require_cases
        self.min_cases = min_cases
        self.max_cases = max_cases

    def _seed_schedule(self) -> list[int]:
        return [index * 100 for index in range(self.correctness_runs)]

    def _payload_filename(self) -> str:
        stem = Path(self.log_name).stem
        return f"{stem}_result.json"

    def _case_count_error(self, case_count: int) -> str | None:
        if self.require_cases and case_count == 0:
            return f"no_cases case_filter={self.case_filter}"
        if self.min_cases is not None and case_count < self.min_cases:
            return f"too_few_cases case_filter={self.case_filter} count={case_count} min={self.min_cases}"
        if self.max_cases is not None and case_count > self.max_cases:
            return f"too_many_cases case_filter={self.case_filter} count={case_count} max={self.max_cases}"
        return None

    def _set_global_seed(self, seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        manual_seed_all = getattr(getattr(torch, "npu", None), "manual_seed_all", None)
        if callable(manual_seed_all):
            manual_seed_all(seed)

    def run(self, standard_dir: Path, build_result: BuildResult, device_id: int = 0) -> CorrectnessResult:
        if self.use_subprocess:
            return self._run_in_subprocess(standard_dir=standard_dir, build_result=build_result, device_id=device_id)
        return self._run_in_process(standard_dir=standard_dir, build_result=build_result, device_id=device_id)

    def _run_in_process(self, standard_dir: Path, build_result: BuildResult, device_id: int = 0) -> CorrectnessResult:
        log_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / self.log_name)
        payload_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / self._payload_filename())
        if not build_result.success or build_result.extension_path is None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("build_failed\n", encoding="utf-8")
            result = CorrectnessResult(False, build_result.op_name, log_path, [], self.correctness_runs)
            self._write_payload(payload_path, result, error="build_failed")
            return result

        op_root = build_result.standard_op_root

        sys.path.insert(0, str(build_result.build_dir))
        try:
            with isolated_runtime_opp_env(
                op_run_root=build_result.work_dir.parent,
                custom_opp_root=build_result.custom_opp_root,
            ):
                torch_npu = _load_torch_npu()
                reference_module = self._load_module("reference_module", op_root / "reference.py")
                binding_module = self._load_module("binding_module", op_root / "binding" / "module.py")
                inputs_module = self._load_module("prepare_inputs_module", op_root / "cases" / "prepare_inputs.py")
                self._validate_contract(reference_module, binding_module, inputs_module)
                policy = load_precision_policy(op_root / "cases" / "proto.yaml")
                rows = select_cases(self._load_cases_for_filter(op_root / "cases"), self.case_filter)
                count_error = self._case_count_error(len(rows))
                if count_error is not None:
                    log_path.write_text(count_error + "\n", encoding="utf-8")
                    result = CorrectnessResult(False, build_result.op_name, log_path, [], self.correctness_runs)
                    self._write_payload(payload_path, result, error=count_error)
                    return result
                torch.npu.set_device(device_id)
                device = torch.device(f"npu:{device_id}")

                results: list[CaseResult] = []
                lines: list[str] = []
                for row in rows:
                    case_id = str(row["case_id"])
                    seeds = self._seed_schedule()
                    passed_runs = 0
                    last_compare: TensorCompareResult | None = None
                    last_precision: PrecisionSummary | None = None
                    first_failed_compare: TensorCompareResult | None = None
                    first_failed_precision: PrecisionSummary | None = None
                    for seed in seeds:
                        prepared = self.prepare_case_evaluation(
                            row=row,
                            reference_module=reference_module,
                            inputs_module=inputs_module,
                            device=device,
                            seed=seed,
                            policy=policy,
                        )

                        self._set_global_seed(seed)
                        model_new = binding_module.ModelNew(*prepared.init_inputs_device).to(device)
                        self._set_global_seed(seed)
                        actual = model_new(*self._clone_inputs(prepared.inputs_device))
                        torch_npu.npu.synchronize()
                        compare = compare_outputs_for_case(
                            row,
                            prepared.golden,
                            actual,
                            max_abs_error=self.max_abs_error,
                            max_rel_error=self.max_rel_error,
                            native=prepared.native,
                            policy=policy,
                        )
                        precision = judge_precision(
                            compare,
                            max_abs_error=self.max_abs_error,
                            max_rel_error=self.max_rel_error,
                        )
                        last_compare = compare
                        last_precision = precision
                        if precision.passed:
                            passed_runs += 1
                        elif first_failed_compare is None:
                            first_failed_compare = compare
                            first_failed_precision = precision
                        if self.dump_io:
                            lines.append(
                                self._format_io_detail_line(
                                    case_id=case_id,
                                    seed=seed,
                                    init_inputs=prepared.init_inputs_cpu,
                                    inputs=prepared.inputs_cpu,
                                    golden=prepared.golden,
                                    actual=actual,
                                    passed=precision.passed,
                                )
                            )
                        lines.append(
                            f"case_id={case_id} seed={seed} passed={precision.passed} "
                            f"max_abs_diff={compare.max_abs_diff:.6f} max_rel_diff={compare.max_rel_diff:.6f} "
                            f"MARE={compare.mare_ratio:.6f} MERE={compare.mere_ratio:.6f} "
                            f"RMSE={compare.rmse_ratio:.6f} "
                            f"small_value_errors={compare.small_value_error_count}/{compare.small_value_count}"
                        )
                    total_runs = len(seeds)
                    pass_rate = float(passed_runs / total_runs) if total_runs else 0.0
                    case_result = CaseResult(
                        case_id=case_id,
                        passed=passed_runs == total_runs,
                        seeds=seeds,
                        passed_runs=passed_runs,
                        total_runs=total_runs,
                        pass_rate=pass_rate,
                        compare=first_failed_compare or last_compare,
                        precision=first_failed_precision or last_precision,
                    )
                    results.append(case_result)
                    lines.append(
                        f"case_id={case_id} passed_runs={passed_runs} total_runs={total_runs} pass_rate={pass_rate:.6f}"
                    )
                success = all(item.passed for item in results)
                log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                result = CorrectnessResult(success, build_result.op_name, log_path, results, self.correctness_runs)
                self._write_payload(payload_path, result)
                return result
        except Exception as exc:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            result = CorrectnessResult(False, build_result.op_name, log_path, [], self.correctness_runs)
            self._write_payload(payload_path, result, error=f"{type(exc).__name__}: {exc}")
            return result
        finally:
            if str(build_result.build_dir) in sys.path:
                sys.path.remove(str(build_result.build_dir))

    def _run_in_subprocess(self, standard_dir: Path, build_result: BuildResult, device_id: int = 0) -> CorrectnessResult:
        log_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / self.log_name)
        if not build_result.success or build_result.extension_path is None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("build_failed\n", encoding="utf-8")
            result = CorrectnessResult(False, build_result.op_name, log_path, [], self.correctness_runs)
            payload_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / self._payload_filename())
            self._write_payload(payload_path, result, error="build_failed")
            return result

        payload_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / self._payload_filename())
        payload_path.unlink(missing_ok=True)
        script = self._write_subprocess_script(
            standard_dir=standard_dir,
            build_result=build_result,
            device_id=device_id,
            payload_path=payload_path,
        )
        runtime_env = prepare_runtime_opp_env(
            op_run_root=build_result.work_dir.parent,
            custom_opp_root=build_result.custom_opp_root,
            env=dict(os.environ),
        )
        command_parts = []
        if self.env_script is not None:
            command_parts.append(f"source {shlex.quote(str(self.env_script))}")
        for key in ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH"):
            value = runtime_env.get(key)
            if value:
                command_parts.append(f"export {key}={shlex.quote(value)}")
        command_parts.append(f"{sys.executable} {shlex.quote(str(script))}")
        proc = run_shell_with_timeout(" && ".join(command_parts), timeout=DEFAULT_CORRECTNESS_TIMEOUT_SEC)
        if payload_path.exists():
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                result = self._result_from_payload(payload, log_path)
                if proc.returncode != 0:
                    log_path.write_text(
                        (proc.stdout or "") + "\n" + (proc.stderr or "")
                        + f"\n[WARNING] process exited with returncode={proc.returncode} but payload indicates success={result.success}\n",
                        encoding="utf-8",
                    )
                return result
            except (json.JSONDecodeError, KeyError):
                pass
        log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
        return CorrectnessResult(False, build_result.op_name, log_path, [], self.correctness_runs)

    def _write_subprocess_script(
        self,
        *,
        standard_dir: Path,
        build_result: BuildResult,
        device_id: int,
        payload_path: Path,
    ) -> Path:
        script = build_result.work_dir.parent / "run_correctness_case.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        # Resolves to the ascend_arena package root
        project_root = Path(__file__).resolve().parents[1]
        script.write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {str(project_root)!r})\n"
            "_threads = os.environ.get('BENCHMARK_TORCH_NUM_THREADS')\n"
            "if _threads:\n"
            "    try:\n"
            "        import torch as _torch\n"
            "        _torch.set_num_threads(max(1, int(_threads)))\n"
            "    except Exception:\n"
            "        pass\n"
            "from ascend_arena_e2e.ascend_arena.runners.build import BuildResult\n"
            "from ascend_arena_e2e.ascend_arena.runners.correctness import CorrectnessRunner\n"
            f"build_result = BuildResult(success={build_result.success!r}, op_name={build_result.op_name!r}, "
            f"standard_op_root=Path({str(build_result.standard_op_root)!r}), work_dir=Path({str(build_result.work_dir)!r}), "
            f"build_dir=Path({str(build_result.build_dir)!r}), log_path=Path({str(build_result.log_path)!r}), "
            f"extension_path=Path({str(build_result.extension_path)!r}), "
            f"custom_opp_root={f'Path({str(build_result.custom_opp_root)!r})' if build_result.custom_opp_root is not None else 'None'}, "
            f"message={build_result.message!r})\n"
            f"runner = CorrectnessRunner(max_abs_error={self.max_abs_error!r}, max_rel_error={self.max_rel_error!r}, "
            f"correctness_runs={self.correctness_runs!r}, use_subprocess=False, case_filter={self.case_filter!r}, "
            f"log_name={self.log_name!r}, dump_io={self.dump_io!r}, require_cases={self.require_cases!r}, "
            f"min_cases={self.min_cases!r}, max_cases={self.max_cases!r})\n"
            f"result = runner.run(standard_dir=Path({str(standard_dir)!r}), build_result=build_result, device_id={device_id})\n"
            "payload = runner._payload_from_result(result)\n"
            f"Path({str(payload_path)!r}).write_text(json.dumps(payload), encoding='utf-8')\n",
            encoding="utf-8",
        )
        script.chmod(0o700)
        return script

    def _payload_from_result(self, result: CorrectnessResult) -> dict[str, Any]:
        return {
            "success": result.success,
            "op_name": result.op_name,
            "correctness_runs": result.correctness_runs,
            "case_results": [
                {
                    "case_id": item.case_id,
                    "passed": item.passed,
                    "compare": asdict(item.compare) if item.compare is not None else None,
                    "precision": asdict(item.precision) if item.precision is not None else None,
                    "seeds": item.seeds,
                    "passed_runs": item.passed_runs,
                    "total_runs": item.total_runs,
                    "pass_rate": item.pass_rate,
                }
                for item in result.case_results
            ],
        }

    def _write_payload(
        self,
        payload_path: Path,
        result: CorrectnessResult,
        *,
        error: str | None = None,
    ) -> None:
        payload = self._payload_from_result(result)
        if error is not None:
            payload["error"] = error
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _result_from_payload(self, payload: dict[str, Any], log_path: Path) -> CorrectnessResult:
        case_results = [
            CaseResult(
                case_id=str(item["case_id"]),
                passed=bool(item["passed"]),
                compare=self._compare_result_from_payload(item.get("compare")),
                precision=PrecisionSummary(**item["precision"]) if item.get("precision") else None,
                seeds=[int(seed) for seed in item.get("seeds", [])],
                passed_runs=int(item.get("passed_runs", 1 if item.get("passed") else 0)),
                total_runs=int(item.get("total_runs", len(item.get("seeds", [])) or 1)),
                pass_rate=float(
                    item.get(
                        "pass_rate",
                        (int(item.get("passed_runs", 1 if item.get("passed") else 0)) / int(item.get("total_runs", len(item.get("seeds", [])) or 1))),
                    )
                ),
            )
            for item in payload.get("case_results", [])
        ]
        return CorrectnessResult(
            bool(payload.get("success")),
            str(payload.get("op_name", "")),
            log_path,
            case_results,
            int(payload.get("correctness_runs", 1)),
        )

    def _compare_result_from_payload(self, value: Any) -> TensorCompareResult | None:
        if not isinstance(value, dict):
            return None
        payload = dict(value)
        payload["output_results"] = [
            OutputCompareResult(**item)
            for item in payload.get("output_results", [])
            if isinstance(item, dict)
        ]
        return TensorCompareResult(**payload)

    def _load_module(self, module_name: str, path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load module from {path}")
        module = importlib.util.module_from_spec(spec)
        standard_dir = str(path.parent.parent)
        added_standard_dir = False
        if standard_dir not in sys.path:
            sys.path.insert(0, standard_dir)
            added_standard_dir = True
        try:
            spec.loader.exec_module(module)
        finally:
            if added_standard_dir and standard_dir in sys.path:
                sys.path.remove(standard_dir)
        return module

    def _validate_contract(self, reference_module: Any, binding_module: Any, inputs_module: Any) -> None:
        """验证算子 benchmark 模块遵循标准接口契约。

        每个算子目录必须提供：
          - reference.py::Model       CPU 参考实现（torch 高阶 API，产生 golden 输出）
          - binding/module.py::ModelNew  NPU 候选实现（调用编译好的 kernel_gen_ops）
          - cases/prepare_inputs.py::get_inputs / get_init_inputs  输入生成函数
        """
        required = [
            (reference_module, "Model", "reference.Model"),
            (binding_module, "ModelNew", "binding.ModelNew"),
            (inputs_module, "get_inputs", "prepare_inputs.get_inputs"),
            (inputs_module, "get_init_inputs", "prepare_inputs.get_init_inputs"),
        ]
        missing = [label for module, attr, label in required if not hasattr(module, attr)]
        if missing:
            raise AttributeError(f"missing required symbols: {', '.join(missing)}")

    def _load_cases(self, path: Path) -> list[dict[str, Any]]:
        return load_test_cases(path)

    def _load_cases_for_filter(self, cases_dir: Path) -> list[dict[str, Any]]:
        from ascend_arena_e2e.ascend_arena.runners.cases import resolve_cases_path
        # resolve_cases_path already tries current + legacy filenames in order.
        path = resolve_cases_path(cases_dir, self.case_filter)
        if not path.exists():
            return []
        return load_test_cases(path)

    def prepare_case_evaluation(
        self,
        *,
        row: dict[str, Any],
        reference_module: Any,
        inputs_module: Any,
        device: str | torch.device,
        seed: int,
        policy: PrecisionPolicy | None = None,
    ) -> PreparedCaseEvaluation:
        self._set_global_seed(seed)
        init_inputs_cpu = inputs_module.get_init_inputs(row, device="cpu")
        self._set_global_seed(seed)
        inputs_cpu = inputs_module.get_inputs(row, device="cpu")
        init_inputs_device = self._to_device(self._clone_inputs(init_inputs_cpu), device)
        inputs_device = self._to_device(self._clone_inputs(inputs_cpu), device)
        reference_inputs_cpu = inputs_cpu

        # A small number of backward operators consume statistics/cache tensors
        # produced by their forward operator.  Such benchmarks may opt in to a
        # device-side precompute hook.  It runs before the candidate model is
        # created (and therefore outside profile timing); all existing
        # benchmarks keep the ordinary CPU-input copy path.
        prepare_device_inputs = getattr(inputs_module, "prepare_device_inputs", None)
        if callable(prepare_device_inputs):
            inputs_device = prepare_device_inputs(
                row,
                self._clone_inputs(inputs_cpu),
                device=device,
            )
            # Backward benchmarks that opt into this hook consume forward
            # products (for example mean/rstd or CTC alpha) as part of their
            # public input contract.  The oracle must see the same products as
            # the candidate; otherwise the comparison measures the forward
            # implementation's device rounding instead of backward accuracy.
            reference_inputs_cpu = self._to_cpu(self._clone_inputs(inputs_device))

        resolved_policy = policy or PrecisionPolicy()
        if resolved_policy.oracle_input_mode == "native":
            oracle_init_inputs = self._clone_inputs(init_inputs_cpu)
            oracle_inputs = self._clone_inputs(reference_inputs_cpu)
        else:
            oracle_init_inputs = self._to_fp64_cpu(self._clone_inputs(init_inputs_cpu))
            oracle_inputs = self._to_fp64_cpu(self._clone_inputs(reference_inputs_cpu))
        oracle_model_type = getattr(reference_module, "OracleModel", reference_module.Model)
        bench_model_type = getattr(reference_module, "BenchModel", reference_module.Model)
        self._set_global_seed(seed)
        golden_model = oracle_model_type(*oracle_init_inputs).to("cpu")
        self._set_global_seed(seed)
        golden = golden_model(*oracle_inputs)
        self._set_global_seed(seed)
        native_model = bench_model_type(*self._clone_inputs(init_inputs_cpu)).to("cpu")
        self._set_global_seed(seed)
        native = native_model(*self._clone_inputs(reference_inputs_cpu))
        return PreparedCaseEvaluation(
            init_inputs_cpu=init_inputs_cpu,
            inputs_cpu=inputs_cpu,
            init_inputs_device=init_inputs_device,
            inputs_device=inputs_device,
            golden=golden,
            native=native,
        )

    def _to_device(self, value: Any, device: str | torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, (list, tuple)):
            return type(value)(self._to_device(item, device) for item in value)
        if isinstance(value, dict):
            return {key: self._to_device(item, device) for key, item in value.items()}
        return value

    def _to_cpu(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().to("cpu")
        if isinstance(value, tuple):
            return tuple(self._to_cpu(item) for item in value)
        if isinstance(value, list):
            return [self._to_cpu(item) for item in value]
        if isinstance(value, dict):
            return {key: self._to_cpu(item) for key, item in value.items()}
        return value

    def _to_fp64_cpu(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to("cpu")
            return tensor.double() if tensor.is_floating_point() else tensor
        if isinstance(value, tuple):
            return tuple(self._to_fp64_cpu(item) for item in value)
        if isinstance(value, list):
            return [self._to_fp64_cpu(item) for item in value]
        if isinstance(value, dict):
            return {key: self._to_fp64_cpu(item) for key, item in value.items()}
        return value

    def _clone_inputs(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, (list, tuple)):
            return type(value)(self._clone_inputs(item) for item in value)
        if isinstance(value, dict):
            return {key: self._clone_inputs(item) for key, item in value.items()}
        return value

    def _format_io_detail_line(
        self,
        *,
        case_id: str,
        seed: int,
        init_inputs: Any,
        inputs: Any,
        golden: Any,
        actual: Any,
        passed: bool,
    ) -> str:
        return json.dumps(
            {
                "kind": "smoke_io",
                "case_id": case_id,
                "seed": seed,
                "passed": passed,
                "init_inputs": self._serialize_value(init_inputs),
                "inputs": self._serialize_value(inputs),
                "golden": self._serialize_value(golden),
                "actual": self._serialize_value(actual),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().to("cpu")
            numel = int(tensor.numel())
            result = {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            if numel <= self.MAX_FULL_TENSOR_LOG_ELEMENTS:
                return {**result, "value": tensor.tolist()}
            flat = tensor.flatten()
            sample_count = min(self.TENSOR_LOG_SAMPLE_ELEMENTS, numel)
            return {
                **result,
                "numel": numel,
                "truncated": True,
                "sample_head": flat[:sample_count].tolist(),
                "sample_tail": flat[-sample_count:].tolist(),
            }
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._serialize_value(item) for key, item in value.items()}
        return value


def _load_torch_npu():
    return importlib.import_module("torch_npu")
