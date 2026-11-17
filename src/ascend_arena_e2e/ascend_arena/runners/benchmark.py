# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


"""BenchmarkRunner — orchestrates build → correctness → profile pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from pathlib import Path

from ascend_arena_e2e.ascend_arena.runners.artifacts import resolve_benchmark_artifact_path
from ascend_arena_e2e.ascend_arena.runners.build import BuildResult, BuildRunner
from ascend_arena_e2e.ascend_arena.runners.cases import (
    load_test_cases,
    select_cases,
    SMOKE_CASES_FILENAME,
    COMPLETE_CASES_FILENAME,
    LEGACY_SMOKE_FILENAME,
    LEGACY_COMPLETE_FILENAME,
    LEGACY_PROFILE_FILENAME,
)
from ascend_arena_e2e.ascend_arena.runners.correctness import CorrectnessResult, CorrectnessRunner
from ascend_arena_e2e.ascend_arena.runners.opp_env import isolated_runtime_opp_env
from ascend_arena_e2e.ascend_arena.runners.profile import ProfileResult, ProfileRunner
from ascend_arena_e2e.ascend_arena.utils.progress import emit_progress

VALID_STAGES = frozenset({"build", "smoke", "complete", "profile"})
ALL_STAGES = frozenset({"smoke", "complete", "profile"})


@dataclass
class BenchmarkResult:
    success: bool
    build_result: BuildResult
    correctness_result: CorrectnessResult
    profile_result: ProfileResult
    smoke_result: CorrectnessResult | None = None
    complete_result: CorrectnessResult | None = None

    @property
    def full_result(self) -> CorrectnessResult | None:
        return self.complete_result


class BenchmarkRunner:
    def __init__(
        self,
        *,
        dependency_root: Path,
        env_script: Path | None = None,
        max_abs_error: float = 1e-3,
        max_rel_error: float = 1e-3,
        correctness_runs: int = 1,
        profile_warmup: int = 10,
        profile_repeat: int = 30,
        profile_inner_iters: int = 1,
        profile_backend: str = "msprof",
        compile_repo: str = "common",
        compile_family: str | None = None,
        compile_op_name: str | None = None,
        compile_jobs: int = 16,
        correctness_use_subprocess: bool = False,
        isolated_build: bool = False,
        profile_requires_correctness: bool = True,
        profile_performance: bool = True,
    ) -> None:
        self.dependency_root = dependency_root
        self.compile_repo = compile_repo
        self.compile_family = compile_family
        self.compile_op_name = compile_op_name
        self.profile_requires_correctness = profile_requires_correctness
        self.profile_performance = profile_performance
        self.build_runner = BuildRunner(
            dependency_root=dependency_root,
            env_script=env_script,
            compile_jobs=compile_jobs,
            isolated_build=isolated_build,
        )
        self.smoke_runner = CorrectnessRunner(
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            correctness_runs=3,
            env_script=env_script,
            use_subprocess=correctness_use_subprocess,
            case_filter="smoke",
            log_name="smoke-correctness.log",
            dump_io=True,
            require_cases=True,
        )
        self.complete_runner = CorrectnessRunner(
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            correctness_runs=correctness_runs,
            env_script=env_script,
            use_subprocess=correctness_use_subprocess,
            case_filter="complete",
            log_name="complete-correctness.log",
            require_cases=True,
        )
        self.profile_correctness_runner = CorrectnessRunner(
            max_abs_error=max_abs_error,
            max_rel_error=max_rel_error,
            correctness_runs=correctness_runs,
            env_script=env_script,
            use_subprocess=correctness_use_subprocess,
            case_filter="profile",
            log_name="profile-correctness.log",
            require_cases=True,
        )
        self.profile_runner = ProfileRunner(
            env_script=env_script,
            warmup=profile_warmup,
            repeat=profile_repeat,
            inner_iters=profile_inner_iters,
            profiler=profile_backend,
            case_filter="profile",
        )

    def _skipped_correctness(
        self,
        build_result: BuildResult,
        reason: str,
        *,
        log_name: str = "profile-correctness.log",
        success: bool = False,
    ) -> CorrectnessResult:
        log_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / log_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(reason + "\n", encoding="utf-8")
        return CorrectnessResult(success, build_result.op_name, log_path, [])

    def _skipped_profile(self, build_result: BuildResult, reason: str, *, success: bool = False) -> ProfileResult:
        log_path = resolve_benchmark_artifact_path(build_result.work_dir.parent / "profile.log")
        output_dir = build_result.work_dir.parent / "profile"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(reason + "\n", encoding="utf-8")
        return ProfileResult(success, build_result.op_name, log_path, output_dir, {}, reason)

    def _has_smoke_cases(self, op_root: Path) -> bool:
        cases_dir = op_root / "cases"
        if (cases_dir / SMOKE_CASES_FILENAME).exists():
            return bool(load_test_cases(cases_dir / SMOKE_CASES_FILENAME))
        if (cases_dir / LEGACY_SMOKE_FILENAME).exists():
            return bool(load_test_cases(cases_dir / LEGACY_SMOKE_FILENAME))
        legacy_profile = cases_dir / LEGACY_PROFILE_FILENAME
        if not legacy_profile.exists():
            return False
        return bool(select_cases(load_test_cases(legacy_profile), "smoke"))

    def _has_complete_cases(self, op_root: Path) -> bool:
        cases_dir = op_root / "cases"
        if (cases_dir / COMPLETE_CASES_FILENAME).exists():
            return bool(load_test_cases(cases_dir / COMPLETE_CASES_FILENAME))
        legacy_complete = cases_dir / LEGACY_COMPLETE_FILENAME
        return legacy_complete.exists() and bool(load_test_cases(legacy_complete))

    def run(
        self,
        *,
        standard_dir: Path,
        op_name: str,
        chip_name: str,
        run_root: Path,
        device_id: int = 0,
        reload_build: bool = False,
        reload_build_root: Path | None = None,
        reload_build_build_path: Path | None = None,
        reload_build_build_out_path: Path | None = None,
        stages: frozenset[str] | None = None,
    ) -> BenchmarkResult:
        run_stages = stages if stages is not None else ALL_STAGES
        build_result: BuildResult | None = None
        result: BenchmarkResult | None = None
        previous_extension = sys.modules.pop("kernel_gen_ops", None)
        try:
            # 中文说明：
            # 主流程固定为 build -> smoke -> complete -> profile。
            # profile 前会额外跑 profile case correctness，避免性能数据来自错误输出。
            # build-only 是特殊调试模式，只有显式 stages={"build"} 时才跳过 correctness/profile。
            build_result = self._run_build_stage(
                standard_dir=standard_dir,
                op_name=op_name,
                chip_name=chip_name,
                run_root=run_root,
                reload_build=reload_build,
                reload_build_root=reload_build_root,
                reload_build_build_path=reload_build_build_path,
                reload_build_build_out_path=reload_build_build_out_path,
            )
            if not build_result.success:
                result = self._build_failed_result(build_result, op_name)
                return result

            # The candidate and baseline stages run in one Python process. Keep the
            # per-build OPP overlay scoped to this benchmark so a candidate vendor
            # cannot become the next stage's build/runtime environment.
            with isolated_runtime_opp_env(
                op_run_root=build_result.work_dir.parent,
                custom_opp_root=build_result.custom_opp_root,
            ):
                smoke_result = self._run_smoke_stage(
                    build_result, standard_dir, device_id, run_stages
                )
                if not smoke_result.success:
                    result = self._stage_failed_result(
                        build_result, smoke_result, op_name, "smoke"
                    )
                    return result

                complete_result = self._run_complete_stage(
                    build_result, standard_dir, device_id, run_stages
                )
                if not complete_result.success:
                    result = self._stage_failed_result(
                        build_result, smoke_result, op_name, "complete",
                        complete_result=complete_result,
                    )
                    return result

                correctness_result, profile_result = self._run_profile_stage(
                    build_result, standard_dir, device_id, run_stages
                )
                if not correctness_result.success:
                    result = self._assemble_result(
                        build_result, smoke_result, complete_result,
                        correctness_result, profile_result, op_name, run_stages,
                    )
                    return result

                result = self._assemble_result(
                    build_result, smoke_result, complete_result,
                    correctness_result, profile_result, op_name, run_stages,
                )
                return result
        finally:
            # kernel_gen_ops is loaded by binding/module.py under a fixed name.
            # Remove this run's extension before the next candidate/baseline run;
            # otherwise Python would reuse the candidate binary from sys.modules.
            sys.modules.pop("kernel_gen_ops", None)
            if previous_extension is not None:
                sys.modules["kernel_gen_ops"] = previous_extension
            if build_result is not None:
                overall_success = result.success if result is not None else False
                self._cleanup(build_result, overall_success=overall_success)

    def _run_build_stage(
        self,
        *,
        standard_dir: Path,
        op_name: str,
        chip_name: str,
        run_root: Path,
        reload_build: bool,
        reload_build_root: Path | None,
        reload_build_build_path: Path | None,
        reload_build_build_out_path: Path | None,
    ) -> BuildResult:
        emit_progress("build_start", op_name=op_name)
        build_result = self.build_runner.run(
            standard_dir=standard_dir,
            chip_name=chip_name,
            run_root=run_root,
            op_name=op_name,
            compile_repo=self.compile_repo,
            compile_family=self.compile_family,
            compile_op_name=self.compile_op_name,
            reload_build=reload_build,
            reload_build_root=reload_build_root,
            reload_build_build_path=reload_build_build_path,
            reload_build_build_out_path=reload_build_build_out_path,
        )
        emit_progress(
            "build_finished",
            op_name=build_result.op_name,
            success=build_result.success,
            log_path=build_result.log_path,
        )
        return build_result

    def _run_smoke_stage(
        self,
        build_result: BuildResult,
        standard_dir: Path,
        device_id: int,
        run_stages: frozenset[str],
    ) -> CorrectnessResult:
        if "smoke" not in run_stages:
            result = self._skipped_correctness(
                build_result, "skipped_not_requested",
                log_name="smoke-correctness.log", success=True,
            )
            emit_progress(
                "smoke_correctness_finished", op_name=build_result.op_name,
                success=True, case_count=0, log_path=result.log_path,
                reason="skipped_not_requested",
            )
            return result

        # 中文说明：
        # 缺少 smoke case 不能形成完整评测结论。
        if not self._has_smoke_cases(build_result.standard_op_root):
            result = self._skipped_correctness(
                build_result, "skipped_no_smoke_cases",
                log_name="smoke-correctness.log", success=False,
            )
            emit_progress(
                "smoke_correctness_finished", op_name=result.op_name,
                success=False, case_count=0, log_path=result.log_path,
                reason="skipped_no_smoke_cases",
            )
            return result

        emit_progress("smoke_correctness_start", op_name=build_result.op_name)
        result = self.smoke_runner.run(
            standard_dir=standard_dir,
            build_result=build_result,
            device_id=device_id,
        )
        emit_progress(
            "smoke_correctness_finished", op_name=result.op_name,
            success=result.success, case_count=len(result.case_results),
            passed_count=sum(item.passed for item in result.case_results),
            log_path=result.log_path,
        )
        return result

    def _run_complete_stage(
        self,
        build_result: BuildResult,
        standard_dir: Path,
        device_id: int,
        run_stages: frozenset[str],
    ) -> CorrectnessResult:
        if "complete" not in run_stages:
            result = self._skipped_correctness(
                build_result, "skipped_not_requested",
                log_name="complete-correctness.log", success=True,
            )
            emit_progress(
                "complete_correctness_finished", op_name=build_result.op_name,
                success=True, case_count=0, log_path=result.log_path,
                reason="skipped_not_requested",
            )
            return result

        # 中文说明：
        # complete case 是主要正确性覆盖，缺失时直接失败。
        if not self._has_complete_cases(build_result.standard_op_root):
            result = self._skipped_correctness(
                build_result, "skipped_no_complete_cases",
                log_name="complete-correctness.log", success=False,
            )
            emit_progress(
                "complete_correctness_finished", op_name=result.op_name,
                success=False, case_count=0, log_path=result.log_path,
                reason="skipped_no_complete_cases",
            )
            return result

        emit_progress("complete_correctness_start", op_name=build_result.op_name)
        result = self.complete_runner.run(
            standard_dir=standard_dir,
            build_result=build_result,
            device_id=device_id,
        )
        emit_progress(
            "complete_correctness_finished", op_name=result.op_name,
            success=result.success, case_count=len(result.case_results),
            passed_count=sum(item.passed for item in result.case_results),
            log_path=result.log_path,
        )
        return result

    def _run_profile_stage(
        self,
        build_result: BuildResult,
        standard_dir: Path,
        device_id: int,
        run_stages: frozenset[str],
    ) -> tuple[CorrectnessResult, ProfileResult]:
        if "profile" not in run_stages:
            correctness_result = self._skipped_correctness(
                build_result, "skipped_not_requested",
                log_name="profile-correctness.log", success=True,
            )
            profile_result = self._skipped_profile(build_result, "skipped_not_requested", success=True)
            emit_progress(
                "correctness_finished", op_name=build_result.op_name,
                success=True, case_count=0, log_path=correctness_result.log_path,
                reason="skipped_not_requested",
            )
            emit_progress(
                "profile_finished", op_name=build_result.op_name,
                success=True, case_count=0, log_path=profile_result.log_path,
                reason="skipped_not_requested",
            )
            return correctness_result, profile_result

        if not self.profile_requires_correctness:
            correctness_result = self._skipped_correctness(
                build_result, "skipped_profile_correctness_not_requested",
                log_name="profile-correctness.log", success=True,
            )
            emit_progress(
                "correctness_finished", op_name=build_result.op_name,
                success=True, case_count=0, log_path=correctness_result.log_path,
                reason="skipped_profile_correctness_not_requested",
            )
            profile_result = self.profile_runner.run(
                standard_dir=standard_dir,
                build_result=build_result,
                correctness_result=correctness_result,
                device_id=device_id,
            )
            emit_progress(
                "profile_finished", op_name=profile_result.op_name,
                success=profile_result.success, case_count=len(profile_result.metrics),
                log_path=profile_result.log_path,
                reason="profile_correctness_not_requested",
            )
            return correctness_result, profile_result

        emit_progress("correctness_start", op_name=build_result.op_name)
        correctness_result = self.profile_correctness_runner.run(
            standard_dir=standard_dir,
            build_result=build_result,
            device_id=device_id,
        )
        emit_progress(
            "correctness_finished", op_name=correctness_result.op_name,
            success=correctness_result.success,
            case_count=len(correctness_result.case_results),
            passed_count=sum(item.passed for item in correctness_result.case_results),
            log_path=correctness_result.log_path,
        )
        if not correctness_result.success:
            profile_result = self._skipped_profile(build_result, "skipped_after_correctness_failed")
            return correctness_result, profile_result

        if not self.profile_performance:
            profile_result = self._skipped_profile(
                build_result, "skipped_performance_disabled", success=True
            )
            emit_progress(
                "profile_finished", op_name=profile_result.op_name,
                success=True, case_count=0, log_path=profile_result.log_path,
                reason="skipped_performance_disabled",
            )
            return correctness_result, profile_result

        emit_progress("profile_start", op_name=build_result.op_name)
        profile_result = self.profile_runner.run(
            standard_dir=standard_dir,
            build_result=build_result,
            correctness_result=correctness_result,
            device_id=device_id,
        )
        return correctness_result, profile_result

    def _cleanup(self, build_result: BuildResult, *, overall_success: bool) -> None:
        result_dir = build_result.work_dir.parent
        run_script = result_dir / "run_correctness_case.py"
        if run_script.exists():
            run_script.unlink()
        if build_result.success:
            for name in ("build", "build_out"):
                path = result_dir / name
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
        if overall_success:
            log_targets = {
                resolve_benchmark_artifact_path(result_dir / "compile.log"),
                resolve_benchmark_artifact_path(result_dir / "smoke-correctness.log"),
                resolve_benchmark_artifact_path(result_dir / "complete-correctness.log"),
                resolve_benchmark_artifact_path(result_dir / "profile-correctness.log"),
                resolve_benchmark_artifact_path(result_dir / "profile.log"),
            }
            for log_file in result_dir.glob("*.log"):
                log_targets.add(log_file)
            for log_file in log_targets:
                if log_file.resolve().is_relative_to(result_dir.resolve()):
                    log_file.unlink(missing_ok=True)

    def _build_failed_result(
        self, build_result: BuildResult, op_name: str
    ) -> BenchmarkResult:
        smoke_result = self._skipped_correctness(
            build_result, "skipped_after_build_failed", log_name="smoke-correctness.log"
        )
        complete_result = self._skipped_correctness(
            build_result, "skipped_after_build_failed", log_name="complete-correctness.log"
        )
        correctness_result = self._skipped_correctness(
            build_result, "skipped_after_build_failed", log_name="profile-correctness.log"
        )
        profile_result = self._skipped_profile(build_result, "skipped_after_build_failed")
        emit_progress(
            "benchmark_finished", op_name=op_name, success=False,
            build_success=False, smoke_success=False,
            complete_success=False, correctness_success=False, profile_success=False,
        )
        return BenchmarkResult(
            success=False, build_result=build_result,
            correctness_result=correctness_result, profile_result=profile_result,
            smoke_result=smoke_result, complete_result=complete_result,
        )

    def _stage_failed_result(
        self,
        build_result: BuildResult,
        smoke_result: CorrectnessResult,
        op_name: str,
        failed_stage: str,
        complete_result: CorrectnessResult | None = None,
    ) -> BenchmarkResult:
        if failed_stage == "smoke":
            complete_result = self._skipped_correctness(
                build_result, "skipped_after_smoke_failed", log_name="complete-correctness.log"
            )
            correctness_result = self._skipped_correctness(
                build_result, "skipped_after_smoke_failed", log_name="profile-correctness.log"
            )
            profile_result = self._skipped_profile(build_result, "skipped_after_smoke_failed")
        else:
            assert complete_result is not None
            correctness_result = self._skipped_correctness(
                build_result, "skipped_after_complete_failed", log_name="profile-correctness.log"
            )
            profile_result = self._skipped_profile(build_result, "skipped_after_complete_failed")

        emit_progress(
            "benchmark_finished", op_name=op_name, success=False,
            build_success=build_result.success,
            smoke_success=smoke_result.success,
            complete_success=complete_result.success if complete_result else False,
            correctness_success=False, profile_success=False,
        )
        return BenchmarkResult(
            success=False, build_result=build_result,
            correctness_result=correctness_result, profile_result=profile_result,
            smoke_result=smoke_result, complete_result=complete_result,
        )

    def _assemble_result(
        self,
        build_result: BuildResult,
        smoke_result: CorrectnessResult,
        complete_result: CorrectnessResult,
        correctness_result: CorrectnessResult,
        profile_result: ProfileResult,
        op_name: str,
        run_stages: frozenset[str],
    ) -> BenchmarkResult:
        build_only = "build" in run_stages and run_stages.isdisjoint({"smoke", "complete", "profile"})
        if build_only:
            success = build_result.success
        else:
            success = (
                build_result.success
                and smoke_result.success
                and complete_result.success
                and correctness_result.success
                and profile_result.success
            )
        emit_progress(
            "benchmark_finished", op_name=op_name, success=success,
            build_success=build_result.success,
            smoke_success=smoke_result.success,
            complete_success=complete_result.success,
            correctness_success=correctness_result.success,
            profile_success=profile_result.success,
        )
        return BenchmarkResult(
            success=success, build_result=build_result,
            correctness_result=correctness_result, profile_result=profile_result,
            smoke_result=smoke_result, complete_result=complete_result,
        )
