"""Runner and generator registry."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from auto_pipeline.generator.base import Runner
from auto_pipeline.generator.akg import AkgAgent
from auto_pipeline.generator.opencode import OpenCodeAgent
from auto_pipeline.generator.pypto import PyptoOrchestratorAgent
from auto_pipeline.core import Generator


RunnerFactory = Callable[[Mapping[str, Any]], Runner]
_RUNNER_FACTORIES: Dict[str, RunnerFactory] = {}
_GENERATOR_FACTORIES: Dict[str, Callable[[Mapping[str, Any]], Generator]] = {}


def register_runner(name: str, factory: RunnerFactory, *, replace: bool = False) -> None:
    key = _normalize_name(name)
    if not replace and key in _RUNNER_FACTORIES:
        raise ValueError(f"runner already registered: {key}")
    _RUNNER_FACTORIES[key] = factory


def register_generator(name: str, factory: Callable[[Mapping[str, Any]], Generator], *, replace: bool = False) -> None:
    key = _normalize_name(name)
    if not replace and key in _GENERATOR_FACTORIES:
        raise ValueError(f"generator already registered: {key}")
    _GENERATOR_FACTORIES[key] = factory


def create_runner(runner_type: str, cfg: Mapping[str, Any]) -> Runner:
    key = _normalize_name(runner_type)
    try:
        factory = _RUNNER_FACTORIES[key]
    except KeyError as exc:
        available = ", ".join(available_runners())
        raise ValueError(f"unsupported runner.type: {runner_type}; available: {available}") from exc
    return factory(cfg)


def create_generator(generator_type: str, cfg: Mapping[str, Any]) -> Generator:
    key = _normalize_name(generator_type)
    try:
        factory = _GENERATOR_FACTORIES[key]
    except KeyError as exc:
        available = ", ".join(available_generators())
        raise ValueError(f"unsupported generator.type: {generator_type}; available: {available}") from exc
    return factory(cfg)


def available_runners() -> list:
    return sorted(_RUNNER_FACTORIES)


def available_generators() -> list:
    return sorted(_GENERATOR_FACTORIES)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _string_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _create_opencode(cfg: Mapping[str, Any]) -> OpenCodeAgent:
    return OpenCodeAgent(
        opencode_bin=str(cfg.get("bin") or cfg.get("opencode_bin") or "opencode"),
        skill=str(cfg.get("skill") or cfg.get("agent") or cfg.get("name") or ""),
        model=str(cfg.get("model") or ""),
        output_format=str(cfg.get("output_format") or "default"),
        dangerously_skip_permissions=bool(cfg.get("dangerously_skip_permissions", True)),
        extra_args=_string_list(cfg.get("extra_args")),
    )


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_bool(value: object, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _string_mapping(value: object) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("env must be a mapping")
    return {str(key): str(val) for key, val in value.items()}


def _int_or_default(value: object, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _create_pypto(cfg: Mapping[str, Any]) -> PyptoOrchestratorAgent:
    repo_root = cfg.get("repo_root") or cfg.get("pypto_repo_root")
    if not repo_root:
        raise ValueError("pypto generator requires repo_root/pypto_repo_root")
    return PyptoOrchestratorAgent(
        pypto_repo_root=repo_root,
        workdir_root=str(cfg.get("workdir_root") or "custom"),
        opencode_bin=str(cfg.get("bin") or cfg.get("opencode_bin") or "opencode"),
        opencode_model=str(cfg.get("model") or cfg.get("opencode_model") or ""),
        agent=str(cfg.get("skill") or cfg.get("agent") or cfg.get("name") or "pypto-op-orchestrator"),
        output_format=str(cfg.get("output_format") or "default"),
        device_id=_optional_int(cfg.get("device_id")),
        device_mode=str(cfg.get("device_mode") or "normal"),
        skip_if_done=_optional_bool(cfg.get("skip_if_done"), True),
        worktree_root=cfg.get("worktree_root") or cfg.get("isolated_worktree_root"),
        worktree_ref=str(cfg.get("worktree_ref") or "HEAD"),
        extra_env=_string_mapping(cfg.get("env")),
    )


def _create_akg(cfg: Mapping[str, Any]) -> AkgAgent:
    repo_root = cfg.get("repo_root") or cfg.get("akg_repo_root")
    if not repo_root:
        raise ValueError("akg-agent generator requires repo_root/akg_repo_root")
    if cfg.get("dsl") is not None:
        raise ValueError("generator.dsl is obsolete; use generator.codegen_target")
    return AkgAgent(
        repo_root=repo_root,
        config_path=cfg.get("config_path"),
        device_id=_int_or_default(cfg.get("device_id"), 0),
        backend=str(cfg.get("backend") or "ascend"),
        arch=str(cfg.get("arch") or "ascend910b4"),
        framework=str(cfg.get("framework") or "torch"),
        codegen_target=str(cfg.get("codegen_target") or "triton_ascend"),
        workflow=str(cfg.get("workflow") or "kernelgen_only_workflow"),
        verify_timeout=_optional_int(cfg.get("verify_timeout")),
        env=_string_mapping(cfg.get("env")),
    )


register_runner("opencode", _create_opencode)
register_generator("akg-agent", _create_akg)
register_generator("pypto", _create_pypto)
