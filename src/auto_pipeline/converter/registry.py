"""Artifact converter registry."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from auto_pipeline.generator.akg.converter.to_cann import AkgToCannConverter
from auto_pipeline.generator.akg.converter.to_stanford import AkgToStanfordConverter
from auto_pipeline.generator.pypto.converter.to_cann import PyptoToCannConverter
from auto_pipeline.generator.pypto.converter.to_stanford import PyptoToStanfordConverter
from auto_pipeline.converter.base import Converter


ConverterFactory = Callable[[Mapping[str, Any]], Converter]
ConverterKey = tuple[str, str]
_CONVERTER_FACTORIES: Dict[ConverterKey, ConverterFactory] = {}


def register_converter(
    source_generator: str,
    target_benchmark: str,
    factory: ConverterFactory,
    *,
    replace: bool = False,
) -> None:
    key = (_normalize_name(source_generator), _normalize_name(target_benchmark))
    if not replace and key in _CONVERTER_FACTORIES:
        raise ValueError(f"converter already registered: {_display_key(key)}")
    _CONVERTER_FACTORIES[key] = factory


def create_converter(source_generator: str, target_benchmark: str, cfg: Mapping[str, Any]) -> Converter:
    key = (_normalize_name(source_generator), _normalize_name(target_benchmark))
    try:
        factory = _CONVERTER_FACTORIES[key]
    except KeyError as exc:
        available = ", ".join(available_converters())
        raise ValueError(
            "unsupported converter for "
            f"generator={source_generator}, benchmark={target_benchmark}; available: {available}"
        ) from exc
    return factory(cfg)


def available_converters() -> list[str]:
    return sorted(_display_key(key) for key in _CONVERTER_FACTORIES)


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _display_key(key: ConverterKey) -> str:
    return f"{key[0]} -> {key[1]}"


def _env_mapping(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(val) for key, val in value.items()}


def _timeout(cfg: Mapping[str, Any]) -> int:
    return int(cfg.get("timeout_sec") or cfg.get("timeout") or 7200)


def _create_pypto_to_cann(cfg: Mapping[str, Any]) -> PyptoToCannConverter:
    return PyptoToCannConverter(timeout_sec=_timeout(cfg), env=_env_mapping(cfg.get("env")))


def _create_pypto_to_stanford(cfg: Mapping[str, Any]) -> PyptoToStanfordConverter:
    return PyptoToStanfordConverter(timeout_sec=_timeout(cfg), env=_env_mapping(cfg.get("env")))


def _create_akg_to_cann(cfg: Mapping[str, Any]) -> AkgToCannConverter:
    return AkgToCannConverter(timeout_sec=_timeout(cfg), env=_env_mapping(cfg.get("env")))


def _create_akg_to_stanford(cfg: Mapping[str, Any]) -> AkgToStanfordConverter:
    return AkgToStanfordConverter(timeout_sec=_timeout(cfg), env=_env_mapping(cfg.get("env")))


register_converter("pypto", "cann", _create_pypto_to_cann)
register_converter("pypto", "stanford", _create_pypto_to_stanford)
register_converter("akg-agent", "cann", _create_akg_to_cann)
register_converter("akg-agent", "stanford", _create_akg_to_stanford)
