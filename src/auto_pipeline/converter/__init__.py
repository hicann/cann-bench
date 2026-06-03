"""Artifact converters."""

__all__ = ["available_converters", "create_converter", "register_converter"]


def __getattr__(name: str):
    if name in __all__:
        from auto_pipeline.converter import registry

        return getattr(registry, name)
    raise AttributeError(name)
