"""Scene/task metadata validation and instruction generation."""

from typing import Any

__all__ = [
    "SceneDefinitionError",
    "generate_instruction",
    "load_scene_task",
    "validate_scene_task",
]


def __getattr__(name: str) -> Any:
    """Load the implementation lazily so ``python -m`` has no runpy warning."""
    if name in __all__:
        from . import generator

        return getattr(generator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
