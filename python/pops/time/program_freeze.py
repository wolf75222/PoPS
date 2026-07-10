"""Deep, detached freeze of Program-owned authoring tables."""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


def freeze_program_tables(program: Any) -> None:
    """Replace every Program-owned container with a detached immutable equivalent."""
    if program._recording:
        raise RuntimeError("Program.freeze() cannot run while an authoring sub-block is active")
    replacements = {
        name: _immutable_copy(value)
        for name, value in vars(program).items()
        if name != "_frozen" and isinstance(
            value, (Mapping, list, tuple, set, frozenset))
    }
    for name, value in replacements.items():
        object.__setattr__(program, name, value)
    object.__setattr__(program, "_frozen", True)


def _immutable_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({
            _immutable_copy(key): _immutable_copy(item)
            for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_copy(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_copy(item) for item in value)
    return value


__all__ = ["freeze_program_tables"]
