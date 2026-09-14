"""Deep, detached freeze of Program-owned authoring tables."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pops._frozen_data import freeze_containers


def freeze_program_tables(program: Any) -> None:
    """Replace every Program-owned container with a detached immutable equivalent."""
    if program._recording:
        raise RuntimeError("Program.freeze() cannot run while an authoring sub-block is active")
    replacements = {
        name: freeze_containers(value)
        for name, value in vars(program).items()
        if name != "_frozen" and isinstance(
            value, (Mapping, list, tuple, set, frozenset))
    }
    for name, value in replacements.items():
        object.__setattr__(program, name, value)
    object.__setattr__(program, "_frozen", True)


__all__ = ["freeze_program_tables"]
