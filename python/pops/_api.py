"""Lazy public phase transitions for the final PoPS lifecycle."""
from __future__ import annotations

from typing import Any


def validate(case: Any) -> Any:
    """Validate and freeze a :class:`pops.Case` without importing native code."""
    from pops.codegen._phases import validate as phase

    return phase(case)


def resolve(case: Any, **authorities: Any) -> Any:
    """Resolve all typed authoring authorities into one immutable plan, still in pure Python."""
    from pops.codegen._phases import resolve as phase

    return phase(case, **authorities)


def compile(plan: Any) -> Any:
    """Authenticate the native toolchain and lower one resolved plan."""
    from pops import _bootstrap  # noqa: F401  # intentional native cut line
    from pops.codegen._phases import compile as phase

    return phase(plan)


def bind(artifact: Any, inputs: Any) -> Any:
    """Authenticate bind inputs, create an InstallPlan and materialize one RuntimeInstance."""
    from pops import _bootstrap  # noqa: F401  # intentional native cut line
    from pops.codegen._phases import bind as phase

    return phase(artifact, inputs)


__all__ = ["bind", "compile", "resolve", "validate"]
