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


def bind(
    artifact: Any,
    *,
    initial_state: Any = None,
    params: Any = None,
    aux: Any = None,
    resources: Any = None,
    initial_values: Any = None,
) -> Any:
    """Bind concrete values and materialize one runtime instance.

    The public boundary accepts the five value/resource families directly. The immutable
    ``BindInputs`` evidence record is an orchestration detail built exactly once here; users never
    import a phase-internal codegen type or choose between two bind spellings.
    """
    from pops import _bootstrap  # noqa: F401  # intentional native cut line
    from pops.codegen._plans import BindInputs
    from pops.codegen._phases import bind as phase

    inputs = BindInputs(
        initial_state={} if initial_state is None else initial_state,
        params={} if params is None else params,
        aux={} if aux is None else aux,
        resources={} if resources is None else resources,
        initial_values={} if initial_values is None else initial_values,
    )
    return phase(artifact, inputs)


__all__ = ["bind", "compile", "resolve", "validate"]
