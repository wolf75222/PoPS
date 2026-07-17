"""Owner-aware conversion from a :class:`ParamHandle` to a symbolic value.

The public data model keeps declaration identity and symbolic algebra separate:
``ParamHandle`` has Boolean equality and is hashable, while this module creates
the distinct ``Expr`` node read by physics formulas.  Conversion always goes
through the declaring ``ParamRegistry``; a handle alone never guesses a value or
silently changes storage class.
"""
from __future__ import annotations

from typing import Any

from .expr import Const, Expr
from .values import RuntimeParamRef


def parameter_value(registry: Any, parameter: Any) -> Expr:
    """Return the symbolic read of an authenticated parameter handle.

    ``ConstParam`` values are inlined, ``RuntimeParam`` and bind-derived values
    become runtime slots, and compile-derived values are evaluated from their
    explicitly declared dependencies.  Later derived phases are refused until a
    runtime evaluator for that phase exists; they are never approximated by a
    declaration default.
    """
    from pops.params import (
        ParamKind,
        ParamPhase,
        ParamStorage,
    )

    declaration = registry.declaration(parameter)
    handle = registry.handle(parameter)
    if declaration.kind is ParamKind.Const:
        return Const(declaration.value, handle=handle)
    if declaration.kind is ParamKind.Runtime:
        seed = declaration.default if declaration.has_default else _neutral_value(
            declaration.dtype
        )
        return RuntimeParamRef(
            handle.local_id,
            seed,
            handle=handle,
            dtype=declaration.dtype,
        )
    if declaration.phase is ParamPhase.Compile:
        if declaration.storage is not ParamStorage.Inline:
            raise ValueError(
                "compile-phase DerivedParam %r must use ParamStorage.Inline"
                % declaration.name
            )
        value = declaration.resolved_value
        return Const(value, handle=handle)
    if declaration.phase is ParamPhase.Bind:
        if declaration.storage is not ParamStorage.DerivedCache:
            raise ValueError(
                "bind-phase DerivedParam %r must use ParamStorage.DerivedCache"
                % declaration.name
            )
        return RuntimeParamRef(
            handle.local_id,
            _neutral_value(declaration.dtype),
            handle=handle,
            dtype=declaration.dtype,
        )
    raise NotImplementedError(
        "DerivedParam %r uses phase=%s; the current native runtime has no %s "
        "parameter evaluator. Use Compile/Inline or Bind/DerivedCache."
        % (declaration.name, declaration.phase.value, declaration.phase.value)
    )


def _evaluate_compile_derived(
    registry: Any,
    declaration: Any,
    *,
    stack: tuple[str, ...],
) -> Any:
    """Evaluate one compile-derived expression from authenticated dependencies."""
    from pops.params import ParamKind, ParamPhase, ParamStorage

    if declaration.name in stack:
        cycle = " -> ".join((*stack, declaration.name))
        raise ValueError("DerivedParam dependency cycle: %s" % cycle)
    env = {}
    next_stack = (*stack, declaration.name)
    for dependency in declaration.depends_on:
        dep = registry.declaration(dependency)
        if dep.kind is ParamKind.Const:
            value = dep.value
        elif dep.kind is ParamKind.Derived:
            if dep.phase is not ParamPhase.Compile or dep.storage is not ParamStorage.Inline:
                raise ValueError(
                    "compile DerivedParam %r depends on non-compile parameter %s"
                    % (declaration.name, dependency.qualified_id)
                )
            value = _evaluate_compile_derived(registry, dep, stack=next_stack)
        else:
            raise ValueError(
                "compile DerivedParam %r depends on runtime parameter %s"
                % (declaration.name, dependency.qualified_id)
            )
        # ValueExpr uses the qualified identity.  The local-id entry supports
        # legacy Var leaves only when the same explicit depends_on handle has
        # authenticated that name; no free-name lookup occurs.
        env[dependency.qualified_id] = value
        env[dependency.local_id] = value
    try:
        return declaration.expression.eval(env)
    except (KeyError, TypeError, ValueError) as exc:
        raise type(exc)(
            "cannot evaluate compile DerivedParam %r from its declared dependencies: %s"
            % (declaration.name, exc)
        ) from None


def _neutral_value(dtype: Any) -> Any:
    """Neutral authoring seed; BindSchema remains the runtime value authority."""
    from pops.math import Bool, Integer

    if dtype is Bool:
        return False
    if dtype is Integer:
        return 0
    return 0.0


def _validate_derived_result(declaration: Any, value: Any) -> None:
    """Apply the declaration's dtype/unit/domain contract to an evaluated value."""
    from pops.params._declaration_data import value_data

    value_data(
        value,
        dtype=declaration.dtype,
        unit=declaration.unit,
        where="compile DerivedParam %r result" % declaration.name,
    )
    if declaration.domain is not None:
        try:
            declaration.domain.check(value, who=declaration.name)
        except (TypeError, ValueError):
            raise ValueError(
                "compile DerivedParam %r result %r is outside domain %s"
                % (declaration.name, value, declaration.domain.to_data())
            ) from None


__all__ = ["parameter_value"]
