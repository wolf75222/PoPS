"""Interaction authoring reusing captured operators and qualified rate projections."""
from __future__ import annotations

from collections.abc import Mapping


def declare_interaction(model, name, *, outputs, preserves=None, dissipates=None):
    from pops._ir.expr import Expr, _wrap
    from pops._ir.quantity import QuantityRef
    from pops._ir.visitors import _children
    if not isinstance(outputs, Mapping) or not outputs:
        raise TypeError("interaction requires outputs={qualified_state: component_expressions}")
    inputs = []
    normalized = {}
    for target, expressions in outputs.items():
        target = model._species_handle("interaction", name, target)
        if target not in inputs:
            inputs.append(target)
        expressions = (expressions,) if isinstance(expressions, Expr) else tuple(expressions)
        normalized[target] = tuple(_wrap(e) for e in expressions)
    stack = [e for values in normalized.values() for e in values]
    seen = set()
    while stack:
        value = stack.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, QuantityRef):
            if value.handle.kind != "state":
                raise NotImplementedError("interaction field input requires an explicit sampled map")
            state = model._states.get(value.handle.local_id)
            if state is None or state != value.handle or state.space != value.space:
                raise ValueError("interaction dependency is not an exact quantity of this model")
            if state not in inputs:
                inputs.append(state)
        stack.extend(_children(value))
    operator = model.coupled_rate(name, inputs=inputs, outputs=normalized,
                                  preserves=preserves, dissipates=dissipates)
    return model.module.apply(operator, *inputs)


def joint_balance_supported(view):
    """The implemented joint-source partition; temporal accumulation stays solve-owned."""
    return (view is not None and view.accumulation.is_identity
            and any(item.kind == "projection" for item in view.occurrences)
            and all(item.kind in ("projection", "source", "flux") for item in view.occurrences)
            and sum(item.kind == "flux" for item in view.occurrences) <= 1
            and all(item.coefficient == -1 for item in view.occurrences if item.kind == "flux"))
