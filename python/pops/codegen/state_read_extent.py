"""Conservative state-read effects of compiler-owned cell expressions.

An extension's children/deps protocol is not a memory-effect declaration. Unknown
nodes therefore remain unknown even when they report no dependencies.
"""
from __future__ import annotations


def cell_state_read_extent(model, expression):
    """Return StateReadExtent@1 (ranked cells), or None for an opaque expression."""
    from pops._ir import expr as ir
    from pops._ir.control_expr import Where, Rounded
    from pops._ir.visitors import _children
    from .module_emit_helpers import _live_prims

    leaves = {ir.Const, ir.Var}
    operators = {ir.Add, ir.Sub, ir.Mul, ir.Div, ir.Pow, ir.Minimum,
                 ir.Maximum, ir.Compare, ir.BooleanAnd, ir.BooleanOr,
                 ir.BooleanNot, ir.Neg, ir.Sqrt, ir.Exp, ir.Abs, ir.Sign,
                 Where, Rounded}
    roots = [expression]
    roots.extend(model.prim_defs[name] for name in _live_prims(model, [expression]))
    permitted_names = set(model.cons_names) | set(model.prim_defs)
    # Runtime parameters are scalar model members, not field/storage accesses.
    seen = set()
    while roots:
        node = roots.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if type(node) not in leaves | operators:
            return None
        if type(node) is ir.Var and node.name not in permitted_names:
            return None
        roots.extend(_children(node))
    from .module_emit_helpers import _ranked_axes
    return {"schema": 1, "unit": "cells", "cells": (0,) * len(_ranked_axes(model)),
            "authority": "compiler.cell-state-ast@1"}
