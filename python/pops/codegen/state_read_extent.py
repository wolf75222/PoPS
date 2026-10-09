"""Conservative state-read effects of compiler-owned cell expressions.

An extension's children/deps protocol is not a memory-effect declaration. Unknown
nodes therefore remain unknown even when they report no dependencies.
"""
from __future__ import annotations


def cell_state_read_extent(model, expression, *, provider_pack=None):
    """Return StateReadExtent@1 (ranked cells), or None for an opaque expression."""
    if provider_pack is not None:
        return _pointwise_state_aux_read_extent(model, expression, provider_pack)
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


def _pointwise_state_aux_read_extent(model, expression, provider_pack):
    """Certify only scalar State/Aux/registered-parameter leaves of the V2 operation."""
    from pops._ir import expr as ir
    from pops._ir.control_expr import Where, Rounded
    from pops._ir.values import RuntimeParamRef
    from pops._ir.visitors import _children
    from pops.model import Module, ParamRegistry
    from pops.model.provider_pack import ComponentContract, ProviderPack
    from .module_emit_helpers import _live_prims, _ranked_axes

    module = getattr(model, "_formula_source_module", None)
    complete = getattr(model, "_component_provider_pack", None)
    if (not isinstance(module, Module) or module.owner_path != model.owner_path
            or type(provider_pack) is not ProviderPack or type(complete) is not ProviderPack):
        return None
    if complete.select(provider_pack).to_data() != provider_pack.to_data():
        return None
    owner = str(module.owner_path.canonical())
    auxiliary_names = set()
    for key in provider_pack:
        space = module.aux().get(key.space_name)
        if (key.owner_qid != owner or key.space_kind != "aux" or space is None
                or key.component != space.name or space.kind != "cell_scalar"
                or space.centering != "cell"):
            return None
        expected = ComponentContract(space.representation, space.centering, space.unit,
                                     space.centering, space.kind)
        if provider_pack.contract(key) != expected:
            return None
        auxiliary_names.add(key.component)
    registry = module.param_registry()
    if not isinstance(registry, ParamRegistry) or registry.owner_path != module.owner_path:
        return None
    leaves = {ir.Const, ir.Var, RuntimeParamRef}
    operators = {ir.Add, ir.Sub, ir.Mul, ir.Div, ir.Pow, ir.Minimum,
                 ir.Maximum, ir.Compare, ir.BooleanAnd, ir.BooleanOr,
                 ir.BooleanNot, ir.Neg, ir.Sqrt, ir.Exp, ir.Abs, ir.Sign,
                 Where, Rounded}
    roots = [expression]
    roots.extend(model.prim_defs[name] for name in _live_prims(model, [expression]))
    seen = set()
    while roots:
        node = roots.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if type(node) not in leaves | operators:
            return None
        if type(node) is ir.Var:
            names = (auxiliary_names if node.kind == "aux" else
                     set(model.cons_names) if node.kind == "cons" else
                     set(model.prim_defs) if node.kind == "prim" else set())
            if node.name not in names:
                return None
        if type(node) is RuntimeParamRef:
            if node.handle is None or node.dtype is None:
                return None
            try:
                declaration = registry.declaration(node.handle)
            except (KeyError, ValueError, TypeError):
                return None
            if getattr(declaration.dtype, "name", None) != node.dtype:
                return None
        roots.extend(_children(node))
    return {"schema": 2, "unit": "cells", "cells": (0,) * len(_ranked_axes(model)),
            "authority": "compiler.pointwise-state-aux-ast@2",
            "provider_count": len(provider_pack)}
