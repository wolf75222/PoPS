"""Authenticate all state coordinates of a physical principal flux."""
from collections.abc import Mapping


def flux_state_inputs(model, target, expressions):
    """Return declared state spaces in physical declaration order, including the target.

    The target is needed for the conservative jump even for a constant physical flux.
    Numerical sampling is deliberately not inferred here.
    """
    from pops._ir import Expr, Var, _children
    from pops._ir.quantity import QuantityRef
    from pops._ir.primitive_expansion import expand_primitive_recipes
    module = model._multi_module
    if module is None:
        return (target.space,)
    spaces = module.state_spaces()
    body = expand_primitive_recipes(expressions, module.primitive_recipes())
    selected = {target.space.name}
    pending, seen = [body], set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, QuantityRef):
            canonical = spaces.get(node.space.name)
            if (node.handle.kind != "state" or canonical is not node.space
                    or node.handle != module.state_handle(canonical)):
                raise ValueError("physical flux reads a foreign or undeclared state quantity")
            selected.add(canonical.name)
        elif isinstance(node, Var) and node.kind in {"cons", "prim"}:
            raise ValueError("physical flux requires declared owner-qualified state coordinates")
        if isinstance(node, Expr):
            pending.extend(_children(node))
        elif isinstance(node, Mapping):
            pending.extend(node.values())
        elif isinstance(node, (tuple, list)):
            pending.extend(node)
    return tuple(space for name, space in spaces.items() if name in selected)


def principal_balance(view, registry):
    """A conservative balance whose physical flux reads multiple states."""
    if view is None or not view.occurrences:
        return False
    if any(term.kind != "flux" for term in view.occurrences):
        return False
    return any(len(tuple(space for space in registry.get(term.payload.reg_name).signature.inputs
                         if space.kind == "state")) > 1 for term in view.occurrences)
