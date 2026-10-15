"""Python algebra for a shared flux consuming two independently owned traces."""

from __future__ import annotations

from pops.model.scalar_contract import (
    Const,
    Var,
    VectorExpr,
    NativeCall,
    scalar_children,
    scalar_expression,
)
from pops.model.hash_data import canonical_hash_data


def TraceFlux(body, *, dimension: int, components: int, stability):
    """Build a NumericalFlux component from ordinary PoPS scalar expressions.

    ``body(left, right, normal, time, dt)`` returns one expression per component;
    ``stability`` has the same arguments and returns a nonnegative normal speed.
    Both callbacks elaborate once in Python. The resulting immutable source package
    is compiled by ``pops.compile`` and evaluated by the native interface scheduler.
    Dimension follows the selected native mesh ABI; component count has no catalogue.
    """
    if type(dimension) is not int or dimension not in (1, 2, 3):
        raise ValueError("TraceFlux dimension must match a native mesh dimension (1, 2, or 3)")
    if type(components) is not int or components < 1:
        raise ValueError("TraceFlux components must be a positive integer")
    if not callable(body) or not callable(stability):
        raise TypeError("TraceFlux requires symbolic body and stability callables")
    vectors = tuple(
        VectorExpr(Var(f"trace_{side}_{i}", "trace_input") for i in range(n))
        for side, n in (("left", components), ("right", components), ("normal", dimension))
    )
    time, dt = (Var("trace_" + name, "trace_input") for name in ("time", "dt"))
    args = (*vectors, time, dt)
    result = body(*args)
    if not isinstance(result, tuple) or len(result) != components:
        raise TypeError("TraceFlux body must return a tuple with the declared component count")
    roots = tuple(scalar_expression(x) for x in (*result, stability(*args)))
    allowed = {id(x) for row in vectors for x in row} | {id(time), id(dt)}
    pending, seen = list(roots), set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, NativeCall):
            raise TypeError("TraceFlux native calls require an explicit fallible provider contract")
        children = scalar_children(node)
        if isinstance(node, Var) and (id(node) not in allowed or node.kind != "trace_input"):
            raise ValueError("TraceFlux contains a foreign variable; capture constants explicitly")
        if not children and not isinstance(node, (Const, Var)):
            raise TypeError("TraceFlux contains a leaf outside its explicit trace/time arguments")
        pending.extend(children)
    evidence = {
        "schema": "pops.trace-flux@1",
        "dimension": dimension,
        "components": components,
        "expressions": canonical_hash_data(roots, where="TraceFlux expressions"),
        "evaluation": "ordered-scalar-dag-kokkos-default-execution@1",
        "publication": "all-faces-finite-nonnegative-bound-before-copy@1",
    }
    from pops.codegen.trace_flux_component import make_component

    return make_component(roots, evidence, dimension, components)


__all__ = ["TraceFlux"]
