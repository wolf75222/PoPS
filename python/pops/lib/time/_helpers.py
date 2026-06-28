"""pops.lib.time._helpers -- shared scheme-builder helpers.

The single-stage RHS assembler ``_stage_rhs`` is the canonical home for the
explicit / split schemes (Spec 4 s6 / s14: the ready schemes live in
``pops.lib.time``). The operator-registry helpers ``_op_space_arity`` / ``_opcall``
introspect the Program-bound registry for the operator-first macros
(predictor_corrector_local_linear, explicit_rk, imex_local_linear) and dispatch
calls with the correct arity.

All three take the live ``pops.time.Program`` instance as their first argument, so
this module needs no ``pops.time`` import (it stays free of any lib -> time
module-scope edge beyond the layering allowance).
"""


def _stage_rhs(P, U, sources, flux):
    """Solve the elliptic fields from U and assemble its RHS for one stage. The FieldContext is
    distinct per stage (no stale global aux). flux=False builds a source-only sub-flow (e.g. Strang S).

    Uses the PRIVATE ``P._rhs_legacy`` builder: the macros author the RHS from the (flux, sources)
    pair, which is the internal lowering of the public typed ``P.rhs(terms=[...])`` -- not a second
    public path."""
    fields = P.solve_fields(U) if flux else None
    return P._rhs_legacy(state=U, fields=fields, flux=flux, sources=list(sources))


def _op_space_arity(P, name):
    """Number of space-typed inputs (State / FieldSpace) of operator @p name in the bound registry."""
    if P._registry is None:
        raise ValueError("operator-first macro: bind a module first (P.bind_operators(module))")
    op = P._registry.get(name)
    return sum(1 for t in op.signature.inputs if getattr(t, "kind", None) in ("state", "field"))


def _opcall(P, name, *candidate_args, value_name=None):
    """Call operator @p name passing exactly as many leading args as its signature's space inputs
    (so an operator that ignores the fields is called with the state alone, and a fields-free linear
    operator with no args)."""
    arity = _op_space_arity(P, name)
    # The PRIVATE _call: the macro resolves the operator by its internal registry name (the user
    # already named it at module-build time), not the public handle-only P.call.
    return P._call(name, *candidate_args[:arity], name=value_name)
