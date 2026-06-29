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

    Uses the PRIVATE ``P._rate_from_transport`` builder: ready-made library macros may author primitive
    transport/source flows, while user Programs should call declared rate operator handles."""
    fields = P._fields_from_state(U) if flux else None
    return P._rate_from_transport(state=U, fields=fields, flux=flux, sources=list(sources))


def _operator_name(selector):
    """Return the registry name of a typed operator selector.

    ``pops.lib.time`` macros are public ready-made scheme builders. They therefore accept the same
    typed operator handles as ``Program.call`` and refuse bare string selectors; the string name only
    reappears after validation as the private registry key used by ``Program._call``.
    """
    from pops.model import Operator, OperatorHandle
    if isinstance(selector, OperatorHandle):
        return selector.name
    if isinstance(selector, Operator):
        return selector.name
    if isinstance(selector, str):
        raise TypeError(
            "operator-first time macros require typed operator handles, not the string %r; "
            "keep the handle returned by m.rate(...), m.rate_operator(...), m.linear_source(...), "
            "or use OperatorHandle(name, kind=...) for built-in operators such as fields_from_state"
            % selector)
    raise TypeError(
        "operator-first time macros require an OperatorHandle/Operator, got %r"
        % type(selector).__name__)


def _op_space_arity(P, selector):
    """Number of space-typed inputs (State / FieldSpace) of an operator selector."""
    if P._registry is None:
        raise ValueError("operator-first macro: bind a module first (P.bind_operators(module))")
    name = _operator_name(selector)
    op = P._registry.get(name)
    return sum(1 for t in op.signature.inputs if getattr(t, "kind", None) in ("state", "field"))


def _opcall(P, selector, *candidate_args, value_name=None):
    """Call a typed operator passing exactly as many leading args as its signature's space inputs
    (so an operator that ignores the fields is called with the state alone, and a fields-free linear
    operator with no args)."""
    name = _operator_name(selector)
    arity = _op_space_arity(P, selector)
    # The PRIVATE _call is only a lowering seam here: the macro has already validated a typed public
    # selector, then reuses the registry name to preserve the byte-identical generated IR.
    return P._call(name, *candidate_args[:arity], name=value_name)
