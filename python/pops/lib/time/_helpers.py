"""pops.lib.time._helpers -- shared scheme-builder helpers.

The single-stage RHS assembler ``_stage_rhs`` is the canonical home for the
explicit / split schemes (Spec 4 s6 / s14: the ready schemes live in
``pops.lib.time``). The operator-registry helpers ``_op_space_arity`` / ``_opcall``
introspect the Program-bound registry for the operator-first factories
(predictor_corrector_local_linear, explicit_rk, IMEX) and dispatch
calls with the correct arity.

The ``@program_macro`` decorator (ADC-554) makes a scheme builder ONE IR route with
the manual Program: called with a live ``Program`` first argument it mutates that Program;
called WITHOUT one (``BlockHandle, state Handle`` first) it builds a fresh ``Program`` and
returns it. Free block/state strings are never accepted.

The stage helpers take the live ``pops.time.Program`` instance as their first argument,
so they need no ``pops.time`` import; the decorator imports ``pops.time.Program`` LAZILY
(function-local), keeping this module free of a lib -> time module-scope edge beyond the
layering allowance.
"""
from __future__ import annotations

import functools
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pops.numerics.terms import DefaultSource, Flux, LocalTerm, SourceTerm


# Ready schemes explicitly include the block model's default/composite source. The selector is a
# typed semantic term; the historical free token ``"default"`` never appears at their public edge.
_DEFAULT_SOURCES = (DefaultSource(),)


def _exact_coefficient(value: Any, where: str) -> Any:
    """Return one finite, unannotated real coefficient without a float round-trip.

    Ready-made schemes use the same accepted scalar domain as the symbolic IR:
    ``int`` / ``Fraction`` / ``Decimal`` / binary64 ``float`` (or an equivalent
    numeric ``ScalarLiteral``).  Units, target annotations and algebraic/custom
    C++ spellings cannot participate in affine coefficient algebra and are
    rejected here, at the preset boundary, with the preset argument name.
    """
    from pops.ir.literals import scalar_literal

    try:
        literal = scalar_literal(value)
    except TypeError as exc:
        raise TypeError(
            "%s must be a finite real coefficient (int, Fraction, Decimal, or float); got %r"
            % (where, value)
        ) from exc
    except ValueError as exc:
        raise ValueError("%s must be a finite real coefficient; got %r" % (where, value)) from exc
    if literal.unit is not None or literal.target is not None:
        raise TypeError(
            "%s cannot carry a unit or target annotation inside a time-scheme coefficient"
            % where)
    try:
        return literal.to_python()
    except TypeError as exc:
        raise TypeError(
            "%s must be numerically composable; algebraic/custom C++ literals belong in "
            "Program scalar expressions, not affine time coefficients" % where
        ) from exc


def _exact_product(*values: Any, where: str) -> Any:
    """Multiply coefficients without silently crossing numeric domains.

    Integers are neutral in every exact domain.  Mixing Decimal, Fraction and
    binary64 in one product requires an explicit conversion by the caller; this
    mirrors Program coefficient algebra and prevents an implicit float fallback.
    """
    from pops.ir.literals import exact_decimal_multiply, numeric_domains_compatible

    normalized = [_exact_coefficient(value, where) for value in values]
    if not normalized:
        return 1
    result = normalized[0]
    for value in normalized[1:]:
        if not numeric_domains_compatible(result, value):
            raise TypeError(
                "%s cannot mix %s and %s without an explicit numeric conversion"
                % (where, type(result).__name__, type(value).__name__))
        if isinstance(result, Decimal) or isinstance(value, Decimal):
            result = exact_decimal_multiply(result, value)
        else:
            result = result * value
    return result


def _exact_reciprocal(value: Any, where: str) -> Any:
    """Return ``1/value`` in value's authoring domain, never via binary64."""
    value = _exact_coefficient(value, where)
    if value == 0:
        raise ValueError("%s must be non-zero" % where)
    if isinstance(value, Decimal):
        from pops.ir.literals import exact_decimal_divide
        result = exact_decimal_divide(1, value)
        if result is None:
            raise TypeError(
                "%s has a non-terminating Decimal reciprocal; use Fraction for an exact ratio"
                % where)
        return result
    if isinstance(value, float):
        return 1.0 / value
    return Fraction(1, 1) / value


def _exact_fraction(value: Any, where: str) -> Fraction:
    """Exact rational view used only to validate tableau identities."""
    value = _exact_coefficient(value, where)
    if isinstance(value, float):
        return Fraction.from_float(value)
    return Fraction(value)


def program_macro(build: Any) -> Any:
    """Make a scheme builder both an in-place mutator AND a Program factory (ADC-554).

    ``build(P, block, state, ...)`` is the in-place builder. The wrapper dispatches on the FIRST
    positional argument:

      - a live ``pops.time.Program`` -> the in-place path, returning ``build``'s own
        result (the final state / value / ``None``) so every existing ``macro(P, block, ...)`` caller
        is byte-identical;
      - anything else -> a fresh ``Program`` (named after the scheme) is created, the builder lowers
        into it, and the PROGRAM is returned. The builder itself validates the required typed
        ``BlockHandle`` and state declaration.

    The two forms lower through the SAME builder into the SAME IR (only the Program's ``name`` differs
    between an explicit ``Program("x")`` and the fresh default), so a macro and the equivalent manual
    Program produce the same logical IR.
    """
    @functools.wraps(build)
    def macro(*args: Any, **kwargs: Any) -> Any:
        from pops.time import Program  # lazy: keep pops.lib.time free of a module-scope time edge
        from pops.provenance import callable_span, source_span
        caller = source_span()
        context = {
            "caller": caller,
            "factory": callable_span(build),
            "authoring_api": "%s.%s" % (build.__module__, build.__qualname__),
        }
        if args and isinstance(args[0], Program):
            prog = args[0]
            previous = prog._provenance_context
            prog._provenance_context = context
            try:
                return build(*args, **kwargs)
            finally:
                prog._provenance_context = previous
        prog = Program(build.__name__)
        prog._provenance_context = context
        try:
            build(prog, *args, **kwargs)
        finally:
            prog._provenance_context = None
        return prog

    return macro


def _time_state(P: Any, block: Any, state: Any = None) -> Any:
    """Return the Program-owned TimeState selected by typed preset arguments.

    Presets accept either ``(block_handle, model_state_handle)`` or a ``TimeState`` already issued
    by ``P``. The latter is convenient when one explicit program composes several preset fragments.
    """
    from pops.time.handles import TimeState
    if isinstance(block, TimeState):
        if state is not None:
            raise TypeError(
                "time preset: pass either a TimeState or (BlockHandle, state Handle), not both")
        return P._require_time_state(block, "time preset")
    if state is None:
        raise TypeError(
            "time preset: provide (BlockHandle, state Handle); free block names are not accepted")
    return P.state(block, state)


def _block_label(state: Any) -> str:
    """Display/runtime label derived from a typed TimeState."""
    from pops.time.references import block_name
    return block_name(state.block)


def _commit(P: Any, state: Any, value: Any) -> Any:
    """Commit through the typed endpoint retained by a preset."""
    return P.commit(state.next, value)


def _stage_point(P: Any, name: Any, offset: Any = 0, *, partitions: Any = None) -> Any:
    """Build one exact named stage coordinate on the Program's logical clock."""
    from pops.time.points import StagePoint, TimePoint

    if partitions is None:
        partitions = {"main": offset}
    return StagePoint(name, {
        partition: TimePoint(P.clock, coordinate)
        for partition, coordinate in partitions.items()
    })


def _at_point(P: Any, value: Any, point: Any) -> Any:
    """Replace one authored SSA record with the same value at an exact evaluation point."""
    return P._replace_value(value, point=point)


def _stage_rhs(
        P: Any, U: Any, sources: Any, flux: Any, *, name: Any, offset: Any = 0,
        partitions: Any = None) -> Any:
    """Solve the elliptic fields from U and assemble its RHS for one stage. The FieldContext is
    distinct per stage (no stale global aux). flux=False builds a source-only sub-flow (e.g. Strang S).

    Ready schemes lower through the same public typed ``P.rhs(terms=[...])`` route as an explicit
    Program. Named sources remain owner-qualified ``OperatorHandle`` values until that boundary."""
    point = _stage_point(P, name, offset, partitions=partitions)
    fields = _at_point(P, P.solve_fields(U), point) if flux else None
    return _at_point(
        P, _typed_rhs(P, U, fields=fields, sources=sources, flux=flux), point)


def _typed_rhs(P: Any, U: Any, *, fields: Any, sources: Any, flux: Any) -> Any:
    """Build one preset RHS from typed source selections and an explicit flux switch."""
    terms = _rhs_terms(sources, flux)
    return P.rhs(state=U, fields=fields, terms=terms)


def _rhs_terms(sources: Any, flux: Any) -> list[Any]:
    """Validate a preset's public source protocol and return typed ``P.rhs`` terms."""
    from pops.model import OperatorHandle

    if not isinstance(flux, bool):
        raise TypeError("time preset: flux must be a Python bool")
    if not isinstance(sources, (list, tuple)):
        raise TypeError(
            "time preset: sources must be a list/tuple of OperatorHandle or typed RHS terms")
    terms: list[Any] = [Flux()] if flux else []
    for source in sources:
        if isinstance(source, str):
            raise TypeError(
                "time preset: source names are not accepted; pass the OperatorHandle returned by "
                "m.source_term(...), SourceTerm(handle), or DefaultSource()")
        if not isinstance(source, (OperatorHandle, DefaultSource, SourceTerm, LocalTerm)):
            raise TypeError(
                "time preset: invalid source selection %r; expected OperatorHandle, "
                "SourceTerm(handle), LocalTerm(handle), or DefaultSource()" % (source,))
        terms.append(source)
    return terms


def _source_names(P: Any, U: Any, sources: Any) -> list[str]:
    """Project typed preset sources to private registry-local lowering names."""
    from pops.time._rhs_terms import terms_to_flux_sources

    _, names, _ = terms_to_flux_sources(P, _rhs_terms(sources, False), state=U)
    return names


def _operator_handle(operator: Any, kwarg: Any) -> Any:
    """Coerce a macro operator selector to a typed :class:`pops.model.OperatorHandle` (ADC-532).

    An operator-first macro takes typed handles (from ``m.rate`` / ``m.field_solve`` /
    ``m.local_linear_map`` / ``m.source_term``), NOT operator name strings. A bare string is REFUSED
    with a clear ``TypeError`` naming the declarer, so a stale ``explicit_operator="lorentz"`` call
    fails loudly instead of silently taking a free-string selector. Returns the handle unchanged."""
    from pops.model import OperatorHandle
    if isinstance(operator, OperatorHandle):
        return operator
    if isinstance(operator, str):
        raise TypeError(
            "operator-first macro: %s must be a typed pops.model.OperatorHandle, not the string %r; "
            "build it with m.rate(...) / m.field_solve(...) / m.local_linear_map(...) / "
            "m.source_term(...)" % (kwarg, operator))
    raise TypeError(
        "operator-first macro: %s must be a pops.model.OperatorHandle (from an m.* declarer), got %r"
        % (kwarg, operator))


def _op_space_arity(P: Any, handle: Any) -> Any:
    """Number of space-typed inputs (State / FieldSpace) of @p handle's operator in the bound
    registry. The handle keeps its complete owner/kind/signature identity until resolution."""
    from pops.time.operator_resolution import resolve_operator_handle
    op = resolve_operator_handle(P, handle, where="operator-first macro")
    return sum(1 for t in op.signature.inputs if getattr(t, "kind", None) in ("state", "field"))


def _opcall(
        P: Any, handle: Any, *candidate_args: Any, value_name: Any = None,
        point: Any = None) -> Any:
    """Call @p handle's operator passing exactly as many leading args as its signature's space inputs
    (so an operator that ignores the fields is called with the state alone, and a fields-free linear
    operator with no args). @p handle is a typed :class:`pops.model.OperatorHandle`.

    The handle is lowered through the INTERNAL ``P._call`` byte-identically to the public
    ``P.call(handle, ...)`` (the same registry lookup, the same primitive-op lowering): the private
    ``_call`` is the allowed internal seam, only the FREE-STRING macro entry is de-stringed."""
    arity = _op_space_arity(P, handle)
    value = P._call(handle, *candidate_args[:arity], name=value_name)
    return _at_point(P, value, point) if point is not None else value
