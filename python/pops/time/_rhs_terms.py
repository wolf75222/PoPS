"""Lower typed ``Program.rhs(terms=[...])`` entries to native RHS route data.

Factored out of :mod:`pops.time._program.operations` (file-size budget): the transformation is pure
(no Program state), so it lives as a module function. The ``pops.numerics`` / ``pops.model``
imports stay function-local so this module adds no module-scope edge to the time package.
"""
from __future__ import annotations

from typing import Any


def lower_rhs_terms(program: Any, terms: Any, *, state: Any) -> Any:
    """Validate typed terms and return their exact native flux/source projection.

    ``Flux()`` selects the default conservative divergence; ``Flux(handle)`` selects an exact
    named grid operator. Their absence sets ``flux=False``.
    :class:`~pops.numerics.terms.DefaultSource` selects the default/composite source of the exact
    model owner carried by ``state``. Every named source is selected by the exact
    :class:`pops.model.OperatorHandle` returned by ``m.source_term``, directly or through a
    :class:`~pops.numerics.terms.SourceTerm` / :class:`~pops.numerics.terms.LocalTerm` wrapper.
    The returned handle tuple is retained in Program IR while the registered names are the explicit
    lowering projection. Free strings are private lowering tokens, not public typed terms.
    """
    from pops.numerics.terms import DefaultSource, Flux, LocalTerm, SourceTerm
    from pops.model import OperatorHandle
    from pops.time.operator_resolution import resolve_operator_handle
    if not isinstance(terms, (list, tuple)):
        raise TypeError("rhs: terms= must be a list or tuple of typed RHS terms")
    default_flux = False
    fluxes = []
    flux_handles = []
    sources = []
    handles = []
    for t in terms:
        if isinstance(t, Flux):
            if t.operator is None:
                if default_flux:
                    raise ValueError("rhs: terms= contains Flux() more than once")
                if fluxes:
                    raise ValueError(
                        "rhs: terms= cannot mix Flux() with named Flux(handle) terms")
                default_flux = True
            else:
                if default_flux:
                    raise ValueError(
                        "rhs: terms= cannot mix Flux() with named Flux(handle) terms")
                operator = resolve_operator_handle(
                    program, t.operator, where="rhs: terms=",
                    expected_kinds="grid_operator", values=(state,))
                if operator.name in fluxes:
                    raise ValueError(
                        "rhs: terms= contains named flux %r more than once" % t.operator.name)
                fluxes.append(operator.name)
                flux_handles.append(t.operator)
        elif isinstance(t, DefaultSource):
            if "default" in sources:
                raise ValueError("rhs: terms= contains DefaultSource() more than once")
            sources.append("default")
        elif isinstance(t, (SourceTerm, LocalTerm)):
            handle = t.operator
            operator = resolve_operator_handle(
                program, handle, where="rhs: terms=", expected_kinds="local_source",
                values=(state,))
            source_name = operator.lowering.get("source", operator.name)
            if source_name in sources:
                raise ValueError(
                    "rhs: terms= contains source %r more than once" % handle.name)
            sources.append(source_name)
            handles.append(handle)
        elif isinstance(t, OperatorHandle):
            operator = resolve_operator_handle(
                program, t, where="rhs: terms=", expected_kinds="local_source",
                values=(state,))
            source_name = operator.lowering.get("source", operator.name)
            if source_name in sources:
                raise ValueError(
                    "rhs: terms= contains source %r more than once" % t.name)
            sources.append(source_name)
            handles.append(t)
        elif isinstance(t, str):
            raise TypeError(
                "rhs: a free source name %r is not accepted in public terms=; pass the "
                "OperatorHandle returned by m.source_term(...), SourceTerm(handle), or "
                "DefaultSource()" % t)
        else:
            raise TypeError(
                "rhs: terms= entries must be Flux(), Flux(handle), DefaultSource(), SourceTerm(handle), "
                "LocalTerm(handle), or the source OperatorHandle returned by m.source_term; got %r "
                "(note: Flux() is a term, not a bool)" % (t,))
    return (
        default_flux or bool(fluxes),
        sources,
        tuple(handles),
        None if default_flux or not fluxes else fluxes,
        tuple(flux_handles),
    )
