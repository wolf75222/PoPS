"""pops.time._rhs_terms -- lower a typed ``P.rhs(terms=[...])`` list onto the legacy
``(flux, sources)`` pair (Spec 5 sec.14.2.4).

Factored out of :mod:`pops.time.program_core` (file-size budget): the transformation is pure
(no Program state), so it lives as a module function. The ``pops.numerics`` / ``pops.model``
imports stay function-local so this module adds no module-scope edge to the time package.
"""
from __future__ import annotations

from typing import Any


def terms_to_flux_sources(program: Any, terms: Any, *, state: Any) -> Any:
    """Lower a typed ``terms=[...]`` list onto the legacy ``(flux, sources)``.

    A :class:`pops.numerics.terms.Flux` sets ``flux=True`` (its absence -> ``flux=False``).
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
    flux = False
    sources = []
    handles = []
    for t in terms:
        if isinstance(t, Flux):
            flux = True
        elif isinstance(t, DefaultSource):
            sources.append("default")
        elif isinstance(t, (SourceTerm, LocalTerm)):
            handle = t.operator
            operator = resolve_operator_handle(
                program, handle, where="rhs: terms=", expected_kinds="local_source",
                values=(state,))
            sources.append(operator.lowering.get("source", operator.name))
            handles.append(handle)
        elif isinstance(t, OperatorHandle):
            operator = resolve_operator_handle(
                program, t, where="rhs: terms=", expected_kinds="local_source",
                values=(state,))
            sources.append(operator.lowering.get("source", operator.name))
            handles.append(t)
        elif isinstance(t, str):
            raise TypeError(
                "rhs: a free source name %r is not accepted in public terms=; pass the "
                "OperatorHandle returned by m.source_term(...), SourceTerm(handle), or "
                "DefaultSource()" % t)
        else:
            raise TypeError(
                "rhs: terms= entries must be Flux(), DefaultSource(), SourceTerm(handle), "
                "LocalTerm(handle), or the source OperatorHandle returned by m.source_term; got %r "
                "(note: Flux() is a term, not a bool)" % (t,))
    return flux, sources, tuple(handles)
