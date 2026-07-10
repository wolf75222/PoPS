"""pops.time._rhs_terms -- lower a typed ``P.rhs(terms=[...])`` list onto the legacy
``(flux, sources)`` pair (Spec 5 sec.14.2.4).

Factored out of :mod:`pops.time.program_core` (file-size budget): the transformation is pure
(no Program state), so it lives as a module function. The ``pops.numerics`` / ``pops.model``
imports stay function-local so this module adds no module-scope edge to the time package.
"""
from __future__ import annotations

from typing import Any


def terms_to_flux_sources(program: Any, terms: Any) -> Any:
    """Lower a typed ``terms=[...]`` list onto the legacy ``(flux, sources)``.

    A :class:`pops.numerics.terms.Flux` sets ``flux=True`` (its absence -> ``flux=False``); every
    source term contributes its registered NAME to ``sources`` in list order. Accepted source forms
    are a named :class:`~pops.numerics.terms.SourceTerm` /
    :class:`~pops.numerics.terms.LocalTerm` descriptor, or the exact
    :class:`pops.model.OperatorHandle` returned by ``m.source_term``. Both are resolved against the
    Program's bound registry and must name a ``local_source``. Free strings are private lowering
    tokens, not public typed terms. A non-term object is a clear ``TypeError``.
    """
    from pops.numerics.terms import Flux, LocalTerm, SourceTerm
    from pops.model import OperatorHandle
    from pops.time.operator_resolution import (
        resolve_operator_handle, resolve_registered_operator,
    )
    flux = False
    sources = []
    for t in terms:
        if isinstance(t, Flux):
            flux = True
        elif isinstance(t, (SourceTerm, LocalTerm)):
            # An unnamed SourceTerm/LocalTerm has no source name to fold in (its .name would be the
            # class name, which is not a declared m.source_term); reject it transparently.
            if t._name is None:
                raise ValueError(
                    "rhs: a %s in terms= must be named (the name of a declared m.source_term); "
                    "got an unnamed %s" % (type(t).__name__, type(t).__name__))
            operator = resolve_registered_operator(
                program, t.name, where="rhs: terms=", expected_kinds="local_source")
            sources.append(operator.name)
        elif isinstance(t, OperatorHandle):
            operator = resolve_operator_handle(
                program, t, where="rhs: terms=", expected_kinds="local_source")
            sources.append(operator.lowering.get("source", operator.name))
        elif isinstance(t, str):
            raise TypeError(
                "rhs: a free source name %r is not accepted in public terms=; pass the "
                "OperatorHandle returned by m.source_term(...) or a typed SourceTerm/LocalTerm "
                "descriptor" % t)
        else:
            raise TypeError(
                "rhs: terms= entries must be a pops.numerics.terms.Flux/SourceTerm/LocalTerm or "
                "the source OperatorHandle returned by m.source_term; got %r "
                "(note: Flux() is a term, not a bool)" % (t,))
    return flux, sources
