"""pops.time._rhs_terms -- lower an internal typed term list onto the legacy
``(flux, sources)`` pair (Spec 5 sec.14.2.4).

Factored out of :mod:`pops.time.program_core` (file-size budget): the transformation is pure
(no Program state), so it lives as a module function. The ``pops.numerics`` / ``pops.model``
imports stay function-local so this module adds no module-scope edge to the time package.
"""


def terms_to_flux_sources(terms):
    """Lower a typed ``terms=[...]`` list onto the legacy ``(flux, sources)``.

    A :class:`pops.numerics.terms.Flux` sets ``flux=True`` (its absence -> ``flux=False``); every
    source term contributes its NAME to ``sources`` in list order. Accepted source forms (each maps
    cleanly onto an existing ``sources=`` name): a :class:`~pops.numerics.terms.SourceTerm` /
    :class:`~pops.numerics.terms.LocalTerm` descriptor (its ``.name`` must be set), an
    :class:`pops.model.OperatorHandle` from ``m.source_term`` (its ``.name``), or a plain source
    name ``str``. A non-term object (e.g. a bare ``bool``) is a clear ``TypeError``.
    """
    from pops.numerics.terms import Flux, LocalTerm, SourceTerm
    from pops.model import OperatorHandle
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
            sources.append(t.name)
        elif isinstance(t, OperatorHandle):
            sources.append(t.name)
        elif isinstance(t, str) and t:
            sources.append(t)
        else:
            raise TypeError(
                "rhs: terms= entries must be a pops.numerics.terms.Flux/SourceTerm/LocalTerm, a "
                "source OperatorHandle (from m.source_term), or a source name str; got %r "
                "(note: Flux() is a term, not a bool)" % (t,))
    return flux, sources
