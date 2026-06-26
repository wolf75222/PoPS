"""pops.lib.moments.basis -- the moment-basis descriptor (inert).

Tracks which representation a stage of the generator works in (the engine's ``C``
central-moment dict and ``S`` standardized-moment dict): RAW raw moments M_pq,
CENTRAL central moments C_pq, STANDARDIZED standardized moments S_pq. It documents
the engine's internal transforms; it computes nothing.
"""

from .ordering import MomentOrdering


class MomentBasis:
    """The moment representation a hierarchy reports: ``RAW`` / ``CENTRAL`` / ``STANDARDIZED``.

    Inert label over the engine's three representations (raw M_pq, central C_pq,
    standardized S_pq). The generator transforms RAW -> CENTRAL -> STANDARDIZED and back;
    this descriptor records the ordering and the order, not numeric data.
    """

    RAW = "raw"
    CENTRAL = "central"
    STANDARDIZED = "standardized"
    _KINDS = (RAW, CENTRAL, STANDARDIZED)

    def __init__(self, order, kind=RAW, ordering=None):
        if order < 2:
            raise ValueError("MomentBasis: order >= 2 required (got %r)" % (order,))
        if kind not in MomentBasis._KINDS:
            raise ValueError("MomentBasis kind %r must be one of %s"
                             % (kind, ", ".join(MomentBasis._KINDS)))
        self.order = int(order)
        self.kind = kind
        self.ordering = ordering or MomentOrdering()

    def names(self):
        """The moment names of this basis at its order (``M{p}{q}`` for the RAW basis)."""
        from .model_builder import moment_names
        return moment_names(self.order)

    def __repr__(self):
        return "MomentBasis(order=%d, kind=%r)" % (self.order, self.kind)


__all__ = ["MomentBasis"]
