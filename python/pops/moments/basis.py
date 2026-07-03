"""pops.moments.basis -- the moment-basis descriptor (inert).

Tracks which representation a stage of the generator works in (the engine's ``C``
central-moment dict and ``S`` standardized-moment dict): RAW raw moments M_pq,
CENTRAL central moments C_pq, STANDARDIZED standardized moments S_pq. It documents
the engine's internal transforms; it computes nothing.
"""
from __future__ import annotations

from typing import Any

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

    def __init__(self, order: Any, kind: Any = RAW, ordering: Any = None) -> None:
        if order < 2:
            raise ValueError("MomentBasis: order >= 2 required (got %r)" % (order,))
        if kind not in MomentBasis._KINDS:
            raise ValueError("MomentBasis kind %r must be one of %s"
                             % (kind, ", ".join(MomentBasis._KINDS)))
        self.order = int(order)
        self.kind = kind
        self.ordering = ordering or MomentOrdering()

    def names(self) -> Any:
        """The moment names of this basis at its order (``M{p}{q}`` for the RAW basis)."""
        from .model_builder import moment_names
        return moment_names(self.order)

    def __repr__(self) -> str:
        return "MomentBasis(order=%d, kind=%r)" % (self.order, self.kind)


class RawMomentBasis(MomentBasis):
    """A :class:`MomentBasis` fixed to the RAW representation (``M_pq``).

    The issue vocabulary names the raw-moment basis explicitly; this thin subclass pins
    ``kind=RAW`` so ``RawMomentBasis(order)`` reads as the transported raw-moment vector while
    staying a ``MomentBasis`` (``isinstance`` still holds). It adds no state and computes nothing.
    """

    def __init__(self, order, ordering=None):
        super().__init__(order, kind=MomentBasis.RAW, ordering=ordering)

    def __repr__(self):
        return "RawMomentBasis(order=%d)" % (self.order,)


__all__ = ["MomentBasis", "RawMomentBasis"]
