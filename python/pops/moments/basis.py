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


class CartesianMonomialBasis:
    """Immutable complete monomial basis with explicit, freely ordered multi-indices.

    No component names, velocity rank or degree ceiling are inferred from a model.
    Each index is an exact nonnegative integer tuple; every monomial through the
    maximum total degree must be present. The supplied order is retained in identity.
    """
    __slots__ = ("_indices", "_dimension", "_order")

    def __init__(self, indices):
        from math import comb
        indices = tuple(tuple(index) for index in indices)
        if not indices or not indices[0]:
            raise ValueError("CartesianMonomialBasis requires nonempty positive-rank indices")
        dimension = len(indices[0])
        if any(len(index) != dimension or any(type(n) is not int or n < 0 for n in index) for index in indices):
            raise ValueError("CartesianMonomialBasis requires exact nonnegative integer multi-indices")
        order = max(map(sum, indices))
        if len(set(indices)) != len(indices) or len(indices) != comb(order + dimension, dimension):
            raise ValueError("CartesianMonomialBasis requires a distinct complete total-degree basis")
        # Distinct nonnegative indices already lie in this degree simplex; its
        # exact cardinality proves completeness without exponential enumeration.
        object.__setattr__(self, '_indices', indices)
        object.__setattr__(self, '_dimension', dimension)
        object.__setattr__(self, '_order', order)

    def __setattr__(self, name, value):
        raise AttributeError("CartesianMonomialBasis is immutable")

    @property
    def indices(self): return self._indices
    @property
    def dimension(self): return self._dimension
    @property
    def order(self): return self._order

    def index(self, multi_index):
        multi_index = tuple(multi_index)
        if len(multi_index) != self.dimension or any(type(n) is not int or n < 0 for n in multi_index):
            raise ValueError("basis lookup requires an exact nonnegative multi-index")
        return self.indices.index(multi_index)

    def to_data(self):
        return {'schema': 'pops.cartesian-monomial-basis@1', 'indices': [list(index) for index in self.indices]}

__all__.append('CartesianMonomialBasis')
