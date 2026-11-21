"""Build a case without the storage constraints it writes (DM3, N10 lot 4, addendum §2).

DM3 says every case conforms without its storage constraints. The b6 cases keep theirs until N15
(Q4), so this context manager drops them while the case is built, without editing the case file:

* ``range_set(..., ownership="replicated")`` loses its ``ownership``: the plan chooses the storage;
* ``Distributed.domain(offsets, capacity=, family=)`` becomes
  ``Distributed.balanced_domain(offsets[-1], capacity=, family=)``: the plan partitions the domain
  over the execution lane unless the reception declares those offsets as a plan constraint.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

from pops.time._program.distributed_arrays import Distributed
from pops.time._program.finite_arrays import _ProgramFiniteArrays


@dataclass(frozen=True)
class PlacementConstraint:
    reason: str
    offsets: tuple[tuple[int, ...], ...]


# The roundtrip checks the actual source owner. The CSR requests and rows must
# also keep their historical per-rank lengths for that same observation.
HISTORICAL_ROUNDTRIP_PLACEMENT = {
    2: PlacementConstraint("Distributed.rank() and rank-local CSR incidence",
                           ((0, 1, 4), (0, 2, 6), (0, 1, 3))),
    3: PlacementConstraint("Distributed.rank() and rank-local CSR incidence",
                           ((0, 1, 1, 4), (0, 2, 2, 6), (0, 1, 1, 3))),
}


@contextmanager
def without_storage_constraints(*, placement: PlacementConstraint | None = None):
    range_set, domain = _ProgramFiniteArrays.range_set, Distributed.domain
    retained = Counter(() if placement is None else placement.offsets)

    def unconstrained_range_set(self, cardinality, *, capacity=None, ownership=None, family=None):
        if ownership not in (None, "replicated"):
            return range_set(self, cardinality, capacity=capacity, ownership=ownership,
                             family=family)
        return range_set(self, cardinality, capacity=capacity, family=family)

    def unconstrained_domain(self, offsets, *, capacity=None, family=None):
        if not isinstance(offsets, (tuple, list)) or len(offsets) < 2:
            return domain(self, offsets, capacity=capacity, family=family)
        if (retained and all(type(offset) is int for offset in offsets)
                and retained[tuple(offsets)] > 0):
            retained[tuple(offsets)] -= 1
            return domain(self, offsets, capacity=capacity, family=family)
        return self.balanced_domain(offsets[-1], capacity=capacity, family=family)

    _ProgramFiniteArrays.range_set = unconstrained_range_set
    Distributed.domain = unconstrained_domain
    try:
        yield
        unapplied = +retained
        if unapplied:
            raise AssertionError(f"unapplied plan placement constraints: {unapplied}")
    finally:
        _ProgramFiniteArrays.range_set = range_set
        Distributed.domain = domain
