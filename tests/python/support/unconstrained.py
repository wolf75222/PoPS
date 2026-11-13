"""Build a case without the storage constraints it writes (DM3, N10 lot 4, addendum §2).

DM3 says every case conforms without its storage constraints. The b6 cases keep theirs until N15
(Q4), so this context manager drops them while the case is built, without editing the case file:

* ``range_set(..., ownership="replicated")`` loses its ``ownership``: the plan chooses the storage;
* ``Distributed.domain(offsets, capacity=, family=)`` becomes
  ``Distributed.balanced_domain(offsets[-1], capacity=, family=)``: the plan partitions the domain
  over the execution lane instead of the offsets the case wrote.
"""
from __future__ import annotations

from contextlib import contextmanager

from pops.time._program.distributed_arrays import Distributed
from pops.time._program.finite_arrays import _ProgramFiniteArrays


@contextmanager
def without_storage_constraints():
    range_set, domain = _ProgramFiniteArrays.range_set, Distributed.domain

    def unconstrained_range_set(self, cardinality, *, capacity=None, ownership=None, family=None):
        if ownership not in (None, "replicated"):
            return range_set(self, cardinality, capacity=capacity, ownership=ownership,
                             family=family)
        return range_set(self, cardinality, capacity=capacity, family=family)

    def unconstrained_domain(self, offsets, *, capacity=None, family=None):
        if not isinstance(offsets, (tuple, list)) or len(offsets) < 2:
            return domain(self, offsets, capacity=capacity, family=family)
        return self.balanced_domain(offsets[-1], capacity=capacity, family=family)

    _ProgramFiniteArrays.range_set = unconstrained_range_set
    Distributed.domain = unconstrained_domain
    try:
        yield
    finally:
        _ProgramFiniteArrays.range_set = range_set
        Distributed.domain = domain
