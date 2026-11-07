"""Optional runtime time-step bound authoring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time._program.value_validation import require_owned, require_region
from pops.time._authoring import atomic_authoring, readonly_authoring_query

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


_READONLY_OPS = frozenset({"state", "solve_fields", "reduce", "compare", "cfl", "hmin",
                           "max_wave_speed", "scalar_op"})


def readonly_dt_bound_nodes(program: Any, sub: Any = None) -> list[Any]:
    """Close one isolated query over authenticated, read-only captures in DAG order.

    This is a lowering view: issued nodes, IDs and authored region order stay intact.
    In particular a captured reduction cannot hide a temporal update in its inputs.
    """
    if sub is None:
        if program._dt_bound is None:
            return []
        sub = program._dt_bound[0]
    return readonly_query_nodes(program, sub, where="set_dt_bound", ops=_READONLY_OPS)


def readonly_query_nodes(program: Any, sub: Any, *, where: str, ops: Any) -> list[Any]:
    """The read-only region @p sub and its transitive captures in DAG order, each of @p ops.

    The dt bound and the Outputs of an elaborated Advance (spec 0.3 CC14) are such regions: they
    read the accepted state between steps and compute scalars, and no node of theirs updates it.
    """
    ordered, complete, visiting = [], set(), set()

    for root in sub:
        pending = [(root, False)]
        while pending:
            value, exiting = pending.pop()
            require_owned(program, value, "%s capture" % where)
            key = id(value)
            if exiting:
                visiting.remove(key)
                complete.add(key)
                ordered.append(value)
                continue
            if key in complete:
                continue
            if key in visiting:
                raise ValueError("%s: cyclic read-only capture dependencies" % where)
            if value.op not in ops:
                raise ValueError(
                    "%s: body and captures may only read state/fields and compute "
                    "scalars; op %r is not allowed" % (where, value.op))
            visiting.add(key)
            pending.append((value, True))
            pending.extend((dependency, False) for dependency in reversed(value.inputs))
    return ordered


class _ProgramDtBound(_ProgramBase):
    """Builder-callback-only dt-bound surface, isolated from general Program authoring."""

    @atomic_authoring
    def set_dt_bound(self, builder: Any) -> Any:
        """Record a read-only scalar sub-program built by ``builder(P, cfl)``.

        A pre-built scalar is intentionally refused: its nodes live in the top-level region and
        cannot designate the result of an isolated dt-bound DAG. The callback form gives the bound
        one exact authoring region; its transitive captures must also be read-only.
        """
        self._guard_mutable("set the dt bound")
        if self._dt_bound is not None:
            raise ValueError("set_dt_bound: a dt bound is already set (set it at most once)")
        if not callable(builder):
            raise TypeError(
                "set_dt_bound requires a builder callable f(P, cfl) -> Scalar; a pre-built "
                "ProgramValue belongs to the top-level region")
        if self._recording:
            raise NotImplementedError("set_dt_bound cannot be opened inside another sub-block")
        sub = []
        with readonly_authoring_query(self):
            self._recording.append(sub)
            try:
                cfl = self._new("scalar", "cfl", (), {}, "cfl", None)
                result = builder(self, cfl)
            finally:
                self._recording.pop()
        if getattr(result, "vtype", None) != "scalar":
            raise ValueError("set_dt_bound: builder must return a Scalar ProgramValue")
        require_region(self, result, self._region_for_block(sub), "set_dt_bound", vtype="scalar")
        readonly_dt_bound_nodes(self, sub)
        self._dt_bound = (sub, result)
        return result

    def dt_bound(self, fn: Any) -> Any:
        """Decorator form of :meth:`set_dt_bound`."""
        self.set_dt_bound(fn)
        return fn

    def has_dt_bound(self) -> bool:
        return self._dt_bound is not None


__all__ = ["_ProgramDtBound", "readonly_dt_bound_nodes", "readonly_query_nodes"]
