"""Optional runtime time-step bound authoring."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time._program.value_validation import require_region
from pops.time._authoring import atomic_authoring

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramDtBound(_ProgramBase):
    """Builder-callback-only dt-bound surface, isolated from general Program authoring."""

    @atomic_authoring
    def set_dt_bound(self, builder: Any) -> Any:
        """Record a read-only scalar sub-program built by ``builder(P, cfl)``.

        A pre-built scalar is intentionally refused: its nodes live in the top-level region and
        cannot be emitted as an isolated dt-bound DAG.  The callback form gives the bound one exact
        authoring region and makes every dependency explicit.
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
        self._recording.append(sub)
        try:
            cfl = self._new("scalar", "cfl", (), {}, "cfl", None)
            result = builder(self, cfl)
        finally:
            self._recording.pop()
        if getattr(result, "vtype", None) != "scalar":
            raise ValueError("set_dt_bound: builder must return a Scalar ProgramValue")
        require_region(self, result, self._region_for_block(sub), "set_dt_bound", vtype="scalar")
        allowed = frozenset({"state", "solve_fields", "reduce", "compare", "cfl", "hmin",
                             "max_wave_speed", "scalar_op"})
        for value in sub:
            if value.op not in allowed:
                raise ValueError(
                    "set_dt_bound: body may only read state/fields and compute scalars; "
                    "op %r is not allowed" % value.op)
        self._dt_bound = (sub, result)
        return result

    def dt_bound(self, fn: Any) -> Any:
        """Decorator form of :meth:`set_dt_bound`."""
        self.set_dt_bound(fn)
        return fn

    def has_dt_bound(self) -> bool:
        return self._dt_bound is not None


__all__ = ["_ProgramDtBound"]
