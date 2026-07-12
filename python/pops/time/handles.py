"""Immutable, owner-qualified temporal declaration handles.

Temporal handles contain identity and inert type metadata only.  All mutable authoring
state -- current-value caching, stage definitions, history configuration/resolution and
endpoint caching -- belongs to :class:`pops.time.Program`.
"""
from __future__ import annotations

from typing import Any

from pops.model.handles import Handle
from pops.time.points import Clock, StagePoint, TimePoint
from pops.time.references import block_name, canonical_handle, state_name


def _state_local_id(block: Any, state: Handle, clock: Clock, suffix: str) -> str:
    """Unambiguous identity from state, clock and a structural suffix."""
    block_id = canonical_handle(block).qualified_id
    state_id = canonical_handle(state).qualified_id
    clock_id = clock.qualified_id
    return "%d:%s|%d:%s|%d:%s|%s" % (
        len(block_id), block_id, len(state_id), state_id,
        len(clock_id), clock_id, suffix)


def _validate_state_metadata(
        where: str, block: Any, state: Any, space: Any, clock: Any) -> None:
    from pops.problem.handles import BlockHandle
    if not isinstance(block, BlockHandle):
        raise TypeError("%s: block must be a BlockHandle" % where)
    if not isinstance(state, Handle) or state.kind != "state" or not state.is_instance:
        raise TypeError("%s: state must be a block-qualified state Handle" % where)
    if state.block_ref is not block:
        raise ValueError("%s: state is qualified by a different block" % where)
    if space is not None and getattr(space, "kind", None) != "state":
        raise TypeError("%s: space must be a StateSpace or None" % where)
    if type(clock) is not Clock:
        raise TypeError("%s: clock must be an exact Clock" % where)


class _ReadableTemporalHandle:
    """Affine proxy whose resolution is exclusively owned by its Program table."""

    __slots__ = ()

    @property
    def value(self) -> Any:
        return self._as_value()

    def _as_value(self) -> Any:
        return self._program._resolve_time_value(self)

    def __add__(self, other: Any) -> Any:
        return self._as_value().__add__(other)

    def __radd__(self, other: Any) -> Any:
        return self._as_value().__radd__(other)

    def __sub__(self, other: Any) -> Any:
        return self._as_value().__sub__(other)

    def __rsub__(self, other: Any) -> Any:
        return self._as_value().__rsub__(other)

    def __neg__(self) -> Any:
        return self._as_value().__neg__()

    def __mul__(self, other: Any) -> Any:
        return self._as_value().__mul__(other)

    def __rmul__(self, other: Any) -> Any:
        return self._as_value().__rmul__(other)

    def __truediv__(self, other: Any) -> Any:
        return self._as_value().__truediv__(other)

    def __rmatmul__(self, other: Any) -> Any:
        """Let a Program operator consume the resolved State without exposing storage here."""
        return other.__matmul__(self._as_value())


class StageHandle(_ReadableTemporalHandle, Handle):
    """Immutable readable handle for one single-assignment temporal stage."""

    __slots__ = ("_program", "block", "state", "key", "space", "clock", "point")

    def __init__(self, *, program: Any, block: Any, state: Any,
                 key: Any, clock: Any, point: Any, space: Any = None) -> None:
        _validate_state_metadata("StageHandle", block, state, space, clock)
        if not isinstance(key, str) or not key:
            raise ValueError("StageHandle: key must be a non-empty string (got %r)" % (key,))
        if type(point) is not StagePoint:
            raise TypeError("StageHandle: point must be an exact StagePoint")
        if point.name != key:
            raise ValueError(
                "StageHandle: key %r must match StagePoint name %r" % (key, point.name))
        if any(candidate.clock != clock for candidate in point.partitions.values()):
            raise ValueError("StageHandle: every partition point must use the TimeState clock")
        suffix = "stage|str|%d:%s" % (len(key), key)
        super().__init__(
            _state_local_id(block, state, clock, suffix),
            kind="state_stage", owner=program.owner_path)
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", point)

    def __repr__(self) -> str:
        return "StageHandle(block=%r, state=%r, key=%r)" % (
            block_name(self.block), state_name(self.state), self.key)


class HistoryHandle(_ReadableTemporalHandle, Handle):
    """Immutable readable handle for one lag of a Program-owned state-history ring.

    ``U.prev`` is the lag-one handle. Calling it as ``U.prev(lag)`` returns the
    Program-cached handle for that lag; resolution to a ProgramValue remains lazy.
    """

    __slots__ = ("_program", "block", "state", "lag", "space", "clock", "point")

    def __init__(self, *, program: Any, block: Any, state: Any,
                 lag: Any, clock: Any, space: Any = None) -> None:
        _validate_state_metadata("HistoryHandle", block, state, space, clock)
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("HistoryHandle: lag must be a Python int >= 1 (got %r)" % (lag,))
        super().__init__(
            _state_local_id(block, state, clock, "history|%d" % lag),
            kind="state_history", owner=program.owner_path)
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "lag", lag)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", TimePoint(clock, step=-lag))

    def __call__(self, lag: Any = 1) -> Any:
        return self._program._history_handle_from(self, lag)

    def __repr__(self) -> str:
        return "HistoryHandle(block=%r, state=%r, lag=%d)" % (
            block_name(self.block), state_name(self.state), self.lag)


class StateEndpointHandle(Handle):
    """Immutable commit-only destination of one end-of-step state.

    It deliberately has no value resolution or symbolic algebra. Program.commit also
    verifies that it is the exact endpoint issued and cached by that Program.
    """

    __slots__ = ("block", "state", "space", "clock", "point")

    @property
    def expression_readable(self) -> bool:
        return False

    def __init__(self, *, owner: Any, block: Any, state: Any,
                 clock: Any, space: Any = None) -> None:
        _validate_state_metadata("StateEndpointHandle", block, state, space, clock)
        super().__init__(
            _state_local_id(block, state, clock, "next"),
            kind="state_endpoint", owner=owner)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", TimePoint(clock, step=1))


class TimeState(Handle):
    """Immutable family handle for ``U.n``, stages, history and ``U.next``."""

    __slots__ = ("_program", "block", "state", "space", "clock", "point")

    @property
    def expression_readable(self) -> bool:
        return False

    def __init__(self, program: Any, block: Any, state: Any, *,
                 clock: Any, space: Any = None) -> None:
        _validate_state_metadata("TimeState", block, state, space, clock)
        super().__init__(
            _state_local_id(block, state, clock, "family"),
            kind="time_state", owner=program.owner_path)
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "clock", clock)
        object.__setattr__(self, "point", TimePoint(clock))

    @property
    def name(self) -> str:
        return state_name(self.state)

    @property
    def n(self) -> Any:
        return self._program._current_time_value(self)

    def stage(self, key: Any, *, point: Any) -> StageHandle:
        return self._program._stage_handle(self, key, point)

    @property
    def next(self) -> StateEndpointHandle:
        return self._program._endpoint_handle(self)

    @property
    def prev(self) -> HistoryHandle:
        return self._program._history_handle(self, 1)

    def __repr__(self) -> str:
        return "TimeState(block=%r, state=%r)" % (
            block_name(self.block), state_name(self.state))


__all__ = ["HistoryHandle", "StageHandle", "StateEndpointHandle", "TimeState"]
