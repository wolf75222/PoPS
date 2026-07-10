"""Immutable, owner-qualified temporal declaration handles.

Temporal handles contain identity and inert type metadata only.  All mutable authoring
state -- current-value caching, stage definitions, history configuration/resolution and
endpoint caching -- belongs to :class:`pops.time.Program`.
"""
from __future__ import annotations

from typing import Any

from pops.model.handles import Handle


def _state_local_id(block: str, state_name: str, suffix: str) -> str:
    """Unambiguous identity from two user-controlled strings plus a structural suffix."""
    return "%d:%s|%d:%s|%s" % (
        len(block), block, len(state_name), state_name, suffix)


def _validate_state_metadata(where: str, block: Any, state_name: Any, space: Any) -> None:
    if not isinstance(block, str) or not block:
        raise ValueError("%s: block must be a non-empty string" % where)
    if not isinstance(state_name, str) or not state_name:
        raise ValueError("%s: state_name must be a non-empty string" % where)
    if space is not None and getattr(space, "kind", None) != "state":
        raise TypeError("%s: space must be a StateSpace or None" % where)


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

    __slots__ = ("_program", "block", "state_name", "key", "space")

    def __init__(self, *, program: Any, block: Any, state_name: Any,
                 key: Any, space: Any = None) -> None:
        _validate_state_metadata("StageHandle", block, state_name, space)
        if not isinstance(key, (int, str)) or isinstance(key, bool):
            raise ValueError("StageHandle: key must be an int or str (got %r)" % (key,))
        key_kind = "int" if isinstance(key, int) else "str"
        key_text = str(key)
        suffix = "stage|%s|%d:%s" % (key_kind, len(key_text), key_text)
        super().__init__(
            _state_local_id(block, state_name, suffix),
            kind="state_stage", owner=program.owner_path)
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "state_name", state_name)
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "space", space)

    def __repr__(self) -> str:
        return "StageHandle(block=%r, state_name=%r, key=%r)" % (
            self.block, self.state_name, self.key)


class HistoryHandle(_ReadableTemporalHandle, Handle):
    """Immutable readable handle for one lag of a Program-owned state-history ring.

    ``U.prev`` is the lag-one handle. Calling it as ``U.prev(lag)`` returns the
    Program-cached handle for that lag; resolution to a ProgramValue remains lazy.
    """

    __slots__ = ("_program", "block", "state_name", "lag", "space")

    def __init__(self, *, program: Any, block: Any, state_name: Any,
                 lag: Any, space: Any = None) -> None:
        _validate_state_metadata("HistoryHandle", block, state_name, space)
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("HistoryHandle: lag must be a Python int >= 1 (got %r)" % (lag,))
        super().__init__(
            _state_local_id(block, state_name, "history|%d" % lag),
            kind="state_history", owner=program.owner_path)
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "state_name", state_name)
        object.__setattr__(self, "lag", lag)
        object.__setattr__(self, "space", space)

    def __call__(self, lag: Any = 1) -> Any:
        return self._program._history_handle_from(self, lag)

    def __repr__(self) -> str:
        return "HistoryHandle(block=%r, state_name=%r, lag=%d)" % (
            self.block, self.state_name, self.lag)


class StateEndpointHandle(Handle):
    """Immutable commit-only destination of one end-of-step state.

    It deliberately has no value resolution or symbolic algebra. Program.commit also
    verifies that it is the exact endpoint issued and cached by that Program.
    """

    __slots__ = ("block", "state_name", "space")

    @property
    def expression_readable(self) -> bool:
        return False

    def __init__(self, *, owner: Any, block: Any, state_name: Any, space: Any = None) -> None:
        _validate_state_metadata("StateEndpointHandle", block, state_name, space)
        super().__init__(
            _state_local_id(block, state_name, "next"),
            kind="state_endpoint", owner=owner)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "state_name", state_name)
        object.__setattr__(self, "space", space)


class TimeState(Handle):
    """Immutable family handle for ``U.n``, stages, history and ``U.next``."""

    __slots__ = ("_program", "block", "state_name", "space")

    @property
    def expression_readable(self) -> bool:
        return False

    def __init__(self, program: Any, block: Any, name: Any = "U", *, space: Any = None) -> None:
        _validate_state_metadata("TimeState", block, name, space)
        super().__init__(
            _state_local_id(block, name, "family"),
            kind="time_state", owner=program.owner_path)
        object.__setattr__(self, "_program", program)
        object.__setattr__(self, "block", block)
        object.__setattr__(self, "state_name", name)
        object.__setattr__(self, "space", space)

    @property
    def name(self) -> str:
        return self.state_name

    @property
    def n(self) -> Any:
        return self._program._current_time_value(self)

    def stage(self, key: Any) -> StageHandle:
        return self._program._stage_handle(self, key)

    @property
    def next(self) -> StateEndpointHandle:
        return self._program._endpoint_handle(self)

    @property
    def prev(self) -> HistoryHandle:
        return self._program._history_handle(self, 1)

    def __repr__(self) -> str:
        return "TimeState(block=%r, name=%r)" % (self.block, self.state_name)


__all__ = ["HistoryHandle", "StageHandle", "StateEndpointHandle", "TimeState"]
