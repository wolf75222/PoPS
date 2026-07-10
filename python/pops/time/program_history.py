"""Typed persistent-history authoring for time Programs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time.program_value_validation import require_history_space, require_top_level
from pops.time.values import _is_field_value

if TYPE_CHECKING:
    from pops.time._program_contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramHistory(_ProgramBase):
    """Full-state and narrow scalar history rings with explicit provenance."""

    def history(self, name: Any, lag: Any = 1, ncomp: Any = None, *,
                space: Any = None, block: Any = None) -> Any:
        self._guard_mutable("declare a history read")
        if not isinstance(name, str) or not name:
            raise ValueError("history: name must be a non-empty string")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("history: lag must be a Python int >= 1 (got %r)" % (lag,))
        if block is not None and (not isinstance(block, str) or not block):
            raise ValueError("history: block must be a non-empty string or None")
        if ncomp is None:
            if space is None and name in self._history_spaces:
                space = self._history_spaces[name]
            if block is None and name in self._history_blocks:
                block = self._history_blocks[name]
            require_history_space(self, name, space)
            self._declare_history_block(name, block)
            self._histories[name] = max(self._histories.get(name, 0), lag)
            return self._new(
                "state", "history", (), {"history": name, "lag": int(lag)}, name, block,
                space=space)
        if space is not None or block is not None:
            raise ValueError("history: a narrow scalar ring has no StateSpace/block provenance")
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp != 1:
            raise ValueError(
                "history: explicit ncomp must be 1; omit it for a full-state history")
        self._histories[name] = max(self._histories.get(name, 0), lag)
        self._histories_ncomp[name] = 1
        return self._new(
            "scalar_field", "history", (), {"history": name, "lag": int(lag), "ncomp": 1},
            name, None)

    def _declare_history_block(self, name: str, block: Any) -> None:
        missing = object()
        prior = self._history_blocks.get(name, missing)
        if prior is missing:
            self._history_blocks[name] = block
        elif prior != block:
            raise ValueError(
                "history %r cannot mix block provenance %r and %r" % (name, prior, block))

    def store_history(self, name: Any, value: Any) -> Any:
        self._guard_mutable("store history")
        if not isinstance(name, str) or not name:
            raise ValueError("store_history: name must be a non-empty string")
        if not _is_field_value(value):
            raise ValueError("store_history: value must be a State/RHS field (got %r)" % (value,))
        require_top_level(self, value, "store_history")
        if value.vtype in ("state", "rhs") or getattr(value.space, "kind", None) == "state":
            require_history_space(self, name, value.space)
            self._declare_history_block(name, value.block)
        elif name in self._history_spaces or name in self._history_blocks:
            raise ValueError("store_history: cannot store a scalar field in a full-state ring")
        self._histories.setdefault(name, 1)
        return self._new(
            "state", "store_history", (value,), {"history": name}, name, value.block,
            space=value.space)

    def keep_history(self, timestate: Any, depth: Any, cold_start: Any = None,
                     checkpoint_policy: Any = None) -> Any:
        self._guard_mutable("configure state history")
        from pops.time.handles import TimeState
        if not isinstance(timestate, TimeState):
            raise ValueError(
                "keep_history: a TimeState handle is required (P.state('U', block=...))")
        return self._configure_time_history(
            timestate, depth, cold_start, checkpoint_policy)


__all__ = ["_ProgramHistory"]
