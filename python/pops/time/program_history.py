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
                space: Any = None, block: Any = None, state_ref: Any = None) -> Any:
        self._guard_mutable("declare a history read")
        if not isinstance(name, str) or not name:
            raise ValueError("history: name must be a non-empty string")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("history: lag must be a Python int >= 1 (got %r)" % (lag,))
        if block is not None:
            from pops.problem.handles import BlockHandle
            if not isinstance(block, BlockHandle):
                raise TypeError("history: block must be a BlockHandle or None")
        if ncomp is None:
            if space is None and name in self._history_spaces:
                space = self._history_spaces[name]
            if block is None and name in self._history_blocks:
                block = self._history_blocks[name]
            require_history_space(self, name, space)
            self._declare_history_block(name, block)
            self._declare_history_state(name, state_ref, block)
            self._histories[name] = max(self._histories.get(name, 0), lag)
            return self._new(
                "state", "history", (),
                {"history": name, "lag": int(lag), "state": state_ref}, name, block,
                space=space, state_ref=state_ref)
        if space is not None or state_ref is not None:
            raise ValueError("history: a narrow scalar ring has no StateSpace/state provenance")
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp != 1:
            raise ValueError(
                "history: explicit ncomp must be 1; omit it for a full-state history")
        self._histories[name] = max(self._histories.get(name, 0), lag)
        self._histories_ncomp[name] = 1
        self._declare_history_block(name, block)
        return self._new(
            "scalar_field", "history", (), {"history": name, "lag": int(lag), "ncomp": 1},
            name, block)

    def _declare_history_block(self, name: str, block: Any) -> None:
        missing = object()
        prior = self._history_blocks.get(name, missing)
        if prior is missing:
            self._history_blocks[name] = block
        elif prior != block:
            raise ValueError(
                "history %r cannot mix block provenance %r and %r" % (name, prior, block))

    def _declare_history_state(self, name: str, state_ref: Any, block: Any) -> None:
        from pops.model.handles import Handle
        if not isinstance(state_ref, Handle) or state_ref.kind != "state" \
                or not state_ref.is_instance:
            raise TypeError(
                "history: a full-state ring requires a block-qualified state Handle")
        if state_ref.block_ref is not block:
            raise ValueError("history: state_ref belongs to a different block")
        missing = object()
        prior = self._history_state_refs.get(name, missing)
        if prior is missing:
            self._history_state_refs[name] = state_ref
        elif prior != state_ref:
            raise ValueError(
                "history %r cannot mix state declarations %s and %s"
                % (name, prior.qualified_id, state_ref.qualified_id))

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
            self._declare_history_state(name, value.state_ref, value.block)
        elif name in self._history_spaces:
            raise ValueError("store_history: cannot store a scalar field in a full-state ring")
        elif name in self._history_blocks and self._history_blocks[name] != value.block:
            raise ValueError("store_history: scalar history block provenance mismatch")
        self._histories.setdefault(name, 1)
        return self._new(
            "state", "store_history", (value,),
            {"history": name, "state": value.state_ref}, name, value.block,
            space=value.space, state_ref=value.state_ref)

    def keep_history(self, timestate: Any, depth: Any, cold_start: Any = None,
                     checkpoint_policy: Any = None) -> Any:
        self._guard_mutable("configure state history")
        from pops.time.handles import TimeState
        if not isinstance(timestate, TimeState):
            raise ValueError(
                "keep_history: a TimeState handle is required (T.state(block, U))")
        return self._configure_time_history(
            timestate, depth, cold_start, checkpoint_policy)


__all__ = ["_ProgramHistory"]
