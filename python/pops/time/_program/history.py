"""Typed persistent-history authoring for time Programs."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pops.time._program.value_validation import require_history_space, require_top_level
from pops.time.values import _is_field_value

if TYPE_CHECKING:
    from pops.time._program.contract import _ProgramBase
else:
    _ProgramBase = object


class _ProgramHistory(_ProgramBase):
    """Full-state and narrow scalar history rings with explicit provenance."""

    @staticmethod
    def _snapshot_history_persistence(checkpoint_policy: Any, ring_slots: int) -> Any:
        """Return one validated, clone-owned and frozen persistence descriptor."""
        from pops.time._history.persistence import (
            HistoryPersistence,
            resolve_history_persistence,
        )

        supplied = resolve_history_persistence(checkpoint_policy)
        policy = HistoryPersistence.from_manifest(supplied.to_manifest())
        policy.validate_for(ring_slots)
        if hasattr(policy, "freeze"):
            policy.freeze()
        return policy

    def _has_history_store(self, name: str) -> bool:
        """Whether the authoring graph already contains a store for ``name``."""
        def _walk(values: Any) -> Any:
            for value in values:
                yield value
                attrs = getattr(value, "attrs", {}) or {}
                for key in (
                    "cond_block", "body_block", "true_block", "false_block",
                    "apply_block", "residual_block",
                ):
                    block = attrs.get(key)
                    if block:
                        yield from _walk(block)

        return any(
            value.op == "store_history" and value.attrs.get("history") == name
            for value in _walk(self._values)
        )

    def _policy_for_history_read(self, name: str, ring_slots: int) -> Any:
        """Materialize Dense for a new ring or validate it at the physical slot count."""
        configured = self._history_persistence.get(name)
        if configured is None:
            return self._snapshot_history_persistence(None, ring_slots)
        _configured_slots, policy = configured
        policy.validate_for(ring_slots)
        return policy

    def _policy_for_history_store(
        self,
        name: str,
        ring_slots: int,
        checkpoint_policy: Any,
    ) -> Any:
        """Resolve one store policy without allowing order-dependent reconfiguration."""
        requested = self._snapshot_history_persistence(checkpoint_policy, ring_slots)
        configured = self._history_persistence.get(name)
        if configured is None:
            return requested
        _configured_slots, current = configured
        if current.to_manifest() == requested.to_manifest():
            current.validate_for(ring_slots)
            return current

        # A read may precede the first store. It materializes the documented Dense default so the
        # compiled ring contract is complete, but the first store remains the authority allowed to
        # select a different typed policy. Once a store exists, changing policy is order-dependent
        # authoring and is refused.
        from pops.time._history.persistence import Dense

        if not self._has_history_store(name) and isinstance(current, Dense):
            return requested
        raise ValueError(
            "store_history: persistence policy for %r is already %s and cannot change to %s"
            % (name, current.name, requested.name)
        )

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
            max_lag = max(self._histories.get(name, 0), lag)
            ring_slots = max_lag + 1
            policy = self._policy_for_history_read(name, ring_slots)
            value = self._new(
                "state", "history", (),
                {"history": name, "lag": int(lag), "state": state_ref}, name, block,
                space=space, state_ref=state_ref)
            self._histories[name] = max_lag
            self._history_persistence[name] = (ring_slots, policy)
            return value
        if space is not None or state_ref is not None:
            raise ValueError("history: a narrow scalar ring has no StateSpace/state provenance")
        if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp != 1:
            raise ValueError(
                "history: explicit ncomp must be 1; omit it for a full-state history")
        max_lag = max(self._histories.get(name, 0), lag)
        ring_slots = max_lag + 1
        policy = self._policy_for_history_read(name, ring_slots)
        self._declare_history_block(name, block)
        value = self._new(
            "scalar_field", "history", (), {"history": name, "lag": int(lag), "ncomp": 1},
            name, block)
        self._histories[name] = max_lag
        self._histories_ncomp[name] = 1
        self._history_persistence[name] = (ring_slots, policy)
        return value

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

    def store_history(
        self,
        name: Any,
        value: Any,
        *,
        depth: Any = None,
        checkpoint_policy: Any = None,
    ) -> Any:
        """Store a ring value with one typed, immutable checkpoint-persistence policy.

        ``depth`` declares the maximum readable lag before reads are authored; the physical ring has
        ``depth + 1`` slots because slot zero is the current value. It is required when a non-Dense
        policy is selected before any read has established that maximum lag, preventing transient
        authoring order from validating a selective policy against the wrong physical slot count.
        """
        return self._store_history(
            name,
            value,
            checkpoint_policy=checkpoint_policy,
            history_depth=depth,
        )

    def _store_history(
        self,
        name: Any,
        value: Any,
        *,
        checkpoint_policy: Any,
        history_depth: Any,
    ) -> Any:
        self._guard_mutable("store history")
        if not isinstance(name, str) or not name:
            raise ValueError("store_history: name must be a non-empty string")
        if not _is_field_value(value):
            raise ValueError("store_history: value must be a State/RHS field (got %r)" % (value,))
        require_top_level(self, value, "store_history")
        declared_lag = self._histories.get(name, 0)
        if history_depth is None:
            from pops.time._history.persistence import Dense, resolve_history_persistence

            supplied = resolve_history_persistence(checkpoint_policy)
            if declared_lag == 0 and not isinstance(supplied, Dense):
                raise ValueError(
                    "store_history: a non-Dense checkpoint_policy requires depth= when no "
                    "history read has declared the final ring depth"
                )
            max_lag = max(declared_lag, 1)
        else:
            if isinstance(history_depth, bool) or not isinstance(history_depth, int) \
                    or history_depth < 1:
                raise ValueError("store_history: maximum history lag must be an int >= 1")
            if declared_lag > history_depth:
                raise ValueError(
                    "store_history: configured depth %d is smaller than declared lag %d for %r"
                    % (history_depth, declared_lag, name)
                )
            max_lag = history_depth
        ring_slots = max_lag + 1
        policy = self._policy_for_history_store(name, ring_slots, checkpoint_policy)
        if value.vtype in ("state", "rhs") or getattr(value.space, "kind", None) == "state":
            require_history_space(self, name, value.space)
            self._declare_history_block(name, value.block)
            self._declare_history_state(name, value.state_ref, value.block)
        elif name in self._history_spaces:
            raise ValueError("store_history: cannot store a scalar field in a full-state ring")
        elif name in self._history_blocks and self._history_blocks[name] != value.block:
            raise ValueError("store_history: scalar history block provenance mismatch")
        node = self._new(
            "state", "store_history", (value,),
            {"history": name, "state": value.state_ref}, name, value.block,
            space=value.space, state_ref=value.state_ref)
        self._histories[name] = max_lag
        self._history_persistence[name] = (ring_slots, policy)
        return node

    def keep_history(self, timestate: Any, depth: Any, cold_start: Any = None,
                     checkpoint_policy: Any = None) -> Any:
        self._guard_mutable("configure state history")
        from pops.time.handles import TimeState
        if not isinstance(timestate, TimeState):
            raise ValueError(
                "keep_history: a TimeState handle is required (T.state(block[U]))")
        return self._configure_time_history(
            timestate, depth, cold_start, checkpoint_policy)


__all__ = ["_ProgramHistory"]
