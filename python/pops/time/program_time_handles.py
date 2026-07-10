"""Program-owned state and resolution tables for immutable temporal handles."""
from __future__ import annotations

from typing import Any

from pops.time.handles import (
    HistoryHandle, StageHandle, StateEndpointHandle, TimeState,
)
from pops.time.program_value_validation import (
    merge_state_spaces, require_compatible_spaces, require_owned,
)
from pops.time.values import ProgramValue, _Affine, _resolve_handle


class _ProgramTimeHandles:
    """Mutable authoring state behind otherwise immutable temporal handles."""

    def _init_time_handle_tables(self) -> None:
        self._time_states = {}
        self._time_current_values = {}
        self._time_stage_handles = {}
        self._time_stage_values = {}
        self._time_history_handles = {}
        self._time_history_values = {}
        self._time_history_configs = {}
        self._time_history_stores = {}
        self._time_endpoint_handles = {}

    @staticmethod
    def _time_state_key(block: str, state_name: str) -> tuple[str, str]:
        return block, state_name

    def _time_state(self, block: str, state_name: str, space: Any) -> TimeState:
        key = self._time_state_key(block, state_name)
        current = self._time_states.get(key)
        if current is not None:
            require_compatible_spaces(
                current.space, space, "TimeState declaration", typed_pair=True)
            return current
        handle = TimeState(self, block, state_name, space=space)
        self._time_states[key] = handle
        return handle

    def _require_time_state(self, state: Any, where: str) -> TimeState:
        if not isinstance(state, TimeState):
            raise TypeError("%s: a TimeState handle is required" % where)
        if state.owner_path != self.owner_path:
            raise ValueError("%s: the TimeState belongs to a different Program" % where)
        issued = self._time_states.get(self._time_state_key(state.block, state.state_name))
        if issued is not state:
            raise ValueError("%s: the TimeState was not issued by this Program" % where)
        return state

    def _current_time_value(self, state: Any) -> ProgramValue:
        state = self._require_time_state(state, "current state")
        value = self._time_current_values.get(state)
        if value is None:
            value = self._new(
                "state", "state", (), {}, state.block, state.block, space=state.space)
            self._time_current_values[state] = value
        else:
            value = self._canonical_value(value)
            if not self._frozen:
                self._time_current_values[state] = value
        return value

    def _stage_handle(self, state: Any, key: Any) -> StageHandle:
        state = self._require_time_state(state, "stage")
        if not isinstance(key, (int, str)) or isinstance(key, bool):
            raise ValueError("TimeState.stage: key must be an int or a str (got %r)" % (key,))
        cache_key = (state, key)
        handle = self._time_stage_handles.get(cache_key)
        if handle is None:
            self._guard_mutable("declare a temporal stage")
            handle = StageHandle(
                program=self, block=state.block, state_name=state.state_name,
                key=key, space=state.space)
            self._time_stage_handles[cache_key] = handle
        return handle

    def _require_stage(self, handle: Any, where: str) -> StageHandle:
        if not isinstance(handle, StageHandle):
            raise TypeError("%s: a StageHandle is required" % where)
        if handle.owner_path != self.owner_path:
            raise ValueError("%s: the StageHandle belongs to a different Program" % where)
        state = self._time_states.get(
            self._time_state_key(handle.block, handle.state_name))
        issued = self._time_stage_handles.get((state, handle.key)) if state is not None else None
        if issued is not handle:
            raise ValueError("%s: the StageHandle was not issued by this Program" % where)
        return handle

    def _define_stage(self, handle: Any, value: Any) -> ProgramValue:
        self._guard_mutable("define a temporal stage")
        handle = self._require_stage(handle, "T.define")
        if handle in self._time_stage_values:
            raise ValueError("SSA stage already defined")
        resolved = _resolve_handle(value)
        if isinstance(resolved, _Affine):
            if not resolved.terms:
                raise TypeError("T.define stage: an empty affine value is not a State")
            for term, _ in resolved.terms:
                require_owned(self, term, "T.define stage")
                if term.vtype not in ("state", "rhs"):
                    raise TypeError(
                        "T.define stage: every affine term must be a State/Rate value; "
                        "got %s value %r" % (term.vtype, term.name))
            terms = [term for term, _ in resolved.terms]
            candidate_space = merge_state_spaces(terms, "T.define stage")
            blocks = {term.block for term in terms if term.block is not None}
            if len(blocks) > 1 or (blocks and handle.block not in blocks):
                raise ValueError("T.define stage: value belongs to a different block")
        else:
            if not isinstance(resolved, ProgramValue):
                raise TypeError(
                    "T.define stage: expected a State value or an affine combination; got %r"
                    % (resolved,))
            require_owned(self, resolved, "T.define stage")
            if resolved.vtype != "state":
                raise TypeError(
                    "T.define stage: expected a State value, got %s value %r"
                    % (resolved.vtype, resolved.name))
            candidate_space = getattr(resolved, "space", None)
            if getattr(resolved, "block", handle.block) not in (None, handle.block):
                raise ValueError("T.define stage: value belongs to a different block")
        require_compatible_spaces(
            handle.space, candidate_space, "T.define stage", typed_pair=True)
        out = self.define(
            "%s_%s_%s" % (handle.block, handle.state_name, handle.key), value)
        self._time_stage_values[handle] = out
        return out

    def _resolve_time_value(self, handle: Any) -> ProgramValue:
        if isinstance(handle, StageHandle):
            handle = self._require_stage(handle, "stage resolution")
            value = self._time_stage_values.get(handle)
            if value is None:
                raise ValueError(
                    "stage %r is undefined (define it with T.define first)" % handle.key)
            value = self._canonical_value(value)
            if not self._frozen:
                self._time_stage_values[handle] = value
            return value
        if isinstance(handle, HistoryHandle):
            return self._resolve_history_handle(handle)
        raise TypeError("temporal value resolution requires a StageHandle or HistoryHandle")

    def _endpoint_handle(self, state: Any) -> StateEndpointHandle:
        state = self._require_time_state(state, "state endpoint")
        handle = self._time_endpoint_handles.get(state)
        if handle is None:
            self._guard_mutable("declare a state endpoint")
            handle = StateEndpointHandle(
                owner=self.owner_path, block=state.block,
                state_name=state.state_name, space=state.space)
            self._time_endpoint_handles[state] = handle
        return handle

    def _require_endpoint(self, endpoint: Any, where: str) -> StateEndpointHandle:
        if not isinstance(endpoint, StateEndpointHandle):
            raise TypeError("%s: a StateEndpointHandle is required" % where)
        if endpoint.owner_path != self.owner_path:
            raise ValueError("%s: the StateEndpointHandle belongs to a different Program" % where)
        state = self._time_states.get(
            self._time_state_key(endpoint.block, endpoint.state_name))
        issued = self._time_endpoint_handles.get(state) if state is not None else None
        if issued is not endpoint:
            raise ValueError("%s: the StateEndpointHandle was not issued by this Program" % where)
        return endpoint

    def _history_handle(self, state: Any, lag: Any) -> HistoryHandle:
        state = self._require_time_state(state, "history")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("TimeState.prev: lag must be a Python int >= 1 (got %r)" % (lag,))
        cache_key = (state, lag)
        handle = self._time_history_handles.get(cache_key)
        if handle is None:
            self._guard_mutable("declare a history handle")
            handle = HistoryHandle(
                program=self, block=state.block, state_name=state.state_name,
                lag=lag, space=state.space)
            self._time_history_handles[cache_key] = handle
        return handle

    def _require_history(self, handle: Any, where: str) -> tuple[HistoryHandle, TimeState]:
        if not isinstance(handle, HistoryHandle):
            raise TypeError("%s: a HistoryHandle is required" % where)
        if handle.owner_path != self.owner_path:
            raise ValueError("%s: the HistoryHandle belongs to a different Program" % where)
        state = self._time_states.get(
            self._time_state_key(handle.block, handle.state_name))
        issued = self._time_history_handles.get((state, handle.lag)) if state is not None else None
        if issued is not handle:
            raise ValueError("%s: the HistoryHandle was not issued by this Program" % where)
        return handle, state

    def _history_handle_from(self, source: Any, lag: Any) -> HistoryHandle:
        source, state = self._require_history(source, "history")
        self._validate_history_lag(state, lag)
        return self._history_handle(state, lag)

    def _validate_history_lag(self, state: TimeState, lag: Any) -> tuple[Any, ...]:
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("TimeState.prev: lag must be a Python int >= 1 (got %r)" % (lag,))
        config = self._time_history_configs.get(state)
        if config is None:
            raise ValueError(
                "%s.prev requires keep_history first: declare T.keep_history(%s, depth=...) "
                "before reading a lagged state" % (state.block, state.state_name))
        if lag > config[0]:
            raise ValueError(
                "%s.prev(%d) exceeds the kept history depth %d; raise the keep_history depth"
                % (state.block, lag, config[0]))
        return config

    def _resolve_history_handle(self, handle: Any) -> ProgramValue:
        handle, state = self._require_history(handle, "history resolution")
        self._validate_history_lag(state, handle.lag)
        value = self._time_history_values.get(handle)
        if value is None:
            value = self.history(
                "%s.%s" % (state.block, state.state_name), handle.lag,
                space=state.space, block=state.block)
            self._time_history_values[handle] = value
        else:
            value = self._canonical_value(value)
            if not self._frozen:
                self._time_history_values[handle] = value
        return value

    def _configure_time_history(self, state: Any, depth: Any, cold_start: Any,
                                checkpoint_policy: Any) -> ProgramValue:
        self._guard_mutable("configure state history")
        state = self._require_time_state(state, "keep_history")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("keep_history: depth must be a Python int >= 1 (got %r)" % (depth,))
        from pops.time.history import CopyCurrent
        from pops.time.history_persistence import (
            HistoryPersistence, resolve_history_persistence,
        )
        cold_start = CopyCurrent() if cold_start is None else cold_start
        if not isinstance(cold_start, CopyCurrent):
            raise TypeError(
                "keep_history: cold_start must be CopyCurrent(); no other cold-start "
                "policy is implemented by the runtime")
        # Program history configuration owns an immutable descriptor snapshot.  Retaining the
        # caller's mutable object would let a post-validation ``policy.k = ...`` change checkpoint
        # semantics and the Program hash behind the builder's back.
        cold_start = CopyCurrent()
        supplied_policy = resolve_history_persistence(checkpoint_policy)
        policy = HistoryPersistence.from_manifest(supplied_policy.to_manifest())
        policy.validate_for(depth)
        if hasattr(policy, "freeze"):
            policy.freeze()
        prior = self._time_history_configs.get(state)
        config = (depth, cold_start, policy)
        if prior is not None:
            raise ValueError("keep_history: history for %s.%s is already configured"
                             % (state.block, state.state_name))
        name = "%s.%s" % (state.block, state.state_name)
        if name in self._histories_ncomp:
            raise ValueError(
                "keep_history: %r is already a narrow scalar history ring, not a State ring"
                % name)
        existing_depth = self._histories.get(name, 0)
        if existing_depth > depth:
            raise ValueError(
                "keep_history: depth %d is smaller than the already-declared lag %d for %r"
                % (depth, existing_depth, name))
        if name in self._history_spaces:
            require_compatible_spaces(
                self._history_spaces[name], state.space,
                "keep_history ring %r" % name, typed_pair=True)
        if name in self._history_blocks and self._history_blocks[name] != state.block:
            raise ValueError(
                "keep_history: ring %r belongs to block %r, not %r"
                % (name, self._history_blocks[name], state.block))
        # Lower the store before publishing the configuration.  If an existing manual ring has
        # incompatible block/StateSpace provenance, ``store_history`` fails and no temporal
        # configuration is left half-installed on the Program.
        store = self.store_history(name, self._current_time_value(state))
        self._time_history_configs[state] = config
        self._history_persistence[name] = (depth, policy)
        self._time_history_stores[state] = store
        return store

    def _rebuild_time_handle_tables(self, out: Any, idmap: Any, representative: Any) -> None:
        """Recreate owner-qualified handles and remap their ProgramValue resolutions after a pass."""
        state_map = {}
        for old_state in self._time_states.values():
            new_state = out._time_state(
                old_state.block, old_state.state_name, old_state.space)
            state_map[old_state] = new_state
            old_current = self._time_current_values.get(old_state)
            if old_current is not None:
                mapped = idmap.get(representative(old_current).id)
                if mapped is not None:
                    out._time_current_values[new_state] = mapped

        for (old_state, key), old_handle in self._time_stage_handles.items():
            new_handle = out._stage_handle(state_map[old_state], key)
            old_value = self._time_stage_values.get(old_handle)
            if old_value is not None:
                mapped = idmap.get(representative(old_value).id)
                if mapped is not None:
                    out._time_stage_values[new_handle] = mapped

        for (old_state, lag), old_handle in self._time_history_handles.items():
            new_handle = out._history_handle(state_map[old_state], lag)
            old_value = self._time_history_values.get(old_handle)
            if old_value is not None:
                mapped = idmap.get(representative(old_value).id)
                if mapped is not None:
                    out._time_history_values[new_handle] = mapped

        for old_state, (depth, cold_start, _policy) in self._time_history_configs.items():
            new_state = state_map[old_state]
            name = "%s.%s" % (new_state.block, new_state.state_name)
            copied_policy = out._history_persistence[name][1]
            out._time_history_configs[new_state] = (depth, cold_start, copied_policy)
            old_store = self._time_history_stores.get(old_state)
            if old_store is not None:
                mapped = idmap.get(representative(old_store).id)
                if mapped is not None:
                    out._time_history_stores[new_state] = mapped

        for old_state in self._time_endpoint_handles:
            out._endpoint_handle(state_map[old_state])


__all__ = ["_ProgramTimeHandles"]
