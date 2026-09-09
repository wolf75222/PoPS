"""Fail-closed slicing of a separable Program into per-layout compiled authorities."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


_UNSLICEABLE_OPS = frozenset({
    "solve_coupled_implicit", "solve_fields_from_blocks",
    "field_solve_from_blocks", "while", "branch", "range", "subcycle",
    "solve_local_nonlinear", "solve_spatial_nonlinear",
})


def _block_id(value: Any) -> str | None:
    block = getattr(value, "block", None)
    if block is None:
        block = getattr(value, "block_ref", None)
    local = getattr(block, "local_id", None)
    return local if isinstance(local, str) and local else None


def _block_ids(value: Any, seen: set[int] | None = None) -> frozenset[str]:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return frozenset()
    seen.add(identity)
    block = _block_id(value)
    if block is not None:
        return frozenset((block,))
    if isinstance(value, Mapping):
        values = tuple(value.keys()) + tuple(value.values())
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = tuple(value)
    else:
        return frozenset()
    return frozenset(
        block_id for item in values for block_id in _block_ids(item, seen))


def slice_program(program: Any, block_names: Any) -> Any:
    """Return an independent Program whose graph is the exact selected-block closure.

    This is intentionally narrower than arbitrary graph partitioning.  Every selected node must be
    block-local and all of its dataflow dependencies must remain in the same partition.  Unsupported
    control/coupled constructs are rejected before compilation instead of being approximated.
    """
    from pops.time import Program

    if type(program) is not Program:
        raise TypeError("slice_program requires an exact pops.Program")
    selected = frozenset(block_names)
    if not selected or any(not isinstance(name, str) or not name for name in selected):
        raise TypeError("slice_program block_names must contain non-empty names")
    if program._dt_bound is not None or program._acceptance_guards:
        raise ValueError(
            "multi-layout Program slicing does not support dt bounds or guards")
    if set(program._histories) - set(program._history_state_refs):
        raise ValueError("multi-layout histories require exact qualified state ownership")
    values = tuple(program._values)
    unsupported = sorted({value.op for value in values if value.op in _UNSLICEABLE_OPS})
    if unsupported:
        raise ValueError("multi-layout Program contains unsliceable operation(s) %s" % unsupported)
    roots = [value for value in values if _block_id(value) in selected]
    if not roots:
        raise ValueError("layout Program partition contains no executable nodes")
    keep: set[int] = set()
    pending = list(roots)
    while pending:
        value = pending.pop()
        if value.id in keep:
            continue
        block = _block_id(value)
        if block is not None and block not in selected:
            raise ValueError(
                "layout Program partition reads block %r outside its selected layout" % block)
        keep.add(value.id)
        pending.extend(getattr(value, "inputs", ()))
        # Matrix-free apply regions can capture block-owned coefficient values.
        # Their inputs are part of the partition closure, even when the operator
        # itself has no direct inputs.
        for key in ("apply_block", "residual_block", "body_block"):
            pending.extend(value.attrs.get(key, ()))
    selected_commits = {
        state: value for state, value in program._commits.items()
        if getattr(getattr(state, "block_ref", None), "local_id", None) in selected
    }
    if not selected_commits:
        raise ValueError("layout Program partition commits no state")
    if any(value.id not in keep for value in selected_commits.values()):
        raise ValueError("layout Program commit is outside the selected dataflow closure")

    registry_owners = frozenset(
        owner
        for value in values if value.id in keep
        for owner in (getattr(getattr(value, "block", None), "model_owner_path", None),)
        if owner is not None
    )

    def keep_state(state: Any) -> bool:
        return _block_id(state) in selected

    # A frozen Program owns MappingProxyType tables and intentionally cannot be deep-copied. The
    # lossless rebuild engine instead re-owns only this dependency closure, renumbers its SSA ids,
    # reconstructs temporal handles, and projects model registries/state tables in one operation.
    clone = program._rebuild(
        lambda value: value.id in keep,
        state_keep=keep_state,
        registry_keep=lambda owner: owner in registry_owners,
        transformation="normalize",
    )
    from pops.codegen.program_emit_field_routes import _walk_program_nodes
    all_values = tuple(_walk_program_nodes(tuple(clone._values)))
    all_ids = sorted({value.id for value in all_values})
    if all_ids != list(range(clone._next_id)):
        raise RuntimeError("sliced Program SSA identity is not a closed contiguous partition")
    if clone._recording:
        raise RuntimeError("sliced Program retained foreign authoring regions")
    active_regions = {value.region for value in all_values}
    if any(region not in active_regions or any(value.id not in all_ids for value in block)
           for block, region in clone._recording_regions.values()):
        raise RuntimeError("sliced Program retained a foreign matrix-free recording region")
    if any(destination not in active_regions or any(source not in active_regions for source in sources)
           for destination, sources in clone._region_imports.items()):
        raise RuntimeError("sliced Program retained a foreign region import")
    for name in (
        "_time_states", "_time_current_values", "_time_stage_handles", "_time_stage_values",
        "_time_history_handles", "_time_history_values", "_time_history_configs",
        "_time_history_stores", "_time_endpoint_handles",
    ):
        table = getattr(clone, name)
        if not isinstance(table, Mapping):
            raise TypeError("Program time-handle table %s is not a mapping" % name)
        retained = _block_ids(table)
        if not retained <= selected:
            raise RuntimeError(
                "sliced Program %s retained foreign block(s) %s"
                % (name, sorted(retained - selected)))
    if _block_ids(clone._commits) - selected or _block_ids(clone._state_spaces) - selected:
        raise RuntimeError("sliced Program retained foreign state authority")
    if _block_ids(clone._history_state_refs) - selected or _block_ids(clone._history_blocks) - selected:
        raise RuntimeError("sliced Program retained foreign history authority")
    expected_histories = {name for name, state in program._history_state_refs.items()
                          if _block_id(state) in selected}
    if set(clone._histories) != expected_histories:
        raise RuntimeError("sliced Program histories differ from their exact state partition")
    if set(clone._operator_registries) - set(registry_owners):
        raise RuntimeError("sliced Program retained a foreign operator registry")
    clone.validate()
    actual = frozenset(getattr(block, "local_id", None) for block in clone._block_indices())
    if actual != selected:
        raise ValueError(
            "sliced Program block routes differ from requested partition: requested=%s actual=%s"
            % (sorted(selected), sorted(actual, key=repr)))
    clone.freeze()
    return clone


__all__ = ["slice_program"]
