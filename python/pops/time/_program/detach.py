"""Detach a compiled time Program from every live authoring authority.

Compilation consumes registries to authenticate semantic references.  A compiled artifact must not
retain those registries (or their mutable builders) afterwards.  ``detach_compiled_program`` rebuilds
the complete SSA graph with clone-owned ``ProgramValue`` objects, interns canonical registry-free
handles, rebuilds every temporal table against the clone, drops operator registries and deep-freezes
the result.  The source Program is never mutated.
"""
from __future__ import annotations

import json
from typing import Any


class _DetachedReferences:
    """Canonical, registry-free Handle interner for one detached Program."""

    def __init__(self) -> None:
        self._by_identity: dict[str, Any] = {}

    def __call__(self, value: Any) -> Any:
        from pops.model.handles import Handle

        if value is None or not isinstance(value, Handle):
            return value
        from pops.time.handles import (
            HistoryHandle,
            StageHandle,
            StateEndpointHandle,
            TimeState,
        )

        if isinstance(value, (TimeState, StageHandle, HistoryHandle, StateEndpointHandle)):
            raise TypeError(
                "temporal handles must be rebuilt by their detached Program tables, not retained "
                "inside ProgramValue metadata"
            )

        from pops.time.references import canonical_handle

        canonical = canonical_handle(value)
        data = canonical.canonical_identity()
        key = json.dumps(data, sort_keys=True, separators=(",", ":"))
        existing = self._by_identity.get(key)
        if existing is not None:
            return existing

        declaration = (
            self(canonical.declaration_ref)
            if canonical.declaration_ref is not None
            else None
        )
        block = self(canonical.block_ref) if canonical.block_ref is not None else None
        detached = canonical._with_owner(
            canonical.owner_path,
            declaration_ref=declaration,
            block_ref=block,
        )

        # Concrete Problem handles carry their issuing registry only as an authoring capability.
        # Canonical identity has already authenticated them; the compiled value must retain no route
        # back into that registry graph, even if the input was an unusual already-resolved handle.
        from pops.problem.handles import BlockHandle, FieldHandle

        if isinstance(detached, BlockHandle):
            object.__setattr__(detached, "model_owner_path", detached.model_owner_path.canonical())
            object.__setattr__(detached, "_instance_registry", None)
        if isinstance(detached, FieldHandle):
            object.__setattr__(detached, "_field_registry", None)
        if detached.canonical_identity() != data:
            raise RuntimeError("detached Handle changed canonical semantic identity")
        self._by_identity[key] = detached
        return detached


def detach_compiled_program(program: Any) -> Any:
    """Return a deeply frozen Program clone with no live authoring graph.

    This is the one post-compilation boundary consumed by codegen/orchestration.  It accepts the
    Program that was successfully lowered, rebuilds every ``ProgramValue`` so ``value.prog`` points
    at the clone, canonicalises/interns block, state, field, parameter and operator handles, rebuilds
    temporal/history/field-provenance tables, removes operator registries/default authoring caches,
    and freezes the clone recursively.  The canonical serialization and IR hash must remain exactly
    identical; a mismatch is a hard internal error.
    """
    from pops.time._program.contract import require_program

    require_program(program, exact=False, where="detach_compiled_program")
    if getattr(program, "_recording", ()):
        raise RuntimeError(
            "detach_compiled_program cannot detach an incomplete active authoring sub-block"
        )

    expected_hash = program._ir_hash()
    references = _DetachedReferences()
    detached = program._rebuild(
        lambda _value: True,
        reference_of=references,
        retain_operator_registries=False,
        canonical_owner=True,
    )
    object.__setattr__(detached, "_compiled_detached", True)
    detached.freeze()

    actual_hash = detached._ir_hash()
    if actual_hash != expected_hash:
        raise RuntimeError(
            "detach_compiled_program changed Program IR identity (%s -> %s)"
            % (expected_hash, actual_hash)
        )
    if detached._operator_registries:
        raise RuntimeError("detached Program unexpectedly retained operator registries")
    return detached
