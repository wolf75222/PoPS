"""One frozen identity for scalar histories used exclusively for inspection/export."""
from __future__ import annotations

import json
from typing import Any


SCALAR_OUTPUT_HISTORY_SPACE = "scalar-output-field-v1"
OUTPUT_HISTORY_PROJECTION_SPACE = "pops.program.scalar-output-history-projection@2"


def _stored_value_ghost_depth(program: Any, value: Any) -> int | str:
    """Mirror the stored producer's allocation, preserving in-place aliases."""
    seen: set[int] = set()
    while True:
        value = program._canonical_value(value)
        if id(value) in seen:
            raise ValueError("history storage contains a cyclic allocation alias")
        seen.add(id(value))
        if value.op == "pointwise_expression" and "field_product_seed" in value.attrs:
            from .program_emit_expressions import pointwise_output_template

            # The typed product has its own exact width and a valid-cell-only
            # allocation; its State input supplies layout authority, not its halo.
            pointwise_output_template(value)
            return 0
        if value.op in ("acceptance_guard", "fill_boundary", "project", "store_history"):
            if not value.inputs:
                raise ValueError("history storage alias requires its allocation source")
            value = value.inputs[0]
            continue
        if value.op == "history":
            source = value.attrs.get("history")
            if type(source) is not str or source not in program._histories:
                raise ValueError("history storage read requires its declared source ring")
            return source
        # Ordinary scalar allocations, including a named linear combination of
        # a typed product, explicitly allocate one ghost cell in the emitter.
        return 1


def history_ghost_depth(program: Any, name: str) -> int:
    """Declare scalar storage independently from its owning State's stencil halo.

    Generic scalar allocations carry one ghost cell; typed product seeds carry
    none. Full State histories keep the owner's native halo, including a scalar
    State with an explicit Space. Every store must agree on the exact allocation.
    """
    if (getattr(program, "_history_spaces", {}).get(name) is not None or
            getattr(program, "_history_state_refs", {}).get(name) is not None):
        return -1
    width = getattr(program, "_histories_ncomp", {}).get(name)
    if width is None:
        return -1
    if type(width) is not int or width < 1:
        raise ValueError("scalar history requires its declared positive component width")
    # A read is the source ring's exact MultiFab, rather than a new scalar
    # allocation. Copying it therefore constrains both rings to the same halo.
    # Solve these equality constraints as finite connected components: a typed
    # allocation can anchor a whole cycle, while unanchored read-first cycles
    # retain their historical one-ghost cold-start allocation.
    names = {name, *getattr(program, "_histories", {}),
             *getattr(program, "_histories_ncomp", {})}
    edges: dict[str, set[str]] = {ring: set() for ring in names}
    anchors: dict[str, set[int]] = {ring: set() for ring in names}
    for ring in names:
        if (getattr(program, "_history_spaces", {}).get(ring) is not None or
                getattr(program, "_history_state_refs", {}).get(ring) is not None or
                getattr(program, "_histories_ncomp", {}).get(ring) is None):
            anchors[ring].add(-1)
    seen: set[int] = set()
    stack = [*getattr(program, "_values", ()),
             *getattr(program, "_commits", {}).values()]
    while stack:
        value = program._canonical_value(stack.pop())
        if id(value) in seen:
            continue
        seen.add(id(value))
        if value.op == "store_history":
            ring = value.attrs.get("history")
            if type(ring) is not str or ring not in edges:
                raise ValueError("history storage requires its declared destination ring")
            if len(value.inputs) != 1:
                raise ValueError("history storage requires one exact stored field")
            if anchors[ring] != {-1}:
                allocation = _stored_value_ghost_depth(program, value.inputs[0])
                if isinstance(allocation, str):
                    edges[ring].add(allocation)
                    edges[allocation].add(ring)
                else:
                    anchors[ring].add(allocation)
        stack.extend(value.inputs)
        stack.extend(program._subblock_value_refs(value))
    depths: set[int] = set()
    visited: set[str] = set()
    pending = [name]
    while pending:
        ring = pending.pop()
        if ring in visited:
            continue
        visited.add(ring)
        depths.update(anchors[ring])
        pending.extend(edges[ring] - visited)
    if len(depths) > 1:
        raise ValueError("history %r stores disagree on their exact ghost depth" % name)
    return next(iter(depths), 1)


def history_space_identity(program: Any, name: str) -> str:
    """Qualify a typed store-only ring; preserve the legacy width-one/two-slot profile.

    This is a native transfer capability, so follow aliases, expression inputs and
    every recorded subblock exactly as Program liveness does. A consumer hidden in
    a matrix-free apply or branch must retain the ordinary history contract.
    """
    space = getattr(program, "_history_spaces", {}).get(name)
    if space is not None:
        return json.dumps(space.to_data(), sort_keys=True, separators=(",", ":"))
    owner = getattr(program, "_history_blocks", {}).get(name)
    width = getattr(program, "_histories_ncomp", {}).get(name)
    lag = program._histories.get(name)
    if (owner is None or getattr(program, "_history_state_refs", {}).get(name) is not None
            or type(width) is not int or width < 1 or type(lag) is not int or lag < 1):
        return "scalar-field"

    seen: set[int] = set()
    stack = [*program._values, *program._commits.values()]
    stored = False
    while stack:
        value = program._canonical_value(stack.pop())
        if id(value) in seen:
            continue
        seen.add(id(value))
        if value.attrs.get("history") == name:
            if (value.op != "store_history" or len(value.inputs) != 1
                    or value.inputs[0].vtype != "scalar_field" or value.block != owner):
                return "scalar-field"
            stored = True
        stack.extend(value.inputs)
        stack.extend(program._subblock_value_refs(value))
    if not stored:
        return "scalar-field"
    # IR16 stores authenticate the physical observation independently of their
    # allocation State. Keep that exact descriptor in state_identity; this URI
    # qualifies only its spatial transfer, never a State/Q-to-observation copy.
    from pops.time._program.global_history_storage import descriptor

    storage = descriptor(program, name)
    if width == 1 and lag == 1 and storage is None:
        return SCALAR_OUTPUT_HISTORY_SPACE
    return OUTPUT_HISTORY_PROJECTION_SPACE
