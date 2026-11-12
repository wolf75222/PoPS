"""The presentation map of one Case and the paths of its snapshot preimages (N11 lot 2b).

:func:`presentation_map` reads the Case registries in registration order and returns the
:class:`~pops.identity.presentation.PresentationMap` that the snapshot applies to its semantic and
artifact preimages. The name-free digest of a model definition is its module digest without the
module name, so two models that differ only by their name have one projected fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pops.identity.presentation import PresentationMap, projected_fingerprint


# Mappings keyed by block names, and bare owner names, in each preimage (N4: projected keys are
# then ordered by block position, which is the block registration order).
SEMANTIC_KEYS = (("blocks",),)
SEMANTIC_VALUES: dict[tuple[str, ...], str] = {}
ARTIFACT_KEYS = (("payload", "blocks"), ("payload", "numerics"))
ARTIFACT_VALUES = {
    ("payload", "name"): "case",
    ("payload", "options", "name"): "case",
    ("payload", "blocks", "*", "model", "manifest", "name"): "model_of_block",
}
# Lists sorted by qualified id, so by name, are sets: sorted again after the projection. A
# LayoutPlan orders its layouts, assignments and mappings so; a Program its commits; the Case its
# initial conditions.
_LAYOUT_SETS = ("assignments", "layouts", "mappings")
SETS = (
    *(("...", "layout", name) for name in _LAYOUT_SETS),
    *(("...", "projections", "fields", name, "$tuple") for name in _LAYOUT_SETS),
    *(("...", "layout_plan", name) for name in _LAYOUT_SETS),
    *(("...", "layout_plan", "value", name) for name in _LAYOUT_SETS),
    ("...", "time", "commits"),
    ("...", "time", "ir", "commits"),
    ("initials",),
    # Resolved-plan reports and rows ordered by their labels.
    ("...", "coverage", "rows"),
    ("...", "coverage", "source_to_targets", "*"),
    ("...", "coverage", "target_to_sources", "*"),
    ("lowering_coverage", "value", "rows"),
    ("lowering_coverage", "value", "source_to_targets", "*"),
    ("lowering_coverage", "value", "target_to_sources", "*"),
    ("...", "providers", "[]", "sources"),
    ("continuation_transitions", "objects"),
    ("initial_condition_plan", "value", "bindings"),
)
# Digests of named data that the preimage holds in full: the LayoutPlan canonical id, the native
# spatial layout identity, the Program IR hash and the identity of each block's numerics.
DERIVED = (
    ("...", "native_spatial_layout", "identity"),
    ("...", "native_spatial_layout", "$object", "projections", "fields", "identity"),
    ("...", "native_spatial_layout", "$object", "projections", "to_data", "identity"),
    ("...", "time", "hash"),
    ("blocks", "*", "numerics", "identity"),
    ("payload", "numerics", "*", "identity"),
    ("blocks", "[]", "numerics", "value", "identity"),
    ("...", "resolved_operations", "value", "identity"),
    ("...", "time", "_ir_hash"),
    ("native_layouts", "*", "value", "identity"),
)
# The resolved plan names each block, and its continuation rows name a state by its block.
PLAN_VALUES = {
    ("blocks", "[]", "name"): "block",
    ("continuation_transitions", "objects", "[]", "name"): "block_if_listed",
}


def presentation_map(problem: Any, *, layout: Any = None) -> PresentationMap:
    """Return the owner-name positions of ``problem`` (and of the compile ``layout``)."""
    from pops.model.ownership import OwnerKind
    from pops.problem._snapshot_payload import _stabilize_problem_models

    _stabilize_problem_models(problem)
    blocks: list[str] = []
    models: list[tuple[str, str | None, str]] = []
    block_models: list[int] = []
    for name, spec in problem._block_registry.items():
        blocks.append(name)
        model = spec["model"]
        owner = model.owner_path.canonical()
        definitions = [node for node in owner.nodes if node.kind is OwnerKind.MODEL_DEFINITION]
        if len(definitions) != 1:
            raise ValueError("block %r model owner %s has no single model definition"
                             % (name, owner))
        fingerprint = owner.definition_fingerprint
        key = (definitions[0].name, fingerprint)
        index = next((i for i, row in enumerate(models) if row[:2] == key), None)
        if index is None:
            index = len(models)
            models.append((key[0], fingerprint, _name_free_fingerprint(model, fingerprint)))
        block_models.append(index)
    names = PresentationMap(
        case=problem.name,
        blocks=tuple(blocks),
        models=tuple(models),
        block_models=tuple(block_models),
        layouts=_layout_names(layout, tuple(blocks)),
    )
    return PresentationMap(
        case=names.case, blocks=names.blocks, models=names.models,
        block_models=names.block_models, layouts=names.layouts,
        digests=_named_digests(problem, layout, names),
    )


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _named_digests(problem: Any, layout: Any, names: PresentationMap) -> tuple[tuple[str, str], ...]:
    """Digests of named records, paired with the digest of the same record projected.

    An initial condition is identified by the sha256 of its canonical identity, and a LayoutPlan by
    the sha256 of its rows: both hold block and case names, so each digest is replaced by the one
    of its projected record.
    """
    from pops.mesh import LayoutPlan

    rows = []
    for initial in problem._initial_registry.resolved():
        data = initial.canonical_identity()
        rows.append((_sha256_json(data),
                     _sha256_json(names.project(data, sets=SETS, derived=DERIVED))))
    if isinstance(layout, LayoutPlan):
        from pops.mesh._layout_plan_contracts import plan_payload

        data = plan_payload(layout.owner, layout.layouts, layout.assignments, layout.mappings)
        projected = names.project(
            json.loads(json.dumps(data)), sets=tuple(("...", name) for name in _LAYOUT_SETS),
            derived=DERIVED)
        rows.append((layout.canonical_id, _sha256_json(projected)))
    return tuple(dict(rows).items())


def _name_free_fingerprint(model: Any, fingerprint: str | None) -> str:
    """The definition fingerprint of ``model`` with its name left out."""
    for candidate in (getattr(model, "module", None), model):
        hasher = getattr(candidate, "module_hash", None)
        free = getattr(candidate, "_name_free_module_hash", None)
        if callable(hasher) and callable(free) \
                and fingerprint == "pops.module:sha256:%s" % hasher():
            return projected_fingerprint(str(free()))
    raise ValueError(
        "model definition fingerprint %r has no name-free module digest" % (fingerprint,))


def _layout_names(layout: Any, blocks: tuple[str, ...]) -> tuple[str, ...]:
    """Layout names in first use over the blocks (registration order), not in name order.

    A LayoutPlan sorts its layouts by qualified id; a layout's position is instead the position
    of the first block assigned to it, then its subjects' kept local names.
    """
    from pops.mesh import LayoutPlan

    if not isinstance(layout, LayoutPlan):
        return ("default",)

    def block_of(subject: Any) -> int:
        reference = subject if subject.kind == "block" else getattr(subject, "block_ref", None)
        name = getattr(reference, "local_id", None)
        return blocks.index(name) if name in blocks else len(blocks)

    first: dict[str, tuple[int, str, str]] = {}
    for row in layout.assignments:
        key = (block_of(row.subject), row.subject.kind, row.subject.local_id)
        local = row.layout.local_id
        first[local] = min(first.get(local, key), key)
    names = [row.handle.local_id for row in layout.layouts]
    return tuple(sorted(names, key=lambda name: first.get(name, (len(blocks) + 1, "", ""))))


__all__ = [
    "ARTIFACT_KEYS", "ARTIFACT_VALUES", "DERIVED", "PLAN_VALUES", "SEMANTIC_KEYS", "SEMANTIC_VALUES", "SETS",
    "presentation_map",
]
