"""The presentation map of one Case and the paths of its snapshot preimages (N11 lot 2b).

:func:`presentation_map` reads the Case registries in registration order and returns the
:class:`~pops.identity.presentation.PresentationMap` that the snapshot applies to its semantic and
artifact preimages. The name-free digest of a model definition is its module digest without the
module name, so two models that differ only by their name have one projected fingerprint.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
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
    ("layout", "value", "providers"),
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
    ("continuation_transitions", "objects", "[]", "name"): "continuation_name",
    ("continuation_transitions", "objects", "[]", "identity"): "continuation_identity",
    ("field_plans", "*", "value", "native_options", "provider_pack", "[]", "owner_block"): "block",
    ("field_plans", "*", "value", "native_options", "provider_identity",
     "contributions", "[]", "owner_block"): "block",
    ("...", "output_route", "owner_block"): "block",
    ("...", "output_route", "publication", "block"): "block",
}

# String references from serialized contracts. Owner and handle records carry their own
# types. The flattened fields below are named by their enclosing schema, so an author name
# in a Space, an IR literal, or a user dictionary is never interpreted as a reference.
REFERENCE_KEYS = (
    ("layout_targets",), ("native_layouts",), ("layout_amr_authorities",),
    *(("...", "coverage", index) for index in ("source_to_targets", "target_to_sources")),
    *(("lowering_coverage", "value", index)
      for index in ("source_to_targets", "target_to_sources")),
)
REFERENCES = (
    # Provider and block ownership.
    ("...", "guarantees", "block_instance"),
    ("blocks", "[]", "instance_owner_qid"),
    ("...", "native_provider_instance", "definition_owner_qid"),
    ("...", "native_provider_instance", "instance_owner_qid"),
    ("blocks", "[]", "state_identities", "[]"),
    ("...", "entries", "[]", "provider", "producer"),
    ("...", "consumer_plans", "*", "[]", "provider", "producer"),
    ("...", "output_route", "owner_identity"),
    # Manifest and resolved operation records.
    ("...", "state_spaces", "*", "qid"),
    ("...", "field_spaces", "*", "qid"),
    ("...", "model", "manifest", "aux", "*", "qid"),
    ("...", "operators", "[]", "qid"),
    ("...", "operator_bindings", "[]", "subject_qid"),
    ("...", "operator_bindings", "[]", "target_qid"),
    ("...", "operations", "[]", "identity"),
    ("...", "operations", "[]", "evaluation"),
    *(("...", "operations", "[]", field, "[]") for field in (
        "consumes", "dependencies", "materialized_inputs", "communication")),
    ("...", "operations", "[]", "inputs", "[]", "reference"),
    ("...", "guarantees", "declaration_operation"),
    ("...", "evaluations", "[]", "identity"),
    ("...", "occurrences", "[]", "identity"),
    ("...", "occurrences", "[]", "operator"),
    ("...", "occurrences", "[]", "target"),
    ("...", "exchanges", "[]", "occurrence"),
    ("...", "exchanges", "[]", "target"),
    ("...", "conditions", "[]", "stencil", "rates", "[]"),
    ("...", "resolved_operations", "value", "execution_order", "[]"),
    # Layout and its named digest, initial condition and its derived provider handle.
    ("...", "native_spatial_layout", "layout_id"),
    ("...", "native_spatial_layout", "$object", "projections", "to_data", "layout_id"),
    ("...", "native_spatial_layout", "$object", "projections", "fields", "layout_id"),
    ("native_layouts", "*", "value", "layout_id"),
    ("layout_amr_authorities", "*", "value", "layout_id"),
    ("layout", "value", "layout_plan"),
    ("...", "quantities", "[]", "layout_id"),
    ("...", "diagnostic_quantities", "[]", "layout_id"),
    *((plan, "value", "layout_plan_id")
      for plan in ("initial_condition_plan", "bootstrap_plan", "amr_transfer")),
    ("...", "authorities", "initial_conditions", "layout_plan_id"),
    ("...", "authorities", "bootstrap", "layout_plan_id"),
    ("...", "authorities", "transfer", "layout_plan_id"),
    ("...", "layout_plan", "value", "canonical_id"),
    ("...", "layout_plan", "value", "qualified_id"),
    ("...", "layout", "$object", "projections", "fields", "canonical_id"),
    ("...", "component_digests", "module"),
    ("...", "model", "module_hash"),
    ("...", "model", "scientific_hash"),
    ("...", "resolved_operations", "value", "source_module_hash"),
    # Coverage report relations and capability providers.
    ("...", "coverage", "rows", "[]", "source"),
    ("...", "coverage", "rows", "[]", "targets", "[]"),
    ("lowering_coverage", "value", "rows", "[]", "source"),
    ("lowering_coverage", "value", "rows", "[]", "targets", "[]"),
    ("capabilities", "resolution", "providers", "[]", "sources", "[]"),
    ("...", "method_provider", "resolution", "evidence", "[]", "source"),
    ("...", "method_provider", "resolution", "evidence", "[]", "targets", "[]"),
    *(path + ("*", "[]") for path in REFERENCE_KEYS if path[-1].endswith(("_sources", "_targets"))),
)


def project_retained_auxiliaries(
    rows: list[dict[str, Any]], blocks: Any, names: PresentationMap,
) -> None:
    """Rebuild auxiliary identifiers from the projected owners in a copied transition plan."""
    from pops.runtime._continuation_transitions import retained_auxiliary_component_identity

    owners = {block.name: block.instance_owner_qid for block in blocks}
    for row in rows:
        if row["kind"] != "auxiliary":
            continue
        owner = owners[row["name"]]
        component = row["validity"]
        if row["identity"] != retained_auxiliary_component_identity(owner, component):
            raise ValueError("auxiliary continuation identity differs from its block and component")
        row["identity"] = retained_auxiliary_component_identity(
            names.project_owner_string(owner), names.project(component))


def ghost_presentation_map(names: PresentationMap, blocks: Any, layout_plan: Any) -> PresentationMap:
    """Extend owner projection with identifiers made by exact ghost composers."""
    from pops.identity import make_identity
    from pops.identity.semantic import semantic_identity
    from pops.mesh.boundaries.composition import GhostPresentationOrigins
    from pops.mesh.boundaries.ghost_plan import GhostProducerPlan

    rows = []
    layout_digests = dict(names.digests)
    for block in blocks:
        if block.numerics is None:
            continue
        for boundary in block.numerics.boundaries:
            if type(boundary) is not GhostProducerPlan:
                continue
            origins = boundary._presentation_origins
            if type(origins) is not GhostPresentationOrigins:
                continue
            if str(origins.numerics.block.instance_owner_path) != block.instance_owner_qid \
                    or origins.layout_plan.canonical_id != layout_plan.canonical_id:
                raise ValueError("ghost composition evidence belongs to another block or layout plan")
            numeric = origins.numerics._payload()
            if semantic_identity(numeric) != origins.numerics.identity:
                raise ValueError("uncomposed numerical plan differs from its ghost authority")
            projected_numeric = semantic_identity(names.project(
                numeric, sets=SETS, derived=DERIVED, references=REFERENCES,
                reference_keys=REFERENCE_KEYS)).to_data()
            named_numeric = origins.numerics.identity.to_data()
            layout_id = origins.layout_plan.canonical_id
            named_transfer = projected_transfer = None
            if origins.amr_transfer is not None:
                transfer = origins.amr_transfer
                data = transfer.canonical_identity()
                if make_identity("amr-transfer", data) != transfer.identity:
                    raise ValueError("AMR ghost transfer evidence differs from its authority")
                named_transfer = transfer.identity.to_data()
                projected_transfer = make_identity("amr-transfer", names.project(
                    data, references=(*REFERENCES, ("layout_plan_id",)))).to_data()
            for origin in origins.handles:
                if origin.handle.owner_path != boundary.topology.owner:
                    raise ValueError("ghost handle belongs to another boundary owner")
                evidence = origin.named_evidence()
                if evidence == named_numeric:
                    projected = projected_numeric
                elif evidence == named_transfer:
                    projected = projected_transfer
                else:
                    projected = names.project(
                        evidence, sets=SETS, derived=DERIVED, references=REFERENCES,
                        reference_keys=REFERENCE_KEYS)
                    if isinstance(evidence, dict) and evidence.get("numerics") == named_numeric:
                        projected["numerics"] = projected_numeric
                    if isinstance(evidence, dict) and evidence.get("plan") == layout_id:
                        projected["plan"] = layout_digests[layout_id]
                    if isinstance(evidence, dict) and "transfer" in evidence:
                        if evidence["transfer"] != named_transfer:
                            raise ValueError("ghost handle references another AMR transfer")
                        projected["transfer"] = projected_transfer
                rows.append((origin.handle.qualified_id, origin.projected_local_id(projected)))
    return names.with_derived_handles(rows)


def project_field_install_references(
    names: PresentationMap, payload: dict[str, Any], field_plans: Any,
) -> PresentationMap:
    """Authenticate and reproject identities carried by resolved field installation rows."""
    from pops.codegen.field_install import field_provider_target
    from pops.codegen.field_install_plan import field_output_target
    from pops.fields._identity import field_identity, strict_field_data
    from pops.identity import canonical_bytes

    slots: dict[str, str] = {}
    labels = {}
    for name, plan in field_plans.items():
        data = payload["field_plans"][name]["value"]
        identity = field_identity("resolved-field-install",
                                  {key: val for key, val in data.items() if key != "identity"})
        if identity != plan.identity or data["identity"] != identity.token:
            raise ValueError("field installation projection lost its named authority")
        # The full authority remains in the projected record.
        data["identity"] = None
        options = data["native_options"]
        provider = options["provider_identity"]
        slot = options["provider_slot"]
        if slot != field_identity("qualified-field-provider", provider).token or \
                options["provider_identity_text"] != canonical_bytes(strict_field_data(provider)).hex():
            raise ValueError("field provider slot differs from its declared contributions")
        projected = names.project(provider, references=REFERENCES, values={
            ("contributions", "[]", "owner_block"): "block",
        })
        projected_slot = field_identity("qualified-field-provider", projected).token
        if slots.setdefault(slot, projected_slot) != projected_slot:
            raise ValueError("conflicting projected field provider identities")
        labels[field_provider_target(name, slot)] = field_provider_target(name, projected_slot)
        options["provider_slot"] = projected_slot
        options["provider_identity_text"] = canonical_bytes(strict_field_data(projected)).hex()

        method = options["method_provider"]
        if method["identity"] != field_identity(
            "prepared-field-lowering-binding",
            {key: val for key, val in method.items() if key != "identity"},
        ).token:
            raise ValueError("field method projection lost its named authority")
        method["identity"] = None
        resolved_options = method["resolution"]["native_options"]
        output = resolved_options["output_route"]
        block_name = output["owner_block"]
        unknown = plan.operator.unknown.qualified_id
        named_output = field_output_target(name, block_name, unknown)
        if any(named_output in row["targets"] for row in method["resolution"]["evidence"]):
            labels[named_output] = field_output_target(
                name, names.block_label(block_name), names.project_qualified_id(unknown))
        if "rhs_identity" in resolved_options:
            rhs = {"rhs": strict_field_data(plan.operator.equation.rhs)}
            named = field_identity("field-rhs", rhs).token
            if resolved_options["rhs_identity"] != named or options["rhs_identity"] != named:
                raise ValueError("field RHS identity differs from the authored expression")
            projected_rhs = field_identity(
                "field-rhs", names.project(rhs, references=REFERENCES)).token
            resolved_options["rhs_identity"] = options["rhs_identity"] = projected_rhs

    return replace(names, field_slots=tuple(slots.items()), field_coverage=tuple(labels.items()))


def presentation_map(problem: Any, *, layout: Any = None, numerics: Any = None) -> PresentationMap:
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
        # K1: without a compile layout (a validated case) a layout the Program captured is not
        # numbered yet; the resolved snapshot numbers it.
        layouts_open=layout is None,
    )
    names = PresentationMap(
        case=names.case, blocks=names.blocks, models=names.models,
        block_models=names.block_models, layouts=names.layouts, layouts_open=names.layouts_open,
        digests=_named_digests(problem, layout, names),
    )
    if numerics is None:
        return names
    from pops.boundary.transport import ResolvedTransportBoundarySet

    return names.with_derived_handles(
        row for numerical in numerics.values()
        for authority in numerical.boundaries
        if type(authority) is ResolvedTransportBoundarySet
        for condition in authority.conditions
        for row in condition.presentation_handles(names)
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
            derived=DERIVED, references=REFERENCES, reference_keys=REFERENCE_KEYS)
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
    "ARTIFACT_KEYS", "ARTIFACT_VALUES", "DERIVED", "PLAN_VALUES", "REFERENCES", "REFERENCE_KEYS",
    "SEMANTIC_KEYS", "SEMANTIC_VALUES", "SETS",
    "presentation_map", "project_retained_auxiliaries",
]
