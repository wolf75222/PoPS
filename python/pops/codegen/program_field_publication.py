"""Resolved exact provider claims from explicit consumed-field publication nodes."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from pops.identity import Identity, canonical_bytes
from pops.model import Handle
from pops.model.provider_pack import ComponentKey, ProviderEntry, ProviderPack
from pops.time._program.serialization import _json_ready


def _key(module: Any, target: Handle, component: str) -> ComponentKey:
    declaration = target.declaration_ref
    if declaration is None or declaration.kind != "field":
        raise ValueError("Program field output requires its original field declaration")
    module.declaration_index().authenticate(declaration._resolved())
    space = module.field_spaces().get(declaration.local_id)
    if space is None or component not in space.components:
        raise ValueError("Program field output selects an undeclared destination component")
    return ComponentKey(str(module.owner_path.canonical()), "field", space.name, component)


def publication_claims(block: Any, program: Any) -> tuple[dict[str, Any], ...]:
    from ._compiler_lowering import require_compiler_lowering
    from ._resolved_block_operations import _program_values
    from pops.fields._program_publication import validate_field_publication

    module = require_compiler_lowering(block.model).source_module
    claims = {}
    for value, _location, _guards in _program_values(program):
        if value.op != "field_publication":
            continue
        rows = validate_field_publication(value)
        for row, source in zip(rows, value.inputs[:len(rows)], strict=True):
            target = row["target"]
            if str(target.owner_path.canonical()) != block.instance_owner_qid:
                continue
            key = _key(module, target, row["component"])
            observed = source if source.op == "field_component" else source.inputs[0]
            claim = {"key": key.to_data(), "target": target._resolved().canonical_identity(),
                     "producer": value.attrs["field_problem_identity"],
                     "unknown": _json_ready(observed.attrs["field_unknown"]),
                     "observation": source.op, "source_component": row["source_component"]}
            prior = claims.get(key)
            if prior is not None and canonical_bytes(prior) != canonical_bytes(claim):
                raise ValueError("Program field output has competing physical publication authorities")
            claims[key] = claim
    return tuple(claims[key] for key in sorted(claims))


def reproject_publication_packs(module: Any, packs: Any, claims: Any) -> Any:
    """Preserve the existing carrier and selections; replace only authenticated field producers."""
    from .component_provider_packs import (
        ComponentProviderPacks, compact_auxiliary_provider_pack, consumer_provider_plan,
    )

    selected = {}
    for claim in claims:
        key = ComponentKey(**claim["key"])
        target = Handle.from_canonical_identity(_json_ready(claim["target"]))
        if _key(module, target, key.component) != key:
            raise ValueError("Program field publication key differs from its exact declaration")
        identity = Identity.from_token(claim["producer"])
        if identity.domain != "field-problem" or key in selected:
            raise ValueError("Program field publication has invalid or repeated producer authority")
        unknown = Handle.from_canonical_identity(_json_ready(claim["unknown"]))
        if unknown.kind != "field" or unknown.block_ref is not None:
            raise ValueError("Program field publication lost independent field storage authority")
        prior = packs.complete.declared_entry(key)
        if prior.producer != "runtime_input" or not prior.available:
            raise ValueError("Program field publication competes with an existing physical producer")
        contract = packs.complete.contract(key)
        if contract.centering != "cell" or contract.layout != "cell":
            raise ValueError("Program field publication requires exact cell storage")
        selected[key] = identity.token
    rows = [(key, packs.complete.contract(key),
             ProviderEntry(selected[key], True, packs.complete.declared_entry(key).slot)
             if key in selected else packs.complete.declared_entry(key)) for key in packs.complete]
    complete = ProviderPack(rows, capacity=packs.complete.capacity)
    by_operator = {name: complete.select(pack) for name, pack in packs.by_operator.items()}
    physical = complete.select(packs.physical_flux)
    return ComponentProviderPacks(complete, by_operator, physical,
        compact_auxiliary_provider_pack(complete), packs.auxiliary_routes, packs.auxiliary_route_metadata,
        {name: consumer_provider_plan(pack) for name, pack in by_operator.items()},
        consumer_provider_plan(physical))


def attach_publication_claims(plan: Any, module: Any, claims: Any) -> Any:
    if not claims:
        return plan
    from ._resolved_operation_inputs import provider_evidence, resolve_plan_provider_packs

    packs = reproject_publication_packs(module, resolve_plan_provider_packs(module, plan.operations), claims)
    evidence = provider_evidence(packs)
    evidence["program_field_publications"] = list(claims)
    return replace(plan, provider_evidence=evidence)


def provider_identity(claim: Any) -> str:
    key = claim["key"]
    return "provider:%s:%s/%s/%s" % (
        claim["producer"], key["owner_qid"], key["space_name"], key["component"])


def remaining_input_pack(pack: Any, target: Handle, space: Any, published: Any) -> Any:
    """Exact static prerequisites of a mixed consumed-field observation context."""
    selected = []
    for key in pack:
        if key.owner_qid != str(target.declaration_ref.owner_path.canonical()) \
                or key.space_kind != "field" or key.space_name != space.name:
            continue
        if key.component not in space.components:
            raise ValueError("field publication provider contains an undeclared component")
        entry = pack.declared_entry(key)
        if key in published:
            continue
        if entry.producer != "runtime_input" or not entry.available:
            raise ValueError("field publication remainder requires exact available runtime inputs")
        selected.append(key)
    return pack.select(selected)
