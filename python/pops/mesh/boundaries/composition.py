"""Total composition of resolved numerical boundary producers.

This module is deliberately post-layout: only then are the state layout, stencil depth, adaptive
coarse/fine requirement, and physical topology all canonical.  It creates the one
``GhostProducerPlan`` consumed by compile/install; no registry priority or insertion-order fallback
survives this phase.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Any

from pops.identity import canonical_bytes
from pops.identity.semantic import semantic_value
from pops.model import Handle, OwnerPath

from .ghost_plan import (
    CoarseFineInterpolation,
    GhostProducerPlan,
    GhostProducerRegistry,
    GhostProduction,
    PhysicalGhost,
    SameLevelHaloMPI,
)
from .ghost_plan_types import (
    GhostCoverageManifest,
    GhostDepthCapability,
    GhostDepthRequirement,
    GhostRegion,
    GhostStencilManifest,
)


@dataclass(frozen=True, slots=True)
class GhostPlanCompositionContext:
    """Post-layout facts offered to one open boundary-composer protocol."""

    numerics: Any
    layout_plan: Any
    amr_transfer: Any
    authorities: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.authorities, tuple) or not self.authorities:
            raise TypeError("ghost-plan composition requires non-empty authority tuple")


def _composer_scope(authority: Any) -> str:
    capability = getattr(authority, "ghost_plan_composer_capability", None)
    compose = getattr(authority, "compose_ghost_plan", None)
    if not callable(capability) or not callable(compose):
        raise TypeError(
            "resolved boundary authorities must implement ghost_plan_composer_capability() "
            "and compose_ghost_plan(context)"
        )
    first, second = capability(), capability()
    if type(first) is not dict or first != second \
            or set(first) != {"schema_version", "scope"} \
            or first.get("schema_version") != 1 \
            or first.get("scope") not in {"self", "all"}:
        raise TypeError(
            "ghost_plan_composer_capability() must return deterministic "
            "{'schema_version': 1, 'scope': 'self'|'all'}"
        )
    return first["scope"]


def _token(data: Any) -> str:
    projected = semantic_value(data, where="ghost-plan composition identity")
    return hashlib.sha256(canonical_bytes(projected)).hexdigest()


def _handle(prefix: str, kind: str, *, owner: Any, evidence: Any) -> Handle:
    return Handle(
        "%s_%s" % (prefix, _token(evidence)[:24]),
        kind=kind,
        owner=owner,
    )


def compose_transport_boundary(
    authority: Any,
    *,
    context: GhostPlanCompositionContext,
) -> GhostProducerPlan:
    numerics = context.numerics
    layout_plan = context.layout_plan
    amr_transfer = context.amr_transfer
    # This validates one-state native support, exact face coverage, dependency kinds, and derived
    # depth before any GhostRegion claims are emitted.
    compile_contract = authority.compile_boundary_data()
    states = {row.state for row in authority.conditions}
    state = next(iter(states))
    topology = authority.plan.topology
    owner = topology.owner
    layout = layout_plan.layout_for(state)
    normalized = layout_plan.normalized(layout)
    depth_value = int(compile_contract["required_depth"])
    dimension = len(topology.boundaries) // 2
    if dimension != 2:
        raise NotImplementedError(
            "the installed ghost-plan composer currently supports exact 2D Cartesian topology"
        )
    depth = (depth_value,) * dimension
    base_evidence = {
        "authority": authority.canonical_identity(),
        "numerics": numerics.identity.to_data(),
        "layout": layout.canonical_identity(),
        "adaptive": normalized.adaptive,
    }
    layout_manifest = _handle(
        "layout_manifest", "layout_manifest", owner=owner,
        evidence={"layout": layout.canonical_identity(), "plan": layout_plan.canonical_id},
    )
    discretization_manifest = _handle(
        "discretization_manifest", "discretization_manifest", owner=owner,
        evidence=numerics.identity.to_data(),
    )
    stencil = GhostStencilManifest(
        _handle("boundary_stencil", "stencil_manifest", owner=owner,
                evidence={"depth": depth, "numerics": numerics.identity.to_data()}),
        depth,
    )
    # The block allocator derives its halo from the same resolved FV methods.  Equality here is an
    # authenticated exact capability, not a user-authored duplicate order/depth knob.
    depth_capability = GhostDepthCapability(
        _handle("allocated_ghosts", "capability", owner=owner, evidence=base_evidence),
        layout_manifest,
        depth,
    )
    requirement = GhostDepthRequirement(stencil, depth_capability)

    def region(name: str, *, boundary: Any = None) -> GhostRegion:
        selector_evidence = {
            **base_evidence,
            "region": name,
            "boundary": (
                None if boundary is None else boundary.canonical_identity()
            ),
        }
        return GhostRegion(
            state,
            layout,
            _handle(name, "ghost_region", owner=owner, evidence=selector_evidence),
            requirement,
            boundary,
        )

    protocol_owner = OwnerPath.shared("pops.boundary.ghost-producers")
    same_level_region = region("same_level_halo")
    same_level = SameLevelHaloMPI(
        handle=_handle("same_level_halo", "ghost_producer", owner=owner,
                       evidence=base_evidence),
        protocol=_handle("same_level_halo", "ghost_producer_protocol",
                         owner=protocol_owner, evidence={"protocol": "same-level-v1"}),
        mpi_capability=_handle(
            "native_neighbor_exchange", "capability", owner=protocol_owner,
            evidence={"capability": "memoized-native-neighbor-exchange-v1"},
        ),
    )
    regions = [same_level_region]
    productions = [GhostProduction(same_level_region, same_level)]
    producers = [same_level]
    predecessor = same_level.handle

    if normalized.adaptive:
        if amr_transfer is None:
            raise ValueError("adaptive ghost composition requires resolved AMR transfer authority")
        transfer_identity = amr_transfer.identity.to_data()
        coarse_fine_region = region("coarse_fine")
        coarse_fine = CoarseFineInterpolation(
            handle=_handle("coarse_fine", "ghost_producer", owner=owner,
                           evidence={**base_evidence, "transfer": transfer_identity}),
            protocol=_handle("coarse_fine", "ghost_producer_protocol",
                             owner=protocol_owner, evidence={"protocol": "coarse-fine-v1"}),
            interpolation=_handle(
                "resolved_amr_transfer", "interpolation", owner=owner,
                evidence=transfer_identity,
            ),
            dependencies=(predecessor,),
        )
        regions.append(coarse_fine_region)
        productions.append(GhostProduction(coarse_fine_region, coarse_fine))
        producers.append(coarse_fine)
        predecessor = coarse_fine.handle

    conditions_by_boundary = {
        row.provider.outputs[0].boundary: row for row in authority.conditions
    }
    for boundary in topology.physical:
        condition = conditions_by_boundary.get(boundary)
        if condition is None:
            raise ValueError(
                "physical topology boundary %s has no executable provider"
                % boundary.qualified_id
            )
        physical_region = region("physical_face", boundary=boundary)
        physical = PhysicalGhost(
            handle=_handle(
                "physical", "ghost_producer", owner=owner,
                evidence={**base_evidence, "boundary": boundary.canonical_identity()},
            ),
            protocol=_handle("physical", "ghost_producer_protocol",
                             owner=protocol_owner, evidence={"protocol": "physical-v1"}),
            provider=condition.provider,
            dependencies=(predecessor,),
        )
        regions.append(physical_region)
        productions.append(GhostProduction(physical_region, physical))
        producers.append(physical)

    coverage = GhostCoverageManifest(
        _handle("ghost_coverage", "ghost_coverage_manifest", owner=owner,
                evidence=base_evidence),
        layout_manifest,
        discretization_manifest,
        tuple(regions),
    )
    return GhostProducerRegistry(*producers).resolve(
        topology,
        coverage,
        tuple(regions),
        tuple(productions),
        execution_authority=authority,
    )


def compose_boundary_plans(
    numerics: Any,
    *,
    layout_plan: Any,
    amr_transfer: Any = None,
) -> Any:
    """Replace resolved boundary authorities with exactly one executable producer plan."""
    from pops.numerics.plan import ResolvedDiscretizationPlan

    if type(numerics) is not ResolvedDiscretizationPlan:
        raise TypeError("boundary composition requires exact ResolvedDiscretizationPlan")
    if not numerics.boundaries:
        if numerics.interfaces:
            raise ValueError(
                "resolved conservative interfaces require an executable boundary authority")
        return numerics
    scopes = tuple(_composer_scope(authority) for authority in numerics.boundaries)
    if len(numerics.boundaries) == 1:
        composer = numerics.boundaries[0]
    else:
        aggregate = [
            authority for authority, scope in zip(
                numerics.boundaries, scopes, strict=True) if scope == "all"
        ]
        if len(aggregate) != 1:
            raise ValueError(
                "multiple boundary authorities require exactly one explicit scope='all' "
                "ghost-plan composer; found %d" % len(aggregate)
            )
        composer = aggregate[0]
    context = GhostPlanCompositionContext(
        numerics=numerics,
        layout_plan=layout_plan,
        amr_transfer=amr_transfer,
        authorities=numerics.boundaries,
    )
    plan = composer.compose_ghost_plan(context)
    if not isinstance(plan, GhostProducerPlan):
        raise TypeError("compose_ghost_plan(context) must return a GhostProducerPlan")
    return replace(numerics, boundaries=(plan,))


def compose_shared_interfaces(blocks: tuple[Any, ...], *, layout_plan: Any) -> tuple[Any, ...]:
    """Consume every cross-block authority once after all physical plans exist.

    The extension protocol is intentionally small: an authority provides canonical data and
    ``compose_resolved_blocks(blocks, layout_plan)``.  This coordinator knows neither a concrete
    interface class nor component ABI details; it only enforces two-sided registration and total
    consumption.
    """
    import json

    groups: dict[str, list[tuple[str, Any]]] = {}
    for block in blocks:
        numerics = getattr(block, "numerics", None)
        for authority in (() if numerics is None else numerics.interfaces):
            projection = getattr(authority, "to_data", None)
            compose = getattr(authority, "compose_resolved_blocks", None)
            if not callable(projection) or not callable(compose):
                raise TypeError(
                    "resolved interface authorities must implement to_data() and "
                    "compose_resolved_blocks(blocks, layout_plan)")
            data = projection()
            if not isinstance(data, dict):
                raise TypeError("resolved interface authority to_data() must return a dict")
            key = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
            groups.setdefault(key, []).append((block.name, authority))

    result = blocks
    claimed_endpoints: dict[str, str] = {}
    # Preflight every extension's exact endpoint claims before the first immutable block rewrite.
    # The coordinator consumes only the small protocol and never branches on an interface class.
    for key in sorted(groups):
        registrations = groups[key]
        names = [name for name, _ in registrations]
        if len(registrations) != 2 or len(set(names)) != 2:
            raise ValueError(
                "one conservative interface must be registered on exactly two distinct endpoint "
                "plans; got %s" % sorted(names))
        claims = getattr(registrations[0][1], "interface_endpoint_claims", None)
        if not callable(claims):
            raise TypeError(
                "resolved interface authorities must implement interface_endpoint_claims()")
        first, second = claims(), claims()
        if type(first) is not tuple or first != second or len(first) != 2 or any(
                not isinstance(claim, dict) for claim in first):
            raise TypeError(
                "interface_endpoint_claims() must return one deterministic two-entry tuple")
        registered_blocks = {
            next(
                block.numerics.block.qualified_id for block in blocks
                if block.name == name)
            for name in names
        }
        normalized_claims = []
        for claim in first:
            if set(claim) != {"schema_version", "block", "boundary", "level"} or \
                    claim.get("schema_version") != 1 or not isinstance(
                        claim.get("block"), str) or not claim["block"] or isinstance(
                            claim.get("level"), bool) or not isinstance(
                                claim.get("level"), int) or claim["level"] < 0:
                raise TypeError(
                    "interface endpoint claim must be exact canonical v1 block/boundary/level data")
            from pops.domain import DomainBoundary
            boundary = DomainBoundary.from_dict(claim.get("boundary"))
            normalized_claims.append({
                "schema_version": 1,
                "block": claim["block"],
                "boundary": boundary.canonical_identity(),
                "level": claim["level"],
            })
        if {claim["block"] for claim in normalized_claims} != registered_blocks:
            raise ValueError(
                "interface endpoint claims differ from their two registered numerical blocks")
        for claim in normalized_claims:
            claim_key = json.dumps(
                claim, sort_keys=True, separators=(",", ":"), allow_nan=False)
            previous = claimed_endpoints.setdefault(claim_key, key)
            if previous != key:
                raise ValueError(
                    "multiple conservative interfaces claim the same block boundary endpoint")

    for key in sorted(groups):
        registrations = groups[key]
        names = [name for name, _ in registrations]
        if len(registrations) != 2 or len(set(names)) != 2:
            raise ValueError(
                "one conservative interface must be registered on exactly two distinct endpoint "
                "plans; got %s" % sorted(names))
        authority = registrations[0][1]
        result = authority.compose_resolved_blocks(result, layout_plan)
        if not isinstance(result, tuple) or len(result) != len(blocks) or \
                tuple(block.name for block in result) != tuple(block.name for block in blocks):
            raise TypeError(
                "compose_resolved_blocks must preserve the ordered ResolvedBlock tuple")

    unconsumed = [
        (block.name, len(block.numerics.interfaces))
        for block in result
        if block.numerics is not None and block.numerics.interfaces
    ]
    if unconsumed:
        raise RuntimeError("resolved interface authorities were not totally consumed: %s" % unconsumed)
    return result


__all__ = [
    "GhostPlanCompositionContext",
    "compose_boundary_plans",
    "compose_shared_interfaces",
    "compose_transport_boundary",
]
