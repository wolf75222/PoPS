"""Public pre-resolve authoring authority for one conservative two-block interface."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import re
from typing import Any

from pops.domain import DomainBoundary
from pops.model import Handle, OwnerPath


_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _handle_data(value: Handle) -> dict[str, Any]:
    return value.canonical_identity() if value.is_resolved else value.inspect()


def _component_data(component: Any) -> dict[str, Any]:
    from pops.external import CompiledComponentArtifact, ExternalComponent

    if type(component) is CompiledComponentArtifact:
        component.verify()
        return {
            "component_id": component.component_id,
            "manifest_identity": component.component_manifest.token,
            "native_interface": component.interface.to_data(),
        }
    if type(component) is ExternalComponent:
        return {
            "component_id": component.component_manifest.component_id,
            "manifest_identity": component.component_manifest.manifest_digest.token,
            "native_interface": component.component_type.interface.to_data(),
        }
    raise TypeError(
        "ConservativeInterface.numerical_flux must be an exact ExternalComponent or "
        "CompiledComponentArtifact")


@dataclass(frozen=True, slots=True)
class BlockInterfaceSide:
    """One authored endpoint: a block-qualified state and one geometric frame boundary."""

    state: Handle
    boundary: DomainBoundary

    def __post_init__(self) -> None:
        if not isinstance(self.state, Handle) or self.state.kind != "state":
            raise TypeError("BlockInterfaceSide.state must be a typed StateHandle")
        if not isinstance(self.boundary, DomainBoundary):
            raise TypeError("BlockInterfaceSide.boundary must be a typed DomainBoundary")

    def resolve_references(self, resolver: Any) -> BlockInterfaceSide:
        state = self.state if self.state.is_resolved else resolver(self.state)
        if not isinstance(state, Handle) or state.kind != "state" or not state.is_resolved:
            raise TypeError("BlockInterfaceSide.state did not resolve to a canonical StateHandle")
        return replace(self, state=state)

    def to_data(self) -> dict[str, Any]:
        return {
            "state": _handle_data(self.state),
            "boundary": self.boundary.canonical_identity(),
        }


@dataclass(frozen=True, slots=True)
class ConservativeInterface:
    """One shared NumericalFlux authority authored before validate/resolve.

    ``attach(left_plan, right_plan)`` registers the same immutable authority on both endpoint
    numerical plans.  Resolution owns the topology/layout qualification and consumes it into two
    executable ``GhostProducerPlan`` values; callers never patch a resolved plan.
    """

    name: str
    left: BlockInterfaceSide
    right: BlockInterfaceSide
    numerical_flux: Any
    permutation: tuple[int, ...]
    tangential_orientation: str = "aligned"
    right_normal_translation: float = 0.0
    right_tangential_offset: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(
                "ConservativeInterface.name must match [A-Za-z][A-Za-z0-9_.-]*")
        if not isinstance(self.left, BlockInterfaceSide) or not isinstance(
                self.right, BlockInterfaceSide):
            raise TypeError("ConservativeInterface endpoints must be BlockInterfaceSide values")
        if self.left.state == self.right.state:
            raise ValueError("ConservativeInterface endpoints must use distinct block states")
        if self.left.boundary.axis.index != self.right.boundary.axis.index or \
                self.left.boundary.outward_sign != -self.right.boundary.outward_sign:
            raise ValueError(
                "ConservativeInterface boundaries must be opposite faces of the same axis")
        if not isinstance(self.permutation, tuple) or not self.permutation or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.permutation):
            raise TypeError("ConservativeInterface.permutation must be a non-empty integer tuple")
        if sorted(self.permutation) != list(range(len(self.permutation))):
            raise ValueError("ConservativeInterface.permutation must be a bijection")
        if self.tangential_orientation not in {"aligned", "reversed"}:
            raise ValueError(
                "ConservativeInterface.tangential_orientation must be aligned or reversed")
        for name in ("right_normal_translation", "right_tangential_offset"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("ConservativeInterface.%s must be a finite real" % name)
            if not math.isfinite(float(value)):
                raise ValueError("ConservativeInterface.%s must be finite" % name)
        _component_data(self.numerical_flux)

    def attach(self, left_plan: Any, right_plan: Any) -> ConservativeInterface:
        """Register this exact authority on both mutable endpoint plans and return it."""
        from pops.numerics import DiscretizationPlan

        if type(left_plan) is not DiscretizationPlan or type(right_plan) is not DiscretizationPlan:
            raise TypeError("ConservativeInterface.attach requires two DiscretizationPlan values")
        if left_plan is right_plan:
            raise ValueError(
                "ConservativeInterface endpoints require distinct per-block numerical plans")
        # Two-plan registration is one authoring transaction.  Prove both destinations first so a
        # frozen/duplicate right plan can never leave a one-sided authority in the left plan.
        left_plan.interfaces.preflight_add(self)
        right_plan.interfaces.preflight_add(self)
        left_plan.interfaces.add(self)
        right_plan.interfaces.add(self)
        return self

    def resolve_references(self, resolver: Any) -> ConservativeInterface:
        return replace(
            self,
            left=self.left.resolve_references(resolver),
            right=self.right.resolve_references(resolver),
        )

    def resolve_for_numerics(self, context: Any) -> ConservativeInterface:
        """Resolve and authenticate the endpoint owned by one numerical-plan context."""
        resolver = getattr(context, "resolve", None)
        frame = getattr(context, "frame", None)
        block = getattr(context, "block", None)
        if not callable(resolver) or frame is None or not isinstance(block, Handle):
            raise TypeError(
                "ConservativeInterface requires a BoundaryResolutionContext-like protocol")
        resolved = self.resolve_references(resolver)
        owned = tuple(
            side for side in (resolved.left, resolved.right) if side.state.block_ref == block)
        if len(owned) != 1:
            raise ValueError(
                "ConservativeInterface must own exactly one endpoint in each attached plan")
        frame_boundaries = getattr(getattr(frame, "boundaries", None), "all", None)
        if not isinstance(frame_boundaries, tuple) or any(
                not isinstance(row, DomainBoundary) for row in frame_boundaries):
            raise TypeError(
                "ConservativeInterface endpoint frame must expose typed boundaries.all")
        if owned[0].boundary not in frame_boundaries:
            raise ValueError(
                "ConservativeInterface endpoint boundary does not belong to its block frame")
        return resolved

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "authority_type": "conservative_block_interface",
            "name": self.name,
            "left": self.left.to_data(),
            "right": self.right.to_data(),
            "numerical_flux": _component_data(self.numerical_flux),
            "permutation": list(self.permutation),
            "tangential_orientation": self.tangential_orientation,
            "right_normal_translation": float(self.right_normal_translation),
            "right_tangential_offset": float(self.right_tangential_offset),
        }

    canonical_identity = to_data

    @property
    def canonical_key(self) -> str:
        return json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"))

    def interface_endpoint_claims(self) -> tuple[dict[str, Any], ...]:
        """Return exact face claims for generic cross-authority conflict detection."""
        claims = []
        for side in (self.left, self.right):
            block = side.state.block_ref
            if block is None or not block.is_resolved:
                raise TypeError(
                    "ConservativeInterface endpoint claims require resolved block states")
            claims.append({
                "schema_version": 1,
                "block": block.qualified_id,
                "boundary": side.boundary.canonical_identity(),
                "level": 0,
            })
        return tuple(claims)

    @staticmethod
    def _block_name(side: BlockInterfaceSide) -> str:
        block = side.state.block_ref
        if block is None or not block.is_resolved:
            raise TypeError(
                "ConservativeInterface states must retain their canonical block qualification")
        return block.local_id

    @staticmethod
    def _boundary(side: BlockInterfaceSide) -> Any:
        from pops.domain import BoundarySide as DomainSide
        from .topology import BoundaryHandle, BoundaryOrientation, BoundarySide

        block = side.state.block_ref
        if block is None:
            raise TypeError("ConservativeInterface endpoint state has no block owner")
        boundary_side = (
            BoundarySide.LOWER if side.boundary.side is DomainSide.LOWER
            else BoundarySide.UPPER)
        return BoundaryHandle(
            "%s@%s" % (side.boundary.name, side.boundary.domain_geometry_id),
            owner=block.instance_owner_path,
            orientation=BoundaryOrientation(side.boundary.axis.index, boundary_side),
        )

    def compose_resolved_blocks(self, blocks: tuple[Any, ...], layout_plan: Any) -> tuple[Any, ...]:
        """Consume this authority into both exact endpoint GhostProducerPlans."""
        from .component_binding import BoundaryComponentBinding
        from .ghost_plan import (
            GhostProducerPlan, GhostProduction, InterfaceGhost, PhysicalGhost,
        )
        from .ghost_plan_types import (
            InterfaceAffineMapping, InterfacePermutation, InterfaceSide, MultiBlockInterface,
            TangentialOrientation,
        )

        left_name, right_name = self._block_name(self.left), self._block_name(self.right)
        if left_name == right_name:
            raise ValueError("ConservativeInterface endpoints resolved to the same block")
        by_name = {block.name: block for block in blocks}
        if len(by_name) != len(blocks) or left_name not in by_name or right_name not in by_name:
            raise ValueError("ConservativeInterface endpoint block is absent from resolved Case")
        endpoint_blocks = (by_name[left_name], by_name[right_name])
        for block in endpoint_blocks:
            if block.numerics is None or len(block.numerics.boundaries) != 1 or not isinstance(
                    block.numerics.boundaries[0], GhostProducerPlan):
                raise TypeError(
                    "ConservativeInterface endpoints require one composed GhostProducerPlan")

        owner = self.left.state.block_ref.owner_path
        if owner != self.right.state.block_ref.owner_path:
            raise ValueError("ConservativeInterface endpoints belong to different Cases")
        left_boundary, right_boundary = self._boundary(self.left), self._boundary(self.right)
        left_layout = layout_plan.layout_for(self.left.state)
        right_layout = layout_plan.layout_for(self.right.state)

        def interface_handle(local_id: str, kind: str) -> Handle:
            return Handle(local_id, kind=kind, owner=owner)

        left_disc = interface_handle(
            "%s_left_%s" % (self.name, by_name[left_name].numerics.identity.token),
            "discretization")
        right_disc = interface_handle(
            "%s_right_%s" % (self.name, by_name[right_name].numerics.identity.token),
            "discretization")
        interface = MultiBlockInterface(
            interface_handle(self.name, "multiblock_interface"),
            InterfaceSide(
                left_boundary, left_layout, left_disc, left_boundary.orientation,
                interface_handle(self.name + "_left_trace", "interface_projection")),
            InterfaceSide(
                right_boundary, right_layout, right_disc, right_boundary.orientation,
                interface_handle(self.name + "_right_trace", "interface_projection")),
            interface_handle(self.name + "_shared_flux", "conservative_flux"),
            InterfacePermutation(
                interface_handle(self.name + "_permutation", "interface_permutation"),
                self.permutation),
            InterfaceAffineMapping(
                interface_handle(self.name + "_mapping", "interface_mapping"),
                tangential_orientation=TangentialOrientation(self.tangential_orientation),
                right_normal_translation=float(self.right_normal_translation),
                right_tangential_scale=(
                    1.0 if self.tangential_orientation == "aligned" else -1.0),
                right_tangential_offset=float(self.right_tangential_offset),
            ),
        )
        binding = BoundaryComponentBinding(
            interface.shared_conservative_flux, self.numerical_flux)
        protocol_owner = OwnerPath.shared("pops.boundary.ghost-producers")

        def consume(block: Any, boundary: Any, state: Handle, side_name: str) -> Any:
            numerics = block.numerics
            plan = numerics.boundaries[0]
            matches = [
                index for index, production in enumerate(plan.productions)
                if production.region.boundary == boundary and production.region.subject == state
            ]
            if len(matches) != 1:
                raise ValueError(
                    "ConservativeInterface %s endpoint must match exactly one physical region"
                    % side_name)
            selected = matches[0]
            production = plan.productions[selected]
            if not isinstance(production.producer, PhysicalGhost):
                raise ValueError(
                    "ConservativeInterface %s endpoint face is already consumed" % side_name)
            producer = InterfaceGhost(
                handle=Handle(
                    "%s_%s_endpoint" % (self.name, side_name),
                    kind="ghost_producer", owner=plan.topology.owner),
                protocol=Handle(
                    "shared_interface_v1", kind="ghost_producer_protocol",
                    owner=protocol_owner),
                interface=interface,
                dependencies=production.producer.dependencies,
            )
            productions = list(plan.productions)
            productions[selected] = GhostProduction(production.region, producer)
            composed = GhostProducerPlan(
                plan.topology, plan.coverage, plan.regions, tuple(productions),
                plan.corner_policies, plan.interfaces + (interface,),
                plan.residual_contributions, plan.linearization_contributions,
                plan.execution_authority, plan.component_bindings + (binding,),
            )
            remaining = tuple(
                row for row in numerics.interfaces
                if getattr(row, "canonical_key", None) != self.canonical_key)
            return replace(
                block, numerics=replace(
                    numerics, boundaries=(composed,), interfaces=remaining))

        updates = {
            left_name: consume(
                by_name[left_name], left_boundary, self.left.state, "left"),
            right_name: consume(
                by_name[right_name], right_boundary, self.right.state, "right"),
        }
        return tuple(updates.get(block.name, block) for block in blocks)


__all__ = ["BlockInterfaceSide", "ConservativeInterface"]
