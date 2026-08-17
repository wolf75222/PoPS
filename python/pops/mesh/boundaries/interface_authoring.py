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


def _trace_projection_contract(block: Any) -> tuple[str, Any, int]:
    """Derive one endpoint trace requirement from its selected reconstruction authority."""
    from .ghost_plan_types import InterfaceTraceOperation
    from pops.numerics.reconstruction import authenticated_reconstruction_route
    from pops.runtime.routes import LIMITER_NONE

    spatial = block.numerics.primary_spatial()
    route = authenticated_reconstruction_route(spatial.reconstruction)
    operation = (
        InterfaceTraceOperation.CELL_AVERAGE
        if route.id == LIMITER_NONE.id
        else InterfaceTraceOperation.RECONSTRUCTED_FACE
    )
    return route.id, operation, spatial.ghost_depth


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
class TangentialTransform:
    """Executable right-to-left transform over the ``Dim - 1`` face tangents.

    Tangents are numbered by increasing spatial-axis order after removing the
    endpoint normal axis.  ``None`` at the interface-authoring layer selects
    the identity transform after the layout has supplied its exact dimension.
    """

    right_tangent_for_left: tuple[int, ...]
    right_tangent_sign: tuple[int, ...]
    right_tangent_offset: tuple[float, ...]

    def __post_init__(self) -> None:
        permutation = self.right_tangent_for_left
        tangent_count = len(permutation) if isinstance(permutation, tuple) else -1
        if tangent_count < 0 or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in permutation):
            raise TypeError("TangentialTransform permutation must be an integer tuple")
        if sorted(permutation) != list(range(tangent_count)):
            raise ValueError("TangentialTransform permutation must be a bijection")
        signs = self.right_tangent_sign
        if not isinstance(signs, tuple) or len(signs) != tangent_count or any(
                isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1)
                for value in signs):
            raise TypeError("TangentialTransform signs must contain one +/-1 per tangent")
        offsets = self.right_tangent_offset
        if not isinstance(offsets, tuple) or len(offsets) != tangent_count or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(float(value)) for value in offsets):
            raise TypeError("TangentialTransform offsets must be finite and cover every tangent")

    @classmethod
    def identity(cls, dimension: int) -> TangentialTransform:
        if isinstance(dimension, bool) or dimension not in (1, 2, 3):
            raise ValueError("TangentialTransform dimension must be 1, 2, or 3")
        tangent_count = dimension - 1
        return cls(tuple(range(tangent_count)), (1,) * tangent_count,
                   (0.0,) * tangent_count)

    @property
    def dimension(self) -> int:
        return len(self.right_tangent_for_left) + 1

    def to_data(self) -> dict[str, Any]:
        return {
            "right_tangent_for_left": list(self.right_tangent_for_left),
            "right_tangent_sign": list(self.right_tangent_sign),
            "right_tangent_offset": [float(value) for value in self.right_tangent_offset],
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
    tangential_transform: TangentialTransform | None = None
    right_normal_translation: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or _NAME.fullmatch(self.name) is None:
            raise ValueError(
                "ConservativeInterface.name must match [A-Za-z][A-Za-z0-9_.-]*")
        if not isinstance(self.left, BlockInterfaceSide) or not isinstance(
                self.right, BlockInterfaceSide):
            raise TypeError("ConservativeInterface endpoints must be BlockInterfaceSide values")
        if self.left.state == self.right.state:
            raise ValueError("ConservativeInterface endpoints must use distinct block states")
        if self.left.boundary.outward_sign != -self.right.boundary.outward_sign:
            raise ValueError(
                "ConservativeInterface boundaries must have opposite outward orientations")
        if not isinstance(self.permutation, tuple) or not self.permutation or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.permutation):
            raise TypeError("ConservativeInterface.permutation must be a non-empty integer tuple")
        if sorted(self.permutation) != list(range(len(self.permutation))):
            raise ValueError("ConservativeInterface.permutation must be a bijection")
        if self.tangential_transform is not None and not isinstance(
                self.tangential_transform, TangentialTransform):
            raise TypeError(
                "ConservativeInterface.tangential_transform must be TangentialTransform or None")
        if isinstance(self.right_normal_translation, bool) or not isinstance(
                self.right_normal_translation, (int, float)):
            raise TypeError("ConservativeInterface.right_normal_translation must be finite")
        if not math.isfinite(float(self.right_normal_translation)):
            raise ValueError("ConservativeInterface.right_normal_translation must be finite")
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
            "schema_version": 2,
            "authority_type": "conservative_block_interface",
            "name": self.name,
            "left": self.left.to_data(),
            "right": self.right.to_data(),
            "numerical_flux": _component_data(self.numerical_flux),
            "permutation": list(self.permutation),
            "tangential_transform": (
                "identity" if self.tangential_transform is None
                else self.tangential_transform.to_data()),
            "right_normal_translation": float(self.right_normal_translation),
        }

    def canonical_identity(self) -> dict[str, Any]:
        data = self.to_data()
        data["right_normal_translation"] = float(self.right_normal_translation).hex()
        transform = self.tangential_transform
        if transform is not None:
            payload = dict(transform.to_data())
            payload["right_tangent_offset"] = [
                float(value).hex() for value in transform.right_tangent_offset
            ]
            data["tangential_transform"] = payload
        return data

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
        from pops.problem.handles import BlockHandle
        from .topology import BoundaryHandle, BoundaryOrientation, BoundarySide

        block = side.state.block_ref
        if not isinstance(block, BlockHandle):
            raise TypeError(
                "ConservativeInterface endpoint state has no typed BlockHandle owner")
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
            GhostProducerPlan, GhostProduction, InterfaceGhost,
        )
        from .ghost_plan_types import (
            InterfaceAffineMapping, InterfacePermutation, InterfaceSide, MultiBlockInterface,
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
        left_native = layout_plan.normalized(left_layout).native_spatial_layout
        right_native = layout_plan.normalized(right_layout).native_spatial_layout
        if left_native is None or right_native is None:
            raise TypeError(
                "ConservativeInterface requires exact native spatial layouts at both endpoints")
        if left_native.dimension != right_native.dimension:
            raise ValueError("ConservativeInterface endpoints have different spatial dimensions")
        dimension = left_native.dimension
        tangent = self.tangential_transform or TangentialTransform.identity(dimension)
        if tangent.dimension != dimension:
            raise ValueError(
                "ConservativeInterface tangential transform rank differs from its layouts")

        def interface_handle(local_id: str, kind: str) -> Handle:
            return Handle(local_id, kind=kind, owner=owner)

        left_disc = interface_handle(
            "%s_left_%s" % (self.name, by_name[left_name].numerics.identity.token),
            "discretization")
        right_disc = interface_handle(
            "%s_right_%s" % (self.name, by_name[right_name].numerics.identity.token),
            "discretization")

        left_trace = _trace_projection_contract(by_name[left_name])
        right_trace = _trace_projection_contract(by_name[right_name])
        interface = MultiBlockInterface(
            interface_handle(self.name, "multiblock_interface"),
            InterfaceSide(
                left_boundary, left_layout, left_disc, left_boundary.orientation,
                interface_handle(self.name + "_left_trace", "interface_projection"),
                *left_trace),
            InterfaceSide(
                right_boundary, right_layout, right_disc, right_boundary.orientation,
                interface_handle(self.name + "_right_trace", "interface_projection"),
                *right_trace),
            interface_handle(self.name + "_shared_flux", "conservative_flux"),
            InterfacePermutation(
                interface_handle(self.name + "_permutation", "interface_permutation"),
                self.permutation),
            InterfaceAffineMapping(
                interface_handle(self.name + "_mapping", "interface_mapping"),
                right_tangent_for_left=tangent.right_tangent_for_left,
                right_tangent_sign=tangent.right_tangent_sign,
                right_tangent_offset=tangent.right_tangent_offset,
                right_normal_translation=float(self.right_normal_translation),
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
            physical_provider = production.producer
            if (
                len(physical_provider.boundary_providers) != 1
                or physical_provider.periodic
                or physical_provider.interfaces
                or physical_provider.operators
            ):
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


__all__ = ["BlockInterfaceSide", "ConservativeInterface", "TangentialTransform"]
