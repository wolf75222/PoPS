"""Canonical data contracts used by boundary ghost-production plans."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

from .providers import BoundaryProvider
from .topology import BoundaryHandle, BoundaryOrientation

if TYPE_CHECKING:
    from pops.model import Handle


_SCHEMA_VERSION = 1


def _handle(value: Any, *, where: str, kinds: frozenset[str] | None = None) -> Handle:
    from pops.model import Handle

    if isinstance(value, str) or not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    if kinds is not None and value.kind not in kinds:
        raise TypeError("%s requires Handle.kind in %s, got %r" %
                        (where, sorted(kinds), value.kind))
    return value


def _handles(values: Any, *, where: str,
             kinds: frozenset[str] | None = None) -> tuple[Handle, ...]:
    if not isinstance(values, tuple):
        raise TypeError("%s must be a tuple" % where)
    rows = tuple(_handle(row, where=where, kinds=kinds) for row in values)
    if len(rows) != len(set(rows)):
        raise ValueError("%s contains duplicate Handles" % where)
    return tuple(sorted(rows, key=lambda row: row.qualified_id))


def _canonical_id(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GhostStencilManifest:
    handle: Handle
    required_depth: tuple[int, ...]

    def __post_init__(self) -> None:
        _handle(self.handle, where="GhostStencilManifest.handle",
                kinds=frozenset(("stencil", "stencil_manifest", "manifest")))
        _depth_tuple(self.required_depth, where="GhostStencilManifest.required_depth")

    def canonical_identity(self) -> dict[str, Any]:
        return {"handle": self.handle.canonical_identity(),
                "required_depth": list(self.required_depth)}


def _depth_tuple(value: Any, *, where: str) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value or any(
            isinstance(row, bool) or not isinstance(row, int) or row < 0 for row in value):
        raise TypeError("%s must be a non-empty tuple of integers >= 0" % where)
    return value


@dataclass(frozen=True, slots=True)
class GhostDepthCapability:
    handle: Handle
    provider_manifest: Handle
    available_depth: tuple[int, ...]

    def __post_init__(self) -> None:
        _handle(self.handle, where="GhostDepthCapability.handle",
                kinds=frozenset(("capability",)))
        _handle(self.provider_manifest, where="GhostDepthCapability.provider_manifest",
                kinds=frozenset(("manifest", "layout_manifest", "stencil_manifest",
                                 "discretization_manifest")))
        _depth_tuple(self.available_depth, where="GhostDepthCapability.available_depth")

    def canonical_identity(self) -> dict[str, Any]:
        return {"handle": self.handle.canonical_identity(),
                "provider_manifest": self.provider_manifest.canonical_identity(),
                "available_depth": list(self.available_depth)}


@dataclass(frozen=True, slots=True, init=False)
class GhostDepthRequirement:
    """Resolved depth derived only from a stencil manifest and authenticated capability."""

    stencil: GhostStencilManifest
    capability: GhostDepthCapability
    depth: tuple[int, ...]

    def __init__(self, stencil: GhostStencilManifest, capability: GhostDepthCapability) -> None:
        if not isinstance(stencil, GhostStencilManifest):
            raise TypeError("GhostDepthRequirement.stencil must be a GhostStencilManifest")
        if not isinstance(capability, GhostDepthCapability):
            raise TypeError("GhostDepthRequirement.capability must be a GhostDepthCapability")
        if len(stencil.required_depth) != len(capability.available_depth):
            raise ValueError("ghost depth dimension is inconsistent with its capability manifest")
        if any(required > available for required, available in zip(
                stencil.required_depth, capability.available_depth, strict=True)):
            raise ValueError("ghost depth capability is insufficient for the stencil manifest")
        object.__setattr__(self, "stencil", stencil)
        object.__setattr__(self, "capability", capability)
        object.__setattr__(self, "depth", stencil.required_depth)

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "depth_type": "derived",
                "stencil": self.stencil.canonical_identity(),
                "capability": self.capability.canonical_identity(),
                "depth": list(self.depth)}


@dataclass(frozen=True, slots=True)
class GhostRegion:
    subject: Handle
    layout: Handle
    selector: Handle
    depth: GhostDepthRequirement
    boundary: BoundaryHandle | None = None

    def __post_init__(self) -> None:
        _handle(self.subject, where="GhostRegion.subject", kinds=frozenset(("state", "field")))
        _handle(self.layout, where="GhostRegion.layout", kinds=frozenset(("layout",)))
        _handle(self.selector, where="GhostRegion.selector",
                kinds=frozenset(("ghost_region",)))
        if not isinstance(self.depth, GhostDepthRequirement):
            raise TypeError("GhostRegion.depth must be derived from GhostDepthRequirement")
        if self.boundary is not None and not isinstance(self.boundary, BoundaryHandle):
            raise TypeError("GhostRegion.boundary must be a BoundaryHandle or None")

    @property
    def overlap_key(self) -> tuple[Handle, Handle, Handle]:
        return (self.subject, self.layout, self.selector)

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "region_type": "ghost",
                "subject": self.subject.canonical_identity(),
                "layout": self.layout.canonical_identity(),
                "selector": self.selector.canonical_identity(),
                "depth": self.depth.canonical_identity(),
                "boundary": (None if self.boundary is None else
                             self.boundary.canonical_identity())}

    @property
    def canonical_id(self) -> str:
        return _canonical_id(self.canonical_identity())


@dataclass(frozen=True, slots=True)
class GhostCoverageManifest:
    """Authoritative expected region set derived from layout/discretization manifests."""

    handle: Handle
    layout_manifest: Handle
    discretization_manifest: Handle
    regions: tuple[GhostRegion, ...]

    def __post_init__(self) -> None:
        _handle(self.handle, where="GhostCoverageManifest.handle",
                kinds=frozenset(("ghost_coverage_manifest",)))
        _handle(self.layout_manifest, where="GhostCoverageManifest.layout_manifest",
                kinds=frozenset(("layout_manifest", "manifest")))
        _handle(self.discretization_manifest,
                where="GhostCoverageManifest.discretization_manifest",
                kinds=frozenset(("discretization_manifest", "manifest")))
        if not isinstance(self.regions, tuple) or any(
                not isinstance(row, GhostRegion) for row in self.regions):
            raise TypeError("GhostCoverageManifest.regions must contain GhostRegion objects")
        coverage = [row.overlap_key for row in self.regions]
        if len(coverage) != len(set(coverage)):
            raise ValueError("GhostCoverageManifest contains overlapping ghost regions")
        object.__setattr__(self, "regions", tuple(sorted(
            self.regions, key=lambda row: row.canonical_id)))

    @property
    def qualified_id(self) -> str:
        return self.handle.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "coverage_type": "ghost",
                "handle": self.handle.canonical_identity(),
                "layout_manifest": self.layout_manifest.canonical_identity(),
                "discretization_manifest": self.discretization_manifest.canonical_identity(),
                "regions": [row.canonical_identity() for row in self.regions]}

    @property
    def canonical_id(self) -> str:
        return _canonical_id(self.canonical_identity())


class CornerCondition(Enum):
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    CHARACTERISTIC = "characteristic"


class CornerMode(Enum):
    ERROR = "error"
    EXPLICIT_RESOLVER = "explicit_resolver"


@dataclass(frozen=True, slots=True)
class CornerConstraint:
    source: BoundaryProvider
    condition: CornerCondition
    datum: Handle | None

    def __post_init__(self) -> None:
        if not isinstance(self.source, BoundaryProvider):
            raise TypeError("CornerConstraint.source must be a BoundaryProvider")
        if not isinstance(self.condition, CornerCondition):
            raise TypeError("CornerConstraint.condition must be a CornerCondition")
        if self.condition is CornerCondition.DIRICHLET and self.datum is None:
            raise ValueError("Dirichlet corner constraint requires an explicit datum Handle")
        if self.datum is not None:
            _handle(self.datum, where="CornerConstraint.datum")

    def canonical_identity(self) -> dict[str, Any]:
        return {"source": self.source.canonical_identity(), "condition": self.condition.value,
                "datum": None if self.datum is None else self.datum.canonical_identity()}


@dataclass(frozen=True, slots=True)
class CornerPolicy:
    corner: GhostRegion
    constraints: tuple[CornerConstraint, ...]
    mode: CornerMode
    resolver: Handle | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.corner, GhostRegion):
            raise TypeError("CornerPolicy.corner must be a GhostRegion")
        if not isinstance(self.constraints, tuple) or len(self.constraints) < 2 or any(
                not isinstance(row, CornerConstraint) for row in self.constraints):
            raise TypeError("CornerPolicy.constraints requires at least two CornerConstraint rows")
        if not isinstance(self.mode, CornerMode):
            raise TypeError("CornerPolicy.mode must be a CornerMode")
        source_ids = [row.source.qualified_id for row in self.constraints]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("CornerPolicy contains the same source twice")
        if self.mode is CornerMode.ERROR and self.resolver is not None:
            raise ValueError("CornerMode.ERROR must not declare a resolver")
        if self.mode is CornerMode.EXPLICIT_RESOLVER:
            _handle(self.resolver, where="CornerPolicy.resolver",
                    kinds=frozenset(("corner_resolver",)))
        dirichlet = [row for row in self.constraints
                     if row.condition is CornerCondition.DIRICHLET]
        if self.mode is CornerMode.ERROR and len({row.datum for row in dirichlet}) > 1:
            first, second = dirichlet[:2]
            raise ValueError(
                "incompatible Dirichlet corner sources %s and %s require an explicit resolver"
                % (first.source.qualified_id, second.source.qualified_id))
        object.__setattr__(self, "constraints", tuple(sorted(
            self.constraints, key=lambda row: row.source.qualified_id)))

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "policy_type": "corner",
                "corner": self.corner.canonical_identity(),
                "constraints": [row.canonical_identity() for row in self.constraints],
                "mode": self.mode.value,
                "resolver": None if self.resolver is None else self.resolver.canonical_identity()}


class InterfaceTraceOperation(str, Enum):
    """Native operation required to project one endpoint state onto its face trace.

    ``CELL_AVERAGE`` is the only operation currently executable by the type-erased
    multi-block scheduler.  Higher-order reconstructions retain their exact
    ``RECONSTRUCTED_FACE`` requirement and fail closed before native installation;
    they are never silently replaced by cell averages.
    """

    CELL_AVERAGE = "cell_average"
    RECONSTRUCTED_FACE = "reconstructed_face"


@dataclass(frozen=True, slots=True)
class InterfaceSide:
    boundary: BoundaryHandle
    layout: Handle
    discretization: Handle
    orientation: BoundaryOrientation
    projection: Handle
    trace_provider: str
    trace_operation: InterfaceTraceOperation
    required_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, BoundaryHandle):
            raise TypeError("InterfaceSide.boundary must be a BoundaryHandle")
        _handle(self.layout, where="InterfaceSide.layout", kinds=frozenset(("layout",)))
        _handle(self.discretization, where="InterfaceSide.discretization",
                kinds=frozenset(("discretization",)))
        if self.orientation != self.boundary.orientation:
            raise ValueError("InterfaceSide orientation does not authenticate its BoundaryHandle")
        _handle(self.projection, where="InterfaceSide.projection",
                kinds=frozenset(("interface_projection",)))
        if not isinstance(self.trace_provider, str) or not self.trace_provider:
            raise TypeError("InterfaceSide.trace_provider must be a non-empty provider identity")
        if not isinstance(self.trace_operation, InterfaceTraceOperation):
            raise TypeError(
                "InterfaceSide.trace_operation must be an InterfaceTraceOperation")
        if isinstance(self.required_depth, bool) or not isinstance(
                self.required_depth, int) or self.required_depth < 1:
            raise TypeError("InterfaceSide.required_depth must be an integer >= 1")

    def canonical_identity(self) -> dict[str, Any]:
        return {"boundary": self.boundary.canonical_identity(),
                "layout": self.layout.canonical_identity(),
                "discretization": self.discretization.canonical_identity(),
                "orientation": self.orientation.canonical_identity(),
                "projection": self.projection.canonical_identity(),
                "trace_provider": self.trace_provider,
                "trace_operation": self.trace_operation.value,
                "required_depth": self.required_depth}

@dataclass(frozen=True, slots=True)
class InterfacePermutation:
    """Executable component permutation, not merely a provenance Handle."""

    handle: Handle
    right_component_for_left: tuple[int, ...]

    def __post_init__(self) -> None:
        _handle(self.handle, where="InterfacePermutation.handle",
                kinds=frozenset(("interface_permutation",)))
        values = self.right_component_for_left
        if not isinstance(values, tuple) or not values or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in values):
            raise TypeError(
                "InterfacePermutation.right_component_for_left must be a non-empty "
                "tuple of integers >= 0"
            )
        if sorted(values) != list(range(len(values))):
            raise ValueError("InterfacePermutation must be a bijection")

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "handle": self.handle.canonical_identity(),
            "right_component_for_left": list(self.right_component_for_left),
        }


@dataclass(frozen=True, slots=True)
class InterfaceAffineMapping:
    """Exact ranked map from the right face into the canonical left-face frame.

    The mapping is deliberately executable data.  The Handle authenticates its
    provenance; it does not stand in for values the native scheduler would have
    to guess. Tangent ordinals follow increasing spatial-axis order with the
    normal axis removed, so the same value is executable in 1-D, 2-D and 3-D.
    """

    handle: Handle
    right_tangent_for_left: tuple[int, ...] = ()
    right_tangent_sign: tuple[int, ...] = ()
    right_tangent_offset: tuple[float, ...] = ()
    right_normal_translation: float = 0.0

    def __post_init__(self) -> None:
        _handle(self.handle, where="InterfaceAffineMapping.handle",
                kinds=frozenset(("interface_mapping",)))
        permutation = self.right_tangent_for_left
        signs = self.right_tangent_sign
        offsets = self.right_tangent_offset
        if not isinstance(permutation, tuple) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in permutation):
            raise TypeError(
                "InterfaceAffineMapping.right_tangent_for_left must be an integer tuple"
            )
        tangent_count = len(permutation)
        if sorted(permutation) != list(range(tangent_count)):
            raise ValueError(
                "InterfaceAffineMapping.right_tangent_for_left must be a bijection")
        if not isinstance(signs, tuple) or len(signs) != tangent_count or any(
                isinstance(value, bool) or not isinstance(value, int) or value not in (-1, 1)
                for value in signs):
            raise TypeError(
                "InterfaceAffineMapping.right_tangent_sign must contain one +/-1 per tangent")
        if not isinstance(offsets, tuple) or len(offsets) != tangent_count:
            raise TypeError(
                "InterfaceAffineMapping.right_tangent_offset must cover every tangent")
        values = (self.right_normal_translation, *offsets)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
               not math.isfinite(float(value)) for value in values):
            raise TypeError("InterfaceAffineMapping coefficients must be finite real values")

    @classmethod
    def identity(cls, handle: Handle, dimension: int) -> InterfaceAffineMapping:
        if isinstance(dimension, bool) or dimension not in (1, 2, 3):
            raise ValueError("InterfaceAffineMapping dimension must be 1, 2, or 3")
        tangent_count = dimension - 1
        return cls(
            handle,
            right_tangent_for_left=tuple(range(tangent_count)),
            right_tangent_sign=(1,) * tangent_count,
            right_tangent_offset=(0.0,) * tangent_count,
        )

    @property
    def dimension(self) -> int:
        return len(self.right_tangent_for_left) + 1

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "handle": self.handle.canonical_identity(),
            "right_tangent_for_left": list(self.right_tangent_for_left),
            "right_tangent_sign": list(self.right_tangent_sign),
            "right_tangent_offset": [float(value).hex() for value in self.right_tangent_offset],
            "right_normal_translation": float(self.right_normal_translation).hex(),
        }


@dataclass(frozen=True, slots=True)
class MultiBlockInterface:
    handle: Handle
    left: InterfaceSide
    right: InterfaceSide
    shared_conservative_flux: Handle
    permutation: InterfacePermutation
    mapping: InterfaceAffineMapping

    def __post_init__(self) -> None:
        _handle(self.handle, where="MultiBlockInterface.handle",
                kinds=frozenset(("multiblock_interface",)))
        if not isinstance(self.left, InterfaceSide) or not isinstance(self.right, InterfaceSide):
            raise TypeError("MultiBlockInterface sides must be InterfaceSide objects")
        if self.left.boundary == self.right.boundary:
            raise ValueError("MultiBlockInterface requires two distinct boundary sides")
        if self.left.orientation.outward_sign != -self.right.orientation.outward_sign:
            raise ValueError("MultiBlockInterface sides require authenticated opposite orientations")
        _handle(self.shared_conservative_flux, where="MultiBlockInterface.shared_conservative_flux",
                kinds=frozenset(("conservative_flux",)))
        if not isinstance(self.permutation, InterfacePermutation):
            raise TypeError("MultiBlockInterface.permutation must be an InterfacePermutation")
        if not isinstance(self.mapping, InterfaceAffineMapping):
            raise TypeError("MultiBlockInterface.mapping must be an InterfaceAffineMapping")

    @property
    def qualified_id(self) -> str:
        return self.handle.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "interface_type": "multiblock",
                "handle": self.handle.canonical_identity(),
                "left": self.left.canonical_identity(), "right": self.right.canonical_identity(),
                "shared_conservative_flux": self.shared_conservative_flux.canonical_identity(),
                "permutation": self.permutation.canonical_identity(),
                "mapping": self.mapping.canonical_identity()}


@dataclass(frozen=True, slots=True)
class BoundaryResidualContribution:
    handle: Handle
    region: GhostRegion
    producer: Handle
    residual: Handle
    states: tuple[Handle, ...] = ()
    fields: tuple[Handle, ...] = ()
    runtime_params: tuple[Handle, ...] = ()
    rate: Handle | None = None

    def __post_init__(self) -> None:
        _handle(self.handle, where="BoundaryResidualContribution.handle",
                kinds=frozenset(("boundary_residual_contribution",)))
        if not isinstance(self.region, GhostRegion):
            raise TypeError("BoundaryResidualContribution.region must be a GhostRegion")
        _handle(self.producer, where="BoundaryResidualContribution.producer",
                kinds=frozenset(("ghost_producer",)))
        _handle(self.residual, where="BoundaryResidualContribution.residual",
                kinds=frozenset(("residual_operator",)))
        object.__setattr__(self, "states", _handles(
            self.states, where="BoundaryResidualContribution.states", kinds=frozenset(("state",))))
        object.__setattr__(self, "fields", _handles(
            self.fields, where="BoundaryResidualContribution.fields", kinds=frozenset(("field",))))
        object.__setattr__(self, "runtime_params", _handles(
            self.runtime_params, where="BoundaryResidualContribution.runtime_params",
            kinds=frozenset(("parameter",))))
        if any(getattr(row, "param_kind", None) != "runtime" for row in self.runtime_params):
            raise TypeError("BoundaryResidualContribution.runtime_params requires RuntimeParam handles")
        if self.rate is not None:
            _handle(self.rate, where="BoundaryResidualContribution.rate",
                    kinds=frozenset(("state", "field")))

    def canonical_identity(self) -> dict[str, Any]:
        return {"handle": self.handle.canonical_identity(),
                "region": self.region.canonical_identity(),
                "producer": self.producer.canonical_identity(),
                "residual": self.residual.canonical_identity(),
                "states": [row.canonical_identity() for row in self.states],
                "fields": [row.canonical_identity() for row in self.fields],
                "runtime_params": [row.canonical_identity() for row in self.runtime_params],
                "rate": None if self.rate is None else self.rate.canonical_identity()}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "boundary_residual_contribution",
                **self.canonical_identity()}


@dataclass(frozen=True, slots=True)
class BoundaryLinearizationContribution:
    handle: Handle
    region: GhostRegion
    producer: Handle
    linearization: Handle
    states: tuple[Handle, ...] = ()
    fields: tuple[Handle, ...] = ()
    runtime_params: tuple[Handle, ...] = ()
    rate: Handle | None = None

    def __post_init__(self) -> None:
        _handle(self.handle, where="BoundaryLinearizationContribution.handle",
                kinds=frozenset(("boundary_linearization_contribution",)))
        if not isinstance(self.region, GhostRegion):
            raise TypeError("BoundaryLinearizationContribution.region must be a GhostRegion")
        _handle(self.producer, where="BoundaryLinearizationContribution.producer",
                kinds=frozenset(("ghost_producer",)))
        _handle(self.linearization, where="BoundaryLinearizationContribution.linearization",
                kinds=frozenset(("linearization_operator",)))
        object.__setattr__(self, "states", _handles(
            self.states, where="BoundaryLinearizationContribution.states",
            kinds=frozenset(("state",))))
        object.__setattr__(self, "fields", _handles(
            self.fields, where="BoundaryLinearizationContribution.fields",
            kinds=frozenset(("field",))))
        object.__setattr__(self, "runtime_params", _handles(
            self.runtime_params, where="BoundaryLinearizationContribution.runtime_params",
            kinds=frozenset(("parameter",))))
        if any(getattr(row, "param_kind", None) != "runtime" for row in self.runtime_params):
            raise TypeError(
                "BoundaryLinearizationContribution.runtime_params requires RuntimeParam handles")
        if self.rate is not None:
            _handle(self.rate, where="BoundaryLinearizationContribution.rate",
                    kinds=frozenset(("state", "field")))

    def canonical_identity(self) -> dict[str, Any]:
        return {"handle": self.handle.canonical_identity(),
                "region": self.region.canonical_identity(),
                "producer": self.producer.canonical_identity(),
                "linearization": self.linearization.canonical_identity(),
                "states": [row.canonical_identity() for row in self.states],
                "fields": [row.canonical_identity() for row in self.fields],
                "runtime_params": [row.canonical_identity() for row in self.runtime_params],
                "rate": None if self.rate is None else self.rate.canonical_identity()}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "boundary_linearization_contribution",
                **self.canonical_identity()}


__all__ = [
    "BoundaryLinearizationContribution", "BoundaryResidualContribution", "CornerCondition",
    "CornerConstraint", "CornerMode", "CornerPolicy", "GhostCoverageManifest",
    "GhostDepthCapability", "GhostDepthRequirement", "GhostRegion", "GhostStencilManifest",
    "InterfaceAffineMapping", "InterfacePermutation", "InterfaceSide",
    "InterfaceTraceOperation", "MultiBlockInterface",
]
