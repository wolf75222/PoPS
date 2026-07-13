"""Immutable ghost-region producer plans with exact, order-independent resolution."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any

from .ghost_plan_types import (
    BoundaryLinearizationContribution, BoundaryResidualContribution, CornerPolicy,
    GhostCoverageManifest, GhostRegion, MultiBlockInterface)
from .providers import BoundaryProvider
from .topology import BoundaryTopology, PeriodicIdentification

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
class GhostProducer:
    """Generic data-only producer; protocol Handles keep the family extensible."""

    handle: Handle
    protocol: Handle
    dependencies: tuple[Handle, ...] = ()
    capabilities: tuple[Handle, ...] = ()
    periodic: tuple[PeriodicIdentification, ...] = ()
    boundary_providers: tuple[BoundaryProvider, ...] = ()
    interfaces: tuple[MultiBlockInterface, ...] = ()
    operators: tuple[Handle, ...] = ()

    def __post_init__(self) -> None:
        _handle(self.handle, where="GhostProducer.handle",
                kinds=frozenset(("ghost_producer",)))
        _handle(self.protocol, where="GhostProducer.protocol",
                kinds=frozenset(("ghost_producer_protocol",)))
        object.__setattr__(self, "dependencies", _handles(
            self.dependencies, where="GhostProducer.dependencies"))
        object.__setattr__(self, "capabilities", _handles(
            self.capabilities, where="GhostProducer.capabilities",
            kinds=frozenset(("capability",))))
        if not isinstance(self.periodic, tuple) or any(
                not isinstance(row, PeriodicIdentification) for row in self.periodic):
            raise TypeError("GhostProducer.periodic must contain PeriodicIdentification rows")
        if not isinstance(self.boundary_providers, tuple) or any(
                not isinstance(row, BoundaryProvider) for row in self.boundary_providers):
            raise TypeError("GhostProducer.boundary_providers must contain BoundaryProvider rows")
        if not isinstance(self.interfaces, tuple) or any(
                not isinstance(row, MultiBlockInterface) for row in self.interfaces):
            raise TypeError("GhostProducer.interfaces must contain MultiBlockInterface rows")
        object.__setattr__(self, "operators", _handles(
            self.operators, where="GhostProducer.operators",
            kinds=frozenset(("interpolation", "numerical_closure"))))
        object.__setattr__(self, "periodic", tuple(sorted(
            self.periodic, key=lambda row: (row.source.qualified_id, row.target.qualified_id))))
        object.__setattr__(self, "boundary_providers", tuple(sorted(
            self.boundary_providers, key=lambda row: row.qualified_id)))
        object.__setattr__(self, "interfaces", tuple(sorted(
            self.interfaces, key=lambda row: row.qualified_id)))

    @property
    def qualified_id(self) -> str:
        return self.handle.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "producer_type": "ghost",
                "handle": self.handle.canonical_identity(),
                "protocol": self.protocol.canonical_identity(),
                "dependencies": [row.canonical_identity() for row in self.dependencies],
                "capabilities": [row.canonical_identity() for row in self.capabilities],
                "periodic": [row.canonical_identity() for row in self.periodic],
                "boundary_providers": [
                    row.canonical_identity() for row in self.boundary_providers],
                "interfaces": [row.canonical_identity() for row in self.interfaces],
                "operators": [row.canonical_identity() for row in self.operators]}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "ghost_producer", **self.canonical_identity()}


def SameLevelHaloMPI(*, handle: Handle, protocol: Handle, mpi_capability: Handle,
                     dependencies: tuple[Handle, ...] = ()) -> GhostProducer:
    _handle(mpi_capability, where="SameLevelHaloMPI.mpi_capability",
            kinds=frozenset(("capability",)))
    return GhostProducer(handle, protocol, dependencies, (mpi_capability,))


def PeriodicGhost(*, handle: Handle, protocol: Handle,
                  identification: PeriodicIdentification,
                  dependencies: tuple[Handle, ...] = ()) -> GhostProducer:
    if not isinstance(identification, PeriodicIdentification):
        raise TypeError("PeriodicGhost.identification must be a PeriodicIdentification")
    return GhostProducer(handle, protocol, dependencies, periodic=(identification,))


def CoarseFineInterpolation(*, handle: Handle, protocol: Handle, interpolation: Handle,
                            dependencies: tuple[Handle, ...] = (),
                            capabilities: tuple[Handle, ...] = ()) -> GhostProducer:
    _handle(interpolation, where="CoarseFineInterpolation.interpolation",
            kinds=frozenset(("interpolation",)))
    return GhostProducer(handle, protocol, dependencies, capabilities,
                         operators=(interpolation,))


def PhysicalGhost(*, handle: Handle, protocol: Handle, provider: BoundaryProvider,
                  dependencies: tuple[Handle, ...] = ()) -> GhostProducer:
    if not isinstance(provider, BoundaryProvider):
        raise TypeError("PhysicalGhost.provider must be a BoundaryProvider")
    return GhostProducer(handle, protocol, dependencies, boundary_providers=(provider,))


def InterfaceGhost(*, handle: Handle, protocol: Handle, interface: MultiBlockInterface,
                   dependencies: tuple[Handle, ...] = ()) -> GhostProducer:
    if not isinstance(interface, MultiBlockInterface):
        raise TypeError("InterfaceGhost.interface must be a MultiBlockInterface")
    return GhostProducer(handle, protocol, dependencies, interfaces=(interface,))


def NumericalClosure(*, handle: Handle, protocol: Handle, closure: Handle,
                     dependencies: tuple[Handle, ...] = ()) -> GhostProducer:
    _handle(closure, where="NumericalClosure.closure",
            kinds=frozenset(("numerical_closure",)))
    return GhostProducer(handle, protocol, dependencies, operators=(closure,))


@dataclass(frozen=True, slots=True)
class GhostProduction:
    region: GhostRegion
    producer: GhostProducer

    def __post_init__(self) -> None:
        if not isinstance(self.region, GhostRegion) or not isinstance(
                self.producer, GhostProducer):
            raise TypeError("GhostProduction requires GhostRegion and GhostProducer objects")

    def canonical_identity(self) -> dict[str, Any]:
        return {"region": self.region.canonical_identity(),
                "producer": self.producer.canonical_identity()}


def _require_topology_case(handle: Handle, topology: BoundaryTopology, *, where: str) -> None:
    from pops.model import OwnerKind

    case_nodes = tuple(
        node for node in handle.owner_path.nodes if node.kind is OwnerKind.CASE)
    if case_nodes and case_nodes[0] != topology.owner.nodes[0]:
        raise ValueError("%s belongs to foreign Case %r; topology root is %r" %
                         (where, case_nodes[0].name, topology.owner.nodes[0].name))


def _provider_handles(provider: BoundaryProvider) -> tuple[Handle, ...]:
    dependencies = provider.dependencies
    flow = dependencies.representation
    rows = [provider.handle, flow.source, flow.target]
    rows.extend(output.subject for output in provider.outputs)
    rows.extend(output.representation for output in provider.outputs)
    rows.extend(dependencies.states + dependencies.fields + dependencies.time +
                dependencies.runtime_params + dependencies.characteristic.characteristics)
    if flow.converter is not None:
        rows.append(flow.converter)
    return tuple(rows)


def _authenticate_cases(
        topology: BoundaryTopology, coverage: GhostCoverageManifest,
        regions: tuple[GhostRegion, ...],
        productions: tuple[GhostProduction, ...], interfaces: tuple[MultiBlockInterface, ...],
        corner_policies: tuple[CornerPolicy, ...],
        residuals: tuple[BoundaryResidualContribution, ...],
        linearizations: tuple[BoundaryLinearizationContribution, ...]) -> None:
    for name, handle in (
        ("handle", coverage.handle), ("layout_manifest", coverage.layout_manifest),
        ("discretization_manifest", coverage.discretization_manifest),
    ):
        _require_topology_case(handle, topology, where="GhostCoverageManifest.%s" % name)
    for index, region in enumerate(regions):
        for name, handle in (
            ("subject", region.subject), ("layout", region.layout),
            ("selector", region.selector),
            ("stencil", region.depth.stencil.handle),
            ("capability", region.depth.capability.handle),
            ("provider_manifest", region.depth.capability.provider_manifest),
        ):
            _require_topology_case(
                handle, topology, where="GhostRegion[%d].%s" % (index, name))
        if region.boundary is not None and not topology.contains(region.boundary):
            raise ValueError("extra GhostRegion boundary is absent from BoundaryTopology")
    for index, production in enumerate(productions):
        producer = production.producer
        rows = (producer.handle, producer.protocol) + producer.dependencies + \
            producer.capabilities + producer.operators
        for handle in rows:
            _require_topology_case(
                handle, topology, where="GhostProduction[%d].producer" % index)
        for provider in producer.boundary_providers:
            for handle in _provider_handles(provider):
                _require_topology_case(
                    handle, topology, where="GhostProduction[%d].boundary_provider" % index)
        for periodic in producer.periodic:
            if not topology.contains(periodic.source) or not topology.contains(periodic.target):
                raise ValueError("extra periodic ghost producer endpoints are absent from topology")
            if periodic not in topology.periodic:
                raise ValueError("periodic ghost producer orientation is absent from topology")
    for interface in interfaces:
        handles = (
            interface.handle, interface.left.layout, interface.left.discretization,
            interface.left.projection, interface.right.layout, interface.right.discretization,
            interface.right.projection, interface.shared_conservative_flux,
            interface.permutation, interface.mapping,
        )
        for handle in handles:
            _require_topology_case(handle, topology, where="MultiBlockInterface")
        if not topology.contains(interface.left.boundary) or not topology.contains(
                interface.right.boundary):
            raise ValueError("MultiBlockInterface boundaries are absent from topology")
    for policy in corner_policies:
        if policy.resolver is not None:
            _require_topology_case(policy.resolver, topology, where="CornerPolicy.resolver")
        for constraint in policy.constraints:
            for handle in _provider_handles(constraint.source):
                _require_topology_case(handle, topology, where="CornerPolicy.source")
            if constraint.datum is not None:
                _require_topology_case(
                    constraint.datum, topology, where="CornerConstraint.datum")
    for contribution in residuals + linearizations:
        for handle in (
                contribution.handle, contribution.producer,
                getattr(contribution, "residual", None) or contribution.linearization):
            _require_topology_case(handle, topology, where="boundary contribution")


@dataclass(frozen=True, slots=True)
class GhostProducerPlan:
    topology: BoundaryTopology
    coverage: GhostCoverageManifest
    regions: tuple[GhostRegion, ...]
    productions: tuple[GhostProduction, ...]
    corner_policies: tuple[CornerPolicy, ...] = ()
    interfaces: tuple[MultiBlockInterface, ...] = ()
    residual_contributions: tuple[BoundaryResidualContribution, ...] = ()
    linearization_contributions: tuple[BoundaryLinearizationContribution, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.topology, BoundaryTopology):
            raise TypeError("GhostProducerPlan.topology must be a BoundaryTopology")
        if not isinstance(self.coverage, GhostCoverageManifest):
            raise TypeError("GhostProducerPlan.coverage must be a GhostCoverageManifest")
        checks = (
            ("regions", GhostRegion), ("productions", GhostProduction),
            ("corner_policies", CornerPolicy), ("interfaces", MultiBlockInterface),
            ("residual_contributions", BoundaryResidualContribution),
            ("linearization_contributions", BoundaryLinearizationContribution),
        )
        for name, expected in checks:
            rows = getattr(self, name)
            if not isinstance(rows, tuple) or any(not isinstance(row, expected) for row in rows):
                raise TypeError("GhostProducerPlan.%s must contain %s rows" %
                                (name, expected.__name__))
        coverage = [row.overlap_key for row in self.regions]
        if len(coverage) != len(set(coverage)):
            raise ValueError("overlapping ghost regions are forbidden")
        expected_coverage = {row.canonical_id for row in self.coverage.regions}
        authored_coverage = {row.canonical_id for row in self.regions}
        if expected_coverage - authored_coverage:
            raise ValueError("missing expected ghost regions from coverage manifest: %s" %
                             sorted(expected_coverage - authored_coverage))
        if authored_coverage - expected_coverage:
            raise ValueError("extra ghost regions outside coverage manifest: %s" %
                             sorted(authored_coverage - expected_coverage))
        expected = {row.canonical_id: row for row in self.regions}
        produced_ids = [row.region.canonical_id for row in self.productions]
        extra = set(produced_ids) - set(expected)
        if extra:
            raise ValueError("extra ghost production region: %s" % sorted(extra))
        if len(produced_ids) != len(set(produced_ids)):
            raise ValueError("overlapping ghost producers for one region")
        missing = set(expected) - set(produced_ids)
        if missing:
            raise ValueError("missing ghost producer for regions: %s" % sorted(missing))
        assignment = {row.region.canonical_id: row.producer.handle for row in self.productions}
        for production in self.productions:
            boundary = production.region.boundary
            producer = production.producer
            if producer.periodic and (boundary is None or not any(
                    boundary in (row.source, row.target) for row in producer.periodic)):
                raise ValueError("periodic ghost producer does not cover its assigned region")
            if producer.boundary_providers and (boundary is None or not any(
                    output.boundary == boundary
                    for provider in producer.boundary_providers
                    for output in provider.outputs)):
                raise ValueError("physical ghost provider does not cover its assigned region")
            if producer.interfaces and (boundary is None or not any(
                    boundary in (row.left.boundary, row.right.boundary)
                    for row in producer.interfaces)):
                raise ValueError("interface ghost provider does not cover its assigned region")
        for policy in self.corner_policies:
            if policy.corner.canonical_id not in expected:
                raise ValueError("extra CornerPolicy references an unknown ghost region")
        policy_regions = [row.corner.canonical_id for row in self.corner_policies]
        if len(policy_regions) != len(set(policy_regions)):
            raise ValueError("overlapping CornerPolicy declarations for one region")
        interface_ids = [row.qualified_id for row in self.interfaces]
        if len(interface_ids) != len(set(interface_ids)):
            raise ValueError("duplicate MultiBlockInterface identity")
        declared_interfaces = set(self.interfaces)
        producer_interfaces = {
            interface for production in self.productions
            for interface in production.producer.interfaces}
        if producer_interfaces - declared_interfaces:
            raise ValueError("missing MultiBlockInterface declaration used by a producer")
        if declared_interfaces - producer_interfaces:
            raise ValueError("extra unused MultiBlockInterface declaration")
        for contribution in self.residual_contributions + self.linearization_contributions:
            region_id = contribution.region.canonical_id
            if region_id not in expected:
                raise ValueError("extra boundary contribution references an unknown ghost region")
            if assignment[region_id] != contribution.producer:
                raise ValueError("boundary contribution references the wrong region producer")
        for name in ("residual_contributions", "linearization_contributions"):
            rows = getattr(self, name)
            ids = [row.handle.qualified_id for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate %s identity" % name)
        _authenticate_cases(
            self.topology, self.coverage, self.regions, self.productions, self.interfaces,
            self.corner_policies, self.residual_contributions,
            self.linearization_contributions)
        sorting = (
            ("regions", lambda row: row.canonical_id),
            ("productions", lambda row: row.region.canonical_id),
            ("corner_policies", lambda row: row.corner.canonical_id),
            ("interfaces", lambda row: row.qualified_id),
            ("residual_contributions", lambda row: row.handle.qualified_id),
            ("linearization_contributions", lambda row: row.handle.qualified_id),
        )
        for name, key in sorting:
            object.__setattr__(self, name, tuple(sorted(getattr(self, name), key=key)))

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "plan_type": "ghost_producers",
                "topology": self.topology.canonical_identity(),
                "coverage": self.coverage.canonical_identity(),
                "regions": [row.canonical_identity() for row in self.regions],
                "productions": [row.canonical_identity() for row in self.productions],
                "corner_policies": [row.canonical_identity() for row in self.corner_policies],
                "interfaces": [row.canonical_identity() for row in self.interfaces],
                "residual_contributions": [
                    row.canonical_identity() for row in self.residual_contributions],
                "linearization_contributions": [
                    row.canonical_identity() for row in self.linearization_contributions]}

    @property
    def canonical_id(self) -> str:
        return _canonical_id(self.canonical_identity())

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "ghost_producer_plan", "canonical_id": self.canonical_id,
                **self.canonical_identity()}


@dataclass(frozen=True, slots=True, init=False)
class GhostProducerRegistry:
    """Local immutable producer registry; it has no insertion-order fallback."""

    producers: tuple[GhostProducer, ...]

    def __init__(self, *producers: GhostProducer) -> None:
        rows = tuple(producers)
        if any(not isinstance(row, GhostProducer) for row in rows):
            raise TypeError("GhostProducerRegistry accepts GhostProducer objects")
        ids = [row.qualified_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ghost producer identity")
        object.__setattr__(self, "producers", tuple(sorted(
            rows, key=lambda row: row.qualified_id)))

    def resolve(
        self, topology: BoundaryTopology, coverage: GhostCoverageManifest,
        regions: tuple[GhostRegion, ...],
        productions: tuple[GhostProduction, ...], *,
        corner_policies: tuple[CornerPolicy, ...] = (),
        interfaces: tuple[MultiBlockInterface, ...] = (),
        residual_contributions: tuple[BoundaryResidualContribution, ...] = (),
        linearization_contributions: tuple[BoundaryLinearizationContribution, ...] = (),
    ) -> GhostProducerPlan:
        if not isinstance(productions, tuple):
            raise TypeError("GhostProducerRegistry productions must be a tuple")
        registered = set(self.producers)
        used = {row.producer for row in productions if isinstance(row, GhostProduction)}
        if used - registered:
            raise ValueError("extra unregistered ghost producer in production map")
        if registered - used:
            raise ValueError("extra unused ghost producers: %s" % sorted(
                row.qualified_id for row in registered - used))
        return GhostProducerPlan(
            topology, coverage, regions, productions, corner_policies, interfaces,
            residual_contributions, linearization_contributions)


__all__ = [
    "CoarseFineInterpolation", "GhostProducer", "GhostProducerPlan", "GhostProducerRegistry",
    "GhostProduction", "InterfaceGhost", "NumericalClosure", "PeriodicGhost", "PhysicalGhost",
    "SameLevelHaloMPI",
]
