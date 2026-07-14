"""Immutable ghost-region producer plans with exact, order-independent resolution."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any

from .component_binding import BoundaryComponentBinding
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
            interface.permutation.handle, interface.mapping.handle,
        )
        for handle in handles:
            _require_topology_case(handle, topology, where="MultiBlockInterface")
        local_endpoints = sum((
            topology.contains(interface.left.boundary),
            topology.contains(interface.right.boundary),
        ))
        if local_endpoints != 1:
            raise ValueError(
                "each block GhostProducerPlan must own exactly one MultiBlockInterface endpoint")
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
    execution_authority: Any = None
    component_bindings: tuple[BoundaryComponentBinding, ...] = ()

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
        bindings = tuple(self.component_bindings)
        if any(type(row) is not BoundaryComponentBinding for row in bindings):
            raise TypeError(
                "GhostProducerPlan.component_bindings must contain exact "
                "BoundaryComponentBinding rows"
            )
        binding_targets = [row.target for row in bindings]
        if len(binding_targets) != len(set(binding_targets)):
            raise ValueError("GhostProducerPlan has multiple native components for one Handle")
        referenced_targets = set(self._component_targets())
        extra_bindings = set(binding_targets) - referenced_targets
        if extra_bindings:
            raise ValueError(
                "GhostProducerPlan component binding targets unused Handle(s) %s"
                % sorted(row.qualified_id for row in extra_bindings)
            )
        object.__setattr__(self, "component_bindings", tuple(sorted(
            bindings, key=lambda row: row.target.qualified_id)))
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
        if self.execution_authority is not None:
            for protocol in ("canonical_identity", "compile_boundary_data",
                             "runtime_boundary_data"):
                if not callable(getattr(self.execution_authority, protocol, None)):
                    raise TypeError(
                        "GhostProducerPlan execution authority must implement %s()" % protocol
                    )
        producer_handles = {row.producer.handle for row in self.productions}
        for production in self.productions:
            ordering_dependencies = {
                dependency for dependency in production.producer.dependencies
                if dependency.kind == "ghost_producer"
            }
            missing_dependencies = ordering_dependencies - producer_handles
            if missing_dependencies:
                raise ValueError(
                    "GhostProducerPlan producer %s depends on absent ghost producer(s) %s"
                    % (production.producer.qualified_id, sorted(
                        row.qualified_id for row in missing_dependencies))
                )
            if production.producer.handle in ordering_dependencies:
                raise ValueError(
                    "GhostProducerPlan producer %s depends on itself"
                    % production.producer.qualified_id
                )
        # Resolve-time construction must reject dependency cycles.  Data dependencies (states,
        # fields, parameters) deliberately remain outside this producer-order graph.
        self.execution_order()

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
                    row.canonical_identity() for row in self.linearization_contributions],
                "component_bindings": [
                    row.canonical_identity() for row in self.component_bindings],
                "execution_authority": (
                    None if self.execution_authority is None else
                    self.execution_authority.canonical_identity())}

    @property
    def canonical_id(self) -> str:
        return _canonical_id(self.canonical_identity())

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "ghost_producer_plan", "canonical_id": self.canonical_id,
                **self.canonical_identity()}

    def resolve_for_numerics(self, context: Any) -> GhostProducerPlan:
        """A fully canonical plan is already resolved; authenticate its Case root only."""
        owner = getattr(context, "owner", None)
        if owner != self.topology.owner:
            raise ValueError("GhostProducerPlan belongs to a different Case owner")
        return self

    def ghost_plan_composer_capability(self) -> dict[str, Any]:
        return {"schema_version": 1, "scope": "self"}

    def compose_ghost_plan(self, context: Any) -> GhostProducerPlan:
        from .composition import GhostPlanCompositionContext

        if not isinstance(context, GhostPlanCompositionContext):
            raise TypeError("GhostProducerPlan composition requires GhostPlanCompositionContext")
        if context.authorities != (self,):
            raise ValueError(
                "a canonical GhostProducerPlan composes only itself; multiple authorities require "
                "an explicit scope='all' composer"
            )
        return self

    def compile_boundary_data(self) -> dict[str, Any]:
        """Prove that every authored producer has an executable lowering before compilation."""
        if self.execution_authority is None:
            raise NotImplementedError(
                "GhostProducerPlan is complete structurally but has no executable lowering "
                "authority; compose it from a numerical boundary provider"
            )
        data = self.execution_authority.compile_boundary_data()
        if type(data) is not dict:
            raise TypeError("boundary compile authority must return a dict")
        periodic = [row for production in self.productions
                    for row in production.producer.periodic]
        for identification in periodic:
            orientation = identification.orientation
            dimension = len(orientation.permutation)
            if orientation.permutation != tuple(range(dimension)) or any(
                    sign != 1 for sign in orientation.signs):
                raise NotImplementedError(
                    "the installed native provider does not execute signed/permuted periodic "
                    "identifications; select an oriented-periodic provider before compile"
                )
        if self.interfaces:
            depth = data.get("required_depth")
            if isinstance(depth, bool) or not isinstance(depth, int) or depth != 1:
                raise NotImplementedError(
                    "shared-interface NumericalFlux requires a prepared trace provider for "
                    "reconstruction order > 1; the current scheduler authenticates cell-average "
                    "traces only (required_depth=1). Physical GhostBoundary providers remain "
                    "available at higher order."
                )
            ncomp = data.get("ncomp")
            if isinstance(ncomp, bool) or not isinstance(ncomp, int) or ncomp < 1:
                raise TypeError(
                    "shared-interface lowering requires an authenticated positive state ncomp"
                )
            incompatible = [
                row.qualified_id for row in self.interfaces
                if len(row.permutation.right_component_for_left) != ncomp
            ]
            if incompatible:
                raise ValueError(
                    "shared-interface component permutation must exactly cover all %d state "
                    "components: %s" % (ncomp, sorted(incompatible))
                )
            missing = [
                row.shared_conservative_flux for row in self.interfaces
                if row.shared_conservative_flux not in self._binding_map()
            ]
            if missing:
                raise NotImplementedError(
                    "shared-interface conservative flux Handle(s) require qualified "
                    "NumericalFlux components: %s"
                    % sorted(row.qualified_id for row in missing)
                )
        if self.corner_policies:
            missing = [
                row.resolver for row in self.corner_policies
                if row.resolver is not None and row.resolver not in self._binding_map()
            ]
            if missing:
                raise NotImplementedError(
                    "explicit corner resolver Handle(s) require qualified GhostBoundary "
                    "components: %s" % sorted(row.qualified_id for row in missing)
                )
        closures = [
            operator for production in self.productions
            for operator in production.producer.operators
            if operator.kind == "numerical_closure"
        ]
        if closures:
            missing = [row for row in closures if row not in self._binding_map()]
            if missing:
                raise NotImplementedError(
                    "numerical boundary closure Handle(s) require qualified GhostBoundary "
                    "components: %s" % sorted(row.qualified_id for row in missing)
                )
        if self.residual_contributions or self.linearization_contributions:
            residual_producers = {row.producer for row in self.residual_contributions}
            linear_producers = {row.producer for row in self.linearization_contributions}
            if residual_producers != linear_producers:
                raise ValueError(
                    "implicit boundary producers must contribute both residual and linearization"
                )
            bindings = self._binding_map()
            missing = [
                handle for handle in (
                    *(row.residual for row in self.residual_contributions),
                    *(row.linearization for row in self.linearization_contributions),
                ) if handle not in bindings
            ]
            if missing:
                raise NotImplementedError(
                    "implicit boundary Handle(s) require qualified FieldBoundaryClosure "
                    "components: %s" % sorted(row.qualified_id for row in missing)
                )
            residual_by_key = {
                (row.region.canonical_id, row.producer): bindings[row.residual]
                for row in self.residual_contributions
            }
            linear_by_key = {
                (row.region.canonical_id, row.producer): bindings[row.linearization]
                for row in self.linearization_contributions
            }
            for key in sorted(residual_by_key, key=lambda row: (row[0], row[1].qualified_id)):
                residual = residual_by_key[key]
                linear = linear_by_key[key]
                if (residual.component_id, residual.component_manifest_identity) != (
                        linear.component_id, linear.component_manifest_identity):
                    raise ValueError(
                        "one implicit boundary residual/JVP pair must use the same exact "
                        "FieldBoundaryClosure component"
                    )
        owned_boundaries = {
            production.region.boundary for production in self.productions
            if production.region.boundary is not None
        }
        omitted_interface_faces = sorted({
            2 * side.boundary.orientation.axis + (
                0 if side.boundary.orientation.side.value == "lower" else 1)
            for interface in self.interfaces
            for side in (interface.left, interface.right)
            if side.boundary in owned_boundaries
        })
        return {
            **data,
            "ghost_plan_identity": self.canonical_id,
            "producer_order": [
                row.producer.qualified_id for row in self.execution_order()],
            "corner_policies": [
                row.canonical_identity() for row in self.corner_policies],
            "interfaces": [row.canonical_identity() for row in self.interfaces],
            "interface_endpoints": [
                {
                    "interface": row.qualified_id,
                    "owned_sides": [
                        name for name, side in (("left", row.left), ("right", row.right))
                        if side.boundary in owned_boundaries
                    ],
                }
                for row in self.interfaces
            ],
            "omitted_interface_faces": omitted_interface_faces,
            "interface_component_bindings": [
                {
                    "interface": row.canonical_identity(),
                    "component": self._binding_map()[
                        row.shared_conservative_flux].canonical_identity(),
                }
                for row in self.interfaces
            ],
            "residual_contributions": [
                row.canonical_identity() for row in self.residual_contributions],
            "linearization_contributions": [
                row.canonical_identity() for row in self.linearization_contributions],
            "component_bindings": [
                row.canonical_identity() for row in self.component_bindings],
            "component_region_templates": self._component_region_rows(None),
        }

    def runtime_boundary_data(self, params: Any) -> dict[str, Any]:
        compiled = self.compile_boundary_data()
        data = self.execution_authority.runtime_boundary_data(params)
        if type(data) is not dict:
            raise TypeError("boundary runtime authority must return a dict")
        result = dict(data)
        result["ghost_plan_identity"] = self.canonical_id
        producer_order = [
            row.producer.qualified_id for row in self.execution_order()
        ]
        result["producer_order"] = producer_order
        result["component_regions"] = self._component_region_rows(params)
        # Shared-interface installation happens after the native blocks exist, whereas physical
        # ghost producers are prepared before their closures are built.  Retain the exact same
        # canonical declarations and qualified component bindings in the runtime payload so the
        # post-block installer never has to recover topology from an authoring object or select a
        # NumericalFlux component by interface uniqueness.
        result["interfaces"] = compiled["interfaces"]
        result["interface_endpoints"] = compiled["interface_endpoints"]
        result["interface_component_bindings"] = compiled[
            "interface_component_bindings"]
        result["identity"] = _canonical_id({
            "schema_version": _SCHEMA_VERSION,
            "prepared_authority": data.get("identity"),
            "ghost_plan_identity": self.canonical_id,
            "producer_order": producer_order,
        })
        return result

    def _component_targets(self) -> tuple[Handle, ...]:
        rows = []
        for production in self.productions:
            rows.extend(provider.handle for provider in production.producer.boundary_providers)
            rows.extend(
                operator for operator in production.producer.operators
                if operator.kind == "numerical_closure"
            )
        rows.extend(
            policy.resolver for policy in self.corner_policies if policy.resolver is not None)
        rows.extend(row.shared_conservative_flux for row in self.interfaces)
        rows.extend(row.residual for row in self.residual_contributions)
        rows.extend(row.linearization for row in self.linearization_contributions)
        return tuple(sorted(set(rows), key=lambda row: row.qualified_id))

    def _binding_map(self) -> dict[Handle, BoundaryComponentBinding]:
        return {row.target: row for row in self.component_bindings}

    def require_component_inputs(self, components: tuple[Any, ...]) -> None:
        """Authenticate every bound component against the explicit resolve input tuple."""
        by_id = {}
        for component in components:
            component_id = getattr(component, "component_id", None)
            if component_id is None:
                component_id = component.component_manifest.component_id
            by_id[component_id] = component
        for binding in self.component_bindings:
            try:
                component = by_id[binding.component_id]
            except KeyError:
                raise ValueError(
                    "boundary Handle %s requires exact component %r in resolve(components=)"
                    % (binding.target.qualified_id, binding.component_id)
                ) from None
            binding.require_component(component)

    def _component_region_rows(self, params: Any | None) -> list[dict[str, Any]]:
        """Return bind-time exact component invocations, including empty dependency tables."""
        from collections.abc import Mapping

        if params is not None and not isinstance(params, Mapping):
            raise TypeError("boundary component lowering requires resolved BindSchema values")
        bindings = self._binding_map()
        rows = []

        def scalar_rows(handles: tuple[Handle, ...]) -> list[dict[str, Any]]:
            result = []
            for handle in handles:
                if params is None:
                    result.append({
                        "qualified_id": handle.qualified_id,
                        "handle": handle.canonical_identity(),
                    })
                    continue
                if handle not in params:
                    raise ValueError(
                        "boundary component parameter %s is absent from resolved BindSchema values"
                        % handle.qualified_id
                    )
                value = params[handle]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError(
                        "boundary component parameter %s must bind to a real scalar"
                        % handle.qualified_id
                    )
                result.append({"qualified_id": handle.qualified_id, "value": float(value)})
            return result

        def region_data(region: GhostRegion, *, kind: str | None = None,
                        boundaries: tuple[Any, ...] = ()) -> dict[str, Any]:
            selected = boundaries
            if region.boundary is not None:
                selected = (region.boundary,)
            axes = tuple(row.orientation.axis for row in selected)
            sides = tuple(row.orientation.outward_sign for row in selected)
            codimension = len(axes)
            if codimension == 0:
                raise ValueError(
                    "component boundary region %s has no exact oriented boundary axes"
                    % region.selector.qualified_id
                )
            region_kind = kind or ("face" if codimension == 1 else "corner")
            return {
                "kind": region_kind,
                "dimension": len(self.topology.boundaries) // 2,
                "codimension": codimension,
                "axes": list(axes),
                "sides": list(sides),
                "region_identity": region.selector.qualified_id,
                "layout_identity": region.layout.qualified_id,
            }

        for production in self.productions:
            region = production.region
            for provider in production.producer.boundary_providers:
                binding = bindings.get(provider.handle)
                if binding is None:
                    continue
                dependencies = provider.dependencies
                rows.append({
                    **binding.canonical_identity(),
                    "producer_identity": production.producer.qualified_id,
                    "state_identity": region.subject.qualified_id,
                    "ghost_identity": region.selector.qualified_id,
                    "region": region_data(region),
                    "states": [row.qualified_id for row in dependencies.states],
                    "directions": [],
                    "fields": [row.qualified_id for row in dependencies.fields],
                    "parameters": scalar_rows(dependencies.runtime_params),
                    "outputs": [row.subject.qualified_id for row in provider.outputs],
                    "rate": None,
                    "nonlinear_iterate": None,
                })
            for operator in production.producer.operators:
                binding = bindings.get(operator)
                if binding is None:
                    continue
                rows.append({
                    **binding.canonical_identity(),
                    "producer_identity": production.producer.qualified_id,
                    "state_identity": region.subject.qualified_id,
                    "ghost_identity": region.selector.qualified_id,
                    "region": region_data(region),
                    "states": [], "directions": [], "fields": [], "parameters": [],
                    "outputs": [region.subject.qualified_id],
                    "rate": None, "nonlinear_iterate": None,
                })
        for policy in self.corner_policies:
            if policy.resolver is None or policy.resolver not in bindings:
                continue
            dependencies = tuple(
                handle for constraint in policy.constraints
                for handle in (
                    constraint.source.dependencies.states +
                    constraint.source.dependencies.fields
                )
            )
            runtime_params = tuple(sorted({
                handle for constraint in policy.constraints
                for handle in constraint.source.dependencies.runtime_params
            }, key=lambda row: row.qualified_id))
            boundaries = tuple(
                output.boundary for constraint in policy.constraints
                for output in constraint.source.outputs
            )
            binding = bindings[policy.resolver]
            rows.append({
                **binding.canonical_identity(),
                "producer_identity": policy.resolver.qualified_id,
                "state_identity": policy.corner.subject.qualified_id,
                "ghost_identity": policy.corner.selector.qualified_id,
                "region": region_data(policy.corner, kind="corner", boundaries=boundaries),
                "states": sorted({
                    row.qualified_id for row in dependencies if row.kind == "state"}),
                "directions": [],
                "fields": sorted({
                    row.qualified_id for row in dependencies if row.kind == "field"}),
                "parameters": scalar_rows(runtime_params),
                "outputs": [policy.corner.subject.qualified_id],
                "rate": None, "nonlinear_iterate": None,
            })
        for contribution, target_name in (
                *((row, "residual") for row in self.residual_contributions),
                *((row, "linearization") for row in self.linearization_contributions)):
            target = getattr(contribution, target_name)
            binding = bindings.get(target)
            if binding is None:
                continue
            rows.append({
                **binding.canonical_identity(),
                "producer_identity": contribution.producer.qualified_id,
                "state_identity": contribution.region.subject.qualified_id,
                "ghost_identity": contribution.region.selector.qualified_id,
                "region": region_data(contribution.region),
                "states": [contribution.region.subject.qualified_id],
                "directions": ([contribution.region.subject.qualified_id]
                               if target_name == "linearization" else []),
                "fields": [], "parameters": [],
                "outputs": [contribution.handle.qualified_id],
                "rate": None,
                "nonlinear_iterate": contribution.region.subject.qualified_id,
            })
        rows.sort(key=lambda row: (row["target"]["qualified_id"],
                                   row["region"]["region_identity"]))
        return rows

    def execution_order(self) -> tuple[GhostProduction, ...]:
        """Stable topological producer order; dependency cycles and foreign edges fail closed."""
        by_handle = {row.producer.handle: row for row in self.productions}
        pending = set(by_handle)
        ordered = []
        while pending:
            ready = sorted(
                (handle for handle in pending
                 if all(dependency not in pending for dependency in
                        by_handle[handle].producer.dependencies
                        if dependency.kind == "ghost_producer")),
                key=lambda handle: handle.qualified_id,
            )
            if not ready:
                raise ValueError("GhostProducerPlan producer dependency graph contains a cycle")
            for handle in ready:
                ordered.append(by_handle[handle])
                pending.remove(handle)
        return tuple(ordered)


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
        execution_authority: Any = None,
        component_bindings: tuple[BoundaryComponentBinding, ...] = (),
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
            residual_contributions, linearization_contributions, execution_authority,
            component_bindings)


__all__ = [
    "CoarseFineInterpolation", "GhostProducer", "GhostProducerPlan", "GhostProducerRegistry",
    "GhostProduction", "InterfaceGhost", "NumericalClosure", "PeriodicGhost", "PhysicalGhost",
    "SameLevelHaloMPI",
]
