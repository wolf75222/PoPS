"""Data-only boundary providers and exact port resolution."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import TYPE_CHECKING, Any

from .ports import (
    BoundaryDependencies, BoundaryPort, ClosureMode, ConstraintResidual, ExteriorTrace,
    GhostState, NumericalFlux)
from .topology import BoundaryTopology

if TYPE_CHECKING:
    from pops.model import Handle


_SCHEMA_VERSION = 1
_PROVIDER_SCHEMA_VERSION = 2


class BoundaryProviderKind(Enum):
    """Exact immutable law selected by one boundary provider specification."""

    INFLOW = "inflow"
    OUTFLOW = "outflow"
    DIRECTIONAL_TRANSPORT = "directional_transport"
    MIXED = "mixed"
    GHOST_FORMULA = "ghost_formula"
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
    NO_FLUX = "no_flux"
    POST_RIEMANN_FLUX = "post_riemann_flux"
    CONSTRAINT_RESIDUAL = "constraint_residual"


_OUTPUT_CONTRACTS: dict[BoundaryProviderKind, type | tuple[type, ...]] = {
    BoundaryProviderKind.INFLOW: (ExteriorTrace, GhostState),
    BoundaryProviderKind.OUTFLOW: (ExteriorTrace, GhostState),
    BoundaryProviderKind.DIRECTIONAL_TRANSPORT: (ExteriorTrace, GhostState),
    BoundaryProviderKind.MIXED: ConstraintResidual,
    BoundaryProviderKind.GHOST_FORMULA: GhostState,
    BoundaryProviderKind.DIRICHLET: ExteriorTrace,
    BoundaryProviderKind.NEUMANN: ConstraintResidual,
    BoundaryProviderKind.NO_FLUX: NumericalFlux,
    BoundaryProviderKind.POST_RIEMANN_FLUX: NumericalFlux,
    BoundaryProviderKind.CONSTRAINT_RESIDUAL: ConstraintResidual,
}


def _handle(value: Any, *, where: str, kind: str) -> Handle:
    from pops.model import Handle

    if isinstance(value, str) or not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    if value.kind != kind:
        raise TypeError("%s requires Handle.kind=%r" % (where, kind))
    if value.canonical_identity().get("qualified_id") != value.qualified_id:
        raise ValueError("%s Handle identity does not authenticate qualified_id" % where)
    return value


def _require_topology_case(handle: Handle, topology: BoundaryTopology, *, where: str) -> None:
    """Reject references projected through a Case other than the topology root."""
    from pops.model import OwnerKind

    case_nodes = tuple(
        node for node in handle.owner_path.nodes if node.kind is OwnerKind.CASE)
    if case_nodes and case_nodes[0] != topology.owner.nodes[0]:
        raise ValueError(
            "%s belongs to foreign Case %r; BoundaryTopology root is %r"
            % (where, case_nodes[0].name, topology.owner.nodes[0].name))


def _authenticate_port_case(
        port: BoundaryPort, topology: BoundaryTopology, *, where: str) -> None:
    _require_topology_case(port.subject, topology, where="%s.subject" % where)
    _require_topology_case(
        port.representation, topology, where="%s.representation" % where)


def _authenticate_provider_case(
        provider: BoundaryProvider, topology: BoundaryTopology) -> None:
    _require_topology_case(provider.handle, topology, where="BoundaryProvider.handle")
    for index, output in enumerate(provider.outputs):
        _authenticate_port_case(
            output, topology, where="BoundaryProvider.outputs[%d]" % index)
    dependencies = provider.dependencies
    for name in ("states", "fields", "time", "runtime_params"):
        for index, handle in enumerate(getattr(dependencies, name)):
            _require_topology_case(
                handle, topology, where="BoundaryDependencies.%s[%d]" % (name, index))
    flow = dependencies.representation
    _require_topology_case(flow.source, topology, where="RepresentationFlow.source")
    _require_topology_case(flow.target, topology, where="RepresentationFlow.target")
    if flow.converter is not None:
        _require_topology_case(
            flow.converter, topology, where="RepresentationFlow.converter")
    for index, handle in enumerate(dependencies.characteristic.characteristics):
        _require_topology_case(
            handle, topology,
            where="CharacteristicClosure.characteristics[%d]" % index)


@dataclass(frozen=True, slots=True)
class BoundaryProvider:
    """Generic provider specification; algorithms live behind its qualified implementation."""

    handle: Handle
    outputs: tuple[BoundaryPort, ...]
    dependencies: BoundaryDependencies
    kind: BoundaryProviderKind

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryProviderKind):
            raise TypeError("BoundaryProvider.kind must be a BoundaryProviderKind")
        handle_kind = (
            "boundary_flux_provider"
            if self.kind is BoundaryProviderKind.POST_RIEMANN_FLUX
            else "boundary_provider"
        )
        _handle(self.handle, where="BoundaryProvider.handle", kind=handle_kind)
        if not isinstance(self.outputs, tuple) or not self.outputs:
            raise TypeError("BoundaryProvider.outputs must be a non-empty tuple")
        if any(not isinstance(row, BoundaryPort) for row in self.outputs):
            raise TypeError("BoundaryProvider.outputs must contain BoundaryPort objects")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("BoundaryProvider contains double output ports")
        if not isinstance(self.dependencies, BoundaryDependencies):
            raise TypeError("BoundaryProvider.dependencies must be explicit")
        allowed = _OUTPUT_CONTRACTS[self.kind]
        if any(not isinstance(row, allowed) for row in self.outputs):
            allowed_names = (allowed.__name__ if isinstance(allowed, type) else
                             "/".join(row.__name__ for row in allowed))
            raise TypeError(
                "BoundaryProvider kind %r requires typed %s outputs"
                % (self.kind.value, allowed_names)
            )
        if self.kind is BoundaryProviderKind.DIRECTIONAL_TRANSPORT and \
                self.dependencies.characteristic.mode is not ClosureMode.DIRECTIONAL:
            raise ValueError(
                "directional_transport provider requires explicit directional characteristic "
                "closure"
            )
        if self.kind is not BoundaryProviderKind.DIRECTIONAL_TRANSPORT and \
                self.dependencies.characteristic.mode is ClosureMode.DIRECTIONAL:
            raise ValueError(
                "directional characteristic closure requires a directional_transport provider"
            )
        target = self.dependencies.representation.target
        if any(row.representation != target for row in self.outputs):
            raise ValueError("provider output representation must match RepresentationFlow.target")
        object.__setattr__(self, "outputs", tuple(sorted(
            self.outputs, key=lambda row: row.canonical_id)))

    @property
    def qualified_id(self) -> str:
        return self.handle.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _PROVIDER_SCHEMA_VERSION, "provider_type": "boundary",
                "provider_kind": self.kind.value,
                "handle": self.handle.canonical_identity(),
                "outputs": [row.canonical_identity() for row in self.outputs],
                "dependencies": self.dependencies.canonical_identity()}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "boundary_provider", **self.canonical_identity()}


def _factory(name: str, kind: BoundaryProviderKind, handle: Any, outputs: Any, dependencies: Any,
             allowed: type | tuple[type, ...], *, directional: bool = False) -> BoundaryProvider:
    if not isinstance(outputs, tuple) or not outputs or any(
            not isinstance(row, allowed) for row in outputs):
        allowed_names = (allowed.__name__ if isinstance(allowed, type) else
                         "/".join(row.__name__ for row in allowed))
        raise TypeError("%s outputs must be typed %s ports" % (name, allowed_names))
    if not isinstance(dependencies, BoundaryDependencies):
        raise TypeError("%s dependencies must be BoundaryDependencies" % name)
    if directional and dependencies.characteristic.mode is not ClosureMode.DIRECTIONAL:
        raise ValueError("DirectionalTransport requires explicit directional characteristic closure")
    return BoundaryProvider(handle, outputs, dependencies, kind)


def Inflow(*, handle: Any, outputs: tuple[BoundaryPort, ...],
           dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory(
        "Inflow", BoundaryProviderKind.INFLOW, handle, outputs, dependencies,
        (ExteriorTrace, GhostState))


def Outflow(*, handle: Any, outputs: tuple[BoundaryPort, ...],
            dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory(
        "Outflow", BoundaryProviderKind.OUTFLOW, handle, outputs, dependencies,
        (ExteriorTrace, GhostState))


def DirectionalTransport(*, handle: Any, outputs: tuple[BoundaryPort, ...],
                         dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory("DirectionalTransport", BoundaryProviderKind.DIRECTIONAL_TRANSPORT,
                    handle, outputs, dependencies,
                    (ExteriorTrace, GhostState), directional=True)


def Mixed(*, handle: Any, outputs: tuple[BoundaryPort, ...],
          dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory(
        "Mixed", BoundaryProviderKind.MIXED, handle, outputs, dependencies, ConstraintResidual)


def GhostFormula(*, handle: Any, outputs: tuple[BoundaryPort, ...],
                 dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory(
        "GhostFormula", BoundaryProviderKind.GHOST_FORMULA, handle, outputs, dependencies,
        GhostState)


def Dirichlet(*, handle: Any, outputs: tuple[BoundaryPort, ...],
              dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory(
        "Dirichlet", BoundaryProviderKind.DIRICHLET, handle, outputs, dependencies,
        ExteriorTrace)


def Neumann(*, handle: Any, outputs: tuple[BoundaryPort, ...],
            dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory(
        "Neumann", BoundaryProviderKind.NEUMANN, handle, outputs, dependencies,
        ConstraintResidual)


def NoFlux(*, handle: Any, output: NumericalFlux,
           dependencies: BoundaryDependencies) -> BoundaryProvider:
    if not isinstance(output, NumericalFlux):
        raise TypeError("NoFlux satisfies NumericalFlux only")
    return BoundaryProvider(handle, (output,), dependencies, BoundaryProviderKind.NO_FLUX)


def PostRiemannFlux(*, handle: Any, output: NumericalFlux,
                    dependencies: BoundaryDependencies) -> BoundaryProvider:
    """Bind one exact native transformation of an already evaluated outward flux.

    The provider owns neither reconstruction nor the Riemann solve.  Its
    ``boundary_flux_provider`` Handle resolves only to the typed
    ``BoundaryFlux.transform_faces`` ABI, so it cannot be mistaken for a ghost
    producer or for the shared-interface ``NumericalFlux.evaluate_faces`` route.
    """
    if not isinstance(output, NumericalFlux):
        raise TypeError("PostRiemannFlux satisfies NumericalFlux only")
    return BoundaryProvider(
        handle, (output,), dependencies, BoundaryProviderKind.POST_RIEMANN_FLUX)


@dataclass(frozen=True, slots=True)
class ResolvedBoundaryBinding:
    need: BoundaryPort
    provider: BoundaryProvider

    def __post_init__(self) -> None:
        if not isinstance(self.need, BoundaryPort) or not isinstance(
                self.provider, BoundaryProvider):
            raise TypeError("resolved boundary binding requires a need and BoundaryProvider")
        if self.need not in self.provider.outputs:
            raise ValueError("resolved boundary binding provider does not satisfy its need")

    def canonical_identity(self) -> dict[str, Any]:
        return {"need": self.need.canonical_identity(),
                "provider": self.provider.canonical_identity()}


@dataclass(frozen=True, slots=True)
class ResolvedBoundaryPlan:
    topology: BoundaryTopology
    needs: tuple[BoundaryPort, ...]
    bindings: tuple[ResolvedBoundaryBinding, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.topology, BoundaryTopology):
            raise TypeError("ResolvedBoundaryPlan.topology must be a BoundaryTopology")
        if not isinstance(self.needs, tuple) or not isinstance(self.bindings, tuple):
            raise TypeError("ResolvedBoundaryPlan needs/bindings must be tuples")
        if tuple(row.need for row in self.bindings) != self.needs:
            raise ValueError("ResolvedBoundaryPlan bindings must exactly cover canonical needs")

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "plan_type": "boundary_providers",
                "topology": self.topology.canonical_identity(),
                "needs": [row.canonical_identity() for row in self.needs],
                "bindings": [row.canonical_identity() for row in self.bindings]}

    @property
    def canonical_id(self) -> str:
        raw = json.dumps(self.canonical_identity(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "resolved_boundary_plan", "canonical_id": self.canonical_id,
                **self.canonical_identity()}


@dataclass(frozen=True, slots=True, init=False)
class BoundaryProviderRegistry:
    """Local immutable provider set; resolution has no process-global registry."""

    providers: tuple[BoundaryProvider, ...]

    def __init__(self, *providers: BoundaryProvider) -> None:
        rows = tuple(providers)
        if any(not isinstance(row, BoundaryProvider) for row in rows):
            raise TypeError("BoundaryProviderRegistry accepts BoundaryProvider objects")
        ids = [row.qualified_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("double boundary provider identity")
        object.__setattr__(self, "providers", tuple(sorted(
            rows, key=lambda row: row.qualified_id)))

    def resolve(self, topology: Any, needs: Any) -> ResolvedBoundaryPlan:
        if not isinstance(topology, BoundaryTopology):
            raise TypeError("boundary resolution requires a BoundaryTopology")
        if not isinstance(needs, tuple) or any(not isinstance(row, BoundaryPort) for row in needs):
            raise TypeError("boundary needs must be a tuple of BoundaryPort objects")
        if len(needs) != len(set(needs)):
            raise ValueError("double boundary need")
        for index, need in enumerate(needs):
            _authenticate_port_case(
                need, topology, where="boundary needs[%d]" % index)
            if not topology.contains(need.boundary):
                raise ValueError("extra boundary need references an undeclared boundary")
            if topology.is_periodic(need.boundary):
                raise ValueError("periodic+physical boundary need is forbidden")
        for provider in self.providers:
            _authenticate_provider_case(provider, topology)
        produced = [output for provider in self.providers for output in provider.outputs]
        for output in produced:
            if not topology.contains(output.boundary):
                raise ValueError("extra provider output references an undeclared boundary")
            if topology.is_periodic(output.boundary):
                raise ValueError("periodic+physical provider output is forbidden")
        extra = set(produced) - set(needs)
        if extra:
            raise ValueError("extra boundary provider outputs: %s" %
                             sorted(row.canonical_id for row in extra))
        bindings = []
        for need in needs:
            matches = [provider for provider in self.providers if need in provider.outputs]
            if not matches:
                raise ValueError("missing boundary provider for %s" % need.canonical_id)
            if len(matches) > 1:
                raise ValueError("ambiguous boundary providers for %s: %s" %
                                 (need.canonical_id,
                                  sorted(row.qualified_id for row in matches)))
            bindings.append(ResolvedBoundaryBinding(need, matches[0]))
        bindings.sort(key=lambda row: row.need.canonical_id)
        return ResolvedBoundaryPlan(
            topology, tuple(sorted(needs, key=lambda row: row.canonical_id)), tuple(bindings))


__all__ = [
    "BoundaryProvider", "BoundaryProviderKind", "BoundaryProviderRegistry", "DirectionalTransport",
    "Dirichlet",
    "GhostFormula", "Inflow", "Mixed", "Neumann", "NoFlux", "Outflow", "PostRiemannFlux",
    "ResolvedBoundaryBinding", "ResolvedBoundaryPlan",
]
