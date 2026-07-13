"""Data-only boundary providers and exact port resolution."""
from __future__ import annotations

from dataclasses import dataclass
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


def _handle(value: Any, *, where: str, kind: str) -> Handle:
    from pops.model import Handle

    if isinstance(value, str) or not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    if value.kind != kind:
        raise TypeError("%s requires Handle.kind=%r" % (where, kind))
    if value.canonical_identity().get("qualified_id") != value.qualified_id:
        raise ValueError("%s Handle identity does not authenticate qualified_id" % where)
    return value


@dataclass(frozen=True, slots=True)
class BoundaryProvider:
    """Generic provider specification; algorithms live behind its qualified implementation."""

    handle: Handle
    outputs: tuple[BoundaryPort, ...]
    dependencies: BoundaryDependencies

    def __post_init__(self) -> None:
        _handle(self.handle, where="BoundaryProvider.handle", kind="boundary_provider")
        if not isinstance(self.outputs, tuple) or not self.outputs:
            raise TypeError("BoundaryProvider.outputs must be a non-empty tuple")
        if any(not isinstance(row, BoundaryPort) for row in self.outputs):
            raise TypeError("BoundaryProvider.outputs must contain BoundaryPort objects")
        if len(self.outputs) != len(set(self.outputs)):
            raise ValueError("BoundaryProvider contains double output ports")
        if not isinstance(self.dependencies, BoundaryDependencies):
            raise TypeError("BoundaryProvider.dependencies must be explicit")
        target = self.dependencies.representation.target
        if any(row.representation != target for row in self.outputs):
            raise ValueError("provider output representation must match RepresentationFlow.target")
        object.__setattr__(self, "outputs", tuple(sorted(
            self.outputs, key=lambda row: row.canonical_id)))

    @property
    def qualified_id(self) -> str:
        return self.handle.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {"schema_version": _SCHEMA_VERSION, "provider_type": "boundary",
                "handle": self.handle.canonical_identity(),
                "outputs": [row.canonical_identity() for row in self.outputs],
                "dependencies": self.dependencies.canonical_identity()}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "boundary_provider", **self.canonical_identity()}


def _factory(name: str, handle: Any, outputs: Any, dependencies: Any,
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
    return BoundaryProvider(handle, outputs, dependencies)


def Inflow(*, handle: Any, outputs: tuple[BoundaryPort, ...],
           dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory("Inflow", handle, outputs, dependencies, (ExteriorTrace, GhostState))


def Outflow(*, handle: Any, outputs: tuple[BoundaryPort, ...],
            dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory("Outflow", handle, outputs, dependencies, (ExteriorTrace, GhostState))


def DirectionalTransport(*, handle: Any, outputs: tuple[BoundaryPort, ...],
                         dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory("DirectionalTransport", handle, outputs, dependencies,
                    (ExteriorTrace, GhostState), directional=True)


def Mixed(*, handle: Any, outputs: tuple[BoundaryPort, ...],
          dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory("Mixed", handle, outputs, dependencies, ConstraintResidual)


def GhostFormula(*, handle: Any, outputs: tuple[BoundaryPort, ...],
                 dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory("GhostFormula", handle, outputs, dependencies, GhostState)


def Dirichlet(*, handle: Any, outputs: tuple[BoundaryPort, ...],
              dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory("Dirichlet", handle, outputs, dependencies, ExteriorTrace)


def Neumann(*, handle: Any, outputs: tuple[BoundaryPort, ...],
            dependencies: BoundaryDependencies) -> BoundaryProvider:
    return _factory("Neumann", handle, outputs, dependencies, ConstraintResidual)


def NoFlux(*, handle: Any, output: NumericalFlux,
           dependencies: BoundaryDependencies) -> BoundaryProvider:
    if not isinstance(output, NumericalFlux):
        raise TypeError("NoFlux satisfies NumericalFlux only")
    return BoundaryProvider(handle, (output,), dependencies)


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
        for need in needs:
            if not topology.contains(need.boundary):
                raise ValueError("extra boundary need references an undeclared boundary")
            if topology.is_periodic(need.boundary):
                raise ValueError("periodic+physical boundary need is forbidden")
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
    "BoundaryProvider", "BoundaryProviderRegistry", "DirectionalTransport", "Dirichlet",
    "GhostFormula", "Inflow", "Mixed", "Neumann", "NoFlux", "Outflow",
    "ResolvedBoundaryBinding", "ResolvedBoundaryPlan",
]
