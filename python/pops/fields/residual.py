"""Resolved field residual contracts, including exact boundary dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pops.model import Handle
    from pops.time import TimePoint


def _handle(value: Any, *, where: str, kinds: frozenset[str] | None = None) -> Handle:
    from pops.model import Handle

    if isinstance(value, str) or not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    if kinds is not None and value.kind not in kinds:
        raise TypeError("%s requires Handle.kind in %s, got %r" %
                        (where, sorted(kinds), value.kind))
    return value


def _handles(values: Any, *, where: str, kinds: frozenset[str]) -> tuple[Handle, ...]:
    if not isinstance(values, tuple):
        raise TypeError("%s must be a tuple" % where)
    rows = tuple(_handle(row, where=where, kinds=kinds) for row in values)
    if len(rows) != len(set(rows)):
        raise ValueError("%s contains duplicate dependencies" % where)
    return tuple(sorted(rows, key=lambda row: row.qualified_id))


def _point(value: Any, *, where: str) -> TimePoint:
    from pops.time import Clock, TimePoint

    if type(value) is not TimePoint or type(value.clock) is not Clock or value.clock.owner is None:
        raise TypeError("%s requires an exact TimePoint on an owner-qualified Clock" % where)
    return value


def _canonical_id(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FieldResidualDependencies:
    iterate: Handle
    states: tuple[Handle, ...]
    fields: tuple[Handle, ...]
    point: TimePoint
    time_sources: tuple[Handle, ...] = ()
    parameters: tuple[Handle, ...] = ()

    def __post_init__(self) -> None:
        _handle(self.iterate, where="FieldResidualDependencies.iterate",
                kinds=frozenset(("field", "aux")))
        object.__setattr__(self, "states", _handles(
            self.states, where="FieldResidualDependencies.states",
            kinds=frozenset(("state",))))
        object.__setattr__(self, "fields", _handles(
            self.fields, where="FieldResidualDependencies.fields",
            kinds=frozenset(("field", "aux"))))
        if self.iterate in self.fields:
            raise ValueError("residual iterate is separate from field dependencies")
        object.__setattr__(self, "time_sources", _handles(
            self.time_sources, where="FieldResidualDependencies.time_sources",
            kinds=frozenset(("time",))))
        object.__setattr__(self, "parameters", _handles(
            self.parameters, where="FieldResidualDependencies.parameters",
            kinds=frozenset(("parameter",))))
        _point(self.point, where="FieldResidualDependencies.point")

    @classmethod
    def merged(cls, *dependencies: FieldResidualDependencies) -> FieldResidualDependencies:
        if not dependencies or any(not isinstance(row, cls) for row in dependencies):
            raise TypeError("dependency merge requires FieldResidualDependencies objects")
        iterate = dependencies[0].iterate
        point = dependencies[0].point
        if any(row.iterate != iterate for row in dependencies):
            raise ValueError("all residual and boundary dependencies require the exact same iterate")
        if any(row.point != point for row in dependencies):
            raise ValueError("all residual and boundary dependencies require the exact TimePoint")
        states = tuple({value for row in dependencies for value in row.states})
        fields = tuple({value for row in dependencies for value in row.fields})
        time_sources = tuple({value for row in dependencies for value in row.time_sources})
        parameters = tuple({value for row in dependencies for value in row.parameters})
        return cls(iterate, states, fields, point, time_sources, parameters)

    def to_data(self) -> dict[str, Any]:
        return {"iterate": self.iterate.canonical_identity(),
                "states": [row.canonical_identity() for row in self.states],
                "fields": [row.canonical_identity() for row in self.fields],
                "time_sources": [row.canonical_identity() for row in self.time_sources],
                "parameters": [row.canonical_identity() for row in self.parameters],
                "point": self.point.to_data()}


@dataclass(frozen=True, slots=True)
class FieldBoundaryContribution:
    region: Any
    provider: Any
    dependencies: FieldResidualDependencies
    residual: Handle
    jacobian: Handle
    jvp: Handle

    contribution_type: ClassVar[str] = "extension"

    def __post_init__(self) -> None:
        self._validate_region()
        if not isinstance(self.dependencies, FieldResidualDependencies):
            raise TypeError("boundary contribution dependencies must be exact")
        self._validate_provider()
        _handle(self.residual, where="FieldBoundaryContribution.residual",
                kinds=frozenset(("field_boundary_residual",)))
        _handle(self.jacobian, where="FieldBoundaryContribution.jacobian",
                kinds=frozenset(("field_boundary_jacobian",)))
        _handle(self.jvp, where="FieldBoundaryContribution.jvp",
                kinds=frozenset(("field_boundary_jvp",)))

    def _validate_region(self) -> None:
        canonical = getattr(self.region, "canonical_identity", None)
        if not callable(canonical):
            raise TypeError("field boundary contribution region must be canonical data")

    def _validate_provider(self) -> None:
        canonical = getattr(self.provider, "canonical_identity", None)
        if not callable(canonical):
            raise TypeError("field boundary contribution provider must be canonical data")

    def validate_topology(self, topology: Any) -> None:
        del topology
        raise TypeError("extension contribution must implement validate_topology()")

    @property
    def region_key(self) -> str:
        return _canonical_id(self.region.canonical_identity())

    def to_data(self) -> dict[str, Any]:
        return {"contribution_type": self.contribution_type,
                "region": self.region.canonical_identity(),
                "provider": self.provider.canonical_identity(),
                "dependencies": self.dependencies.to_data(),
                "residual": self.residual.canonical_identity(),
                "jacobian": self.jacobian.canonical_identity(),
                "jvp": self.jvp.canonical_identity()}


class _PhysicalContribution(FieldBoundaryContribution):
    def _validate_region(self) -> None:
        from pops.mesh.boundaries import BoundaryHandle

        if not isinstance(self.region, BoundaryHandle):
            raise TypeError("physical field contribution requires a BoundaryHandle")

    def _validate_provider(self) -> None:
        from pops.mesh.boundaries import BoundaryProvider

        if not isinstance(self.provider, BoundaryProvider):
            raise TypeError("physical field contribution requires a BoundaryProvider")
        provider_dependencies = self.provider.dependencies
        checks = (
            (provider_dependencies.states, self.dependencies.states, "states"),
            (provider_dependencies.fields, self.dependencies.fields, "fields"),
            (provider_dependencies.time, self.dependencies.time_sources, "time"),
            (provider_dependencies.runtime_params, self.dependencies.parameters, "parameters"),
        )
        for required, declared, name in checks:
            if not set(required).issubset(declared):
                raise ValueError(
                    "field boundary contribution omits provider %s dependencies" % name)

    def validate_topology(self, topology: Any) -> None:
        from pops.mesh.boundaries import ConstraintResidual

        if not topology.contains(self.region) or topology.is_periodic(self.region):
            raise ValueError("physical contribution must target a physical topology boundary")
        if not any(isinstance(output, ConstraintResidual) and
                   output.boundary == self.region and
                   output.subject == self.dependencies.iterate
                   for output in self.provider.outputs):
            raise ValueError(
                "physical BoundaryProvider does not cover the contribution region/iterate")


@dataclass(frozen=True, slots=True)
class DirichletContribution(_PhysicalContribution):
    contribution_type = "dirichlet"


@dataclass(frozen=True, slots=True)
class NeumannContribution(_PhysicalContribution):
    contribution_type = "neumann"


@dataclass(frozen=True, slots=True)
class MixedContribution(_PhysicalContribution):
    contribution_type = "mixed"


@dataclass(frozen=True, slots=True)
class PeriodicContribution(FieldBoundaryContribution):
    contribution_type = "periodic"

    def _validate_region(self) -> None:
        from pops.mesh.boundaries import PeriodicIdentification

        if not isinstance(self.region, PeriodicIdentification):
            raise TypeError("periodic field contribution requires a PeriodicIdentification")

    def _validate_provider(self) -> None:
        _handle(self.provider, where="PeriodicContribution.provider",
                kinds=frozenset(("periodic_field_provider",)))

    def validate_topology(self, topology: Any) -> None:
        if self.region not in topology.periodic:
            raise ValueError("periodic contribution is absent from BoundaryTopology")


@dataclass(frozen=True, slots=True)
class FieldDependencyCoverage:
    residual: FieldResidualDependencies
    jacobian: FieldResidualDependencies
    jvp: FieldResidualDependencies
    restart: FieldResidualDependencies

    def __post_init__(self) -> None:
        if any(not isinstance(row, FieldResidualDependencies) for row in (
                self.residual, self.jacobian, self.jvp, self.restart)):
            raise TypeError("FieldDependencyCoverage entries must be FieldResidualDependencies")

    def to_data(self) -> dict[str, Any]:
        return {name: getattr(self, name).to_data()
                for name in ("residual", "jacobian", "jvp", "restart")}


@dataclass(frozen=True, slots=True)
class FieldRestartContract:
    handle: Handle
    dependencies: FieldResidualDependencies
    payloads: tuple[Handle, ...]

    def __post_init__(self) -> None:
        _handle(self.handle, where="FieldRestartContract.handle",
                kinds=frozenset(("field_restart_contract",)))
        if not isinstance(self.dependencies, FieldResidualDependencies):
            raise TypeError("FieldRestartContract.dependencies must be exact")
        object.__setattr__(self, "payloads", _handles(
            self.payloads, where="FieldRestartContract.payloads",
            kinds=frozenset(("field_residual", "field_jacobian", "field_jvp"))))

    def to_data(self) -> dict[str, Any]:
        return {"handle": self.handle.canonical_identity(),
                "dependencies": self.dependencies.to_data(),
                "payloads": [row.canonical_identity() for row in self.payloads]}


def _case_root(handle: Handle) -> Any:
    from pops.model import OwnerKind

    return next((node for node in handle.owner_path.nodes if node.kind is OwnerKind.CASE), None)


@dataclass(frozen=True, slots=True)
class FieldResidualContract:
    handle: Handle
    operator: Handle
    unknown: Handle
    topology: Any
    dependencies: FieldResidualDependencies
    boundaries: tuple[FieldBoundaryContribution, ...]
    residual: Handle
    jacobian: Handle
    jvp: Handle
    coverage: FieldDependencyCoverage
    restart: FieldRestartContract

    def __post_init__(self) -> None:
        from pops.mesh.boundaries import BoundaryTopology

        _handle(self.handle, where="FieldResidualContract.handle",
                kinds=frozenset(("field_residual_contract",)))
        _handle(self.operator, where="FieldResidualContract.operator",
                kinds=frozenset(("field_operator",)))
        _handle(self.unknown, where="FieldResidualContract.unknown",
                kinds=frozenset(("field", "aux")))
        if not isinstance(self.topology, BoundaryTopology):
            raise TypeError("FieldResidualContract.topology must be a BoundaryTopology")
        if not isinstance(self.dependencies, FieldResidualDependencies):
            raise TypeError("FieldResidualContract.dependencies must be exact")
        if self.dependencies.iterate != self.unknown:
            raise ValueError("field residual iterate must be the exact unknown Handle")
        if not isinstance(self.boundaries, tuple) or any(
                not isinstance(row, FieldBoundaryContribution) for row in self.boundaries):
            raise TypeError("FieldResidualContract.boundaries contains invalid contributions")
        if len({row.region_key for row in self.boundaries}) != len(self.boundaries):
            raise ValueError("more than one field contribution covers the same boundary region")
        for row in self.boundaries:
            row.validate_topology(self.topology)
        for value, kind, where in (
            (self.residual, "field_residual", "residual"),
            (self.jacobian, "field_jacobian", "jacobian"),
            (self.jvp, "field_jvp", "jvp"),
        ):
            _handle(value, where="FieldResidualContract.%s" % where, kinds=frozenset((kind,)))
        expected = FieldResidualDependencies.merged(
            self.dependencies, *(row.dependencies for row in self.boundaries))
        if not isinstance(self.coverage, FieldDependencyCoverage):
            raise TypeError("FieldResidualContract.coverage must be explicit")
        for name in ("residual", "jacobian", "jvp", "restart"):
            if getattr(self.coverage, name) != expected:
                raise ValueError("%s dependency coverage omits boundary or field dependencies" % name)
        if not isinstance(self.restart, FieldRestartContract):
            raise TypeError("FieldResidualContract.restart must be explicit")
        if self.restart.dependencies != expected:
            raise ValueError("restart contract omits boundary or field dependencies")
        if set(self.restart.payloads) != {self.residual, self.jacobian, self.jvp}:
            raise ValueError("restart contract must cover residual, Jacobian and JVP payloads")
        handles = [self.handle, self.operator, self.unknown, self.residual, self.jacobian, self.jvp,
                   self.restart.handle]
        handles.extend(expected.states + expected.fields + expected.time_sources +
                       expected.parameters + (expected.iterate,))
        handles.extend(value for row in self.boundaries for value in (
            getattr(row.provider, "handle", row.provider), row.residual, row.jacobian, row.jvp))
        for row in self.boundaries:
            provider = row.provider
            if not hasattr(provider, "dependencies"):
                continue
            provider_dependencies = provider.dependencies
            handles.extend(provider_dependencies.states + provider_dependencies.fields +
                           provider_dependencies.time + provider_dependencies.runtime_params)
            handles.extend(output.subject for output in provider.outputs)
            handles.extend(output.representation for output in provider.outputs)
            flow = provider_dependencies.representation
            handles.extend((flow.source, flow.target))
            if flow.converter is not None:
                handles.append(flow.converter)
            handles.extend(provider_dependencies.characteristic.characteristics)
        foreign = [value.qualified_id for value in handles
                   if _case_root(value) is not None and
                   _case_root(value) != self.topology.owner.nodes[0]]
        if foreign:
            raise ValueError("field residual contract contains foreign Case handles: %s" % foreign)
        from pops.model import OwnerKind

        point_clock = expected.point.clock
        if point_clock is None:
            raise ValueError("field residual TimePoint must carry its owning clock")
        point_owner = point_clock.owner
        if point_owner is None:
            raise ValueError("field residual TimePoint clock must carry its owner")
        point_case = next((node for node in point_owner.nodes
                           if node.kind is OwnerKind.CASE), None)
        if point_case is not None and point_case != self.topology.owner.nodes[0]:
            raise ValueError("field residual TimePoint belongs to a foreign Case")
        object.__setattr__(self, "boundaries", tuple(sorted(
            self.boundaries, key=lambda row: row.region_key)))

    @property
    def identity(self) -> str:
        return _canonical_id(self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "contract_type": "field_residual",
                "handle": self.handle.canonical_identity(),
                "operator": self.operator.canonical_identity(),
                "unknown": self.unknown.canonical_identity(),
                "topology": self.topology.canonical_identity(),
                "dependencies": self.dependencies.to_data(),
                "boundaries": [row.to_data() for row in self.boundaries],
                "residual": self.residual.canonical_identity(),
                "jacobian": self.jacobian.canonical_identity(),
                "jvp": self.jvp.canonical_identity(),
                "coverage": self.coverage.to_data(), "restart": self.restart.to_data()}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "field_residual_contract", "identity": self.identity,
                **self.to_data()}


__all__ = [
    "DirichletContribution", "FieldBoundaryContribution", "FieldDependencyCoverage",
    "FieldResidualContract", "FieldResidualDependencies", "FieldRestartContract",
    "MixedContribution", "NeumannContribution", "PeriodicContribution",
]
