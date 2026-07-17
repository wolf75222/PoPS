"""Capability-resolved field solve plans and publication-safe outcomes."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from ._identity import field_identity
from .context import Accepted, FieldContext, LayoutBinding
from .nullspace import NullspaceCompatibility
from .residual import FieldResidualContract

if TYPE_CHECKING:
    from pops.model import Handle


def _handle(value: Any, *, where: str, kinds: frozenset[str]) -> Handle:
    from pops.model import Handle

    if isinstance(value, str) or not isinstance(value, Handle) or not value.is_resolved:
        raise TypeError("%s requires a canonical owner-qualified Handle" % where)
    if value.kind not in kinds:
        raise TypeError("%s requires Handle.kind in %s, got %r" %
                        (where, sorted(kinds), value.kind))
    return value


def _tags(values: Any, *, where: str, allowed: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or any(
            not isinstance(row, str) or row not in allowed for row in values):
        raise TypeError("%s must be a tuple using %s" % (where, sorted(allowed)))
    if len(values) != len(set(values)):
        raise ValueError("%s contains duplicate capabilities" % where)
    return tuple(sorted(values))


class FieldArtifactUnavailable(RuntimeError):
    """Structured refusal raised before codegen/native artifact creation."""

    def __init__(self, *, code: str, missing: tuple[str, ...], capability: Handle) -> None:
        self.report = {
            "phase": "field_resolution", "severity": "error", "code": code,
            "missing": list(missing), "capability": capability.canonical_identity(),
            "artifact_created": False,
        }
        super().__init__("%s: %s" % (code, ", ".join(missing)))


@dataclass(frozen=True, slots=True)
class ResolvedHierarchyPolicy:
    mode: str
    capability: Handle

    def __post_init__(self) -> None:
        if self.mode not in ("composite", "level_local"):
            raise ValueError("resolved hierarchy mode must be composite or level_local")
        _handle(self.capability, where="ResolvedHierarchyPolicy.capability",
                kinds=frozenset(("field_solve_capability",)))

    def to_data(self) -> dict[str, Any]:
        return {"mode": self.mode, "capability": self.capability.canonical_identity()}


@dataclass(frozen=True, slots=True)
class FieldSolveCapabilities:
    handle: Handle
    hierarchy_modes: tuple[str, ...]
    layout_mode: str
    native_contracts: tuple[str, ...]
    boundary_contributions: tuple[str, ...]

    _HIERARCHY = frozenset(("composite", "level_local"))
    _NATIVE = frozenset(("residual", "jacobian", "jvp", "restart"))
    _BOUNDARY = frozenset(("dirichlet", "neumann", "mixed", "periodic"))

    def __post_init__(self) -> None:
        _handle(self.handle, where="FieldSolveCapabilities.handle",
                kinds=frozenset(("field_solve_capability",)))
        object.__setattr__(self, "hierarchy_modes", _tags(
            self.hierarchy_modes, where="FieldSolveCapabilities.hierarchy_modes",
            allowed=self._HIERARCHY))
        if self.layout_mode not in self._HIERARCHY:
            raise ValueError("FieldSolveCapabilities.layout_mode is unsupported")
        object.__setattr__(self, "native_contracts", _tags(
            self.native_contracts, where="FieldSolveCapabilities.native_contracts",
            allowed=self._NATIVE))
        object.__setattr__(self, "boundary_contributions", _tags(
            self.boundary_contributions,
            where="FieldSolveCapabilities.boundary_contributions", allowed=self._BOUNDARY))

    def resolve_hierarchy(self, requested: str) -> ResolvedHierarchyPolicy:
        mode = self.layout_mode if requested == "infer_from_layout" else requested
        if mode not in self._HIERARCHY:
            raise ValueError("unknown hierarchy policy %r" % requested)
        if mode not in self.hierarchy_modes:
            raise FieldArtifactUnavailable(
                code="field.hierarchy.unsupported", missing=(mode,), capability=self.handle)
        return ResolvedHierarchyPolicy(mode, self.handle)

    def require(self, residual: FieldResidualContract) -> None:
        missing_native = tuple(sorted(self._NATIVE - set(self.native_contracts)))
        if missing_native:
            raise FieldArtifactUnavailable(
                code="field.native_contract.unsupported", missing=missing_native,
                capability=self.handle)
        required_boundaries = {row.contribution_type for row in residual.boundaries}
        missing_boundaries = tuple(sorted(
            required_boundaries - set(self.boundary_contributions)))
        if missing_boundaries:
            raise FieldArtifactUnavailable(
                code="field.boundary_contract.unsupported", missing=missing_boundaries,
                capability=self.handle)

    def to_data(self) -> dict[str, Any]:
        return {"handle": self.handle.canonical_identity(),
                "hierarchy_modes": list(self.hierarchy_modes),
                "layout_mode": self.layout_mode,
                "native_contracts": list(self.native_contracts),
                "boundary_contributions": list(self.boundary_contributions)}


@dataclass(frozen=True, slots=True)
class FieldOperatorDomain:
    residual_contract_identity: str
    unknown: Handle
    layout: LayoutBinding

    def __post_init__(self) -> None:
        if not isinstance(self.residual_contract_identity, str) or len(
                self.residual_contract_identity) != 64 or any(
                    char not in "0123456789abcdef" for char in self.residual_contract_identity):
            raise ValueError("FieldOperatorDomain requires a residual contract sha256 identity")
        _handle(self.unknown, where="FieldOperatorDomain.unknown",
                kinds=frozenset(("field", "aux")))
        if not isinstance(self.layout, LayoutBinding):
            raise TypeError("FieldOperatorDomain.layout must be a LayoutBinding")

    def to_data(self) -> dict[str, Any]:
        return {"residual_contract_identity": self.residual_contract_identity,
                "unknown": self.unknown.canonical_identity(), "layout": self.layout.to_data()}


@dataclass(frozen=True, slots=True)
class PreconditionerBinding:
    preconditioner: Handle
    domain: FieldOperatorDomain

    def __post_init__(self) -> None:
        _handle(self.preconditioner, where="PreconditionerBinding.preconditioner",
                kinds=frozenset(("preconditioner",)))
        if not isinstance(self.domain, FieldOperatorDomain):
            raise TypeError("PreconditionerBinding.domain must be a FieldOperatorDomain")

    def to_data(self) -> dict[str, Any]:
        return {"preconditioner": self.preconditioner.canonical_identity(),
                "domain": self.domain.to_data()}


@dataclass(frozen=True, slots=True)
class FieldSolvePlan:
    residual: FieldResidualContract
    hierarchy: ResolvedHierarchyPolicy
    capabilities: FieldSolveCapabilities
    domain: FieldOperatorDomain
    preconditioner: PreconditionerBinding | None = None
    nullspace: NullspaceCompatibility | None = None
    gauge: Handle | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.residual, FieldResidualContract):
            raise TypeError("FieldSolvePlan.residual must be a FieldResidualContract")
        if not isinstance(self.hierarchy, ResolvedHierarchyPolicy):
            raise TypeError("FieldSolvePlan.hierarchy must already be capability-resolved")
        if not isinstance(self.capabilities, FieldSolveCapabilities):
            raise TypeError("FieldSolvePlan.capabilities must be typed")
        if self.hierarchy.capability != self.capabilities.handle:
            raise ValueError("resolved hierarchy policy uses a foreign capability proof")
        self.capabilities.require(self.residual)
        if not isinstance(self.domain, FieldOperatorDomain):
            raise TypeError("FieldSolvePlan.domain must be a FieldOperatorDomain")
        if self.domain.residual_contract_identity != self.residual.identity or \
                self.domain.unknown != self.residual.unknown:
            raise ValueError("FieldSolvePlan domain does not authenticate its residual")
        if self.preconditioner is not None:
            if not isinstance(self.preconditioner, PreconditionerBinding):
                raise TypeError("FieldSolvePlan.preconditioner must be a PreconditionerBinding")
            if self.preconditioner.domain != self.domain:
                raise ValueError("preconditioner domain does not authenticate the residual domain")
        if self.nullspace is not None and not isinstance(
                self.nullspace, NullspaceCompatibility):
            raise TypeError("FieldSolvePlan.nullspace must be a NullspaceCompatibility proof")
        if self.nullspace is not None and self.gauge is None:
            raise ValueError("nullspace and gauge are separate; an explicit gauge is required")
        if self.gauge is not None:
            _handle(self.gauge, where="FieldSolvePlan.gauge",
                    kinds=frozenset(("field_gauge",)))

    @property
    def identity(self) -> Any:
        return field_identity("field-solve-plan", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {"schema_version": 1, "residual": self.residual.to_data(),
                "hierarchy": self.hierarchy.to_data(),
                "capabilities": self.capabilities.to_data(), "domain": self.domain.to_data(),
                "preconditioner": (None if self.preconditioner is None else
                                   self.preconditioner.to_data()),
                "nullspace": None if self.nullspace is None else self.nullspace.to_data(),
                "gauge": None if self.gauge is None else self.gauge.canonical_identity()}

    def inspect(self) -> dict[str, Any]:
        return {"report_type": "field_solve_plan", "identity": self.identity.token,
                **self.to_data()}


@dataclass(frozen=True, slots=True)
class FieldSolveResolver:
    """Local resolver; capabilities are checked before a plan can reach codegen."""

    capabilities: FieldSolveCapabilities

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, FieldSolveCapabilities):
            raise TypeError("FieldSolveResolver requires FieldSolveCapabilities")

    def resolve(
        self, residual: FieldResidualContract, hierarchy_policy: Any, layout: LayoutBinding, *,
        preconditioner: PreconditionerBinding | None = None,
        nullspace: NullspaceCompatibility | None = None, gauge: Handle | None = None,
    ) -> FieldSolvePlan:
        self.capabilities.require(residual)
        resolver = getattr(hierarchy_policy, "resolve", None)
        if not callable(resolver):
            raise TypeError("field solve requires a typed FieldHierarchyPolicy")
        hierarchy = resolver(self.capabilities)
        if not isinstance(hierarchy, ResolvedHierarchyPolicy):
            raise TypeError(
                "FieldHierarchyPolicy.resolve() must return ResolvedHierarchyPolicy"
            )
        domain = FieldOperatorDomain(residual.identity, residual.unknown, layout)
        if preconditioner is not None and preconditioner.domain != domain:
            raise ValueError("preconditioner domain does not authenticate the residual domain")
        return FieldSolvePlan(
            residual, hierarchy, self.capabilities, domain, preconditioner, nullspace, gauge)


class SolveStatus(Enum):
    CONVERGED = "converged"
    NON_CONVERGED = "non_converged"
    INCOMPATIBLE_RHS = "incompatible_rhs"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class SolveOutcome:
    plan: FieldSolvePlan
    status: SolveStatus
    iterations: int
    witness: Handle
    reason: str
    context: FieldContext | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.plan, FieldSolvePlan):
            raise TypeError("SolveOutcome.plan must be a FieldSolvePlan")
        if not isinstance(self.status, SolveStatus):
            raise TypeError("SolveOutcome.status must be a SolveStatus")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int) or \
                self.iterations < 0:
            raise ValueError("SolveOutcome.iterations must be an integer >= 0")
        _handle(self.witness, where="SolveOutcome.witness",
                kinds=frozenset(("field_solve_witness",)))
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("SolveOutcome.reason must be a non-empty string")
        if self.context is not None and not isinstance(self.context, FieldContext):
            raise TypeError("SolveOutcome.context must be a FieldContext or None")
        if self.context is not None and (
                self.context.operator != self.plan.residual.operator or
                self.context.point != self.plan.residual.dependencies.point or
                self.context.layout != self.plan.domain.layout):
            raise ValueError("FieldContext does not authenticate the solve plan")
        if self.status is SolveStatus.CONVERGED:
            if self.context is None or not isinstance(self.context.materialization, Accepted):
                raise ValueError("converged SolveOutcome requires an Accepted FieldContext")
            if not self.context.validity.contains(self.context.point, self.context.layout):
                raise ValueError("converged SolveOutcome requires a currently valid FieldContext")
            expected = set(self.plan.residual.coverage.residual.states +
                           self.plan.residual.coverage.residual.fields +
                           self.plan.residual.coverage.residual.time_sources +
                           self.plan.residual.coverage.residual.parameters)
            if {row.reference for row in self.context.inputs} != expected:
                raise ValueError("Accepted FieldContext omits field or boundary dependencies")
        elif self.context is not None and isinstance(self.context.materialization, Accepted):
            raise ValueError(
                "non-converged/incompatible-RHS solve cannot publish Accepted FieldContext")

    def publish(self) -> FieldContext:
        if self.status is not SolveStatus.CONVERGED or self.context is None:
            raise RuntimeError("only a converged solve may publish a FieldContext")
        return self.context

    def to_data(self) -> dict[str, Any]:
        return {"plan_identity": self.plan.identity.token, "status": self.status.value,
                "iterations": self.iterations, "witness": self.witness.canonical_identity(),
                "reason": self.reason,
                "context": None if self.context is None else self.context.to_data()}


__all__ = [
    "FieldArtifactUnavailable", "FieldOperatorDomain", "FieldSolveCapabilities",
    "FieldSolvePlan", "FieldSolveResolver", "PreconditionerBinding",
    "ResolvedHierarchyPolicy", "SolveOutcome", "SolveStatus",
]
