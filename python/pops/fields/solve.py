"""Capability-resolved field solve plans and publication-safe outcomes."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
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


def _hierarchy_policy_options(value: Any) -> Mapping[str, bool | int | float | str]:
    """Freeze the provider-owned scalar option envelope used by the native authority."""
    if not isinstance(value, Mapping):
        raise TypeError("resolved hierarchy policy options must be a mapping")
    result: dict[str, bool | int | float | str] = {}
    for key, item in value.items():
        if type(key) is not str or not key:
            raise TypeError("resolved hierarchy policy option names must be exact strings")
        if type(item) is bool or type(item) is str:
            result[key] = item
        elif type(item) is int:
            if item < -(1 << 63) or item > (1 << 63) - 1:
                raise ValueError("resolved hierarchy policy integer option exceeds int64")
            result[key] = item
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("resolved hierarchy policy options require finite binary64")
            result[key] = item
        else:
            raise TypeError(
                "resolved hierarchy policy options support exact bool/int/float/string values"
            )
    return MappingProxyType(result)


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
    """Opaque, versioned hierarchy-policy authority selected before provider validation."""

    policy_id: str
    interface_version: int
    option_schema: str
    options: Mapping[str, bool | int | float | str]
    capability: Handle | None = None

    def __post_init__(self) -> None:
        for name in ("policy_id", "option_schema"):
            value = getattr(self, name)
            if type(value) is not str or not value:
                raise TypeError("ResolvedHierarchyPolicy.%s must be an exact identity" % name)
        if type(self.interface_version) is not int or self.interface_version < 1:
            raise TypeError("ResolvedHierarchyPolicy.interface_version must be positive")
        object.__setattr__(self, "options", _hierarchy_policy_options(self.options))
        if self.capability is not None:
            _handle(
                self.capability,
                where="ResolvedHierarchyPolicy.capability",
                kinds=frozenset(("field_solve_capability",)),
            )

    def authority(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "interface_version": self.interface_version,
            "option_schema": self.option_schema,
            "options": dict(self.options),
        }

    def with_capability(self, capability: Handle) -> ResolvedHierarchyPolicy:
        if self.capability is not None and self.capability != capability:
            raise ValueError("resolved hierarchy policy uses a foreign capability proof")
        return ResolvedHierarchyPolicy(
            self.policy_id,
            self.interface_version,
            self.option_schema,
            self.options,
            capability,
        )

    def to_data(self) -> dict[str, Any]:
        return {
            "authority": self.authority(),
            "capability": (
                None if self.capability is None else self.capability.canonical_identity()
            ),
        }


@dataclass(frozen=True, slots=True)
class FieldSolveCapabilities:
    handle: Handle
    inferred_hierarchy: ResolvedHierarchyPolicy
    native_contracts: tuple[str, ...]
    boundary_contributions: tuple[str, ...]

    _NATIVE = frozenset(("residual", "jacobian", "jvp", "restart"))
    _BOUNDARY = frozenset(("dirichlet", "neumann", "mixed", "periodic"))

    def __post_init__(self) -> None:
        _handle(self.handle, where="FieldSolveCapabilities.handle",
                kinds=frozenset(("field_solve_capability",)))
        if not isinstance(self.inferred_hierarchy, ResolvedHierarchyPolicy):
            raise TypeError(
                "FieldSolveCapabilities.inferred_hierarchy must be a resolved authority"
            )
        object.__setattr__(
            self,
            "inferred_hierarchy",
            self.inferred_hierarchy.with_capability(self.handle),
        )
        object.__setattr__(self, "native_contracts", _tags(
            self.native_contracts, where="FieldSolveCapabilities.native_contracts",
            allowed=self._NATIVE))
        object.__setattr__(self, "boundary_contributions", _tags(
            self.boundary_contributions,
            where="FieldSolveCapabilities.boundary_contributions", allowed=self._BOUNDARY))

    def inferred_hierarchy_policy(self) -> ResolvedHierarchyPolicy:
        return self.inferred_hierarchy

    def bind_hierarchy_policy(
        self, policy: ResolvedHierarchyPolicy
    ) -> ResolvedHierarchyPolicy:
        if not isinstance(policy, ResolvedHierarchyPolicy):
            raise TypeError("field hierarchy policy must return ResolvedHierarchyPolicy")
        return policy.with_capability(self.handle)

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
                "inferred_hierarchy": self.inferred_hierarchy.to_data(),
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
        hierarchy = self.capabilities.bind_hierarchy_policy(hierarchy)
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
