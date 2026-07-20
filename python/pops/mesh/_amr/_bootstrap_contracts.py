"""Private deterministic level-zero initialization and recursive AMR hierarchy bootstrap."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pops.identity import Identity, make_identity
from ._contracts import canonical_handle
from .hierarchy import CanonicalOptions
from .tagging_resolution import ResolvedTaggingGraph


def _handle(value: Any, *, where: str, kind: str | None = None) -> Any:
    projection = getattr(value, "canonical_identity", None)
    data = projection() if callable(projection) else None
    actual = data.get("kind") if isinstance(data, Mapping) else None
    if not isinstance(actual, str):
        raise TypeError("%s requires an owner-qualified Handle protocol" % where)
    return canonical_handle(value, where=where, kinds=kind or actual)


def _freeze(value: Any, *, where: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError("%s mapping keys must be non-empty strings" % where)
        return MappingProxyType(
            {key: _freeze(value[key], where="%s.%s" % (where, key)) for key in sorted(value)}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item, where="%s[]" % where) for item in value)
    raise TypeError("%s contains non-canonical data %s" % (where, type(value).__name__))


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AnalyticReprojection:
    """Evaluate the authenticated initial-condition source on every created level."""

    def to_data(self) -> dict[str, Any]:
        return {"method": "analytic_reprojection"}


@dataclass(frozen=True, slots=True)
class ProlongFromParent:
    """Use the resolved prolongation provider from the immediate parent level."""

    def to_data(self) -> dict[str, Any]:
        return {"method": "prolongation"}


@dataclass(frozen=True, slots=True)
class BootstrapSelection:
    subject: Any
    method: AnalyticReprojection | ProlongFromParent
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _handle(self.subject, where="BootstrapSelection.subject")
        if type(self.method) not in (AnalyticReprojection, ProlongFromParent):
            raise TypeError("bootstrap method must be AnalyticReprojection or ProlongFromParent")

    def to_data(self) -> dict[str, Any]:
        return {"subject": self.subject.canonical_identity(), **self.method.to_data()}


@dataclass(frozen=True, slots=True)
class ConstraintProvider:
    subject: Any
    provider: Any
    options: CanonicalOptions = CanonicalOptions()
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        _handle(self.subject, where="ConstraintProvider.subject")
        _handle(
            self.provider,
            where="ConstraintProvider.provider",
            kind="amr_constraint_provider",
        )
        if type(self.options) is not CanonicalOptions:
            raise TypeError("ConstraintProvider.options must be CanonicalOptions")

    @property
    def qualified_id(self) -> str:
        return self.provider.qualified_id

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "subject": self.subject.canonical_identity(),
            "provider": self.provider.canonical_identity(),
            "qualified_id": self.qualified_id,
            "options": self.options.to_data(),
        }


@dataclass(frozen=True, slots=True)
class BootstrapOrdering:
    phases: tuple[str, ...]
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        phases = tuple(self.phases)
        if len(phases) != 3 or set(phases) != {"transfer", "projection", "constraint"}:
            raise ValueError(
                "BootstrapOrdering must explicitly order transfer, projection, constraint"
            )
        object.__setattr__(self, "phases", phases)

    def to_data(self) -> dict[str, Any]:
        return {"phases": list(self.phases)}


@dataclass(frozen=True, slots=True)
class BootstrapAction:
    level: int
    phase: str
    operation: str
    subject_id: str | None
    evidence: Any
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if isinstance(self.level, bool) or not isinstance(self.level, int) or self.level < 0:
            raise ValueError("BootstrapAction.level must be an integer >= 0")
        for name in ("phase", "operation"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise TypeError("BootstrapAction.%s must be non-empty" % name)
        if self.subject_id is not None and (
            not isinstance(self.subject_id, str) or not self.subject_id
        ):
            raise TypeError("BootstrapAction.subject_id must be non-empty or None")
        object.__setattr__(self, "evidence", _freeze(self.evidence, where="BootstrapAction.evidence"))

    def to_data(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "phase": self.phase,
            "operation": self.operation,
            "subject_id": self.subject_id,
            "evidence": _thaw(self.evidence),
        }

    @property
    def identity(self) -> Identity:
        return make_identity("amr-bootstrap-action", self.to_data())


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    layout_plan_id: str
    hierarchy_identity: Identity
    transfer_identity: Identity
    initial_identity: Identity
    tagging: ResolvedTaggingGraph
    ordering: BootstrapOrdering
    selections: tuple[BootstrapSelection, ...]
    constraints: tuple[Any, ...]
    actions: tuple[BootstrapAction, ...]
    __pops_ir_immutable__ = True

    def __post_init__(self) -> None:
        if not isinstance(self.layout_plan_id, str) or not self.layout_plan_id:
            raise TypeError("BootstrapPlan.layout_plan_id must be non-empty")
        for name, domain in (
            ("hierarchy_identity", "resolved-amr-hierarchy"),
            ("transfer_identity", "amr-transfer"),
            ("initial_identity", "initial-condition-plan"),
        ):
            identity = getattr(self, name)
            if type(identity) is not Identity or identity.domain != domain:
                raise TypeError("BootstrapPlan.%s must be an exact %s Identity" % (name, domain))
        if type(self.ordering) is not BootstrapOrdering:
            raise TypeError("BootstrapPlan.ordering must be BootstrapOrdering")
        if type(self.tagging) is not ResolvedTaggingGraph:
            raise TypeError("BootstrapPlan.tagging must be a ResolvedTaggingGraph")
        selections = tuple(self.selections)
        if not selections or any(type(row) is not BootstrapSelection for row in selections):
            raise TypeError("BootstrapPlan.selections must contain selections")
        subject_ids = [row.subject.qualified_id for row in selections]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("BootstrapPlan contains duplicate selections")
        constraints = tuple(self.constraints)
        if any(type(constraint) is not ConstraintProvider for constraint in constraints):
            raise TypeError("BootstrapPlan.constraints must contain ConstraintProvider values")
        actions = tuple(self.actions)
        if not actions or any(type(row) is not BootstrapAction for row in actions):
            raise TypeError("BootstrapPlan.actions must contain actions")
        if not any(row.level == 0 and row.operation == "initialize_level_zero" for row in actions):
            raise ValueError("BootstrapPlan must initialize level zero")
        object.__setattr__(self, "selections", selections)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "actions", actions)

    @property
    def identity(self) -> Identity:
        return make_identity("amr-bootstrap-plan", self.canonical_identity())

    def canonical_identity(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "layout_plan_id": self.layout_plan_id,
            "hierarchy_identity": self.hierarchy_identity.to_data(),
            "transfer_identity": self.transfer_identity.to_data(),
            "initial_identity": self.initial_identity.to_data(),
            "tagging": self.tagging.canonical_identity(),
            "ordering": self.ordering.to_data(),
            "selections": [selection.to_data() for selection in self.selections],
            "constraints": [constraint.canonical_identity() for constraint in self.constraints],
            "actions": [action.to_data() for action in self.actions],
        }

    def inspect(self) -> dict[str, Any]:
        return {
            "report_type": "amr_bootstrap_plan",
            "identity": self.identity.token,
            **self.canonical_identity(),
        }


def _action(level: int, phase: str, operation: str, subject: Any = None, evidence: Any = None) \
        -> BootstrapAction:
    return BootstrapAction(
        level,
        phase,
        operation,
        None if subject is None else subject.qualified_id,
        evidence,
    )
