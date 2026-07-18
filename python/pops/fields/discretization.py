"""Numerical field plans, strictly separated from physical FieldOperator declarations."""

from __future__ import annotations

from copy import copy
from typing import Any, Protocol, runtime_checkable

from pops.descriptors import BrickDescriptor, Descriptor, reject_string_selector
from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.identity import Identity

from ._identity import field_identity, strict_field_data
from .bcs import BoundaryCondition
from .gauges import FieldGauge
from ._references import collect_references, resolve_value
from .solve import ResolvedHierarchyPolicy


FIELD_DISCRETIZATION_SCHEMA_VERSION = 2
_FIELD_DISCRETIZATION_DATA_KEYS = frozenset({
    "schema_version",
    "provider_id",
    "method",
    "boundaries",
    "solver",
    "nonlinear",
    "preconditioner",
    "nullspace",
    "gauge",
    "hierarchy_policy",
})


@runtime_checkable
class FieldDiscretizationProtocol(Protocol):
    """Small structural interface consumed by field registration and lowering.

    A third-party plan owns its Python type and provider identity.  It does not register that
    type in PoPS: implementing this exact surface is sufficient.  The numerical method, solver
    and policies remain independently extensible descriptors.
    """

    provider_id: str
    method: Any
    boundaries: tuple[Any, ...]
    solver: Any
    nonlinear: Any
    preconditioner: Any
    nullspace: Any
    gauge: Any
    hierarchy_policy: Any

    def to_data(self) -> dict[str, Any]: ...
    def available(self, context: Any = None) -> Any: ...
    def validate(self, context: Any = None) -> Any: ...
    def declaration_references(self) -> tuple[Any, ...]: ...
    def resolve_references(self, resolver: Any) -> Any: ...
    def inspect(self) -> dict[str, Any]: ...
    def freeze(self) -> Any: ...


def require_field_discretization(value: Any, *, where: str) -> FieldDiscretizationProtocol:
    """Authenticate one field plan through the structural protocol, without class dispatch."""
    if not isinstance(value, FieldDiscretizationProtocol):
        raise TypeError("%s must implement FieldDiscretizationProtocol" % where)
    if not isinstance(value.provider_id, str) or not value.provider_id.strip():
        raise TypeError("%s provider_id must be a non-empty string" % where)
    return value


def field_discretization_data(value: Any, *, where: str) -> dict[str, Any]:
    """Return exact versioned identity data for a protocol-conforming field plan."""
    plan = require_field_discretization(value, where=where)
    data = plan.to_data()
    if not isinstance(data, dict) or set(data) != _FIELD_DISCRETIZATION_DATA_KEYS:
        raise TypeError(
            "%s to_data() must return exactly the FieldDiscretization v%d keys"
            % (where, FIELD_DISCRETIZATION_SCHEMA_VERSION)
        )
    if data["schema_version"] != FIELD_DISCRETIZATION_SCHEMA_VERSION:
        raise ValueError("%s uses an unsupported field discretization schema" % where)
    if data["provider_id"] != plan.provider_id:
        raise ValueError("%s provider_id disagrees with its canonical data" % where)
    return data


class FieldHierarchyPolicy(Descriptor):
    """Small extension interface producing an opaque resolved hierarchy authority."""

    category = "field_hierarchy_policy"

    def to_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "options": self.options()}

    def resolved_authority(self) -> ResolvedHierarchyPolicy:
        raise NotImplementedError(
            "%s must provide a resolved hierarchy authority" % type(self).__name__
        )

    def resolve(self, capabilities: Any) -> ResolvedHierarchyPolicy:
        binder = getattr(capabilities, "bind_hierarchy_policy", None)
        if not callable(binder):
            raise TypeError("field hierarchy policy requires typed solve capabilities")
        resolved = binder(self.resolved_authority())
        if not isinstance(resolved, ResolvedHierarchyPolicy):
            raise TypeError("field hierarchy policy binder did not return a resolved authority")
        return resolved


class InferHierarchyFromLayout(FieldHierarchyPolicy):
    """Resolve uniform/level-local/composite behavior from method and layout capabilities."""

    def options(self) -> dict[str, Any]:
        return {
            "policy_id": "pops.field-hierarchy.infer-from-layout",
            "interface_version": 1,
        }

    def resolve(self, capabilities: Any) -> ResolvedHierarchyPolicy:
        resolver = getattr(capabilities, "inferred_hierarchy_policy", None)
        if not callable(resolver):
            raise TypeError("inferred field hierarchy requires a typed layout authority")
        resolved = resolver()
        if not isinstance(resolved, ResolvedHierarchyPolicy):
            raise TypeError("inferred hierarchy authority is not resolved")
        return resolved


class LevelByLevelSolve(FieldHierarchyPolicy):
    def options(self) -> dict[str, Any]:
        return self.resolved_authority().authority()

    def resolved_authority(self) -> ResolvedHierarchyPolicy:
        return ResolvedHierarchyPolicy(
            "pops.field-hierarchy.level-local",
            1,
            "pops.field-hierarchy.options.empty@1",
            {},
        )


class CompositeHierarchySolve(FieldHierarchyPolicy):
    def options(self) -> dict[str, Any]:
        return self.resolved_authority().authority()

    def resolved_authority(self) -> ResolvedHierarchyPolicy:
        return ResolvedHierarchyPolicy(
            "pops.field-hierarchy.composite",
            1,
            "pops.field-hierarchy.options.empty@1",
            {},
        )


def _typed_descriptor(value: Any, *, field: str, required: bool = True) -> Any:
    if isinstance(value, str):
        reject_string_selector(value, field, "a typed descriptor")
    if value is None and not required:
        return None
    if not isinstance(value, (Descriptor, BrickDescriptor)):
        raise TypeError("FieldDiscretization %s must be a typed Descriptor" % field)
    return value


def _validate_route(value: Any, context: Any) -> None:
    result = value.validate(context)
    raise_if_error = getattr(result, "raise_if_error", None)
    if callable(raise_if_error):
        raise_if_error()


class FieldDiscretization(Descriptor):
    """All numerical choices for lowering one FieldOperator.

    No redundant ``order``, ghost depth, or AMR boolean is accepted: those consequences are
    derived from the selected method and hierarchy policy during resolution/lowering.
    """

    category = "field_discretization"
    provider_id = "pops.fields.discretization.v2"

    def __init__(
        self,
        *,
        method: Any,
        boundaries: Any,
        solver: Any,
        nonlinear: Any = None,
        preconditioner: Any = None,
        nullspace: Any = None,
        gauge: Any = None,
        hierarchy_policy: Any = None,
    ) -> None:
        self.method = _typed_descriptor(method, field="method")
        boundary_tuple = tuple(boundaries)
        if any(not isinstance(item, BoundaryCondition) for item in boundary_tuple):
            raise TypeError(
                "FieldDiscretization boundaries must contain BoundaryCondition descriptors"
            )
        self.boundaries = boundary_tuple
        self.solver = _typed_descriptor(solver, field="solver")
        self.nonlinear = _typed_descriptor(
            nonlinear, field="nonlinear", required=False)
        self.preconditioner = _typed_descriptor(
            preconditioner, field="preconditioner", required=False
        )
        self.nullspace = _typed_descriptor(nullspace, field="nullspace", required=False)
        if gauge is not None and not isinstance(gauge, FieldGauge):
            raise TypeError("FieldDiscretization gauge must be a FieldGauge")
        self.gauge = gauge
        if hierarchy_policy is None:
            hierarchy_policy = InferHierarchyFromLayout()
        if not isinstance(hierarchy_policy, FieldHierarchyPolicy):
            raise TypeError("FieldDiscretization hierarchy_policy must be a FieldHierarchyPolicy")
        self.hierarchy_policy = hierarchy_policy

    @property
    def identity(self) -> Identity:
        for reference in self.declaration_references():
            reference.canonical_identity()
        return field_identity("field-discretization", self.to_data())

    def to_data(self) -> dict[str, Any]:
        return {
            "schema_version": FIELD_DISCRETIZATION_SCHEMA_VERSION,
            "provider_id": self.provider_id,
            "method": strict_field_data(self.method),
            "boundaries": strict_field_data(self.boundaries),
            "solver": strict_field_data(self.solver),
            "nonlinear": strict_field_data(self.nonlinear),
            "preconditioner": strict_field_data(self.preconditioner),
            "nullspace": strict_field_data(self.nullspace),
            "gauge": strict_field_data(self.gauge),
            "hierarchy_policy": strict_field_data(self.hierarchy_policy),
        }

    def semantic_data(self) -> dict[str, Any]:
        """Exact numerical semantics, including nested descriptor configuration."""
        return self.to_data()

    def artifact_data(self) -> dict[str, Any]:
        return self.to_data()

    def options(self) -> dict[str, Any]:
        def label(value: Any) -> str | None:
            return None if value is None else getattr(value, "name", type(value).__name__)

        return {
            "method": label(self.method),
            "boundaries": [boundary.options() for boundary in self.boundaries],
            "solver": label(self.solver),
            "nonlinear": label(self.nonlinear),
            "preconditioner": label(self.preconditioner),
            "nullspace": label(self.nullspace),
            "gauge": label(self.gauge),
            "hierarchy_policy": self.hierarchy_policy.options(),
        }

    def requirements(self) -> RequirementSet:
        return RequirementSet(
            {
                "layout": True,
                "boundary_topology": bool(self.boundaries),
                "nullspace_declared": self.nullspace is not None,
                "gauge_declared": self.gauge is not None,
            }
        )

    def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            {
                "derives_order": True,
                "derives_ghost_depth": True,
                "hierarchy_policy": self.hierarchy_policy.options(),
            }
        )

    def validate(self, context: Any = None) -> bool:
        _validate_route(self.method, context)
        _validate_route(self.solver, context)
        if self.nonlinear is not None:
            _validate_route(self.nonlinear, context)
        if self.preconditioner is not None:
            _validate_route(self.preconditioner, context)
        if self.nullspace is not None and self.gauge is None:
            raise ValueError("FieldDiscretization with a nullspace requires an explicit gauge")
        seen: set[str] = set()
        for boundary in self.boundaries:
            key = field_identity("field-boundary-selector", boundary.selector.options()).token
            if key in seen:
                raise ValueError(
                    "FieldDiscretization has more than one condition for the same boundary"
                )
            seen.add(key)
            _validate_route(boundary, context)
        if all(reference.is_resolved for reference in self.declaration_references()):
            _ = self.identity
        return True

    def declaration_references(self) -> tuple[Any, ...]:
        return collect_references(
            (
                self.method,
                self.boundaries,
                self.solver,
                self.nonlinear,
                self.preconditioner,
                self.nullspace,
                self.gauge,
                self.hierarchy_policy,
            )
        )

    def resolve_references(self, resolver: Any) -> FieldDiscretization:
        resolved = copy(self)
        resolved.method = resolve_value(self.method, resolver, where="FieldDiscretization method")
        resolved.boundaries = tuple(
            resolve_value(self.boundaries, resolver, where="FieldDiscretization boundaries")
        )
        resolved.solver = resolve_value(self.solver, resolver, where="FieldDiscretization solver")
        resolved.nonlinear = resolve_value(
            self.nonlinear, resolver, where="FieldDiscretization nonlinear")
        resolved.preconditioner = resolve_value(
            self.preconditioner,
            resolver,
            where="FieldDiscretization preconditioner",
        )
        resolved.nullspace = resolve_value(
            self.nullspace, resolver, where="FieldDiscretization nullspace"
        )
        resolved.gauge = resolve_value(self.gauge, resolver, where="FieldDiscretization gauge")
        resolved.hierarchy_policy = resolve_value(
            self.hierarchy_policy,
            resolver,
            where="FieldDiscretization hierarchy_policy",
        )
        return resolved

    def inspect(self) -> dict[str, Any]:
        info = super().inspect()
        info["identity"] = (
            self.identity.token
            if all(reference.is_resolved for reference in self.declaration_references())
            else None
        )
        info["derived"] = {
            "order": "from method capabilities",
            "ghost_depth": "from method stencil",
            "hierarchy": "from hierarchy_policy + materialized layout",
        }
        return info


__all__ = [
    "CompositeHierarchySolve",
    "FieldDiscretization",
    "FieldDiscretizationProtocol",
    "FieldHierarchyPolicy",
    "InferHierarchyFromLayout",
    "LevelByLevelSolve",
    "field_discretization_data",
    "require_field_discretization",
]
