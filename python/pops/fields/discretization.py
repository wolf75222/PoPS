"""Numerical field plans, strictly separated from physical FieldOperator declarations."""

from __future__ import annotations

from copy import copy
from typing import Any

from pops.descriptors import BrickDescriptor, Descriptor, reject_string_selector
from pops.descriptors_report import CapabilitySet, RequirementSet
from pops.identity import Identity

from ._identity import (
    _register_builtin_descriptor_projection,
    _seal_builtin_descriptor_projections,
    field_identity,
    strict_field_data,
)
from .bcs import BoundaryCondition
from .gauges import FieldGauge
from ._references import collect_references, resolve_value


class FieldHierarchyPolicy(Descriptor):
    """Small extension interface for hierarchy coupling semantics."""

    category = "field_hierarchy_policy"

    def to_data(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "options": self.options()}

    def resolve(self, capabilities: Any) -> Any:
        resolver = getattr(capabilities, "resolve_hierarchy", None)
        if not callable(resolver):
            raise TypeError("field hierarchy policy requires typed solve capabilities")
        return resolver(self.options()["policy"])


class InferHierarchyFromLayout(FieldHierarchyPolicy):
    """Resolve uniform/level-local/composite behavior from method and layout capabilities."""

    def options(self) -> dict[str, Any]:
        return {"policy": "infer_from_layout"}


class LevelByLevelSolve(FieldHierarchyPolicy):
    def options(self) -> dict[str, Any]:
        return {"policy": "level_local"}


class CompositeHierarchySolve(FieldHierarchyPolicy):
    def options(self) -> dict[str, Any]:
        return {"policy": "composite"}


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


def _brick_data(value: BrickDescriptor) -> dict[str, Any]:
    if value.expression is not None or value.builder is not None:
        raise TypeError(
            "field methods with expression/builder payloads require an exact registered projection"
        )
    return {
        "name": value.name,
        "brick_type": value.brick_type,
        "category": value.category,
        "native_id": value.native_id,
        "scheme": value.scheme,
        "requirements": value.requirements,
        "capabilities": value.capabilities,
        "options": value.options,
        "available": value.available().ok,
    }


_register_builtin_descriptor_projection(BrickDescriptor, _brick_data)


def _geometric_mg_data(value: Any) -> dict[str, Any]:
    return {
        "scheme": value.scheme,
        "smoother": {"type": type(value.smoother).__name__, "options": value.smoother.options()},
        "coarse": {"type": type(value.coarse).__name__, "options": value.coarse.options()},
        "tolerance": {"type": type(value.tolerance).__name__, "options": value.tolerance.options()},
        "max_cycles": value.max_cycles,
        "min_coarse": value.min_coarse,
        "pre_sweeps": value.pre_sweeps,
        "post_sweeps": value.post_sweeps,
        "bottom_sweeps": value.bottom_sweeps,
        "amr_composite": (
            None
            if value.amr_composite is None
            else {
                "type": type(value.amr_composite).__name__,
                "options": value.amr_composite.options(),
            }
        ),
    }


def _register_builtin_solver_projections() -> None:
    from pops.solvers.elliptic import FFT, GeometricMG

    _register_builtin_descriptor_projection(GeometricMG, _geometric_mg_data)
    _register_builtin_descriptor_projection(
        FFT, lambda value: {"scheme": value.scheme, "spectral": value.spectral}
    )


_register_builtin_solver_projections()
_seal_builtin_descriptor_projections()


class FieldDiscretization(Descriptor):
    """All numerical choices for lowering one FieldOperator.

    No redundant ``order``, ghost depth, or AMR boolean is accepted: those consequences are
    derived from the selected method and hierarchy policy during resolution/lowering.
    """

    category = "field_discretization"

    def __init__(
        self,
        *,
        method: Any,
        boundaries: Any,
        solver: Any,
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
            "schema_version": 1,
            "method": strict_field_data(self.method),
            "boundaries": strict_field_data(self.boundaries),
            "solver": strict_field_data(self.solver),
            "preconditioner": strict_field_data(self.preconditioner),
            "nullspace": strict_field_data(self.nullspace),
            "gauge": strict_field_data(self.gauge),
            "hierarchy_policy": strict_field_data(self.hierarchy_policy),
        }

    def options(self) -> dict[str, Any]:
        def label(value: Any) -> str | None:
            return None if value is None else getattr(value, "name", type(value).__name__)

        return {
            "method": label(self.method),
            "boundaries": [boundary.options() for boundary in self.boundaries],
            "solver": label(self.solver),
            "preconditioner": label(self.preconditioner),
            "nullspace": label(self.nullspace),
            "gauge": label(self.gauge),
            "hierarchy_policy": self.hierarchy_policy.options()["policy"],
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
                "hierarchy_policy": self.hierarchy_policy.options()["policy"],
            }
        )

    def validate(self, context: Any = None) -> bool:
        _validate_route(self.method, context)
        _validate_route(self.solver, context)
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
    "FieldHierarchyPolicy",
    "InferHierarchyFromLayout",
    "LevelByLevelSolve",
]
