"""Normalize combined legacy constructors into physical and numerical records."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .discretization import field_discretization_data
from .gauges import MeanValueGauge
from .nullspace import ConstantNullspace
from .problem import FieldBoundary, FieldProblem, FieldProblemError, SharedMeanGauge


@dataclass(frozen=True, slots=True)
class NormalizedFieldDefinition:
    problem: FieldProblem
    numerical: dict[str, Any]

    def to_data(self) -> dict[str, Any]:
        return {"problem": self.problem.to_data(), "numerical": self.numerical}


def normalize_field_definition(operator: FieldProblem, discretization: Any) -> NormalizedFieldDefinition:
    """Separate physical BC/kernel/gauge from the chosen discrete enforcement.

    This projection retains the complete original method and solver options. It is
    shared by the legacy adapter and general field route; it never chooses physics.
    """
    from .operator import FieldOperator
    if not isinstance(operator, FieldProblem):
        raise TypeError("field definition requires a FieldProblem")
    numerical = field_discretization_data(discretization, where="field normalization")
    combined_boundaries = tuple(discretization.boundaries)
    authored_boundaries = () if isinstance(operator, FieldOperator) else operator.boundaries
    unknowns = operator.unknowns
    if len(unknowns) > 1 and combined_boundaries:
        raise FieldProblemError("field.boundary.joint_ownership_required",
                                "joint field boundaries must name their exact unknowns")
    combined = tuple(FieldBoundary(operator.unknowns[0], relation)
                     for relation in combined_boundaries)
    if authored_boundaries and combined and tuple(row.to_data() for row in authored_boundaries) \
            != tuple(row.to_data() for row in combined):
        raise FieldProblemError("field.boundary.conflicting_authorities",
                                "physical and combined boundary declarations disagree")
    boundaries = authored_boundaries or combined
    gauge = None if isinstance(operator, FieldOperator) else operator.gauge
    if discretization.gauge is not None or discretization.nullspace is not None:
        if len(operator.unknowns) != 1 or not isinstance(discretization.nullspace, ConstantNullspace) \
                or not isinstance(discretization.gauge, MeanValueGauge):
            raise FieldProblemError("field.gauge.joint_required",
                                    "a joint physical gauge cannot be replaced by independent numerical gauges")
        combined_gauge = SharedMeanGauge(operator.unknowns, discretization.gauge.value)
        if gauge is not None and gauge.to_data() != combined_gauge.to_data():
            raise FieldProblemError("field.gauge.conflicting_authorities", "field gauges disagree")
        gauge = combined_gauge
    physical = FieldProblem(operator.name, unknowns=operator.unknowns,
                            equations=operator.equations, boundaries=boundaries, gauge=gauge,
                            branch=getattr(operator, "branch", None), outputs=operator.outputs)
    numerical = dict(numerical)
    numerical.pop("boundaries")
    numerical.pop("gauge")
    numerical.pop("nullspace")
    numerical["boundary_enforcement"] = {
        "method": numerical["method"], "physical_relation_count": len(boundaries)}
    return NormalizedFieldDefinition(physical, numerical)
