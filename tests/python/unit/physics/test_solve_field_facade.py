"""The final field path separates physics, numerics, and Case ownership."""
from __future__ import annotations

import pytest

from pops.fields import (
    CellCenteredSecondOrder,
    FieldDiscretization,
    FieldOperator,
    FieldOutput,
    GradientOutput,
)
from pops.fields.bcs import (
    AllPhysicalBoundaries,
    BoundaryCondition,
    Dirichlet,
)
from pops.math import laplacian
from pops.physics import Model
from pops.problem import Case
from pops.solvers.elliptic import GeometricMG


def _final_field_assembly():
    model = Model("electrostatic")
    (charge,) = model.state("U", components=["charge"])
    potential = model.field("potential")
    operator = model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=(-laplacian(potential) == charge),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential),
        ),
    )
    discretization = FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(
            BoundaryCondition(AllPhysicalBoundaries(), Dirichlet(0.0)),
        ),
        solver=GeometricMG(),
    )
    case = Case("field-case")
    case.block("material", model)
    field = case.field(operator, discretization)
    return case, model, operator, discretization, field


def test_case_field_binds_one_physical_operator_to_one_numerical_plan() -> None:
    case, model, operator, discretization, field = _final_field_assembly()

    assert isinstance(operator, FieldOperator)
    assert isinstance(discretization, FieldDiscretization)
    assert model.field_operators[operator.name] is operator
    assert case.fields() == {operator.name: field}
    assert field.local_id == operator.name
    assert case.resolve(field).owner_path == case.owner_path.canonical()

    registered = case._fields.get(operator.name)
    assert registered.operator is operator
    assert registered.discretization is discretization


def test_physics_and_numerics_are_not_mixed() -> None:
    _, _, operator, discretization, _ = _final_field_assembly()

    for numerical_name in ("method", "boundaries", "solver", "hierarchy_policy"):
        assert not hasattr(operator, numerical_name)
    for physical_name in ("unknown", "equation", "providers", "outputs"):
        assert not hasattr(discretization, physical_name)


def test_case_field_rejects_duplicate_physical_operator_identity() -> None:
    case, _, operator, discretization, _ = _final_field_assembly()

    with pytest.raises(ValueError, match="already exists"):
        case.field(operator, discretization)
