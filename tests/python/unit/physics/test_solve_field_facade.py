"""The final field path separates physics, numerics, and Case ownership."""
from __future__ import annotations

import pytest

from pops.domain import Rectangle
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
from pops.frames import Cartesian2D
from pops.math import ddt, div, laplacian
from pops.physics import Model
from pops.problem import Case
from pops.solvers.elliptic import GeometricMG


def _final_field_assembly(*, gradient_sign: int = -1):
    frame = Rectangle(
        "electrostatic-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("electrostatic", frame=frame)
    state = model.state("U", components=["charge"])
    (charge,) = state
    transport = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (charge,), y_axis: (0.0 * charge,)},
        waves={x_axis: (1.0,), y_axis: (0.0,)},
    )
    model.rate("transport_rate", equation=ddt(state) == -div(transport))
    potential = model.field("potential")
    operator = model.field_operator(
        "electrostatic",
        unknown=potential,
        equation=(-laplacian(potential) == charge),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential, sign=gradient_sign),
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


def test_gradient_output_sign_reaches_model_hash_and_both_native_loaders() -> None:
    _, negative, _, _, _ = _final_field_assembly(gradient_sign=-1)
    _, positive, _, _, _ = _final_field_assembly(gradient_sign=1)
    negative_emit = negative.__pops_compiler_lowering__().emit_model
    positive_emit = positive.__pops_compiler_lowering__().emit_model

    assert negative_emit._m._elliptic_fields["electrostatic"]["gradient_sign"] == -1
    assert positive_emit._m._elliptic_fields["electrostatic"]["gradient_sign"] == 1
    assert negative_emit._m._model_hash() != positive_emit._m._model_hash()
    for target in ("system", "amr_system"):
        negative_loader = negative_emit._m.emit_cpp_native_loader(target=target)
        positive_loader = positive_emit._m.emit_cpp_native_loader(target=target)
        assert 'register_elliptic_field(name, "electrostatic", 5, 6, 7, -1);' \
            in negative_loader
        assert 'register_elliptic_field(name, "electrostatic", 5, 6, 7, 1);' \
            in positive_loader
