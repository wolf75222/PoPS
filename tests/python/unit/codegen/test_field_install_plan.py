from __future__ import annotations

import pytest

from pops.codegen._orchestration_compile import capture_field_plans
from pops.codegen.lowering_coverage import LoweringRejection
from pops.descriptors import Descriptor
from pops.fields import (
    CellCenteredSecondOrder,
    CompositeHierarchySolve,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    LevelByLevelSolve,
    MeanValueGauge,
    boundary_value,
    logical_time,
)
from pops.fields._identity import strict_field_data
from pops.fields.bcs import (
    AllPhysicalBoundaries,
    BoundaryCondition,
    Dirichlet,
    Mixed,
    Neumann,
)
from pops.math import Laplacian, elliptic_terms, laplacian
from pops.layouts import Uniform
from pops.physics import Model
from pops.problem import Case
from pops.solvers.elliptic import GeometricMG
from pops.solvers.options import CompositeFAC
from tests.python.support.layout_plan import cartesian_grid, final_amr_layout


_LAYOUT = Uniform(cartesian_grid(n=16, periodic=False))
_ONE_LEVEL_AMR_LAYOUT = final_amr_layout(
    cartesian_grid(n=16, periodic=False), max_levels=1)


class ExternalFieldPlan(Descriptor):
    """Test-owned plan provider; no PoPS registry knows this concrete class."""

    category = "external_field_discretization"
    provider_id = "pops.test.external-field-plan.v1"

    def __init__(self, inner: FieldDiscretization) -> None:
        self._inner = inner

    @property
    def method(self):
        return self._inner.method

    @property
    def boundaries(self):
        return self._inner.boundaries

    @property
    def solver(self):
        return self._inner.solver

    @property
    def nonlinear(self):
        return self._inner.nonlinear

    @property
    def preconditioner(self):
        return self._inner.preconditioner

    @property
    def nullspace(self):
        return self._inner.nullspace

    @property
    def gauge(self):
        return self._inner.gauge

    @property
    def hierarchy_policy(self):
        return self._inner.hierarchy_policy

    def to_data(self):
        data = self._inner.to_data()
        data["provider_id"] = self.provider_id
        return data

    def available(self, context=None):
        return self._inner.available(context)

    def validate(self, context=None):
        return self._inner.validate(context)

    def declaration_references(self):
        return self._inner.declaration_references()

    def resolve_references(self, resolver):
        return type(self)(self._inner.resolve_references(resolver))

    def inspect(self):
        data = self._inner.inspect()
        data["provider_id"] = self.provider_id
        return data


def _field(model: Model, name: str, rhs: object):
    unknown = model.field(name)
    return model.field_operator(
        name,
        unknown=unknown,
        equation=(-laplacian(unknown) == rhs),
        outputs=(FieldOutput(name, unknown),),
    )


def _disc(condition: object, *, singular: bool = False) -> FieldDiscretization:
    return FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), condition),),
        solver=GeometricMG(),
        nullspace=ConstantNullspace() if singular else None,
        gauge=MeanValueGauge(0) if singular else None,
    )


def _case(*conditions: object):
    model = Model("field-model-%d" % len(conditions))
    state = model.state("U", components=["rho", "sigma"])
    operators = [
        _field(model, "potential_%d" % index, state[index])
        for index in range(len(conditions))
    ]
    problem = Case(name="field-case-%d" % len(conditions))
    problem.block("material", model)
    handles = []
    for operator, condition in zip(operators, conditions, strict=True):
        handles.append(problem.field(
            operator,
            _disc(condition, singular=isinstance(condition, Neumann)),
        ))
    return problem, handles


def _one_level_amr_plan(*, hierarchy_policy, solver):
    model = Model("one-level-amr-field-model")
    (rho,) = model.state("U", components=["rho"])
    operator = _field(model, "potential", rho)
    problem = Case(name="one-level-amr-field-case")
    problem.block("material", model)
    problem.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Dirichlet(0.0)),),
        solver=solver,
        hierarchy_policy=hierarchy_policy,
    ))
    return capture_field_plans(
        problem, lambda value: value, target="amr_system", layout=_ONE_LEVEL_AMR_LAYOUT,
    )["potential"]


def test_one_level_amr_resolve_preserves_level_by_level_capability() -> None:
    plan = _one_level_amr_plan(
        hierarchy_policy=LevelByLevelSolve(), solver=GeometricMG())

    assert plan.native_options["hierarchy"] == "level_local"


def test_composite_fac_refuses_a_single_level_amr_backend() -> None:
    with pytest.raises(LoweringRejection, match="multi-level AMR backend"):
        _one_level_amr_plan(
            hierarchy_policy=CompositeHierarchySolve(),
            solver=GeometricMG(fac=CompositeFAC()),
        )


@pytest.mark.parametrize(
    "condition",
    (Dirichlet(2.0), Mixed(alpha=2.0, beta=0.5, value=5.0), Neumann(0.0)),
)
def test_field_plan_lowers_complete_boundary_and_qualified_provider(condition: object) -> None:
    problem, _ = _case(condition)
    plan = next(iter(capture_field_plans(
        problem, lambda value: value, target="system", layout=_LAYOUT).values()))

    options = plan.native_options
    assert options["provider_identity"]["contributions"][0]["provider"]["block_ref"][
        "local_id"] == "material"
    assert options["provider_slot"]
    assert options["provider_identity_text"]
    assert options["provider_pack"][0]["owner_block"] == "material"
    assert options["provider_pack"][0]["key"] == "potential_0"
    assert "kind" not in options["provider_pack"][0]
    assert len(options["boundary_faces"]) == 4
    assert all(row.disposition != "rejected" for row in plan.coverage)


def test_two_fields_keep_distinct_qualified_provider_slots_and_solver_plans() -> None:
    problem, _ = _case(Dirichlet(0.0), Mixed(1.0, 1.0, 0.0))
    plans = capture_field_plans(
        problem, lambda value: value, target="system", layout=_LAYOUT)

    assert set(plans) == {"potential_0", "potential_1"}
    assert len({plan.native_options["provider_slot"] for plan in plans.values()}) == 2
    assert len({plan.identity.token for plan in plans.values()}) == 2


@pytest.mark.parametrize("target", ("system", "amr_system"))
@pytest.mark.parametrize("sign", (-1, 1))
def test_gradient_output_sign_is_part_of_the_exact_native_output_route(
    target: str, sign: int,
) -> None:
    model = Model("signed-gradient-%s-%d" % (target, sign))
    (rho,) = model.state("U", components=["rho"])
    unknown = model.field("potential")
    operator = model.field_operator(
        "potential",
        unknown=unknown,
        equation=(-laplacian(unknown) == rho),
        outputs=(
            FieldOutput("potential", unknown),
            GradientOutput("electric_field", unknown, sign=sign),
        ),
    )
    problem = Case(name="signed-gradient-case-%s-%d" % (target, sign))
    problem.block("material", model)
    problem.field(operator, _disc(Dirichlet(0.0)))

    plan = capture_field_plans(
        problem, lambda value: value, target=target, layout=_LAYOUT)["potential"]

    assert plan.native_options["output_route"]["components"] == (
        "potential", "electric_field_x", "electric_field_y")
    assert plan.native_options["output_route"]["gradient_sign"] == sign


def test_non_unit_laplacian_scale_is_preserved_by_exact_rhs_normalization() -> None:
    model = Model("scaled-laplacian-model")
    (rho,) = model.state("U", components=["rho"])
    unknown = model.field("potential")
    operator = model.field_operator(
        "potential",
        unknown=unknown,
        equation=(Laplacian(unknown, scale=-2.0) == rho),
        outputs=(FieldOutput("potential", unknown),),
    )

    lowered_model = model.__pops_compiler_lowering__().emit_model._m
    lowered_rhs = lowered_model._elliptic_fields["potential"]["rhs"]
    assert strict_field_data(lowered_rhs) == strict_field_data(rho / 2.0)

    problem = Case(name="scaled-laplacian-case")
    problem.block("material", model)
    problem.field(operator, _disc(Dirichlet(0.0)))
    plan = capture_field_plans(
        problem, lambda value: value, target="system", layout=_LAYOUT)["potential"]

    (laplacian_term,) = elliptic_terms(plan.operator.equation.lhs)
    assert laplacian_term.scale == -2.0


def test_native_lowering_rejects_a_mutated_same_owner_output_source() -> None:
    model = Model("defensive-output-source-model")
    (rho,) = model.state("U", components=["rho"])
    unknown = model.field("potential")
    peer = model.field("peer")
    operator = model.field_operator(
        "potential",
        unknown=unknown,
        equation=(-laplacian(unknown) == rho),
        outputs=(FieldOutput("potential", unknown),),
    )
    problem = Case(name="defensive-output-source-case")
    problem.block("material", model)
    problem.field(operator, _disc(Dirichlet(0.0)))

    # Registration normally validates this invariant.  Mutate afterwards to prove the native
    # lowering boundary independently fails closed instead of routing the solved ``potential``
    # values under a descriptor claiming they came from ``peer``.
    operator.outputs = (FieldOutput("potential", peer),)
    with pytest.raises(ValueError, match="FieldOutput source disagrees.*solved unknown"):
        capture_field_plans(
            problem, lambda value: value, target="system", layout=_LAYOUT)


def test_external_field_plan_crosses_registration_resolution_and_lowering_structurally() -> None:
    model = Model("external-field-plan-model")
    (rho,) = model.state("U", components=["rho"])
    operator = _field(model, "potential", rho)
    problem = Case(name="external-field-plan-case")
    problem.block("material", model)
    problem.field(operator, ExternalFieldPlan(_disc(Dirichlet(0.0))))

    plan = capture_field_plans(
        problem, lambda value: value, target="system", layout=_LAYOUT)["potential"]

    assert plan.discretization.provider_id == ExternalFieldPlan.provider_id
    assert plan.to_data()["discretization"]["provider_id"] == ExternalFieldPlan.provider_id


def test_dynamic_boundary_lowers_to_generated_parameter_launcher() -> None:
    model = Model("dynamic-field-model")
    (rho,) = model.state("U", components=["rho"])
    unknown = model.field("potential")
    operator = model.field_operator(
        "potential", unknown=unknown, equation=(-laplacian(unknown) == rho),
        outputs=(FieldOutput("potential", unknown),),
    )
    problem = Case(name="dynamic-field-case")
    problem.block("material", model)
    from pops.params import RuntimeParam
    boundary_value = RuntimeParam("wall_value")
    problem.param(boundary_value)
    problem.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(
            AllPhysicalBoundaries(), Dirichlet(problem.value(boundary_value))),),
        solver=GeometricMG(),
    ))

    plans = capture_field_plans(
        problem, lambda value: value, target="system", layout=_LAYOUT)
    plan = plans["potential"]
    assert plan.native_options["boundary_kernel_required"] is True
    assert plan.native_options["boundary_iterate_dependent"] is False
    assert [handle.local_id for handle in plan.boundary_parameter_handles()] == ["wall_value"]

    from pops.codegen.program_emit_field_boundaries import emit_field_boundaries
    source = emit_field_boundaries(None, None, plans, "system")
    assert 'extern "C" void pops_install_field_boundaries(void* sys)' in source
    assert "prepare_field_boundary_residual_route_0" in source
    assert "params[0]" in source


def test_iterate_dependent_boundary_requires_newton_and_emits_exact_jvp() -> None:
    from pops.math import ValueExpr
    from pops.solvers.nonlinear import Newton

    model = Model("nonlinear-boundary-model")
    (rho,) = model.state("U", components=["rho"])
    unknown = model.field("potential")
    operator = model.field_operator(
        "potential", unknown=unknown, equation=(-laplacian(unknown) == rho),
        outputs=(FieldOutput("potential", unknown),),
    )
    problem = Case(name="nonlinear-boundary-case")
    problem.block("material", model)
    u = ValueExpr(unknown)
    problem.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(
            AllPhysicalBoundaries(), Mixed(alpha=1.0, beta=1.0, value=u * u)),),
        solver=GeometricMG(),
        nonlinear=Newton(),
    ))

    plans = capture_field_plans(
        problem, lambda value: value, target="system", layout=_LAYOUT)
    plan = plans["potential"]
    assert plan.native_options["boundary_iterate_dependent"] is True
    assert plan.native_options["nonlinear"]["target"] == "system"

    from pops.codegen.program_emit_field_boundaries import emit_field_boundaries
    source = emit_field_boundaries(None, None, plans, "system")
    assert "prepare_field_boundary_jvp_route_0" in source
    assert ":boundary-jvp" in source
    assert "dvalue" in source
    assert "const auto& params = *context.parameters" not in source


def test_boundary_state_component_and_logical_time_lower_to_direct_provider_pack() -> None:
    model = Model("prepared-boundary-model")
    state = model.state("U", components=["rho", "momentum"])
    rho, _ = state
    unknown = model.field("potential")
    operator = model.field_operator(
        "potential", unknown=unknown, equation=(-laplacian(unknown) == rho),
        outputs=(FieldOutput("potential", unknown),),
    )
    problem = Case(name="prepared-boundary-case")
    block = problem.block("material", model)
    prepared_rho = boundary_value(block[state], "rho")
    problem.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(
            AllPhysicalBoundaries(), Dirichlet(prepared_rho + logical_time("time"))),),
        solver=GeometricMG(),
    ))

    plans = capture_field_plans(
        problem, lambda value: value, target="system", layout=_LAYOUT)
    plan = plans["potential"]
    dependencies = plan.native_options["boundary_dependencies"]
    assert [(row["owner_block"], row["component"])
            for row in dependencies["states"]] == [("material", 0)]
    assert dependencies["logical_time"] == ("time",)

    from pops.codegen.program_emit_field_boundaries import emit_field_boundaries
    source = emit_field_boundaries(None, None, plans, "system")
    assert "context.states[0]->fab(li).const_array()" in source
    assert "state0(i, j, 0)" in source
    assert "context.point.time" in source


def test_boundary_state_value_requires_explicit_component_contract() -> None:
    from pops.math import ValueExpr

    model = Model("ambiguous-boundary-model")
    state = model.state("U", components=["rho", "momentum"])
    rho, _ = state
    unknown = model.field("potential")
    operator = model.field_operator(
        "potential", unknown=unknown, equation=(-laplacian(unknown) == rho),
        outputs=(FieldOutput("potential", unknown),),
    )
    problem = Case(name="ambiguous-boundary-case")
    block = problem.block("material", model)
    problem.field(operator, FieldDiscretization(
        method=CellCenteredSecondOrder(),
        boundaries=(BoundaryCondition(
            AllPhysicalBoundaries(), Dirichlet(ValueExpr(block[state]))),),
        solver=GeometricMG(),
    ))

    with pytest.raises(Exception, match="boundary_value"):
        capture_field_plans(
            problem, lambda value: value, target="system", layout=_LAYOUT)
