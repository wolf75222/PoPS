"""Physical NCP declarations must survive authoring and reject flux-only realization."""
from fractions import Fraction

import pytest

from pops.math import ddt, div
from pops.numerics import FiniteVolume, reconstruction, riemann, variables
from pops.physics import Model
from tests.python.support.physics_roles import FRAME, X_AXIS, Y_AXIS


def _model():
    model = Model("nonconservative contract", frame=FRAME)
    state = model.state("U", components=("mass", "stress"))
    mass, stress = state
    flux = model.flux("F", frame=FRAME, state=state,
                      components={X_AXIS: (mass, stress), Y_AXIS: (mass, stress)},
                      waves={X_AXIS: (1, 1), Y_AXIS: (1, 1)})
    product = model.nonconservative_product("B grad U", state=state,
        matrices={X_AXIS: ((0, 0), (stress / mass, 2)),
                  Y_AXIS: ((0, 0), (3, mass))}, conservative_components=("mass",))
    return model, state, flux, product


def test_complete_signed_product_and_constitutive_body_remain_in_module():
    model, state, flux, product = _model()
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    assert [(row.kind, row.coefficient) for row in rate.occurrences] == [
        ("flux", -1), ("nonconservative", -1)]
    assert rate.occurrences[1].payload is product
    assert model.rate_contract(rate)["nonconservative_products"] == (product,)
    module = model.module
    operator = module.operator_registry().get(product.reg_name)
    assert operator.lowering["nonconservative_law"] is product.law
    assert len(operator.body) == 2
    assert product.law.conservative_components == ("mass",)
    assert module.operator_registry().get(rate.local_id).lowering["physical_balance"] is rate.view
    assert "path-conservative" in rate.view.legacy_incompatibility()


def test_partition_keeps_each_signed_nonconservative_occurrence():
    model, state, flux, product = _model()
    rate = model.rate("balance", equation=ddt(state) == (
        -div(flux) - product + Fraction(2, 3) * product))
    selected = rate.select(product)
    assert selected.view.ordinals == (1, 2)
    assert [row.coefficient for row in selected.occurrences] == [-1, Fraction(2, 3)]
    assert all(row.payload is product for row in selected.occurrences)


def test_conservative_finite_volume_cannot_discard_the_product():
    model, state, flux, product = _model()
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    method = FiniteVolume(flux=flux, variables=variables.Conservative(state),
                          reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov())
    with pytest.raises(ValueError, match="cannot omit a physical nonconservative product"):
        method.validate_rate_contract(model.rate_contract(rate))


def test_foreign_product_with_identical_names_is_not_accepted():
    model, state, flux, _ = _model()
    foreign, _, _, product = _model()
    assert foreign is not model
    with pytest.raises(ValueError, match="nonconservative"):
        model.rate("foreign", equation=ddt(state) == -div(flux) - product)


def test_declared_conservative_rows_must_be_exactly_zero():
    model, state, _, _ = _model()
    before = dict(model._nonconservative_products)
    with pytest.raises(ValueError, match="exactly zero"):
        model.nonconservative_product("false conservation", state=state,
            matrices={X_AXIS: ((1, 0), (0, 0)), Y_AXIS: ((0, 0), (0, 0))},
            conservative_components=("mass",))
    assert model._nonconservative_products == before


@pytest.mark.parametrize("matrices", [
    {X_AXIS: ((0, 0), (0, 0))},
    {"x": ((0, 0), (0, 0)), "y": ((0, 0), (0, 0))},
    {X_AXIS: ((0,),), Y_AXIS: ((0,),)},
])
def test_incomplete_or_unqualified_directional_matrices_are_refused(matrices):
    model, state, _, _ = _model()
    with pytest.raises(ValueError):
        model.nonconservative_product("bad matrix", state=state, matrices=matrices)


def test_foreign_coefficient_and_duplicate_name_are_atomic_refusals():
    model, state, _, _ = _model()
    foreign, other, _, _ = _model()
    before = dict(model._nonconservative_products)
    with pytest.raises(ValueError, match="foreign quantity owner"):
        model.nonconservative_product("foreign matrix", state=state,
            matrices={X_AXIS: ((0, 0), (other[0], 0)), Y_AXIS: ((0, 0), (0, 0))})
    with pytest.raises(ValueError, match="already declared"):
        model.nonconservative_product("B grad U", state=state,
            matrices={X_AXIS: ((0, 0), (0, 0)), Y_AXIS: ((0, 0), (0, 0))})
    assert model._nonconservative_products == before


def test_constitutive_result_is_an_ephemeral_output_not_a_stored_field():
    from pops.codegen.resolved_operations import build_resolved_operations
    model, state, flux, product = _model()
    model.rate("balance", equation=ddt(state) == -div(flux) - product)
    resolved = build_resolved_operations(model.module)
    constitutive = [operation for operation in resolved.operations
                    if operation.native_route == "program:nonconservative_constitutive_law"]
    assert len(constitutive) == 1
    assert product.name + "_product" not in model.module.field_spaces()


def test_typed_auxiliary_persists_without_creating_a_second_provider_route():
    from pops.analytic import coordinate
    from pops.fields import AnalyticAux
    from pops.model.provider_pack import build_operator_provider_pack
    from pops.codegen.resolved_operations import build_resolved_operations
    model = Model("typed path geometry", frame=FRAME)
    state = model.state("U", components=("mass", "stress"))
    mass, stress = state
    metric = model.auxiliary("metric", frame=FRAME.canonical_id)
    first_module = model.module
    assert set(first_module.aux()) == {"metric"}
    assert not first_module.field_spaces()["fields"].components
    flux = model.flux("F", frame=FRAME, state=state,
                      components={X_AXIS: (mass * metric, stress), Y_AXIS: (mass, stress)})
    product = model.nonconservative_product("B", state=state,
        matrices={X_AXIS: ((0, 0), (metric, stress)), Y_AXIS: ((0, 0), (0, 0))},
        conservative_components=("mass",))
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    module = model.module
    assert module is not first_module
    assert module.aux()["metric"] == first_module.aux()["metric"]
    target = module.aux_handle(module.aux()["metric"])
    module.aux_provider(AnalyticAux(target, 0 * coordinate(FRAME, X_AXIS) + 2, frame=FRAME))
    for name in (flux.reg_name, product.reg_name, rate.local_id):
        pack = build_operator_provider_pack(module, module.operator_registry().get(name))
        assert [(key.space_kind, key.space_name, key.component) for key in pack] == [
            ("aux", "metric", "metric")]
    build_resolved_operations(module)
    with pytest.raises(ValueError, match="already declared"):
        model.auxiliary("metric")
    with pytest.raises(ValueError, match="already declares"):
        model.aux("metric")
    legacy = Model("legacy field", frame=FRAME)
    legacy.aux("metric")
    with pytest.raises(ValueError, match="already declared"):
        legacy.auxiliary("metric")


def test_undeclared_auxiliary_still_cannot_enter_the_physical_law():
    from pops._ir import Var
    model, state, _, _ = _model()
    with pytest.raises(ValueError, match="declared field quantities"):
        model.nonconservative_product("unowned metric", state=state,
            matrices={X_AXIS: ((0, 0), (Var("metric", "aux"), 0)),
                      Y_AXIS: ((0, 0), (0, 0))})
