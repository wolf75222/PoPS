"""Callback-only provider reads retain exact roles through native brick emission."""

from __future__ import annotations

import pytest

from pops.codegen._native_model_provider_plan import (
    native_model_provider_plan,
    project_provider_locals,
    provider_slot_projection,
)
from pops.codegen.module_codegen import _emit_bricks, emit_cpp_source
from pops.physics._facade import Model


def _scalar_model(name: str) -> tuple[Model, object]:
    model = Model(name)
    (u,) = model.conservative_vars("u")
    model.primitive_vars(u)
    model.conservative_from([u])
    model.flux(x=[0 * u], y=[0 * u])
    model.eigenvalues(x=[0 * u], y=[0 * u])
    return model, u


def _callback_model(*, transitive_callbacks: bool = False) -> Model:
    model, u = _scalar_model("callback_provider_dependencies")
    nu = model.aux("nu")
    jac_aux = model.aux("jac_aux")
    speed_aux = model.aux("speed_aux")
    dt_aux = model.aux("dt_aux")
    model.source([-u * u])
    frequency = (
        model.primitive("nu_squared", nu * nu)
        if transitive_callbacks else nu * nu
    )
    model.source_frequency(frequency)
    # At the native oracle's u=2, the supplied jac_aux=-4 is exactly d(-u*u)/du.
    jacobian = model.primitive("jac_value", jac_aux) if transitive_callbacks else jac_aux
    speed = speed_aux + u * u
    timestep = dt_aux / (1 + u * u)
    if transitive_callbacks:
        speed = model.primitive("speed_value", speed)
        timestep = model.primitive("dt_value", timestep)
    model.source_jacobian([[jacobian]])
    model.stability_speed(speed)
    model.stability_dt(timestep)
    model._model_hash()
    return model


def _components(plan: tuple) -> set[str]:
    return {row["key"]["component"] for row in plan}


def _method(source: str, signature: str) -> str:
    start = source.index(signature)
    return source[start:source.index("\n  }", start)]


def _read(component: str, slot: int) -> str:
    return "const pops::Real %s = pops::provider_value<%d>(a);" % (component, slot)


def test_callback_only_inputs_are_in_their_exact_native_provider_roles() -> None:
    model = _callback_model()
    source_role = model._m._component_operator_consumer_plans["source_default"]
    flux_role = model._m._component_flux_consumer_plan
    assert _components(source_role) == {"nu", "jac_aux"}
    assert _components(flux_role) == {"speed_aux", "dt_aux"}
    native = native_model_provider_plan(model._m)
    assert _components(native) == {"nu", "jac_aux", "speed_aux", "dt_aux"}
    slots = {row["key"]["component"]: row["consumer_slot"] for row in native}
    _, bricks, _ = _emit_bricks(model._m, name="CallbackInputs")

    for signature, component in (
        ("pops::Real frequency(", "nu"),
        ("void jacobian(", "jac_aux"),
        ("pops::Real stability_speed(", "speed_aux"),
        ("pops::Real stability_dt(", "dt_aux"),
    ):
        assert _read(component, slots[component]) in _method(bricks, signature)
    assert "static constexpr int n_flux_providers = 2;" in bricks
    assert bricks.count("static constexpr int n_providers = 4;") == 2


def test_frequency_only_provider_survives_a_field_free_source() -> None:
    model, u = _scalar_model("frequency_only_provider")
    nu = model.aux("nu")
    model.source([-u * u])
    model.source_frequency(nu * nu)
    model._model_hash()

    role = model._m._component_operator_consumer_plans["source_default"]
    assert _components(role) == {"nu"}
    assert model._m._component_flux_consumer_plan == ()
    native = native_model_provider_plan(model._m)
    assert _components(native) == {"nu"}
    source = emit_cpp_source(model._m, name="FrequencyOnly", native_input_plan=native)
    assert "static constexpr int n_providers = 1;" in source
    assert _read("nu", native[0]["consumer_slot"]) in _method(source, "pops::Real frequency(")


def test_callback_primitive_recipe_reads_its_provider_before_using_it() -> None:
    model = _callback_model(transitive_callbacks=True)
    source_role = model._m._component_operator_consumer_plans["source_default"]
    assert _components(source_role) == {"nu", "jac_aux"}
    native = native_model_provider_plan(model._m)
    slots = {row["key"]["component"]: row["consumer_slot"] for row in native}
    source = emit_cpp_source(model._m, name="TransitiveFrequency", native_input_plan=native)
    frequency = _method(source, "pops::Real frequency(")
    read = frequency.index(_read("nu", slots["nu"]))
    primitive = frequency.index("const pops::Real nu_squared =")
    assert read < primitive < frequency.index("return nu_squared;")


def test_callback_primitive_recipes_retain_source_and_stability_provider_order() -> None:
    model = _callback_model(transitive_callbacks=True)
    assert _components(model._m._component_flux_consumer_plan) == {"speed_aux", "dt_aux"}
    native = native_model_provider_plan(model._m)
    slots = {row["key"]["component"]: row["consumer_slot"] for row in native}
    _, bricks, _ = _emit_bricks(model._m, name="TransitiveCallbacks")
    for signature, component, primitive in (
        ("void jacobian(", "jac_aux", "jac_value"),
        ("pops::Real stability_speed(", "speed_aux", "speed_value"),
        ("pops::Real stability_dt(", "dt_aux", "dt_value"),
    ):
        method = _method(bricks, signature)
        assert method.index(_read(component, slots[component])) < method.index(
            "const pops::Real %s =" % primitive
        )


@pytest.mark.parametrize("role_name", ("source_default", "physical_flux"))
@pytest.mark.parametrize("corruption", ("missing", "foreign_key", "contract", "provider"))
def test_callback_provider_projection_rejects_unauthenticated_union_rows(
    role_name: str, corruption: str,
) -> None:
    model = _callback_model()
    role = (
        model._m._component_flux_consumer_plan
        if role_name == "physical_flux"
        else model._m._component_operator_consumer_plans[role_name]
    )
    native = native_model_provider_plan(model._m)
    # Copy only the serializable row values. The resolved immutable plans remain untouched.
    damaged = [
        {**row, "key": dict(row["key"]), "contract": dict(row["contract"]),
         "provider": dict(row["provider"])}
        for row in native
    ]
    index = next(i for i, row in enumerate(damaged) if row["key"] == role[0]["key"])
    if corruption == "missing":
        damaged.pop(index)
    elif corruption == "foreign_key":
        damaged[index]["key"]["owner_qid"] += "/foreign"
    elif corruption == "contract":
        damaged[index]["contract"]["centering"] = "foreign-centering"
    else:
        damaged[index]["provider"]["producer"] = "foreign-provider"
    damaged_plan = tuple(damaged)
    read = _read(role[0]["key"]["component"], role[0]["consumer_slot"])
    for project in (
        lambda: provider_slot_projection(role, damaged_plan),
        lambda: project_provider_locals([read], role, damaged_plan),
    ):
        with pytest.raises(ValueError, match="do not authenticate an operation provider"):
            project()


@pytest.mark.compiler
def test_native_callbacks_read_their_distinct_qualified_inputs() -> None:
    from pops._native_selector import select_native_dimension
    from tests.python.unit.codegen.test_dsl_brick import _compile_and_run

    select_native_dimension(2)
    model = _callback_model(transitive_callbacks=True)
    _, bricks, _ = _emit_bricks(model._m, name="CallbackOracle")
    inputs = native_model_provider_plan(model._m)
    slots = {row["key"]["component"]: row["consumer_slot"] for row in inputs}
    assert set(slots) == {"nu", "jac_aux", "speed_aux", "dt_aux"}
    source = r'''
#include <cmath>
#include <iostream>
#include <pops/physics/bricks/bricks.hpp>
%s
using Hyp = pops_generated::CallbackOracleHyp;
using Src = pops_generated::CallbackOracleSrc;
static_assert(Hyp::n_providers == %d);
static_assert(Src::n_providers == Hyp::n_providers);
static_assert(Hyp::n_flux_providers == 2);
int main() {
  Hyp hyp;
  Src src;
  Hyp::State state{};
  state[0] = 2;
  pops::ProviderValues<Hyp::n_providers> values{};
  values[%d] = 3;
  values[%d] = -4;
  values[%d] = 5;
  values[%d] = 7;
  const auto source = src.apply(state, values);
  const auto frequency = src.frequency(state, values);
  pops::Real jacobian[1][1]{};
  src.jacobian(state, values, jacobian);
  const auto speed = hyp.template stability_speed<0>(state, values);
  const auto dt = hyp.stability_dt(state, values);
  if (source[0] != -4 || frequency != 9 || jacobian[0][0] != -4 || speed != 9
      || !std::isfinite(dt) || std::abs(dt - 1.4) > 1e-14) {
    return 1;
  }
  std::cout << "callback-values-ok\n";
  return 0;
}
''' % (bricks, len(inputs), slots["nu"], slots["jac_aux"], slots["speed_aux"], slots["dt_aux"])
    assert _compile_and_run(source, "callback_provider_dependencies") == "callback-values-ok\n"
