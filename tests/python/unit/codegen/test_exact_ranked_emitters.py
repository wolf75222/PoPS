"""Exact-ranked C++ emission from the authenticated x[/y[/z]] physics map."""
from __future__ import annotations

import subprocess
from types import MappingProxyType, SimpleNamespace

import pytest

from pops._ir import Var
from pops.codegen.module_emit_helpers import _axis_values
from pops.codegen.module_lowering import lower_and_validate
from pops.codegen.toolchain import native_compile_environment, pops_loader_build_flags
from pops.frames import X_AXIS, Y_AXIS, Z_AXIS
from pops.math import sqrt
from pops.params import RuntimeParam
from pops.physics import Density, Energy, Momentum
from pops.physics._facade import Model
from tests.python.support.requirements import repo_include


AXES = ("x", "y", "z")
ROLE_AXES = (X_AXIS, Y_AXIS, Z_AXIS)


def _emit_cpp_brick(model: Model, **kwargs: object) -> tuple[str, object]:
    """Emit only after canonical Module/provider-pack lowering."""
    emit_model, source_module = lower_and_validate(model, facade=model)
    return emit_model._m.emit_cpp_brick(**kwargs), source_module


def _scalar_model(dimension: int, *, explicit_waves: bool = True,
                  eigenvalues: bool = True) -> Model:
    model = Model("ranked_scalar_%d" % dimension)
    (q,) = model.conservative_vars("q")
    axes = AXES[:dimension]
    model.flux(**{
        axis: [(ordinal + 1) * q] for ordinal, axis in enumerate(axes)
    })
    if explicit_waves:
        model.wave_speeds(**{
            axis: (-(ordinal + 1) + 0 * q, ordinal + 1 + 0 * q)
            for ordinal, axis in enumerate(axes)
        })
    elif eigenvalues:
        model.eigenvalues(**{
            axis: [ordinal + 1 + 0 * q] for ordinal, axis in enumerate(axes)
        })
    model.primitive_vars(q)
    model.conservative_from([q])
    return model


def _euler_model(dimension: int) -> Model:
    gamma = 1.4
    model = Model("ranked_euler_%d" % dimension)
    names = ("rho", "rho_u", "rho_v", "rho_w", "E")
    selected_names = (names[0], *names[1:dimension + 1], names[-1])
    selected_roles = (
        Density(),
        *(Momentum(ROLE_AXES[axis]) for axis in range(dimension)),
        Energy(),
    )
    conservative = model.conservative_vars(*selected_names, roles=selected_roles)
    rho, momenta, energy = conservative[0], conservative[1:-1], conservative[-1]
    velocities = [
        model.primitive(("u", "v", "w")[axis], momenta[axis] / rho)
        for axis in range(dimension)
    ]
    kinetic = sum(velocity * velocity for velocity in velocities)
    pressure = model.primitive(
        "p", (gamma - 1.0) * (energy - 0.5 * rho * kinetic)
    )
    enthalpy = (energy + pressure) / rho
    sound = sqrt(gamma * pressure / rho)
    fluxes = {}
    eigenvalues = {}
    for normal, axis in enumerate(AXES[:dimension]):
        normal_velocity = velocities[normal]
        components = [momenta[normal]]
        components.extend(
            momenta[tangent] * normal_velocity
            + (pressure if tangent == normal else 0)
            for tangent in range(dimension)
        )
        components.append(rho * enthalpy * normal_velocity)
        fluxes[axis] = components
        eigenvalues[axis] = [
            normal_velocity - sound,
            normal_velocity,
            normal_velocity + sound,
        ]
    model.flux(**fluxes)
    model.eigenvalues(**eigenvalues)
    model.primitive_vars(rho, *velocities, pressure)
    model.conservative_from([
        rho,
        *(rho * velocity for velocity in velocities),
        pressure / (gamma - 1.0) + 0.5 * rho * kinetic,
    ])
    return model


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_physical_methods_are_axis_static_and_ranked_from_flux(dimension: int) -> None:
    model = _scalar_model(dimension)
    source, _ = _emit_cpp_brick(model, name="RankedScalar")

    assert "static constexpr int dimension = %d;" % dimension in source
    assert "static constexpr pops::PreparedProviderIdentity provider_identity()" in source
    assert "serialize_exact_parameters(pops::ExactContractBuilder& contract)" in source
    assert '.text("%s")' % model._m._model_hash() in source
    assert "template <int Axis>\n  POPS_HD State flux(" in source
    assert "template <int Axis>\n  POPS_HD pops::Real max_wave_speed(" in source
    assert "template <int Axis>\n  POPS_HD void wave_speeds(" in source
    assert "return flux<Axis>(U, a);" in source
    assert "return max_wave_speed<Axis>(U, a);" in source
    assert source.count("if constexpr (Axis + 1 < dimension)") == 2
    assert "invalid[component] = std::numeric_limits<pops::Real>::quiet_NaN();" in source
    assert "return std::numeric_limits<pops::Real>::quiet_NaN();" in source
    assert "if (axis == 0)" not in source
    assert "switch (axis)" not in source
    assert "if constexpr (Axis == %d)" % (dimension - 1) in source
    assert "if constexpr (Axis == %d)" % dimension not in source
    assert source.count("F[0] =") == dimension


def test_fd_jacobian_calls_the_same_axis_static_flux() -> None:
    model = _scalar_model(3, explicit_waves=False, eigenvalues=False)
    model.wave_speeds_from_jacobian(eig="fd")

    source, _ = _emit_cpp_brick(model, name="RankedFd")
    assert "flux<Axis>(Up_, a)" in source
    assert "flux<Axis>(Um_, a)" in source
    assert "flux(Up_, a, dir)" not in source


def test_runtime_parameter_values_are_part_of_the_emitted_exact_contract() -> None:
    authored = Model("exact_runtime_parameter")
    (state,) = authored.conservative_vars("q")
    speed = authored.value(authored.param(RuntimeParam("speed", default=2.0)))
    authored.flux(x=[speed * state], y=[state])
    authored.eigenvalues(x=[speed + 0 * state], y=[1 + 0 * state])
    authored.primitive_vars(state)
    authored.conservative_from([state])

    source, _ = _emit_cpp_brick(authored, name="ExactRuntimeParameter")
    assert "params.count < 0 || params.count > pops::kMaxRuntimeParams" in source
    assert "contract.scalar(params.values[index]);" in source
    assert '.text("%s")' % authored._m._model_hash() in source


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_role_hllc_and_roe_iterate_ranked_momenta(dimension: int) -> None:
    model = _euler_model(dimension)
    model.enable_hllc()
    model.enable_roe()

    source, _ = _emit_cpp_brick(model, name="RankedEuler")
    assert "template <int Axis>\n  POPS_HD pops::Real contact_speed(" in source
    assert "template <int Axis>\n  POPS_HD State hllc_star_state(" in source
    assert "template <int Axis>\n  POPS_HD State roe_dissipation(" in source
    assert "int dir" not in source
    assert "constexpr int in_ = %d;" % dimension in source
    if dimension == 3:
        assert "const pops::Real q2 = u0 * u0 + u1 * u1 + u2 * u2;" in source
        assert "const pops::Real at1" in source
        assert "const pops::Real at2" in source


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_provided_roe_rows_cover_the_exact_rank(dimension: int) -> None:
    model = _scalar_model(dimension)
    q = Var("q", "cons")
    jump = model.right(q) - model.left(q)
    model.roe_dissipation(**{
        axis: [(ordinal + 1) * jump]
        for ordinal, axis in enumerate(AXES[:dimension])
    })

    source, _ = _emit_cpp_brick(model, name="ProvidedRoe")
    assert "template <int Axis>\n  POPS_HD State roe_dissipation(" in source
    assert "if constexpr (Axis == %d)" % (dimension - 1) in source
    assert "int dir" not in source


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_roe_jacobian_is_derived_on_every_ranked_axis(dimension: int) -> None:
    model = _scalar_model(dimension)
    model.roe_from_jacobian()

    assert tuple(
        key for key in model._m._roe_jacobian if key != "entropy_fix"
    ) == AXES[:dimension]
    source, _ = _emit_cpp_brick(model, name="JacobianRoe")
    assert "template <int Axis>\n  POPS_HD State roe_dissipation(" in source
    assert "if constexpr (Axis == %d)" % (dimension - 1) in source


def test_provided_roe_refuses_a_partial_ranked_map() -> None:
    model = _scalar_model(3)
    q = Var("q", "cons")
    jump = model.right(q) - model.left(q)

    with pytest.raises(ValueError, match="exactly match set_flux"):
        model.roe_dissipation(x=[jump], y=[jump])


def test_axis_values_accepts_immutable_exact_maps_and_refuses_axis_drift() -> None:
    model = SimpleNamespace(_flux={"x": [0], "y": [0]})
    expected = ["flux_x", "flux_y"]

    assert (
        _axis_values(
            model,
            MappingProxyType({"x": ["flux_x"], "y": ["flux_y"]}),
            where="immutable fluxes",
        )
        == expected
    )

    for values in (
        MappingProxyType({"x": ["flux_x"]}),
        MappingProxyType({"y": ["flux_y"], "x": ["flux_x"]}),
        MappingProxyType({"x": ["flux_x"], "y": ["flux_y"], "z": ["flux_z"]}),
    ):
        with pytest.raises(ValueError, match="exact emitted axis set"):
            _axis_values(model, values, where="immutable fluxes")


def test_ranked_hllc_and_roe_templates_are_cpp_well_formed(tmp_path) -> None:
    try:
        compiler, authenticated_compile_flags, link_flags = pops_loader_build_flags()
    except RuntimeError as exc:
        pytest.skip(str(exc))
    include = repo_include()

    for dimension in (1, 2, 3):
        model = _euler_model(dimension)
        model.enable_hllc()
        model.enable_roe()
        source, _ = _emit_cpp_brick(
            model,
            name="Euler%d" % dimension, namespace="rank%d" % dimension
        )
        calls = [
            "  { using Model = rank%d::Euler%d; Model model; Model::State state{};"
            % (dimension, dimension),
            "    using Providers = pops::ProviderValues<pops::provider_count<Model>()>;",
            "    Providers providers{}; pops::Real lower{}, upper{};",
            "    static_assert(pops::HyperbolicModel<Model>);",
            "    pops::ExactContractBuilder brick_contract;",
            "    model.serialize_exact_parameters(brick_contract);",
        ]
        for axis in range(dimension):
            calls.append("    (void)model.template flux<%d>(state, providers);" % axis)
            calls.append("    (void)model.template max_wave_speed<%d>(state, providers);" % axis)
            calls.append("    (void)model.flux(state, providers, %d);" % axis)
            calls.append("    (void)model.max_wave_speed(state, providers, %d);" % axis)
            calls.append("    model.template wave_speeds<%d>(state, providers, lower, upper);" % axis)
            calls.append(
                "    (void)model.template contact_speed<%d>(state, state, 1, 1, -1, 1);"
                % axis
            )
            calls.append(
                "    (void)model.template hllc_star_state<%d>(state, 1, -1, 0);" % axis
            )
            calls.append(
                "    (void)model.template roe_dissipation<%d>(state, providers, state, providers);"
                % axis
            )
        calls += [
            "    const auto invalid_flux = model.flux(state, providers, Model::dimension);",
            "    for (int component = 0; component < Model::State::size(); ++component)",
            "      if (!std::isnan(invalid_flux[component])) return 1;",
            "    if (!std::isnan(model.max_wave_speed(state, providers, Model::dimension))) return 1;",
        ]
        calls.append("  }")
        calls += [
            "  { using ExactModel = pops::CompositeModel<rank%d::Euler%d, "
            "pops::NoSource, pops::BackgroundDensity>;" % (dimension, dimension),
            "    static_assert(requires(const ExactModel& model, "
            "pops::ExactContractBuilder& contract) {",
            "      { ExactModel::provider_identity() } -> "
            "std::same_as<pops::PreparedProviderIdentity>;",
            "      model.serialize_exact_parameters(contract);",
            "    });",
            "    ExactModel model; pops::ExactContractBuilder contract;",
            "    model.serialize_exact_parameters(contract);",
            "    if (contract.view().empty()) return 2;",
            "  }",
        ]
        translation_unit = "\n".join((
            "#include <cmath>",
            "#include <pops/core/model/physical_model.hpp>",
            "#include <pops/physics/bricks/bricks.hpp>",
            source,
            "int main() {",
            *calls,
            "  return 0;",
            "}",
        ))
        source_path = tmp_path / ("ranked_emitters_%dd.cpp" % dimension)
        executable = tmp_path / ("ranked_emitters_%dd" % dimension)
        source_path.write_text(translation_unit, encoding="utf-8")
        compile_flags = [
            flag for flag in authenticated_compile_flags
            if not flag.startswith("-DPOPS_NATIVE_DIM=")
        ]
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O0",
                "-DPOPS_NATIVE_DIM=%d" % dimension,
                "-I",
                include,
                *compile_flags,
                source_path,
                *link_flags,
                "-o",
                executable,
            ],
            check=True,
            env=native_compile_environment(),
        )
        subprocess.run([executable], check=True, env=native_compile_environment())
