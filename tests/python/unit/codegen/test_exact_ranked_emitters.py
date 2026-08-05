"""Exact-ranked C++ emission from the authenticated x[/y[/z]] physics map."""
from __future__ import annotations

import shutil
import subprocess

import pytest

from pops._ir import Var
from pops.math import sqrt
from pops.physics._model import HyperbolicModel
from tests.python.support.requirements import repo_include


AXES = ("x", "y", "z")


def _scalar_model(dimension: int, *, explicit_waves: bool = True) -> HyperbolicModel:
    model = HyperbolicModel("ranked_scalar_%d" % dimension)
    (q,) = model.conservative_vars("q")
    axes = AXES[:dimension]
    model.set_flux(**{
        axis: [(ordinal + 1) * q] for ordinal, axis in enumerate(axes)
    })
    if explicit_waves:
        model.set_wave_speeds(**{
            axis: (-(ordinal + 1) + 0 * q, ordinal + 1 + 0 * q)
            for ordinal, axis in enumerate(axes)
        })
    else:
        model.set_eigenvalues(**{
            axis: [ordinal + 1 + 0 * q] for ordinal, axis in enumerate(axes)
        })
    model.set_primitive_state(q)
    model.set_conservative_from([q])
    return model


def _euler_model(dimension: int) -> HyperbolicModel:
    gamma = 1.4
    model = HyperbolicModel("ranked_euler_%d" % dimension)
    names = ("rho", "rho_u", "rho_v", "rho_w", "E")
    roles = ("Density", "MomentumX", "MomentumY", "MomentumZ", "Energy")
    selected_names = (names[0], *names[1:dimension + 1], names[-1])
    selected_roles = (roles[0], *roles[1:dimension + 1], roles[-1])
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
    model.set_flux(**fluxes)
    model.set_eigenvalues(**eigenvalues)
    model.set_primitive_state(rho, *velocities, pressure)
    model.set_conservative_from([
        rho,
        *(rho * velocity for velocity in velocities),
        pressure / (gamma - 1.0) + 0.5 * rho * kinetic,
    ])
    return model


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_physical_methods_are_axis_static_and_ranked_from_flux(dimension: int) -> None:
    source = _scalar_model(dimension).emit_cpp_brick(name="RankedScalar")

    assert "static constexpr int dimension = %d;" % dimension in source
    assert "template <int Axis>\n  POPS_HD State flux(" in source
    assert "template <int Axis>\n  POPS_HD pops::Real max_wave_speed(" in source
    assert "template <int Axis>\n  POPS_HD void wave_speeds(" in source
    assert "int dir" not in source
    assert "if constexpr (Axis == %d)" % (dimension - 1) in source
    assert "if constexpr (Axis == %d)" % dimension not in source
    assert source.count("F[0] =") == dimension


def test_fd_jacobian_calls_the_same_axis_static_flux() -> None:
    model = _scalar_model(3, explicit_waves=False)
    model._eig = {}
    model.set_wave_speeds_from_jacobian(eig="fd")

    source = model.emit_cpp_brick(name="RankedFd")
    assert "flux<Axis>(Up_, a)" in source
    assert "flux<Axis>(Um_, a)" in source
    assert "flux(Up_, a, dir)" not in source


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_role_hllc_and_roe_iterate_ranked_momenta(dimension: int) -> None:
    model = _euler_model(dimension)
    model.enable_hllc()
    model.enable_roe()

    source = model.emit_cpp_brick(name="RankedEuler")
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

    source = model.emit_cpp_brick(name="ProvidedRoe")
    assert "template <int Axis>\n  POPS_HD State roe_dissipation(" in source
    assert "if constexpr (Axis == %d)" % (dimension - 1) in source
    assert "int dir" not in source


@pytest.mark.parametrize("dimension", (1, 2, 3))
def test_roe_jacobian_is_derived_on_every_ranked_axis(dimension: int) -> None:
    model = _scalar_model(dimension)
    model.roe_from_jacobian()

    assert tuple(
        key for key in model._roe_jacobian if key != "entropy_fix"
    ) == AXES[:dimension]
    source = model.emit_cpp_brick(name="JacobianRoe")
    assert "template <int Axis>\n  POPS_HD State roe_dissipation(" in source
    assert "if constexpr (Axis == %d)" % (dimension - 1) in source


def test_provided_roe_refuses_a_partial_ranked_map() -> None:
    model = _scalar_model(3)
    q = Var("q", "cons")
    jump = model.right(q) - model.left(q)

    with pytest.raises(ValueError, match="exactly match set_flux"):
        model.roe_dissipation(x=[jump], y=[jump])


def test_ranked_hllc_and_roe_templates_are_cpp_well_formed(tmp_path) -> None:
    compiler = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    include = repo_include()
    if compiler is None:
        pytest.skip("a C++20 compiler is required for exact-ranked emitter syntax")

    for dimension in (1, 2, 3):
        model = _euler_model(dimension)
        model.enable_hllc()
        model.enable_roe()
        source = model.emit_cpp_brick(
            name="Euler%d" % dimension, namespace="rank%d" % dimension
        )
        calls = [
            "  { rank%d::Euler%d model; rank%d::Euler%d::State state{};"
            % (dimension, dimension, dimension, dimension),
            "    pops::Aux providers{}; pops::Real lower{}, upper{};",
        ]
        for axis in range(dimension):
            calls.append("    (void)model.template flux<%d>(state, providers);" % axis)
            calls.append("    (void)model.template max_wave_speed<%d>(state, providers);" % axis)
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
        calls.append("  }")
        translation_unit = "\n".join((
            "#include <pops/core/state/variables.hpp>",
            source,
            "int main() {",
            *calls,
            "  return 0;",
            "}",
        ))
        source_path = tmp_path / ("ranked_emitters_%dd.cpp" % dimension)
        executable = tmp_path / ("ranked_emitters_%dd" % dimension)
        source_path.write_text(translation_unit, encoding="utf-8")
        subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-O0",
                "-DPOPS_NATIVE_DIM=%d" % dimension,
                "-I",
                include,
                source_path,
                "-o",
                executable,
            ],
            check=True,
        )
