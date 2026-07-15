"""Numerical regressions for the final board-authored Roe capability.

The test deliberately crosses the authenticated compiler-provider boundary instead of importing
the former PDE facade.  Both models are ordinary :class:`pops.physics.Model` values and execute in
the native ``System`` route with the generic ``Roe`` descriptor.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops.runtime._engine_descriptors as engine
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import ddt, div, sqrt
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Roe
from pops.physics import Density, Energy, Model, Momentum
from pops.runtime._system import System
from tests.python.support.requirements import (
    default_cxx,
    missing_native_compile_requirement,
    repo_include,
)


INCLUDE = repo_include()
GAMMA = 1.4

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _frame(name: str):
    return Rectangle(name, (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())


def _compile_native(model: Model, path: Path):
    """Compile through the final explicit provider, never through ``Model.compile``."""
    lowering = model.__pops_compiler_lowering__()
    assert lowering.facade is model
    assert lowering.source_module is model.module
    return lowering.emit_model.compile(str(path), INCLUDE, backend="production")


def _require_native_toolchain() -> None:
    reason = missing_native_compile_requirement(INCLUDE, default_cxx())
    if reason:
        pytest.skip(reason)


def _euler(name: str) -> Model:
    frame = _frame(name)
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state(
        "U",
        components=("rho", "rho_u", "rho_v", "E"),
        roles={
            "rho": Density(),
            "rho_u": Momentum(axis=x_axis),
            "rho_v": Momentum(axis=y_axis),
            "E": Energy(),
        },
    )
    rho, rho_u, rho_v, energy = state
    u = model.primitive("u", rho_u / rho)
    v = model.primitive("v", rho_v / rho)
    pressure = model.scalar(
        "p", (GAMMA - 1.0) * (energy - 0.5 * rho * (u * u + v * v)))
    enthalpy = model.scalar("H", (energy + pressure) / rho)
    sound_speed = model.scalar("c", sqrt(GAMMA * pressure / rho))
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (rho_u, rho_u * u + pressure, rho_u * v, rho * enthalpy * u),
            y_axis: (rho_v, rho_v * u, rho_v * v + pressure, rho * enthalpy * v),
        },
        # Four entries retain the repeated entropy/shear eigenvalue explicitly.
        waves={
            x_axis: (u - sound_speed, u, u, u + sound_speed),
            y_axis: (v - sound_speed, v, v, v + sound_speed),
        },
    )
    model.riemann(Roe(), flux=flux, pressure=pressure)
    model.rate("transport", equation=ddt(state) == -div(flux))
    return model


def _isothermal(name: str) -> Model:
    frame = _frame(name)
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state(
        "U",
        components=("rho", "rho_u", "rho_v"),
        roles={
            "rho": Density(),
            "rho_u": Momentum(axis=x_axis),
            "rho_v": Momentum(axis=y_axis),
        },
    )
    rho, rho_u, rho_v = state
    cs2 = 0.5
    u = model.primitive("u", rho_u / rho)
    v = model.primitive("v", rho_v / rho)
    pressure = model.scalar("p", cs2 * rho)
    sound_speed = sqrt(cs2)
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (rho_u, rho_u * u + pressure, rho_u * v),
            y_axis: (rho_v, rho_v * u, rho_v * v + pressure),
        },
        waves={
            x_axis: (u - sound_speed, u, u + sound_speed),
            y_axis: (v - sound_speed, v, v + sound_speed),
        },
    )
    model.riemann(Roe(), flux=flux, pressure=pressure)
    model.rate("transport", equation=ddt(state) == -div(flux))
    return model


def _smooth_density(n: int, amplitude: float = 0.4) -> np.ndarray:
    points = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(points, points, indexing="xy")
    return 1.0 + amplitude * np.exp(-60.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def test_board_roe_runs_euler_and_preserves_stationary_shear(tmp_path: Path) -> None:
    _require_native_toolchain()

    n = 24
    roe_spatial = engine.Spatial(limiter=FirstOrder(), flux=Roe())

    euler = _compile_native(_euler("final_euler_roe"), tmp_path / "euler_roe.so")
    assert euler.has_roe
    assert not euler.has_hllc  # compiling Roe-only must not require the unrelated HLLC capability
    rho = _smooth_density(n)
    u = np.full_like(rho, 0.1)
    v = np.zeros_like(rho)
    pressure = np.ones_like(rho)
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    initial = np.stack((rho, rho * u, rho * v, energy))

    gas = System(n=n, L=1.0, periodic=True)
    gas.add_equation(
        "gas", model=euler, spatial=roe_spatial, time=engine.Explicit())
    gas.set_state("gas", initial)
    for _ in range(8):
        gas.step(2.0e-4)
    final = np.asarray(gas.get_state("gas"))
    assert np.isfinite(final).all()
    np.testing.assert_allclose(final[[0, 3]].sum(axis=(1, 2)),
                               initial[[0, 3]].sum(axis=(1, 2)), rtol=1.0e-12, atol=1.0e-12)

    isothermal = _compile_native(
        _isothermal("final_isothermal_roe"), tmp_path / "isothermal_roe.so")
    assert isothermal.has_roe
    assert not isothermal.has_hllc
    points = (np.arange(n) + 0.5) / n
    transverse_velocity = np.tile(0.3 * np.sin(2.0 * np.pi * points), (n, 1))
    shear = np.stack((np.ones((n, n)), np.zeros((n, n)), transverse_velocity))
    fluid = System(n=n, L=1.0, periodic=True)
    fluid.add_equation(
        "fluid", model=isothermal, spatial=roe_spatial, time=engine.Explicit())
    fluid.set_state("fluid", shear)
    before = np.asarray(fluid.get_state("fluid")).copy()
    for _ in range(6):
        fluid.step_cfl(0.3)
    after = np.asarray(fluid.get_state("fluid"))
    # The contact/shear wave has zero normal speed, so Roe adds exactly no dissipation.
    np.testing.assert_array_equal(after, before)
