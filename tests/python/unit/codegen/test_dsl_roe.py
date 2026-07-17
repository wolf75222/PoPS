"""Numerical regressions for board-authored Roe through the final public lifecycle."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Density, Energy, Model, Momentum
from pops.time import FixedDt


ROOT = Path(__file__).resolve().parents[4]
GAMMA = 1.4

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _frame(name: str):
    return Rectangle(name, (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())


def _compile_public(
    model: Model,
    *,
    case_name: str,
    block_name: str,
    n: int,
    dt: float,
    cxx: str,
):
    """Compile one Roe case through ``validate -> resolve -> compile`` only."""
    state = model.states["U"]
    flux = model.fluxes["transport"]
    rate = model.operators["transport"]
    case = pops.Case(case_name)
    block = case.block(block_name, model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Roe(),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(dt))
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=model.frame,
            cells=(n, n),
            periodic=PeriodicAxes(model.frame.axes),
        )
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    assert len(artifact.blocks) == 1
    return artifact


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
    model.riemann(riemann.Roe(), flux=flux, pressure=pressure)
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
    model.riemann(riemann.Roe(), flux=flux, pressure=pressure)
    model.rate("transport", equation=ddt(state) == -div(flux))
    return model


def _smooth_density(n: int, amplitude: float = 0.4) -> np.ndarray:
    points = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(points, points, indexing="xy")
    return 1.0 + amplitude * np.exp(-60.0 * ((x - 0.5) ** 2 + (y - 0.5) ** 2))


def test_board_roe_runs_euler_and_preserves_stationary_shear(
    isolated_native_cache: Path,
    native_cxx: str,
    kokkos_root: Path,
) -> None:
    del isolated_native_cache, kokkos_root
    n = 24
    euler_dt = 2.0e-4

    euler = _compile_public(
        _euler("final_euler_roe"),
        case_name="final_euler_roe_case",
        block_name="gas",
        n=n,
        dt=euler_dt,
        cxx=native_cxx,
    )
    assert euler.blocks[0].model.has_roe
    # Compiling Roe-only must not require the unrelated HLLC capability.
    assert not euler.blocks[0].model.has_hllc
    rho = _smooth_density(n)
    u = np.full_like(rho, 0.1)
    v = np.zeros_like(rho)
    pressure = np.ones_like(rho)
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
    initial = np.ascontiguousarray(np.stack((rho, rho * u, rho * v, energy)))

    gas = pops.bind(euler, initial_state={"gas": initial})
    report = pops.run(gas, t_end=8 * euler_dt, max_steps=8)
    assert report.accepted_steps == 8
    final = np.asarray(gas.get_state("gas"), dtype=np.float64).reshape(initial.shape)
    assert np.isfinite(final).all()
    np.testing.assert_allclose(
        final[[0, 3]].sum(axis=(1, 2)),
        initial[[0, 3]].sum(axis=(1, 2)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    shear_dt = 1.0e-3
    isothermal = _compile_public(
        _isothermal("final_isothermal_roe"),
        case_name="final_isothermal_roe_case",
        block_name="fluid",
        n=n,
        dt=shear_dt,
        cxx=native_cxx,
    )
    assert isothermal.blocks[0].model.has_roe
    assert not isothermal.blocks[0].model.has_hllc
    points = (np.arange(n) + 0.5) / n
    transverse_velocity = np.tile(0.3 * np.sin(2.0 * np.pi * points), (n, 1))
    shear = np.ascontiguousarray(
        np.stack((np.ones((n, n)), np.zeros((n, n)), transverse_velocity))
    )
    fluid = pops.bind(isothermal, initial_state={"fluid": shear})
    before = np.asarray(fluid.get_state("fluid"), dtype=np.float64).reshape(shear.shape).copy()
    report = pops.run(fluid, t_end=6 * shear_dt, max_steps=6)
    assert report.accepted_steps == 6
    after = np.asarray(fluid.get_state("fluid"), dtype=np.float64).reshape(shear.shape)
    # The contact/shear wave has zero normal speed, so Roe adds exactly no dissipation.
    np.testing.assert_array_equal(after, before)
