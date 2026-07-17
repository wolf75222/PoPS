"""Explicit signed wave speeds drive native HLL through the final public lifecycle."""
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
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.riemann import ExplicitPair, provider_of
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model
from pops.time import FixedDt


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 1.0e-4
AX = 1.0
AY = 0.6
SPEED_X = (-1.25, 1.5)
SPEED_Y = (-0.9, 0.8)

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _model() -> tuple[Model, object, object, object]:
    frame = Rectangle(
        "explicit-hll-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = Model("explicit_hll", frame=frame)
    state = model.state("U", components=("q1", "q2"))
    q1, q2 = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={
            x_axis: (AX * q2, AX * q1),
            y_axis: (AY * q2, AY * q1),
        },
    )
    # Deliberately asymmetric signed bounds: HLL cannot collapse algebraically to Rusanov.
    model.wave_speeds(
        flux,
        frame=frame,
        values={x_axis: SPEED_X, y_axis: SPEED_Y},
    )
    rate = model.rate("transport", equation=ddt(state) == -div(flux))
    return model, state, flux, rate


def _initial_state() -> np.ndarray:
    points = (np.arange(N) + 0.5) / N
    x, y = np.meshgrid(points, points, indexing="xy")
    return np.stack(
        (
            1.0 + 0.25 * np.sin(2.0 * np.pi * x) * np.cos(4.0 * np.pi * y),
            0.2 * np.cos(2.0 * np.pi * x) + 0.1 * np.sin(2.0 * np.pi * y),
        )
    )


def _physical_flux(state: np.ndarray, direction: int) -> np.ndarray:
    speed = AX if direction == 0 else AY
    return np.stack((speed * state[1], speed * state[0]))


def _hll_rhs(state: np.ndarray) -> np.ndarray:
    rhs = np.zeros_like(state)
    for direction, array_axis, (speed_left, speed_right) in (
        (0, 2, SPEED_X),
        (1, 1, SPEED_Y),
    ):
        left = state
        right = np.roll(state, -1, axis=array_axis)
        flux_left = _physical_flux(left, direction)
        flux_right = _physical_flux(right, direction)
        interface_flux = (
            speed_right * flux_left
            - speed_left * flux_right
            + speed_left * speed_right * (right - left)
        ) / (speed_right - speed_left)
        rhs -= N * (interface_flux - np.roll(interface_flux, 1, axis=array_axis))
    return rhs


def _rusanov_rhs(state: np.ndarray) -> np.ndarray:
    rhs = np.zeros_like(state)
    for direction, array_axis, pair in (
        (0, 2, SPEED_X),
        (1, 1, SPEED_Y),
    ):
        left = state
        right = np.roll(state, -1, axis=array_axis)
        flux_left = _physical_flux(left, direction)
        flux_right = _physical_flux(right, direction)
        radius = max(abs(pair[0]), abs(pair[1]))
        interface_flux = 0.5 * (flux_left + flux_right) - 0.5 * radius * (right - left)
        rhs -= N * (interface_flux - np.roll(interface_flux, 1, axis=array_axis))
    return rhs


def test_final_case_consumes_explicit_signed_pair_in_hll_without_pressure(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, kokkos_root
    model, state, flux, rate = _model()
    provider = provider_of(model)
    assert provider is not None
    assert provider.kind == ExplicitPair().kind
    hll = riemann.HLL(waves=ExplicitPair())
    assert hll.options["waves"] == "explicit_pair"

    case = pops.Case("explicit_hll_case")
    block = case.block("transport", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=hll,
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=model.frame,
            cells=(N, N),
            periodic=PeriodicAxes(model.frame.axes),
        )
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    assert artifact.blocks[0].model.has_wave_speeds

    initial = np.ascontiguousarray(_initial_state())
    expected_rhs = _hll_rhs(initial)
    assert float(np.max(np.abs(expected_rhs - _rusanov_rhs(initial)))) > 1.0e-3
    simulation = pops.bind(artifact, initial_state={"transport": initial})
    report = pops.run(simulation, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    final = np.asarray(simulation.get_state("transport"), dtype=np.float64).reshape(
        initial.shape
    )
    expected = initial + DT * expected_rhs
    np.testing.assert_allclose(final, expected, rtol=0.0, atol=2.0e-12)

    # A Rusanov fallback would give a different update for these asymmetric signed bounds.
    assert np.max(np.abs(final - initial)) > 1.0e-6
