"""The final compiled Program honors a generic staircase embedded boundary."""

from __future__ import annotations

import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.boundary import ZeroFlux
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.mesh.geometry import Disc, EmbeddedBoundary, difference
from pops.mesh.masks import Staircase
from pops.numerics import DiscretizationPlan, FiniteVolume, reconstruction, riemann, variables
from pops.time import FixedDt


CELLS = 16
DT = 2.0e-3

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _case_and_grid() -> tuple[pops.Case, CartesianGrid]:
    frame = Rectangle(
        "program-eb-box", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("program-eb-advection", frame=frame)
    state = model.state("U", components=("density",))
    (density,) = state
    flux = model.flux(
        "constant-x-transport",
        frame=frame,
        state=state,
        components={x_axis: (density,), y_axis: (0.0 * density,)},
        waves={x_axis: (1.0,), y_axis: (0.0,)},
    )
    rate = model.rate("transport-rate", equation=ddt(state) == -div(flux))
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )

    case = pops.Case("program-eb-runtime")
    block = case.block("tracer", model=model)
    case.numerics(numerics, block=block)
    program = libtime.ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    grid = CartesianGrid(
        frame=frame,
        cells=(CELLS, CELLS),
        periodic=PeriodicAxes(frame.axes),
    )
    return case, grid


def _run(case: pops.Case, layout: Uniform, initial: np.ndarray) -> np.ndarray:
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    runtime = pops.bind(artifact, initial_state={"tracer": initial})
    report = pops.run(runtime, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    return np.asarray(runtime.state_global("tracer"), dtype=np.float64).reshape(
        1, CELLS, CELLS
    )


def test_program_rhs_uses_csg_staircase_metrics(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, native_cxx, kokkos_root
    case, grid = _case_and_grid()
    annulus = difference(
        Disc(center=(0.5, 0.5), radius=0.42),
        Disc(center=(0.5, 0.5), radius=0.18),
    )
    embedded_layout = Uniform(
        grid,
        embedded_boundary=EmbeddedBoundary(annulus, Staircase(), ZeroFlux()),
    )
    cartesian_layout = Uniform(grid)

    coordinate = (np.arange(CELLS, dtype=np.float64) + 0.5) / CELLS
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    initial = (1.0 + 0.2 * np.sin(2.0 * np.pi * x) + 0.07 * y)[None, :, :]
    baseline = initial.copy()
    cartesian = _run(case, cartesian_layout, initial.copy())
    staircase = _run(case, embedded_layout, initial.copy())

    radius = np.hypot(x - 0.5, y - 0.5)
    active = (radius < 0.42) & (radius >= 0.18)
    inactive = ~active

    # This proves the generated Program's rate/commit path used the metric-aware native residual:
    # inactive cells have exactly zero RHS, active cells evolve, and the result differs from the
    # Cartesian Program. Merely installing or reading a mask cannot satisfy these assertions.
    assert np.array_equal(staircase[0, inactive], baseline[0, inactive])
    assert np.max(np.abs(staircase[0, active] - baseline[0, active])) > 1.0e-6
    assert np.max(np.abs(cartesian[0, inactive] - baseline[0, inactive])) > 1.0e-6
    assert np.max(np.abs(staircase - cartesian)) > 1.0e-6
    np.testing.assert_allclose(
        staircase[0, active].sum(), baseline[0, active].sum(), rtol=0.0, atol=2.0e-13
    )
