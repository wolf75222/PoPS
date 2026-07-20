"""Native end-to-end execution of a generic symbolic local transform."""

from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.boundary import ZeroFlux
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.mesh.geometry import Disc, EmbeddedBoundary
from pops.mesh.masks import CutCell, Staircase
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt


CELLS = 4
DT = 0.125

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _case_and_layout(transport_mask=None):
    frame = Rectangle(
        "local_transform_square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes

    model = pops.Model("native_local_transform", frame=frame)
    state = model.state(
        "U",
        components=("q",),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    q = state[0]
    flux = model.flux(
        "zero_flux",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * q,), y_axis: (0.0 * q,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    rate = model.rate("zero_rate", equation=ddt(state) == -div(flux))
    shift = model.local_transform("positive_shift", (q + 1.0,), valid_if=q > 0.0)

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

    case = pops.Case("native_local_transform_case")
    block = case.block("field", model=model)
    case.numerics(numerics, block=block)

    program = pops.Program("native-local-transform")
    temporal = program.state(block[state])
    rhs = rate(temporal.n)
    candidate = program.value(
        "candidate", temporal.n + program.dt * rhs, at=temporal.next.point)
    transformed = program.transform(
        candidate, transform=shift, name="shifted_candidate")
    program.commit(temporal.next, transformed)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    grid = CartesianGrid(
        frame=frame,
        cells=(CELLS, CELLS),
        periodic=PeriodicAxes(frame.axes),
    )
    layout = Uniform(grid)
    if transport_mask is not None:
        layout = Uniform(
            grid,
            embedded_boundary=EmbeddedBoundary(
                Disc(center=(0.5, 0.5), radius=0.36),
                transport_mask,
                ZeroFlux(),
            ),
        )
    return case, layout


def test_local_transform_executes_natively_and_rejects_before_publication(
    isolated_native_cache, native_cxx, kokkos_root,
):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout = _case_and_layout()
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))

    initial = np.full((1, CELLS, CELLS), 2.0, dtype=np.float64)
    accepted = pops.bind(artifact, initial_state={"field": initial})
    report = pops.run(accepted, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    actual = np.asarray(accepted.state_global("field"), dtype=np.float64).reshape(initial.shape)
    np.testing.assert_array_equal(actual, initial + 1.0)

    invalid = np.full((1, CELLS, CELLS), -1.0, dtype=np.float64)
    rejected = pops.bind(artifact, initial_state={"field": invalid})
    with pytest.raises(RuntimeError, match="local_transform|positive_shift"):
        pops.run(rejected, t_end=DT, max_steps=1)
    unchanged = np.asarray(rejected.state_global("field"), dtype=np.float64).reshape(invalid.shape)
    np.testing.assert_array_equal(unchanged, invalid)


@pytest.mark.parametrize(
    "transport_mask", [Staircase(), CutCell()], ids=["staircase", "cutcell"])
def test_local_transform_preserves_and_does_not_validate_inactive_cells(
    isolated_native_cache, native_cxx, kokkos_root, transport_mask,
):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout = _case_and_layout(transport_mask)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))

    coordinate = (np.arange(CELLS, dtype=np.float64) + 0.5) / CELLS
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    active = np.hypot(x - 0.5, y - 0.5) < 0.36
    initial = np.where(active[None, :, :], 2.0, -2.0)
    runtime = pops.bind(artifact, initial_state={"field": initial})

    report = pops.run(runtime, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    actual = np.asarray(runtime.state_global("field"), dtype=np.float64).reshape(initial.shape)
    np.testing.assert_array_equal(actual[0, active], initial[0, active] + 1.0)
    np.testing.assert_array_equal(actual[0, ~active], initial[0, ~active])
