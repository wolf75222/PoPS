"""The public pointwise projection hook consumes bound aux data and executes on AMR."""

from __future__ import annotations

import numpy as np
import pytest

import pops
from pops.amr import (
    AMRExecution,
    AMRHierarchy,
    AMRRegrid,
    AMRTagging,
    AMRTransfer,
    Buffer,
    ConflictPolicy,
    EqualityPolicy,
    Hysteresis,
    Tag,
)
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR
from pops.lib.amr import StateTransfer
from pops.lib.initial import Constant
from pops.math import ValueExpr, ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt, every
from tests.python.support.requirements import repo_include


GRID_CELLS = 8
PROJECTION_DT = 0.03125
FLOOR_VALUE = 0.25
INCLUDE = repo_include()


def _projected_model():
    frame = Rectangle("projection_domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = pops.Model("projected_scalar", frame=frame)
    state = model.state("U", components=("q",))
    (q,) = state
    floor = model.aux("floor")
    flux = model.flux(
        "zero_transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * q,), y_axis: (0.0 * q,)},
        waves={x_axis: (0.0 * q,), y_axis: (0.0 * q,)},
    )
    rate = model.rate("zero_rate", equation=ddt(state) == -div(flux))
    model.projection(((q + floor + sqrt((q - floor) * (q - floor))) / 2.0,))
    return frame, model, state, flux, rate


def test_public_projection_oracle_reads_the_declared_aux_field():
    _frame, model, _state, _flux, _rate = _projected_model()
    values = np.array([[[-1.0, 2.0], [3.0, -4.0]]])
    floor = np.array([[0.5, 0.25], [4.0, -2.0]])

    projected = model.projection_value(values, aux={"floor": floor})

    np.testing.assert_array_equal(projected, np.maximum(values, floor[None, ...]))


def _projection_case() -> tuple[pops.Case, AMR]:
    frame, model, state, flux, rate = _projected_model()
    case = pops.Case("projection_case")
    block = case.block("scalar", model)
    state_instance = block[state]
    # The bound state starts below the floor.  Tagging against -2 forces a real fine level before
    # the projection, so the proof covers aux publication and projection on both AMR levels.
    threshold = case.param(RuntimeParam("refine_threshold", default=-2.0))

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
    case.numerics(numerics, block=block)

    program = pops.Program("project_after_step")
    temporal = program.state(state_instance)
    candidate = program.value(
        "candidate",
        temporal.n + program.dt * rate(temporal.n),
        at=temporal.next.point,
    )
    program.commit(temporal.next, program.project(candidate))
    program.step_strategy(FixedDt(PROJECTION_DT))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=state_instance,
            value=Constant((-1.0,)),
            projection=ConservativeCellAverage(),
        )
    )

    transfer = AMRTransfer()
    transfer.state(state_instance, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(GRID_CELLS, GRID_CELLS),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(state_instance) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(100, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )

    return case, layout


def _resolve_projection_case(*, cxx: str | None = None):
    case, layout = _projection_case()
    compile_options = None if cxx is None else {"include": INCLUDE, "cxx": cxx}
    return pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options=compile_options,
    )


def test_public_projection_program_resolves_on_amr():
    resolved = _resolve_projection_case()

    assert resolved.target == "amr_system"
    amr_program = resolved.capabilities["resolution"]["amr_program"]
    assert amr_program["status"] == "proven"
    assert [dict(group) for group in amr_program["groups"]] == [
        {"name": "projection", "status": "green"}
    ]


@pytest.mark.compiler
@pytest.mark.native_loader
def test_bound_aux_drives_public_projection_in_a_native_amr_step(
    isolated_native_cache, native_cxx, kokkos_root
) -> None:
    del isolated_native_cache, kokkos_root
    artifact = pops.compile(_resolve_projection_case(cxx=native_cxx))
    floor = np.full((GRID_CELLS, GRID_CELLS), FLOOR_VALUE, dtype=np.float64)
    simulation = pops.bind(artifact, aux={"floor": floor})
    level_count = simulation.n_levels()
    assert level_count == 2
    before = tuple(
        np.asarray(simulation.block_level_state_global("scalar", level), dtype=np.float64).copy()
        for level in range(level_count)
    )
    assert all(np.all(level == -1.0) for level in before)

    report = pops.run(simulation, t_end=PROJECTION_DT, max_steps=1)

    assert report.accepted_steps == 1
    assert report.final_time == PROJECTION_DT
    assert simulation.n_levels() == level_count
    for level in range(level_count):
        published = np.asarray(
            simulation.block_level_state_global("scalar", level), dtype=np.float64
        )
        np.testing.assert_array_equal(published, np.full_like(published, FLOOR_VALUE))
