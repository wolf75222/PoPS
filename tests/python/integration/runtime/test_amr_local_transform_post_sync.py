"""AMR local_transform runs after reflux through Program.after_synchronization."""

from __future__ import annotations

import numpy as np
import pops
import pytest

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
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt, every
from tests.python.support.requirements import repo_include


CELLS = 8
DT = 0.125
INCLUDE = repo_include()

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def _case_and_layout():
    frame = Rectangle(
        "amr_post_sync_square", lower=(0.0, 0.0), upper=(1.0, 1.0)
    ).frame(Cartesian2D())
    x_axis, y_axis = frame.axes

    model = pops.Model("amr_post_sync_transform", frame=frame)
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

    case = pops.Case("amr_post_sync_transform_case")
    block = case.block("field", model=model)
    block_state = block[state]
    case.numerics(numerics, block=block)

    program = pops.Program("amr-post-sync-transform")
    temporal = program.state(block_state)
    rhs = rate(temporal.n)
    candidate = program.value(
        "candidate", temporal.n + program.dt * rhs, at=temporal.next.point)
    program.commit(temporal.next, candidate)

    def _relax(program_body):
        synced = program_body.value(
            "synced", 1.0 * temporal.n, at=temporal.next.point)
        relaxed = program_body.transform(
            synced, transform=shift, name="shifted_candidate")
        program_body.commit(temporal.next, relaxed)

    program.after_synchronization(_relax)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    case.initials.add(InitialCondition(
        state=block_state,
        value=Constant((2.0,)),
        projection=ConservativeCellAverage(),
    ))
    refine_threshold = case.param(RuntimeParam("post_sync_refine", default=1.0))
    transfer = AMRTransfer()
    transfer.state(block_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(CELLS, CELLS),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(block_state) > case.value(refine_threshold)),
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


def test_amr_after_synchronization_applies_transform_on_a_refined_hierarchy(
    isolated_native_cache, native_cxx, kokkos_root,
) -> None:
    del isolated_native_cache, kokkos_root
    case, layout = _case_and_layout()
    artifact = pops.compile(
        pops.resolve(
            pops.validate(case),
            layout=layout,
            backend=Production(),
            compile_options={"include": INCLUDE, "cxx": native_cxx},
        )
    )
    simulation = pops.bind(artifact)
    assert simulation.n_levels() == 2
    report = pops.run(simulation, t_end=DT, max_steps=1)
    assert report.accepted_steps == 1
    actual = np.asarray(
        simulation.block_level_state_global("field", 0), dtype=np.float64
    ).reshape((1, CELLS, CELLS))
    np.testing.assert_array_equal(actual, np.full((1, CELLS, CELLS), 3.0, dtype=np.float64))
