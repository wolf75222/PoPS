"""Generated Strang splitting on a genuinely refined AMR hierarchy.

ADC-700 requires time composition to remain an ordinary compiled ``Program`` when
AMR is active.  This test exercises the public resolve/compile/bind/run lifecycle
with a non-commuting Strang composition, a live two-level hierarchy, and no
native/legacy temporal facade.

The two local shears are nilpotent, so each authored Euler subflow is also its
exact exponential.  Their ordering therefore has a closed-form matrix oracle.
Running two accepted macro steps additionally proves that stage scratch from the
first step cannot leak into the next accepted state.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pops
import pops.lib.time as libtime
import pytest
from pops.amr import (
    AMRClockRelation,
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
from pops.lib.initial import BindArray, Gaussian
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.representations import Conservative
from pops.spaces import CellState
from pops.time import FixedDt, every


ROOT = Path(__file__).resolve().parents[4]
N = 16
DT = 2.0e-2
NSTEPS = 2
FIRST_RATE = 1.0
SECOND_RATE = -2.0

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _inert_scalar_model(name, frame):
    x_axis, y_axis = frame.axes
    model = Model(name, frame=frame)
    state = model.state("U", components=("marker",))
    (marker,) = state
    flux = model.flux(
        "inert_marker_flux",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * marker,), y_axis: (0.0 * marker,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    rate = model.rate("inert_marker_rate", equation=ddt(state) == -div(flux))
    return model, state, flux, rate


def _split_model(frame):
    x_axis, y_axis = frame.axes
    model = Model("amr-refined-strang-shears", frame=frame)
    state = model.state(
        "U",
        components=("first_amplitude", "second_amplitude"),
        representation=Conservative(),
        space=CellState(frame=frame),
    )
    first_amplitude, second_amplitude = state
    zero_flux = (0.0 * first_amplitude, 0.0 * second_amplitude)
    flux = model.flux(
        "inert_split_flux",
        frame=frame,
        state=state,
        components={x_axis: zero_flux, y_axis: zero_flux},
        waves={x_axis: (0.0, 0.0), y_axis: (0.0, 0.0)},
    )
    rate = model.rate("inert_split_rate", equation=ddt(state) == -div(flux))
    first = model.operator(
        "upper_shear",
        returns=model.local_linear_operator(
            "upper_shear",
            on=state,
            matrix=((0.0, FIRST_RATE), (0.0, 0.0)),
        ),
    )
    second = model.operator(
        "lower_shear",
        returns=model.local_linear_operator(
            "lower_shear",
            on=state,
            matrix=((0.0, 0.0), (SECOND_RATE, 0.0)),
        ),
    )
    return model, state, flux, rate, first, second


def _numerics(state, flux, rate):
    plan = DiscretizationPlan()
    plan.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    return plan


def _subflow(operator, label):
    def build(program, current, fraction, *, at):
        linear = program.value(
            "%s_map" % label,
            operator(program=program),
            at=current.point,
        )
        change = program.apply(linear, current)
        return program.value(
            "%s_flow" % label,
            current + (fraction * program.dt) * change,
            at=at,
        )

    return build


def _resolved(native_cxx):
    frame = Rectangle("amr-refined-strang-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    split_model, split_state, split_flux, split_rate, first, second = _split_model(frame)
    marker_model, marker_declaration, marker_flux, marker_rate = _inert_scalar_model(
        "amr-refined-strang-marker", frame
    )

    case = pops.Case("amr-refined-strang-case")
    oscillator = case.block("oscillator", split_model, states=(split_state,))
    marker = case.block("marker", marker_model, states=(marker_declaration,))
    oscillator_state = oscillator[split_state]
    marker_state = marker[marker_declaration]
    case.numerics(_numerics(split_state, split_flux, split_rate), block=oscillator)
    case.numerics(
        _numerics(marker_declaration, marker_flux, marker_rate),
        block=marker,
    )

    program = libtime.Strang(
        oscillator_state,
        first=_subflow(first, "upper"),
        second=_subflow(second, "lower"),
    )
    marker_temporal = program.state(marker_state)
    marker_next = program.value(
        "held_marker",
        marker_temporal.n,
        at=marker_temporal.next.point,
    )
    program.commit(marker_temporal.next, marker_next)
    program.step_strategy(FixedDt(DT))
    case.program(program)

    case.initials.add(
        InitialCondition(
            state=oscillator_state,
            value=BindArray(),
            projection=ConservativeCellAverage(),
        )
    )
    x_axis, y_axis = frame.axes
    case.initials.add(
        InitialCondition(
            state=marker_state,
            value=Gaussian(
                frame=frame,
                center={x_axis: 0.5, y_axis: 0.5},
                background=0.0,
                amplitude=1.0,
                inverse_width=100.0,
            ),
            projection=ConservativeCellAverage(),
        )
    )

    threshold = case.param(RuntimeParam("refine_threshold", default=0.2))
    transfer = AMRTransfer()
    transfer.state(oscillator_state, StateTransfer())
    transfer.state(marker_state, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(marker_state) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.subcycled((AMRClockRelation(0, 1, 2),)),
    )
    return (
        pops.resolve(
            pops.validate(case),
            layout=layout,
            backend=Production(),
            compile_options={"include": str(ROOT / "include"), "cxx": native_cxx},
        ),
        oscillator_state,
    )


def _initial_state():
    return np.broadcast_to(
        np.array((1.0, 0.3), dtype=np.float64)[:, None, None],
        (2, N, N),
    ).copy()


def _strang_matrix(dt):
    identity = np.eye(2, dtype=np.float64)
    first = np.array(((0.0, FIRST_RATE), (0.0, 0.0)), dtype=np.float64)
    second = np.array(((0.0, 0.0), (SECOND_RATE, 0.0)), dtype=np.float64)
    first_half = identity + 0.5 * dt * first
    return first_half @ (identity + dt * second) @ first_half


def _fine_valid_mask(simulation):
    side = 2 * N
    valid = np.zeros((side, side), dtype=bool)
    fine = next(row for row in simulation.amr.patch_table().per_level if int(row["level"]) == 1)
    for ilo, jlo, ihi, jhi in fine["boxes"]:
        valid[jlo : jhi + 1, ilo : ihi + 1] = True
    assert np.any(valid)
    return valid


def test_generated_strang_runs_only_through_program_on_refined_amr(
    native_cxx,
    isolated_native_cache,
    kokkos_root,
):
    del isolated_native_cache, kokkos_root
    plan, oscillator_state = _resolved(native_cxx)
    artifact = pops.compile(plan)
    initial = _initial_state()

    def run_once():
        simulation = pops.bind(
            artifact,
            initial_values={oscillator_state: initial},
        )
        assert simulation.n_levels() == 2
        assert simulation.patch_boxes()
        assert simulation.installed_program_hash()
        report = pops.run(
            simulation,
            t_end=NSTEPS * DT,
            max_steps=NSTEPS,
            console=False,
        )
        return simulation, report

    simulation, report = run_once()
    evolved = np.asarray(
        simulation.block_level_state_global("oscillator", 0),
        dtype=np.float64,
    ).reshape(initial.shape)
    fine_valid = _fine_valid_mask(simulation)
    coarse_covered = fine_valid.reshape(N, 2, N, 2).any(axis=(1, 3))
    assert np.any(coarse_covered)
    assert np.any(~coarse_covered)

    # The every(1) regrid materializes the fine patches after the first accepted coarse step.
    # Uncovered coarse cells therefore take two full Strang steps. Fine cells inherit the state
    # after the first full step, then take the two ratio-2 substeps of the second interval.
    coarse_expected = np.linalg.matrix_power(_strang_matrix(DT), NSTEPS) @ initial[:, 0, 0]
    fine_expected = (
        np.linalg.matrix_power(_strang_matrix(0.5 * DT), 2 * (NSTEPS - 1))
        @ _strang_matrix(DT)
        @ initial[:, 0, 0]
    )

    assert report.accepted_steps == simulation.macro_step() == NSTEPS
    assert simulation.n_levels() == 2
    assert simulation.patch_boxes()
    assert np.isfinite(evolved).all()
    coarse_actual = evolved[:, ~coarse_covered]
    np.testing.assert_allclose(
        coarse_actual,
        np.broadcast_to(coarse_expected[:, None], coarse_actual.shape),
        rtol=0.0,
        atol=2.0e-13,
    )
    covered_actual = evolved[:, coarse_covered]
    np.testing.assert_allclose(
        covered_actual,
        np.broadcast_to(fine_expected[:, None], covered_actual.shape),
        rtol=0.0,
        atol=2.0e-13,
    )

    fine = np.asarray(
        simulation.block_level_state_global("oscillator", 1),
        dtype=np.float64,
    ).reshape(2, 2 * N, 2 * N)
    assert np.isfinite(fine[:, fine_valid]).all()
    fine_actual = fine[:, fine_valid]
    np.testing.assert_allclose(
        fine_actual,
        np.broadcast_to(fine_expected[:, None], fine_actual.shape),
        rtol=0.0,
        atol=2.0e-13,
    )

    level_clocks = [
        row for row in simulation._executor.program_clock_manifest() if row and row[0] == "level"
    ]
    assert [int(row[1]) for row in level_clocks] == [0, 1]
    assert all(
        int(row[2]) == NSTEPS
        and int(row[3]) == 0
        and int(row[4]) == 1
        and abs(float(row[5]) - NSTEPS * DT) < 1.0e-15
        for row in level_clocks
    )
    sync = simulation._executor.program_sync_manifest()
    assert any(row[3] == "average_down" for row in sync)

    # A fresh bind of the same immutable artifact must reproduce the accepted image exactly.
    # This catches stage/candidate storage escaping the previous Program execution.
    replay, replay_report = run_once()
    replay_state = np.asarray(
        replay.block_level_state_global("oscillator", 0),
        dtype=np.float64,
    ).reshape(initial.shape)
    assert replay_report.accepted_steps == NSTEPS
    np.testing.assert_array_equal(replay_state, evolved)
