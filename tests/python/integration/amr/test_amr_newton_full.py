"""Nonlinear local IMEX execution through the compiled AMR Program.

This is the executable replacement for the former ``engine.IMEX`` rejection
test.  A public Program advances

    du/dt = -k u**2

with implicit Euler.  Every active AMR level consumes a typed
``SolveOutcome`` produced by ``LocalNewton``; no spatial-runtime time
integrator or hidden Newton option participates.  Fine cells and uncovered
coarse cells are compared with the closed-form implicit-Euler root; a forced
iteration limit proves that ``FailRun`` preserves both accepted levels and the
clock. A separate no-root equation under the fine footprint proves that covered
parent cells still participate because their old/new states feed temporal
coarse-to-fine fill-patch interpolation. A NaN injected into one active fine
cell must raise ``InvalidEvaluation`` and roll back both hierarchy levels and
the clock.
"""
from __future__ import annotations

from pathlib import Path

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
from pops.lib.initial import Gaussian
from pops.lib.time import IMEX, IMEX_EULER_TABLEAU
from pops.math import ValueExpr, ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.params import RuntimeParam
from pops.physics import Model
from pops.projection import ConservativeCellAverage
from pops.solvers import LocalNewton
from pops.time import FailRun, FixedDt, every


ROOT = Path(__file__).resolve().parents[4]
N = 12
DT = 0.05
REACTION_RATE = 2.0
HIGH_BUDGET_MAX_ITERATIONS = 30

pytestmark = [
    pytest.mark.compiler,
    pytest.mark.kokkos,
    pytest.mark.native_loader,
    pytest.mark.regression,
]


def _reaction_model(frame):
    x_axis, y_axis = frame.axes
    model = Model("amr-nonlinear-reaction", frame=frame)
    state = model.state("U", components=("u",))
    (u,) = state
    flux = model.flux(
        "transport",
        frame=frame,
        state=state,
        components={x_axis: (0.0 * u,), y_axis: (0.0 * u,)},
        waves={x_axis: (0.0,), y_axis: (0.0,)},
    )
    explicit_rate = model.rate(
        "explicit_rhs",
        equation=ddt(state) == -div(flux),
    )
    model.source(
        "quadratic_decay",
        on=state,
        value=(-REACTION_RATE * u * u,),
    )
    implicit_source = model.operators["quadratic_decay"]
    return model, state, flux, explicit_rate, implicit_source


def _resolved(native_cxx, *, max_iterations=HIGH_BUDGET_MAX_ITERATIONS):
    frame = Rectangle(
        "amr-nonlinear-reaction-domain",
        lower=(0.0, 0.0),
        upper=(1.0, 1.0),
    ).frame(Cartesian2D())
    model, state, flux, explicit_rate, implicit_source = _reaction_model(frame)

    case = pops.Case("amr-nonlinear-reaction-case")
    block = case.block("reactant", model=model, states=(state,))
    instance = block[state]
    numerics = DiscretizationPlan()
    numerics.rates.add(
        explicit_rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=riemann.Rusanov(),
        ),
    )
    case.numerics(numerics, block=block)

    program = IMEX(
        instance,
        explicit_operator=explicit_rate,
        implicit_operator=implicit_source,
        tableau=IMEX_EULER_TABLEAU,
        implicit_solver=LocalNewton(
            tolerance=1.0e-12,
            max_iterations=max_iterations,
        ),
        solve_action=FailRun(),
    )
    program.step_strategy(FixedDt(DT))
    case.program(program)

    x_axis, y_axis = frame.axes
    case.initials.add(
        InitialCondition(
            state=instance,
            value=Gaussian(
                frame=frame,
                center={x_axis: 0.5, y_axis: 0.5},
                background=0.1,
                amplitude=0.9,
                inverse_width=80.0,
            ),
            projection=ConservativeCellAverage(),
        )
    )
    threshold = case.param(RuntimeParam("reaction_refine_threshold", default=0.2))
    transfer = AMRTransfer()
    transfer.state(instance, StateTransfer())
    layout = AMR(
        grid=CartesianGrid(
            frame=frame,
            cells=(N, N),
            periodic=PeriodicAxes(frame.axes),
        ),
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(
                Tag(ValueExpr(instance) > case.value(threshold)),
                Buffer(cells=1),
            ),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1, clock=program.clock)),
        transfer=transfer,
        execution=AMRExecution.synchronous(),
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={
            "include": str(ROOT / "include"),
            "cxx": native_cxx,
        },
    )
    return resolved, program


def _bind(artifact):
    communicator = artifact.platform_manifest.communicator.require(
        "AMR nonlinear Program communicator"
    )
    if communicator == "serial":
        return pops.bind(artifact)
    if communicator == "MPI_COMM_WORLD":
        return pops.bind(
            artifact,
            resources={"execution_context": pops.ExecutionContext.mpi_world(artifact)},
        )
    raise RuntimeError("unsupported AMR nonlinear Program communicator %r" % communicator)


def _implicit_euler_root(values):
    coefficient = DT * REACTION_RATE
    return (
        -1.0 + np.sqrt(1.0 + 4.0 * coefficient * values)
    ) / (2.0 * coefficient)


def _program_solve_contract(program):
    graph = program.to_graph()
    nonlinear_solves = [
        node for node in graph.nodes
        if getattr(node, "op", None) == "solve_local_nonlinear"
    ]
    consumed = [
        node for node in graph.nodes
        if getattr(node, "op", None) == "solve_outcome"
    ]
    solve_ids = {node.node_id for node in nonlinear_solves}
    assert nonlinear_solves
    assert len(consumed) == len(nonlinear_solves)
    assert {
        node.inputs[0].node_id for node in consumed
    } == solve_ids
    assert all(
        node.attrs.to_data()["attrs"]["action"]["kind"] == "fail_run"
        for node in consumed
    )
    assert program.validate() is True


def _level_values(simulation, level):
    side = N * (2 ** level)
    return np.asarray(
        simulation.block_level_state_global("reactant", level),
        dtype=np.float64,
    ).reshape(side, side)


def _fine_valid_mask(simulation):
    side = 2 * N
    valid = np.zeros((side, side), dtype=bool)
    patch_table = simulation.amr.patch_table()
    fine = next(
        row for row in patch_table.per_level
        if int(row["level"]) == 1
    )
    for ilo, jlo, ihi, jhi in fine["boxes"]:
        valid[jlo:jhi + 1, ilo:ihi + 1] = True
    assert np.any(valid)
    return valid


def _coarse_uncovered_from_fine(fine_valid):
    covered = fine_valid.reshape(N, 2, N, 2).any(axis=(1, 3))
    assert np.any(covered)
    assert np.any(~covered)
    return ~covered


@pytest.fixture(scope="module")
def nonlinear_artifact(native_cxx, kokkos_root, tmp_path_factory):
    del kokkos_root
    cache = tmp_path_factory.mktemp("amr-nonlinear-native-cache")
    environment = pytest.MonkeyPatch()
    environment.setenv("POPS_CACHE_DIR", str(cache))
    environment.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
    try:
        resolved, program = _resolved(native_cxx)
        yield pops.compile(resolved), program
    finally:
        environment.undo()


def test_nonlinear_local_imex_executes_on_the_refined_amr_program(
    nonlinear_artifact,
):
    artifact, program = nonlinear_artifact
    _program_solve_contract(program)

    simulation = _bind(artifact)
    assert simulation.n_levels() == 2
    fine_valid = _fine_valid_mask(simulation)
    coarse_uncovered = _coarse_uncovered_from_fine(fine_valid)
    coarse_before = _level_values(simulation, 0).copy()
    fine_before = _level_values(simulation, 1).copy()
    assert np.count_nonzero(fine_before[fine_valid]) > 0

    report = pops.run(
        simulation,
        t_end=DT,
        max_steps=1,
        console=False,
    )
    assert report.accepted_steps == simulation.macro_step() == 1
    assert simulation.time() == pytest.approx(DT)
    assert simulation.n_levels() == 2
    assert np.array_equal(_fine_valid_mask(simulation), fine_valid)
    coarse_after = _level_values(simulation, 0)
    fine_after = _level_values(simulation, 1)

    np.testing.assert_allclose(
        coarse_after[coarse_uncovered],
        _implicit_euler_root(coarse_before[coarse_uncovered]),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        fine_after[fine_valid],
        _implicit_euler_root(fine_before[fine_valid]),
        rtol=2.0e-11,
        atol=2.0e-12,
    )
    assert np.all(np.isfinite(coarse_after[coarse_uncovered]))
    assert np.all(np.isfinite(fine_after[fine_valid]))
    assert np.all(
        coarse_after[coarse_uncovered]
        <= coarse_before[coarse_uncovered] + 2.0e-12
    )
    assert np.all(
        fine_after[fine_valid] <= fine_before[fine_valid] + 2.0e-12
    )
    program_report = simulation.program_report()
    assert program_report.installed
    assert tuple(sorted({int(row["level"]) for row in program_report.flux_ledger})) == (0, 1)


def test_nonlinear_local_imex_high_budget_covered_parent_failure_rolls_back(
    nonlinear_artifact,
):
    artifact, program = nonlinear_artifact
    _program_solve_contract(program)

    simulation = _bind(artifact)
    fine_valid = _fine_valid_mask(simulation)
    coarse_covered = ~_coarse_uncovered_from_fine(fine_valid)
    covered_poison = -4.0
    assert 1.0 + 4.0 * DT * REACTION_RATE * covered_poison < 0.0

    coarse_poisoned = _level_values(simulation, 0).copy()
    coarse_poisoned[coarse_covered] = covered_poison
    simulation._executor.set_block_level_state(
        "reactant",
        0,
        np.ascontiguousarray(coarse_poisoned),
    )
    before = tuple(_level_values(simulation, level).copy() for level in (0, 1))
    before_time = simulation.time()
    before_step = simulation.macro_step()

    with pytest.raises(
        RuntimeError,
        match=r"local_nonlinear failed: (iteration_limit|singular|invalid_evaluation).*action=fail_run",
    ):
        pops.run(
            simulation,
            t_end=DT,
            max_steps=1,
            console=False,
        )

    assert simulation.time() == before_time == 0.0
    assert simulation.macro_step() == before_step == 0
    for level, expected in enumerate(before):
        np.testing.assert_array_equal(_level_values(simulation, level), expected)


def test_nonlinear_local_imex_active_fine_nan_rolls_back(
    nonlinear_artifact,
):
    artifact, program = nonlinear_artifact
    _program_solve_contract(program)

    simulation = _bind(artifact)
    fine_valid = _fine_valid_mask(simulation)
    fine_poisoned = _level_values(simulation, 1).copy()
    poison_j, poison_i = np.argwhere(fine_valid)[0]
    fine_poisoned[poison_j, poison_i] = np.nan
    simulation._executor.set_block_level_state(
        "reactant",
        1,
        np.ascontiguousarray(fine_poisoned),
    )
    failed_before = tuple(_level_values(simulation, level).copy() for level in (0, 1))
    before_failure_time = simulation.time()
    before_failure_step = simulation.macro_step()

    with pytest.raises(
        RuntimeError,
        match=r"local_nonlinear failed: invalid_evaluation.*action=fail_run",
    ):
        pops.run(
            simulation,
            t_end=DT,
            max_steps=1,
            console=False,
        )

    assert simulation.time() == before_failure_time == 0.0
    assert simulation.macro_step() == before_failure_step == 0
    for level, expected in enumerate(failed_before):
        np.testing.assert_allclose(
            _level_values(simulation, level),
            expected,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )


def test_nonlinear_local_imex_failrun_rolls_back_refined_amr_attempt(
    native_cxx,
    isolated_native_cache,
    kokkos_root,
):
    del isolated_native_cache, kokkos_root
    resolved, program = _resolved(native_cxx, max_iterations=1)
    _program_solve_contract(program)

    simulation = _bind(pops.compile(resolved))
    assert simulation.n_levels() == 2
    before = tuple(_level_values(simulation, level).copy() for level in (0, 1))
    before_time = simulation.time()
    before_step = simulation.macro_step()

    with pytest.raises(
        RuntimeError,
        match=r"local_nonlinear failed: iteration_limit.*action=fail_run",
    ):
        pops.run(
            simulation,
            t_end=DT,
            max_steps=1,
            console=False,
        )

    assert simulation.time() == before_time == 0.0
    assert simulation.macro_step() == before_step == 0
    assert simulation.n_levels() == 2
    for level, expected in enumerate(before):
        np.testing.assert_array_equal(_level_values(simulation, level), expected)
