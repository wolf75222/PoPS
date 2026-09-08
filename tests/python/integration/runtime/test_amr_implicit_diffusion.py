"""M7.1 synchronized composite implicit diffusion against a conservative fine-grid reference.

Native qualification requires all N=16,32,64 pairs, four full accepted steps, partial refinement,
physical exchange accounting, and rejected-attempt rollback. Source collection alone is not evidence.
"""

from __future__ import annotations
import numpy as np
import pops
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
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import AMR, Uniform
from pops.lib.amr import StateTransfer
from pops.lib.initial import Gaussian
from pops.math import ValueExpr, CoeffGradient, ddt, div, sqrt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import Diffusion, DiscretizationPlan
from pops.projection import ConservativeCellAverage
from pops.params import RuntimeParam
from pops.solvers import Newton
from pops.time import (
    DerivativeStrategy,
    FailRun,
    FixedDt,
    ImplicitDiffusionStage,
    SolveUnknown,
    every,
)
from tests.python.support.amr_snapshots import composite_active_mask, composite_active_block_state
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]
DT = 0.00025


def build(n, *, kind="constant", refined=True, boundary=False, invalid=False, subcycled=False):
    from pops.physics.diffusion import DiffusiveBoundary

    frame = Rectangle("composite-implicit-square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(
        Cartesian2D()
    )
    model = pops.Model("composite-implicit-heat", frame=frame)
    state = model.state("U", components=("energy",))
    u = state[0]
    variable = (sqrt(1 + 4 * u) - 1) / 2 if kind == "nonlinear_accumulation" else u
    coefficient = 0.1 * (1 + 0.2 * u) if kind == "variable" else 0.1
    if invalid:
        coefficient = 0.1 * (1.1 - u)
    physical = (
        tuple(
            DiffusiveBoundary(axis, side, "conormal", 0.01 if axis == 0 else 0.0)
            for axis in range(2)
            for side in ("lower", "upper")
        )
        if boundary
        else None
    )
    flux = model.diffusive_flux(
        "conduction", state=state, value=CoeffGradient(variable, coefficient), boundaries=physical
    )
    rate = model.rate("heat", equation=ddt(state) == div(flux))
    accumulation = (
        model.local_transform("temperature_to_energy", (u + u * u,), valid_if=u > -0.5)
        if kind == "nonlinear_accumulation"
        else None
    )
    case = pops.Case("composite-implicit-" + kind)
    block = case.block("heat", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, Diffusion(flux=flux))
    case.numerics(numerics, block=block)
    program = pops.Program("composite-implicit-stage")
    temporal = program.state(block[state])
    if invalid:
        program.store_history("prior_energy", temporal.n, depth=1)
    coordinates = program.value("coordinates", temporal.n, at=temporal.next.point)
    request = ImplicitDiffusionStage(
        rate, temporal.n, program.dt, accumulation=accumulation
    ).request(
        unknown=SolveUnknown("coordinate", coordinates),
        seed=temporal.n,
        derivative=DerivativeStrategy("finite_difference"),
    )
    solved = program.solve(
        request,
        solver=Newton(
            tolerance=1e-12,
            max_iterations=20,
            linear_tolerance=1e-8,
            linear_max_iterations=100,
            restart=30,
        ),
    ).consume(action=FailRun())[0]
    conserved = (
        program.transform(solved, transform=accumulation) if accumulation is not None else solved
    )
    candidate = program.value("updated", conserved, at=temporal.next.point)
    program.commit(temporal.next, candidate)
    program.step_strategy(FixedDt(DT))
    case.program(program)
    case.initials.add(
        InitialCondition(
            state=block[state],
            value=Gaussian(
                frame=frame,
                center={frame.x: 0.37, frame.y: 0.43},
                background=1.0,
                amplitude=0.3,
                inverse_width=40.0,
            ),
            projection=ConservativeCellAverage(),
        )
    )
    grid = CartesianGrid(
        frame=frame, cells=(n, n), periodic=None if boundary else PeriodicAxes(frame.axes)
    )
    if not refined:
        return case, Uniform(grid)
    threshold = case.param(RuntimeParam("refine-threshold", default=1.12))
    transfer = AMRTransfer()
    transfer.state(block[state], StateTransfer())
    execution = (
        AMRExecution.subcycled((AMRClockRelation(0, 1, 2),))
        if subcycled
        else AMRExecution.synchronous()
    )
    return case, AMR(
        grid=grid,
        hierarchy=AMRHierarchy(max_levels=2, ratios=(2,)),
        tagging=AMRTagging(
            rules=(Tag(ValueExpr(block[state]) > case.value(threshold)), Buffer(cells=1)),
            hysteresis=Hysteresis(0, EqualityPolicy.HOLD),
            conflict_policy=ConflictPolicy.REFINE_WINS,
        ),
        regrid=AMRRegrid(schedule=every(1000, clock=program.clock)),
        transfer=transfer,
        execution=execution,
    )


def bind(n, **options):
    case, layout = build(n, **options)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    return pops.bind(
        artifact, resources={"execution_context": artifact_execution_context(artifact)}
    )


def mass(runtime, n):
    return sum(
        float(composite_active_block_state(runtime, "heat", level, refinement_ratio=2).sum())
        / (n * 2**level) ** 2
        for level in range(runtime.n_levels())
    )


def assert_no_second_reflux(runtime):
    ledger = tuple(tuple(map(str, row)) for row in runtime._executor.program_flux_ledger_manifest())
    assert not ledger, (
        "composite residual already closes the face flux; accepted reflux must remain empty"
    )


@pytest.mark.parametrize("kind", ["constant", "variable", "nonlinear_accumulation"])
def test_composite_implicit_matches_conservative_reference_and_converges(
    kind, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    errors = []
    for n in (16, 32, 64):
        amr, reference = bind(n, kind=kind), bind(2 * n, kind=kind, refined=False)
        assert amr.n_levels() == 2
        mask0 = composite_active_mask(amr, 0, refinement_ratio=2)
        assert mask0.any() and (~mask0).any(), "the matrix requires an actual coarse/fine interface"
        before = mass(amr, n)
        for runtime in (amr, reference):
            report = pops.run(runtime, t_end=4 * DT, max_steps=4, console=False)
            assert report.accepted_steps == 4 and report.rejected_steps == 0
        assert abs(mass(amr, n) - before) < 5e-11
        assert_no_second_reflux(amr)
        fine = np.asarray(reference.state_global("heat")).reshape(2 * n, 2 * n)
        coarse = fine.reshape(n, 2, n, 2).mean(axis=(1, 3))
        squared = 0.0
        for level, expected in ((0, coarse), (1, fine)):
            mask = composite_active_mask(amr, level, refinement_ratio=2)
            actual = np.asarray(amr.block_level_state_global("heat", level)).reshape(expected.shape)
            squared += float(np.sum((actual[mask] - expected[mask]) ** 2)) / (n * 2**level) ** 2
        errors.append(squared**0.5)
    assert errors[1] < errors[0] / 1.5 and errors[2] < errors[1] / 1.5
    record_property(kind + "_conservative_reference_l2_errors", errors)


def test_composite_implicit_physical_boundary_inventory(
    isolated_native_cache, native_cxx, kokkos_root
):
    n = 32
    runtime = bind(n, boundary=True)
    previous = mass(runtime, n)
    for step in range(4):
        report = pops.run(runtime, t_end=(step + 1) * DT, max_steps=1, console=False)
        assert report.accepted_steps == 1
        records = runtime._executor._program_exchange_records()
        amount = sum(row["integrated_amount"] for row in records)
        assert records and abs(amount - 0.02 * DT) < 2e-12
        current = mass(runtime, n)
        assert abs(current - previous - amount) < 5e-11
        previous = current
        assert_no_second_reflux(runtime)


def test_composite_implicit_failure_restores_every_level_history_and_exchange(
    isolated_native_cache, native_cxx, kokkos_root
):
    runtime = bind(32, invalid=True)
    native = runtime._executor

    def envelope():
        return (
            runtime.time(),
            runtime.macro_step(),
            tuple(runtime.patch_boxes()),
            tuple(
                np.asarray(runtime.block_level_state_global("heat", level)).tobytes()
                for level in range(runtime.n_levels())
            ),
            native._program_exchange_records(),
            tuple(tuple(map(str, row)) for row in native.program_flux_ledger_manifest()),
            tuple(
                (
                    name,
                    native.history_initialized(name),
                    native.history_fill_count(name),
                    tuple(
                        np.asarray(native.history_global(name, slot)).tobytes()
                        for slot in range(native.history_depth(name))
                    ),
                )
                for name in native.history_names()
            ),
        )

    before = envelope()
    with pytest.raises(RuntimeError, match="(?i)(invalid|diffus|spatial)"):
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
    assert envelope() == before


def test_composite_implicit_refuses_true_subcycling_before_publication(
    isolated_native_cache, native_cxx, kokkos_root
):
    runtime = bind(16, subcycled=True)
    before = tuple(
        np.asarray(runtime.block_level_state_global("heat", level)).tobytes()
        for level in range(runtime.n_levels())
    )
    with pytest.raises(RuntimeError, match="(?i)(synchron|subcycl|ratio)"):
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
    assert runtime.time() == 0 and runtime.macro_step() == 0
    assert before == tuple(
        np.asarray(runtime.block_level_state_global("heat", level)).tobytes()
        for level in range(runtime.n_levels())
    )
