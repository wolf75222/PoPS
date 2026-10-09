"""Installed-package receipts for persistent integrals; no private runtime mutations."""
import numpy as np
import pops
import pytest

from pops.boundary import TransportBoundarySet
from pops.boundary.transport import Outflow
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.projection import ConservativeCellAverage
from pops.time import FixedDt
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import (
    collective_directory as _directory, save_public_snapshot as _save,
    selected_external_records, ssprk_upwind_step,
)
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N, DT, TOL = 8, .01, 3e-13


def build_case(*, proposed_dt=DT, selected_axis=0):
    frame = Rectangle("receipt", lower=(0., 0.), upper=(1., 1.)).frame(Cartesian2D())
    model = pops.Model("transport", frame=frame)
    state = model.state("density", components=("mass",))
    x, y = frame.axes
    flux = model.flux("transport", frame=frame, state=state,
                      components={x: (state[0],), y: (0 * state[0],)},
                      waves={x: (1.,), y: (0.,)})
    rate = model.rate("balance", equation=ddt(state) == -div(flux))
    case = pops.Case("integral_receipt")
    block = case.block("fluid", model)
    plan = DiscretizationPlan()
    plan.rates.add(rate, FiniteVolume(flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(),
        riemann=riemann.User(state=state,
            body=lambda left, right, fl, fr, speed: .5 * (fl + fr) - .5 * speed * (right - left),
            stability=lambda left, right, fl, fr, speed: speed)))
    plan.boundaries.add(TransportBoundarySet({
        frame.boundaries.x_min: Outflow(state=block[state]),
        frame.boundaries.x_max: Outflow(state=block[state]),
    }, periodic=PeriodicAxes((y,))))
    case.numerics(plan, block=block)
    program = pops.Program("ssprk2_integral")
    temporal = program.state(block[state])
    first = rate(temporal.n)
    stage = program.stage("predictor", c=1)
    predictor = program.value("predictor", temporal.n + program.dt * first, at=stage)
    last = rate(predictor)
    candidate = program.value("accepted", .5 * temporal.n + .5 * predictor
                              + .5 * program.dt * last, at=temporal.next.point)
    program.commit(temporal.next, candidate)
    quantity = program.integral_state("q", initial=.7)
    for evaluation in (first, last):
        program.accept_external_trace(quantity, rate=evaluation, axis=selected_axis,
                                      side=1, component=0, scale=-1.)
    program.step_strategy(FixedDt(proposed_dt))
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                       projection=ConservativeCellAverage()))
    # One y cell retains a genuine Dim2 route and forces the MPI2 split along x.
    layout = Uniform(CartesianGrid(frame=frame, cells=(N, 1), periodic=PeriodicAxes((y,))))
    initial = 1. + .2 * (np.arange(N) + .5) / N  # exact averages of a linear density
    return case, layout, quantity, initial


def _world():
    from pops import _pops
    from pops.codegen._native_mpi import native_mpi_communicator
    return _pops.mpi_world() if native_mpi_communicator(_pops) == "MPI_COMM_WORLD" else None


def _root(world):
    return world is None or int(world.rank) == 0


def _compile(world, case, layout):
    resolved = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    if world is None:
        return pops.compile(resolved)
    from tests.python.integration.mpi._compile_once import compile_resolved_plan_once
    return compile_resolved_plan_once(world, resolved, route="integral-public-restart",
                                      compile_artifact=pops.compile)


def _bind(world, artifact, initial):
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    return collective_call(world, lambda: pops.bind(artifact,
        initial_values={subject: np.ascontiguousarray(initial[None, None, :])},
        resources={"execution_context": artifact_execution_context(artifact)}))


def _assert_quantity(world, runtime, quantity, expected):
    with collective_check(world):
        assert abs(runtime.integral_state(quantity) - expected) < TOL


def _run(world, runtime, end, max_steps):
    return collective_call(world, lambda: pops.run(runtime, t_end=end, max_steps=max_steps,
                                                 console=False))


def _check_stage_ledger(ledgers, quantity, expected_q, expected_increments):
    groups = {}
    selected_counts = []
    for ledger in ledgers:
        assert ledger["quantities"][quantity.identity] == (.7, expected_q)
        selected = selected_external_records(ledger)
        selected_counts.append(len(selected))
        expected_keys = {(r["operation"], r["occurrence"], r["context"], r["quadrature"])
                         for r in selected}
        assert set(ledger["consumed"]) == expected_keys
        for row in selected:
            assert row["orientation"] == -1 and row["multiplicity"] == 1
            assert abs(row["weight"] - .5 * DT) < 1e-17
            assert row["measure"] == 1.
            groups[row["evaluation"]] = groups.get(row["evaluation"], 0.) + (
                row["measure"] * row["flux"] * row["weight"])
    assert len(groups) == 2
    np.testing.assert_allclose(sorted(groups.values()), sorted(expected_increments), atol=TOL, rtol=0)
    assert sum(selected_counts) == 2  # one physical face, observed at two SSPRK stages
    if len(ledgers) > 1:
        assert 0 in selected_counts, "MPI receipt must include a rank owning no selected face"


def test_public_ssprk_integral_restart_in_fresh_instance(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property):
    world = _world()
    case, layout, quantity, initial = build_case()
    artifact = _compile(world, case, layout)
    directory = _directory(world, tmp_path)
    runtime = _bind(world, artifact, initial)
    _assert_quantity(world, runtime, quantity, .7)
    _save(world, runtime, artifact, quantity, directory, "initial")
    expected1, increments1 = ssprk_upwind_step(initial, DT)
    expected2, increments2 = ssprk_upwind_step(expected1, DT)
    q1, q2 = .7 + sum(increments1), .7 + sum(increments1) + sum(increments2)
    _run(world, runtime, DT, 1)
    checkpoint, accepted = _save(world, runtime, artifact, quantity, directory, "accepted")
    _assert_quantity(world, runtime, quantity, q1)
    with collective_check(world):
        if _root(world):
            np.testing.assert_allclose(accepted[0].reshape(-1), expected1, atol=TOL, rtol=0)
            _check_stage_ledger(accepted[2], quantity, runtime.integral_state(quantity), increments1)
    _run(world, runtime, 2 * DT, 1)
    _, continuous = _save(world, runtime, artifact, quantity, directory, "continuous")
    restarted = _bind(world, artifact, initial)
    _assert_quantity(world, restarted, quantity, .7)
    collective_call(world, lambda: restarted.restart(checkpoint))
    _, restored = _save(world, restarted, artifact, quantity, directory, "restored")
    with collective_check(world):
        assert restarted.time() == DT and restarted.macro_step() == 1
        if _root(world):
            np.testing.assert_array_equal(restored[0], accepted[0])
            assert restored[1] == accepted[1]
    _run(world, restarted, 2 * DT, 1)
    _, continued = _save(world, restarted, artifact, quantity, directory, "continued")
    _assert_quantity(world, restarted, quantity, q2)
    with collective_check(world):
        assert restarted.time() == 2 * DT and restarted.macro_step() == 2
        if _root(world):
            np.testing.assert_allclose(continued[0].reshape(-1), expected2, atol=TOL, rtol=0)
            np.testing.assert_array_equal(continued[0], continuous[0])
            assert continued[1] == continuous[1]
            _check_stage_ledger(continued[2], quantity, restarted.integral_state(quantity), increments2)
    record_property("integral_receipts", str(directory))
    record_property("artifact_identity", artifact.artifact_identity.token)


@pytest.mark.parametrize("failure", ("absent_periodic_trace", "unsafe_dt"))
def test_public_integral_refusal_restores_envelope_and_safe_retry(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property, failure):
    world = _world()
    absent = failure == "absent_periodic_trace"
    case, layout, quantity, initial = build_case(proposed_dt=DT if absent else .5,
                                                selected_axis=1 if absent else 0)
    artifact = _compile(world, case, layout)
    runtime = _bind(world, artifact, initial)
    directory = _directory(world, tmp_path)
    _, before = _save(world, runtime, artifact, quantity, directory, "before")
    foreign = pops.Program("foreign").integral_state("q", initial=.7)
    _, foreign_errors = collective_attempt(world, lambda: runtime.integral_state(foreign))
    with collective_check(world):
        assert all(error and error[0] == "ValueError" and "absent" in error[1]
                   for error in foreign_errors), foreign_errors
    _, errors = collective_attempt(world, lambda: pops.run(
        runtime, t_end=DT if absent else .5, max_steps=1, console=False))
    with collective_check(world):
        diagnostic = "no unconsumed face contribution" if absent else "user_face_numerical_stability"
        assert all(error and (error[2] or (absent and error[0] == "ValueError"))
                   and diagnostic in error[1] for error in errors), errors
        assert runtime.time() == 0. and runtime.macro_step() == 0
        assert runtime.integral_state(quantity) == .7
    _, rejected = _save(world, runtime, artifact, quantity, directory, "rejected")
    with collective_check(world):
        if _root(world):
            np.testing.assert_array_equal(rejected[0], before[0])
            assert rejected[1] == before[1]
    if not absent:
        # Public endpoint clipping supplies a safe effective dt to the same artifact/runtime.
        report = _run(world, runtime, DT, 1)
        expected, increments = ssprk_upwind_step(initial, DT)
        _, retried = _save(world, runtime, artifact, quantity, directory, "retried")
        _assert_quantity(world, runtime, quantity, .7 + sum(increments))
        with collective_check(world):
            assert report.accepted_steps == 1
            assert runtime.time() == DT and runtime.macro_step() == 1
            if _root(world):
                np.testing.assert_allclose(retried[0].reshape(-1), expected, atol=TOL, rtol=0)
                _check_stage_ledger(retried[2], quantity, runtime.integral_state(quantity), increments)
    record_property("integral_receipts", str(directory))
    record_property("artifact_identity", artifact.artifact_identity.token)
