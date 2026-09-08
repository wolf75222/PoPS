"""M6.2 finite retry evidence through compiled attempts and the public controller."""
from __future__ import annotations

import json

import numpy as np
import pops
import pytest
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.initial import InitialCondition
from pops.layouts import Uniform
from pops.lib.initial import BindArray
from pops.math import ddt
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, StateStorage
from pops.projection import ConservativeCellAverage
from pops.time import ErrorControlledDt, FixedDt, GuardRole, RejectAttempt
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def bind_case(*, exhausted=False, direct=False):
    frame = Rectangle("retry_square", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    model = pops.Model("retry_source", frame=frame)
    state = model.state("U", components=("mass",))
    (u,) = state
    source = model.source("injection", on=state, value=(1.0 + 0.0 * u,))
    rate = model.rate("evolution", equation=ddt(state) == source)
    case = pops.Case("native_retry")
    block = case.block("material", model=model, states=(state,))
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, StateStorage())
    case.numerics(numerics, block=block)
    program = pops.Program("finite_retry")
    temporal = program.state(block[state])
    candidate = program.value("candidate", temporal.n + program.dt * rate(temporal.n),
                              at=temporal.next.point)
    # This provisional history write precedes the actual compiled rejection.
    program.store_history("attempted", candidate, depth=2)
    increment = program.value("increment", candidate - temporal.n, at=temporal.next.point)
    guarded = program.guard("dt_error_estimate", candidate,
                            program.norm_inf(increment) <= (0.0 if exhausted else 0.0625),
                            action=RejectAttempt(), role=GuardRole.ERROR_ESTIMATE)
    program.commit(temporal.next, guarded)
    strategy = FixedDt(0.0625) if direct else ErrorControlledDt(
        dt_init=0.125, rtol=1e-3, atol=1e-8, dt_min=0.001, dt_max=0.125,
        max_rejections=1, shrink=0.5, growth=1.0)
    program.step_strategy(strategy)
    case.program(program)
    case.initials.add(InitialCondition(state=block[state], value=BindArray(),
                                      projection=ConservativeCellAverage()))
    layout = Uniform(CartesianGrid(frame=frame, cells=(16, 16), periodic=PeriodicAxes(frame.axes)))
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    subject = artifact.plan.initial_condition_plan.bindings[0].subject
    return pops.bind(artifact, initial_values={subject: np.ones((1, 16, 16))},
                     resources={"execution_context": artifact_execution_context(artifact)})


def envelope(runtime):
    native = runtime._executor
    return (runtime.time(), runtime.macro_step(),
            np.asarray(runtime.state_global("material")).tobytes(),
            native._program_exchange_records(), tuple(
                (name, native.history_initialized(name), native.history_fill_count(name), tuple(
                    np.asarray(native.history_global(name, slot)).tobytes()
                    for slot in range(native.history_depth(name))))
                for name in native.history_names()))


def test_native_retry_matches_successful_attempt_without_rejected_history(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    retry = bind_case()
    direct = bind_case(direct=True)
    retry_report = pops.run(retry, t_end=0.125, max_steps=2)
    direct_report = pops.run(direct, t_end=0.125, max_steps=2)
    assert retry_report.accepted_steps == direct_report.accepted_steps == 2
    assert retry_report.rejected_steps == 1
    assert direct_report.rejected_steps == 0
    assert envelope(retry) == envelope(direct)
    assert retry.time() == 0.125
    np.testing.assert_array_equal(retry.state_global("material"), np.full((1, 16, 16), 1.125))
    native = retry._executor
    assert native.history_fill_count("attempted") == 2
    for label, report in (("retry_run", retry_report), ("control_run", direct_report)):
        record_property(label, json.dumps({"accepted_steps": report.accepted_steps,
            "rejected_steps": report.rejected_steps, "final_time": report.final_time,
            "final_macro_step": report.final_macro_step, "run_identity": report.run_identity.token}))


def test_native_finite_retry_exhaustion_restores_accepted_envelope(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    runtime = bind_case(exhausted=True)
    before = envelope(runtime)
    with pytest.raises(RuntimeError, match="dt_error_estimate") as rejected:
        pops.run(runtime, t_end=0.125, max_steps=1)
    assert envelope(runtime) == before
    report = runtime._executor._last_step_transaction_report
    assert report.attempts == 2
    assert any("finite retry budget exhausted" in row for row in report.diagnostics)
    assert report.action == "reject_attempt"
    record_property("failure", str(rejected.value))
    record_property("retry_transaction", json.dumps(report.to_data()))
