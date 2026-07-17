"""Public, immutable and evidence-backed ``pops.run`` reports."""
from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from types import ModuleType

import pops
import pytest

from pops.runtime._runtime_instance import RuntimeInstance
from pops.time import ErrorControlledDt
from tests.python.unit.runtime.test_runtime_instance_gate import _Executor
from tests.python.unit.runtime.test_runtime_planning import _install


@pytest.fixture(autouse=True)
def _pure_python_rejection_type(monkeypatch):
    """Exercise the Python controller without requiring the final native link in this unit test."""

    class StepAttemptRejected(RuntimeError):
        pass

    bootstrap = ModuleType("pops._bootstrap")
    bootstrap.StepAttemptRejected = StepAttemptRejected
    sys.modules.pop("pops.runtime._step_strategy", None)
    monkeypatch.setitem(sys.modules, "pops._bootstrap", bootstrap)
    yield
    sys.modules.pop("pops.runtime._step_strategy", None)


def _runtime(executor_type=_Executor):
    plan = _install()
    engine = RuntimeInstance(plan, executor=executor_type(plan))
    return plan, engine, engine


def test_public_run_returns_exact_immutable_report():
    plan, engine, simulation = _runtime()

    report = pops.run(simulation, t_end=2.0, max_steps=2)

    assert type(report) is pops.RunReport
    assert report.accepted_steps == 2
    assert report.rejected_steps == 0
    assert report.final_time == 2.0
    assert report.final_macro_step == 2
    assert report.stop_reason is pops.RunStopReason.TARGET_TIME_REACHED
    assert report.run_identity == engine.last_run_identity
    assert report.bind_identity == plan.bind_identity
    assert report.execution_identity == plan.execution_context.identity
    assert report.artifact_identity == plan.artifact.artifact_identity
    assert report.to_data() == {
        "accepted_steps": 2,
        "rejected_steps": 0,
        "final_time": 2.0,
        "final_macro_step": 2,
        "stop_reason": "target_time_reached",
        "run_identity": report.run_identity.to_data(),
        "bind_identity": report.bind_identity.to_data(),
        "execution_identity": report.execution_identity.to_data(),
        "artifact_identity": report.artifact_identity.to_data(),
        "field_providers": [],
    }
    with pytest.raises(FrozenInstanceError):
        report.accepted_steps = 3
    with pytest.raises(TypeError, match="no implicit truth value"):
        bool(report)


def test_run_at_reached_target_reports_zero_local_steps_without_faking_progress():
    _plan, _engine, simulation = _runtime()
    first = pops.run(simulation, t_end=1.0, max_steps=1)

    already_reached = pops.run(simulation, t_end=1.0, max_steps=0)

    assert first.accepted_steps == 1
    assert already_reached.accepted_steps == 0
    assert already_reached.rejected_steps == 0
    assert already_reached.final_time == 1.0
    assert already_reached.final_macro_step == 1
    assert already_reached.stop_reason is pops.RunStopReason.TARGET_TIME_REACHED
    assert already_reached.run_identity != first.run_identity


def test_report_counts_only_rejected_attempts_from_this_run():
    from pops._bootstrap import StepAttemptRejected

    class RejectOnceExecutor(_Executor):
        def __init__(self, plan):
            super().__init__(plan)
            self._step_strategy = ErrorControlledDt(
                dt_init=1.0,
                rtol=1.0e-3,
                atol=1.0e-6,
                dt_min=0.1,
                dt_max=1.0,
                max_rejections=2,
            )
            self._rejected_once = False

        def step(self, dt):
            if not self._rejected_once:
                self._rejected_once = True
                raise StepAttemptRejected("step attempt rejected during guard: test")
            super().step(dt)

    _plan, engine, simulation = _runtime(RejectOnceExecutor)

    report = pops.run(simulation, t_end=1.0, max_steps=2)

    assert report.accepted_steps == 2
    assert report.rejected_steps == 1
    assert engine._attempt == 3
    assert engine._executor._temporal_restart_state.transaction_stats == {
        "accepted": 2,
        "rejected": 1,
        "failed": 0,
    }


def test_failed_max_steps_run_raises_instead_of_returning_a_success_report():
    _plan, _engine, simulation = _runtime()

    with pytest.raises(RuntimeError, match="max_steps exhausted before t_end"):
        pops.run(simulation, t_end=2.0, max_steps=1)
