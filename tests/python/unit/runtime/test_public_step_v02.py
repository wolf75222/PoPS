"""Public mono-step shares the actual runtime lifecycle, never the raw native step."""
import pops
import pytest
from pops.runtime._run_manifest import RunManifest
from pops.time import ErrorControlledDt
from tests.python.unit.runtime.test_run_report import _runtime, _pure_python_rejection_type
from tests.python.unit.runtime.test_runtime_instance_gate import _Executor


def test_step_stops_after_one_accepted_step_with_exact_envelope():
    plan, runtime, _ = _runtime()
    report = pops.step(runtime, t_bound=3.)
    assert report.accepted_steps == 1 and report.rejected_steps == 0
    assert report.final_time == 1. and report.final_macro_step == 1
    assert report.stop_reason is pops.RunStopReason.ACCEPTED_STEP_LIMIT_REACHED
    temporal = runtime._executor._temporal_restart_state
    assert temporal.status == "accepted" and temporal.synchronized
    assert temporal.transaction_stats == {"accepted": 1, "rejected": 0, "failed": 0}
    manifest = runtime._executor._last_run_manifest
    assert manifest.controls["stop_condition"] == "one_accepted_step"
    assert RunManifest.from_dict(manifest.to_dict()).run_identity == manifest.run_identity
    assert report.bind_identity == plan.bind_identity
    second = pops.step(runtime, t_bound=3.)
    assert second.final_time == 2. and second.accepted_steps == 1
    assert second.run_identity != report.run_identity


def test_step_retries_until_one_acceptance_without_counting_rejections_as_steps():
    from pops._bootstrap import StepAttemptRejected
    class RejectOnce(_Executor):
        def __init__(self, plan):
            super().__init__(plan)
            self._step_strategy = ErrorControlledDt(dt_init=1., rtol=1.e-3, atol=1.e-6,
                dt_min=.1, dt_max=1., max_rejections=2)
            self.once = False
        def step(self, dt):
            if not self.once:
                self.once = True
                raise StepAttemptRejected("step attempt rejected during guard: public-step-test")
            super().step(dt)
    _, runtime, _ = _runtime(RejectOnce)
    report = pops.step(runtime, t_bound=2.)
    assert report.accepted_steps == 1 and report.rejected_steps == 1
    assert report.final_macro_step == 1 and report.final_time == .5
    assert runtime._executor._temporal_restart_state.transaction_stats == {
        "accepted": 1, "rejected": 1, "failed": 0}


def test_run_keeps_target_contract_and_different_request_identity():
    _, run, _ = _runtime()
    _, step, _ = _runtime()
    with pytest.raises(RuntimeError, match="max_steps exhausted"):
        pops.run(run, t_end=3., max_steps=1, console=False)
    result = pops.step(step, t_bound=3.)
    assert run._executor._last_run_manifest.run_identity != result.run_identity
    with pytest.raises(TypeError, match="use pops.step"):
        pops.run(step, t_end=3., max_steps=1, _accepted_step_limit=1)


@pytest.mark.parametrize("bound", (0., -1., float("nan"), float("inf"), True))
def test_step_rejects_invalid_frontier_before_advancing(bound):
    _, runtime, _ = _runtime()
    with pytest.raises(ValueError, match="finite future"):
        pops.step(runtime, t_bound=bound)
    assert runtime.time() == 0 and runtime.macro_step() == 0
