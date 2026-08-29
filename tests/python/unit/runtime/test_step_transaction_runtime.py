"""ADC-666: RuntimeInstance envelopes native state and accepted consumers atomically."""
from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from pops._bootstrap import StepAttemptRejected
from pops.output._consumer_contracts import ConsumerCursorSet, ScheduleCursor
from pops.output._writers.common import _OutputRecoveryRequired, _StagedOutputFile
from pops.runtime._consumer_transaction import ConsumerTransactionReport
from pops.runtime._multi_layout_executor import (
    _CompositeTemporalRestartState,
    _MultiLayoutUniformExecutor,
)
from pops.runtime._runtime_instance import RuntimeInstance
from pops.runtime._step_strategy import prepare_program_run
from pops.runtime._temporal_restart import TemporalRestartState
from pops.time import (
    ALL_PROVISIONAL_STORES,
    BlockProjection,
    ErrorControlledDt,
    FixedDt,
    ProjectAndRecheck,
    StepTransactionReport,
)


class _Native:
    def __init__(
        self,
        *,
        fail_begin: bool = False,
        fail_commit: bool = False,
        fault_phase: str | None = None,
        fail_finalize: bool = False,
        fail_stop: bool = False,
    ):
        self.t = 0.0
        self.step_index = 0
        self._accepted = None
        self._committed = False
        self.fail_begin = fail_begin
        self.fail_commit = fail_commit
        self.fail_finalize = fail_finalize
        self.fail_stop = fail_stop
        self.events = []
        self.scope_events = []
        self.scope_active = False
        self._step_transaction_plan = SimpleNamespace(stores=ALL_PROVISIONAL_STORES)
        self._step_controller = None
        self._last_step_transaction_report = None
        self._executed_step_projections = []
        self.fault_phase = fault_phase
        self.states = {"left": [1.0, 2.0], "right": [3.0, 4.0]}
        self.cache = {"rhs": [5.0]}
        self.history = {"left.U": [[0.5, 1.5]]}
        self.diagnostics = {"accepted_norm": 7.0}

    def time(self):
        return self.t

    def macro_step(self):
        return self.step_index

    @contextmanager
    def _provisional_read_scope(self):
        self.scope_events.append("enter")
        self.scope_active = True
        try:
            yield
        finally:
            self.scope_active = False
            self.scope_events.append("exit")

    def _accepted_transaction_fail_stop_(self):
        return self.fail_stop

    def _consume_step_projections(self):
        result = tuple(self._executed_step_projections)
        self._executed_step_projections.clear()
        return result

    def accepted_stores(self):
        return copy.deepcopy({
            "states": self.states,
            "clock": (self.t, self.step_index),
            "cache": self.cache,
            "history": self.history,
            "diagnostics": self.diagnostics,
        })

    def _mutate_provisional_stores(self, dt):
        self.t += float(dt)
        self.step_index += 1
        self.states["left"][0] += 11.0
        self.states["right"][1] -= 13.0
        self.cache["rhs"].append(17.0)
        self.history["left.U"].append([19.0, 23.0])
        self.diagnostics["provisional_norm"] = 29.0

    def _reject_fault(self, phase):
        error = StepAttemptRejected(f"fault injected during {phase}")
        error.status = "invalid_evaluation"
        error.phase = phase
        error.detail = f"fault injected during {phase}"
        error.disposition = "reject"
        error.reason_code = 666
        raise error

    def step(self, dt):
        self._mutate_provisional_stores(dt)
        if self.fault_phase in {"stage", "solve", "synchronize", "guard"}:
            self._reject_fault(self.fault_phase)
        return float(dt)

    def _begin_step_transaction(self):
        if self._accepted is not None:
            raise RuntimeError("nested transaction")
        self._accepted = self.accepted_stores()
        self._committed = False
        self.events.append("begin")
        if self.fault_phase == "prepare":
            self._mutate_provisional_stores(0.375)
            raise RuntimeError("fault injected during prepare")
        if self.fail_begin:
            self.t = 0.375
            self.step_index = 3
            raise RuntimeError("fault injected during native begin")

    def _commit_step_transaction(self):
        if self._accepted is None:
            raise RuntimeError("missing transaction")
        self.events.append("commit")
        if self.fail_commit:
            raise RuntimeError("fault injected during native commit")
        self._committed = True

    def _finalize_step_transaction(self):
        if self._accepted is None or not self._committed:
            raise RuntimeError("missing committed transaction")
        self.events.append("finalize")
        if self.fail_finalize:
            self._accepted = None
            self._committed = False
            raise RuntimeError("fault injected during native finalization")
        self._accepted = None
        self._committed = False

    def _rollback_step_transaction(self):
        if self._accepted is None:
            raise RuntimeError("missing transaction")
        accepted = self._accepted
        self.states = accepted["states"]
        self.t, self.step_index = accepted["clock"]
        self.cache = accepted["cache"]
        self.history = accepted["history"]
        self.diagnostics = accepted["diagnostics"]
        self._accepted = None
        self._committed = False
        self.events.append("rollback")


class _ScopedChild:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    @contextmanager
    def _provisional_read_scope(self):
        self.events.append(("enter", self.name))
        try:
            yield
        finally:
            self.events.append(("exit", self.name))


class _EffectTransaction:
    def __init__(self, owner, *, at_start=False, at_end=False):
        self.owner = owner
        self.report = (at_start, at_end)
        self.state = "staged"
        owner.temporaries.add("sample.tmp")
        owner.scope_observations.append(("stage", owner._executor.scope_active))

    def accept(self):
        self.owner.scope_observations.append(("accept", self.owner._executor.scope_active))
        self.owner._executor.events.append("publish")
        self.owner.temporaries.discard("sample.tmp")
        self.owner.artifacts.add("sample.out")
        if self.owner.fail_effect:
            raise RuntimeError("fault injected during effect publication")
        self.state = "accepted"
        if self.owner.fail_finalize:
            return ConsumerTransactionReport(
                "accepted", self.owner._consumer_cursors, ("sample",))
        return self.report

    @property
    def cursor_updates(self):
        return (ScheduleCursor("sample", "accepted", 1),)

    @property
    def recoveries(self):
        return self.owner.recovery_authorities

    def abort(self):
        if self.state in {"staged", "accepted"}:
            self.owner.temporaries.discard("sample.tmp")
            self.owner.artifacts.discard("sample.out")
            self.state = "rejected"

    def seal(self):
        assert self.state in {"accepted", "sealed"}
        self.state = "sealed"
        self.owner.finalize_calls += 1
        if self.owner.finalize_calls > 1:
            self.owner.saw_retained_finalizer = any(
                pending.transaction is self
                for pending in self.owner._consumer_finalize_pending
            )
        if self.owner.finalize_failures_remaining:
            self.owner.finalize_failures_remaining -= 1
            raise RuntimeError("fault injected during consumer finalization")
        return ()


class _Runtime(RuntimeInstance):
    def __init__(
        self, native, *, fail_effect=False, fail_finalize=False, recoveries=(),
    ):
        self._executor = native
        self._consumer_cursors = ConsumerCursorSet()
        self._consumer_reports = ()
        self._consumer_finalize_pending = ()
        self._consumer_recoveries = {}
        self._checkpoint_cursor_override = None
        self._attempt = 4
        self.fail_effect = fail_effect
        self.fail_finalize = fail_finalize
        self.finalize_failures_remaining = int(fail_finalize)
        self.finalize_calls = 0
        self.saw_retained_finalizer = False
        self.recovery_authorities = tuple(recoveries)
        self.scope_observations = []
        self.temporaries = set()
        self.artifacts = set()

    def _stage_consumers(self, *, at_start=False, at_end=False):
        return (_EffectTransaction(self, at_start=at_start, at_end=at_end),)


def test_effect_failure_restores_native_and_python_envelopes_and_reports_phase():
    native = _Native()
    runtime = _Runtime(native, fail_effect=True)

    with pytest.raises(RuntimeError, match="fault injected"):
        runtime._accepted_step_transaction(lambda: (native.step(0.25), 1), at_end=True)

    assert (native.time(), native.macro_step()) == (0.0, 0)
    assert runtime._attempt == 4
    assert runtime.consumer_cursors.rows == ()
    assert runtime._consumer_reports == ()
    assert runtime.temporaries == set()
    assert runtime.artifacts == set()
    assert native.events == ["begin", "commit", "publish", "rollback"]
    report = native._last_step_transaction_report
    assert (report.status, report.phase, report.action) == ("failed", "effect", "fail_run")
    assert report.rolled_back_effects == tuple(store.value for store in ALL_PROVISIONAL_STORES)


def test_accepted_step_freezes_consumer_payload_before_hidden_publish_without_late_native_reads():
    native = _Native()
    runtime = _Runtime(native)

    runtime._accepted_step_transaction(lambda: (native.step(0.25), 1))

    assert runtime.scope_observations == [("stage", True), ("accept", False)]
    assert native.scope_events == ["enter", "exit"]


def test_accepted_step_refuses_a_native_executor_without_provisional_read_scope():
    native = _Native()
    native._provisional_read_scope = None
    runtime = _Runtime(native)

    with pytest.raises(RuntimeError, match="_provisional_read_scope"):
        runtime._accepted_step_transaction(lambda: (native.step(0.25), 1))

    assert native.events == ["begin", "rollback"]
    assert native.time() == 0.0
    assert runtime.consumer_cursors.rows == ()


def test_multi_layout_provisional_scopes_enter_in_layout_order_and_unwind_in_reverse():
    events = []
    executor = object.__new__(_MultiLayoutUniformExecutor)
    executor._plan = SimpleNamespace(artifact=SimpleNamespace(layout_programs=(
        SimpleNamespace(layout_id="layout-left"),
        SimpleNamespace(layout_id="layout-right"),
    )))
    executor._engines = {
        "layout-left": _ScopedChild("layout-left", events),
        "layout-right": _ScopedChild("layout-right", events),
    }

    with executor._provisional_read_scope():
        events.append(("body", None))

    assert events == [
        ("enter", "layout-left"),
        ("enter", "layout-right"),
        ("body", None),
        ("exit", "layout-right"),
        ("exit", "layout-left"),
    ]


def test_native_finalize_fail_stop_keeps_accepted_consumers_and_never_rolls_back():
    native = _Native(fail_finalize=True, fail_stop=True)
    runtime = _Runtime(native, fail_finalize=True)

    with pytest.raises(RuntimeError, match="native finalization"):
        runtime._accepted_step_transaction(lambda: (native.step(0.25), 1))

    assert native.events == ["begin", "commit", "publish", "finalize"]
    assert (native.time(), native.macro_step()) == (0.25, 1)
    assert runtime.consumer_cursors.for_consumer("sample").committed_samples == 1
    assert len(runtime._consumer_reports) == 1
    assert any("fail-stop" in value for value in runtime._consumer_reports[0].diagnostics)
    assert runtime._consumer_finalize_pending
    assert runtime._native_fail_stop is True

    runtime._retry_consumer_finalizers()
    assert not runtime._consumer_finalize_pending
    assert any("fail-stop" in value for value in runtime._consumer_reports[0].diagnostics)


def test_missing_native_finalize_witness_fails_closed_without_authorizing_rollback():
    error = RuntimeError("native finalization failed after seal")

    assert RuntimeInstance._native_finalize_fail_stop(SimpleNamespace(), error) is True
    assert any("witness is missing" in note for note in getattr(error, "__notes__", ()))


def test_success_commits_native_clock_cursors_and_attempt_counter_together():
    native = _Native()
    runtime = _Runtime(native)

    result = runtime._accepted_step_transaction(lambda: (native.step(0.125), 2))

    assert result == 0.125
    assert (native.time(), native.macro_step()) == (0.125, 1)
    assert native._accepted is None
    assert runtime._attempt == 6
    assert runtime.consumer_cursors.for_consumer("sample").committed_samples == 1
    assert runtime._consumer_reports == ((False, False),)
    assert runtime.artifacts == {"sample.out"}
    assert native.events == ["begin", "commit", "publish", "finalize"]


def test_passing_guard_does_not_report_an_unexecuted_projection():
    native = _Native()
    strategy = FixedDt(0.125)
    native._step_transaction_plan = SimpleNamespace(
        strategy=strategy,
        stores=ALL_PROVISIONAL_STORES,
        guards=(
            SimpleNamespace(
                name="realizability",
                action=ProjectAndRecheck(BlockProjection()),
            ),
        ),
    )

    native._step_strategy = strategy
    report = prepare_program_run(native).run_step(native, t_end=0.125)

    assert report.projections == ()
    assert report.to_data()["projections"] == []


def test_controller_report_carries_only_the_executed_projection_identity():
    class ProjectingNative(_Native):
        def step(self, dt):
            self._executed_step_projections.append("realizability")
            return super().step(dt)

    native = ProjectingNative()
    strategy = FixedDt(0.125)
    native._step_transaction_plan = SimpleNamespace(
        strategy=strategy,
        stores=ALL_PROVISIONAL_STORES,
        guards=(
            SimpleNamespace(
                name="realizability",
                action=ProjectAndRecheck(BlockProjection()),
            ),
        ),
    )

    native._step_strategy = strategy
    report = prepare_program_run(native).run_step(native, t_end=0.125)

    assert report.projections == ("realizability",)
    assert report.to_data()["projections"] == ["realizability"]


def test_rejected_projection_rolls_back_without_hot_report_materialization():
    class RejectingProjectedNative(_Native):
        def step(self, dt):
            self._executed_step_projections.append("realizability")
            return super().step(dt)

    native = RejectingProjectedNative(fault_phase="guard")
    native._step_transaction_plan = SimpleNamespace(
        stores=ALL_PROVISIONAL_STORES,
        guards=(
            SimpleNamespace(
                name="realizability",
                action=ProjectAndRecheck(BlockProjection()),
            ),
        ),
    )
    accepted = native.accepted_stores()
    runtime = _Runtime(native)

    with pytest.raises(StepAttemptRejected, match="fault injected during guard"):
        runtime._accepted_controller_step(
            native,
            native,
            FixedDt(0.125),
            t_end=0.125,
            controls={},
        )

    assert native.accepted_stores() == accepted
    assert native._executed_step_projections == []
    report = native._last_step_transaction_report
    assert (report.status, report.phase, report.action) == (
        "rejected",
        "guard",
        "reject_attempt",
    )
    assert report.projections == ()


def test_outer_envelope_materializes_projection_report_only_after_native_finalize():
    class ProjectingNative(_Native):
        def step(self, dt):
            self._executed_step_projections.append("realizability")
            return super().step(dt)

        def _consume_step_projections(self):
            assert self.events[-1] == "finalize"
            self.events.append("consume-projections")
            return super()._consume_step_projections()

    native = ProjectingNative()
    native._step_transaction_plan = SimpleNamespace(
        stores=ALL_PROVISIONAL_STORES,
        guards=(
            SimpleNamespace(
                name="realizability",
                action=ProjectAndRecheck(BlockProjection()),
            ),
        ),
    )
    runtime = _Runtime(native)

    report = runtime._accepted_controller_step(
        native,
        native,
        FixedDt(0.125),
        t_end=0.125,
        controls={},
    )

    assert report.projections == ("realizability",)
    assert native._last_step_transaction_report.projections == ("realizability",)
    assert native.events[-2:] == ["finalize", "consume-projections"]


def test_error_controlled_retry_rolls_back_before_opening_the_next_attempt():
    class RejectOnceNative(_Native):
        def __init__(self):
            super().__init__()
            self.native_attempts = 0

        def step(self, dt):
            self.native_attempts += 1
            self._mutate_provisional_stores(dt)
            if self.native_attempts == 1:
                self._reject_fault("guard")
            return float(dt)

    native = RejectOnceNative()
    runtime = _Runtime(native)
    accepted = native.accepted_stores()
    strategy = ErrorControlledDt(
        dt_init=0.2,
        rtol=1.0e-4,
        atol=1.0e-8,
        dt_min=0.01,
        dt_max=0.5,
        max_rejections=2,
        shrink=0.5,
        growth=1.5,
    )

    report = runtime._accepted_controller_step(
        native,
        native,
        strategy,
        t_end=0.5,
        controls={},
    )

    assert report.attempts == 2
    assert native.native_attempts == 2
    assert (native.time(), native.macro_step()) == (0.1, 1)
    assert native.states["left"][0] == accepted["states"]["left"][0] + 11.0
    assert native.states["right"][1] == accepted["states"]["right"][1] - 13.0
    assert native.cache["rhs"] == accepted["cache"]["rhs"] + [17.0]
    assert native.history["left.U"] == accepted["history"]["left.U"] + [[19.0, 23.0]]
    assert native.events == [
        "begin",
        "rollback",
        "begin",
        "commit",
        "publish",
        "finalize",
    ]
    assert runtime._attempt == 6
    assert runtime.consumer_cursors.for_consumer("sample").committed_samples == 1
    assert native._step_controller.next_dt == pytest.approx(0.15)
    assert native._last_step_transaction_report == report


def test_error_controlled_retry_preserves_composite_temporal_attempt_stats():
    class RejectOnceNative(_Native):
        def __init__(self):
            super().__init__()
            self.native_attempts = 0
            self._temporal_restart_state = _CompositeTemporalRestartState(
                (TemporalRestartState(), TemporalRestartState())
            )

        def step(self, dt):
            self.native_attempts += 1
            self._mutate_provisional_stores(dt)
            if self.native_attempts == 1:
                self._reject_fault("guard")
            return float(dt)

    native = RejectOnceNative()
    runtime = _Runtime(native)
    strategy = ErrorControlledDt(
        dt_init=0.2,
        rtol=1.0e-4,
        atol=1.0e-8,
        dt_min=0.01,
        dt_max=0.5,
        max_rejections=2,
        shrink=0.5,
        growth=1.5,
    )

    report = runtime._accepted_controller_step(
        native,
        native,
        strategy,
        t_end=0.5,
        controls={},
    )

    assert report.attempts == 2
    assert native.native_attempts == 2
    for state in native._temporal_restart_state.states:
        assert state.transaction_stats == {
            "accepted": 1,
            "rejected": 1,
            "failed": 0,
        }
        assert state.status == "accepted"
        assert state.synchronized is True


def test_attempt_stats_restore_failure_does_not_mask_the_initiating_error():
    native = _Native()
    native._temporal_restart_state = TemporalRestartState()
    runtime = _Runtime(native)

    def fail_with_malformed_stats():
        native._temporal_restart_state.transaction_stats["failed"] = "malformed"
        raise ValueError("initiating failure")

    with pytest.raises(ValueError, match="initiating failure") as caught:
        runtime._accepted_step_transaction(fail_with_malformed_stats)

    assert any(
        "attempt-statistics restoration also failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )


def test_error_controlled_retry_exhaustion_restores_the_last_accepted_boundary():
    class RejectAlwaysNative(_Native):
        def __init__(self):
            super().__init__()
            self.native_attempts = 0

        def step(self, dt):
            self.native_attempts += 1
            self._mutate_provisional_stores(dt)
            self._reject_fault("guard")

    native = RejectAlwaysNative()
    runtime = _Runtime(native)
    accepted = native.accepted_stores()
    strategy = ErrorControlledDt(
        dt_init=0.2,
        rtol=1.0e-4,
        atol=1.0e-8,
        dt_min=0.01,
        dt_max=0.5,
        max_rejections=1,
        shrink=0.5,
        growth=1.5,
    )

    with pytest.raises(StepAttemptRejected, match="fault injected during guard"):
        runtime._accepted_controller_step(
            native,
            native,
            strategy,
            t_end=0.5,
            controls={},
        )

    assert native.native_attempts == 2
    assert native.accepted_stores() == accepted
    assert native.events == ["begin", "rollback", "begin", "rollback"]
    assert runtime._attempt == 4
    assert runtime.consumer_cursors.rows == ()
    assert runtime.artifacts == set()
    assert native._step_controller.next_dt == pytest.approx(0.2)
    report = native._last_step_transaction_report
    assert (report.status, report.phase, report.action, report.attempts) == (
        "rejected",
        "guard",
        "reject_attempt",
        2,
    )


def test_post_native_finalize_failure_retries_with_owner_and_keeps_acceptance():
    native = _Native()
    runtime = _Runtime(native, fail_finalize=True)

    result = runtime._accepted_step_transaction(lambda: (native.step(0.125), 2))

    assert result == 0.125
    assert (native.time(), native.macro_step()) == (0.125, 1)
    assert native._accepted is None
    assert runtime._attempt == 6
    assert runtime.consumer_cursors.for_consumer("sample").committed_samples == 1
    assert runtime.artifacts == {"sample.out"}
    assert native.events == ["begin", "commit", "publish", "finalize"]
    (report,) = runtime._consumer_reports
    assert report.status == "accepted"
    assert report.diagnostics == ()
    assert runtime.finalize_calls == 2
    assert runtime.saw_retained_finalizer
    assert runtime._consumer_finalize_pending == ()


def test_runtime_instance_retains_and_operates_typed_output_recovery(tmp_path):
    public = Path(tmp_path) / "raced-output.bin"
    public.write_bytes(b"runtime-owned")
    original = public.lstat()
    owner = (int(original.st_dev), int(original.st_ino))
    replacement = public.with_name("raced-output.third-party.bin")
    replacement.write_bytes(b"third-party")
    os.replace(replacement, public)
    with pytest.raises(_OutputRecoveryRequired) as failure:
        _StagedOutputFile._quarantine_owned_path(
            public,
            owner,
            replaced_message="injected RuntimeInstance recovery",
        )
    recovery = failure.value.recovery
    native = _Native()
    runtime = _Runtime(native, recoveries=(recovery,))

    runtime._accepted_step_transaction(lambda: (native.step(0.125), 1))

    (record,) = runtime.consumer_recoveries
    assert record.public_path == public
    assert record.quarantine_path.is_file()
    assert record.state == "retained"
    restored = runtime.restore_consumer_recovery(record.recovery_id)
    assert restored.state == "restored"
    assert public.read_bytes() == b"third-party"
    runtime.cleanup_consumer_recovery(record.recovery_id)
    assert runtime.consumer_recoveries == ()
    assert not record.quarantine_path.exists()


def test_native_failure_rolls_back_even_when_the_fault_happens_after_mutation():
    native = _Native()
    runtime = _Runtime(native)

    def fault():
        native.step(0.5)
        raise RuntimeError("fault injected during synchronize")

    with pytest.raises(RuntimeError, match="synchronize"):
        runtime._accepted_step_transaction(fault)

    assert (native.time(), native.macro_step()) == (0.0, 0)
    assert runtime._attempt == 4


def test_native_begin_failure_rolls_back_partial_mutation_and_python_envelope():
    native = _Native(fail_begin=True)
    native._last_step_transaction_report = StepTransactionReport(
        status="accepted",
        phase="commit",
        action="commit",
        projections=("previous-attempt",),
    )
    runtime = _Runtime(native)

    with pytest.raises(RuntimeError, match="native begin"):
        runtime._accepted_step_transaction(lambda: (native.step(0.25), 1))

    assert (native.time(), native.macro_step()) == (0.0, 0)
    assert native._accepted is None
    assert runtime._attempt == 4
    assert runtime.consumer_cursors.rows == ()
    assert runtime._consumer_reports == ()
    assert native.events == ["begin", "rollback"]
    assert native._last_step_transaction_report.status == "failed"
    assert native._last_step_transaction_report.projections == ()


def test_native_commit_failure_discards_prepared_outputs_before_they_become_visible():
    native = _Native(fail_commit=True)
    runtime = _Runtime(native)

    with pytest.raises(RuntimeError, match="native commit"):
        runtime._accepted_step_transaction(lambda: (native.step(0.25), 1))

    assert (native.time(), native.macro_step()) == (0.0, 0)
    assert runtime.consumer_cursors.rows == ()
    assert runtime._consumer_reports == ()
    assert runtime.temporaries == set()
    assert runtime.artifacts == set()
    assert "publish" not in native.events
    assert native.events == ["begin", "commit", "rollback"]


@pytest.mark.parametrize(
    ("phase", "status", "action", "diagnostic"),
    (
        ("prepare", "failed", "fail_run", "fault injected during prepare"),
        ("stage", "rejected", "reject_attempt", "fault injected during stage"),
        ("solve", "rejected", "reject_attempt", "fault injected during solve"),
        (
            "synchronize",
            "rejected",
            "reject_attempt",
            "fault injected during synchronize",
        ),
        ("guard", "rejected", "reject_attempt", "fault injected during guard"),
        ("effect", "failed", "fail_run", "fault injected during effect publication"),
        ("commit", "failed", "fail_run", "fault injected during native commit"),
    ),
)
def test_fault_injection_matrix_restores_every_available_store_and_reports_exact_phase(
    phase,
    status,
    action,
    diagnostic,
):
    native = _Native(
        fault_phase=phase if phase in {
            "prepare", "stage", "solve", "synchronize", "guard"
        } else None,
        fail_commit=phase == "commit",
    )
    strategy = FixedDt(0.25)
    native._step_transaction_plan = SimpleNamespace(
        strategy=strategy,
        stores=ALL_PROVISIONAL_STORES,
        guards=(
            SimpleNamespace(
                name="realizability",
                action=ProjectAndRecheck(BlockProjection()),
            ),
        ),
    )
    runtime = _Runtime(native, fail_effect=phase == "effect")
    runtime._consumer_cursors = ConsumerCursorSet((
        ScheduleCursor("accepted-sample", "accepted-occurrence", 3),
    ))
    native._last_step_transaction_report = StepTransactionReport(
        status="accepted",
        phase="commit",
        action="commit",
        staged_effects=("states",),
        committed_effects=("states",),
    )
    runtime._consumer_reports = ("accepted-report",)
    runtime._checkpoint_cursor_override = "accepted-checkpoint-cursor"
    runtime.temporaries = {"accepted.tmp"}
    runtime.artifacts = {"accepted.out"}

    accepted_native = native.accepted_stores()
    accepted_cursors = runtime.consumer_cursors.to_data()
    accepted_reports = runtime._consumer_reports
    accepted_checkpoint_cursor = runtime._checkpoint_cursor_override
    accepted_temporaries = set(runtime.temporaries)
    accepted_artifacts = set(runtime.artifacts)
    accepted_attempt = runtime._attempt

    def advance():
        native._step_strategy = strategy
        report = prepare_program_run(native).run_step(native, t_end=0.25)
        return report, report.attempts

    with pytest.raises(RuntimeError, match=diagnostic):
        runtime._accepted_step_transaction(advance)

    assert native.accepted_stores() == accepted_native
    assert native._accepted is None
    assert native._step_controller is None
    assert runtime.consumer_cursors.to_data() == accepted_cursors
    assert runtime._consumer_reports == accepted_reports
    assert runtime._checkpoint_cursor_override == accepted_checkpoint_cursor
    assert runtime.temporaries == accepted_temporaries
    assert runtime.artifacts == accepted_artifacts
    assert runtime._attempt == accepted_attempt

    report = native._last_step_transaction_report
    assert (report.status, report.phase, report.action) == (status, phase, action)
    stores = tuple(store.value for store in ALL_PROVISIONAL_STORES)
    assert report.staged_effects == stores
    assert report.committed_effects == ()
    assert report.rolled_back_effects == stores
    assert report.attempts == 1
    assert report.projections == ()
    assert report.diagnostics == (diagnostic,)
