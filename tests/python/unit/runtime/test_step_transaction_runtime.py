"""ADC-666: RuntimeInstance envelopes native state and accepted consumers atomically."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from pops.output._consumer_contracts import ConsumerCursorSet, ScheduleCursor
from pops.output._writers.common import _OutputRecoveryRequired, _StagedOutputFile
from pops.runtime._runtime_instance import RuntimeInstance
from pops.runtime._consumer_transaction import ConsumerTransactionReport
from pops.time import ALL_PROVISIONAL_STORES


class _Native:
    def __init__(self, *, fail_begin=False, fail_commit=False):
        self.t = 0.0
        self.step_index = 0
        self._accepted = None
        self._committed = False
        self.fail_begin = fail_begin
        self.fail_commit = fail_commit
        self.events = []
        self._step_transaction_plan = SimpleNamespace(stores=ALL_PROVISIONAL_STORES)
        self._step_controller = None
        self._last_step_transaction_report = None

    def time(self):
        return self.t

    def macro_step(self):
        return self.step_index

    def step(self, dt):
        self.t += float(dt)
        self.step_index += 1
        return float(dt)

    def _begin_step_transaction(self):
        if self._accepted is not None:
            raise RuntimeError("nested transaction")
        self._accepted = (self.t, self.step_index)
        self._committed = False
        self.events.append("begin")
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
        self._accepted = None
        self._committed = False

    def _rollback_step_transaction(self):
        if self._accepted is None:
            raise RuntimeError("missing transaction")
        self.t, self.step_index = self._accepted
        self._accepted = None
        self._committed = False
        self.events.append("rollback")


class _EffectTransaction:
    def __init__(self, owner, *, at_start=False, at_end=False):
        self.owner = owner
        self.report = (at_start, at_end)
        self.state = "staged"
        owner.temporaries.add("sample.tmp")

    def accept(self):
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
    public.unlink()
    public.write_bytes(b"third-party")
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
    runtime = _Runtime(native)

    with pytest.raises(RuntimeError, match="native begin"):
        runtime._accepted_step_transaction(lambda: (native.step(0.25), 1))

    assert (native.time(), native.macro_step()) == (0.0, 0)
    assert native._accepted is None
    assert runtime._attempt == 4
    assert runtime.consumer_cursors.rows == ()
    assert runtime._consumer_reports == ()
    assert native.events == ["begin", "rollback"]


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
