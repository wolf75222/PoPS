"""ADC-666: RuntimeInstance envelopes native state and accepted consumers atomically."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from pops.runtime._consumer_contracts import ConsumerCursorSet, ScheduleCursor
from pops.runtime.runtime_instance import RuntimeInstance
from pops.time import ALL_PROVISIONAL_STORES


class _Native:
    def __init__(self, *, fail_commit=False):
        self.t = 0.0
        self.step_index = 0
        self._accepted = None
        self._committed = False
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
        self.owner.native_executor.events.append("publish")
        self.owner.temporaries.discard("sample.tmp")
        self.owner.artifacts.add("sample.out")
        if self.owner.fail_effect:
            raise RuntimeError("fault injected during effect publication")
        self.state = "accepted"
        return self.report

    @property
    def cursor_updates(self):
        return (ScheduleCursor("sample", "accepted", 1),)

    def abort(self):
        if self.state in {"staged", "accepted"}:
            self.owner.temporaries.discard("sample.tmp")
            self.owner.artifacts.discard("sample.out")
            self.state = "rejected"

    def seal(self):
        assert self.state == "accepted"
        self.state = "sealed"


class _Runtime(RuntimeInstance):
    def __init__(self, native, *, fail_effect=False):
        self.native_executor = native
        self.consumer_cursors = ConsumerCursorSet()
        self.consumer_reports = ()
        self._checkpoint_cursor_override = None
        self._attempt = 4
        self.fail_effect = fail_effect
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
    assert runtime.consumer_reports == ()
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
    assert runtime.consumer_reports == ((False, False),)
    assert runtime.artifacts == {"sample.out"}
    assert native.events == ["begin", "commit", "publish", "finalize"]


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


def test_native_commit_failure_discards_prepared_outputs_before_they_become_visible():
    native = _Native(fail_commit=True)
    runtime = _Runtime(native)

    with pytest.raises(RuntimeError, match="native commit"):
        runtime._accepted_step_transaction(lambda: (native.step(0.25), 1))

    assert (native.time(), native.macro_step()) == (0.0, 0)
    assert runtime.consumer_cursors.rows == ()
    assert runtime.consumer_reports == ()
    assert runtime.temporaries == set()
    assert runtime.artifacts == set()
    assert "publish" not in native.events
    assert native.events == ["begin", "commit", "rollback"]
