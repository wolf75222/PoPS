"""Ordered physical mapping rejects preserve the aggregate transaction boundary."""
from types import SimpleNamespace
import sys

import pytest

from pops.runtime._multi_layout_executor import _MultiLayoutUniformExecutor


class _Rejected(RuntimeError):
    pass


class _Child:
    def __init__(self, name, events):
        self.name, self.events = name, events
        self.value = 0
        self.snapshot = None
        self.reject = False

    def time(self):
        return self.value

    def macro_step(self):
        return self.value

    def step(self, dt):
        self.events.append((self.name, "step"))
        self.value += 1
        if self.reject:
            raise _Rejected(self.name)

    def _begin_step_transaction(self):
        assert self.snapshot is None
        self.events.append((self.name, "begin"))
        self.snapshot = self.value

    def _rollback_step_transaction(self):
        assert self.snapshot is not None
        self.events.append((self.name, "rollback"))
        self.value, self.snapshot = self.snapshot, None


class _Session:
    def __init__(self, name, target, events):
        self.name, self.target, self.events = name, target, events
        self.captured = 0
        self.fail_reset = False

    def capture(self, generation, attempt):
        assert generation == 1 and self.captured == 0
        self.events.append((self.name, "capture", attempt))
        self.captured = attempt

    def apply(self, generation, attempt):
        assert generation == 1 and self.captured == attempt
        self.events.append((self.name, "apply", attempt))
        self.target.value += 10
        return self.name

    def reject_attempt(self, generation, attempt):
        # Retain the native guard: rejecting an untouched route is a bug.
        assert generation == 1 and self.captured == attempt
        self.events.append((self.name, "reject", attempt))
        if self.fail_reset:
            raise RuntimeError("route reset failed")
        self.captured = 0


def _executor(monkeypatch):
    monkeypatch.setitem(sys.modules, "pops._bootstrap",
                        SimpleNamespace(StepAttemptRejected=_Rejected))
    events = []
    field, phase = (_Child(name, events) for name in ("field", "phase"))
    routes = []
    for name, operation, source, target, source_subject, target_subject, timing in (
        ("moment", 2, "phase", "field", "f", "rho", "before-step"),
        ("pullback", 3, "field", "phase", "E", "observation", "after-source-step"),
    ):
        transfer = SimpleNamespace(
            mapping_id=name, operation_abi=operation, source_layout_id=source,
            target_layout_id=target, source_subject_id=source_subject,
            target_subject_id=target_subject,
            synchronization_uri=f"pops://synchronization/{timing}@1",
        )
        routes.append(SimpleNamespace(
            transfer=transfer, session=_Session(name, field if operation == 2 else phase, events)))
    executor = _MultiLayoutUniformExecutor.__new__(_MultiLayoutUniformExecutor)
    executor._engines = {"field": field, "phase": phase}
    executor._transfer_routes = tuple(routes)
    executor._active_transfer_generation = 1
    executor._transfer_attempt = 0
    executor._mapping_evaluations = {"moment": 0, "pullback": 0}
    executor._last_mapping_receipts = ()
    executor._authenticate_mapping_receipt = lambda *args, **kwargs: None
    for child in executor._engines.values():
        child._begin_step_transaction()
    events.clear()
    return executor, events


@pytest.mark.parametrize("rejecting_layout", ["field", "phase"])
def test_ordered_rejection_resets_only_captured_routes_and_retries(monkeypatch, rejecting_layout):
    executor, events = _executor(monkeypatch)
    executor._engines[rejecting_layout].reject = True
    with pytest.raises(_Rejected, match=rejecting_layout):
        executor.step(0.01)

    expected_routes = ["moment"] if rejecting_layout == "field" else ["moment", "pullback"]
    assert [event[0] for event in events if event[1] == "capture"] == expected_routes
    assert [event[0] for event in events if event[1] == "reject"] == expected_routes[::-1]
    assert [event[0] for event in events if event[1] == "rollback"] == ["phase", "field"]
    assert [event[0] for event in events if event[1] == "begin"] == ["field", "phase"]
    assert all(child.value == child.snapshot == 0 for child in executor._engines.values())
    assert executor.mapping_report() == {"moment": 0, "pullback": 0}
    assert executor._last_mapping_receipts == ()

    executor._engines[rejecting_layout].reject = False
    executor.step(0.01)
    assert all(child.value == 11 for child in executor._engines.values())
    assert executor.mapping_report() == {"moment": 1, "pullback": 1}
    assert executor._last_mapping_receipts == ("moment", "pullback")
    assert executor._transfer_attempt == 2


def test_route_reset_failure_still_restores_every_child(monkeypatch):
    executor, events = _executor(monkeypatch)
    executor._engines["field"].reject = True
    executor._transfer_routes[0].session.fail_reset = True
    with pytest.raises(RuntimeError, match="route reset failed"):
        executor.step(0.01)
    assert [event[0] for event in events if event[1] == "rollback"] == ["phase", "field"]
    assert all(child.value == 0 and child.snapshot is None for child in executor._engines.values())
    assert not any(event[1] == "begin" for event in events)
    assert executor.mapping_report() == {"moment": 0, "pullback": 0}
