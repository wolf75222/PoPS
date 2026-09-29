"""Two participant test seam: failure must converge before the next operation."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest

from tests.python.support import collective_checks as checks


def two_ranks(monkeypatch, operation):
    barrier = Barrier(2, timeout=3)
    rows = [None, None]

    def gather(world, error):
        rows[world.rank] = error
        barrier.wait()
        result = tuple(rows)
        barrier.wait()  # neither participant may overwrite the current round
        return result

    with monkeypatch.context() as patch:
        patch.setattr(checks, "_gather", gather)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(operation, SimpleNamespace(rank=rank)) for rank in range(2)]
            return [future.result(timeout=5) for future in futures]


@pytest.mark.parametrize("error", [AssertionError("clock advanced"), ValueError("bad envelope")])
def test_rank_local_check_failure_stops_both_before_next_collective(monkeypatch, error):
    reached = []

    def participant(world):
        try:
            with checks.collective_check(world):
                if world.rank == 1:
                    raise error
            reached.append(world.rank)
        except AssertionError as failure:
            return str(failure)

    messages = two_ranks(monkeypatch, participant)
    assert messages[0] == messages[1]
    assert type(error).__name__ in messages[0]
    assert not reached


def test_unexpected_native_exception_stops_successful_peer(monkeypatch):
    def participant(world):
        def native_call():
            if world.rank == 1:
                raise TypeError("unexpected native result")
            return object()
        with pytest.raises(AssertionError, match="TypeError") as error:
            checks.collective_call(world, native_call)
        return str(error.value)

    messages = two_ranks(monkeypatch, participant)
    assert messages[0] == messages[1]


def test_runtime_error_family_preserves_subclass_diagnostic(monkeypatch):
    class StepAttemptRejected(RuntimeError):
        pass

    def participant(world):
        def refused():
            raise StepAttemptRejected("domain rejected")
        _, errors = checks.collective_attempt(world, refused)
        return errors

    results = two_ranks(monkeypatch, participant)
    assert results[0] == results[1] == (
        ("StepAttemptRejected", "domain rejected", True),) * 2
    _, errors = checks.collective_attempt(None, lambda: int("invalid"))
    assert errors[0][0] == "ValueError" and errors[0][2] is False


def test_snapshot_conversion_failure_prevents_second_state_gather(monkeypatch):
    calls = [[], []]

    class BadArray:
        def __array__(self, *args, **kwargs):
            raise ValueError("bad rank-local array")

    def participant(world):
        def state_global(name):
            calls[world.rank].append(name)
            return BadArray() if world.rank == 1 else [1.]
        runtime = SimpleNamespace(state_global=state_global)
        with pytest.raises(AssertionError, match="bad rank-local array"):
            checks.state_snapshots(runtime, world, ("first", "second"))

    two_ranks(monkeypatch, participant)
    assert calls == [["first"], ["first"]]


def test_success_returns_local_values_and_serial_checks_are_preserved(monkeypatch):
    results = two_ranks(monkeypatch, lambda world: checks.collective_call(world, lambda: world.rank))
    assert results == [0, 1]
    assert checks.collective_call(None, lambda: 17) == 17
    with pytest.raises(AssertionError, match="AssertionError.*unchanged threshold"):
        with checks.collective_check(None):
            assert False, "unchanged threshold"
