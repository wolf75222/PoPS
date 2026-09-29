"""Test-only convergence at boundaries between native collective operations.

A check block contains local work only. A call contains at most one native
collective operation: this helper cannot repair a hang inside that operation.
Every rank must enter these boundaries in the same order. Returned values stay
local; only exception diagnostics are gathered, including unexpected types.
Each diagnostic is (actual type name, message, isinstance(RuntimeError)), so a
StepAttemptRejected subclass retains the same admission as pytest.raises(RuntimeError).
"""
from contextlib import contextmanager


def _gather(world, failure):
    if world is None:
        return (failure,)
    from pops._native_collectives import allgather_value
    return tuple(allgather_value(world, failure))


def collective_attempt(world, operation):
    value, failure = None, None
    try:
        value = operation()
    except Exception as error:
        failure = (type(error).__name__, str(error), isinstance(error, RuntimeError))
    return value, _gather(world, failure)


def collective_call(world, operation):
    value, failures = collective_attempt(world, operation)
    assert not any(failures), failures
    return value


@contextmanager
def collective_check(world):
    """Converge assertions and other exceptions from a local-only block."""
    failure = None
    try:
        yield
    except Exception as error:
        failure = (type(error).__name__, str(error), isinstance(error, RuntimeError))
    failures = _gather(world, failure)
    assert not any(failures), failures


def state_snapshots(runtime, world, names):
    """Gather one state at a time, checking local conversion before the next."""
    import numpy as np
    rows = []
    for name in names:
        value = collective_call(world, lambda name=name: runtime.state_global(name))
        with collective_check(world):
            rows.append(np.asarray(value).copy())
    return tuple(rows)
