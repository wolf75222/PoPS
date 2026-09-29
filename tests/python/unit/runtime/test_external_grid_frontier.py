"""C22: authored binary64 grid points are distinct temporal frontiers."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from pops.runtime._runtime_instance import RuntimeInstance
from pops.runtime._step_strategy import prepare_program_run
from pops.time import ExternalTimeGrid
from tests.python.unit.runtime.test_step_strategy import _Engine, _Native
from tests.python.unit.runtime.test_runtime_instance_gate import _Executor, _install


def _prepared(grid):
    return prepare_program_run(_Engine(ExternalTimeGrid("grid")), {"grid": grid})


@pytest.mark.parametrize("next_time,limit", [
    (1.e-18, 5.e-19),
    (math.ulp(0.), 0.),
    (math.nextafter(.1, math.inf), .1),
    (math.nextafter(1.e100, math.inf), 1.e100),
])
def test_next_grid_point_cannot_cross_the_requested_frontier(next_time, limit):
    native = _Native()
    with pytest.raises(RuntimeError, match="final time is not a declared grid point"):
        _prepared((0., next_time)).run_step(native, t_end=limit)
    assert native.time() == 0. and native.macro_step() == 0
    assert native.calls == []


@pytest.mark.parametrize("now", (5.e-19, math.nextafter(.1, math.inf)))
def test_nearby_undeclared_current_time_is_not_relabelled(now):
    native = _Native()
    native.t = now
    with pytest.raises(RuntimeError, match="current time is not a declared grid point"):
        _prepared((0., .1, .2)).run_step(native, t_end=.2)
    assert native.time() == now and native.calls == []


@pytest.mark.parametrize("grid", [
    (0., .1, .2, .3),
    (0., math.ulp(0.), 2 * math.ulp(0.)),
    (0., 1.e-18, 2.e-18),
    (1., math.nextafter(1., math.inf), 1. + 4 * math.ulp(1.)),
    (1.e100, math.nextafter(1.e100, math.inf), 1.e100 + 4 * math.ulp(1.e100)),
])
def test_each_representable_grid_point_is_reached_exactly(grid):
    native = _Native()
    native.t = grid[0]
    prepared = _prepared(grid)
    for index, expected in enumerate(grid[1:], 1):
        prepared.run_step(native, t_end=grid[-1])
        assert native.time() == expected and native.macro_step() == index


@pytest.mark.parametrize("start,end", ((-1.e308, 1.e308), (-1., 1.e-18)))
def test_unrepresentable_interval_is_refused_before_native_execution(start, end):
    native = _Native()
    native.t = start
    with pytest.raises(RuntimeError, match="representable.*interval"):
        _prepared((start, end)).run_step(native, t_end=end)
    assert native.time() == start and native.calls == []


@pytest.mark.parametrize("wrong_clock", ("time", "step"))
def test_wrong_reached_clock_rolls_back_the_real_publication_envelope(wrong_clock):
    class WrongClock(_Executor):
        def step(self, dt):
            super().step(dt)
            if wrong_clock == "time":
                self._time = math.nextafter(self._time, math.inf)
            else:
                self._step += 1

    executor = WrongClock(_install())
    executor._step_strategy = ExternalTimeGrid("grid")
    runtime = RuntimeInstance(executor._plan, executor=executor)
    before = np.asarray(runtime.state_global("fluid")).copy()
    temporal = executor._temporal_restart_state.to_data()
    with pytest.raises(RuntimeError, match="ExternalTimeGrid.*reached"):
        runtime._run(t_end=.1, max_steps=1, console=False, grid=(0., .1))
    assert runtime.time() == 0. and runtime.macro_step() == 0
    np.testing.assert_array_equal(runtime.state_global("fluid"), before)
    assert executor._temporal_restart_state.to_data() == temporal
    assert runtime._publisher.post_commit_reports == ()


@pytest.mark.parametrize("local_bad", (False, True))
def test_preflight_converges_rank_local_entry_error_before_any_step(monkeypatch, local_bad):
    from pops import _native_collectives
    from pops.runtime import _step_strategy

    native = _Native()
    if local_bad:
        native.t = math.ulp(0.)
    votes = []

    def world(_engine, *, preparing=False):
        assert preparing
        return SimpleNamespace(rank=0, size=2)

    def gather(_world, envelope):
        votes.append(envelope)
        peer = {"contract": None, "error": "peer current time is not a declared grid point"}
        return (envelope, peer)

    monkeypatch.setattr(_step_strategy, "_attempt_world", world)
    monkeypatch.setattr(_native_collectives, "allgather_value", gather)
    with pytest.raises(RuntimeError, match="collective ExternalTimeGrid preparation failed"):
        _prepared((0., .1)).run_step(native, t_end=.1)
    assert len(votes) == 1 and native.calls == []


@pytest.mark.parametrize("field", range(7))
def test_preflight_compares_every_exact_interval_authority(monkeypatch, field):
    from pops import _native_collectives
    from pops.runtime import _step_strategy

    native = _Native()

    def gather(_world, envelope):
        peer = list(envelope["contract"])
        peer[field] = "different exact authority"
        return (envelope, {"contract": tuple(peer), "error": None})

    monkeypatch.setattr(_step_strategy, "_attempt_world",
                        lambda _engine, **_kwargs: SimpleNamespace(rank=0, size=2))
    monkeypatch.setattr(_native_collectives, "allgather_value", gather)
    with pytest.raises(RuntimeError, match="preparation differs between ranks"):
        _prepared((0., .1)).run_step(native, t_end=.1)
    assert native.calls == []
