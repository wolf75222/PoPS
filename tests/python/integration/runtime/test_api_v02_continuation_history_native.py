"""History must remain an executable part of a structural accepted state."""
import numpy as np
import pops
import pytest

from examples.api_v02.continuation_history_uniform import build_continuation_history_uniform
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.api_v02_receipts import _decode, _image
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _snapshot(runtime, world):
    image = collective_call(world, lambda: _image(runtime))
    state = collective_call(world, lambda: runtime.state_global("method"))
    slots = [collective_call(world, lambda slot=slot: runtime.history_global("prior", slot))
             for slot in (0, 1)]
    return image, state, slots


def _equal(left, right, world):
    with collective_check(world):
        assert left[0] == right[0]
        for a, b in zip([left[1], *left[2]], [right[1], *right[2]], strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_array_equal(a, b)


def _verify(spec, runtime, world, step):
    image, state, slots = _snapshot(runtime, world)
    decoded = collective_call(world, lambda: _decode(spec, image,
        rank=0 if world is None else int(world.rank)))
    # Independent recurrence, with all valid cells inspected in both stores.
    factors = [1.]
    previous = 1.
    for _ in range(step):
        current = factors[-1]
        factors.append((1.+1.5*spec.dt)*current-.5*spec.dt*previous)
        previous = current
    with collective_check(world):
        root = decoded["root"]
        assert root["step"] == runtime.macro_step() == step
        assert root["time"] == runtime.time()
        assert root["time"] == pytest.approx(step*spec.dt, rel=0, abs=2.e-16)
        if state is not None:
            initial = np.broadcast_to(1.+.2*np.sin(2*np.pi*(np.arange(8)+.5)/8), (1, 4, 8))
            np.testing.assert_allclose(np.asarray(state).reshape(initial.shape),
                                       factors[-1]*initial, rtol=0, atol=5.e-13)
            np.testing.assert_allclose(np.asarray(slots[1]).reshape(initial.shape),
                                       factors[-2]*initial, rtol=0, atol=5.e-13)
        retained = root["previous"]
        for ordinal, payload in retained["patches"].items():
            (lx, hx, _), (ly, hy, _) = retained["layout"][ordinal]
            gx, gy = (axis[0] for axis in retained["axes"])
            nx, ny = hx-lx+1, hy-ly+1
            values = payload.reshape(1, ny+2*gy, nx+2*gx)[0, gy:gy+ny, gx:gx+nx]
            expected = factors[max(0, step-2)]*(1.+.2*np.sin(2*np.pi*(np.arange(lx, hx+1)+.5)/8))
            np.testing.assert_allclose(values, np.broadcast_to(expected, values.shape), rtol=0, atol=5.e-13)
    return image, state, slots


def test_history_and_continuation_restart_and_next_step(
        tmp_path, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("history-continuation", builder=build_continuation_history_uniform)
    runtime = _bind(spec, artifact, world)
    for step in (1, 2):
        collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
        before = _verify(spec, runtime, world, step)
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: runtime.checkpoint(str(directory/"joint")))
    replay = _bind(spec, artifact, world)
    collective_call(world, lambda: replay.restart(path))
    _equal(before, _verify(spec, replay, world, 2), world)
    after = []
    for candidate in (runtime, replay):
        collective_call(world, lambda candidate=candidate:
            pops.step(candidate, t_bound=spec.t_end, console=False))
        after.append(_verify(spec, candidate, world, 3))
    _equal(*after, world)


def test_rejected_continuation_restores_every_history_slot(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("history-continuation-rejection",
        builder=build_continuation_history_uniform, reject=True)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    before = _verify(spec, runtime, world, 1)
    _, errors = collective_attempt(world, lambda:
        pops.step(runtime, t_bound=spec.t_end, console=False))
    with collective_check(world):
        assert errors and all(errors)
    _equal(before, _verify(spec, runtime, world, 1), world)
