"""Execute nonlinear coordinate History through restart and rejected publication."""
import numpy as np
import pops
import pytest

from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once
from tests.python.support.nonlinear_representation_history import build, physical_step, history_metadata

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _prepare(world, *, reject):
    case, layout, initial, dt = collective_call(world, lambda: build(reject_after_first=reject))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route="nonlinear representation History", compile_artifact=pops.compile)
    return artifact, initial, dt


def _bind(artifact, initial, world):
    def bind():
        context = artifact_execution_context(artifact)
        subject = artifact.plan.initial_condition_plan.bindings[0].subject
        return pops.bind(artifact, initial_values={subject: initial}, resources={"execution_context": context})
    return collective_call(world, bind)


def _snapshot(runtime, world):
    image = collective_call(world, lambda: bytes(runtime._executor._checkpoint_program_exchanges()))
    state = collective_call(world, lambda: runtime.state_global("material"))
    slots = [collective_call(world, lambda index=index: runtime.history_global("coordinates", index)) for index in (0, 1)]
    diagnostics = collective_call(world, lambda: dict(runtime._executor.program_diagnostics()))
    return image, state, slots, diagnostics, runtime.time(), runtime.macro_step()


def _equal(left, right, world):
    with collective_check(world):
        assert left[0] == right[0]
        assert left[3:] == right[3:]
        for a, b in zip([left[1], *left[2]], [right[1], *right[2]], strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_array_equal(a, b)


def _verify(runtime, initial, dt, step, world):
    value = _snapshot(runtime, world)
    metadata = collective_call(world, lambda: history_metadata(value[0]))
    states = [initial]
    for _ in range(step):
        states.append(physical_step(states[-1], dt))
    lag = max(1, step-1)
    with collective_check(world):
        assert value[5] == step
        assert value[4] == pytest.approx(step*dt, abs=2.e-16, rel=0)
        if value[1] is not None:
            np.testing.assert_allclose(np.asarray(value[1]).reshape(initial.shape), states[-1], rtol=0, atol=2.e-10)
            # Tail rotation leaves the just-stored sample in read slot 1 and
            # recycles the previous sample into write slot 0 at this depth.
            for slot, physical in zip(value[2], (states[lag], states[-1]), strict=True):
                expected = .5*(np.sqrt(1+4*physical)-1)
                np.testing.assert_allclose(np.asarray(slot).reshape(initial.shape), expected, rtol=0, atol=2.e-10)
        for component, name in enumerate(("retained.physical.u", "retained.physical.v")):
            assert abs(value[3][name]-float(states[lag][component].sum())) < 4.e-9
    return value, metadata


def test_coordinate_history_keeps_origin_across_restart_and_next_q(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    native, world = native_profile(scope="nonlinear coordinate History restart")
    artifact, initial, dt = _prepare(world, reject=False)
    runtime = _bind(artifact, initial, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=3*dt, console=False))
    _, first_metadata = _verify(runtime, initial, dt, 1, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=3*dt, console=False))
    before, second_metadata = _verify(runtime, initial, dt, 2, world)
    with collective_check(world):
        assert second_metadata[0] == first_metadata[1]
        assert second_metadata[0][0][1] != second_metadata[1][0][1]
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: runtime.checkpoint(str(directory/"coordinates")))
    restored = _bind(artifact, initial, world)
    collective_call(world, lambda: restored.restart(path))
    _equal(before, _verify(restored, initial, dt, 2, world)[0], world)
    received = []
    for candidate in (runtime, restored):
        collective_call(world, lambda candidate=candidate: pops.step(candidate, t_bound=3*dt, console=False))
        received.append(_verify(candidate, initial, dt, 3, world)[0])
    _equal(*received, world)
    record_property("native_path", str(native.__file__))


def test_failure_after_retaining_coordinate_restores_every_slot_and_obligation(
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    native, world = native_profile(scope="nonlinear coordinate History rollback")
    artifact, initial, dt = _prepare(world, reject=True)
    runtime = _bind(artifact, initial, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=2*dt, console=False))
    before, _ = _verify(runtime, initial, dt, 1, world)
    _, errors = collective_attempt(world, lambda: pops.step(runtime, t_bound=2*dt, console=False))
    with collective_check(world):
        assert errors and all(errors), errors
        assert all("reject_after_history_write" in error[1] for error in errors)
    _equal(before, _verify(runtime, initial, dt, 1, world)[0], world)
    record_property("native_path", str(native.__file__))
