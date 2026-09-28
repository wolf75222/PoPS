"""Independent Uniform path checks for CFL selection and rejected publication."""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops.layouts import Uniform
from pops.lib.time import ForwardEuler
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import (DiscretizationPlan, PathConservativeFiniteVolume,
                           reconstruction, riemann, variables)
from pops.time import AdaptiveCFL, FixedDt, StagePoint, TimePoint
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.symbolic_path_case import declarations


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
N = 8


def _resolved(strategy, *, hot_stage=False):
    model, state, flux, product, path = declarations()
    rate = model.rate("balance", equation=ddt(state) == -div(flux) - product)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, PathConservativeFiniteVolume(
        flux=flux, path=path, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case = pops.Case("uniform-path-protocol")
    block = case.block("transport", model)
    case.numerics(numerics, block=block)
    if hot_stage:
        program = pops.Program("hot-uniform-path-stage")
        temporal = program.state(block[state])
        stage = StagePoint("hot", {"main": TimePoint(program.clock, 0)})
        hot = program.value("hot_input", 100. * temporal.n, at=stage)
        rhs = program.value("path_rhs", rate(hot), at=stage)
        result = program.value("next", temporal.n + program.dt * rhs, at=temporal.next.point)
        program.commit(temporal.next, result)
    else:
        program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(strategy)
    case.program(program)
    layout = Uniform(CartesianGrid(frame=path.frame, cells=(N, N),
                                   periodic=PeriodicAxes(path.frame.axes)))
    return pops.resolve(pops.validate(case), layout=layout)


def _initial():
    x = (np.arange(N) + .5) / N
    u = np.broadcast_to(1. + .1 * np.sin(2. * np.pi * x), (N, N))
    v = np.full((N, N), .3)
    return np.ascontiguousarray(np.stack((u, v)))


def _step(initial, dt):
    right = np.roll(initial, -1, axis=2)
    speed = .5 + np.maximum(initial[1] ** 2, right[1] ** 2)
    flux = .25 * (initial + right) - .5 * speed * (right - initial)
    action = np.zeros_like(initial)
    action[0] = (initial[1] ** 2 + initial[1] * right[1] + right[1] ** 2) \
        * (right[0] - initial[0]) / 3.
    return initial + dt * N * (np.roll(flux, 1, axis=2) - flux
                               - .5 * (action + np.roll(action, 1, axis=2)))


def _bind(strategy, *, hot_stage=False):
    artifact = pops.compile(_resolved(strategy, hot_stage=hot_stage))
    initial = _initial()
    runtime = pops.bind(artifact, initial_state={"transport": initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    return runtime, initial


def test_uniform_path_adaptive_cfl_without_max_dt_matches_two_step_oracle(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    cfl = .25
    runtime, initial = _bind(AdaptiveCFL(cfl=cfl))
    # The generated proposal includes both incident faces of each of two axes.
    # v stays constant, so the exact proposal does not change after the first step.
    dt = cfl / (2 * 2 * (.5 + .3 ** 2) * N)
    report = pops.run(runtime, t_end=2 * dt, max_steps=2)
    assert report.accepted_steps == runtime.macro_step() == 2
    assert runtime.time() == pytest.approx(2 * dt, rel=0., abs=1.e-15)
    actual = np.asarray(runtime.state_global("transport")).reshape(initial.shape)
    np.testing.assert_allclose(actual, _step(_step(initial, dt), dt), rtol=0., atol=6.e-14)
    np.testing.assert_array_equal(actual[1], initial[1])
    assert np.max(np.abs(actual[0] - initial[0])) > 1.e-5


@pytest.mark.parametrize("hot_stage", (False, True), ids=("fixed_dt", "actual_stage_cfl"))
def test_uniform_path_refusal_preserves_accepted_state_and_clock(
        isolated_native_cache, native_cxx, kokkos_root, hot_stage):
    del isolated_native_cache, native_cxx, kokkos_root
    dt = 1.e-4
    strategy = AdaptiveCFL(cfl=.25, max_dt=dt) if hot_stage else FixedDt(dt)
    runtime, initial = _bind(strategy, hot_stage=hot_stage)
    message = ("actual incident-face CFL exceeds Courant" if hot_stage
               else "active authored step_cfl Courant")
    # A serial invalid_argument maps to ValueError; collective refusals and the
    # numerical stage guard map to RuntimeError. Both must retain the diagnostic.
    with pytest.raises((ValueError, RuntimeError), match=message):
        pops.run(runtime, t_end=dt, max_steps=1)
    np.testing.assert_array_equal(
        np.asarray(runtime.state_global("transport")).reshape(initial.shape), initial)
    assert runtime.time() == 0.
    assert runtime.macro_step() == 0


def test_uniform_path_checkpoint_restart_does_not_request_an_absent_poisson_field(
        isolated_native_cache, native_cxx, kokkos_root, tmp_path):
    del isolated_native_cache, native_cxx, kokkos_root
    dt = 1.e-4
    artifact = pops.compile(_resolved(AdaptiveCFL(cfl=.25, max_dt=dt)))
    initial = _initial()

    def bind():
        return pops.bind(artifact, initial_state={"transport": initial}, resources={
            "execution_context": artifact_execution_context(artifact)})

    uninterrupted = bind()
    pops.run(uninterrupted, t_end=dt, max_steps=1)
    checkpoint = uninterrupted.checkpoint(tmp_path / "path-without-default-field")
    pops.run(uninterrupted, t_end=2 * dt, max_steps=1)
    expected = np.asarray(uninterrupted.state_global("transport")).copy()

    restarted = bind()
    restarted.restart(checkpoint)
    assert restarted.time() == pytest.approx(dt, rel=0., abs=1.e-16)
    assert restarted.macro_step() == 1
    report = pops.run(restarted, t_end=2 * dt, max_steps=1)
    np.testing.assert_array_equal(np.asarray(restarted.state_global("transport")), expected)
    assert report.accepted_steps == 1
    assert restarted.macro_step() == 2
    assert restarted.time() == pytest.approx(2 * dt, rel=0., abs=1.e-16)
