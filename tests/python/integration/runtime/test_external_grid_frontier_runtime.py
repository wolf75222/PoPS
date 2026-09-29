"""C22 reception: public native run, exact grid, and rank-local clock fault rollback."""

import math

import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_balance_multiplicity_runtime import _case, CELLS
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


def test_native_external_grid_frontier_and_rank_local_landing_rollback(
        isolated_native_cache, native_cxx, kokkos_root, monkeypatch):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, _, _ = _case("double")
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    context = artifact_execution_context(artifact)
    world = context.communicator.handle
    rank = 0 if world is None else int(world.rank)
    size = 1 if world is None else int(world.size)
    subject = artifact.plan.initial_condition_plan.bindings[0].subject

    def bind():
        return pops.bind(artifact, initial_values={subject: np.ones((1, CELLS, CELLS))},
                         resources={"execution_context": context})

    refused = bind()
    before = np.asarray(refused.state_global("material")).copy()
    with pytest.raises(RuntimeError, match="final time is not a declared grid point"):
        pops.run(refused, t_end=5.e-19, max_steps=1, console=False,
                 time_grid=(0., 1.e-18, 2.e-18))
    assert refused.time() == 0. and refused.macro_step() == 0
    np.testing.assert_array_equal(refused.state_global("material"), before)

    valid = bind()
    report = pops.run(valid, t_end=.3, max_steps=3, console=False,
                      time_grid=(0., .1, .2, .3))
    assert report.accepted_steps == 3
    assert valid.time() == .3 and valid.macro_step() == 3
    actual = np.asarray(valid.state_global("material"))
    if actual.size:
        # Genuine source balance U'=4U, three forward-Euler intervals.
        expected = np.prod([1. + 4. * (b - a)
                            for a, b in zip((0., .1, .2), (.1, .2, .3), strict=True)])
        np.testing.assert_allclose(actual, expected, rtol=0., atol=3.e-14)

    failed = bind()
    before = np.asarray(failed.state_global("material")).copy()
    executor = failed._executor
    raw = executor._native_step_target()
    temporal_before = executor._temporal_restart_state.to_data()

    class WrongLanding:
        """Fault only the observation after the genuine native step has mutated fields."""
        stepped = False

        def time(self):
            value = raw.time()
            if self.stepped and rank == (1 if size > 1 else 0):
                return math.nextafter(value, math.inf)
            return value

        def macro_step(self):
            return raw.macro_step()

        def step(self, dt):
            result = raw.step(dt)
            self.stepped = True
            return result

    proxy = WrongLanding()
    with monkeypatch.context() as patch:
        patch.setattr(executor, "_native_step_target", lambda: proxy)
        with pytest.raises(RuntimeError, match="ExternalTimeGrid reached clock"):
            pops.run(failed, t_end=.1, max_steps=1, console=False, time_grid=(0., .1))
    assert failed.time() == 0. and failed.macro_step() == 0
    np.testing.assert_array_equal(failed.state_global("material"), before)
    assert executor._temporal_restart_state.to_data() == temporal_before
    retry = pops.run(failed, t_end=.1, max_steps=1, console=False, time_grid=(0., .1))
    assert retry.accepted_steps == 1 and failed.time() == .1
