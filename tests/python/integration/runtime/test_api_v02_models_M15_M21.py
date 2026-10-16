"""Native finite-profile receptions; no transfer to untested continuum regimes."""
import numpy as np
import pops
import pytest

from examples.api_v02.models_M15_M28 import build_model
from tests.python.support.public_native_profile import native_profile
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.api_v02_moment_profiles import _bind, run_native_moment_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


# M15 orders2/4 are received in the separate true-Dim1 HyQMOM module.
# M19-M21 use a second receiver to bound cold-compilation shard cost.
@pytest.mark.parametrize("model", ("M16", "M17", "M18"))
def test_native_closed_multicell_profile(isolated_native_cache, model, record_property):
    del isolated_native_cache
    run_native_moment_profile(model, record_property)

def test_original_B1_oblique_nonhyperbolic_state_is_refused_without_commit(isolated_native_cache):
    del isolated_native_cache
    _, world = native_profile(scope="API v0.2 original B1 nonreal oblique spectrum")
    spec = collective_call(world, lambda: build_model("M16", cells=16, steps=4, oblique_refusal=True))
    runtime = _bind(world, spec)
    initial = collective_call(world, lambda: runtime.state_global("moments"))
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=spec.dt, max_steps=1, console=False))
    final = collective_call(world, lambda: runtime.state_global("moments"))
    clock = collective_call(world, lambda: (float(runtime.time()), int(runtime.macro_step())))
    with collective_check(world):
        assert all(failure is not None and failure[2] and
            "hyperbolic face evaluation refused publication" in failure[1] for failure in failures), failures
        assert clock == (0., 0), clock
        if initial is not None:
            np.testing.assert_array_equal(final, initial)
