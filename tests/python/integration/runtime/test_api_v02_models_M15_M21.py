"""Native finite-profile receptions; no transfer to untested continuum regimes."""
import numpy as np
import pops
import pytest

from examples.api_v02.models_M15_M28 import build_model
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _bind(world, spec):
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route=spec.profile, compile_artifact=pops.compile)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    return collective_call(world, lambda: pops.bind(artifact, initial_values=spec.initial_values,
        resources={"execution_context": context}))


# M15 orders2/4 are received in the separate true-Dim1 HyQMOM module.
@pytest.mark.parametrize("model", ("M16", "M17", "M18", "M19", "M20", "M21"))
def test_native_closed_multicell_profile(isolated_native_cache, model, record_property):
    del isolated_native_cache
    native, world = native_profile(scope="API v0.2 moment and finite phase-space PDE")
    spec = collective_call(world, lambda: build_model(model, cells=16, steps=4))
    runtime = _bind(world, spec)
    collective_call(world, lambda: pops.run(runtime, t_end=spec.t_end, max_steps=4, console=False))
    # This callback performs collective native gathers on all ranks.
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    with collective_check(world):
        assert metrics["passed"], metrics
    record_property("profile", spec.profile)
    record_property("native_path", str(native.__file__))
    record_property("metrics", str(metrics))


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
