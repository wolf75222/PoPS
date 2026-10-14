"""Final delivery is selected from real native provisional evaluations."""
import numpy as np
import pops
import pytest

from examples.api_v02.realization_choices import build_choice
from tests.python.support.api_v02_native import execute_case
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("kind,dt", (("discarded", .02), ("ancestral", .02),
                                    ("branch", .01), ("branch", .02),
                                    ("scalar_capture", .02), ("inactive_division", .02),
                                    ("scalar_elementary", .02), ("scalar_domain_capture", .02)))
def test_provisional_choices_and_scalar_capture(kind, dt, isolated_native_cache,
                                               native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API02 selected realization")
    _, artifact, metrics = execute_case(lambda: build_choice(kind, dt=dt), world, "choice-"+kind)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("metrics", str(metrics))


def test_unhandled_scalar_failure_preserves_accepted_state(isolated_native_cache,
                                                         native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API02 scalar failure rollback")
    spec = collective_call(world, lambda: build_choice("scalar_reject"))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case), layout=spec.layout))
    artifact = compile_profile_plan_once(world, plan, route="scalar-reject", compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(artifact, initial_values=dict(spec.initial_values),
                              resources={"execution_context": artifact_execution_context(artifact)}))
    before = collective_call(world, lambda: runtime.state_global("material"))
    if before is not None:
        before = np.asarray(before).copy()
    with pytest.raises((ValueError, RuntimeError), match="iteration|structured|rejected|scalar"):
        pops.run(runtime, t_end=spec.t_end, max_steps=1, console=False)
    after = collective_call(world, lambda: runtime.state_global("material"))
    with collective_check(world):
        if after is not None:
            np.testing.assert_array_equal(after, before)
        assert runtime.time() == 0 and runtime.macro_step() == 0
