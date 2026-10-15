"""Typed error arrays cross real AMR solves, and accepted images restart exactly."""
import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_api_v02_original_field_sequence_native import _same
from tests.python.integration.runtime.test_public_amr_original_field import (
    DT, TOL, build, capture, check_original_saved,
)
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize(("mode", "expected"), (("capture_typed", 3.45),
    ("iterate_typed_success", .32), ("iterate_typed_error", 3.9), ("iterate_typed_budget", -7)))
def test_original_typed_recovery_keeps_array_values_code_and_restart(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, mode, expected
):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API0.2 original AMR typed recovery")
    case, layout = collective_call(world, lambda: build(16, (0,), control_region=mode,
        right_preconditioner="SpatialBasisJacobi@1"))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route="original AMR typed recovery",
                                         compile_artifact=pops.compile)

    def bind():
        return collective_call(world, lambda: pops.bind(artifact,
            resources={"execution_context": artifact_execution_context(artifact)}))

    def accepted_image(runtime, time):
        image = capture(world, runtime)
        with collective_check(world):
            check_original_saved(image[0], 16, 1, time)
            diagnostics = dict(image[2][1])
            np.testing.assert_allclose(diagnostics["structured_original_observation"], expected,
                                       rtol=0, atol=TOL)
        return image

    runtime = bind()
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    accepted = accepted_image(runtime, DT)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"original_typed_recovery")))
    resumed = bind()
    collective_call(world, lambda: resumed.restart(checkpoint))
    _same(world, accepted, accepted_image(resumed, DT))
    for candidate in (runtime, resumed):
        collective_call(world, lambda candidate=candidate:
            pops.run(candidate, t_end=2*DT, max_steps=1, console=False))
    _same(world, accepted_image(runtime, 2*DT), accepted_image(resumed, 2*DT))
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("recovery", mode)
    record_property("error_code", -(1 << 63))
    record_property("equation_residual_tolerance", TOL)
