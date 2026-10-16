"""Native hierarchy recovery resumes authored equations and preserves restart images."""
import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_api_v02_original_field_sequence_native import _same
from tests.python.integration.runtime.test_public_amr_original_field import (
    DT, TOL, build, capture, check_original_saved,
)
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("mode", ("capture_success", "capture_runtime", "capture_terminal", "capture_user",
                                  "iterate_success", "iterate_budget", "iterate_runtime"))
def test_original_field_recovery_preserves_result_and_complete_restart(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, mode
):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API0.2 original AMR asynchronous recovery")
    case, layout = collective_call(world, lambda: build(16, (0,), control_region=mode,
        right_preconditioner="SpatialBasisJacobi@1"))
    final_report_prefixes = {"amr_original_field_%d" % value.id for value in case._time._values
                             if value.op == "solve_spatial_field"}
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route="original AMR captured evaluation",
                                         compile_artifact=pops.compile)

    def bind():
        return collective_call(world, lambda: pops.bind(artifact,
            resources={"execution_context": artifact_execution_context(artifact)}))

    def accepted_image(runtime, time):
        image = capture(world, runtime)
        with collective_check(world):
            check_original_saved(image[0], 16, 1, time)
            diagnostics = dict(image[2][1])
            expected = 5 if mode.endswith("success") else -7
            assert diagnostics["structured_original_observation"] == expected
            expected_origin = 0 if mode.endswith("success") else 1 if mode == "capture_user" else 2
            assert diagnostics["structured_original_failure_origin"] == expected_origin
            residuals = [value for key, value in diagnostics.items() if key.endswith(".residual_norm")]
            # A rejected SolveOutcome transfers control before the successful
            # observation/diagnostic publication. Its runtime-failure code is
            # consumed by the Python error branch above; the terminal twin case
            # verifies the iteration-limit reason. The final solve must converge.
            expected_reports = 1 if mode.endswith("runtime") else 2
            assert len(residuals) == expected_reports
            assert all(np.isfinite(value) and 0 <= value <= 1e-10 for value in residuals)
            if mode.endswith("runtime"):
                assert {key.split(".", 1)[0] for key in diagnostics
                        if key.startswith("amr_original_field_")} == final_report_prefixes
        return image

    runtime = bind()
    if mode == "capture_terminal":
        before = capture(world, runtime)
        _, failures = collective_attempt(world, lambda: pops.run(runtime,
            t_end=DT, max_steps=1, console=False))
        with collective_check(world):
            assert all(failures)
            assert all("iteration_limit" in str(error) for error in failures)
        _same(world, before, capture(world, runtime))
        record_property("artifact", artifact.artifact_identity.hexdigest)
        record_property("recovery", mode)
        return
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    accepted = accepted_image(runtime, DT)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"original_recovery")))
    resumed = bind()
    collective_call(world, lambda: resumed.restart(checkpoint))
    _same(world, accepted, accepted_image(resumed, DT))
    for candidate in (runtime, resumed):
        collective_call(world, lambda candidate=candidate:
            pops.run(candidate, t_end=2*DT, max_steps=1, console=False))
    _same(world, accepted_image(runtime, 2*DT), accepted_image(resumed, 2*DT))
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("recovery", mode)
    record_property("equation_residual_tolerance", TOL)
