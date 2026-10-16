"""Python branch/fold nonlinear equations, genuine AMR C++, restart and rejection."""
import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_api_v02_original_field_sequence_native import _same
from tests.python.integration.runtime.test_public_amr_original_field import (
    DT, TARGET, TOL, build, capture, check_original_saved,
)
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("mode", ("branch_true", "branch_false", "fold", "fold_overflow"))
def test_original_field_region_resumes_exact_graph_and_restarts(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, mode
):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API0.2 original AMR field structured regions")
    case, layout = collective_call(world, lambda: build(16, (0,), control_region=mode,
        right_preconditioner="SpatialBasisJacobi@1"))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route="original nonlinear AMR structured region",
                                         compile_artifact=pops.compile)

    def bind():
        return collective_call(world, lambda: pops.bind(artifact,
            resources={"execution_context": artifact_execution_context(artifact)}))

    def accepted_image(runtime, time):
        image = capture(world, runtime)
        with collective_check(world):
            check_original_saved(image[0], 16, 1, time)
            diagnostics = dict(image[2][1])
            factor = 1 if mode == "branch_false" else 2
            np.testing.assert_allclose(diagnostics["structured_original_observation"],
                factor*float(TARGET[0]), rtol=0, atol=TOL)
            # Only the selected branch runs. A repeated loop node reuses its
            # authored identity, whereas the final consumer has a second one.
            stems = [key[:-len(".residual_norm")] for key in diagnostics if key.endswith(".residual_norm")]
            assert len(stems) == 2
            for stem in stems:
                residual = diagnostics[stem+".residual_norm"]
                reference = diagnostics[stem+".reference_residual_norm"]
                assert np.isfinite(residual) and residual >= 0 and reference > 0
                assert residual <= 1e-10*max(1, reference)
        return image

    runtime = bind()
    if mode == "fold_overflow":
        before = capture(world, runtime)
        _, failures = collective_attempt(world, lambda: pops.run(runtime,
            t_end=DT, max_steps=1, console=False))
        with collective_check(world):
            assert all(failures), "every rank must reject the authored capacity violation"
            assert all("fold count outside declared capacity" in str(error) for error in failures)
        _same(world, before, capture(world, runtime))
    else:
        collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
        accepted = accepted_image(runtime, DT)
        directory = collective_directory(world, tmp_path)
        checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"original_region")))
        resumed = bind()
        collective_call(world, lambda: resumed.restart(checkpoint))
        _same(world, accepted, accepted_image(resumed, DT))
        for candidate in (runtime, resumed):
            collective_call(world, lambda candidate=candidate:
                pops.run(candidate, t_end=2*DT, max_steps=1, console=False))
        _same(world, accepted_image(runtime, 2*DT), accepted_image(resumed, 2*DT))
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("region", mode)
    record_property("equation_residual_tolerance", TOL)
