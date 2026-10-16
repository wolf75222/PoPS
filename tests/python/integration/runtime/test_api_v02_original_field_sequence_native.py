"""Two Python-authored nonlinear equations through actual composite C++ barriers."""
import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_public_amr_original_field import (
    DT, TARGET, TOL, build, capture,
)
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _check(world, runtime, *, reuse_request, time):
    image = capture(world, runtime)
    factor = 1 if reuse_request else 2
    with collective_check(world):
        assert len(image[0]) == 2
        assert np.any(image[0][0][3]) and not np.all(image[0][0][3])
        for level, (response, forcing, parameter, active) in enumerate(image[0]):
            shape = (16*2**level,)*2
            q = response.reshape(shape)/time
            a = parameter.reshape(shape)
            f = forcing.reshape(shape)
            target = factor*float(TARGET[0])
            np.testing.assert_allclose(q[active], target, rtol=0, atol=TOL)
            rhs = f if reuse_request else 2*f + 6*float(TARGET[0])**3
            assert np.max(np.abs((a*q+q**3-rhs)[active])) < TOL
        diagnostics = dict(image[2][1])
        assert diagnostics["first_original_observation"] > 0
        for suffix, lower in ((".full_residual_evaluations", 1), (".finite_difference_jvps", 0)):
            work = [value for key, value in diagnostics.items() if key.endswith(suffix)]
            assert len(work) == 2
            assert all(np.isfinite(value) and value == int(value) and value > lower for value in work)
        stems = [key[:-len(".residual_norm")] for key in diagnostics if key.endswith(".residual_norm")]
        assert len(stems) == 2
        for stem in stems:
            residual, reference = diagnostics[stem+".residual_norm"], diagnostics[stem+".reference_residual_norm"]
            assert np.isfinite(residual) and residual >= 0 and reference > 0
            assert residual <= 1e-10*max(1, reference)
    return image


def _same(world, left, right):
    with collective_check(world):
        assert left[1:] == right[1:]
        for a, b in zip(left[0], right[0], strict=True):
            for x, y in zip(a, b, strict=True):
                assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


@pytest.mark.parametrize("reuse_request", (True, False))
def test_original_field_sequence_executes_both_equations_and_restarts(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, reuse_request
):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API0.2 original AMR field sequence")
    case, layout = collective_call(world, lambda: build(16, (0,), second_solve=True,
        reuse_request=reuse_request, right_preconditioner="SpatialBasisJacobi@1"))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route="two original nonlinear AMR barriers",
                                         compile_artifact=pops.compile)

    def bind():
        return collective_call(world, lambda: pops.bind(artifact,
            resources={"execution_context": artifact_execution_context(artifact)}))

    runtime = bind()
    collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
    accepted = _check(world, runtime, reuse_request=reuse_request, time=DT)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"two_original_solves")))
    resumed = bind()
    collective_call(world, lambda: resumed.restart(checkpoint))
    _same(world, accepted, _check(world, resumed, reuse_request=reuse_request, time=DT))
    for candidate in (runtime, resumed):
        collective_call(world, lambda candidate=candidate:
            pops.run(candidate, t_end=2*DT, max_steps=1, console=False))
    _same(world, _check(world, runtime, reuse_request=reuse_request, time=2*DT),
          _check(world, resumed, reuse_request=reuse_request, time=2*DT))
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("request_reused", reuse_request)
    record_property("equation_residual_tolerance", TOL)
