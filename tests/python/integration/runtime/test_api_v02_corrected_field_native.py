"""T12 face replacement, same-time field re-solve and a dependent native state."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_flux_correction import (
    build_t12_corrected_field, reference_screened_poisson, reference_step,
)
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.public_native_profile import profile_gather_value


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


@pytest.mark.parametrize("field_input", ("accepted", "provisional", "old"))
def test_native_replacement_recomputes_consumed_field(
        field_input, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T12_corrected_field",
        builder=build_t12_corrected_field, field_input=field_input)
    runtime = _bind(spec, artifact, world)
    report = collective_call(world, lambda: pops.run(
        runtime, t_end=spec.t_end, max_steps=1, console=False))
    metrics = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        assert abs(runtime.time()-spec.t_end) < 2.e-13
        # state_global gathers the state into every rank, so every rank evaluates the same
        # metrics from the same image.
        owners = [row for row in metrics if row.get("metrics_owner", True)]
        assert len(owners) == len(metrics) and all(row == owners[0] for row in owners), metrics
        for row in owners:
            # All variants execute the same conservative replacement correctly.
            assert row["reference_error"] < 3.e-13
            assert row["mass_defect"] < 3.e-14
            assert row["retained_corrected_error"] < 3.e-13
            assert row["provisional_field_error"] < 2.e-11
            # The response is an accepted second block, so stale observation
            # changes computation; merely exporting two diagnostic fields fails.
            assert row["response_uses_executed_field_error"] < 2.e-12
            assert row["stale_provisional_response_gap"] > 1.e-8
            assert row["stale_old_response_gap"] > 1.e-7
            if field_input == "accepted":
                assert row["passed"], row
            else:
                assert not row["passed"], row
                assert row["field_input_error"] > 1.e-3
                assert row["accepted_field_residual"] > 1.e-3
                expected_gap = row["stale_"+field_input+"_response_gap"]
                assert abs(row["response_error"]-expected_gap) < 2.e-12
    # The Fourier reference is independent of the live CG, stencil evaluator
    # and expression graph. Check its defining equation at the actual data.
    initial = np.full((1, 4, 32), .15)
    initial[..., 6:12] = [.15, .06, .005, 1., .65, .3]
    corrected = reference_step(initial, dt=.4/32, dx=1/32)["corrected"]
    phi = reference_screened_poisson(corrected, cells=(32, 4))
    applied = phi + sum(n*n*(2*phi-np.roll(phi, 1, axis)-np.roll(phi, -1, axis))
                        for n, axis in ((32, 2), (4, 1)))
    with collective_check(world):
        np.testing.assert_allclose(applied, corrected, rtol=0, atol=3.e-13)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("field_input", field_input)
    record_property("metrics", str(metrics))
