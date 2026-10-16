"""Nonquadratic authored integrands in the real PoPS AMR/Kokkos/MPI pipeline."""
import numpy as np
import pops
import pytest
from examples.api_v02.method_relaxed_quartic import build_t10_quartic
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.public_native_profile import profile_gather_value
from tests.python.support.relaxed_composite_oracle import affine_error
from tests.python.support.relaxed_flux_receipts import assert_scaled
from tests.python.support import relaxed_quartic_oracle as reference

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _states(runtime, world):
    return [collective_call(world, lambda level=level:
        runtime.block_level_state_global("method", level)) for level in range(runtime.n_levels())]


@pytest.mark.parametrize("ratio", (1, 2))
def test_authored_quartic_integral_selects_global_nonzero_root(ratio,
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    base_spec, base_artifact, world = _compile("T10_quartic_provisional", builder=build_t10_quartic,
                                               temporal_ratio=ratio, relaxed=False)
    spec, artifact, world = _compile("T10_quartic_relaxed", builder=build_t10_quartic,
                                      temporal_ratio=ratio)
    baseline, runtime = _bind(base_spec, base_artifact, world), _bind(spec, artifact, world)
    old = _states(runtime, world)
    boxes = runtime.patch_boxes()
    with collective_check(world):
        assert runtime.n_levels() == baseline.n_levels() == 2
        assert boxes == baseline.patch_boxes()
    for instance in (baseline, runtime):
        # This nonquadratic map has gamma>1. The declared future frontier must
        # contain its computed endpoint; the requested step remains exactly .5.
        report = collective_call(world, lambda: pops.step(instance,
            t_bound=instance.time()+1., console=False))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            assert report.stop_reason is pops.RunStopReason.ACCEPTED_STEP_LIMIT_REACHED
    provisional, actual = _states(baseline, world), _states(runtime, world)
    def inspect():
        if old[0] is None: return None
        gamma = reference.relaxation(old, provisional, boxes)
        return dict(gamma=gamma, error=affine_error(actual, old, provisional, gamma),
                    initial=reference.energy(old, boxes), energy=reference.energy(actual, boxes))
    rows = profile_gather_value(world, collective_call(world, inspect))
    gamma = next(row["gamma"] for row in rows if row is not None)
    with collective_check(world):
        assert .25 < gamma < 2. and abs(runtime.time()-.5*gamma) < 2.e-13
        for row in rows:
            if row is not None:
                assert row["error"] < 3.e-12
                assert abs(row["energy"]-row["initial"]) < 2.e-12
    image0 = collective_call(world, lambda: bytes(baseline._executor.program_accepted_state()))
    image1 = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world): assert_scaled(image0, image1, gamma, requested_duration=.5)
    trajectory = _bind(spec, artifact, world)
    report = collective_call(world, lambda: pops.run(trajectory, t_end=runtime.time(),
                                                    max_steps=1, console=False))
    trajectory_state = _states(trajectory, world)
    with collective_check(world):
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        for left, right in zip(actual, trajectory_state, strict=True):
            if left is not None: np.testing.assert_array_equal(left, right)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("quartic_metrics", str(rows))


def test_nonfinite_authored_integrand_rejects_before_any_accepted_publication(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T10_nonfinite_composite", builder=build_t10_quartic,
                                      temporal_ratio=2, invalid_integrand=True)
    runtime = _bind(spec, artifact, world)
    old = _states(runtime, world)
    image = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=1., max_steps=1, console=False))
    actual = _states(runtime, world)
    restored = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world):
        assert failures and all(row and "composite_integral" in row[1] for row in failures), failures
        assert runtime.time() == 0 and runtime.macro_step() == 0 and image == restored
        for left, right in zip(old, actual, strict=True):
            if left is not None: np.testing.assert_array_equal(left, right)
