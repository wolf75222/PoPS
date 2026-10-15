"""Whole-macro computed time, volume norm, genuine flux, restart and rollback."""
import numpy as np
import pops
import pytest
from examples.api_v02.method_relaxed_composite import build_t10_composite
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value
from tests.python.support import relaxed_composite_oracle as reference
from tests.python.support.relaxed_flux_receipts import assert_scaled

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _states(runtime, world):
    return [collective_call(world, lambda level=level:
        runtime.block_level_state_global("method", level)) for level in range(runtime.n_levels())]


@pytest.mark.parametrize("ratio", (1, 2))
def test_macro_affine_finalization_scales_state_time_and_reflux(ratio, tmp_path,
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    candidate_spec, candidate_artifact, world = _compile("T10_candidate", builder=build_t10_composite,
        temporal_ratio=ratio, relaxed=False)
    spec, artifact, world = _compile("T10_affine", builder=build_t10_composite, temporal_ratio=ratio)
    candidate = _bind(candidate_spec, candidate_artifact, world)
    runtime = _bind(spec, artifact, world)
    old = _states(runtime, world)
    boxes = runtime.patch_boxes()
    with collective_check(world):
        assert runtime.n_levels() == candidate.n_levels() == 2
        assert boxes == candidate.patch_boxes()
    for instance in (candidate, runtime):
        # Public mono-step uses the full controller/consumer/checkpoint lifecycle.
        # Its bound is distinct from the computed accepted endpoint.
        report = collective_call(world, lambda: pops.step(instance,
            t_bound=instance.time()+1., console=False))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            assert report.stop_reason is pops.RunStopReason.ACCEPTED_STEP_LIMIT_REACHED
    provisional, actual = _states(candidate, world), _states(runtime, world)
    def inspect():
        if old[0] is None: return None
        gamma = reference.relaxation(old, provisional, boxes)
        return dict(gamma=gamma, error=reference.affine_error(actual, old, provisional, gamma),
                    initial=reference.pair(old, old, boxes), energy=reference.pair(actual, actual, boxes))
    metrics = profile_gather_value(world, collective_call(world, inspect))
    gamma = next(row["gamma"] for row in metrics if row is not None)
    with collective_check(world):
        assert .79 < gamma < .96 and abs(runtime.time()-gamma) < 1.e-14
        assert runtime.macro_step() == 1
        for row in metrics:
            if row is not None:
                assert row["error"] < 2.e-13 and abs(row["energy"]-row["initial"]) < 2.e-13
    baseline_image = collective_call(world, lambda: bytes(candidate._executor.program_accepted_state()))
    relaxed_image = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world):
        assert_scaled(baseline_image, relaxed_image, gamma)
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: runtime.checkpoint(str(directory/"relaxed")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(path))
    for instance in (runtime, resumed):
        report = collective_call(world, lambda: pops.step(instance,
            t_bound=instance.time()+1., console=False))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            assert report.stop_reason is pops.RunStopReason.ACCEPTED_STEP_LIMIT_REACHED
    a, b = _states(runtime, world), _states(resumed, world)
    with collective_check(world):
        assert runtime.time() == resumed.time() and runtime.macro_step() == resumed.macro_step() == 2
        for left, right in zip(a, b, strict=True):
            if left is not None: np.testing.assert_array_equal(left, right)
    _expected_state = a
    a = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    b = collective_call(world, lambda: bytes(resumed._executor.program_accepted_state()))
    with collective_check(world): assert a == b
    trajectory = _bind(spec, artifact, world)
    report = collective_call(world, lambda: pops.run(trajectory, t_end=runtime.time(),
                                                    max_steps=2, console=False))
    trajectory_state = _states(trajectory, world)
    with collective_check(world):
        assert report.accepted_steps == 2 and report.rejected_steps == 0
        for left, right in zip(trajectory_state, _expected_state, strict=True):
            if left is not None: np.testing.assert_array_equal(left, right)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("composite_metrics", str(metrics))


def test_rejection_after_affine_publication_restores_all_levels_and_measures(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T10_parent_reject", builder=build_t10_composite,
                                     temporal_ratio=2, reject_parent=True)
    runtime = _bind(spec, artifact, world)
    before = _states(runtime, world)
    image = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=1., max_steps=1, console=False))
    after = _states(runtime, world)
    restored = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world):
        assert failures and all(row and "reject_relaxed_parent" in row[1] for row in failures), failures
        assert runtime.time() == 0 and runtime.macro_step() == 0
        assert image == restored
        for left, right in zip(before, after, strict=True):
            if left is not None: np.testing.assert_array_equal(left, right)
