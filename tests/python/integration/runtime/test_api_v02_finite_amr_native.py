"""Real PoPS AMR execution of Python finite regions, restart and parent rollback."""
import numpy as np
import pops
import pytest

from examples.api_v02.finite_amr import build_finite_amr
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.public_native_profile import native_profile, profile_gather_value
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _images(runtime, world):
    return [collective_call(world, lambda level=level: runtime.block_level_state_global("method", level))
            for level in range(runtime.n_levels())]


@pytest.mark.parametrize("distributed,ratio", ((False, 1), (False, 2), (True, 1), (True, 2)))
def test_finite_regions_execute_on_every_native_amr_invocation(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, distributed, ratio
):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="finite AMR regions")
    ranks = 1 if world is None else int(world.size)
    spec, artifact, world = _compile("finite_amr", builder=build_finite_amr,
        distributed=distributed, ranks=ranks, ratio=ratio, steps=3)
    runtime = _bind(spec, artifact, world)
    first = collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    with collective_check(world):
        assert first.accepted_steps == 1 and runtime.n_levels() == 2
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"finite_amr")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    for candidate in (runtime, resumed):
        report = collective_call(world, lambda candidate=candidate:
            pops.run(candidate, t_end=spec.t_end, max_steps=2, console=False))
        with collective_check(world):
            assert report.accepted_steps == 2 and report.rejected_steps == 0
        metrics = profile_gather_value(world, collective_call(world, lambda candidate=candidate: spec.invariants(candidate)))
        with collective_check(world):
            assert all(row["passed"] for row in metrics), metrics
    before, after = _images(runtime, world), _images(resumed, world)
    with collective_check(world):
        for left, right in zip(before, after, strict=True):
            if left is not None:
                np.testing.assert_array_equal(left, right)
        assert runtime.time() == resumed.time() == spec.t_end
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("distributed", distributed)
    record_property("temporal_ratio", ratio)


def test_finite_amr_parent_rejection_restores_every_level(
    isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("finite_amr_reject", builder=build_finite_amr, reject_parent=True)
    runtime = _bind(spec, artifact, world)
    before = _images(runtime, world)
    accepted = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    _, failures = collective_attempt(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    after = _images(runtime, world)
    restored = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world):
        assert failures
        assert runtime.time() == 0. and runtime.macro_step() == 0
        assert accepted and accepted == restored
        for left, right in zip(before, after, strict=True):
            if left is not None:
                np.testing.assert_array_equal(left, right)
    record_property("artifact", artifact.artifact_identity.hexdigest)
