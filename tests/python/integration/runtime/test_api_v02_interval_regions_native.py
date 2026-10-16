"""Issued interval captures execute/reject collectively in native Uniform and AMR loops."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_interval_regions import build_interval_regions
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_attempt, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _states(runtime, world, amr):
    if not amr:
        return [collective_call(world, lambda: runtime.state_global("method"))]
    return [collective_call(world, lambda level=level:
        runtime.block_level_state_global("method", level)) for level in range(runtime.n_levels())]


@pytest.mark.parametrize("amr", (False, True))
def test_interval_captures_execute_all_regions_and_restart(amr, tmp_path,
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("interval_regions", builder=build_interval_regions, amr=amr)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.dt, console=False))
    checks = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        assert all(row["passed"] for row in checks), checks
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"interval_regions")))
    restored = _bind(spec, artifact, world)
    collective_call(world, lambda: restored.restart(checkpoint))
    for candidate in (runtime, restored):
        collective_call(world, lambda candidate=candidate:
            pops.step(candidate, t_bound=2*spec.dt, console=False))
        checks = profile_gather_value(world, collective_call(
            world, lambda candidate=candidate: spec.invariants(candidate)))
        with collective_check(world):
            assert all(row["passed"] for row in checks), checks
    before, after = _states(runtime, world, amr), _states(restored, world, amr)
    with collective_check(world):
        for a, b in zip(before, after, strict=True):
            if a is not None:
                np.testing.assert_array_equal(a, b)
        assert runtime.time() == restored.time() == 2*spec.dt


@pytest.mark.parametrize("amr", (False, True))
def test_interval_rejection_inside_fold_restores_all_accepted_fields(amr,
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("interval_regions_reject", builder=build_interval_regions,
                                     amr=amr, reject=True)
    runtime = _bind(spec, artifact, world)
    before = _states(runtime, world, amr)
    _, errors = collective_attempt(world, lambda: pops.step(runtime, t_bound=spec.dt, console=False))
    after = _states(runtime, world, amr)
    with collective_check(world):
        assert all(errors), errors
        assert runtime.time() == 0 and runtime.macro_step() == 0
        for a, b in zip(before, after, strict=True):
            if a is not None:
                np.testing.assert_array_equal(a, b)
