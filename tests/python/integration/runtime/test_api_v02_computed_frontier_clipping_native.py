"""Real AMR computed-frontier clipping across checkpoint and subsequent steps."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_relaxed_composite import build_t10_composite
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def states(runtime, world):
    return [collective_call(world, lambda level=level:
        runtime.block_level_state_global("method", level)) for level in range(runtime.n_levels())]


@pytest.mark.parametrize("nominal,end", [(.4, .7), (.75, 1.3)])
def test_clipped_computed_frontier_restarts_with_exact_state_clock_and_receipt(
        nominal, end, tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T10_clipped_restart", builder=build_t10_composite,
        temporal_ratio=2, requested_dt=nominal, t_end=end, clip_to_frontier=True)
    original = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(original, t_bound=end, console=False))
    with collective_check(world):
        assert 0 < original.time() < end and original.macro_step() == 1
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: original.checkpoint(str(directory/"clipped")))
    restored = _bind(spec, artifact, world)
    collective_call(world, lambda: restored.restart(path))
    old, copied = states(original, world), states(restored, world)
    with collective_check(world):
        assert original.time() == restored.time() and restored.macro_step() == 1
        for left, right in zip(old, copied, strict=True):
            if left is not None:
                np.testing.assert_array_equal(left, right)
    metrics = []
    for runtime in (original, restored):
        collective_call(world, lambda runtime=runtime: pops.run(runtime, t_end=end, max_steps=32, console=False))
        rows = profile_gather_value(world, collective_call(world, lambda runtime=runtime: spec.invariants(runtime)))
        with collective_check(world):
            owners = [row for row in rows if row.get("metrics_owner", True)]
            assert owners and all(row["passed"] for row in owners), rows
        metrics.append(rows)
    left, right = states(original, world), states(restored, world)
    with collective_check(world):
        assert original.time() == restored.time()
        assert abs(original.time()-end) < 2.e-13
        assert original.macro_step() == restored.macro_step() > 1
        for a, b in zip(left, right, strict=True):
            if a is not None:
                np.testing.assert_array_equal(a, b)
        original_temporal = original._executor._temporal_restart_state
        restored_temporal = restored._executor._temporal_restart_state
        a = original_temporal.controller_state["program_frontier"]
        b = restored_temporal.controller_state["program_frontier"]
        assert a == b
        assert a["requested_duration"] == min(nominal, float.fromhex(a["limit"])
                                               - float.fromhex(a["start"])).hex()
        assert a["rejections"] == 0 and float.fromhex(a["requested_duration"]) < nominal
        assert original_temporal.strategy == restored_temporal.strategy
    record_property("frontier_metrics", str(metrics))
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
