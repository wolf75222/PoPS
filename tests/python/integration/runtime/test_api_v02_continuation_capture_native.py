"""Full child image promotion/recovery across real AMR field solve suspension."""
import numpy as np
import pops
import pytest

from examples.api_v02.continuation_capture_amr import build_continuation_capture_amr
from tests.python.support.amr_continuation_receipts import forest
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _verify(spec, runtime, world, step, recover):
    image = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    decoded = collective_call(world, lambda: forest(spec, image,
        rank=0 if world is None else int(world.rank)))
    metrics = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        assert all(row["passed"] for row in metrics), metrics
        assert [row["level"] for row in decoded["records"]] == [-1, 0, 1]
        for row in decoded["records"]:
            root = row["root"]
            assert root["time"] == runtime.time() == step*spec.dt
            assert root["step"] == step
            assert root["counter"] == step*(10 if recover else 1)*(2 if row["level"] == -1 else 1)
    return decoded


@pytest.mark.parametrize("recover", (False, True))
def test_assessed_child_scope_suspends_recovers_and_restarts(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, recover
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("continuation_capture_amr",
        builder=build_continuation_capture_amr, recover=recover)
    runtime = _bind(spec, artifact, world)
    report = collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    with collective_check(world):
        assert report.accepted_steps == 1 and report.rejected_steps == 0
    saved = _verify(spec, runtime, world, 1, recover)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"captured")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = _verify(spec, resumed, world, 1, recover)
    with collective_check(world):
        assert [row["image"] for row in saved["records"]] == [row["image"] for row in restored["records"]]
    for instance in (runtime, resumed):
        report = collective_call(world, lambda instance=instance:
            pops.step(instance, t_bound=spec.t_end, console=False))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
        _verify(spec, instance, world, 2, recover)
    for level in range(runtime.n_levels()):
        left = collective_call(world, lambda level=level: runtime.block_level_state_global("method", level))
        right = collective_call(world, lambda level=level: resumed.block_level_state_global("method", level))
        with collective_check(world):
            if left is not None:
                np.testing.assert_array_equal(left, right)
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("recovered_after_assessment", recover)
