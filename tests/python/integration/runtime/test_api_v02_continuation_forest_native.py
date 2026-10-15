"""Native PoPS forest acceptance, full checkpoint, and late rejection receivers."""
import numpy as np
import pops
import pytest

from examples.api_v02.continuation_forest_amr import build_continuation_forest_amr
from tests.python.support.amr_continuation_receipts import forest
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _images(runtime, world):
    return [collective_call(world, lambda level=level: runtime.block_level_state_global("method", level))
            for level in range(runtime.n_levels())]


def _accepted(runtime, world):
    return collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))


def _verify(spec, runtime, world, *, step, ratio):
    rank = 0 if world is None else int(world.rank)
    image = _accepted(runtime, world)
    decoded = collective_call(world, lambda: forest(spec, image, rank=rank))
    metrics = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        assert all(row["passed"] for row in metrics), metrics
        assert runtime.n_levels() == 2
        records = decoded["records"]
        assert [record["level"] for record in records] == [-1, 0, 1]
        parent = records[0]["root"]
        assert parent["step"] == step
        assert parent["time"] == runtime.time()
        assert parent["counter"] == (1+ratio)*step*(step+1)//2
        assert abs(parent["mass"]-1.5) < 3.e-13
        for record, substeps in zip(records[1:], (1, ratio), strict=True):
            child = record["root"]
            assert child["step"] == step and child["time"] == parent["time"]
            assert child["counter"] == step*substeps
            assert child["history"]["values"] == list(range(step, step+3))
            assert child["state"]["components"] == 1
        assert any(np.any(np.abs(item["payload"]) > 0) for item in decoded["faces"])
    return decoded


@pytest.mark.parametrize("ratio", (1, 2))
def test_complete_amr_forest_acceptance_checkpoint_and_restart(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, ratio
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("continuation_forest", builder=build_continuation_forest_amr,
                                     ratio=ratio, steps=3)
    runtime = _bind(spec, artifact, world)
    report = collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    with collective_check(world):
        assert report.accepted_steps == 1 and report.rejected_steps == 0
    saved = _verify(spec, runtime, world, step=1, ratio=ratio)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"forest")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = _verify(spec, resumed, world, step=1, ratio=ratio)
    with collective_check(world):
        assert [record["image"] for record in saved["records"]] == [
            record["image"] for record in restored["records"]]
    for candidate in (runtime, resumed):
        report = collective_call(world, lambda candidate=candidate:
            pops.run(candidate, t_end=spec.t_end, max_steps=2, console=False))
        with collective_check(world):
            assert report.accepted_steps == 2 and report.rejected_steps == 0
        _verify(spec, candidate, world, step=3, ratio=ratio)
    before, after = _images(runtime, world), _images(resumed, world)
    with collective_check(world):
        for left, right in zip(before, after, strict=True):
            if left is not None:
                np.testing.assert_array_equal(left, right)
        assert runtime.time() == resumed.time() == spec.t_end
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("temporal_ratio", ratio)


@pytest.mark.parametrize("rejection", ("reject_last_child", "reject_parent"))
def test_late_forest_rejection_restores_all_previously_accepted_images(
    isolated_native_cache, native_cxx, kokkos_root, record_property, rejection
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("continuation_forest_reject", builder=build_continuation_forest_amr,
                                     **{rejection: True})
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    _verify(spec, runtime, world, step=1, ratio=2)
    before, accepted = _images(runtime, world), _accepted(runtime, world)
    _, failures = collective_attempt(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    after, restored = _images(runtime, world), _accepted(runtime, world)
    with collective_check(world):
        assert failures
        assert runtime.time() == spec.dt and runtime.macro_step() == 1
        assert accepted == restored
        for left, right in zip(before, after, strict=True):
            if left is not None:
                np.testing.assert_array_equal(left, right)
    _verify(spec, runtime, world, step=1, ratio=2)
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("rejection", rejection)
