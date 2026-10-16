"""Producer-exact Field errors cross AMR recovery and complete-image restart."""
import numpy as np
import pops
import pytest

from examples.api_v02.continuation_capture_amr import build_continuation_capture_amr
from tests.python.support.amr_continuation_receipts import forest
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _image(runtime, world):
    return collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))


def _states(runtime, world):
    return [collective_call(world, lambda level=level:
        runtime.block_level_state_global("method", level)) for level in range(runtime.n_levels())]


def _verify(spec, runtime, world, step):
    image = _image(runtime, world)
    decoded = collective_call(world, lambda: forest(spec, image,
        rank=0 if world is None else int(world.rank)))
    with collective_check(world):
        assert [row["level"] for row in decoded["records"]] == [-1, 0, 1]
        assert decoded["records"][0]["root"]["counter"] == 20*step
        retained = []
        for row in decoded["records"][1:]:
            root = row["root"]
            assert root["counter"] == 10*step and root["step"] == step
            assert root["time"] == runtime.time() == step*spec.dt
            sample = root["retained"]
            assert sample["field_support"]["producer_level"] == 1
            assert sample["field_support"]["program_owner"] == 0
            for ordinal, payload in sample["patches"].items():
                (lo_x, hi_x, _), (lo_y, hi_y, _) = sample["layout"][ordinal]
                gx, gy = (axis[0] for axis in sample["axes"])
                nx, ny = hi_x-lo_x+1, hi_y-lo_y+1
                values = payload.reshape(1, ny+2*gy, nx+2*gx)[0, gy:gy+ny, gx:gx+nx]
                expected = np.where(np.arange(lo_x, hi_x+1)//2 >= 8, 2., 1.)
                np.testing.assert_array_equal(values, np.broadcast_to(expected, values.shape))
            retained.append(sample)
        # The coarse image retains the exact fine support instead of relabelling it.
        assert retained[0]["layout"] == retained[1]["layout"]
        assert retained[0]["field_support"] == retained[1]["field_support"]
        assert retained[0]["patches"].keys() == retained[1]["patches"].keys()
        for ordinal in retained[0]["patches"]:
            np.testing.assert_array_equal(retained[0]["patches"][ordinal], retained[1]["patches"][ordinal])
    return decoded


def test_fine_field_error_is_retained_opaque_on_coarse_and_survives_restart(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("fine_field_error_capsule",
        builder=build_continuation_capture_amr, field_payload=True)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    saved = _verify(spec, runtime, world, 1)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"fine_error")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = _verify(spec, resumed, world, 1)
    with collective_check(world):
        assert [row["image"] for row in saved["records"]] == [row["image"] for row in restored["records"]]
    for candidate in (runtime, resumed):
        collective_call(world, lambda candidate=candidate:
            pops.step(candidate, t_bound=spec.t_end, console=False))
        _verify(spec, candidate, world, 2)
    left, right = _states(runtime, world), _states(resumed, world)
    with collective_check(world):
        for a, b in zip(left, right, strict=True):
            if a is not None:
                np.testing.assert_array_equal(a, b)
    record_property("artifact", artifact.artifact_identity.hexdigest)


def test_foreign_field_error_numeric_read_rejects_and_restores_whole_macro_image(
    isolated_native_cache, native_cxx, kokkos_root
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("foreign_field_error_read",
        builder=build_continuation_capture_amr, field_payload=True, illegal_read=True)
    runtime = _bind(spec, artifact, world)
    before, states, time, step = _image(runtime, world), _states(runtime, world), runtime.time(), runtime.macro_step()
    _, failures = collective_attempt(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    after, next_states = _image(runtime, world), _states(runtime, world)
    with collective_check(world):
        assert failures and all(failures), failures
        assert all("producer support" in failure[1] for failure in failures), failures
        assert after == before and runtime.time() == time and runtime.macro_step() == step
        for a, b in zip(states, next_states, strict=True):
            if a is not None:
                np.testing.assert_array_equal(a, b)
