"""A FieldProblem value keeps its scientific owner, old time, and producing level."""
import numpy as np
import pops
import pytest
from pops._native_collectives import allgather_value
from examples.api_v02.continuation_capture_amr import build_continuation_capture_amr
from tests.python.support.amr_continuation_receipts import forest
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.integration.runtime.test_api_v02_field_error_capsule_native import _image, _states

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _verify(spec, runtime, world, step):
    image = _image(runtime, world)
    decoded = collective_call(world, lambda: forest(spec, image,
        rank=0 if world is None else int(world.rank)))
    report = collective_call(world, runtime.program_report)
    local_sum = 0.
    with collective_check(world):
        assert [row["level"] for row in decoded["records"]] == [-1, 0, 1]
        assert decoded["records"][0]["root"]["counter"] == 20*step
        samples = []
        for row in decoded["records"][1:]:
            root = row["root"]
            assert root["counter"] == 10*step and root["step"] == step
            sample = root["retained"]
            support = sample["field_support"]
            assert support["program_owner"] == -1 and support["producer_level"] == 1
            assert support["field_provider"] == sample["field_provider"]
            assert "field_problem" in sample["field_provider"]
            assert support["origin_point"].startswith("pops.field-observation.point@1:sha256:")
            samples.append(sample)
        assert samples[0]["field_support"] == samples[1]["field_support"]
        assert samples[0]["layout"] == samples[1]["layout"]
        assert samples[0]["patches"].keys() == samples[1]["patches"].keys()
        for ordinal, payload in samples[1]["patches"].items():
            np.testing.assert_array_equal(payload, samples[0]["patches"][ordinal])
            (lo_x, hi_x, _), (lo_y, hi_y, _) = samples[1]["layout"][ordinal]
            gx, gy = (axis[0] for axis in samples[1]["axes"])
            nx, ny = hi_x-lo_x+1, hi_y-lo_y+1
            values = payload.reshape(1, ny+2*gy, nx+2*gx)[0, gy:gy+ny, gx:gx+nx]
            assert np.isfinite(values).all() and (values > 0).all() and (values < 1.5).all()
            local_sum += float(values.sum())
    sums = collective_call(world, lambda: [local_sum] if world is None else allgather_value(world, local_sum))
    total = sum(sums)
    with collective_check(world):
        assert report.diagnostics["capsule_sample"] == pytest.approx(total, rel=2e-13, abs=2e-13)
    return decoded, total, report


def test_independent_field_retains_old_sample_through_capture_and_restart(
        tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("independent_field_continuation",
        builder=build_continuation_capture_amr, field_payload=True, independent_field=True)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    saved, old_sum, _ = _verify(spec, runtime, world, 1)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"independent_field")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored_image = _image(resumed, world)
    restored = collective_call(world, lambda: forest(spec, restored_image,
        rank=0 if world is None else int(world.rank)))
    with collective_check(world):
        assert [row["image"] for row in restored["records"]] == [row["image"] for row in saved["records"]]
    results = []
    for candidate in (runtime, resumed):
        collective_call(world, lambda candidate=candidate: pops.step(candidate, t_bound=spec.t_end, console=False))
        decoded, total, report = _verify(spec, candidate, world, 2)
        with collective_check(world):
            assert decoded["records"][0]["root"]["previous"] == pytest.approx(old_sum, rel=2e-13, abs=2e-13)
            assert total > old_sum + 1e-5
            assert decoded["records"][2]["root"]["retained"]["field_support"]["origin_point"] != saved["records"][2]["root"]["retained"]["field_support"]["origin_point"]
        results.append(decoded)
    with collective_check(world):
        assert [row["image"] for row in results[0]["records"]] == [row["image"] for row in results[1]["records"]]
    record_property("artifact", artifact.artifact_identity.hexdigest)


def test_independent_field_foreign_read_restores_every_accepted_leaf(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("independent_field_foreign_read",
        builder=build_continuation_capture_amr, field_payload=True, independent_field=True, illegal_read=True)
    runtime = _bind(spec, artifact, world)
    before, states = _image(runtime, world), _states(runtime, world)
    _, errors = collective_attempt(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    after, current = _image(runtime, world), _states(runtime, world)
    with collective_check(world):
        assert errors and all(errors), errors
        assert all("producer support" in error[1] for error in errors), errors
        assert after == before and runtime.time() == 0 and runtime.macro_step() == 0
        for a, b in zip(states, current, strict=True):
            if a is not None:
                np.testing.assert_array_equal(a, b)
