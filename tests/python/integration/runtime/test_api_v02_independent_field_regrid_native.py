"""An independent solved Field moves through Python remap and remains consumed by AB2."""
import numpy as np
import pops
import pytest

from examples.api_v02.continuation_independent_regrid import build_independent_continuation_regrid
from tests.python.support.amr_continuation_receipts import forest
from tests.python.support.api_v02_native import _bind, _compile, typed_failure_values
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _factor(step, dt=.1):
    # Independent closed two-step recurrence; never evaluates the authored DAG.
    older, current = 1., 1.
    for _ in range(step):
        older, current = current, (1.+1.5*dt)*current-.5*dt*older
    return current


def _field_error(field, level, factor, *, cells=(16, 8)):
    assert field["components"] == 1
    gx, gy = (axis[0] for axis in field["axes"])
    error = 0.
    for ordinal, payload in field["patches"].items():
        (lo_x, hi_x, _), (lo_y, hi_y, _) = field["layout"][ordinal]
        nx, ny = hi_x-lo_x+1, hi_y-lo_y+1
        values = payload.reshape(1, ny+2*gy, nx+2*gx)[0, gy:gy+ny, gx:gx+nx]
        coarse_x = np.arange(lo_x, hi_x+1)//(2**level)
        expected = factor*(1.+.5*np.sin(2*np.pi*(coarse_x+.5)/cells[0]))
        error = max(error, float(np.max(abs(values-expected[None, :]))))
    return error


def _accepted(runtime, world):
    return collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))


def _images(runtime, world):
    return [collective_call(world, lambda level=level: runtime.block_level_state_global("method", level))
            for level in range(runtime.n_levels())]


def _verify(spec, runtime, world, *, step, regridded):
    image = _accepted(runtime, world)
    rank = 0 if world is None else int(world.rank)
    decoded = collective_call(world, lambda: forest(spec, image, rank=rank))
    metrics = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        assert all(item["passed"] for item in metrics), metrics
        assert runtime.n_levels() == (2 if regridded else 1)
        records = decoded["records"]
        assert [record["level"] for record in records] == list(range(-1, runtime.n_levels()))
        parent = records[0]["root"]
        assert parent["step"] == step and parent["time"] == runtime.time()
        assert parent["counter"] == step
        assert parent["regrids"] >= int(regridded)
        if not regridded:
            assert parent["regrids"] == 0
        assert abs(parent["mass"]-_factor(step)) < 5.e-13
        for record in records[1:]:
            child, level = record["root"], record["level"]
            assert child["step"] == step and child["time"] == parent["time"]
            assert child["counter"] == step
            assert child["controls"]["values"] == [0, 1, 2]
            assert child["previous"]["layout"] == child["state"]["layout"]
            provider = child["previous"]["field_provider"]
            support = child["previous"]["field_support"]
            assert provider and support["field_provider"] == provider
            assert support["program_owner"] == -1 and support["producer_level"] == level
            assert support["topology_epoch"] <= record["epoch"]
            assert support["materialization_generation"] <= record["generation"]
            if regridded and step == 2:
                assert support["topology_epoch"] == record["epoch"]
                assert support["materialization_generation"] == record["generation"]
            assert support["origin_point"].startswith("pops.field-observation.point@1:sha256:")
            assert _field_error(child["state"], level, _factor(step)) < 5.e-13
            assert _field_error(child["previous"], level, _factor(max(0, step-1))) < 5.e-13
        supports = [row["root"]["previous"]["field_support"] for row in records[1:]]
        assert len({row["origin_point"] for row in supports}) == 1
    return decoded


def test_python_remap_moves_independent_solved_history_and_survives_restart(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("independent_field_regrid_ab2", builder=build_independent_continuation_regrid)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    _verify(spec, runtime, world, step=1, regridded=False)
    initial_boxes = runtime.patch_boxes()
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    saved = _verify(spec, runtime, world, step=2, regridded=True)
    with collective_check(world):
        assert runtime.patch_boxes() != initial_boxes, "the fixture must create actual new fine storage"
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"ab2_regrid")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = _verify(spec, resumed, world, step=2, regridded=True)
    with collective_check(world):
        assert [row["image"] for row in saved["records"]] == [row["image"] for row in restored["records"]]
    for candidate in (runtime, resumed):
        report = collective_call(world, lambda candidate=candidate:
            pops.run(candidate, t_end=spec.t_end, max_steps=1, console=False))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
        _verify(spec, candidate, world, step=3, regridded=True)
    left, right = _images(runtime, world), _images(resumed, world)
    with collective_check(world):
        for a, b in zip(left, right, strict=True):
            if a is not None:
                np.testing.assert_array_equal(a, b)
        assert runtime.time() == resumed.time() == spec.t_end
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("history", "independent FieldProblem phi=q: retained phi(n-1) remapped by Python requests and consumed by AB2")


def test_late_regrid_rejection_restores_independent_field_and_complete_forest(
    isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("independent_field_regrid_reject", builder=build_independent_continuation_regrid,
                                     reject_regrid_parent=True)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    _verify(spec, runtime, world, step=1, regridded=False)
    before, accepted, boxes = _images(runtime, world), _accepted(runtime, world), runtime.patch_boxes()
    error = None
    try:
        pops.step(runtime, t_bound=spec.t_end, console=False)
    except Exception as caught:
        error = caught
    after, restored = _images(runtime, world), _accepted(runtime, world)
    with collective_check(world):
        _, failures = typed_failure_values(error, world)
        assert all(item == {"tag": "assessment_rejected", "value": {"checks": {"stage": 71}}}
                   for item in failures), failures
        assert runtime.time() == spec.dt and runtime.macro_step() == 1
        assert runtime.patch_boxes() == boxes
        assert accepted == restored
        for left, right in zip(before, after, strict=True):
            if left is not None:
                np.testing.assert_array_equal(left, right)
    _verify(spec, runtime, world, step=1, regridded=False)
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("rejection", "late parent after child remap and assessment")
