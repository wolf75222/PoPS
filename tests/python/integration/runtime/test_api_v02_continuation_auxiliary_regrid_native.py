"""A scalar Field history with storage distinct from its owning AMR State."""
import numpy as np
import pops
import pytest

from examples.api_v02.continuation_regrid_auxiliary import build_continuation_regrid_auxiliary
from tests.python.support.amr_continuation_receipts import forest
from tests.python.support.api_v02_native import _bind
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _factor(step):
    previous, current = 1., 1.
    for _ in range(step):
        previous, current = current, 1.15*current-.05*previous
    return current


def _verify_field(field, *, level, step, components, ghosts):
    assert field["components"] == components
    assert [axis[0] for axis in field["axes"]] == [ghosts, ghosts]
    for ordinal, payload in field["patches"].items():
        (lo_x, hi_x, _), (lo_y, hi_y, _) = field["layout"][ordinal]
        nx, ny = hi_x-lo_x+1, hi_y-lo_y+1
        values = payload.reshape(components, ny+2*ghosts, nx+2*ghosts)
        valid = values[:, ghosts:ghosts+ny, ghosts:ghosts+nx]
        coarse_x = np.arange(lo_x, hi_x+1)//(2**level)
        reference = _factor(step)*(1.+.5*np.sin(2*np.pi*(coarse_x+.5)/16))
        for component in range(components):
            np.testing.assert_allclose(valid[component], np.broadcast_to(
                (component+1)*reference[None, :], (ny, nx)), rtol=0, atol=5.e-13)


def _verify(spec, runtime, world, step):
    raw = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    rank = 0 if world is None else int(world.rank)
    decoded = collective_call(world, lambda: forest(spec, raw, rank=rank))
    with collective_check(world):
        assert runtime.n_levels() == (1 if step == 1 else 2)
        root = decoded["records"][0]["root"]
        assert root["step"] == step and root["time"] == runtime.time()
        assert root["regrids"] >= int(step >= 2)
        assert abs(root["mass"]-_factor(step)) < 5.e-13
        for record in decoded["records"][1:]:
            child, level = record["root"], record["level"]
            assert child["time"] == root["time"] and child["step"] == step
            assert child["controls"]["values"] == [0, 1, 2]
            assert child["state"]["layout"] == child["previous"]["layout"]
            _verify_field(child["state"], level=level, step=step, components=2, ghosts=2)
            _verify_field(child["previous"], level=level, step=max(0, step-1), components=1, ghosts=1)
    return decoded


def test_explicit_target_storage_remaps_and_consumes_a_different_field_space(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property
):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="API0.2 independently shaped retained AMR Field")
    spec, providers = collective_call(world, build_continuation_regrid_auxiliary)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
        layout=spec.layout, layout_providers=providers))
    artifact = compile_profile_plan_once(world, plan, route="explicit target Field allocation",
                                         compile_artifact=pops.compile)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    _verify(spec, runtime, world, 1)
    old_boxes = runtime.patch_boxes()
    collective_call(world, lambda: pops.step(runtime, t_bound=spec.t_end, console=False))
    saved = _verify(spec, runtime, world, 2)
    with collective_check(world):
        assert runtime.patch_boxes() != old_boxes
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"distinct_field")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    restored = _verify(spec, resumed, world, 2)
    with collective_check(world):
        assert [row["image"] for row in saved["records"]] == [row["image"] for row in restored["records"]]
    for candidate in (runtime, resumed):
        report = collective_call(world, lambda candidate=candidate:
            pops.run(candidate, t_end=spec.t_end, max_steps=1, console=False))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
        _verify(spec, candidate, world, 3)
    left = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    right = collective_call(world, lambda: bytes(resumed._executor.program_accepted_state()))
    # Epoch/provenance may be requalified at restart. Compare scientific images,
    # which carry the exact native values and explicit Field storage contracts.
    a = collective_call(world, lambda: forest(spec, left, rank=0 if world is None else int(world.rank)))
    b = collective_call(world, lambda: forest(spec, right, rank=0 if world is None else int(world.rank)))
    with collective_check(world):
        assert [row["image"] for row in a["records"]] == [row["image"] for row in b["records"]]
        assert runtime.time() == resumed.time() == spec.t_end
    record_property("artifact", artifact.artifact_identity.hexdigest)
    record_property("storage", "State(components=2,ghosts=2), retained Field(components=1,ghosts=1)")
