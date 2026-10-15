"""Real two-trace generated component, coupled AMR, affine publication and restart."""
import numpy as np
import pops
import pytest
from examples.api_v02.method_relaxed_multiblock import build_t10_multiblock, transport_trace_flux
from examples.api_v02.method_amr_reception import composite_base_projection
from tests.python.support.api_v02_native import _bind
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import native_profile, compile_profile_plan_once
from tests.python.support.relaxed_flux_receipts import assert_scaled, assert_interface_scaled

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _compile(*, ratio=2, relaxed=True, reject_parent=False):
    _native, world = native_profile(dimension=2, scope="T10 two-trace affine AMR")
    component = collective_call(world, transport_trace_flux)
    spec = collective_call(world, lambda: build_t10_multiblock(component,
        temporal_ratio=ratio, relaxed=relaxed, reject_parent=reject_parent))
    plan = collective_call(world, lambda: pops.resolve(pops.validate(spec.case),
        layout=spec.layout, components=(component,)))
    artifact = compile_profile_plan_once(world, plan, route="T10 two-trace affine AMR",
                                        compile_artifact=pops.compile)
    return spec, artifact, world


def _states(runtime, world):
    return {name: [collective_call(world, lambda name=name, level=level:
        runtime.block_level_state_global(name, level)) for level in range(runtime.n_levels())]
        for name in ("left", "right")}


def _mass(states, boxes):
    total = 0.
    for values in states.values():
        arrays = [np.asarray(value).reshape(16*2**level, 32*2**level)
                  for level, value in enumerate(values)]
        projection = composite_base_projection(arrays, boxes, cells=(32, 16))
        assert all(count > 0 for count in projection["active_counts"])
        total += float(projection["means"].mean())
    return total


@pytest.mark.parametrize("ratio", (1, 2))
def test_affine_multiblock_transports_native_interface_measures(ratio, tmp_path,
        isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    base_spec, base_artifact, world = _compile(ratio=ratio, relaxed=False)
    spec, artifact, world = _compile(ratio=ratio)
    base, runtime = _bind(base_spec, base_artifact, world), _bind(spec, artifact, world)
    old = _states(runtime, world)
    boxes = runtime.patch_boxes()
    with collective_check(world):
        assert runtime.n_levels() == base.n_levels() == 2
        assert boxes == base.patch_boxes()
    for instance in (base, runtime):
        report = collective_call(world, lambda: pops.step(instance,
            t_bound=instance.time()+1., console=False))
        with collective_check(world):
            assert report.accepted_steps == 1 and report.rejected_steps == 0
            assert report.stop_reason is pops.RunStopReason.ACCEPTED_STEP_LIMIT_REACHED
    provisional, accepted = _states(base, world), _states(runtime, world)
    with collective_check(world):
        assert runtime.time() == .8 and runtime.macro_step() == 1
        if old["left"][0] is not None:
            for name in old:
                for a, b, c in zip(old[name], provisional[name], accepted[name], strict=True):
                    a, b = np.asarray(a), np.asarray(b)
                    np.testing.assert_allclose(c, a+.8*(b-a), rtol=0, atol=3.e-14, equal_nan=True)
            # Exterior inflow is zero; the right exterior transports constant .1.
            # Shared interface exchange and internal coarse/fine corrections cancel globally.
            assert abs(_mass(provisional, boxes)-_mass(old, boxes)+.0002) < 3.e-13
            assert abs(_mass(accepted, boxes)-_mass(old, boxes)+.00016) < 3.e-13
    original = collective_call(world, lambda: bytes(base._executor.program_accepted_state()))
    image = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world):
        assert_scaled(original, image, .8)
        assert_interface_scaled(original, image, .8)
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: runtime.checkpoint(str(directory/"two-block-affine")))
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
        for name in a:
            for left, right in zip(a[name], b[name], strict=True):
                if left is not None: np.testing.assert_array_equal(left, right)
    a = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    b = collective_call(world, lambda: bytes(resumed._executor.program_accepted_state()))
    with collective_check(world): assert a == b
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)


def test_rejection_after_multiblock_affine_publication_restores_interface_ledger(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(reject_parent=True)
    runtime = _bind(spec, artifact, world)
    before = _states(runtime, world)
    image = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=1., max_steps=1, console=False))
    after = _states(runtime, world)
    restored = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world):
        assert failures and all(row and "reject_two_block_parent" in row[1] for row in failures), failures
        assert runtime.time() == 0 and runtime.macro_step() == 0 and restored == image
        for name in before:
            for left, right in zip(before[name], after[name], strict=True):
                if left is not None: np.testing.assert_array_equal(left, right)
