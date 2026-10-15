"""Exact generated faces enter real AMR subcycling, reflux and rollback."""
import numpy as np
import pops
import pytest
from examples.api_v02.method_flux_correction_amr import build_t12_amr
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.public_native_profile import profile_gather_value
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def test_corrected_generated_faces_reflux_and_restart(tmp_path, isolated_native_cache,
        native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T12_AMR", builder=build_t12_amr)
    runtime = _bind(spec, artifact, world)
    report = collective_call(world, lambda: pops.run(runtime, t_end=spec.dt,
                                                     max_steps=1, console=False))
    rows = profile_gather_value(world, collective_call(world, lambda: spec.invariants(runtime)))
    with collective_check(world):
        owners = [row for row in rows if row.get("metrics_owner", True)]
        assert owners and all(row["passed"] for row in owners), rows
        assert all(row["active_counts"][0] and row["active_counts"][1] for row in owners)
        assert report.accepted_steps == 1 and report.rejected_steps == 0
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"corrected")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    for instance in (runtime, resumed):
        collective_call(world, lambda: pops.run(instance, t_end=spec.t_end,
                                                max_steps=2, console=False))
        rows = profile_gather_value(world, collective_call(world, lambda: spec.invariants(instance)))
        with collective_check(world):
            assert all(row["passed"] for row in rows), rows
    for level in range(runtime.n_levels()):
        left = collective_call(world, lambda: runtime.block_level_state_global("method", level))
        right = collective_call(world, lambda: resumed.block_level_state_global("method", level))
        with collective_check(world):
            if left is not None:
                np.testing.assert_array_equal(left, right)
    a = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    b = collective_call(world, lambda: bytes(resumed._executor.program_accepted_state()))
    with collective_check(world):
        assert a and a == b
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("composite_metrics", str(rows))


def test_parent_rejection_restores_generated_face_envelope(isolated_native_cache,
        native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T12_AMR_reject", builder=build_t12_amr,
                                     reject_parent=True)
    runtime = _bind(spec, artifact, world)
    levels = runtime.n_levels()
    before = [collective_call(world, lambda level=level:
        runtime.block_level_state_global("method", level)) for level in range(levels)]
    envelope = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    boxes = runtime.patch_boxes()
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=spec.dt,
                                                              max_steps=1, console=False))
    after = [collective_call(world, lambda level=level:
        runtime.block_level_state_global("method", level)) for level in range(levels)]
    restored = collective_call(world, lambda: bytes(runtime._executor.program_accepted_state()))
    with collective_check(world):
        assert failures and all(row and "reject_after_reflux" in row[1] for row in failures), failures
        assert runtime.time() == 0. and runtime.macro_step() == 0
        assert levels == 2 and runtime.n_levels() == levels and runtime.patch_boxes() == boxes
        assert envelope and envelope == restored
        for left, right in zip(before, after, strict=True):
            if left is not None:
                np.testing.assert_array_equal(left, right)
