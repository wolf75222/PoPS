"""Authored retained-history matrix after the real refluxed AMR macro step."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_relaxed_history import build_t10_history
from tests.python.integration.runtime.test_api_v02_history_transfer_native import (
    _image, _same, _sample_rows,
)
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import collective_call, collective_check, collective_attempt
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.history_flux_receipts import accepted_state_history_flux

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 900


def _accepted_step(runtime, world):
    report = collective_call(world, lambda: pops.step(
        runtime, t_bound=runtime.time()+1., console=False))
    with collective_check(world):
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        assert report.stop_reason is pops.RunStopReason.ACCEPTED_STEP_LIMIT_REACHED


def _verify_matrix(before, after, world):
    start, duration = before["time"], after["time"]-before["time"]
    with collective_check(world):
        assert duration > 0 and len(after["states"]) == 2
        assert set(after["metadata"]) == {(name, level) for name in
            ("method.state", "retained_transport") for level in (0, 1)}
        for level in after["states"]:
            raw, intervals = after["metadata"]["method.state", level]
            actual_level, samples = _sample_rows(raw)
            assert actual_level == level and len(samples) == 3
            assert all(row["kind"] == 3 and any(row["digest"]) for row in samples)
            np.testing.assert_allclose([row["time"] for row in samples],
                [start, start+.5*duration, start], rtol=0, atol=2.e-14)
            np.testing.assert_allclose(intervals, [.5*duration]*3, rtol=0, atol=2.e-14)
            for slot, sample in enumerate(samples):
                assert sample["interval"] == intervals[slot]
                actual = after["rings"]["method.state", level, slot]
                if actual is None:
                    continue
                initial, accepted = (np.asarray(image["states"][level])
                                     for image in (before, after))
                expected = .5*(initial+accepted) if slot == 1 else initial
                np.testing.assert_allclose(actual, expected, rtol=0, atol=2.e-13)
            if ("method.state", level) in before["metadata"]:
                assert raw != before["metadata"]["method.state", level][0]
            _, rhs_samples = _sample_rows(after["metadata"]["retained_transport", level][0])
            assert all(row["kind"] == 3 and any(row["digest"]) for row in rhs_samples)
            first = after["rings"]["retained_transport", level, 0]
            if first is not None:
                first = np.asarray(first)
                assert float(np.max(abs(first))) > 0
                for slot in (1, 2):
                    np.testing.assert_array_equal(first, after["rings"]["retained_transport", level, slot])
        flux_rings, raw_flux = accepted_state_history_flux(after["accepted"])
        rings = [row for row in flux_rings if row["name"].endswith("retained_transport")]
        assert raw_flux and len(rings) == 2
        assert all(len(row["slots"]) == 3 and all(len(slot) == 1 for slot in row["slots"])
                   for row in rings)
        assert all(basis["faces"] for row in rings for slot in row["slots"] for basis in slot)


@pytest.mark.parametrize("ratio", (1, 2))
def test_affine_history_uses_macro_entry_and_accepted_state_then_restarts(
        ratio, tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T10_affine_history", builder=build_t10_history,
                                     temporal_ratio=ratio)
    runtime = _bind(spec, artifact, world)
    before = _image(runtime, world)
    _accepted_step(runtime, world)
    accepted = _image(runtime, world)
    _verify_matrix(before, accepted, world)
    report = collective_call(world, runtime.program_report)
    with collective_check(world):
        # This scalar was computed by a generated native kernel reading lag1
        # after publication, not by the host-side inspection oracle.
        observed = report.diagnostics["rrk.history_midpoint_x"]
        if before["states"][0] is not None:
            initial = np.asarray(before["states"][0]).reshape(3, 16, 32)
            final = np.asarray(accepted["states"][0]).reshape(3, 16, 32)
            expected = .5*(initial[0]+final[0])
            assert abs(observed-float(np.min(expected))) < 2.e-13
            rhs = np.asarray(accepted["rings"]["retained_transport", 0, 1]).reshape(3, 16, 32)
            assert abs(report.diagnostics["rrk.retained_rhs_minimum"]-float(np.min(rhs[2]))) < 2.e-13
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory/"relaxed_history")))
    resumed = _bind(spec, artifact, world)
    collective_call(world, lambda: resumed.restart(checkpoint))
    _same(accepted, _image(resumed, world), world)
    for active in (runtime, resumed):
        _accepted_step(active, world)
        _verify_matrix(accepted, _image(active, world), world)
    _same(_image(runtime, world), _image(resumed, world), world)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("mpi_ranks", 1 if world is None else int(world.size))


@pytest.mark.parametrize("failure", ("invalid_history", "reject_parent"))
def test_affine_history_invalid_matrix_or_late_rejection_rolls_back_every_authority(
        failure, isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("T10_affine_history_rollback", builder=build_t10_history,
                                     temporal_ratio=2, **{failure: True})
    runtime = _bind(spec, artifact, world)
    before = _image(runtime, world)
    _, errors = collective_attempt(world, lambda: pops.step(runtime, t_bound=1., console=False))
    with collective_check(world):
        expected = "invalid target window" if failure == "invalid_history" else "reject_relaxed_parent"
        assert errors and all(error and expected in error[1] for error in errors), errors
    _same(before, _image(runtime, world), world)
