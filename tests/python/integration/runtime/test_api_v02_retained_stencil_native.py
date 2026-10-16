"""Actual C++ AB2 with retained face evaluations, full restart and atomic rejection."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_retained_stencil import build_retained_stencil, initial_state, reference
from tests.python.integration.runtime.test_api_v02_scoped_stencil_native import _receipts
from tests.python.support.api_v02_native import _bind, _compile, execute_case
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _check_faces(world, records, *, steps):
    initial = initial_state()
    _, changes, terms = reference(initial, steps=steps)
    with collective_check(world):
        assert len(records) == 16*4*4*len(terms[-1])
        increment = np.zeros_like(initial)
        observed = {}
        for row in records:
            cell, axis, side = row["quadrature_identity"].split("/")
            i, j = map(int, cell.split(":")[1:])
            assert row["multiplicity"] == 1
            increment[0, j, i] += row["integrated_amount"]
            if axis == "axis:0" and side == "side:0":
                key = (row["evaluation_context"], row["temporal_weight"])
                observed.setdefault(key, np.zeros_like(initial))[0, j, i] = row["numerical_flux"]
        assert len(observed) == len(terms[-1])
        for weight, expected_faces in terms[-1]:
            candidates = [faces for (_, actual_weight), faces in observed.items() if actual_weight == weight]
            assert len(candidates) == 1, "each source evaluation retains its signed AB2 weight"
            np.testing.assert_allclose(candidates[0], expected_faces, rtol=0, atol=4.e-14)
        np.testing.assert_allclose(increment, changes[-1]/64, rtol=0, atol=4.e-17)
        assert np.max(abs(changes[-1])) > .005, "no-op must fail the recurrence"


@pytest.mark.parametrize("retention", ("history", "continuation"))
def test_retained_ab2_faces_restart_without_replaying_accepted_origin(retention, tmp_path,
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="retained AB2 "+retention)
    spec = collective_call(world, lambda: build_retained_stencil(retention=retention, steps=2))
    runtime, artifact, _ = execute_case(lambda: spec, world, "retained AB2 "+retention)
    _check_faces(world, _receipts(world, runtime), steps=2)
    if retention == "history":
        image = collective_call(world, runtime._executor._checkpoint_program_exchanges)
        with collective_check(world):
            assert image.startswith(b"POPSEX05")
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: runtime.checkpoint(str(directory / ("AB2-"+retention))))
    replay = _bind(spec, artifact, world)
    collective_call(world, lambda: replay.restart(path))
    for candidate in (runtime, replay):
        collective_call(world, lambda candidate=candidate: pops.run(candidate, t_end=.03, max_steps=1, console=False))
    left = collective_call(world, lambda: runtime.state_global("method"))
    right = collective_call(world, lambda: replay.state_global("method"))
    left_records, right_records = _receipts(world, runtime), _receipts(world, replay)
    _check_faces(world, left_records, steps=3)
    expected, _, _ = reference(initial_state(), steps=3)
    with collective_check(world):
        assert left_records == right_records
        if left is not None:
            np.testing.assert_array_equal(left, right)
            np.testing.assert_allclose(np.asarray(left).reshape(expected.shape), expected, rtol=0, atol=4.e-14)
    if retention == "continuation":
        lhs = collective_call(world, runtime._executor._s._checkpoint_continuation_image)
        rhs = collective_call(world, replay._executor._s._checkpoint_continuation_image)
        with collective_check(world):
            assert lhs == rhs, "complete retained Field metadata and accepted State origins restart exactly"


@pytest.mark.parametrize("retention", ("history", "continuation"))
def test_retained_ab2_parent_rejection_restores_fields_history_and_mailbox(retention,
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="retained AB2 rollback "+retention)
    runtime, _, _ = execute_case(lambda: build_retained_stencil(retention=retention, steps=1, reject_after=1),
                                 world, "retained AB2 rollback "+retention)
    before = collective_call(world, lambda: runtime.state_global("method"))
    metadata = collective_call(world, runtime._executor._checkpoint_program_exchanges)
    continuation = (collective_call(world, runtime._executor._s._checkpoint_continuation_image)
                    if retention == "continuation" else None)
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=.02, max_steps=1, console=False))
    after = collective_call(world, lambda: runtime.state_global("method"))
    restored = collective_call(world, runtime._executor._checkpoint_program_exchanges)
    restored_continuation = (collective_call(world, runtime._executor._s._checkpoint_continuation_image)
                             if retention == "continuation" else None)
    with collective_check(world):
        assert failures and all(failures)
        assert metadata == restored and continuation == restored_continuation
        assert runtime.time() == .01 and runtime.macro_step() == 1
        if before is not None:
            np.testing.assert_array_equal(before, after)


def test_retained_face_budget_rejects_before_any_accepted_publication(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("retained AB2 insufficient budget",
        builder=lambda: build_retained_stencil(steps=1, basis_capacity=1))
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
    before = collective_call(world, lambda: runtime.state_global("method"))
    metadata = collective_call(world, runtime._executor._checkpoint_program_exchanges)
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=.02, max_steps=1, console=False))
    after = collective_call(world, lambda: runtime.state_global("method"))
    restored = collective_call(world, runtime._executor._checkpoint_program_exchanges)
    with collective_check(world):
        assert failures and all(failures)
        assert all("budget" in error[1].lower() or "capacity" in error[1].lower() for error in failures), failures
        assert metadata == restored
        assert runtime.time() == .01 and runtime.macro_step() == 1
        if before is not None:
            np.testing.assert_array_equal(before, after)
