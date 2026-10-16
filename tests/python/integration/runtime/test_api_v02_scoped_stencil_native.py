"""Scoped face values, actual exchange records and rollback on native PoPS."""
import numpy as np
import pops
import pytest

from examples.api_v02.method_scoped_stencil import build_scoped_stencil, reference_step
from tests.python.support.api_v02_native import _bind, _compile, execute_case
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.public_native_profile import native_profile, profile_gather_value

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _initial(cells=(16, 4)):
    x, y = np.meshgrid((np.arange(cells[0]) + .5) / cells[0],
                       (np.arange(cells[1]) + .5) / cells[1])
    return (1 + .2*np.cos(2*np.pi*x) + .1*np.sin(2*np.pi*y))[None]


def _receipts(world, runtime):
    shards = profile_gather_value(world, collective_call(
        world, runtime._executor._program_exchange_records))
    result = {}
    with collective_check(world):
        for shard in shards:
            for row in shard:
                key = tuple(row[name] for name in ("operation_identity", "occurrence_identity",
                                                  "evaluation_context", "quadrature_identity"))
                assert key not in result or result[key] == row, "conflicting MPI face replicas"
                result[key] = row
    return tuple(result.values())


@pytest.mark.parametrize("mode,positive", (("branch", True), ("branch", False), ("fold", True)))
def test_scoped_faces_follow_executed_values_and_restart(mode, positive, tmp_path,
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="scoped face expression")
    def builder():
        return build_scoped_stencil(mode=mode, positive=positive, dead_invalid=True)
    spec = collective_call(world, builder)
    runtime, artifact, _ = execute_case(lambda: spec, world, "scoped face " + mode + str(positive))
    records = _receipts(world, runtime)
    actual = collective_call(world, lambda: runtime.state_global("method"))
    initial = _initial()
    iterations = 2 if mode == "fold" else 1
    expected, faces = reference_step(initial, dt=.01, dx=1/16,
                                    velocity=1. if positive else -.6, iterations=iterations)
    with collective_check(world):
        assert len(records) == 16 * 4 * 4 * iterations
        groups = {}
        increments = np.zeros_like(initial)
        for row in records:
            groups.setdefault(row["evaluation_context"], []).append(row)
            cell, axis_token, side_token = row["quadrature_identity"].split("/")
            i, j = map(int, cell.split(":")[1:])
            axis, side = int(axis_token.split(":")[1]), int(side_token.split(":")[1])
            assert row["temporal_weight"] == .01 / iterations
            assert row["multiplicity"] == 1
            assert row["orientation"] == (1 if side == 0 else -1)
            assert row["face_measure"] == (1/4 if axis == 0 else 1/16)
            increments[0, j, i] += row["integrated_amount"]
        assert len(groups) == iterations, "each invocation needs its own retained evaluation"
        observed = []
        for group in groups.values():
            snapshot = np.zeros_like(initial)
            for row in group:
                cell, axis, side = row["quadrature_identity"].split("/")
                if axis != "axis:0" or side != "side:0":
                    continue
                i, j = map(int, cell.split(":")[1:])
                snapshot[0, j, i] = row["numerical_flux"]
            observed.append(snapshot)
        for expected_faces in faces:
            assert min(np.max(abs(item-expected_faces)) for item in observed) < 3.e-14
        np.testing.assert_allclose(increments, (expected-initial)/64, rtol=0, atol=3.e-17)
        if actual is not None:
            np.testing.assert_allclose(np.asarray(actual).reshape(initial.shape), expected, rtol=0, atol=3.e-14)
    directory = collective_directory(world, tmp_path)
    path = collective_call(world, lambda: runtime.checkpoint(str(directory / "scoped")))
    replay = _bind(spec, artifact, world)
    collective_call(world, lambda: replay.restart(path))
    for candidate in (runtime, replay):
        collective_call(world, lambda candidate=candidate: pops.run(candidate, t_end=.02, max_steps=1, console=False))
    left = collective_call(world, lambda: runtime.state_global("method"))
    right = collective_call(world, lambda: replay.state_global("method"))
    left_records, right_records = _receipts(world, runtime), _receipts(world, replay)
    with collective_check(world):
        if left is not None:
            np.testing.assert_array_equal(left, right)
        assert left_records == right_records


def test_scoped_rejection_restores_state_and_exact_exchange_mailbox(tmp_path,
        isolated_native_cache, native_cxx, kokkos_root):
    del tmp_path, isolated_native_cache, native_cxx, kokkos_root
    _, world = native_profile(dimension=2, scope="scoped exchange rollback")
    runtime, _, _ = execute_case(lambda: build_scoped_stencil(mode="fold", reject_second=True),
                                 world, "scoped rollback")
    before = collective_call(world, lambda: runtime.state_global("method"))
    mailbox = collective_call(world, runtime._executor._checkpoint_program_exchanges)
    _, failures = collective_attempt(world, lambda: pops.run(runtime, t_end=.02, max_steps=1, console=False))
    after = collective_call(world, lambda: runtime.state_global("method"))
    restored = collective_call(world, runtime._executor._checkpoint_program_exchanges)
    with collective_check(world):
        assert failures and all(failures)
        assert all("reject" in error[1].lower() for error in failures)
        assert runtime.time() == .01 and runtime.macro_step() == 1
        assert mailbox == restored
        if before is not None:
            np.testing.assert_array_equal(before, after)


def test_nonaffine_write_cannot_discard_a_scoped_exchange_expression(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("scoped_stencil_nonaffine",
        builder=lambda: build_scoped_stencil(mode="branch", nonaffine=True))
    runtime = _bind(spec, artifact, world)
    before = collective_call(world, lambda: runtime.state_global("method"))
    _, failures = collective_attempt(world,
        lambda: pops.run(runtime, t_end=.01, max_steps=1, console=False))
    after = collective_call(world, lambda: runtime.state_global("method"))
    records = collective_call(world, runtime._executor._program_exchange_records)
    with collective_check(world):
        assert failures and all(failures)
        assert all(any(word in error[1].lower() for word in ("stencil", "provenance", "affine"))
                   for error in failures), failures
        assert runtime.time() == 0 and runtime.macro_step() == 0
        assert not records
        if before is not None:
            np.testing.assert_array_equal(before, after)
