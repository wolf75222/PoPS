"""Independent Field history is executed, remapped, restarted and rolled back by PoPS."""

import numpy as np
import pops
import pytest
from examples.api_v02.independent_field_history import build_independent_history
from examples.api_v02.method_temporal_field import build_temporal_field
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import (
    collective_call,
    collective_check,
    collective_attempt,
)
from tests.python.support.integral_state_receipts import collective_directory
from tests.python.support.history_flux_receipts import accepted_state_history_flux
from tests.python.integration.runtime.test_api_v02_history_transfer_native import _image, _same
from tests.python.integration.runtime.test_api_v02_output_history_remap_native import (
    _check_projected_observation,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200


def _uniform_snapshot(runtime, world):
    state = collective_call(world, lambda: runtime.state_global("method"))
    ring = [
        collective_call(world, lambda slot=slot: runtime.history_global("potential", slot))
        for slot in range(3)
    ]
    metadata = collective_call(world, runtime._executor._checkpoint_program_exchanges)
    return state, ring, metadata, runtime.time(), runtime.macro_step()


def _uniform_equal(left, right, world):
    with collective_check(world):
        assert left[2:] == right[2:]
        for a, b in zip([left[0], *left[1]], [right[0], *right[1]], strict=True):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_array_equal(a, b)


def test_independent_uniform_history_reads_old_provider_and_restarts(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile("independent_history", builder=build_independent_history)
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=0.06, max_steps=3, console=False))
    metrics = collective_call(world, lambda: spec.invariants(runtime))
    before = _uniform_snapshot(runtime, world)
    with collective_check(world):
        assert metrics["passed"], metrics
        assert before[2].startswith(b"POPSEX06")
        assert b"field_problem" in before[2], (
            "retained metadata contains the actual independent provider"
        )
        if before[0] is not None:
            np.testing.assert_allclose(before[1][1], 0.52, rtol=0, atol=3.0e-12)
            np.testing.assert_allclose(before[1][2], 0.51, rtol=0, atol=3.0e-12)
            assert not np.array_equal(before[1][1], before[1][2]), "CopyCurrent cannot pass"
    directory = collective_directory(world, tmp_path)
    path = collective_call(
        world, lambda: runtime.checkpoint(str(directory / "independent-history"))
    )
    replay = _bind(spec, artifact, world)
    collective_call(world, lambda: replay.restart(path))
    _uniform_equal(before, _uniform_snapshot(replay, world), world)
    for candidate in (runtime, replay):
        collective_call(
            world,
            lambda candidate=candidate: pops.run(candidate, t_end=0.08, max_steps=1, console=False),
        )
        metrics = collective_call(world, lambda candidate=candidate: spec.invariants(candidate))
        with collective_check(world):
            assert metrics["passed"], metrics
    _uniform_equal(_uniform_snapshot(runtime, world), _uniform_snapshot(replay, world), world)


def _metadata(image):
    decoded = {}
    accepted_state_history_flux(image["accepted"], metadata_output=decoded)
    rows = [slots for key, slots in decoded["fields"].items() if key.endswith("observed")]
    assert len(rows) == 2
    for slots in rows:
        assert all(
            row["support"]
            and row["support"]["program_owner"] == -1
            and row["support"]["field_provider"]
            and row["representation"] is None
            for row in slots
        )
        assert len({row["support"]["origin_point"] for row in slots}) == 3
    return decoded


def _check_read_sum(image, world):
    with collective_check(world):
        if image["states"][0] is None:
            return
        states = [
            np.asarray(image["states"][level]).reshape(3, 16 * 2**level, 16 * 2**level)
            for level in range(2)
        ]
        fine = states[1][1] > 0
        for level, state in enumerate(states):
            live = state[1] > 0
            if level == 0:
                live &= ~fine.reshape(16, 2, 16, 2).any(axis=(1, 3))
            assert live.any()
            previous = np.asarray(image["rings"]["observed", level, 2])
            expected = float(previous.sum())
            # The PDE oracle checks these retained values independently; this
            # second assertion checks the actual History reader/reduction consumer.
            np.testing.assert_allclose(state[2][live], expected, rtol=0, atol=3.0e-9)


@pytest.mark.parametrize("reject", (False, True))
def test_independent_amr_history_keeps_provider_through_authored_regrid_and_restart(
    reject, tmp_path, isolated_native_cache, native_cxx, kokkos_root
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(
        "independent_amr_history",
        builder=build_temporal_field,
        regrid_observation=True,
        retained_history_read=True,
        reject_parent=reject,
    )
    runtime = _bind(spec, artifact, world)
    patches = tuple(runtime.patch_boxes())
    collective_call(world, lambda: pops.run(runtime, t_end=0.16, max_steps=4, console=False))
    before = _image(runtime, world)
    with collective_check(world):
        assert before["patches"] != patches
        _metadata(before)
    _check_projected_observation(before, world)
    directory = collective_directory(world, tmp_path)
    path = collective_call(
        world, lambda: runtime.checkpoint(str(directory / "independent-amr-history"))
    )
    replay = _bind(spec, artifact, world)
    collective_call(world, lambda: replay.restart(path))
    _same(before, _image(replay, world), world)
    for candidate in (runtime, replay):
        collective_call(
            world,
            lambda candidate=candidate: pops.run(candidate, t_end=0.18, max_steps=1, console=False),
        )
    after = _image(runtime, world)
    _same(after, _image(replay, world), world)
    _check_read_sum(after, world)
    if reject:
        _, failures = collective_attempt(
            world, lambda: pops.run(runtime, t_end=0.20, max_steps=1, console=False)
        )
        with collective_check(world):
            assert failures and all(failures), failures
        _same(after, _image(runtime, world), world)
