"""Real temporal reconstruction composed with a suspended composite solve."""

import numpy as np
import pops
import pytest

from examples.api_v02.method_temporal_field import build_temporal_field
from tests.python.support.api_v02_native import _bind, _compile
from tests.python.support.collective_checks import (
    collective_attempt,
    collective_call,
    collective_check,
)
from tests.python.support.integral_state_receipts import collective_directory

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]
POPS_PROCESS_TIMEOUT = 1200
_HISTORIES = ("observed", "observed_before_barrier", "observed_after_barrier", "synchronized")


def _image(runtime, world, *, initialized_histories=True):
    return dict(
        accepted=collective_call(world, lambda: bytes(runtime._executor.program_accepted_state())),
        time=runtime.time(),
        step=runtime.macro_step(),
        patches=tuple(runtime.patch_boxes()),
        states=[
            collective_call(
                world, lambda level=level: runtime.block_level_state_global("method", level)
            )
            for level in range(runtime.n_levels())
        ],
        histories={
            name: [
                collective_call(
                    world, lambda name=name, level=level: runtime.history_global(name, level, 1)
                )
                for level in range(runtime.n_levels())
            ]
            for name in _HISTORIES if initialized_histories
        },
    )


def _same(left, right, world):
    with collective_check(world):
        for key in ("accepted", "time", "step", "patches"):
            assert left[key] == right[key], key
        for x, y in zip(left["states"], right["states"], strict=True):
            if x is None:
                assert y is None
            else:
                np.testing.assert_array_equal(x, y)
        assert left["histories"].keys() == right["histories"].keys()
        for name in left["histories"]:
            for x, y in zip(left["histories"][name], right["histories"][name], strict=True):
                if x is None:
                    assert y is None
                else:
                    np.testing.assert_array_equal(x, y)


def _equations(image, world, *, held):
    with collective_check(world):
        assert len(image["states"]) == 2
        if image["states"][0] is None:
            return
        temporal_error = 0.0
        for level, state in enumerate(image["states"]):
            n = 16 * 2**level
            values = np.asarray(state).reshape(3, n, n)
            live = values[1] > 0
            assert live.any()
            # The scalar reductions occur on either side of the suspension. They
            # must read the same immutable composite value, even on the coarse
            # origin whose mutable provider is reused by the fine origin.
            np.testing.assert_array_equal(values[2][live], np.zeros(int(live.sum())))
            histories = {
                name: np.asarray(rows[level]).reshape(n, n)
                for name, rows in image["histories"].items()
            }
            # Same SSA observation survives the suspension exactly, including bootstrap steps.
            np.testing.assert_array_equal(
                histories["observed_before_barrier"], histories["observed_after_barrier"]
            )
            np.testing.assert_array_equal(
                histories["observed"], histories["observed_after_barrier"]
            )
            temporal_error = max(
                temporal_error,
                float(np.max(np.abs(histories["observed"][live] - (1 + values[1][live] ** 2)))),
            )
            # For -Laplace(phi)+phi=3+t² on the periodic composite mesh, the
            # independent solution is constant. A no-op or swapped solve misses +2.
            np.testing.assert_allclose(
                histories["synchronized"][live], 3 + values[1][live] ** 2, rtol=0, atol=3.0e-9
            )
        if held:
            assert temporal_error > 1.0e-5, temporal_error
        else:
            assert temporal_error < 3.0e-9, temporal_error


@pytest.mark.parametrize("constant_control", (False, True))
def test_temporal_reconstruction_and_hierarchy_barrier_preserve_fields_and_restart(
    tmp_path, isolated_native_cache, native_cxx, kokkos_root, record_property, constant_control
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(
        "temporal_hierarchy_" + str(constant_control),
        builder=build_temporal_field,
        hierarchy_companion=True,
        constant_control=constant_control,
    )
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=0.16, max_steps=4, console=False))
    _equations(_image(runtime, world), world, held=constant_control)
    directory = collective_directory(world, tmp_path)
    checkpoint = collective_call(world, lambda: runtime.checkpoint(str(directory / "mixed-field")))
    restarted = _bind(spec, artifact, world)
    collective_call(world, lambda: restarted.restart(checkpoint))
    _same(_image(runtime, world), _image(restarted, world), world)
    for active in (runtime, restarted):
        collective_call(
            world, lambda active=active: pops.run(active, t_end=0.18, max_steps=1, console=False)
        )
        _equations(_image(active, world), world, held=constant_control)
    _same(_image(runtime, world), _image(restarted, world), world)
    record_property("artifact_identity", artifact.artifact_identity.hexdigest)
    record_property("mpi_ranks", 1 if world is None else int(world.size))


def test_rejection_after_mixed_field_barriers_restores_both_histories_and_sources(
    isolated_native_cache, native_cxx, kokkos_root
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(
        "temporal_hierarchy_reject",
        builder=build_temporal_field,
        hierarchy_companion=True,
        reject_parent=True,
    )
    runtime = _bind(spec, artifact, world)
    collective_call(world, lambda: pops.run(runtime, t_end=0.18, max_steps=5, console=False))
    before = _image(runtime, world)
    _, failures = collective_attempt(
        world, lambda: pops.run(runtime, t_end=0.20, max_steps=1, console=False)
    )
    with collective_check(world):
        assert failures and all(failures), failures
        assert all("reject" in failure[1].lower() for failure in failures), failures
    _same(before, _image(runtime, world), world)


def test_composite_observation_budget_rejects_before_state_or_history_publication(
    isolated_native_cache,
    native_cxx,
    kokkos_root,
):
    del isolated_native_cache, native_cxx, kokkos_root
    spec, artifact, world = _compile(
        "temporal_hierarchy_snapshot_budget",
        builder=build_temporal_field,
        hierarchy_companion=True,
        observation_byte_capacity=1,
    )
    runtime = _bind(spec, artifact, world)
    # Before the first step these output rings have no readable sample. Their
    # exact empty descriptors are included in the accepted-state image.
    before = _image(runtime, world, initialized_histories=False)
    _, failures = collective_attempt(
        world, lambda: pops.run(runtime, t_end=0.04, max_steps=1, console=False)
    )
    with collective_check(world):
        assert failures and all(failures), failures
        assert all("capacity" in failure[1].lower() for failure in failures), failures
    _same(before, _image(runtime, world, initialized_histories=False), world)
