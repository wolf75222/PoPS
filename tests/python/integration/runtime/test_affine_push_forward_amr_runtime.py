"""Installed AMR reception of the public affine body on real local state storage."""

import numpy as np
import pops
import pytest

from tests.python.integration.runtime.test_user_numerical_bodies_runtime import _compile, _root_check
from tests.python.support.affine_push_forward_amr_case import (
    CELLS, DT, INDICES, affine_amr_case, expected_level_cell_averages,
    mapped_particle_moments, raw_particle_moments,
)
from tests.python.support.amr_snapshots import level_valid_mask
from tests.python.support.collective_checks import collective_attempt, collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context


pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _states(runtime, levels, world):
    rows = []
    for level in range(levels):
        value = collective_call(world, lambda level=level: runtime.block_level_state_global(
            "moments", level))
        with collective_check(world):
            if world is None or world.rank == 0:
                width = CELLS * 2**level
                rows.append(np.asarray(value, dtype=np.float64).reshape(
                    len(INDICES), width, width).copy())
    return rows


@pytest.mark.parametrize("levels", (1, 2))
def test_public_affine_body_and_local_solve_match_particles_on_amr(
        isolated_native_cache, native_cxx, kokkos_root, levels):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, _ = affine_amr_case(levels=levels)
    artifact, world = _compile(case, layout, "affine-local-amr%d" % levels)
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(
        artifact, resources={"execution_context": context}))
    with collective_check(world):
        assert runtime.n_levels() == levels
    before = _states(runtime, levels, world)

    def check_initial():
        for level, actual in enumerate(before):
            valid = level_valid_mask(runtime, level, refinement_ratio=2)
            if level == 1:
                assert 0 < valid.sum() < valid.size, "a real coarse/fine interface is required"
            expected = expected_level_cell_averages(raw_particle_moments(), level)
            np.testing.assert_allclose(actual[:, valid], expected[:, valid],
                                       rtol=0, atol=3.e-12)
    _root_check(world, check_initial)

    report = collective_call(world, lambda: pops.run(runtime, t_end=DT, max_steps=1,
                                                     console=False))
    after = _states(runtime, levels, world)
    with collective_check(world):
        assert report.accepted_steps == 1 and report.rejected_steps == 0
        assert runtime.time() == pytest.approx(DT) and runtime.macro_step() == 1

    def check_accepted():
        for level, actual in enumerate(after):
            valid = level_valid_mask(runtime, level, refinement_ratio=2)
            expected = expected_level_cell_averages(mapped_particle_moments(), level)
            np.testing.assert_allclose(actual[:, valid], expected[:, valid],
                                       rtol=0, atol=3.e-12)
    _root_check(world, check_accepted)


def test_impossible_local_residual_rejects_without_amr_publication(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    case, layout, _ = affine_amr_case(
        levels=2, case_name="affine_amr_impossible_residual",
        impossible_residual=True)
    artifact, world = _compile(case, layout, "affine-local-amr-reject")
    context = collective_call(world, lambda: artifact_execution_context(artifact))
    runtime = collective_call(world, lambda: pops.bind(
        artifact, resources={"execution_context": context}))
    before = _states(runtime, 2, world)
    with collective_check(world):
        boxes = tuple(runtime.patch_boxes())
    for _ in range(2):
        _, failures = collective_attempt(
            world, lambda: pops.run(runtime, t_end=DT, max_steps=1, console=False))
        with collective_check(world):
            assert all(row is not None and row[2] for row in failures), failures
            assert runtime.time() == 0 and runtime.macro_step() == 0
            assert runtime.n_levels() == 2 and tuple(runtime.patch_boxes()) == boxes
        after = _states(runtime, 2, world)
        def check_rollback(after=after):
            for actual, original in zip(after, before, strict=True):
                np.testing.assert_array_equal(actual, original)
        _root_check(world, check_rollback)
