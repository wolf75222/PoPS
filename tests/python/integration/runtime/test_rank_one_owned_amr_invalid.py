"""MPI ownership witness: a user body fails on a rank-one-owned AMR cell.

The model and active invalid body come from the general public User-numerics
fixture. This file changes only the layout and independently verifies actual
rank-local level-0 boxes before choosing the injected cell.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest

from pops._native_collectives import allgather_value
from pops.amr import PatchLayout
from pops.layouts import AMR
from tests.python.integration.runtime.test_user_numerical_bodies_runtime import (
    DT, N, _bind, _case, _compile, _snapshot, _world,
)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _distributed(layout):
    return AMR(
        grid=layout.grid, hierarchy=layout.hierarchy, tagging=layout.tagging,
        regrid=layout.regrid, transfer=layout.transfer, execution=layout.execution,
        embedded_boundary=layout.embedded_boundary,
        patch_layout=PatchLayout(distribute_coarse=True, coarse_max_grid=4),
        load_balance=layout.load_balance, tagger=layout.tagger,
        clustering=layout.clustering, reflux=layout.reflux,
    )


def _cell_in(box, x, y):
    lower, upper = box
    return lower[0] <= x < upper[0] and lower[1] <= y < upper[1]


def test_rank_one_owned_invalid_body_collectively_rolls_back(
        isolated_native_cache, native_cxx, kokkos_root):
    world = _world()
    if world is None or world.size != 2:
        pytest.skip("requires two actual native MPI ranks")

    case, base_layout, handles, subjects, _, _ = _case(
        1, layout_kind="amr1", invalid="body")
    artifact, world = _compile(case, _distributed(base_layout),
                               "user-invalid-owned-rank-one-amr")
    initials = (np.ones((1, N, N)), np.ones((1, N, N)))
    probe = _bind(artifact, handles, subjects, initials,
                  ((0., .75), (0., .6)), gate=2.)

    local = probe.amr.coarse_local_box_bounds()
    report = probe.amr.patch_table()
    assert len(local) == report.coarse_local_boxes
    assert report.coarse_total_boxes > len(local) > 0
    owned = allgather_value(world, local)
    assert len(owned) == 2 and all(owned)
    assert sum(len(row) for row in owned) == report.coarse_total_boxes
    coverage = np.zeros((N, N), dtype=np.int8)
    for boxes in owned:
        for (lx, ly), (ux, uy) in boxes:
            assert 0 <= lx < ux <= N and 0 <= ly < uy <= N
            coverage[ly:uy, lx:ux] += 1
    np.testing.assert_array_equal(coverage, np.ones_like(coverage))

    # A rank-one interior cell is selected from the native MultiFab local boxes.
    # It is not inferred from a round-robin assumption or global patch listing.
    candidates = [
        (x, y) for (lx, ly), (ux, uy) in owned[1]
        for x in range(lx + 1, ux - 1) for y in range(ly + 1, uy - 1)
        if not any(_cell_in(box, x, y) for box in owned[0])
    ]
    assert candidates, "rank one needs a stencil-interior owned cell"
    x, y = candidates[0]
    disturbed = initials[0].copy()
    disturbed[0, y, x] = 3.
    runtime = _bind(artifact, handles, subjects, (disturbed, initials[1]),
                    ((0., .75), (0., .6)), gate=2.)
    assert runtime.amr.coarse_local_box_bounds() == tuple(owned[world.rank])
    before = tuple(_snapshot(runtime, name, 1, "amr1", world)
                   for name in ("first", "second"))
    failure = ""
    try:
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
    except RuntimeError as exc:
        failure = str(exc)
    failures = allgather_value(world, failure)
    assert all(failures), "the rank-one failure must reach both ranks"
    assert runtime.time() == 0. and runtime.macro_step() == 0
    after = tuple(_snapshot(runtime, name, 1, "amr1", world)
                  for name in ("first", "second"))
    if world.rank == 0:
        for old, new in zip(before, after, strict=True):
            np.testing.assert_array_equal(new, old)
