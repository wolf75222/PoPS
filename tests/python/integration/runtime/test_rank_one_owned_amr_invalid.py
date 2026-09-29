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


def _agree(world, condition, message):
    failures = allgather_value(world, "" if condition else message)
    assert not any(failures), "; ".join(item for item in failures if item)


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
    _agree(world, len(local) == report.coarse_local_boxes and
           report.coarse_total_boxes > len(local) > 0,
           "native local AMR box count disagrees with the live patch report")
    owned = allgather_value(world, local)
    _agree(world, len(owned) == 2 and all(owned) and
           sum(len(row) for row in owned) == report.coarse_total_boxes,
           "two nonempty rank-owned box sets must match global box count")
    coverage = np.zeros((N, N), dtype=np.int8)
    valid_bounds = True
    for boxes in owned:
        for (lx, ly), (ux, uy) in boxes:
            valid_bounds &= 0 <= lx < ux <= N and 0 <= ly < uy <= N
            if valid_bounds:
                coverage[ly:uy, lx:ux] += 1
    _agree(world, valid_bounds and bool(np.all(coverage == 1)),
           "native rank-owned AMR boxes must cover each level-0 cell exactly once")

    # A rank-one interior cell is selected from the native MultiFab local boxes.
    # It is not inferred from a round-robin assumption or global patch listing.
    candidates = [
        (x, y) for (lx, ly), (ux, uy) in owned[1]
        for x in range(lx + 1, ux - 1) for y in range(ly + 1, uy - 1)
        if not any(_cell_in(box, x, y) for box in owned[0])
    ]
    _agree(world, bool(candidates), "rank one needs a stencil-interior owned cell")
    x, y = candidates[0]

    # Same artifact/layout/parameters without an active invalid branch must
    # accept one step. This rules out an unrelated preflight or layout failure.
    pops.run(probe, t_end=DT, max_steps=1, console=False)
    _agree(world, probe.time() == DT and probe.macro_step() == 1,
           "inactive User body did not complete the positive control")

    disturbed = initials[0].copy()
    disturbed[0, y, x] = 3.
    runtime = _bind(artifact, handles, subjects, (disturbed, initials[1]),
                    ((0., .75), (0., .6)), gate=2.)
    _agree(world, runtime.amr.coarse_local_box_bounds() == tuple(owned[world.rank]),
           "rank ownership changed between positive and negative bind")
    before = tuple(_snapshot(runtime, name, 1, "amr1", world)
                   for name in ("first", "second"))
    failure = ("", "")
    try:
        pops.run(runtime, t_end=DT, max_steps=1, console=False)
    except Exception as exc:
        failure = (type(exc).__name__, str(exc))
    failures = allgather_value(world, failure)
    _agree(world, all(kind == "RuntimeError" and message for kind, message in failures),
           "the active User body must reject collectively as RuntimeError")
    _agree(world, runtime.time() == 0. and runtime.macro_step() == 0,
           "rejected step advanced accepted time or macro-step")
    after = tuple(_snapshot(runtime, name, 1, "amr1", world)
                  for name in ("first", "second"))
    if world.rank == 0:
        for old, new in zip(before, after, strict=True):
            np.testing.assert_array_equal(new, old)
