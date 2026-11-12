"""N17 L1b on a native build of rank two: an advection on a plane graded along x (ROM-110, C1, C2).

The plane of examples/api_v02/models_graded_spacing.py is written with the surface: a scalar
advected at the velocity (1, 1/2) by ``rusanov()`` on the periodic unit square, 16 x 8 cells, its
nodes along x those of the graded ring (``pops.Graded(nodes={plane.x: ring_nodes})``) and uniform
along y, with the SSPRK2 of ``pops.lib.time.methods``. A graded box of rank two is node positions:
its faces are faces of a box, of measure the node differences of the other axis, its cell measures
the products of the node differences, bitwise. The run takes the steps of an independent NumPy
assembly of the same Rusanov faces on the graded cells and keeps the inventory sum |K| u to
rounding.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import pops.surface as surface
from examples.api_v02.models_graded_spacing import (
    PLANE_CELLS, PLANE_DT, PLANE_STEPS, PLANE_VELOCITY, RING_GRADING, build_graded_plane)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]


def graded_nodes(cells):
    """The node function of x at s = k / cells, as the analytic program of bind evaluates it."""
    nodes = np.array([s - RING_GRADING * math.sin(2 * math.pi * s) / (2 * math.pi)
                      for s in (k / cells for k in range(cells + 1))])
    nodes[0], nodes[-1] = 0.0, 1.0
    return nodes


def grid(values, shape):
    """An array of the snapshot, whose first axis runs fastest, indexed [i, j]."""
    return np.asarray(values).reshape(shape[::-1]).transpose()


def oracle(x_nodes, y_nodes, initial, steps, dt):
    """SSPRK2 of the Rusanov faces F.n |S| on the faces of the node positions: an x face has the
    measure of the y interval of its cell, a y face that of its x interval."""
    dx, dy = np.diff(x_nodes)[:, None], np.diff(y_nodes)[None, :]
    measures = dx * dy
    a, b = PLANE_VELOCITY

    def rate(u):
        west = np.roll(u, 1, axis=0)
        south = np.roll(u, 1, axis=1)
        fx = (0.5 * (a * west + a * u) - 0.5 * abs(a) * (u - west)) * dy
        fy = (0.5 * (b * south + b * u) - 0.5 * abs(b) * (u - south)) * dx
        return -((np.roll(fx, -1, axis=0) - fx) + (np.roll(fy, -1, axis=1) - fy)) / measures

    u = np.array(initial, dtype=float)
    for _ in range(steps):
        first = u + dt * rate(u)
        u = (u + first + dt * rate(first)) / 2
    return u


@pytest.fixture(scope="module")
def compiled(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        case, u, u0 = build_graded_plane()
        yield surface.compile(case, target=surface.OpenMP()), u, u0


def test_the_plane_graded_along_x_runs_on_its_node_positions(compiled, tmp_path):
    """CF15: the faces and cells of a graded box of rank two are read from the arrays."""
    artifact, u, u0 = compiled
    x_nodes = graded_nodes(PLANE_CELLS[0])
    y_nodes = np.arange(PLANE_CELLS[1] + 1) / PLANE_CELLS[1]
    xc, yc = 0.5 * (x_nodes[:-1] + x_nodes[1:]), 0.5 * (y_nodes[:-1] + y_nodes[1:])
    initial = (1.0 + 0.5 * np.sin(2 * math.pi * xc)[:, None] * np.cos(2 * math.pi * yc)[None, :]
               + 0.25 * np.cos(2 * math.pi * xc)[:, None])
    simulation = surface.bind(artifact, data={u0: initial}, directory=tmp_path)
    result = simulation.run(until=PLANE_STEPS * PLANE_DT, max_steps=PLANE_STEPS)
    assert result.failure is None and result.steps == PLANE_STEPS
    arrays = simulation._simulation._runtime._executor._s._static_geometry_snapshot()
    assert arrays["kind"] == "nodes"
    dx = np.broadcast_to(np.diff(x_nodes)[:, None], PLANE_CELLS)
    dy = np.broadcast_to(np.diff(y_nodes)[None, :], PLANE_CELLS)
    assert np.array_equal(grid(arrays["width"][0], PLANE_CELLS), dx)
    assert np.array_equal(grid(arrays["width"][1], PLANE_CELLS), dy)
    assert np.array_equal(grid(arrays["cell_measure"], PLANE_CELLS), dx * dy)
    x_faces = (PLANE_CELLS[0] + 1, PLANE_CELLS[1])
    y_faces = (PLANE_CELLS[0], PLANE_CELLS[1] + 1)
    assert np.array_equal(grid(arrays["face_measure"][0], x_faces),
                          np.broadcast_to(np.diff(y_nodes)[None, :], x_faces))
    assert np.array_equal(grid(arrays["face_measure"][1], y_faces),
                          np.broadcast_to(np.diff(x_nodes)[:, None], y_faces))
    final = np.asarray(result.state[u])
    assert final.shape == PLANE_CELLS
    expected = oracle(x_nodes, y_nodes, initial, PLANE_STEPS, PLANE_DT)
    np.testing.assert_allclose(final, expected, rtol=1e-13, atol=0.0)
    uniform = oracle(np.arange(PLANE_CELLS[0] + 1) / PLANE_CELLS[0], y_nodes, initial,
                     PLANE_STEPS, PLANE_DT)
    assert np.max(np.abs(final - uniform)) > 1e3 * np.max(np.abs(final - expected))
    measures = dx * dy
    inventory = np.sum(measures * initial)
    assert abs(np.sum(measures * final) - inventory) <= 1e-13 * inventory
