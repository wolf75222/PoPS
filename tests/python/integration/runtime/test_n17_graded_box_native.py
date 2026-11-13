"""N17 L1b on a native build of rank three: the diffusion of a graded box (ROM-110, C1, C2).

The test of rank three that the design note asks of L1 (section 10): the widths and the diffusion
inventory of a graded box of rank three. The box of examples/api_v02/models_graded_spacing.py is
written with the surface: a scalar diffused by ``two_point()`` on the periodic box
[0, 1) x [0, 2) x [-1, 1), 8 x 6 x 4 cells, x graded as the ring, y uniform and z graded on
[-1, 1] (s = (z + 1) / 2), with the SSPRK2 of ``pops.lib.time.methods``. The widths are the node
differences of each axis, the cell measures and the face measures their products, bitwise; the run
takes the steps of an independent NumPy assembly of the same two-point fluxes on the graded cells
and keeps the inventory sum |K| u to rounding.

N17 L2b-2 reads the centred slope of the same arrays in rank three, through the runtime-private
seam of the System (design note N17, section 20): an affine field read at the neighbour centroids
returns its slope in every cell, across the three seams, bitwise the quotient of its differences.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import pops.surface as surface
from examples.api_v02.models_graded_spacing import (
    BOX_CELLS, BOX_DT, BOX_STEPS, BOX_Z_GRADING, CONDUCTIVITY, RING_GRADING, build_graded_box)
from tests.python.support.static_geometry_arrays import (
    affine_differences, centroid_rows, largest_value, layout_static_system, native, ranked,
    rounding_floor)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]


def axis_nodes():
    """The nodes of x, y and z as the analytic programs of bind evaluate them: the logical node k
    is lower + (upper - lower) (k / c), the last one upper; s = (node - lower) / (upper - lower)."""
    cells_x, cells_y, cells_z = BOX_CELLS
    x = np.array([s - RING_GRADING * math.sin(2 * math.pi * s) / (2 * math.pi)
                  for s in (k / cells_x for k in range(cells_x + 1))])
    y = np.array([0.0 + 2.0 * (k / cells_y) for k in range(cells_y)] + [2.0])
    logical = [-1.0 + 2.0 * (k / cells_z) for k in range(cells_z)] + [1.0]
    z = np.array([-1 + 2 * (s + BOX_Z_GRADING * math.sin(2 * math.pi * s) / (2 * math.pi))
                  for s in ((value - -1.0) / 2.0 for value in logical)])
    x[0], x[-1], z[0], z[-1] = 0.0, 1.0, -1.0, 1.0
    return x, y, z


def grid(values, shape):
    """An array of the snapshot, whose first axis runs fastest, indexed [i, j, l]."""
    return np.asarray(values).reshape(shape[::-1]).transpose()


def seam_distances(nodes):
    centroids = 0.5 * (nodes[:-1] + nodes[1:])
    left = np.roll(centroids, 1)
    left[0] -= nodes[-1] - nodes[0]
    return centroids - left


def geometry(x, y, z):
    """Widths, cell measures and face measures of the box of node positions."""
    dx, dy, dz = np.diff(x)[:, None, None], np.diff(y)[None, :, None], np.diff(z)[None, None, :]
    return (dx, dy, dz), dx * dy * dz, (dy * dz, dx * dz, dx * dy)


def oracle(nodes, initial, steps, dt):
    """SSPRK2 of the two-point fluxes k (u_R - u_L) / d |S| on the faces of the box."""
    _, measures, areas = geometry(*nodes)
    shapes = ((-1, 1, 1), (1, -1, 1), (1, 1, -1))
    distances = [seam_distances(axis).reshape(shape)
                 for axis, shape in zip(nodes, shapes, strict=True)]

    def rate(u):
        total = np.zeros_like(u)
        for axis in range(3):
            flux = CONDUCTIVITY * (u - np.roll(u, 1, axis=axis)) / distances[axis] * areas[axis]
            total = total + (np.roll(flux, -1, axis=axis) - flux)
        return total / measures

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
        case, u, u0 = build_graded_box()
        yield surface.compile(case, target=surface.OpenMP()), u, u0


def test_the_widths_and_the_diffusion_inventory_of_a_graded_box_of_rank_three(compiled,
                                                                              tmp_path):
    """CF15: rank three, refused at the base, generates the arrays of a graded box and diffuses
    on them."""
    artifact, u, u0 = compiled
    nodes = axis_nodes()
    centroids = [0.5 * (axis[:-1] + axis[1:]) for axis in nodes]
    initial = (1.0 + 0.5 * np.sin(2 * math.pi * centroids[0])[:, None, None]
               * np.cos(math.pi * centroids[1])[None, :, None]
               + 0.25 * np.sin(math.pi * centroids[2])[None, None, :])
    simulation = surface.bind(artifact, data={u0: initial}, directory=tmp_path)
    result = simulation.run(until=BOX_STEPS * BOX_DT, max_steps=BOX_STEPS)
    assert result.failure is None and result.steps == BOX_STEPS
    arrays = simulation._simulation._runtime._executor._s._static_geometry_snapshot()
    assert arrays["kind"] == "nodes"
    widths, measures, areas = geometry(*nodes)
    for axis in range(3):
        assert np.array_equal(grid(arrays["width"][axis], BOX_CELLS),
                              np.broadcast_to(widths[axis], BOX_CELLS)), axis
        faces = tuple(cells + (other == axis) for other, cells in enumerate(BOX_CELLS))
        assert np.array_equal(grid(arrays["face_measure"][axis], faces),
                              np.broadcast_to(areas[axis], faces)), axis
    assert np.array_equal(grid(arrays["cell_measure"], BOX_CELLS), measures)
    final = np.asarray(result.state[u])
    assert final.shape == BOX_CELLS
    expected = oracle(nodes, initial, BOX_STEPS, BOX_DT)
    np.testing.assert_allclose(final, expected, rtol=1e-13, atol=0.0)
    uniform = (np.arange(BOX_CELLS[0] + 1) / BOX_CELLS[0], nodes[1],
               -1.0 + 2.0 * np.arange(BOX_CELLS[2] + 1) / BOX_CELLS[2])
    assert np.max(np.abs(final - oracle(uniform, initial, BOX_STEPS, BOX_DT))) \
        > 1e3 * np.max(np.abs(final - expected))
    inventory = np.sum(measures * initial)
    assert abs(np.sum(measures * final) - inventory) <= 1e-13 * inventory


def test_an_affine_field_at_the_neighbour_centroids_returns_its_slope_on_the_box(native_cxx,
                                                                                  kokkos_root):
    """N17 L2b-2 (DN38, DN39): read at the neighbour centroids that NumPy computes from the
    snapshot alone, across the three translated seams of the box, an affine field returns its
    slope through the seam of the System (``pops::centroid_slope``) in every cell. Delta X of node
    positions is diagonal, its off-axis entries exact zeros, and the slope is the quotient of the
    differences by its diagonal, bitwise."""
    del native_cxx, kokkos_root
    system = layout_static_system(build_graded_box()[0], dimension=3)
    arrays = system._static_geometry_snapshot()
    assert tuple(arrays["periodic"]) == (True, True, True)
    assert np.array_equal(np.asarray(arrays["seam_translation"]), np.diag([1.0, 2.0, 2.0]))
    rows = centroid_rows(arrays)
    assert np.all(rows[..., ~np.eye(3, dtype=bool)] == 0.0)
    offset, slope = 0.75, (-1.25, 0.5, 2.0)
    differences = affine_differences(arrays, offset, slope)
    slopes = np.asarray(system._static_geometry_centroid_slopes(ranked(differences)))
    assert slopes.shape == (3,) + BOX_CELLS[::-1]
    slopes = np.stack([native(component) for component in slopes])
    floor = rounding_floor(rows, largest_value(arrays, offset, slope), slope)
    for axis in range(3):
        assert np.array_equal(slopes[axis], differences[axis] / rows[..., axis, axis]), axis
        assert np.all(np.abs(slopes[axis] - slope[axis]) <= floor), axis
