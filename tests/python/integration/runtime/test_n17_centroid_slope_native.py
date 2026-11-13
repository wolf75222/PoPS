"""N17 L2b-2 on a native build of rank two: the centred slope of the static arrays (DN38 to DN40).

The geometric part of the centred slope is one header function of the view of the arrays,
``pops::centroid_slope`` with ``pops::neighbour_centroid`` (include/pops/mesh/geometry/
centroid_slope.hpp): the slope g_K of a cell solves delta X g = delta e by elimination with
partial pivoting, row a of delta X the difference c(K + 1_a) - c(K - 1_a) of the centroids of the
two neighbours of K along a. A neighbour inside the domain is its stored centroid; across the seam
of a periodic axis, the centroid of the cell at the other end translated by the seam; across a
physical end, the centroid of K reflected through the centroid of its end face, 2 m_f - c_K. The
Program operation of centred differences calls it (rank one, test_n17_centroid_slope_rod_native.py)
and the generated cell read of N15 will call it. No Program consumes a slope of two components
today (design note N17, section 20), so in rank two the function is read through the
runtime-private seam of the System, ``_static_geometry_centroid_slopes``, from differences that the
test computes.

On each layout an affine field e = a + b . x, read at the neighbour centroids that NumPy computes
from the snapshot alone, returns b in every cell to the rounding floor of its solve, walls and
seams included:

* the annulus 1 <= r <= 2 of L2b-1 (``build_annulus_diffusion``), a chart whose theta seam is
  closed, with walls at r = 1 and r = 2;
* the sheared square x = s + t / 4 of L2b-1 (``build_chart_diffusion("sheared")``), a doubly
  periodic chart whose seams translate by (1, 0) and (1/4, 1);
* the graded plane of L1b (``build_graded_plane``), node positions, doubly periodic;
* a plane of node positions with walls on both axes, x x / 2 + x / 2 by (3 y + y y) / 4.

On node positions delta X is diagonal, its off-axis entries exact zeros across seams and walls
alike, and the slope is the quotient of the differences by the diagonal, bitwise (DN39). The
snapshot publishes the periodicity and the seam translations of the arrays, and the face
centroids that the reflected centroid reads.
"""
from __future__ import annotations

import numpy as np
import pops
import pytest
from pops.analytic import coordinates
from pops.domain import CartesianDomain
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, GeometryEvolution, MovingControlVolumes

from examples.api_v02.models_charted_supports import PLANE_SHEAR, build_chart_diffusion
from examples.api_v02.models_graded_spacing import build_graded_plane
from examples.api_v02.models_static_geometry import build_annulus_diffusion
from tests.python.support.static_geometry_arrays import (
    affine_differences, centroid_rows, largest_value, layout_static_system, native, ranked,
    rounding_floor)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

# Two affine fields, (a, b): e = a + b . x.
FIELDS = ((0.75, (-1.25, 0.5)), (-0.3, (0.2, 2.0)))


def walled_plane():
    """Node positions of rank two with physical ends on both axes, written below the surface."""
    frame = CartesianDomain("n17_walled_plane", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
    x, y = coordinates(frame)
    layout = MovingControlVolumes(Uniform(CartesianGrid(frame=frame, cells=(6, 5))),
                                  evolution=GeometryEvolution((x*x/2 + x/2, (3*y + y*y)/4)))
    return pops.Case("N17_walled_plane"), layout


def annulus_system():
    case, layout, _, _ = build_annulus_diffusion()
    return layout_static_system(dimension=2, layout=layout, owner=case)


def walled_plane_system():
    case, layout = walled_plane()
    return layout_static_system(dimension=2, layout=layout, owner=case)


# name: (the bare System of its arrays, kind, periodicity, seam translations).
LAYOUTS = {
    "annulus": (annulus_system, "chart", (False, True), ((0.0, 0.0), (0.0, 0.0))),
    "sheared square": (lambda: layout_static_system(build_chart_diffusion("sheared")[0],
                                                    dimension=2),
                       "chart", (True, True), ((1.0, 0.0), (PLANE_SHEAR, 1.0))),
    "graded plane": (lambda: layout_static_system(build_graded_plane()[0], dimension=2),
                     "nodes", (True, True), ((1.0, 0.0), (0.0, 1.0))),
    "walled plane": (walled_plane_system, "nodes", (False, False), ((0.0, 0.0), (0.0, 0.0))),
}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_an_affine_field_at_the_neighbour_centroids_returns_its_slope(native_cxx, kokkos_root,
                                                                       name):
    """Every cell, walls and seams included, returns the slope b of an affine field read at the
    neighbour centroids, to the rounding floor of its solve; on node positions the slope is the
    quotient of the differences by the diagonal of delta X, bitwise."""
    del native_cxx, kokkos_root
    build, kind, periodic, translation = LAYOUTS[name]
    system = build()
    arrays = system._static_geometry_snapshot()
    assert arrays["kind"] == kind
    assert tuple(arrays["periodic"]) == periodic
    assert np.array_equal(np.asarray(arrays["seam_translation"]), np.asarray(translation))
    rows = centroid_rows(arrays)
    cells = rows.shape[:2]
    for offset, slope in FIELDS:
        differences = affine_differences(arrays, offset, slope)
        slopes = np.asarray(system._static_geometry_centroid_slopes(ranked(differences)))
        assert slopes.shape == (2,) + cells[::-1]
        slopes = np.stack([native(component) for component in slopes])
        floor = rounding_floor(rows, largest_value(arrays, offset, slope), slope)
        for component in range(2):
            assert np.all(np.abs(slopes[component] - slope[component]) <= floor), \
                (name, offset, component)
        if kind == "nodes":
            assert np.all(rows[..., 0, 1] == 0.0) and np.all(rows[..., 1, 0] == 0.0)
            for axis in range(2):
                assert np.array_equal(slopes[axis], differences[axis] / rows[..., axis, axis])


def test_the_snapshot_publishes_the_face_centroids_that_the_reflected_centroid_reads(
        native_cxx, kokkos_root):
    """The face centroids of the snapshot are those of its faces: on node positions the node of
    the axis and the midpoint of the other; the upper copy of a seam face is the lower one
    translated by the seam."""
    del native_cxx, kokkos_root
    arrays = walled_plane_system()._static_geometry_snapshot()
    nodes = [native(axis) for axis in arrays["nodes"]]
    x, y = nodes[0][:, 0], nodes[1][0, :]
    faces = [[native(component) for component in axis] for axis in arrays["face_centroid"]]
    assert faces[0][0].shape == (7, 5) and faces[1][0].shape == (6, 6)
    assert np.array_equal(faces[0][0], np.broadcast_to(x[:, None], (7, 5)))
    assert np.array_equal(faces[0][1], np.broadcast_to((0.5*(y[:-1] + y[1:]))[None, :], (7, 5)))
    assert np.array_equal(faces[1][1], np.broadcast_to(y[None, :], (6, 6)))
    assert np.array_equal(faces[1][0], np.broadcast_to((0.5*(x[:-1] + x[1:]))[:, None], (6, 6)))
    sheared = layout_static_system(build_chart_diffusion("sheared")[0], dimension=2)
    arrays = sheared._static_geometry_snapshot()
    for axis, translation in enumerate(((1.0, 0.0), (PLANE_SHEAR, 1.0))):
        faces = [native(component) for component in arrays["face_centroid"][axis]]
        for component in range(2):
            lower = np.take(faces[component], 0, axis=axis)
            upper = np.take(faces[component], faces[component].shape[axis] - 1, axis=axis)
            assert np.array_equal(upper, lower + translation[component]), (axis, component)


def test_the_seam_refuses_differences_of_another_shape(native_cxx, kokkos_root):
    """The differences are one array per axis of the cells of the arrays, in the NumPy order of
    the state; another shape is refused, never read."""
    del native_cxx, kokkos_root
    system = walled_plane_system()
    with pytest.raises(ValueError, match="differences"):
        system._static_geometry_centroid_slopes(np.zeros((2, 6, 5)))
    with pytest.raises(ValueError, match="differences"):
        system._static_geometry_centroid_slopes(np.zeros((1, 5, 6)))
