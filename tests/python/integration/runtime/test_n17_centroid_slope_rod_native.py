"""N17 L2b-2 on a native build of rank one: the centred slope of a Program on graded nodes (DN38).

The Program operation of centred differences (``gradient`` of ``pops.Program``, authored below the
surface; the surface word ``centred_gradient()`` belongs to a later lot of N15) reads the static
geometry arrays on every static layout. The slope g_K of a cell solves delta X g = delta e, row a
of delta X the difference of the centroids of the two neighbours of K along a, delta e_a the
difference of their values (``pops::centroid_slope`` of
include/pops/mesh/geometry/centroid_slope.hpp). The values are those of the boundary session of
the Program: a physical end of the mesh repeats the end value, the halo of a periodic axis copies
the cells of the other end. The neighbour centroids are the stored centroids inside the domain,
the centroid of the cell at the other end translated by the seam across a periodic axis, and at a
physical end the centroid of the end cell reflected through the centroid of its end face,
2 m_f - c_K.

``build_graded_slope`` of examples/api_v02/models_static_geometry.py takes one step that commits
the slope of the state, on 24 cells:

* the rod, x = tanh(3 s) / tanh(3), with physical ends: an affine field e = a + b x bound at the
  centroids returns b to rounding in every cell but the two end cells, and every cell returns the
  NumPy oracle of the same ghosts on the same centroids, bitwise, the end cells included;
* the ring, x = s - 0.6 sin(2 pi s) / (2 pi), periodic, whose seam translates by 1: a periodic
  field returns the NumPy oracle of the translated seam, bitwise.

On both, the runtime-private seam of the System (``_static_geometry_centroid_slopes``) evaluates
the same function from the differences of the same ghosts and returns the slopes of the Program,
bitwise: the Program operation calls the function that the generated cell read of N15 will call.
Without a compiled Program, an affine field read at the neighbour centroids, which NumPy computes
from the snapshot alone, returns its slope through the seam in every cell, ends and seam included.
At the base the Program operation was refused on the arrays (static_geometry_consumer_unavailable)
and the System had no such seam.
"""
from __future__ import annotations

import math

import numpy as np
import pops
import pytest

from examples.api_v02.models_static_geometry import SLOPE_CELLS, build_graded_slope, graded_nodes
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile
from tests.python.support.static_geometry_arrays import (
    affine_differences, centroid_rows, largest_value, layout_static_system, ranked,
    rounding_floor)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

# The affine field of the rod, e = OFFSET + SLOPE x.
OFFSET, SLOPE = 0.75, -1.25


def rod_values(centroids):
    return OFFSET + SLOPE * centroids


def ring_values(centroids):
    return (1.0 + 0.5 * np.sin(2 * math.pi * centroids)
            + 0.25 * np.cos(6 * math.pi * centroids))


def session_differences(values, periodic):
    """e(K + 1) - e(K - 1) with the ghosts of the boundary session of the Program: the halo of the
    periodic axis, or the end value repeated at each physical end."""
    if periodic:
        upper, lower = np.roll(values, -1), np.roll(values, 1)
    else:
        upper = np.append(values[1:], values[-1])
        lower = np.insert(values[:-1], 0, values[0])
    return upper - lower


def run_slope(periodic):
    """Generate the arrays of the layout in a bare System, bind the field of the case at their
    centroids, take the one step of the Program and return the bare System, the arrays, the bound
    values, the report and the slopes of the Program."""
    scope = "N17 L2b-2 slope " + ("ring" if periodic else "rod")
    _, world = native_profile(dimension=1, scope=scope)
    case, layout, subject = collective_call(world, lambda: build_graded_slope(periodic=periodic))
    system = collective_call(world, lambda: layout_static_system(dimension=1, layout=layout,
                                                                 owner=case))
    arrays = collective_call(world, system._static_geometry_snapshot)
    centroids = np.asarray(arrays["centroid"][0])
    values = (ring_values if periodic else rod_values)(centroids)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route=scope, compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values={subject: values[None]},
        resources={"execution_context": artifact_execution_context(artifact)}))
    report = collective_call(world, lambda: pops.run(runtime, t_end=1.0, max_steps=1,
                                                     console=False))
    slopes = np.asarray(collective_call(world, lambda: runtime.state_global("rod")))[0]
    bound = collective_call(world, lambda: runtime._executor._s._static_geometry_snapshot())
    return world, system, arrays, bound, values, report, slopes


def test_the_slope_of_the_rod_reads_its_centroids_and_its_reflected_ends(
        isolated_native_cache, native_cxx, kokkos_root):
    """An affine field returns its slope in the interior of the rod; the end cells read the
    repeated end value on the reflected centroid 2 m_f - c_K, as the oracle of the same ghosts
    does; the seam of the System returns the slopes of the Program, bitwise."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, system, arrays, bound, values, report, slopes = run_slope(periodic=False)
    differences = session_differences(values, periodic=False)
    rows = centroid_rows(arrays)[:, 0, 0]
    through_seam = collective_call(
        world, lambda: system._static_geometry_centroid_slopes(ranked(differences[None])))
    with collective_check(world):
        assert report.accepted_steps == 1
        assert arrays["kind"] == "nodes" and tuple(arrays["periodic"]) == (False,)
        assert np.array_equal(np.asarray(arrays["seam_translation"]), np.zeros((1, 1)))
        for key in ("centroid", "face_centroid"):
            assert np.array_equal(np.asarray(bound[key]), np.asarray(arrays[key])), key
        nodes = np.asarray(arrays["nodes"][0])
        np.testing.assert_allclose(nodes, graded_nodes(SLOPE_CELLS), rtol=0.0, atol=4e-16)
        assert np.array_equal(np.asarray(arrays["face_centroid"][0][0]), nodes)
        assert slopes.shape == (SLOPE_CELLS,)
        assert np.array_equal(slopes, differences / rows)
        floor = rounding_floor(rows[1:-1, None, None], np.max(np.abs(values)), (SLOPE,))
        assert np.all(np.abs(slopes[1:-1] - SLOPE) <= floor)
        # The end cells read the reflected centroid: c_1 + c_0 - 2 m_0 below, about half the slope.
        centroids = np.asarray(arrays["centroid"][0])
        assert rows[0] == centroids[1] - (2 * nodes[0] - centroids[0])
        assert rows[-1] == (2 * nodes[-1] - centroids[-1]) - centroids[-2]
        assert np.array_equal(np.asarray(through_seam)[0], slopes)


def test_the_slope_of_the_ring_reads_the_translated_seam(isolated_native_cache, native_cxx,
                                                         kokkos_root):
    """A periodic field on the graded ring returns the oracle of the seam translated by 1 (the
    halo copies the cells of the other end); the seam of the System returns the slopes of the
    Program, bitwise."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, system, arrays, bound, values, report, slopes = run_slope(periodic=True)
    differences = session_differences(values, periodic=True)
    rows = centroid_rows(arrays)[:, 0, 0]
    through_seam = collective_call(
        world, lambda: system._static_geometry_centroid_slopes(ranked(differences[None])))
    with collective_check(world):
        assert report.accepted_steps == 1
        assert arrays["kind"] == "nodes" and tuple(arrays["periodic"]) == (True,)
        assert np.array_equal(np.asarray(arrays["seam_translation"]), np.ones((1, 1)))
        for key in ("centroid", "face_centroid"):
            assert np.array_equal(np.asarray(bound[key]), np.asarray(arrays[key])), key
        centroids = np.asarray(arrays["centroid"][0])
        assert rows[0] == centroids[1] - (centroids[-1] - 1.0)
        assert rows[-1] == (centroids[0] + 1.0) - centroids[-2]
        assert np.array_equal(slopes, differences / rows)
        assert np.array_equal(np.asarray(through_seam)[0], slopes)


@pytest.mark.parametrize("periodic", [False, True], ids=["rod", "ring"])
def test_an_affine_field_at_the_neighbour_centroids_returns_its_slope(native_cxx, kokkos_root,
                                                                       periodic):
    """Read at the neighbour centroids that NumPy computes from the snapshot alone (the reflected
    end, the translated seam), an affine field returns its slope through the seam of the System
    in every cell, bitwise the quotient of its differences: a row of rank one is its diagonal."""
    del native_cxx, kokkos_root
    case, layout, _ = build_graded_slope(periodic=periodic)
    system = layout_static_system(dimension=1, layout=layout, owner=case)
    arrays = system._static_geometry_snapshot()
    differences = affine_differences(arrays, OFFSET, (SLOPE,))
    rows = centroid_rows(arrays)
    slopes = np.asarray(system._static_geometry_centroid_slopes(ranked(differences)))
    assert slopes.shape == (1, SLOPE_CELLS)
    assert np.array_equal(slopes[0], differences[0] / rows[:, 0, 0])
    floor = rounding_floor(rows, largest_value(arrays, OFFSET, (SLOPE,)), (SLOPE,))
    assert np.all(np.abs(slopes[0] - SLOPE) <= floor)
