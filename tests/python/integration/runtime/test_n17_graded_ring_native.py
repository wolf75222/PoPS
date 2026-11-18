"""N17 L1b on a native build of rank one: graded spacing at the surface, run (ROM-110, C1, C2).

The ring of examples/api_v02/models_graded_spacing.py is written with the surface: a scalar on the
periodic ring 0 <= x < 1 whose nodes are s - 0.6 sin(2 pi s) / (2 pi), ``pops.Graded``, diffused by
``two_point()`` or advected by ``rusanov()``, with the SSPRK2 of ``pops.lib.time.methods``; it is
compiled, bound and run with ``pops.compile``, ``pops.bind`` and ``simulation.run``. The static
geometry arrays are those of its node positions: the nodes are the node function at s = k / 32 as
the analytic program of bind evaluates it, with the ends set to the bounds, and the widths and
cell measures are their node differences, bitwise; the distance across the seam joins the two
half cells. Each run takes the steps of an independent NumPy assembly of the same scheme on the
graded cells and keeps the inventory sum |K| u to rounding.

Both words are faces authored in source since N3 lot 1: rusanov() a ``riemann.User`` face of the
generated block, two_point() a ``StencilFlux``. Their one stability guard (reason 503) reads the
measures of the graded cells: a step 5 % above the bound of the seam cells is refused, although the
uniform cells of the logical ring would admit it, and a step 5 % below is admitted.

The nodes nest bitwise (note, 3.8): node r k of r c cells is node k of c cells, for c = 32 and 24
and r = 2, 4 and 8, and a coarse width is the sum of its fine widths to rounding. Probe 3 (note,
6.3) reads the same mechanism on the wall nodes tanh(3 s) / tanh(3) of its layout from 32 to 256
cells; its steady layer waits on N9 and G04 (tests/python/unit/surface/test_n17_graded.py).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import pops.surface as surface
from examples.api_v02.models_graded_spacing import (
    CONDUCTIVITY, RING_ADVECTION_DT, RING_CELLS, RING_DIFFUSION_DT, RING_GRADING, RING_SPEED,
    RING_STEPS, WALL_CELLS, WALL_GRADING, build_graded_ring, build_wall_layer)
from tests.python.support.static_geometry_arrays import layout_static_arrays

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
DT = {"diffusion": RING_DIFFUSION_DT, "advection": RING_ADVECTION_DT}


def ring_nodes(cells):
    """The node function at s = k / cells, operation by operation as the analytic program of bind
    evaluates it, the ends set to the bounds."""
    nodes = np.array([s - RING_GRADING * math.sin(2 * math.pi * s) / (2 * math.pi)
                      for s in (k / cells for k in range(cells + 1))])
    nodes[0], nodes[-1] = 0.0, 1.0
    return nodes


def wall_nodes(cells):
    """tanh(3 s) / tanh(3) of probe 3, as its node function writes it with exp."""
    edge = math.exp(-2 * WALL_GRADING)
    values = []
    for s in (k / cells for k in range(cells + 1)):
        growth = math.exp(-2 * WALL_GRADING * s)
        values.append((1 - growth) * (1 + edge) / ((1 + growth) * (1 - edge)))
    nodes = np.array(values)
    nodes[0], nodes[-1] = 0.0, 1.0
    return nodes


def initial_values(nodes):
    """Cell values that vary around the ring, at the centroids of the graded cells."""
    centroids = 0.5 * (nodes[:-1] + nodes[1:])
    return 1.0 + 0.5 * np.sin(2 * math.pi * centroids) + 0.25 * np.cos(6 * math.pi * centroids)


def seam_distances(nodes):
    """The distance between the centroids across each face; face 0 joins the last cell, translated
    by the length of the ring, to the first."""
    centroids = 0.5 * (nodes[:-1] + nodes[1:])
    left = np.roll(centroids, 1)
    left[0] -= nodes[-1] - nodes[0]
    return centroids - left


def oracle(term, nodes, initial, steps, dt):
    """The steps of SSPRK2 on the graded ring: two-point fluxes k (u_i - u_i-1) / d_i, or Rusanov
    fluxes of a u, on the faces of the node positions, divided by the cell measures."""
    measures = np.diff(nodes)
    distances = seam_distances(nodes)

    def rate(u):
        left = np.roll(u, 1)
        if term == "diffusion":
            faces = CONDUCTIVITY * (u - left) / distances
            return (np.roll(faces, -1) - faces) / measures
        faces = 0.5 * (RING_SPEED * left + RING_SPEED * u) - 0.5 * abs(RING_SPEED) * (u - left)
        return -(np.roll(faces, -1) - faces) / measures

    u = np.array(initial, dtype=float)
    for _ in range(steps):
        first = u + dt * rate(u)
        u = (u + first + dt * rate(first)) / 2
    return u


def guard_frequency(term, nodes):
    """The largest frequency that the guard of the faces reads on a cell of the ring: the sum over
    its two faces of the face conductance k / d (diffusion) or of the face speed |a| (advection),
    each times |S| / |K|, which is 1 / w at rank one."""
    widths = np.diff(nodes)
    if term == "advection":
        return float(np.max(abs(RING_SPEED) / widths))
    conductance = CONDUCTIVITY / seam_distances(nodes)
    return float(np.max((conductance + np.roll(conductance, -1)) / widths))


def guard_steps(term):
    """The steps 5 % below and 5 % above the bound of the graded cells (SSPRK2 and FixedDt give
    each stage the budget 1)."""
    bound = 1.0 / guard_frequency(term, ring_nodes(RING_CELLS))
    return {"admitted": 0.95 * bound, "refused": 1.05 * bound}


@pytest.fixture(scope="module")
def compiled(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        cases = {term: build_graded_ring(term) for term in DT}
        yield {term: (surface.compile(case, target=surface.OpenMP()), u, u0)
               for term, (case, u, u0) in cases.items()}


@pytest.fixture(scope="module")
def guarded(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        cases = {(term, side): build_graded_ring(term, dt=dt)
                 for term in DT for side, dt in guard_steps(term).items()}
        yield {key: (surface.compile(case, target=surface.OpenMP()), u0)
               for key, (case, u, u0) in cases.items()}


@pytest.mark.parametrize("term", ["diffusion", "advection"])
def test_the_stability_guard_reads_the_measures_of_the_graded_cells(guarded, tmp_path, term):
    """The guard of the faces bounds the step by the frequency of the graded cells, so that the
    step 5 % above the bound of the seam cells (0.4 / 32 wide) is refused with the reason 503,
    although the uniform cells 1 / 32 of the logical ring would admit twice that step, and the
    step 5 % below the bound is admitted."""
    steps = guard_steps(term)
    uniform = 1.0 / guard_frequency(term, np.arange(RING_CELLS + 1) / RING_CELLS)
    assert 2 * steps["refused"] < uniform
    initial = initial_values(ring_nodes(RING_CELLS))
    artifact, u0 = guarded[(term, "admitted")]
    simulation = surface.bind(artifact, data={u0: initial}, directory=tmp_path / "admitted")
    result = simulation.run(until=steps["admitted"], max_steps=1)
    assert result.failure is None and result.steps == 1
    artifact, u0 = guarded[(term, "refused")]
    simulation = surface.bind(artifact, data={u0: initial}, directory=tmp_path / "refused")
    with pytest.raises(Exception, match="user_face_numerical_stability|reason_code=503"):
        simulation.run(until=steps["refused"], max_steps=1)


@pytest.mark.parametrize("term", ["diffusion", "advection"])
def test_the_graded_ring_runs_on_its_node_positions(compiled, tmp_path, term):
    """CF15: the surface case runs on the arrays of its graded nodes, not on the uniform spacing
    of its logical ring, and its steps are those of the scheme on the graded cells."""
    artifact, u, u0 = compiled[term]
    nodes = ring_nodes(RING_CELLS)
    initial = initial_values(nodes)
    simulation = surface.bind(artifact, data={u0: initial}, directory=tmp_path)
    result = simulation.run(until=RING_STEPS * DT[term], max_steps=RING_STEPS)
    assert result.failure is None and result.steps == RING_STEPS
    arrays = simulation._simulation._runtime._executor._s._static_geometry_snapshot()
    assert arrays["kind"] == "nodes"
    native = np.asarray(arrays["nodes"][0])
    assert native.tobytes() == nodes.tobytes()
    widths = np.asarray(arrays["width"][0])
    measures = np.asarray(arrays["cell_measure"])
    assert np.array_equal(widths, np.diff(nodes)) and np.array_equal(measures, np.diff(nodes))
    distances = np.asarray(arrays["face_distance"][0])
    np.testing.assert_allclose(distances[:-1], seam_distances(nodes), rtol=4e-16, atol=0.0)
    assert distances[-1] == distances[0]
    final = np.asarray(result.state[u])
    expected = oracle(term, nodes, initial, RING_STEPS, DT[term])
    np.testing.assert_allclose(final, expected, rtol=1e-13, atol=0.0)
    uniform = oracle(term, np.arange(RING_CELLS + 1) / RING_CELLS, initial, RING_STEPS, DT[term])
    assert np.max(np.abs(final - uniform)) > 1e3 * np.max(np.abs(final - expected))
    inventory = np.sum(measures * initial)
    assert abs(np.sum(measures * final) - inventory) <= 1e-13 * inventory


@pytest.mark.parametrize("base", [32, 24])
def test_graded_nodes_nest_bitwise_across_resolutions(base):
    """Node k of c cells and node r k of r c cells read one binary64 point, s = k / c correctly
    rounded (note, 3.8), so a coarse width is the sum of its r fine widths to rounding."""
    coarse = layout_static_arrays(build_graded_ring("diffusion", cells=base)[0], dimension=1)
    nodes = np.asarray(coarse["nodes"][0])
    assert nodes.tobytes() == ring_nodes(base).tobytes()
    widths = np.asarray(coarse["width"][0])
    for ratio in (2, 4, 8):
        fine = layout_static_arrays(build_graded_ring("diffusion", cells=ratio * base)[0],
                                    dimension=1)
        fine_nodes = np.asarray(fine["nodes"][0])
        assert fine_nodes[::ratio].tobytes() == nodes.tobytes(), ratio
        sums = np.asarray(fine["width"][0]).reshape(base, ratio).sum(axis=1)
        assert np.max(np.abs(sums - widths)) <= ratio * np.spacing(1.0), ratio


def test_the_wall_nodes_of_probe_3_nest_and_cluster_at_the_wall():
    """Probe 3 at mechanism level (note, 6.3): its layout, pops.Graded with the wall nodes, gives
    from 32 to 256 cells nested node positions whose widths shrink toward the wall x = 1, in the
    ratio of the slopes of tanh(3 s) at the two ends."""
    arrays = {cells: layout_static_arrays(build_wall_layer(cells, diffusion=False)[0],
                                          dimension=1)
              for cells in (WALL_CELLS, 2 * WALL_CELLS, 4 * WALL_CELLS, 8 * WALL_CELLS)}
    nodes = np.asarray(arrays[WALL_CELLS]["nodes"][0])
    assert nodes.tobytes() == wall_nodes(WALL_CELLS).tobytes()
    for cells, fine in arrays.items():
        ratio = cells // WALL_CELLS
        assert np.asarray(fine["nodes"][0])[::ratio].tobytes() == nodes.tobytes(), cells
        widths = np.asarray(fine["width"][0])
        assert np.all(np.diff(widths) < 0), cells
        slopes = math.cosh(WALL_GRADING) ** 2   # f'(0) / f'(1) for tanh(3 s) / tanh(3)
        assert 0.85 * slopes < widths[0] / widths[-1] < slopes, cells
    assert np.asarray(arrays[WALL_CELLS]["face_distance"][0])[-1] == 2 * (1.0 - 0.5 * (
        nodes[-2] + nodes[-1]))
