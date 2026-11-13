"""N17 L2a on a native build of rank two: a scalar advected on the charted annulus (ROM-110, C2).

The annulus of L1a is written with the surface (examples/api_v02/models_charted_supports.py):
``ring = pops.Box(r=(1, 2), theta=pops.Periodic(0, 2 * pi))`` and
``annulus = ring.chart(x=ring.r * cos(ring.theta), y=ring.r * sin(ring.theta))``, a scalar advected
at the Cartesian velocity (1, 1/2) by ``rusanov()``, ``outflow`` at both walls, the SSPRK2 of
``pops.lib.time.methods``, 8 x 16 cells laid out with ``geometry=GaussLegendre(2)``; it is
compiled, bound and run with ``pops.compile``, ``pops.bind`` and ``simulation.run``. The case
elaborates onto the chart of C1 that L1a serves below the surface (design note N17, DN25): its
arrays are a chart, its cell measures the polar measures (r2^2 - r1^2) dtheta / 2, which the rule
integrates exactly (det DX = r), a constant state stays constant to rounding because the faces of
every cell close, and a moving state takes the steps of an independent NumPy assembly of the same
Rusanov faces on the polar chords.

The graded chart (``pops.Graded(base=..., nodes={annulus.r: f})``, f(s) = 1 + s (1 + s) / 2) runs
on the same chart composed with its radial nodes, X(f(s), theta) (DN29): its measures are the polar
measures of the graded radii, the rule still exact (det DX = f f', a cubic), and its steps those of
the same assembly on the chords between the graded radii.

On a physical normal the face input s of ``rusanov()`` is the bound of the normal spectral radius
that b6 provides, sum_k |n_k| |dF_k/du| (normal_flux.hpp), equal to |dF/du . n| on an axis normal
(DN31 of the note): the oracle uses that bound, and the run is told apart from the oracle with the
exact radius |a n_x + b n_y|.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import pops.surface as surface
from examples.api_v02.models_charted_supports import (
    ANNULUS_CELLS, ANNULUS_DT, ANNULUS_RADII, ANNULUS_STEPS, ANNULUS_VELOCITY, build_chart_annulus,
    radial_nodes)

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
GRADINGS = {"uniform": None, "graded": radial_nodes}


def grid(values, shape):
    """An array of the snapshot, whose first axis runs fastest, indexed [i, j]."""
    return np.asarray(values).reshape(shape[::-1]).transpose()


def radii(grading):
    """The radial nodes: 1 + k / 8, or f(k / 8) on the graded chart, the ends at the bounds."""
    cells = ANNULUS_CELLS[0]
    s = np.arange(cells + 1) / cells
    nodes = np.array([ANNULUS_RADII[0] + value for value in s]) if grading is None else \
        np.array([grading(value) for value in s])
    nodes[0], nodes[-1] = ANNULUS_RADII
    return nodes


def chords(nodes):
    """The area vectors of the faces from the polar map, independently of the arrays, indexed
    [component, i, j] like the state: the rotated chords between the images of the logical nodes,
    oriented toward increasing logical index, the seam theta = 2 pi identified with theta = 0, and
    the polar measures."""
    cells_theta = ANNULUS_CELLS[1]
    angles = 2 * math.pi * np.arange(cells_theta + 1) / cells_theta
    x = nodes[:, None] * np.cos(angles)[None, :]
    y = nodes[:, None] * np.sin(angles)[None, :]
    along_theta = (x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1])
    along_r = (x[1:, :] - x[:-1, :], y[1:, :] - y[:-1, :])
    radial = np.stack((along_theta[1], -along_theta[0]))
    angular = np.stack((-along_r[1], along_r[0]))
    angular[:, :, -1] = angular[:, :, 0]
    measures = 0.5 * (nodes[1:] ** 2 - nodes[:-1] ** 2) * (2 * math.pi / cells_theta)
    return radial, angular, np.broadcast_to(measures[:, None], ANNULUS_CELLS)


def rusanov(left, right, area, exact=False):
    """F.n |S| of the Rusanov face of F = (a u, b u) on the unit normal n of the area vector S;
    its speed is the bound |n_x| |a| + |n_y| |b| that b6 provides, or the exact |a n_x + b n_y|."""
    a, b = ANNULUS_VELOCITY
    magnitude = np.hypot(area[0], area[1])
    nx, ny = area[0] / magnitude, area[1] / magnitude
    normal = a * nx + b * ny
    speed = np.abs(normal) if exact else np.abs(nx) * abs(a) + np.abs(ny) * abs(b)
    return (0.5 * normal * (left + right) - 0.5 * speed * (right - left)) * magnitude


def oracle(nodes, initial, steps, dt, exact=False):
    """SSPRK2 of the Rusanov finite volumes on the physical faces of the annulus, outflow in r (the
    ghost repeats the cell) and theta periodic."""
    radial, angular, measures = chords(nodes)

    def rate(u):
        along_r = np.concatenate((u[:1, :], u, u[-1:, :]), axis=0)
        along_theta = np.concatenate((u[:, -1:], u, u[:, :1]), axis=1)
        through_r = rusanov(along_r[:-1, :], along_r[1:, :], radial, exact)
        through_theta = rusanov(along_theta[:, :-1], along_theta[:, 1:], angular, exact)
        return -((through_r[1:, :] - through_r[:-1, :])
                 + (through_theta[:, 1:] - through_theta[:, :-1])) / measures

    u = np.array(initial, dtype=float)
    for _ in range(steps):
        first = u + dt * rate(u)
        u = (u + first + dt * rate(first)) / 2
    return u


def moving_state(nodes):
    """A state that varies along both logical axes, at the logical centres of the cells."""
    centres = 0.5 * (nodes[:-1] + nodes[1:])
    angles = 2 * math.pi * (np.arange(ANNULUS_CELLS[1]) + 0.5) / ANNULUS_CELLS[1]
    r, theta = np.meshgrid(centres, angles, indexing="ij")
    return 1.0 + 0.5 * np.sin(theta) * (r - 1.0) + 0.25 * np.cos(2 * theta)


@pytest.fixture(scope="module")
def compiled(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        artifacts = {}
        for label, grading in GRADINGS.items():
            case, u, u0 = build_chart_annulus(graded=grading)
            artifacts[label] = (surface.compile(case, target=surface.OpenMP()), u, u0)
        yield artifacts


def run(artifact, u, u0, initial, directory):
    simulation = surface.bind(artifact, data={u0: initial}, directory=directory)
    result = simulation.run(until=ANNULUS_STEPS * ANNULUS_DT, max_steps=ANNULUS_STEPS)
    assert result.failure is None and result.steps == ANNULUS_STEPS
    arrays = simulation._simulation._runtime._executor._s._static_geometry_snapshot()
    return np.asarray(result.state[u], dtype=float), arrays


@pytest.mark.parametrize("label", list(GRADINGS))
def test_the_charted_annulus_runs_on_the_physical_faces_of_its_chart(compiled, label, tmp_path):
    """CF15: the case written with σ.chart and geometry= runs on C1's chart arrays."""
    artifact, u, u0 = compiled[label]
    nodes = radii(GRADINGS[label])
    _, _, polar = chords(nodes)

    constant = np.full(ANNULUS_CELLS, 1.3)
    still, arrays = run(artifact, u, u0, constant, tmp_path / "constant")
    assert arrays["kind"] == "chart" and 0.0 <= arrays["closure_ratio"] <= 1.0
    measures = grid(arrays["cell_measure"], ANNULUS_CELLS)
    assert measures.shape == ANNULUS_CELLS
    np.testing.assert_allclose(measures, polar, rtol=1e-13, atol=0.0)
    assert np.max(np.abs(still - constant)) <= 1e-13 * 1.3

    initial = moving_state(nodes)
    final, _ = run(artifact, u, u0, initial, tmp_path / "moving")
    assert final.shape == ANNULUS_CELLS and np.max(np.abs(final - initial)) > 1e-3
    expected = oracle(nodes, initial, ANNULUS_STEPS, ANNULUS_DT)
    np.testing.assert_allclose(final, expected, rtol=1e-13, atol=0.0)
    exact = oracle(nodes, initial, ANNULUS_STEPS, ANNULUS_DT, exact=True)
    assert np.max(np.abs(final - exact)) > 1e3 * np.max(np.abs(final - expected))
