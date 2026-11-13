"""N17 L2b-1 on a native build of rank two: the two-point face on a chart (ROM-110, DN35, DN36).

A two-point face reads the difference of the values of its two cells over the distance between
them, along the normal of the face. On a chart it is consistent where the line of the two
centroids is along the normal (K-orthogonality, design note N17, 3.6); the arrays measure how far
it is from it, and a law that reads the normal only through n.n = 1 (one conductivity on every
axis) gives the same row on every logical axis, so its stencil is the flux density along the
physical normal (DN35). Two structures run here:

* below the surface, the annulus of L1a, 8 x 16 cells, diffused with one conductivity on every
  axis, theta periodic, its walls at r = 1 and r = 2 held at the values 1 and 0
  (``build_annulus_diffusion`` of examples/api_v02/models_static_geometry.py). Its interior faces
  read the distance of the centroids, which the rule computes, and its end faces the mirrored
  distance 2 |m_f - c_K| of the chart (design note N17, 3.3), at the concave wall and at the convex
  one, which C1 stated as NaN before this lot. The polar faces are orthogonal to the line of the
  centroids, and the run takes the steps of an independent NumPy assembly of the same faces;
* at the surface, ``two_point()`` on the doubly periodic unit square under a chart
  (``build_chart_diffusion`` of examples/api_v02/models_charted_supports.py): turned by pi / 6, an
  isometry of the box, whose faces are orthogonal to the line of the centroids and whose run is the
  run of the box to rounding; sheared, x = s + t / 4, whose faces are not (the arrays publish the
  sine 1 / sqrt(17)) and whose run takes the steps of the NumPy assembly of its faces.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pops
import pytest

import pops.surface as surface
from examples.api_v02.models_charted_supports import (
    PLANE_CELLS, PLANE_CONDUCTIVITY, PLANE_DT, PLANE_SHEAR, PLANE_STEPS, build_chart_diffusion)
from examples.api_v02.models_static_geometry import (
    DIFFUSION_CELLS, DIFFUSION_CONDUCTIVITY, DIFFUSION_DT, DIFFUSION_STEPS, DIFFUSION_WALLS,
    build_annulus_diffusion)
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
# The tensor rule GaussLegendre(2) on [0, 1].
RULE_NODES = (0.5 - 0.5/math.sqrt(3.0), 0.5 + 0.5/math.sqrt(3.0))
RULE_WEIGHTS = (0.5, 0.5)


def logical_nodes(lower, upper, cells):
    """lower + (upper - lower) (k / cells), the last node the upper bound (DN15)."""
    nodes = np.array([lower + (upper - lower)*(k/cells) for k in range(cells + 1)])
    nodes[-1] = upper
    return nodes


def chart_geometry(chart, jacobian, first, second):
    """The geometry of a chart of rank two from its map, independently of the arrays, indexed
    [i, j] by the logical cells (i along the first axis): the cell measures and centroids by the
    rule, the area vectors of the faces of each axis (the rotated chords, toward increasing index)
    and their centroids (the chord midpoints)."""
    cells = (len(first) - 1, len(second) - 1)
    measure = np.zeros(cells)
    centroid = np.zeros((2,) + cells)
    for i in range(cells[0]):
        for j in range(cells[1]):
            da, db = first[i + 1] - first[i], second[j + 1] - second[j]
            for ta, wa in zip(RULE_NODES, RULE_WEIGHTS, strict=True):
                for tb, wb in zip(RULE_NODES, RULE_WEIGHTS, strict=True):
                    a, b = first[i] + da*ta, second[j] + db*tb
                    weight = wa*wb*da*db*abs(jacobian(a, b))
                    measure[i, j] += weight
                    centroid[:, i, j] += weight*np.asarray(chart(a, b))
    centroid /= measure
    a, b = np.meshgrid(first, second, indexing="ij")
    x, y = chart(a, b)
    along_second = (x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1])
    along_first = (x[1:, :] - x[:-1, :], y[1:, :] - y[:-1, :])
    faces = (np.stack((along_second[1], -along_second[0])),
             np.stack((-along_first[1], along_first[0])))
    middles = (np.stack(((x[:, 1:] + x[:, :-1])/2, (y[:, 1:] + y[:, :-1])/2)),
               np.stack(((x[1:, :] + x[:-1, :])/2, (y[1:, :] + y[:-1, :])/2)))
    return measure, centroid, faces, middles


def sine_to_normal(gap, area):
    """The sine of the angle between the line ``gap`` and the normal of the area vector."""
    cross = gap[0]*area[1] - gap[1]*area[0]
    return np.abs(cross)/(np.hypot(gap[0], gap[1])*np.hypot(area[0], area[1]))


# --- below the surface: the annulus diffused, its walls held at 1 and 0 --------------------------

def polar(r, theta):
    return r*np.cos(theta), r*np.sin(theta)


def annulus_geometry():
    radii = logical_nodes(1.0, 2.0, DIFFUSION_CELLS[0])
    angles = logical_nodes(0.0, 2*math.pi, DIFFUSION_CELLS[1])
    measure, centroid, (radial, angular), (radial_mid, _) = chart_geometry(
        polar, lambda r, theta: r, radii, angles)
    # The seam theta = 2 pi is the face theta = 0.
    angular[:, :, -1] = angular[:, :, 0]
    inner = np.linalg.norm(radial_mid[:, 0, :] - centroid[:, 0, :], axis=0)
    outer = np.linalg.norm(radial_mid[:, -1, :] - centroid[:, -1, :], axis=0)
    across_r = np.linalg.norm(centroid[:, 1:, :] - centroid[:, :-1, :], axis=0)
    wrapped = np.concatenate((centroid[:, :, -1:], centroid), axis=2)
    across_theta = np.linalg.norm(wrapped[:, :, 1:] - wrapped[:, :, :-1], axis=0)
    distance_r = np.concatenate((2*inner[None], across_r, 2*outer[None]))
    distance_theta = np.concatenate((across_theta, across_theta[:, :1]), axis=1)
    sines = [sine_to_normal(centroid[:, 1:, :] - centroid[:, :-1, :], radial[:, 1:-1, :]),
             sine_to_normal(wrapped[:, :, 1:] - wrapped[:, :, :-1], angular[:, :, :-1]),
             sine_to_normal(radial_mid[:, 0, :] - centroid[:, 0, :], radial[:, 0, :]),
             sine_to_normal(radial_mid[:, -1, :] - centroid[:, -1, :], radial[:, -1, :])]
    return {"measure": measure, "radial": radial, "angular": angular,
            "distance": (distance_r, distance_theta),
            "sine": max(float(np.max(value)) for value in sines)}


def annulus_oracle(initial, steps, dt):
    """SSPRK2 of the two-point faces k (u_R - u_L) / d |S| on the faces of the annulus: across a
    wall the face reads the wall value V at half the mirrored distance, k (V - u_K) / (d / 2)."""
    geometry = annulus_geometry()
    k = DIFFUSION_CONDUCTIVITY
    measure = geometry["measure"]
    radial = np.hypot(*geometry["radial"])
    angular = np.hypot(*geometry["angular"])
    distance_r, distance_theta = geometry["distance"]
    lower, upper = DIFFUSION_WALLS

    def rate(u):
        through_r = np.empty((u.shape[0] + 1, u.shape[1]))
        through_r[1:-1] = k*(u[1:] - u[:-1])/distance_r[1:-1]
        through_r[0] = -k*2*(lower - u[0])/distance_r[0]
        through_r[-1] = k*2*(upper - u[-1])/distance_r[-1]
        wrapped = np.concatenate((u[:, -1:], u), axis=1)
        through_theta = k*(wrapped[:, 1:] - wrapped[:, :-1])/distance_theta[:, :-1]
        through_theta = np.concatenate((through_theta, through_theta[:, :1]), axis=1)
        flux_r, flux_theta = through_r*radial, through_theta*angular
        return ((flux_r[1:] - flux_r[:-1]) + (flux_theta[:, 1:] - flux_theta[:, :-1]))/measure

    u = np.array(initial, dtype=float)
    for _ in range(steps):
        first = u + dt*rate(u)
        u = 0.5*u + 0.5*(first + dt*rate(first))
    return u


def run_below_the_surface(build, *, scope, steps, dt):
    _, world = native_profile(dimension=2, scope=scope)
    case, layout, initial_values, initial = collective_call(world, build)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route=scope, compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values=initial_values,
        resources={"execution_context": artifact_execution_context(artifact)}))
    report = collective_call(world, lambda: pops.run(runtime, t_end=steps*dt, max_steps=steps,
                                                     console=False))
    return world, runtime, report, initial


def test_the_annulus_diffuses_across_its_walls_on_the_mirrored_distance_of_its_chart(
        isolated_native_cache, native_cxx, kokkos_root):
    """DN35 below the surface: one conductivity on every axis is served on the chart, whose end
    faces read 2 |m_f - c_K| at the concave wall r = 1 and at the convex wall r = 2."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, runtime, report, initial = run_below_the_surface(
        build_annulus_diffusion, scope="N17 L2b-1 annulus diffusion", steps=DIFFUSION_STEPS,
        dt=DIFFUSION_DT)
    state = collective_call(world, lambda: runtime.state_global("ring"))
    arrays = collective_call(world, lambda: runtime._executor._s._static_geometry_snapshot())
    geometry = annulus_geometry()
    with collective_check(world):
        assert report.accepted_steps == DIFFUSION_STEPS
        assert arrays["kind"] == "chart"
        # Arrays of the snapshot are (theta, r); the oracle indexes [r, theta].
        for axis, expected in enumerate(geometry["distance"]):
            served = np.asarray(arrays["face_distance"][axis]).T
            assert served.shape == expected.shape and np.all(np.isfinite(served))
            np.testing.assert_allclose(served, expected, rtol=1e-13, atol=0.0)
        assert 0.0 <= arrays["orthogonality"] <= 1e-13
        assert geometry["sine"] <= 1e-13
        initial_grid = np.asarray(initial)[0].T
        final = np.asarray(state)[0].T
        assert np.max(np.abs(final - initial_grid)) > 1e-3
        expected = annulus_oracle(initial_grid, DIFFUSION_STEPS, DIFFUSION_DT)
        np.testing.assert_allclose(final, expected, rtol=1e-13, atol=0.0)
        # The walls held at 1 and 0 heat the inner ring and cool the outer one.
        assert np.mean(final[0] - initial_grid[0]) > 0 > np.mean(final[-1] - initial_grid[-1])


# --- at the surface: two_point() on the turned and the sheared square ----------------------------

def grid(values, shape):
    """An array of the snapshot, whose first axis runs fastest, indexed [i, j]."""
    return np.asarray(values).reshape(shape[::-1]).transpose()


def square_chart(name):
    """The map, its Jacobian determinant and the translations of its two seams."""
    if name == "turned":
        c, s = math.cos(math.pi/6), math.sin(math.pi/6)
        return (lambda a, b: (c*a - s*b, s*a + c*b)), (lambda a, b: 1.0)
    return (lambda a, b: (a + PLANE_SHEAR*b, b)), (lambda a, b: 1.0)


def square_geometry(name):
    nodes = logical_nodes(0.0, 1.0, PLANE_CELLS[0])
    chart, jacobian = square_chart(name)
    measure, centroid, faces, _ = chart_geometry(chart, jacobian, nodes, nodes)
    shift = (np.asarray(chart(1.0, 0.0)) - np.asarray(chart(0.0, 0.0)),
             np.asarray(chart(0.0, 1.0)) - np.asarray(chart(0.0, 0.0)))
    distances, sines = [], []
    for axis in (0, 1):
        before = np.roll(centroid, 1, axis=1 + axis)
        index = [slice(None), slice(None), slice(None)]
        index[1 + axis] = 0
        before[tuple(index)] -= shift[axis][:, None]
        gap = centroid - before
        cut = [slice(None), slice(None), slice(None)]
        cut[1 + axis] = slice(0, -1)
        area = faces[axis][tuple(cut)]
        distances.append(np.hypot(gap[0], gap[1]))
        sines.append(float(np.max(sine_to_normal(gap, area))))
    return measure, tuple(np.hypot(*faces[axis][tuple(
        [slice(None)] + [slice(0, -1) if a == axis else slice(None) for a in (0, 1)])])
        for axis in (0, 1)), tuple(distances), max(sines)


def square_oracle(name, initial, steps, dt):
    """SSPRK2 of the two-point faces -k (u - u_before) / d |S| on the periodic square."""
    measure, areas, distances, _ = square_geometry(name)
    k = PLANE_CONDUCTIVITY

    def rate(u):
        total = np.zeros_like(u)
        for axis in (0, 1):
            flux = -k*(u - np.roll(u, 1, axis=axis))/distances[axis]*areas[axis]
            total -= np.roll(flux, -1, axis=axis) - flux
        return total/measure

    u = np.array(initial, dtype=float)
    for _ in range(steps):
        first = u + dt*rate(u)
        u = 0.5*u + 0.5*(first + dt*rate(first))
    return u


def square_state():
    """A state that varies along both logical axes, at the logical centres of the cells."""
    centres = (np.arange(PLANE_CELLS[0]) + 0.5)/PLANE_CELLS[0]
    s, t = np.meshgrid(centres, centres, indexing="ij")
    return 1.0 + 0.5*np.sin(2*math.pi*s)*np.cos(2*math.pi*t) + 0.25*np.cos(4*math.pi*t)


@pytest.fixture(scope="module")
def squares(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        artifacts = {}
        for name in ("turned", "sheared", None):
            case, u, u0 = build_chart_diffusion(name)
            artifacts[name] = (surface.compile(case, target=surface.OpenMP()), u, u0)
        yield artifacts


def run_square(squares, name, directory):
    artifact, u, u0 = squares[name]
    simulation = surface.bind(artifact, data={u0: square_state()}, directory=directory)
    result = simulation.run(until=PLANE_STEPS*PLANE_DT, max_steps=PLANE_STEPS)
    assert result.failure is None and result.steps == PLANE_STEPS
    arrays = simulation._simulation._runtime._executor._s._static_geometry_snapshot()
    return np.asarray(result.state[u], dtype=float), arrays


def test_two_point_on_the_turned_square_is_the_box_to_rounding(squares, tmp_path):
    """DN35 and DN36 at the surface: the turned square is an isometry of the box, its faces are
    orthogonal to the line of the centroids, and two_point() on it runs as on the box."""
    box, none = run_square(squares, None, tmp_path / "box")
    turned, arrays = run_square(squares, "turned", tmp_path / "turned")
    assert none is None
    assert arrays["kind"] == "chart" and 0.0 <= arrays["orthogonality"] <= 1e-13
    initial = square_state()
    assert np.max(np.abs(box - initial)) > 1e-3
    np.testing.assert_allclose(turned, box, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(turned, square_oracle("turned", initial, PLANE_STEPS, PLANE_DT),
                               rtol=1e-13, atol=0.0)


def test_two_point_on_the_sheared_square_takes_the_steps_of_its_faces(squares, tmp_path):
    """DN36: the sheared square publishes the sine 1 / sqrt(17) between its faces and the line of
    the centroids; its two-point faces read the distance of the centroids all the same, and the
    run takes the steps of the NumPy assembly of those faces, away from the box."""
    sheared, arrays = run_square(squares, "sheared", tmp_path / "sheared")
    _, _, distances, sine = square_geometry("sheared")
    assert arrays["kind"] == "chart"
    assert abs(sine - 1/math.sqrt(17.0)) <= 1e-14
    assert abs(arrays["orthogonality"] - sine) <= 1e-14
    for axis in (0, 1):
        faces = list(PLANE_CELLS)
        faces[axis] += 1
        served = grid(arrays["face_distance"][axis], tuple(faces))
        cut = tuple(slice(0, -1) if a == axis else slice(None) for a in (0, 1))
        np.testing.assert_allclose(served[cut], distances[axis], rtol=1e-13, atol=0.0)
    initial = square_state()
    expected = square_oracle("sheared", initial, PLANE_STEPS, PLANE_DT)
    np.testing.assert_allclose(sheared, expected, rtol=1e-13, atol=0.0)
    box = square_oracle("turned", initial, PLANE_STEPS, PLANE_DT)
    assert np.max(np.abs(sheared - box)) > 1e3*np.max(np.abs(sheared - expected))
