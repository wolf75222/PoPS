"""The two graphs of C1, static geometry arrays of a structured configuration (N17, CF15).

Spec 0.3 D0b: the measures, face vectors and normals of a mapped configuration are geometry arrays
generated « as for a moving configuration (D18, D21) at zero velocity ». At the base the geometry
evolution contract (``pops.mesh.moving``) had one provider, the periodic Uniform1D carrier of a
moving interval, and refused a clock-free map of rank one or two with
``native_geometry_evolution_unavailable``. The two graphs were committed failing first; they are
written below the surface with the descriptors that exist there (design note N17, section 4,
DN14), so that each failed for C1's reason and for no missing word of C2:

* graph (1), a vector balance in Cartesian components on a charted support (the structure of the
  diocotron on a disk, probe 2): isothermal Euler on a 4 x 8 annulus 1 <= r <= 2, theta periodic,
  the map (r cos theta, r sin theta), which reads no clock, Rusanov faces, a constant state. The
  rule of its measures is stated (D0b), GaussLegendre(2) on each logical axis;
* graph (2), graded nodes (the structure of the wall-graded sheath, probe 3): diffusion with
  no-flux ends on 32 cells of a non-periodic 1-D layout whose nodes are x = tanh(3 s) / tanh(3),
  written with exp, which reads no clock either. A map of rank one is read as node positions with
  exact node differences, so it states no rule.

Their builders now live in examples/api_v02/models_static_geometry.py, where the core ledger names
them (FC7). The reference box of each layout is logical and the physical domain is the image of the
map, fixed in time (note, section 4, « How C1 reads the geometry-evolution contract »). Both graphs
resolve to static geometry arrays, and graph (1) runs natively: the constant state stays constant
to rounding on the annulus, because the faces of every cell close, and the cell measures are the
polar ones. Graph (2) runs on a native build of rank one, in
test_n17_static_geometry_graded_native.py.
"""
from __future__ import annotations

import math

import numpy as np
import pops
import pytest
from examples.api_v02.models_static_geometry import (
    ANNULUS_CELLS, ANNULUS_DT, ANNULUS_STEPS, SOUND_SPEED, build_annulus_free_stream,
    build_graded_diffusion)
from pops.codegen._layout_resolution import resolve_native_spatial_layouts
from pops.codegen._native_spatial_layout import native_spatial_layouts
from pops.codegen._static_geometry_consumers import static_geometry_layouts
from pops.mesh import normalize_layout_plan
from tests.python.support.collective_checks import collective_call, collective_check
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.support.public_native_profile import compile_profile_plan_once, native_profile


GRAPHS = {"graph (1) annulus free stream": build_annulus_free_stream,
          "graph (2) graded diffusion": build_graded_diffusion}


@pytest.mark.parametrize("label,kind", [("graph (1) annulus free stream", "chart"),
                                        ("graph (2) graded diffusion", "nodes")])
def test_the_graph_resolves_to_static_geometry_arrays(label, kind):
    """CF15, first half of the flip: the clock-free map of each graph is now read as static
    geometry arrays at resolution (a chart with its stated rule, or node positions), instead of
    the refusal native_geometry_evolution_unavailable of the base."""
    case, layout, _, _ = GRAPHS[label]()
    plan = normalize_layout_plan(layout, owner=case.owner_path.canonical())
    evolution = plan.layouts[0].requirements["geometry_evolution"]
    assert evolution["domain_motion"] == "fixed_physical_domain"
    assert not any(value.get("op") == "time" for value in _nodes(evolution["coordinate_map"]))
    assert native_spatial_layouts(plan)
    assert resolve_native_spatial_layouts(plan)
    resolved = pops.resolve(pops.validate(case), layout=layout)
    (reading,) = static_geometry_layouts(resolved.layout_plan).values()
    assert reading.kind == kind and reading.rank == len(layout.evolution.coordinate_map)


def _nodes(data):
    """Every node of the serialized analytic expressions."""
    pending = list(data)
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            yield node
            pending.extend(node.values())
        elif isinstance(node, list):
            pending.extend(node)


def run_graph(build, *, dimension, scope, steps, dt):
    """Resolve, compile, bind and run one graph natively; return its runtime and its report."""
    _, world = native_profile(dimension=dimension, scope=scope)
    case, layout, initial_values, initial = collective_call(world, build)
    plan = collective_call(world, lambda: pops.resolve(pops.validate(case), layout=layout))
    artifact = compile_profile_plan_once(world, plan, route=scope, compile_artifact=pops.compile)
    runtime = collective_call(world, lambda: pops.bind(
        artifact, initial_values=initial_values,
        resources={"execution_context": artifact_execution_context(artifact)}))
    report = collective_call(world, lambda: pops.run(runtime, t_end=steps*dt, max_steps=steps,
                                                     console=False))
    return world, runtime, report, initial


def perturbed_annulus_state():
    """Cell averages (rho, mx, my) by (theta, r) that vary along both logical axes."""
    radii = 1.0 + (np.arange(ANNULUS_CELLS[0]) + 0.5)/ANNULUS_CELLS[0]
    angles = 2*math.pi*(np.arange(ANNULUS_CELLS[1]) + 0.5)/ANNULUS_CELLS[1]
    r, theta = np.meshgrid(radii, angles)
    return np.stack((1.3 + 0.2*np.cos(theta)*(r - 1.0), 0.7 + 0.1*np.sin(theta),
                     -0.4 + 0.1*np.cos(2*theta)*(r - 1.0)))


def annulus_faces():
    """The area vectors and cell measures of the annulus from its polar map, independently of the
    arrays: the rotated chords between the images of the logical nodes, oriented toward increasing
    logical index, the seam theta = 2 pi identified with theta = 0, and the polar measures."""
    cells_r, cells_theta = ANNULUS_CELLS
    radii = 1.0 + np.arange(cells_r + 1)/cells_r
    angles = 2*math.pi*np.arange(cells_theta + 1)/cells_theta
    x = radii[None, :]*np.cos(angles)[:, None]
    y = radii[None, :]*np.sin(angles)[:, None]
    along_theta = (x[1:, :] - x[:-1, :], y[1:, :] - y[:-1, :])
    along_r = (x[:, 1:] - x[:, :-1], y[:, 1:] - y[:, :-1])
    radial = np.stack((along_theta[1], -along_theta[0]))
    angular = np.stack((-along_r[1], along_r[0]))
    angular[:, -1, :] = angular[:, 0, :]
    measures = 0.5*(radii[1:]**2 - radii[:-1]**2)*(2*math.pi/cells_theta)
    return radial, angular, np.broadcast_to(measures, (cells_theta, cells_r))


def _normal_flux(state, nx, ny):
    rho, mx, my = state
    vx, vy = mx/rho, my/rho
    c2 = SOUND_SPEED*SOUND_SPEED
    return np.stack((mx*nx + my*ny, (mx*vx + c2*rho)*nx + mx*vy*ny,
                     my*vx*nx + (my*vy + c2*rho)*ny))


def _rusanov(left, right, area):
    """F.n |S| of the Rusanov face on the unit normal n of the area vector S, with the speed
    max over both traces of sum_k |n_k| (|v_k| + c)."""
    magnitude = np.hypot(area[0], area[1])
    nx, ny = area[0]/magnitude, area[1]/magnitude

    def bound(state):
        rho, mx, my = state
        return (np.abs(nx)*(np.abs(mx/rho) + SOUND_SPEED)
                + np.abs(ny)*(np.abs(my/rho) + SOUND_SPEED))

    speed = np.maximum(bound(left), bound(right))
    return (0.5*(_normal_flux(left, nx, ny) + _normal_flux(right, nx, ny))
            - 0.5*speed*(right - left))*magnitude


def annulus_oracle(initial, steps, dt):
    """Two authored SSPRK2 stages of the Rusanov finite volumes on the physical faces of the
    annulus, with Outflow ends in r (the ghost repeats the cell) and theta periodic."""
    radial, angular, measures = annulus_faces()

    def rate(state):
        along_r = np.concatenate((state[:, :, :1], state, state[:, :, -1:]), axis=2)
        along_theta = np.concatenate((state[:, -1:, :], state, state[:, :1, :]), axis=1)
        through_r = _rusanov(along_r[:, :, :-1], along_r[:, :, 1:], radial)
        through_theta = _rusanov(along_theta[:, :-1, :], along_theta[:, 1:, :], angular)
        return -((through_r[:, :, 1:] - through_r[:, :, :-1])
                 + (through_theta[:, 1:, :] - through_theta[:, :-1, :]))/measures

    state = np.array(initial, dtype=float)
    for _ in range(steps):
        predictor = state + dt*rate(state)
        state = 0.5*state + 0.5*(predictor + dt*rate(predictor))
    return state


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_graph_1_runs_on_the_physical_faces_of_the_annulus(isolated_native_cache, native_cxx,
                                                           kokkos_root):
    """CF15, second half of the flip for graph (1): the faces read the physical normals and
    measures of the arrays, so a constant Cartesian state stays constant to rounding, the cell
    measures are the polar measures (r2^2 - r1^2) dtheta / 2 (the rule integrates det DX = r
    exactly), and a state that varies along both axes takes the steps of an independent NumPy
    assembly of the same Rusanov faces on the polar chords (on the logical box it would not)."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, runtime, report, initial = run_graph(
        build_annulus_free_stream, dimension=2, scope="N17 graph 1 annulus", steps=ANNULUS_STEPS,
        dt=ANNULUS_DT)
    state = collective_call(world, lambda: runtime.state_global("gas"))
    arrays = collective_call(world, lambda: runtime._executor._s._static_geometry_snapshot())
    perturbed = perturbed_annulus_state()
    world, varied, varied_report, _ = run_graph(
        lambda: build_annulus_free_stream(state=perturbed), dimension=2,
        scope="N17 graph 1 annulus", steps=ANNULUS_STEPS, dt=ANNULUS_DT)
    varied_state = collective_call(world, lambda: varied.state_global("gas"))
    _, _, polar = annulus_faces()
    with collective_check(world):
        assert report.accepted_steps == ANNULUS_STEPS
        assert arrays["kind"] == "chart" and 0.0 <= arrays["closure_ratio"] <= 1.0
        measures = np.asarray(arrays["cell_measure"])
        assert measures.shape == (ANNULUS_CELLS[1], ANNULUS_CELLS[0])
        np.testing.assert_allclose(measures, polar, rtol=1e-13, atol=0.0)
        scale = np.max(np.abs(initial))
        assert np.max(np.abs(np.asarray(state) - initial))/scale <= 1e-13
        assert varied_report.accepted_steps == ANNULUS_STEPS
        expected = annulus_oracle(perturbed, ANNULUS_STEPS, ANNULUS_DT)
        np.testing.assert_allclose(np.asarray(varied_state), expected, rtol=1e-13, atol=0.0)


def _source_rusanov(flux, state):
    """master's Rusanov written in source: the riemann.User body of examples/api_v02/b6_numerics.py,
    the parity body of rusanov() (N3 lot 1)."""
    from examples.api_v02.b6_numerics import rusanov

    return rusanov(flux, state).riemann


@pytest.mark.compiler
@pytest.mark.kokkos
@pytest.mark.native_loader
def test_graph_1_runs_a_face_authored_in_source_on_the_physical_normals(
        isolated_native_cache, native_cxx, kokkos_root):
    """The words of pops.lib.numerics are faces authored in source since N3 lot 1. On the arrays
    such a face reads what a physical normal provides: the normal fluxes F.n, the bound of the
    normal spectral radius and, by name, n, which is the physical unit normal of the face. master's
    Rusanov written in source takes the steps of the oracle of the catalogue face on the chords of
    the annulus, and the stability guard of a source face, which reads the measures of the arrays,
    admits them."""
    del isolated_native_cache, native_cxx, kokkos_root
    perturbed = perturbed_annulus_state()
    world, runtime, report, _ = run_graph(
        lambda: build_annulus_free_stream(state=perturbed, faces=_source_rusanov), dimension=2,
        scope="N17 graph 1 annulus source face", steps=ANNULUS_STEPS, dt=ANNULUS_DT)
    state = collective_call(world, lambda: runtime.state_global("gas"))
    with collective_check(world):
        assert report.accepted_steps == ANNULUS_STEPS
        expected = annulus_oracle(perturbed, ANNULUS_STEPS, ANNULUS_DT)
        np.testing.assert_allclose(np.asarray(state), expected, rtol=1e-13, atol=0.0)
