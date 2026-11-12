"""G03 lot R-b (ROM-21): a state represented by cell totals, run natively (note §4.3, §4.6).

The lowering decodes ū = U/μ(K) into a stage field, runs today's rate on ū (the boundary session
fills the ghosts of ū by the laws of the case) and encodes the rate of U as μ(K) times it. Each
test runs a total and checks it against the update written by hand on ū, or against the same
case run as an average, which is today's route: a total divided by its cell measures is the
average to rounding.
"""
from __future__ import annotations

import numpy as np
import pytest

from examples.api_v02.models_G03_totals import (
    BURGERS_CELLS, BURGERS_DT, BURGERS_STEPS, DIFFUSION_DT, DIFFUSION_STEPS, GRADED_BURGERS_DT,
    GRADED_BURGERS_STEPS, INFLOW_VALUE, build_burgers, build_graded_burgers,
    build_graded_diffusion_total)
from examples.api_v02.models_static_geometry import graded_nodes
from tests.python.integration.runtime.test_n17_static_geometry_graded_native import graded_oracle
from tests.python.integration.runtime.test_n17_static_geometry_graphs import run_graph
from tests.python.support.collective_checks import collective_call, collective_check

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def _run(build, scope, steps, dt, *, block="line", **keywords):
    world, runtime, report, initial = run_graph(
        lambda: build(**keywords), dimension=1, scope=scope, steps=steps, dt=dt)
    state = collective_call(world, lambda: runtime.state_global(block))
    return world, report, np.asarray(state)[0], initial


def burgers_oracle(initial, steps, dt, *, periodic):
    """SSPRK2 of first-order Rusanov faces on 16 uniform cells, written on the density.

    Not periodic: the ghost 2g - u_0 at the lower end, the face flux zeroed at the upper end."""
    h = 1.0/BURGERS_CELLS

    def rate(u):
        if periodic:
            left, right = np.roll(u, 1), u
            faces = 0.5*(0.5*left**2 + 0.5*right**2) \
                - 0.5*np.maximum(abs(left), abs(right))*(right - left)
            return -(np.roll(faces, -1) - faces)/h
        extended = np.concatenate(([2*INFLOW_VALUE - u[0]], u))
        left, right = extended[:-1], extended[1:]
        faces = np.zeros(BURGERS_CELLS + 1)
        faces[:-1] = 0.5*(0.5*left**2 + 0.5*right**2) \
            - 0.5*np.maximum(abs(left), abs(right))*(right - left)
        return -np.diff(faces)/h

    u = np.array(initial, dtype=float)
    for _ in range(steps):
        predictor = u + dt*rate(u)
        u = 0.5*u + 0.5*(predictor + dt*rate(predictor))
    return u


@pytest.mark.parametrize("periodic", [False, True], ids=["inflow-and-closed", "periodic"])
def test_a_total_runs_the_density_update_with_its_ghosts_on_the_density(
        isolated_native_cache, native_cxx, kokkos_root, periodic):
    """Burgers as a total: ū = U/h follows the update written by hand on ū (the inflow ghost
    2g - ū, the closed end, or the periodic seam), and equals the run of the average."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, total, initial = _run(
        build_burgers, "G03 R-b burgers total", BURGERS_STEPS, BURGERS_DT, represent="total",
        periodic=periodic)
    _, _, average, _ = _run(
        build_burgers, "G03 R-b burgers average", BURGERS_STEPS, BURGERS_DT,
        represent="average", periodic=periodic)
    h = 1.0/BURGERS_CELLS
    expected = burgers_oracle(initial[0], BURGERS_STEPS, BURGERS_DT, periodic=periodic)
    with collective_check(world):
        assert report.accepted_steps == BURGERS_STEPS
        np.testing.assert_allclose(total/h, expected, rtol=1e-13, atol=0.0)
        np.testing.assert_allclose(total/h, average, rtol=1e-14, atol=0.0)
        if periodic:
            inventory = np.sum(initial[0])*h
            assert abs(np.sum(total) - inventory) <= 1e-14*inventory


def test_burgers_as_a_total_on_the_graded_cells_is_the_average(
        isolated_native_cache, native_cxx, kokkos_root):
    """O-e of the review of rev 2.3: on N17's graded cells, widths in a ratio of about 92, the
    total divided by the measures is the run of the average, at the same fixed step."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, total, _ = _run(
        build_graded_burgers, "G03 R-b graded burgers total", GRADED_BURGERS_STEPS,
        GRADED_BURGERS_DT, represent="total")
    _, _, average, _ = _run(
        build_graded_burgers, "G03 R-b graded burgers average", GRADED_BURGERS_STEPS,
        GRADED_BURGERS_DT, represent="average")
    measures = np.diff(graded_nodes())
    with collective_check(world):
        assert report.accepted_steps == GRADED_BURGERS_STEPS
        assert measures.max()/measures.min() > 90
        np.testing.assert_allclose(total/measures, average, rtol=1e-13, atol=0.0)


def test_a_diffusion_of_a_total_is_the_two_point_scheme_on_its_density(
        isolated_native_cache, native_cxx, kokkos_root):
    """The graded diffusion of N17 under a total: ū = U/|K| follows the independent assembly of
    the two-point scheme on the graded cells, and the inventory Σ U is kept to rounding."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, total, initial = _run(
        build_graded_diffusion_total, "G03 R-b graded diffusion total", DIFFUSION_STEPS,
        DIFFUSION_DT, block="rod")
    nodes = graded_nodes()
    measures = np.diff(nodes)
    expected = graded_oracle(nodes, initial[0], DIFFUSION_STEPS, DIFFUSION_DT)
    with collective_check(world):
        assert report.accepted_steps == DIFFUSION_STEPS
        np.testing.assert_allclose(total/measures, expected, rtol=1e-13, atol=0.0)
        inventory = np.sum(measures*initial[0])
        assert abs(np.sum(total) - inventory) <= 1e-13*inventory
