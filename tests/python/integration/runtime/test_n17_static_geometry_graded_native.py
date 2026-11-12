"""Graph (2) of C1 on a native build of rank one: diffusion on graded node positions (N17, CF15).

The 32 cells of the layout are the images x = tanh(3 s) / tanh(3) of a uniform logical segment
(examples/api_v02/models_static_geometry.py builds the Case). The static geometry arrays of these
node positions are generated at bind; the diffusion of the Program reads their distances, face
measures and cell measures instead of the uniform spacing of the logical segment. The run is
checked against an independent NumPy assembly of the same two-point scheme on the graded cells, and
the inventory sum |K| u of the no-flux segment is kept to rounding.
"""
from __future__ import annotations

import numpy as np
import pytest

from examples.api_v02.models_static_geometry import (
    CONDUCTIVITY, GRADED_CELLS, GRADED_DT, GRADED_STEPS, build_graded_diffusion, graded_nodes)
from tests.python.integration.runtime.test_n17_static_geometry_graphs import run_graph
from tests.python.support.collective_checks import collective_call, collective_check

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]


def graded_oracle(nodes, initial, steps, dt):
    """Two authored SSPRK2 stages of the two-point flux on cells [x_i, x_i+1] with no-flux ends."""
    measures = np.diff(nodes)
    centroids = 0.5*(nodes[:-1] + nodes[1:])
    distances = np.diff(centroids)

    def rate(u):
        faces = np.zeros(len(nodes))
        faces[1:-1] = CONDUCTIVITY*np.diff(u)/distances
        return np.diff(faces)/measures

    u = np.array(initial, dtype=float)
    for _ in range(steps):
        predictor = u + dt*rate(u)
        u = 0.5*u + 0.5*(predictor + dt*rate(predictor))
    return u


def test_graph_2_diffuses_on_the_graded_node_positions(isolated_native_cache, native_cxx,
                                                       kokkos_root):
    """CF15, second half of the flip for graph (2): the widths are the node differences, bitwise,
    the run is the two-point scheme on the graded cells, and the inventory stays to rounding."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, runtime, report, initial = run_graph(
        build_graded_diffusion, dimension=1, scope="N17 graph 2 graded", steps=GRADED_STEPS,
        dt=GRADED_DT)
    state = collective_call(world, lambda: runtime.state_global("rod"))
    arrays = collective_call(world, lambda: runtime._executor._s._static_geometry_snapshot())
    with collective_check(world):
        assert report.accepted_steps == GRADED_STEPS
        assert arrays["kind"] == "nodes"
        nodes = np.asarray(arrays["nodes"][0])
        np.testing.assert_allclose(nodes, graded_nodes(), rtol=0.0, atol=4e-16)
        widths = np.asarray(arrays["width"][0])
        measures = np.asarray(arrays["cell_measure"])
        assert widths.shape == (GRADED_CELLS,)
        assert np.array_equal(widths, np.diff(nodes)) and np.array_equal(measures, np.diff(nodes))
        expected = graded_oracle(nodes, initial[0], GRADED_STEPS, GRADED_DT)
        actual = np.asarray(state)[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0.0)
        inventory = np.sum(measures*initial[0])
        assert abs(np.sum(measures*actual) - inventory) <= 1e-13*inventory
