"""Native 2D extrusion of a three-state coordinated face against a pure FV oracle.

The two signed side sources split the exact path integral 30/70. A legacy
Rusanov 50/50 path dispatch cannot match this witness. Run only after the
corresponding source and native artifact have been rebuilt together.
"""

import numpy as np
import pytest

import pops
from tests.python.support.api040_coordinated_face_oracle_independent import (
    COMPONENTS, PERMUTATIONS, forward_euler, initial_cell_means)
from tests.python.support.native_execution_context import artifact_execution_context
from tests.python.unit.numerics.test_api040_coordinated_face_public_independent import _case


pytestmark = [pytest.mark.compiler, pytest.mark.native_loader]


@pytest.mark.parametrize("order", PERMUTATIONS)
def test_native_asymmetric_face_matches_independent_oracle_and_preserves_transverse_invariance(
        isolated_native_cache, native_cxx, kokkos_root, order):
    del isolated_native_cache, native_cxx, kokkos_root
    cells, steps, beta, gamma = 8, 8, .7, -.4
    dt = .05/cells
    case, layout, _ = _case(order, beta=beta, gamma=gamma, dimension=2)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    initial_canonical = initial_cell_means(cells)
    initial_ordered = initial_canonical[[COMPONENTS.index(name) for name in order]]
    initial = np.broadcast_to(initial_ordered[:, None, :], (3, cells, cells)).copy()
    expected_line = forward_euler(cells, beta=beta, gamma=gamma, order=order, steps=steps)
    expected = np.broadcast_to(expected_line[:, None, :], initial.shape)

    runtime = pops.bind(artifact, initial_state={"transport": initial},
                        resources={"execution_context": artifact_execution_context(artifact)})
    report = pops.run(runtime, t_end=steps*dt, max_steps=steps)
    assert report.accepted_steps == steps
    actual = np.asarray(runtime.state_global("transport")).reshape(initial.shape)
    np.testing.assert_allclose(actual, expected, rtol=0., atol=4.e-13)
    np.testing.assert_allclose(actual,
        np.broadcast_to(actual[:, :1, :], actual.shape), rtol=0., atol=4.e-13)
    assert np.max(np.abs(actual-initial)) > 1.e-5


def test_one_coordinated_artifact_two_bind_time_dissipations_match_separate_oracles(
        isolated_native_cache, native_cxx, kokkos_root):
    del isolated_native_cache, native_cxx, kokkos_root
    cells, beta, gamma, dt = 8, .7, -.4, .05/8
    case, layout, _, parameter = _case(dimension=2, runtime_margin=True)
    artifact = pops.compile(pops.resolve(pops.validate(case), layout=layout))
    line = initial_cell_means(cells)
    initial = np.broadcast_to(line[:, None, :], (3, cells, cells)).copy()
    results = []
    for margin in (.5, 1.5):
        runtime = pops.bind(artifact, initial_state={"transport": initial.copy()},
                            params={parameter: margin},
                            resources={"execution_context": artifact_execution_context(artifact)})
        report = pops.run(runtime, t_end=dt, max_steps=1)
        assert report.accepted_steps == 1
        actual = np.asarray(runtime.state_global("transport")).reshape(initial.shape)
        expected_line = forward_euler(cells, beta=beta, gamma=gamma, steps=1,
                                      margin=margin)
        expected = np.broadcast_to(expected_line[:, None, :], actual.shape)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=4.e-13)
        results.append(actual)
    assert np.max(np.abs(results[0]-results[1])) > 1.e-6
