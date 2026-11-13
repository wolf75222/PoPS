"""N16 (ROM-109), the lot that lifts the table for three words: the oracle tests of N16 5.1 (A1)
for ``CellTotal()``, ``PointValue()`` and ``CellAverage(quadrature=GaussLegendre(3))``, run
natively at rank one on b6 rows (``examples/api_v02/models_N16_three_words.py``).

Under one constant mode c the Riesz map 1/(c² μ(K)) and the weight c of the test compose into
1/(c μ(K)) for every builder that delivers loads, so the updates written by hand are

    dU_K/dt = -(F_+ - F_-) + μ(K) S(U_K/μ(K))       a total (c = 1/μ(K)),
    du_K/dt = -(F_+ - F_-)/μ(K) + S(u_K)            an average, a sample, a stated rule (c = 1),

with F the Rusanov flux of Burgers (FaceFlux) and S(u) = -DECAY u a density (CellRate). Each
word runs from the encode of u0 that the example computes from its closed form and binds as data.

- On N17's 32 graded cells (widths in a ratio of about 92), FaceFlux: a mapped layout refuses a
  named source today, so CellRate is not run there.
- On 16 uniform cells of [0, 3], μ(K) = 3/16, FaceFlux and CellRate: the density enters a total
  multiplied by μ(K) and an average as it is.
- A sample and a stated rule have the rate of an average, so each runs bit for bit as the case
  without a row, from the same coefficients.
"""
from __future__ import annotations

import numpy as np
import pytest

from examples.api_v02.models_N16_three_words import (
    DECAY, GRADED_DT, GRADED_STEPS, UNIFORM_DT, UNIFORM_STEPS, build_words, layout_nodes)
from tests.python.integration.runtime.test_n17_static_geometry_graphs import run_graph
from tests.python.support.collective_checks import collective_call, collective_check

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

_RUNS = {"graded": (GRADED_STEPS, GRADED_DT), "uniform": (UNIFORM_STEPS, UNIFORM_DT)}


def _run(layout, represent, coefficients=None):
    steps, dt = _RUNS[layout]
    world, runtime, report, initial = run_graph(
        lambda: build_words(represent=represent, layout=layout, coefficients=coefficients),
        dimension=1, scope="N16 three words %s %s" % (layout, represent), steps=steps, dt=dt)
    state = collective_call(world, lambda: runtime.state_global("line"))
    return world, report, np.asarray(state)[0], initial[0]


def words_oracle(coefficients, layout, represent, *, scaled=True):
    """SSPRK2 of first-order Rusanov faces with outflow ends (the ghost is the interior value), and
    on the uniform layout the decay density, written on the coefficients of each word.
    ``scaled=False`` adds the density to a total without μ(K), the error that A1 names."""
    steps, dt = _RUNS[layout]
    measures = np.diff(layout_nodes(layout))
    decay = DECAY if layout == "uniform" else 0.0

    def faces(u):
        extended = np.concatenate(([u[0]], u, [u[-1]]))
        left, right = extended[:-1], extended[1:]
        return 0.5*(0.5*left**2 + 0.5*right**2) \
            - 0.5*np.maximum(abs(left), abs(right))*(right - left)

    def rate(c):
        if represent == "total":
            density = c/measures
            return -np.diff(faces(density)) + (measures if scaled else 1.0)*(-decay*density)
        return -np.diff(faces(c))/measures + (-decay*c)

    c = np.array(coefficients, dtype=float)
    for _ in range(steps):
        predictor = c + dt*rate(c)
        c = 0.5*c + 0.5*(predictor + dt*rate(predictor))
    return c


@pytest.mark.parametrize("represent", ["total", "sample", "rule"])
@pytest.mark.parametrize("layout", ["graded", "uniform"])
def test_a_word_of_one_constant_mode_follows_the_update_written_by_hand(
        isolated_native_cache, native_cxx, kokkos_root, layout, represent):
    """N16 5.1 (A1): FaceFlux, and on the uniform layout CellRate of a density, under a total, a
    sample and a stated rule, against the update written by hand on the coefficients."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, state, initial = _run(layout, represent)
    measures = np.diff(layout_nodes(layout))
    expected = words_oracle(initial, layout, represent)
    with collective_check(world):
        assert report.accepted_steps == _RUNS[layout][0]
        if layout == "graded":
            assert measures.max()/measures.min() > 90
        else:
            assert np.all(measures == measures[0]) and measures[0] not in (1.0, 1.0/measures.size)
        np.testing.assert_allclose(state, expected, rtol=1e-13, atol=0.0)


def test_the_density_of_a_total_is_scaled_by_the_measure_and_not_of_an_average(
        isolated_native_cache, native_cxx, kokkos_root):
    """A1, the scale of CellTotal under CellRate: the total divided by μ(K) is the run of the
    average from the same field, and the run is far from the update that adds the density to the
    total without μ(K), the error that A1 names."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, total, initial = _run("uniform", "total")
    _, _, average, _ = _run("uniform", "average")
    measures = np.diff(layout_nodes("uniform"))
    unscaled = words_oracle(initial, "uniform", "total", scaled=False)
    with collective_check(world):
        assert report.accepted_steps == UNIFORM_STEPS
        np.testing.assert_allclose(total/measures, average, rtol=1e-13, atol=0.0)
        assert np.max(abs(total - unscaled)) > 1e-3*np.max(abs(total))


@pytest.mark.parametrize("represent", ["sample", "rule"])
def test_a_sample_and_a_stated_rule_run_bit_for_bit_as_the_average_from_the_same_coefficients(
        isolated_native_cache, native_cxx, kokkos_root, represent):
    """MG1: a sample and a stated rule emit nothing new; bound with the same coefficients, the run
    is the run of the case without a row, bit for bit, on the uniform layout with CellRate."""
    del isolated_native_cache, native_cxx, kokkos_root
    world, report, state, initial = _run("uniform", represent)
    _, _, bare, _ = _run("uniform", None, coefficients=initial)
    with collective_check(world):
        assert report.accepted_steps == UNIFORM_STEPS
        np.testing.assert_array_equal(state, bare)
