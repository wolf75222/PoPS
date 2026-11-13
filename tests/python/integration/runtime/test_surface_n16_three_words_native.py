"""N16 (ROM-109), the lot that lifts the table for three words, native: a case of the surface under
``CellTotal()``, ``PointValue()`` and ``CellAverage(quadrature=GaussLegendre(3))`` of
``pops.lib.representations``, compiled, bound and run at rank one.

The case is the advection of L1 of §17.1 on 16 periodic cells of [0, 3] (μ(K) = 3/16):
``upwind()``, the SSPRK2 of ``pops.lib.time.methods``, a fixed step and the default acceptance test
``finite()``. Its diffusion is left out: G03 lot R-b refuses the StencilFlux of ``two_point()``
under a total until a case runs it. Its initial value is the datum u0, the coefficients of each
representation supplied at bind (the projection by encode is N16 lot 2's).

- A total, bound with μ(K) times the averages, runs the averages times μ(K), to rounding: the row
  of b6 decodes ū = U/μ(K), runs today's rate and encodes μ(K) R (G03 lot R-b; N16 5.1, A1).
- A sample and a stated rule have the rate of an average: bound with the same coefficients, each
  runs bit for bit as the case under ``CellAverage()``.
- The arrays that a run publishes keep the identity of their representation (K4): the totals of a
  run are refused as the datum of the case under averages, with the same shape.
"""
from pathlib import Path

import numpy as np
import pytest

import pops.surface as surface
from pops.lib.numerics import upwind
from pops.lib.quadrature import GaussLegendre
from pops.lib.representations import CellAverage, CellTotal, PointValue
from pops.lib.time.methods import ssprk2
from pops.surface.math import ddt, div

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = 16
LENGTH = 3.0
DT = 1e-3
UNTIL = 0.02
WORDS = {"average": CellAverage, "total": CellTotal, "sample": PointValue,
         "rule": lambda: CellAverage(quadrature=GaussLegendre(3))}


def _case(word):
    domain = surface.Box(x=surface.Periodic(0, LENGTH))
    model = surface.Model("advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    u0 = model.data("u0", surface.Scalar(domain))
    advection = model.flux(of=u, value=a * u)
    model.balance(ddt(u) == -div(advection))
    numerics = surface.Numerics(represent={u: WORDS[word]()}, schemes={advection: upwind()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=DT),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return case, u, u0


def _averages():
    centres = (np.arange(CELLS) + 0.5) * LENGTH / CELLS
    return 1.0 + 0.5 * np.sin(2 * np.pi * centres / LENGTH)


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache-n16-words")
    rows = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        for word in WORDS:
            case, u, u0 = _case(word)
            rows[word] = (surface.compile(case, target=surface.OpenMP()), u, u0)
    return rows


def _run(artifacts, word, coefficients):
    artifact, u, u0 = artifacts[word]
    result = surface.bind(artifact, data={u0: coefficients}).run(until=UNTIL)
    return result, result.state[u]


def test_a_total_runs_the_average_times_the_measure(artifacts):
    """CellTotal() at the surface: bound with μ(K) ū, the run is μ(K) times the run of the
    averages, to rounding, and its inventory is the average's."""
    measure = LENGTH / CELLS
    averages = _averages()
    total_run, total = _run(artifacts, "total", measure * averages)
    average_run, average = _run(artifacts, "average", averages)
    assert total_run.steps == average_run.steps == round(UNTIL / DT)
    np.testing.assert_allclose(np.asarray(total) / measure, np.asarray(average), rtol=1e-13,
                               atol=0.0)
    inventory = np.sum(averages) * measure
    assert abs(np.sum(total) - inventory) <= 1e-13 * inventory


@pytest.mark.parametrize("word", ["sample", "rule"])
def test_a_sample_and_a_stated_rule_run_bit_for_bit_as_the_average(artifacts, word):
    """PointValue() and CellAverage(quadrature=GaussLegendre(3)) at the surface: from the same
    coefficients, the run is the run under CellAverage(), bit for bit (MG1)."""
    coefficients = _averages()
    _, state = _run(artifacts, word, coefficients)
    _, average = _run(artifacts, "average", coefficients)
    assert np.asarray(state).tobytes() == np.asarray(average).tobytes()


def test_published_totals_are_not_the_datum_of_the_averages(artifacts):
    """K4: the totals that a run publishes carry the identity of CellTotal(); bound as the datum
    of the case under CellAverage(), of the same shape and support, they are refused."""
    _, totals = _run(artifacts, "total", LENGTH / CELLS * _averages())
    artifact, _, u0 = artifacts["average"]
    with pytest.raises(TypeError) as caught:
        surface.bind(artifact, data={u0: totals})
    assert "of kind 'total'" in str(caught.value) and "of kind 'average'" in str(caught.value)
