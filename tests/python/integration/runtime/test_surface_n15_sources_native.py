"""N15 FaceFlux/CellRate lot FC-1 (ROM-98, note §4), native: a source of the surface under
``pointwise()``, compiled, bound and run at rank one.

The case is du/dt = -div(a u) + S, S = -k u (a = 1, k = 2), on 16 periodic cells of [0, 3]
(μ(K) = 3/16), with ``upwind()``, the SSPRK2 of ``pops.lib.time.methods`` and a fixed step. Its
initial value is the datum u0, the coefficients of each representation supplied at bind.

- Under ``CellAverage()`` the run follows the update written in NumPy: the upwind face of a u is
  a u_left, the source -k u_K, and SSPRK2 (note §4).
- Under ``CellTotal()``, bound with μ(K) times the averages, the run is μ(K) times the run under
  ``CellAverage()``, to rounding: the named source reads ū = U/μ(K) and its rate is multiplied by
  μ(K), as the rate of the transport (G03 lot R-b, route line 6; N16 5.1, A1).
- Under ``PointValue()`` the value read at the centre is the rate itself: the run is the run under
  ``CellAverage()``, bit for bit.
- ``- S`` with S = k u runs as ``+ S`` with S = -k u.
- A source alone, by forward Euler, is u_{n+1} = (1 - k dt) u_n in each cell, to rounding.
"""
from pathlib import Path

import numpy as np
import pytest

import pops.surface as surface
from pops.lib.numerics import pointwise, upwind
from pops.lib.representations import CellAverage, CellTotal, PointValue
from pops.lib.time.methods import forward_euler, ssprk2
from pops.surface.math import ddt, div

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = 16
LENGTH = 3.0
DT = 1e-3
UNTIL = 0.02
STEPS = 20
DECAY = 2.0
WORDS = {"average": CellAverage, "total": CellTotal, "sample": PointValue}


def _case(word="average", minus=False, transport=True):
    domain = surface.Box(x=surface.Periodic(0, LENGTH))
    model = surface.Model("decaying_advection")
    u = model.quantity("u", surface.Scalar(domain))
    a = model.constant("a", value=1.0)
    k = model.parameter("k", default=DECAY)
    u0 = model.data("u0", surface.Scalar(domain))
    schemes = {}
    if minus:
        decay = model.source(of=u, value=k * u)
        source = -decay
    else:
        decay = model.source(of=u, value=-k * u)
        source = decay
    schemes[decay] = pointwise()
    if transport:
        advection = model.flux(of=u, value=a * u)
        schemes[advection] = upwind()
        model.balance(ddt(u) == -div(advection) + source)
        method = ssprk2(rate=model)
    else:
        model.balance(ddt(u) == source)
        method = forward_euler(rate=model)
    numerics = surface.Numerics(represent={u: WORDS[word]()}, schemes=schemes)
    program = surface.Program(advance=surface.Advance(method, dt=DT), initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={domain: surface.Uniform(cells=CELLS)})
    return case, u, u0


ROWS = {"average": {}, "total": {}, "sample": {}, "minus": {"minus": True},
        "alone": {"transport": False}}


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache-n15-sources")
    rows = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        for name, options in ROWS.items():
            word = name if name in WORDS else "average"
            case, u, u0 = _case(word, **options)
            rows[name] = (surface.compile(case, target=surface.OpenMP()), u, u0)
    return rows


def _averages():
    centres = (np.arange(CELLS) + 0.5) * LENGTH / CELLS
    return 1.0 + 0.5 * np.sin(2 * np.pi * centres / LENGTH)


def _run(artifacts, name, coefficients):
    artifact, u, u0 = artifacts[name]
    result = surface.bind(artifact, data={u0: coefficients}).run(until=UNTIL)
    return np.asarray(result.state[u], dtype=float)


def _oracle(values):
    h = LENGTH / CELLS

    def rate(u):
        return -(u - np.roll(u, 1)) / h - DECAY * u

    for _ in range(STEPS):
        first = values + DT * rate(values)
        values = (values + first + DT * rate(first)) / 2
    return values


def test_a_source_under_an_average_follows_the_update_written_by_hand(artifacts):
    np.testing.assert_allclose(_run(artifacts, "average", _averages()), _oracle(_averages()),
                               rtol=1e-12, atol=0)


def test_a_source_under_a_total_runs_the_average_times_the_measure(artifacts):
    measure = LENGTH / CELLS
    average = _run(artifacts, "average", _averages())
    total = _run(artifacts, "total", measure * _averages())
    assert np.all(np.isfinite(average))
    assert not np.allclose(average, _averages())  # the case moved
    np.testing.assert_allclose(total / measure, average, rtol=1e-13, atol=0)


def test_a_source_under_a_sample_runs_the_average_bit_for_bit(artifacts):
    np.testing.assert_array_equal(_run(artifacts, "sample", _averages()),
                                  _run(artifacts, "average", _averages()))


def test_minus_a_source_runs_as_the_source_of_opposite_sign(artifacts):
    np.testing.assert_allclose(_run(artifacts, "minus", _averages()),
                               _run(artifacts, "average", _averages()), rtol=1e-14, atol=0)


def test_a_source_alone_by_forward_euler_decays_in_each_cell(artifacts):
    expected = _averages() * (1.0 - DECAY * DT) ** STEPS
    np.testing.assert_allclose(_run(artifacts, "alone", _averages()), expected, rtol=1e-13,
                               atol=0)
