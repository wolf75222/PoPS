"""N15 FaceFlux/CellRate lot FC-2 (ROM-98, note §3.6, §4), native: ``cell.load(e)`` and the
coordinates read at the centre of the cell, compiled, bound and run at rank one.

The cases are a source alone, du/dt = S, by forward Euler with a fixed step on 16 periodic cells of
[0, 3] (h = 3/16), so that each cell follows its own update, written here in NumPy:

- ``CellRate(cell.load(sin(2 pi x / 3)))`` under ``CellAverage(quadrature=GaussLegendre(2))``: the
  rate is the Gauss mean of the forcing on the cell, m_K = Σ_q (w_q / 2) sin(2 pi x_q / 3) with
  x_q = x_K + (h / 2) ξ_q, and u_{n+1} = u_n + dt m_K;
- ``CellRate(-x * cell.value(u))`` under ``CellAverage()``: x is the centre x_K, and
  u_{n+1} = u_n - dt x_K u_n;
- the law -x u of a source under ``pointwise()`` runs as that scheme, bit for bit.
"""
from pathlib import Path

import numpy as np
import pytest

import pops.surface as surface
from pops.lib.numerics import pointwise
from pops.lib.quadrature import GaussLegendre
from pops.lib.representations import CellAverage
from pops.lib.time.methods import forward_euler
from pops.surface.discrete import CellRate
from pops.surface.math import ddt, pi, sin

pytestmark = [pytest.mark.compiler, pytest.mark.kokkos, pytest.mark.native_loader]

REPO_ROOT = Path(__file__).resolve().parents[4]
CELLS = 16
LENGTH = 3.0
DT = 1e-3
UNTIL = 0.02
STEPS = 20
DOMAIN = surface.Box(x=surface.Periodic(0, LENGTH))
X = DOMAIN.x


@surface.scheme
def forcing(cell, term):
    return CellRate(cell.load(sin(2 * pi * X / LENGTH)))


@surface.scheme
def at_centre(cell, term):
    q = term.quantity
    return CellRate(-X * cell.value(q))


def _case(name):
    model = surface.Model("cell_load_" + name)
    u = model.quantity("u", surface.Scalar(DOMAIN))
    u0 = model.data("u0", surface.Scalar(DOMAIN))
    if name == "load":
        source = model.source(of=u, value=0 * u)
        scheme, representation = forcing(), CellAverage(quadrature=GaussLegendre(2))
    elif name == "centre":
        source = model.source(of=u, value=-u)
        scheme, representation = at_centre(), CellAverage()
    else:
        source = model.source(of=u, value=-X * u)
        scheme, representation = pointwise(), CellAverage()
    model.balance(ddt(u) == source)
    numerics = surface.Numerics(represent={u: representation}, schemes={source: scheme})
    program = surface.Program(advance=surface.Advance(forward_euler(rate=model), dt=DT),
                              initial={u: u0})
    case = surface.Case(model=model, numerics=numerics, program=program,
                        layout={DOMAIN: surface.Uniform(cells=CELLS)})
    return case, u, u0


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory, native_cxx, kokkos_root):
    del native_cxx, kokkos_root
    cache = tmp_path_factory.mktemp("pops-native-cache-n15-cell-load")
    rows = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("POPS_CACHE_DIR", str(cache))
        patch.setenv("POPS_NATIVE_CACHE_DIR", str(cache))
        patch.chdir(REPO_ROOT)
        for name in ("load", "centre", "law"):
            case, u, u0 = _case(name)
            rows[name] = (surface.compile(case, target=surface.OpenMP()), u, u0)
    return rows


def _centres():
    return (np.arange(CELLS) + 0.5) * LENGTH / CELLS


def _initial():
    return 1.0 + 0.5 * np.cos(2 * np.pi * _centres() / LENGTH)


def _run(artifacts, name):
    artifact, u, u0 = artifacts[name]
    result = surface.bind(artifact, data={u0: _initial()}).run(until=UNTIL)
    return np.asarray(result.state[u], dtype=float)


def test_a_load_under_a_rule_runs_the_gauss_mean_of_its_integrand(artifacts):
    nodes, weights = np.polynomial.legendre.leggauss(2)
    h = LENGTH / CELLS
    points = _centres()[:, None] + (h / 2) * nodes[None, :]
    mean = (np.sin(2 * np.pi * points / LENGTH) * (weights / 2)[None, :]).sum(axis=1)
    expected = _initial()
    for _ in range(STEPS):
        expected = expected + DT * mean
    run = _run(artifacts, "load")
    assert not np.allclose(run, _initial())
    np.testing.assert_allclose(run, expected, rtol=1e-13, atol=1e-15)


def test_a_coordinate_read_by_a_cell_rate_is_the_centre_of_the_cell(artifacts):
    expected = _initial()
    for _ in range(STEPS):
        expected = expected + DT * (-_centres() * expected)
    np.testing.assert_allclose(_run(artifacts, "centre"), expected, rtol=1e-13, atol=0)


def test_the_law_of_a_source_under_pointwise_reads_the_centre_as_the_cell_rate(artifacts):
    np.testing.assert_array_equal(_run(artifacts, "law"), _run(artifacts, "centre"))
