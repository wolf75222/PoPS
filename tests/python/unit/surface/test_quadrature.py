"""``pops.Quadrature`` in the core and the named rules of ``pops.lib.quadrature`` (spec 0.3, D0,
2.6.2, 2.6.9; review of 11/11, P3).

The core knows one quadrature construct, ``pops.Quadrature(nodes=, weights=, interval=)``; the core
names no rule (CF2). ``GaussLegendre(n)`` and ``GaussHermite(n)`` are functions of the library that
return one, each with its parity test (CF4): the rule integrates exactly what its degree promises,
and the same nodes and weights written by hand give the same rule, so the library has no privilege
over a rule that a user writes (A4).
"""
from __future__ import annotations

import math

import pytest

import pops.surface as surface
from pops.lib import quadrature
from pops.lib.numerics.schemes import two_point, upwind
from pops.lib.quadrature import GaussHermite, GaussLegendre
from pops.lib.time.methods import ssprk2
from pops.surface.math import ddt, div, grad, infinity, pi


def _integrate(rule, f):
    return math.fsum(w * f(v) for v, w in zip(rule.nodes, rule.weights))


# --- the named rules of the library, with their parity tests (CF4) ---------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 16])
def test_gauss_legendre_is_exact_up_to_degree_2n_minus_1(n):
    rule = GaussLegendre(n)
    assert isinstance(rule, surface.Quadrature)
    assert (rule.n, rule.interval) == (n, (-1.0, 1.0))
    for k in range(2 * n):
        exact = 2.0 / (k + 1) if k % 2 == 0 else 0.0
        assert _integrate(rule, lambda v: v ** k) == pytest.approx(exact, abs=1e-13), k
    # the degree is the rule's: x^(2n) is not integrated exactly
    assert abs(_integrate(rule, lambda v: v ** (2 * n)) - 2.0 / (2 * n + 1)) > 1e-12


@pytest.mark.parametrize("n", [1, 2, 5, 24])
def test_gauss_hermite_is_exact_for_p_exp_minus_v2_up_to_degree_2n_minus_1(n):
    rule = GaussHermite(n)
    assert isinstance(rule, surface.Quadrature)
    assert (rule.n, rule.interval) == (n, (-math.inf, math.inf))
    for k in range(2 * n):
        exact = math.gamma((k + 1) / 2) if k % 2 == 0 else 0.0
        value = _integrate(rule, lambda v: v ** k * math.exp(-v * v))
        assert value == pytest.approx(exact, rel=1e-10, abs=1e-12), k


@pytest.mark.parametrize("rule, n", [(GaussLegendre, 4), (GaussHermite, 24)])
def test_a_named_rule_is_the_rule_written_by_hand(rule, n):
    named = rule(n)
    written = surface.Quadrature(nodes=list(named.nodes), weights=list(named.weights),
                                 interval=named.interval)
    assert (written.nodes, written.weights, written.interval) \
        == (named.nodes, named.weights, named.interval)


@pytest.mark.parametrize("rule", [GaussLegendre, GaussHermite])
@pytest.mark.parametrize("n", [0, -1, True, 2.0, "3"])
def test_a_named_rule_takes_a_number_of_nodes(rule, n):
    with pytest.raises(TypeError, match=r"number of nodes n >= 1"):
        rule(n)


def test_the_library_module_lists_its_rules():
    assert quadrature.__all__ == ["GaussHermite", "GaussLegendre"]


# --- pops.Quadrature, a rule of one's own ----------------------------------------------------

def test_a_rule_of_ones_own_realizes_its_integrals():
    simpson = surface.Quadrature(nodes=[-1, 0, 1], weights=[1 / 3, 4 / 3, 1 / 3], interval=(-1, 1))
    assert simpson.n == 3
    assert _integrate(simpson, lambda v: v ** 3 + v ** 2) == pytest.approx(2 / 3)
    # an end of the interval may be a node, a weight may be negative, an end may be pi; the
    # nodes and weights are Python numbers (E4), never a symbol
    lobatto = surface.Quadrature(nodes=(0.0, math.pi), weights=(math.pi / 2, math.pi / 2),
                                 interval=(0, pi))
    assert lobatto.interval == (0.0, math.pi) and lobatto.nodes == (0.0, math.pi)
    assert surface.Quadrature(nodes=[0.5], weights=[-1.0], interval=(0, 1)).weights == (-1.0,)
    unbounded = surface.Quadrature(nodes=[0.0], weights=[math.sqrt(math.pi)],
                                   interval=(-infinity, infinity))
    assert unbounded.interval == (-math.inf, math.inf)
    assert repr(simpson).startswith("pops.Quadrature(nodes=(-1.0, 0.0, 1.0), weights=")


@pytest.mark.parametrize("kwargs, kind, words", [
    (dict(nodes=[0.0, 0.5], weights=[1.0], interval=(-1, 1)), ValueError, "as many weights"),
    (dict(nodes=[], weights=[], interval=(-1, 1)), ValueError, "at least one number"),
    (dict(nodes="0.5", weights=[1.0], interval=(-1, 1)), TypeError, "sequence of numbers"),
    (dict(nodes=0.5, weights=[1.0], interval=(-1, 1)), TypeError, "sequence of numbers"),
    (dict(nodes=[True], weights=[1.0], interval=(-1, 1)), TypeError, "finite Python numbers"),
    (dict(nodes=[0.0], weights=[math.nan], interval=(-1, 1)), TypeError, "finite Python numbers"),
    (dict(nodes=[math.inf], weights=[1.0], interval=(-infinity, infinity)), TypeError,
     "finite Python numbers"),
    (dict(nodes=[pi], weights=[1.0], interval=(0, 4)), TypeError, "finite Python numbers"),
    (dict(nodes=[0.0], weights=[2.0], interval=[-1, 1]), TypeError, "reference interval"),
    (dict(nodes=[0.0], weights=[2.0], interval=(1, -1)), ValueError, "lower < upper"),
    (dict(nodes=[1.5], weights=[2.0], interval=(-1, 1)), ValueError, "outside the interval"),
])
def test_a_rule_that_is_not_data_is_refused(kwargs, kind, words):
    with pytest.raises(kind, match=words):
        surface.Quadrature(**kwargs)


def test_the_core_knows_no_rule_by_name():
    assert "Quadrature" in surface.__all__
    assert not {"GaussLegendre", "GaussHermite"} & set(surface.__all__)
    for name in ("GaussLegendre", "GaussHermite"):
        with pytest.raises(surface.SymbolNotInSurface, match="pops.lib.quadrature"):
            getattr(surface, name)


def test_a_rule_as_the_layout_of_a_support_waits_for_g03():
    domain = surface.Box(x=surface.Periodic(0, 1))
    model = surface.Model("advection_diffusion")
    u = model.quantity("u", surface.Scalar(domain))
    advection = model.flux(of=u, value=u)
    diffusion = model.diffusive_flux(of=u, value=-1e-2 * grad(u))
    model.balance(ddt(u) == -div(advection + diffusion))
    u0 = model.data("u0", surface.Scalar(domain))
    numerics = surface.Numerics(represent={u: surface.CellAverage()},
                                schemes={advection: upwind(), diffusion: two_point()})
    program = surface.Program(advance=surface.Advance(ssprk2(rate=model), dt=1e-3),
                              initial={u: u0})
    with pytest.raises(surface.NotYetElaborated, match="supports of G03"):
        surface.Case(model=model, numerics=numerics, program=program,
                     layout={domain: GaussLegendre(2)})
