"""Named quadrature rules (spec 0.3, 2.6.9 and D0; review of 11/11, P3).

The core knows one quadrature construct, ``pops.Quadrature(nodes=, weights=, interval=)``; a named
rule is a function of this module that returns one, its nodes and weights computed in Python when
the case is written (E4), with no privilege over a rule that a user writes (A4)::

    from pops.lib.quadrature import GaussLegendre, GaussHermite

    numerics = pops.Numerics(represent={phi: pops.CellAverage(quadrature=GaussLegendre(1))})
    layout = {space: pops.Uniform(cells=128), velocity: GaussHermite(24)}

Each rule has its parity test (CF4): ``tests/python/unit/surface/test_quadrature.py`` checks that
the rule integrates exactly what its degree promises and that the same nodes and weights written by
hand as ``pops.Quadrature`` give the same rule.
"""
from __future__ import annotations

import numpy

import pops.surface as pops
from pops.surface.math import infinity


def _order(n, rule):
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise TypeError("%s(n) takes a number of nodes n >= 1, got %r" % (rule, n))
    return n


def GaussLegendre(n):
    """``GaussLegendre(n)``: the n Gauss-Legendre nodes and weights on [-1, 1], exact for the
    polynomials of degree 2n - 1; on a cell or a segment, the plan maps the rule to each axis."""
    nodes, weights = numpy.polynomial.legendre.leggauss(_order(n, "GaussLegendre"))
    return pops.Quadrature(nodes=nodes.tolist(), weights=weights.tolist(), interval=(-1, 1))


def GaussHermite(n):
    """``GaussHermite(n)``: the n Gauss-Hermite nodes v_k on (-infinity, infinity) with the
    measures w_k exp(v_k^2), w_k the Gauss-Hermite weights, so that the sum of
    w_k exp(v_k^2) f(v_k) is exact for f = p exp(-v^2) with p of degree 2n - 1 (D0)."""
    nodes, weights = numpy.polynomial.hermite.hermgauss(_order(n, "GaussHermite"))
    measures = weights * numpy.exp(nodes * nodes)
    return pops.Quadrature(nodes=nodes.tolist(), weights=measures.tolist(),
                           interval=(-infinity, infinity))


__all__ = ["GaussHermite", "GaussLegendre"]
