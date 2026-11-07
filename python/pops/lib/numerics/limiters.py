"""Slope limiters written in Python (spec 0.3 section 4.4, catalogue of ``pops.lib.numerics``; N3).

A limiter is a plain function ``limiter(a, b)`` of the backward and forward differences of a cell,
passed uncalled: ``muscl(Uh, vanleer)``. It runs once, when the reconstruction is authored, and
returns the expression of the limited slope, which becomes the generated C++ of the face (G2).
No native limiter is named.

Each function writes the formula of its catalogue row and keeps the guards of the b6 native
policy of the same name (``include/pops/numerics/fv/reconstruction.hpp``), so that the two
agree on every input, not only where the row's formula is finite:

* a non-finite difference is returned as the slope, so that the face guard of the flux refuses
  it instead of a limited value hiding it (the first finite check returns ``a``);
* where the row's formula would overflow or underflow, the same value is formed from the
  smaller magnitude, as the native policy forms it.

On finite differences of ordinary size each function is the row's formula, in the native order
of operations.
"""
from __future__ import annotations

import sys
from typing import Any

from pops import math as pm

# The largest and the smallest normal binary64 values: std::numeric_limits<double>::max() and
# ::min() of the native policies.
_LARGEST = sys.float_info.max
_SMALLEST_NORMAL = sys.float_info.min


def _finite(value: Any) -> Any:
    # NaN fails both comparisons, and so does an infinity of either sign.
    return (value <= _LARGEST) & (value >= -_LARGEST)


def _guarded(a: Any, b: Any, same_sign: Any) -> Any:
    """Return ``a`` where it is not finite, else ``b`` where it is not finite, else ``same_sign()``
    where a and b have the same strict sign, else 0."""
    positive = (a > 0) & (b > 0)
    negative = (a < 0) & (b < 0)

    def limited():
        return pm.where(positive | negative, lambda: same_sign(positive), lambda: 0.)

    return pm.where(~_finite(a), lambda: a,
                    lambda: pm.where(~_finite(b), lambda: b, limited))


def _magnitudes(a: Any, b: Any, positive: Any) -> tuple[Any, Any]:
    # |a| and |b| where a and b share a sign, the sign given by ``positive``.
    return (pm.where(positive, lambda: a, lambda: -a),
            pm.where(positive, lambda: b, lambda: -b))


def _signed(positive: Any, magnitude: Any) -> Any:
    return pm.where(positive, lambda: magnitude, lambda: -magnitude)


def minmod(a: Any, b: Any) -> Any:
    """0 where ab <= 0; elsewhere a where |a| < |b|, and b otherwise (master's Minmod).

    The sign tests replace the product ab, which can underflow.
    """
    return _guarded(a, b, lambda positive: pm.where(abs(a) < abs(b), lambda: a, lambda: b))


def vanleer(a: Any, b: Any) -> Any:
    """0 where ab <= 0; elsewhere 2ab / (a + b) (master's VanLeer).

    Where ab is below the smallest normal value or above half the largest one, or a + b
    overflows, the same value is formed as m (2 / (1 + m / M)), m and M the smaller and the
    larger of |a| and |b|, with the common sign. A generated face refuses a trace whose
    intermediate values are not all finite, so there the product ab beyond the largest double
    (|a|, |b| above about 1e154) is refused rather than formed into the bounded value.
    """
    def same_sign(positive):
        product = a*b
        total = a + b
        ordinary = ((product >= _SMALLEST_NORMAL) & (product <= .5*_LARGEST)
                    & (total <= _LARGEST) & (total >= -_LARGEST))

        def bounded():
            aa, bb = _magnitudes(a, b, positive)
            smaller = pm.where(aa < bb, lambda: aa, lambda: bb)
            larger = pm.where(aa < bb, lambda: bb, lambda: aa)
            return _signed(positive, smaller*(2./(1. + smaller/larger)))

        return pm.where(ordinary, lambda: 2.*product/total, bounded)

    return _guarded(a, b, same_sign)


def mc(a: Any, b: Any) -> Any:
    """0 where ab <= 0; elsewhere min((|a| + |b|) / 2, 2|a|, 2|b|) with the common sign
    (monotonized central, master's MC).

    The centred candidate is |a|/2 + |b|/2; below twice the smallest normal value it is
    m + (M - m)/2, m and M the smaller and the larger magnitude, so that two equal subnormal
    differences are not lost. The bound 2m is compared as centred/2 <= m, which cannot overflow.
    """
    def same_sign(positive):
        aa, bb = _magnitudes(a, b, positive)
        smaller = pm.where(aa < bb, lambda: aa, lambda: bb)
        larger = pm.where(aa < bb, lambda: bb, lambda: aa)
        centred = pm.where(smaller >= 2.*_SMALLEST_NORMAL, lambda: .5*aa + .5*bb,
                           lambda: smaller + .5*(larger - smaller))
        magnitude = pm.where(.5*centred <= smaller, lambda: centred, lambda: 2.*smaller)
        return _signed(positive, magnitude)

    return _guarded(a, b, same_sign)


def superbee(a: Any, b: Any) -> Any:
    """0 where ab <= 0; elsewhere max(min(2|a|, |b|), min(|a|, 2|b|)) with the common sign
    (master's Superbee).

    Each doubled magnitude is formed only where it is bounded by the other one, so finite
    differences give a finite slope.
    """
    def same_sign(positive):
        aa, bb = _magnitudes(a, b, positive)
        twice_a = pm.where(aa <= .5*bb, lambda: 2.*aa, lambda: bb)
        twice_b = pm.where(bb <= .5*aa, lambda: 2.*bb, lambda: aa)
        magnitude = pm.where(twice_a > twice_b, lambda: twice_a, lambda: twice_b)
        return _signed(positive, magnitude)

    return _guarded(a, b, same_sign)


__all__ = ["mc", "minmod", "superbee", "vanleer"]
