"""MUSCL and its slope limiters, written with the surface (spec 0.3, 2.6.9; A4, SU5, CF4). N3 lot 2
(ROM-87).

``muscl(limiter=)`` is a scheme over ``cell``, ``term`` and ``axis`` that returns ``Reconstruct``:
the trace u_c + limiter(u_c - u_(c-1), u_(c+1) - u_c) / 2 on the face on the positive side of the
axis, mirrored on the negative side (2.6.4), which a face scheme takes as its ``reconstruction=``:
``rusanov(reconstruction=muscl(limiter=minmod))``. A limiter is a plain function ``limiter(a, b)``
of the backward and the forward differences, passed uncalled (SU5); it runs once, when the scheme
runs, and returns the expression of the limited slope.

The bodies are in the arithmetic of the b6 bodies that they replace
(``examples/api_v02/b6_reconstructions.py``, the parity oracles, D5): ``0.5 * x``, never ``x / 2``,
the same nested ``where`` and the same constants, so that the generated C++ of the trace is the one
of b6 operation by operation (P1). ``where`` takes values; the elaboration gives b6 its branches
(P3, P4). The body of ``muscl`` is the one of the gallery (``examples/surface/gallery/lib_schemes.py``,
D8), in that arithmetic.

With the limiters below, the generated trace of ``muscl`` is master's native MUSCL trace bit for bit
wherever the latter is finite, except for the sign of a zero trace of a cell value -0 on the
negative face (the native policy multiplies the slope by the orientation of the face; the
mirrored body does not), and except where vanleer's product of the two differences passes the
largest double, which the generated face refuses (E2 of the lot note, as for the b6 bodies).

Each limiter writes the formula of its catalogue row and keeps the guards of the b6 policy of
master of the same name (``include/pops/numerics/fv/reconstruction.hpp``), so that the two agree
on every input, not only where the row's formula is finite (P2):

* a non-finite difference is returned as the slope, so that the face guard of the flux refuses it
  instead of a limited value hiding it (the first finite check returns ``a``);
* where the row's formula would overflow or underflow, the same value is formed from the smaller
  magnitude, as the native policy forms it.

On finite differences of ordinary size each limiter is the formula of its row, in the native order
of operations. The constants are the binary64 bounds of the native policies, written as literals:
1.7976931348623157e+308 is ``std::numeric_limits<double>::max()``, 8.988465674311579e+307 its half,
2.2250738585072014e-308 ``::min()`` and 4.450147717014403e-308 twice that.
"""
import pops.surface as pops
from pops.surface.discrete import Reconstruct
from pops.surface.math import all_of, any_of, none_of, where


# The reconstruction

@pops.scheme
def muscl(cell, term, axis, limiter):
    """u_(c+1/2) = u_c + limiter(u_c - u_(c-1), u_(c+1) - u_c) / 2, the trace on the face on the
    positive side of `axis`; the trace on the negative side mirrors it (2.6.4). A record state is
    limited field by field, since record arithmetic acts field by field (SD71). As master's MUSCL
    face value."""
    q = term.quantity
    centre = cell.value(q)
    backward = centre - cell.shift(axis, -1).value(q)
    forward = cell.shift(axis, 1).value(q) - centre
    return Reconstruct(centre + 0.5 * limiter(backward, forward))


# The limiters

def _finite(value):
    # NaN fails both comparisons, and so does an infinity of either sign.
    return all_of([value <= 1.7976931348623157e+308, value >= -1.7976931348623157e+308])


def _guarded(a, b, limited):
    """a where it is not finite, else b where it is not finite, else ``limited``."""
    return where(none_of([_finite(a)]), a, where(none_of([_finite(b)]), b, limited))


def _same_sign(a, b):
    """The predicates a > 0 and b > 0, and a < 0 and b < 0."""
    return all_of([a > 0, b > 0]), all_of([a < 0, b < 0])


def _magnitudes(a, b, positive):
    # |a| and |b| where a and b share a sign, the sign given by ``positive``.
    return where(positive, a, -a), where(positive, b, -b)


def _signed(positive, magnitude):
    return where(positive, magnitude, -magnitude)


def minmod(a, b):
    """0 where ab <= 0; elsewhere a where |a| < |b|, and b otherwise (master's Minmod). The sign
    tests replace the product ab, which can underflow."""
    positive, negative = _same_sign(a, b)
    return _guarded(a, b, where(any_of([positive, negative]), where(abs(a) < abs(b), a, b), 0.0))


def vanleer(a, b):
    """0 where ab <= 0; elsewhere 2ab / (a + b) (master's VanLeer).

    Where ab is below the smallest normal value or above half the largest one, or a + b
    overflows, the same value is formed as m (2 / (1 + m / M)), m and M the smaller and the
    larger of |a| and |b|, with the common sign. A generated face refuses a trace whose
    intermediate values are not all finite, so there the product ab beyond the largest double
    (|a|, |b| above about 1e154) is refused rather than formed into the bounded value."""
    positive, negative = _same_sign(a, b)
    product = a * b
    total = a + b
    ordinary = all_of([product >= 2.2250738585072014e-308, product <= 8.988465674311579e+307,
                       total <= 1.7976931348623157e+308, total >= -1.7976931348623157e+308])
    aa, bb = _magnitudes(a, b, positive)
    smaller = where(aa < bb, aa, bb)
    larger = where(aa < bb, bb, aa)
    bounded = _signed(positive, smaller * (2.0 / (1.0 + smaller / larger)))
    same_sign = where(ordinary, 2.0 * product / total, bounded)
    return _guarded(a, b, where(any_of([positive, negative]), same_sign, 0.0))


def mc(a, b):
    """0 where ab <= 0; elsewhere min((|a| + |b|) / 2, 2|a|, 2|b|) with the common sign
    (monotonized central, master's MC).

    The centred candidate is |a|/2 + |b|/2; below twice the smallest normal value it is
    m + (M - m)/2, m and M the smaller and the larger magnitude, so that two equal subnormal
    differences are not lost. The bound 2m is compared as centred/2 <= m, which cannot overflow."""
    positive, negative = _same_sign(a, b)
    aa, bb = _magnitudes(a, b, positive)
    smaller = where(aa < bb, aa, bb)
    larger = where(aa < bb, bb, aa)
    centred = where(smaller >= 4.450147717014403e-308, 0.5 * aa + 0.5 * bb,
                    smaller + 0.5 * (larger - smaller))
    magnitude = where(0.5 * centred <= smaller, centred, 2.0 * smaller)
    return _guarded(a, b, where(any_of([positive, negative]), _signed(positive, magnitude), 0.0))


def superbee(a, b):
    """0 where ab <= 0; elsewhere max(min(2|a|, |b|), min(|a|, 2|b|)) with the common sign
    (master's Superbee).

    Each doubled magnitude is formed only where it is bounded by the other one, so finite
    differences give a finite slope."""
    positive, negative = _same_sign(a, b)
    aa, bb = _magnitudes(a, b, positive)
    twice_a = where(aa <= 0.5 * bb, 2.0 * aa, bb)
    twice_b = where(bb <= 0.5 * aa, 2.0 * bb, aa)
    magnitude = where(twice_a > twice_b, twice_a, twice_b)
    return _guarded(a, b, where(any_of([positive, negative]), _signed(positive, magnitude), 0.0))
