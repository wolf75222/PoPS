"""WENO5, written with the surface (spec 0.3, 2.6.9; A4, SU5, CF4). N3 lot 3 (ROM-87).

``weno5(epsilon=None)`` is a scheme over ``cell``, ``term`` and ``axis`` that returns
``Reconstruct``: the WENO5-Z trace of Borges et al. (2008) over the cells -2 to 2 along the axis, on
the face on the positive side, mirrored on the negative side (2.6.4), which a face scheme takes as
its ``reconstruction=``: ``rusanov(reconstruction=weno5())``. It is master's ``WENO5()`` (and
``WENO5Z()``, which runs the same ``weno5z``).

The body is the b6 body that it replaces (``examples/api_v02/b6_reconstructions.py``, the parity
oracle, D5) in the same arithmetic, operation by operation (P1): the same nested ``where``, the same
constants and the same shared subexpressions, so that the generated C++ of the trace is the one of
b6, which follows master's native policy (``include/pops/numerics/fv/reconstruction.hpp``,
``weno5z``). ``where`` takes values; the elaboration gives b6 its branches and lowers each shared
node once (P3, P4). The guards of the native policy are kept (P2): a uniform stencil returns its
value, a stencil with a sample beyond 1e37 in magnitude is rescaled by its largest magnitude, and
the weights are formed relative to the smallest denominator where tau5 / (epsilon + beta) passes
1e100, so that the trace stays finite.

The constants are the binary64 values of the native policy, written as literals: 1e-40 is
``kWenoEpsilon``, 1e37 the magnitude above which the stencil is rescaled, 1e100 the ratio above
which the weights are formed relative to the smallest denominator, 1.7976931348623157e+308
``std::numeric_limits<double>::max()`` and 1.0833333333333333 the binary64 value of 13/12.
"""
import math

import pops.surface as pops
from pops.surface.discrete import Reconstruct
from pops.surface.math import all_of, any_of, equal, where


# The reconstruction

@pops.scheme
def weno5(cell, term, axis, epsilon=None):
    """WENO5 reconstruction: the trace on the positive face of `axis` combines the three
    third-order traces of the sub-stencils with the weights d_k (1 + (tau5 / (epsilon +
    beta_k))^2), d = (1/10, 6/10, 3/10), beta_k the smoothness indicators of Jiang and Shu and
    tau5 = |beta_0 - beta_2|; the trace on the negative side mirrors it (2.6.4). `epsilon` is the
    regulariser, a finite positive Python number, by default master's kWenoEpsilon = 1e-40. A
    record state is reconstructed field by field (SD71). As master's WENO5."""
    eps = 1e-40 if epsilon is None else _regulariser(epsilon)
    q = term.quantity
    samples = tuple(cell.shift(axis, k).value(q) for k in (-2, -1, 0, 1, 2))
    return Reconstruct(_weno5z(samples, eps))


def _regulariser(epsilon):
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("weno5(epsilon=) must be a Python number, got %r" % (epsilon,))
    value = float(epsilon)
    if not (math.isfinite(value) and value > 0):
        raise ValueError("weno5(epsilon=) must be finite and positive; got %r" % (epsilon,))
    return value


# The WENO5-Z trace, in the order of operations of the native weno5z

def _smaller(x, y):
    return where(x < y, x, y)


def _weights(d0, d1, d2, tau5, least, q):
    """The bounded form of the WENO-Z weights: the alphas times a common factor, finite even
    where tau5 / (eps + beta) would overflow."""
    q0, q1, q2 = q
    # A zero denominator dominates with its ideal weight.
    i0 = where(equal(d0, 0), 0.1, 0.0)
    i1 = where(equal(d1, 0), 0.6, 0.0)
    i2 = where(equal(d2, 0), 0.3, 0.0)
    indicator = (i0 * q0 + i1 * q1 + i2 * q2) / (i0 + i1 + i2)
    r0, r1, r2 = tau5 / d0, tau5 / d1, tau5 / d2
    a0 = 0.1 * (1.0 + r0 * r0)
    a1 = 0.6 * (1.0 + r1 * r1)
    a2 = 0.3 * (1.0 + r2 * r2)
    ratio = (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)
    common = least / tau5
    m0 = 0.1 * (common * common + (least / d0) * (least / d0))
    m1 = 0.6 * (common * common + (least / d1) * (least / d1))
    m2 = 0.3 * (common * common + (least / d2) * (least / d2))
    relative = (m0 * q0 + m1 * q1 + m2 * q2) / (m0 + m1 + m2)
    return where(equal(tau5, 0), 0.1 * q0 + 0.6 * q1 + 0.3 * q2,
                 where(equal(least, 0), indicator,
                       where(tau5 <= least, ratio, relative)))


def _core(vm2, vm1, v0, vp1, vp2, eps, bounded):
    """WENO5-Z at the face between v0 and vp1, as the native ``weno5z_core``; ``bounded`` is
    Python data, true on the rescaled stencil."""
    # The three third-order traces of the face.
    q0 = (2.0 * vm2 - 7.0 * vm1 + 11.0 * v0) / 6.0
    q1 = (-vm1 + 5.0 * v0 + 2.0 * vp1) / 6.0
    q2 = (2.0 * v0 + 5.0 * vp1 - vp2) / 6.0
    # The smoothness indicators of Jiang and Shu.
    e0, f0 = vm2 - 2.0 * vm1 + v0, vm2 - 4.0 * vm1 + 3.0 * v0
    e1, f1 = vm1 - 2.0 * v0 + vp1, vm1 - vp1
    e2, f2 = v0 - 2.0 * vp1 + vp2, 3.0 * v0 - 4.0 * vp1 + vp2
    b0 = 1.0833333333333333 * e0 * e0 + 0.25 * f0 * f0
    b1 = 1.0833333333333333 * e1 * e1 + 0.25 * f1 * f1
    b2 = 1.0833333333333333 * e2 * e2 + 0.25 * f2 * f2
    tau5 = where(b0 - b2 < 0, b2 - b0, b0 - b2)
    d0, d1, d2 = eps + b0, eps + b1, eps + b2
    least = where(d0 < d1, _smaller(d0, d2), _smaller(d1, d2))
    q = (q0, q1, q2)
    if bounded:
        return _weights(d0, d1, d2, tau5, least, q)
    r0, r1, r2 = tau5 / (eps + b0), tau5 / (eps + b1), tau5 / (eps + b2)
    a0 = 0.1 * (1.0 + r0 * r0)
    a1 = 0.6 * (1.0 + r1 * r1)
    a2 = 0.3 * (1.0 + r2 * r2)
    inv = 1.0 / (a0 + a1 + a2)
    ordinary = (a0 * q0 + a1 * q1 + a2 * q2) * inv
    return where(tau5 > 1e100 * least, _weights(d0, d1, d2, tau5, least, q), ordinary)


def _weno5z(v, eps):
    vm2, vm1, v0, vp1, vp2 = v
    # The largest magnitude of the five samples; a NaN never replaces it.
    scale = 0.0
    for value in v:
        magnitude = where(value < 0, -value, value)
        scale = where(magnitude > scale, magnitude, scale)
    wm2, wm1, w0, wp1, wp2 = (value / scale for value in v)
    scaled = scale * _core(wm2, wm1, w0, wp1, wp2, (eps / scale) / scale, True)
    rescaled = where(scale <= 1.7976931348623157e+308, scaled, scale)
    large = any_of([any_of([value > 1e37, value < -1e37]) for value in v])
    uniform = all_of([equal(vm2, vm1), equal(vm1, v0), equal(v0, vp1), equal(vp1, vp2)])
    return where(uniform, v0, where(large, rescaled, _core(vm2, vm1, v0, vp1, vp2, eps, False)))
