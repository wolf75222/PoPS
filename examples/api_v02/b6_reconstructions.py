"""The b6 bodies of the per-axis reconstructions of ``pops.lib.numerics``, kept as the parity oracles
of the surface words (N3 lots 2 and 3, D5; MG10): the builder ``Reconstruct(rep, value)``,
``weno5(rep)``, ``muscl(rep, limiter)`` and its four slope limiters.

Until N3 lot 2 these functions were ``pops.lib.numerics.reconstructions`` and
``pops.lib.numerics.limiters``. The library now writes ``muscl(limiter=)`` (lot 2) and
``weno5(epsilon=)`` (lot 3) with ``@pops.scheme`` and the limiters over the expressions of the
surface, and these bodies move here unchanged: the parity tests compare each word with the body
below, C++ for C++ (P1), and the native parity of N3 (``test_lib_numerics_reconstructions*.py``)
keeps them against master's MUSCL and WENO5. The b6 ``Reconstruct`` is not ported (D4): a case
writes ``Reconstruct`` from ``pops.surface.discrete``. Text of the earlier modules:

Per-axis reconstructions written in Python (spec 0.3 sections 4.3 and 4.4, IR25; N3).

Since N3 lot 2 ``muscl`` and its limiters are written with the surface in
:mod:`pops.lib.numerics.limiters`, and their b6 bodies are the oracles of
``examples/api_v02/b6_reconstructions.py``. This module keeps the b6 ``Reconstruct`` and ``weno5``
until lot 3 ports ``weno5`` (MG9, ``[[earlier]]``).

``Reconstruct(rep, value)`` is the builder of section 4.3: ``value(c, a)`` gives the trace of
the cell ``c`` on its face on the positive side of the axis ``a``, and the trace on the face on
the negative side is the same body with its offsets along ``a`` mirrored. ``weno5(rep)`` is the
WENO5-Z face value of master's ``WENO5()``, written with it. The body runs once, now, and
its expression becomes the generated C++ of the face reconstruction; no native reconstruction or
limiter is named.

The realization is the source-compiled reconstruction ``pops.numerics.reconstruction.User``,
which a face flux of :mod:`pops.lib.numerics.fluxes` or a ``FiniteVolume`` takes as its
``reconstruction=``. It fixes three properties of this lot, each stated where it applies:

* componentwise: ``c.value(rep)`` is one component of ``rep``, and the body is applied to each
  component of a record state on its own;
* one body for every axis: ``a`` is the axis of the face, the same for each axis of the support,
  and the generated kernel applies the body along each axis, mirrored on the negative face. A
  body that asks which axis it is (``a == x``) is refused, since it would choose one branch for
  every axis;
* reads along ``a`` only: ``c.shift(a, k)`` reaches the cells of the face's line; a read across
  the face's line is refused.

The stencil, the halo and the derived ghost depth come from the shifts that the body reads
(D5a); the author writes none of them.

Slope limiters written in Python (spec 0.3 section 4.4, catalogue of ``pops.lib.numerics``; N3).

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

import math
import sys
from collections.abc import Callable
from typing import Any

from pops import math as pm


class _Axis:
    """The axis ``a`` of a reconstruction body: the axis of the face, the same for every axis."""

    __slots__ = ()

    def __eq__(self, other: object) -> bool:
        raise TypeError(
            "Reconstruct: the axis of a reconstruction body is the axis of the face, and the same "
            "body serves every axis; a body cannot compare it with an axis of the frame")

    def __hash__(self) -> int:
        raise TypeError("Reconstruct: the axis of a reconstruction body is not Python data")

    def __repr__(self) -> str:
        return "<axis of the face>"


_FACE_AXIS = _Axis()


class _Cell:
    """The cell ``c`` of a reconstruction body, ``offset`` cells from the cell of the trace along
    the axis of the face."""

    __slots__ = ("_rep", "_sample", "_offset")

    def __init__(self, rep: Any, sample: Callable[[int], Any], offset: int) -> None:
        self._rep = rep
        self._sample = sample
        self._offset = offset

    def value(self, rep: Any) -> Any:
        """The value of ``rep`` in this cell: one component, the body being componentwise."""
        if rep is not self._rep:
            raise TypeError("Reconstruct(rep, value): the body reads only the representation "
                            "that it reconstructs")
        return self._sample(self._offset)

    def shift(self, axis: Any, k: int) -> _Cell:
        """The cell ``k`` steps along ``axis``, which must be the axis ``a`` of the body."""
        if axis is not _FACE_AXIS:
            raise TypeError("Reconstruct: c.shift(a, k) reads along the axis a of the face only; "
                            "a read across the face's line is not a per-axis reconstruction")
        if type(k) is not int:
            raise TypeError("Reconstruct: c.shift(a, k) takes an exact integer k")
        return _Cell(self._rep, self._sample, self._offset + k)

    def __repr__(self) -> str:
        return "<cell %+d along the axis of the face>" % self._offset


def _trace(rep: Any, value: Callable[..., Any], sample: Callable[[int], Any]) -> Any:
    return value(_Cell(rep, sample, 0), _FACE_AXIS)


def Reconstruct(rep: Any, value: Callable[..., Any], *, formal_order: int | None = None) -> Any:
    """Reconstruction of ``rep``: ``value(c, a)`` is the trace of ``c`` on its face on the
    positive side of the axis ``a`` (section 4.3).

    ``c.value(rep)`` reads the cell, and ``c.shift(a, k)`` the cell ``k`` steps along ``a``;
    the trace on the negative face is the same body with its offsets mirrored. ``formal_order``
    is the claim of section 13 that the realization ``reconstruction.User`` carries (IR); left
    out, it is the number of cells that the body spans, the largest order that such a stencil can
    have, so that a halo or a look-ahead sized by the order is never short.
    """
    from pops.numerics import reconstruction

    if not callable(value):
        raise TypeError("Reconstruct(rep, value) requires a function value(c, a)")
    if formal_order is not None and (type(formal_order) is not int or formal_order < 1):
        raise ValueError("Reconstruct(formal_order=) must be a positive exact integer")

    def body(sample):
        return _trace(rep, value, sample)

    if formal_order is None:
        # The span of the stencil is known once the body has run: author it once to read the
        # offsets, then with the order that they bound.
        offsets = reconstruction.User(body, formal_order=1).options["sample_offsets"]
        return reconstruction.User(
            body, formal_order=int(max(offsets, default=0) - min(offsets, default=0) + 1))
    return reconstruction.User(body, formal_order=formal_order)


# The constants of the native WENO5-Z policy (include/pops/numerics/fv/reconstruction.hpp and
# include/pops/runtime/numerical_defaults.hpp): the default smoothness regulariser kWenoEpsilon,
# the sample magnitude above which the stencil is rescaled, and the ratio tau5 / (eps + beta)
# above which the weights are formed relative to the smallest denominator.
_WENO_EPSILON = 1e-40
_WENO_SCALE_LIMIT = 1e37
_WENO_RATIO_LIMIT = 1e100
# The largest binary64 value, std::numeric_limits<double>::max() of the native policy.
_LARGEST = sys.float_info.max


def _smaller(x: Any, y: Any) -> Any:
    return pm.where(x < y, lambda: x, lambda: y)


def _weno5z_weights(d0: Any, d1: Any, d2: Any, tau5: Any, smallest: Any,
                    q: tuple[Any, Any, Any]) -> Any:
    """The bounded form of the WENO-Z weights: the alphas times a common factor, finite even
    where tau5 / (eps + beta) would overflow."""
    q0, q1, q2 = q

    def indicator_weights():
        # A zero denominator dominates with its ideal weight.
        a0 = pm.where(d0 == 0, lambda: .1, lambda: 0.)
        a1 = pm.where(d1 == 0, lambda: .6, lambda: 0.)
        a2 = pm.where(d2 == 0, lambda: .3, lambda: 0.)
        return (a0*q0 + a1*q1 + a2*q2)/(a0 + a1 + a2)

    def ratio_weights():
        r0, r1, r2 = tau5/d0, tau5/d1, tau5/d2
        a0 = .1*(1. + r0*r0)
        a1 = .6*(1. + r1*r1)
        a2 = .3*(1. + r2*r2)
        return (a0*q0 + a1*q1 + a2*q2)/(a0 + a1 + a2)

    def relative_weights():
        common = smallest/tau5
        a0 = .1*(common*common + (smallest/d0)*(smallest/d0))
        a1 = .6*(common*common + (smallest/d1)*(smallest/d1))
        a2 = .3*(common*common + (smallest/d2)*(smallest/d2))
        return (a0*q0 + a1*q1 + a2*q2)/(a0 + a1 + a2)

    return pm.where(tau5 == 0, lambda: .1*q0 + .6*q1 + .3*q2,
                    lambda: pm.where(smallest == 0, indicator_weights,
                                     lambda: pm.where(tau5 <= smallest, ratio_weights,
                                                      relative_weights)))


def _weno5z_core(vm2: Any, vm1: Any, v0: Any, vp1: Any, vp2: Any, eps: Any,
                 bounded: bool) -> Any:
    """WENO5-Z (Borges et al. 2008) at the face between v0 and vp1, in the order of operations of
    the native ``weno5z_core``; ``bounded`` is Python data, true on the rescaled stencil."""
    # The three third-order traces of the face.
    q0 = (2.*vm2 - 7.*vm1 + 11.*v0)/6.
    q1 = (-vm1 + 5.*v0 + 2.*vp1)/6.
    q2 = (2.*v0 + 5.*vp1 - vp2)/6.
    # The smoothness indicators of Jiang and Shu.
    c = 13./12.
    e0, f0 = vm2 - 2.*vm1 + v0, vm2 - 4.*vm1 + 3.*v0
    e1, f1 = vm1 - 2.*v0 + vp1, vm1 - vp1
    e2, f2 = v0 - 2.*vp1 + vp2, 3.*v0 - 4.*vp1 + vp2
    b0 = c*e0*e0 + .25*f0*f0
    b1 = c*e1*e1 + .25*f1*f1
    b2 = c*e2*e2 + .25*f2*f2
    tau5 = pm.where(b0 - b2 < 0, lambda: b2 - b0, lambda: b0 - b2)
    d0, d1, d2 = eps + b0, eps + b1, eps + b2
    smallest = pm.where(d0 < d1, lambda: _smaller(d0, d2), lambda: _smaller(d1, d2))
    q = (q0, q1, q2)
    if bounded:
        return _weno5z_weights(d0, d1, d2, tau5, smallest, q)

    def ordinary():
        r0, r1, r2 = tau5/(eps + b0), tau5/(eps + b1), tau5/(eps + b2)
        a0 = .1*(1. + r0*r0)
        a1 = .6*(1. + r1*r1)
        a2 = .3*(1. + r2*r2)
        inv = 1./(a0 + a1 + a2)
        return (a0*q0 + a1*q1 + a2*q2)*inv

    return pm.where(tau5 > _WENO_RATIO_LIMIT*smallest,
                    lambda: _weno5z_weights(d0, d1, d2, tau5, smallest, q), ordinary)


def _weno5z(v: tuple[Any, Any, Any, Any, Any], eps: float) -> Any:
    vm2, vm1, v0, vp1, vp2 = v

    def rescaled():
        # The largest magnitude of the five samples; a NaN never replaces it.
        scale: Any = 0.
        for value in v:
            magnitude = pm.where(value < 0, lambda value=value: -value, lambda value=value: value)
            scale = pm.where(magnitude > scale, lambda m=magnitude: m, lambda s=scale: s)

        def scaled():
            wm2, wm1, w0, wp1, wp2 = (value/scale for value in v)
            return scale*_weno5z_core(wm2, wm1, w0, wp1, wp2, (eps/scale)/scale, True)

        return pm.where(scale <= _LARGEST, scaled, lambda: scale)

    large = None
    for value in v:
        outside = (value > _WENO_SCALE_LIMIT) | (value < -_WENO_SCALE_LIMIT)
        large = outside if large is None else large | outside
    uniform = (vm2 == vm1) & (vm1 == v0) & (v0 == vp1) & (vp1 == vp2)
    return pm.where(uniform, lambda: v0, lambda: pm.where(
        large, rescaled, lambda: _weno5z_core(vm2, vm1, v0, vp1, vp2, eps, False)))


def weno5(rep: Any, epsilon: float | None = None) -> Any:
    """WENO5 reconstruction of ``rep``: the WENO-Z weights of Borges et al. (2008) over the
    shifts -2 to 2 (IR table of section 13: halo 3, nonlinear), master's ``WENO5()`` and
    ``WENO5Z()``, which both run ``weno5z``.

    The trace on the positive face of each axis combines the three third-order traces of the
    sub-stencils with the weights d_k (1 + (tau5 / (epsilon + beta_k))^2), d = (1/10, 6/10,
    3/10), beta_k the smoothness indicators of Jiang and Shu and tau5 = |beta_0 - beta_2|;
    the negative face mirrors it. ``epsilon`` is the regulariser, by default the native
    ``kWenoEpsilon`` = 1e-40. As the native policy, a uniform stencil returns its value, a
    stencil with a sample beyond 1e37 in magnitude is rescaled by its largest magnitude, and the
    weights are formed relative to the smallest denominator where tau5 / (epsilon + beta) passes
    1e100, so that the trace stays finite. Claimed order 5 (section 13); derived ghost depth 3.
    """
    eps = _WENO_EPSILON if epsilon is None else epsilon
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise TypeError("weno5(epsilon=) must be a Python number")
    eps = float(eps)
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError("weno5(epsilon=) must be finite and positive; got %r" % (epsilon,))

    def value(c, a):
        samples = tuple(c.shift(a, k).value(rep) for k in (-2, -1, 0, 1, 2))
        return _weno5z(samples, eps)

    return Reconstruct(rep, value, formal_order=5)


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


def muscl(rep: Any, limiter: Callable[[Any, Any], Any]) -> Any:
    """MUSCL reconstruction of ``rep`` with ``limiter`` (section 4.4): the trace
    u_c + limiter(u_c - u_(c-1), u_(c+1) - u_c) / 2 on the positive face of each axis, mirrored
    on the negative one; master's ``MUSCL(VanLeer())`` is ``muscl(Uh, vanleer)``.

    ``limiter`` is passed uncalled, for example :func:`vanleer`. The
    trace is second order where the limiter keeps the slope (claim of section 13), and the body
    reads one cell on each side, so the derived ghost depth is 2.

    With the limiters above, the generated trace is the native
    MUSCL trace bit for bit wherever the latter is finite, except for the sign of a zero trace of
    a cell value -0 on the negative face (the native policy multiplies the slope by the face's
    orientation; the mirrored body does not), and except where vanleer's product of the two
    differences passes the largest double, which the generated face refuses.
    """
    if not callable(limiter):
        raise TypeError("muscl(rep, limiter) requires a limiter function limiter(a, b), passed "
                        "uncalled")

    def value(c, a):
        u = c.value(rep)
        return u + .5*limiter(u - c.shift(a, -1).value(rep), c.shift(a, 1).value(rep) - u)

    return Reconstruct(rep, value, formal_order=2)


__all__ = ["Reconstruct", "mc", "minmod", "muscl", "superbee", "vanleer", "weno5"]
