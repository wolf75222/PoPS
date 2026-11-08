"""Per-axis reconstructions written in Python (spec 0.3 sections 4.3 and 4.4, IR25; N3).

``Reconstruct(rep, value)`` is the builder of section 4.3: ``value(c, a)`` gives the trace of
the cell ``c`` on its face on the positive side of the axis ``a``, and the trace on the face on
the negative side is the same body with its offsets along ``a`` mirrored. ``muscl(rep, limiter)``
is the MUSCL face value of the catalogue, written with it:

    u_c + limiter(u_c - u_(c-1), u_(c+1) - u_c) / 2

so that master's ``MUSCL(VanLeer())`` is ``muscl(Uh, vanleer)``, and ``weno5(rep)`` is the
WENO5-Z face value of master's ``WENO5()``. The body runs once, now, and
its expression becomes the generated C++ of the face reconstruction; no native reconstruction or
limiter is named.

The realization is the source-compiled reconstruction ``pops.numerics.reconstruction.User``,
which a face flux of :mod:`pops.lib.numerics.fluxes` or a ``FiniteVolume`` takes as its
``reconstruction=``. It fixes three properties of this lot, each stated where it applies:

* componentwise: ``c.value(rep)`` is one component of ``rep``, and the body is applied to each
  component of a record state on its own, as section 4.4 applies ``muscl`` to each field;
* one body for every axis: ``a`` is the axis of the face, the same for each axis of the support,
  and the generated kernel applies the body along each axis, mirrored on the negative face. A
  body that asks which axis it is (``a == x``) is refused, since it would choose one branch for
  every axis;
* reads along ``a`` only: ``c.shift(a, k)`` reaches the cells of the face's line; a read across
  the face's line is refused.

The stencil, the halo and the derived ghost depth come from the shifts that the body reads
(D5a); the author writes none of them.
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


def muscl(rep: Any, limiter: Callable[[Any, Any], Any]) -> Any:
    """MUSCL reconstruction of ``rep`` with ``limiter`` (section 4.4): the trace
    u_c + limiter(u_c - u_(c-1), u_(c+1) - u_c) / 2 on the positive face of each axis, mirrored
    on the negative one; master's ``MUSCL(VanLeer())`` is ``muscl(Uh, vanleer)``.

    ``limiter`` is passed uncalled, for example :func:`pops.lib.numerics.limiters.vanleer`. The
    trace is second order where the limiter keeps the slope (claim of section 13), and the body
    reads one cell on each side, so the derived ghost depth is 2.

    With the limiters of :mod:`pops.lib.numerics.limiters`, the generated trace is the native
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


__all__ = ["Reconstruct", "muscl", "weno5"]
