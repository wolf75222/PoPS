"""Rusanov, upwind and HLL face fluxes written in Python (spec 0.3 section 4.4, P11, A4; N3a).

:func:`rusanov`, :func:`upwind` and :func:`hll` return the finite-volume realization of one
physical flux ``F`` of the state ``Uh``: a :class:`pops.numerics.spatial.FiniteVolume` whose face is
a :func:`riemann.User` body. The body runs once, now, and its expressions become the generated C++
face policy; no native numerical flux is named, and the native ``riemann.Rusanov()`` /
``riemann.HLL()`` routes are not called.

A face body reads, for one face and its canonical positive normal ``n``:

* ``uL``, ``uR``: the two traces of ``Uh`` (one expression per component);
* ``fL``, ``fR``: the normal physical fluxes F(uL).n and F(uR).n, generated from the components of
  ``F``;
* ``s``: the larger spectral radius of ``F`` at the two traces along ``n``, generated from the wave
  speeds that ``F`` declares with ``waves=`` (R9).

A speed function given as an option is called with the same five arguments,
``f(uL, uR, fL, fR, s)``, and returns expressions of them. Section 4.4 writes ``smax(uL, uR, n)``;
the body of ``riemann.User`` receives neither ``n`` nor the signed wave speeds along it, so these
functions take the five face inputs above until that face input exists (CF15, N3a report).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pops import math as pm


def _realization(F: Any, Uh: Any, face: Any, reconstruction: Any) -> Any:
    from pops.numerics import reconstruction as reconstructions, variables
    from pops.numerics.spatial import FiniteVolume

    if reconstruction is None:
        reconstruction = reconstructions.FirstOrder()
    return FiniteVolume(flux=F, variables=variables.Conservative(Uh),
                        reconstruction=reconstruction, riemann=face)


def _face(body: Callable[..., tuple], Uh: Any, stability: Callable[..., Any]) -> Any:
    from pops.numerics import riemann

    return riemann.User(body=body, state=Uh, stability=stability)


def spectral_radius(uL: Any, uR: Any, fL: Any, fR: Any, s: Any) -> Any:
    """The larger spectral radius of the flux at the two traces along the face normal (R9)."""
    del uL, uR, fL, fR
    return s


def rusanov(F: Any, Uh: Any, smax: Callable[..., Any] = spectral_radius, *,
            reconstruction: Any = None) -> Any:
    """Rusanov (local Lax-Friedrichs) flux of ``F`` on ``Uh``.

    F* = (F(uL).n + F(uR).n) / 2 - smax (uR - uL) / 2, with ``smax`` recorded as the speed of the
    flux. The default ``smax`` is :func:`spectral_radius`, which gives master's Rusanov.
    ``reconstruction=None`` is first order.
    """
    if not callable(smax):
        raise TypeError("rusanov(smax=) requires a speed function f(uL, uR, fL, fR, s)")

    def body(uL, uR, fL, fR, s):
        speed = smax(uL, uR, fL, fR, s)
        return tuple(.5*(fl + fr) - .5*speed*(ur - ul)
                     for ul, ur, fl, fr in zip(uL, uR, fL, fR, strict=True))

    return _realization(F, Uh, _face(body, Uh, smax), reconstruction)


def upwind(F: Any, Uh: Any, speed: Callable[..., Any] | None = None, *,
           reconstruction: Any = None) -> Any:
    """Upwind flux of ``F`` on ``Uh``: F(uL).n where the normal speed is positive, else F(uR).n.

    ``speed=None`` is :func:`rusanov` with the spectral radius of R9, which is the upwind flux of a
    flux linear in u. A given ``speed`` returns the signed normal speed; since the body receives no
    face normal, its sign must hold on every canonical normal of the case (for M01 the velocity
    (.7, .3) gives a positive speed on both axes). The recorded speed is the spectral radius ``s``.
    """
    if speed is None:
        return rusanov(F, Uh, reconstruction=reconstruction)
    if not callable(speed):
        raise TypeError("upwind(speed=) requires a speed function f(uL, uR, fL, fR, s)")

    def body(uL, uR, fL, fR, s):
        sign = speed(uL, uR, fL, fR, s)
        return tuple(pm.where(sign > 0, lambda fl=fl: fl, lambda fr=fr: fr)
                     for fl, fr in zip(fL, fR, strict=True))

    return _realization(F, Uh, _face(body, Uh, spectral_radius), reconstruction)


def hll(F: Any, Uh: Any, speeds: Callable[..., tuple] | None = None, *,
        reconstruction: Any = None) -> Any:
    """Harten-Lax-van Leer flux of ``F`` on ``Uh``.

    F* = F(uL).n where sL >= 0, F(uR).n where sR <= 0, and otherwise
    (sR F(uL).n - sL F(uR).n + sL sR (uR - uL)) / (sR - sL), where ``speeds`` returns the pair
    (sL, sR): the smallest slowest and the largest fastest signed wave speed of the two traces
    along the face normal. The recorded speed of the flux is the spectral radius ``s``.

    The face of ``riemann.User`` receives no signed wave speed, so ``speeds`` has no derived
    default yet: a case states the pair from its own wave-speed bounds. The derived default, the
    slowest and fastest speeds of R9 along the face normal, needs a face input that the frozen
    engine does not provide (CF15; see the N3a report).
    """
    if speeds is None:
        raise NotImplementedError(
            "hll(F, Uh) needs the signed wave speeds of F along the face normal; the "
            "riemann.User face receives only the spectral radius. State them with "
            "hll(F, Uh, speeds=f), f(uL, uR, fL, fR, s) -> (sL, sR).")
    if not callable(speeds):
        raise TypeError("hll(speeds=) requires a function f(uL, uR, fL, fR, s) -> (sL, sR)")

    def body(uL, uR, fL, fR, s):
        pair = speeds(uL, uR, fL, fR, s)
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError("hll(speeds=) must return the pair (sL, sR)")
        lower, upper = pair

        def component(ul, ur, fl, fr):
            # The two-wave state exists only where sL < 0 < sR, so sR - sL > 0 there.
            def two_wave():
                return (upper*fl - lower*fr + lower*upper*(ur - ul))*(1./(upper - lower))
            return pm.where(lower >= 0, lambda: fl,
                            lambda: pm.where(upper <= 0, lambda: fr, two_wave))

        return tuple(component(*row) for row in zip(uL, uR, fL, fR, strict=True))

    return _realization(F, Uh, _face(body, Uh, spectral_radius), reconstruction)


__all__ = ["rusanov", "upwind", "hll", "spectral_radius"]
