"""Rusanov, upwind and HLL face fluxes written in Python (spec 0.3 section 4.4, 2.6.9, P11, A4).

:func:`rusanov`, :func:`upwind` and :func:`hll` return the finite-volume realization of one
physical flux ``F`` of the state ``Uh``: a :class:`pops.numerics.spatial.FiniteVolume` whose face is
a :func:`riemann.User` body. The body runs once, now, and its expressions become the generated C++
face policy; no native numerical flux is named, and the native ``riemann.Rusanov()`` /
``riemann.HLL()`` routes are not called. The native face loop that runs the body is the same for
every flux: it holds no formula, which the body-swap parity of CF4 checks with the L1 upwind body
and the L5 Scharfetter-Gummel body (tests/python/integration/runtime/test_lib_face_body_swap_native.py).

A face body reads, for one face and its canonical positive normal ``n``:

* ``uL``, ``uR``: the two traces of ``Uh`` (one expression per component);
* ``fL``, ``fR``: the normal physical fluxes F(uL).n and F(uR).n, generated from the components of
  ``F``;
* ``s``: the larger spectral radius of ``F`` at the two traces along ``n``, generated from the wave
  speeds that ``F`` declares with ``waves=`` (R9);
* ``n``, the normal itself, and ``speeds``, the pair (sL, sR) of the slowest and the fastest wave
  speeds that the model declares along ``n`` (``Model.wave_speeds`` or
  ``wave_speeds_from_jacobian``, R9), united over the two traces. The face evaluates these two
  only where an expression reads them (CF15, ROM-99).

Each flux records the speed that its row of 2.6.9 states (SD44; adjudication section 20):
:func:`rusanov` the speed it dissipates with, :func:`upwind` with ``speed=f`` the magnitude of
f(face), :func:`hll` max(|sL|, |sR|) of the pair it uses.

The speed function given as an option takes the arguments of section 4.4: ``smax(uL, uR, n)`` for
:func:`rusanov` (the third parameter of section 4.4's ``rusanov(F, Uh, smax)``) and
``speed(face)`` for :func:`upwind`. The ``spectral_radius(F, u, n)`` of section 4.3 belongs to this
module but is not written yet; the face reads its larger value over the two traces as the input
``s``.
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


class _Face:
    """The face that :func:`upwind` gives its speed function (section 4.4).

    ``face.normal`` is the canonical normal n, and ``face.left(Uh)`` and ``face.right(Uh)`` are
    the two traces of the realized state. Section 4.3 evaluates any expression on either side;
    this face holds the traces of ``Uh`` only, so it refuses any other expression.
    """

    __slots__ = ("normal", "_state", "_traces")

    def __init__(self, state: Any, left: Any, right: Any, normal: Any) -> None:
        self.normal = normal
        self._state = state
        self._traces = (left, right)

    def _trace(self, side: int, value: Any) -> Any:
        if value is not self._state:
            raise TypeError("upwind(speed=): the face holds the traces of the realized state only")
        return self._traces[side]

    def left(self, value: Any) -> Any:
        return self._trace(0, value)

    def right(self, value: Any) -> Any:
        return self._trace(1, value)


def rusanov(F: Any, Uh: Any, smax: Callable[..., Any] | None = None, *,
            reconstruction: Any = None) -> Any:
    """Rusanov (local Lax-Friedrichs) flux of ``F`` on ``Uh``.

    F* = (F(uL).n + F(uR).n) / 2 - smax (uR - uL) / 2, with ``smax`` recorded as the speed of the
    flux. ``smax(uL, uR, n)`` gives the face speed from the two traces and the normal (section
    4.4); ``smax=None`` takes the face input ``s``, the larger spectral radius of the two traces
    (R9), which gives master's Rusanov. ``reconstruction=None`` is first order.
    """
    if smax is not None and not callable(smax):
        raise TypeError("rusanov(smax=) requires a speed function smax(uL, uR, n)")

    def speed(uL, uR, fL, fR, s, *, n):
        del fL, fR
        return s if smax is None else smax(uL, uR, n)

    def body(uL, uR, fL, fR, s, *, n):
        face_speed = speed(uL, uR, fL, fR, s, n=n)
        return tuple(.5*(fl + fr) - .5*face_speed*(ur - ul)
                     for ul, ur, fl, fr in zip(uL, uR, fL, fR, strict=True))

    return _realization(F, Uh, _face(body, Uh, speed), reconstruction)


def upwind(F: Any, Uh: Any, speed: Callable[..., Any] | None = None, *,
           reconstruction: Any = None) -> Any:
    """Upwind flux of ``F`` on ``Uh`` (row ``upwind`` of 2.6.9, section 4.4).

    With ``speed=f``, s = f(face) is the signed normal speed at the face, for example the velocity
    of the case dotted with ``face.normal``; ``face.left(Uh)`` and ``face.right(Uh)`` are the two
    traces. F* = s uL where s > 0, else s uR, component by component, and the flux records |s| as
    its speed (SD44; adjudication section 20). Nothing is checked against ``F``: the stated speed
    is the case's.

    ``speed=None`` is section 4.4's form: :func:`rusanov` with the larger spectral radius of R9,
    which is the upwind flux of a scalar flux linear in u. The row of 2.6.9 instead takes
    s = F(1).n and refuses a flux that is not F(1) u (``RealizationMismatch``, naming ``rusanov()``
    and ``hll()``); that needs the scheme to read the form of its key (``term.expect``) and a face
    input F(1).n, which the face of ``riemann.User`` does not have (CF15, ROM-87 handoff).
    """
    if speed is None:
        return rusanov(F, Uh, reconstruction=reconstruction)
    if not callable(speed):
        raise TypeError("upwind(speed=) requires a speed function speed(face)")

    def stated(uL, uR, n):
        return speed(_Face(Uh, uL, uR, n))

    def body(uL, uR, fL, fR, s, *, n):
        del fL, fR, s
        sign = stated(uL, uR, n)
        return tuple(pm.where(sign > 0, lambda ul=ul: sign*ul, lambda ur=ur: sign*ur)
                     for ul, ur in zip(uL, uR, strict=True))

    def recorded(uL, uR, fL, fR, s, *, n):
        del fL, fR, s
        return abs(stated(uL, uR, n))

    return _realization(F, Uh, _face(body, Uh, recorded), reconstruction)


def hll(F: Any, Uh: Any, *, reconstruction: Any = None) -> Any:
    """Harten-Lax-van Leer flux of ``F`` on ``Uh`` (row ``hll`` of 2.6.9).

    F* = F(uL).n where sL >= 0, F(uR).n where sR <= 0, and otherwise
    (sR F(uL).n - sL F(uR).n + sL sR (uR - uL)) / (sR - sL). (sL, sR) is the face input
    ``speeds``: the slowest and the fastest wave speeds that the model declares along the face
    normal (``Model.wave_speeds`` or ``wave_speeds_from_jacobian``, R9), united over the two
    traces, as the native ``riemann.HLL()`` takes them. A case whose model declares none is
    refused when it is validated. The flux records max(|sL|, |sR|) as its speed, the larger
    spectral radius of the two traces (adjudication section 20). The row states no other way to
    give the pair, so the pilot's ``speeds=`` option is gone (I22, CF16).
    """
    def body(uL, uR, fL, fR, s, *, speeds):
        del s
        lower, upper = speeds

        def component(ul, ur, fl, fr):
            # The two-wave state exists only where sL < 0 < sR, so sR - sL > 0 there.
            def two_wave():
                return (upper*fl - lower*fr + lower*upper*(ur - ul))*(1./(upper - lower))
            return pm.where(lower >= 0, lambda: fl,
                            lambda: pm.where(upper <= 0, lambda: fr, two_wave))

        return tuple(component(*row) for row in zip(uL, uR, fL, fR, strict=True))

    def recorded(uL, uR, fL, fR, s, *, speeds):
        del uL, uR, fL, fR, s
        lower, upper = speeds
        return pm.maximum(abs(lower), abs(upper))

    return _realization(F, Uh, _face(body, Uh, recorded), reconstruction)


__all__ = ["rusanov", "upwind", "hll"]
