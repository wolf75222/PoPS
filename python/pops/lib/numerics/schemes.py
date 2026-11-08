"""The scheme words of the surface for a flux and a diffusive flux (spec 0.3, 2.6.9; A4, D5).

``upwind()``, ``rusanov()``, ``hll()`` and ``two_point()`` are the words that
``pops.Numerics(schemes={F: s})`` takes, each called with its options (SU5). Spec 0.3 writes them
with ``@pops.scheme`` over the entities ``face`` and ``term`` (2.6.4); until that construct is
elaborated, each word is a ``SchemeValue`` whose ``realize`` gives the b6 realization of its key
and checks the form of the key as ``term.expect`` would. A key of another form is refused
(``RealizationMismatch``), and a form that this lot does not realize is ``NotYetElaborated``. CF4
compares each word with the same scheme written with ``@pops.scheme`` once that construct exists.

At b6 the three face fluxes are b6's native first-order routes, as §17.1 states for L1: the
waves of the elaborated flux are dF/du, which the elaborator derives (R9), and Rusanov's flux with
those waves is the upwind flux of a flux F(1) u. ``two_point()`` is b6's two-point diffusion,
-k (u_R - u_L) / dx for a coefficient k that does not vary with the state. The b6 builders are
imported where a realization is built, as ``fluxes`` does: the import graph keeps ``pops.lib``
above the surface and below ``pops.numerics``.
"""
from __future__ import annotations

from typing import Any

from pops.surface import (
    AmbiguousElaboration, NotYetElaborated, RealizationMismatch, SchemeValue, TermView,
)

_FACE = "a later lot of N15 with the face entities of 2.6.4 and @pops.scheme (A4)"
_RECONSTRUCTION = "a later lot of N15 with the reconstructions of N3"
_DERIVATIVES = "N11 (expression closure, C1)"


def _first_order(view: TermView, face: str) -> Any:
    """The b6 first-order finite volume of the key, with the native face ``face`` of b6."""
    from pops.numerics import reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume

    return FiniteVolume(flux=view.elaborated, variables=variables.Conservative(view.state),
                        reconstruction=reconstruction.FirstOrder(),
                        riemann=getattr(riemann, face)())


def _derivative_speed(word: str, view: TermView, site: Any) -> None:
    raise AmbiguousElaboration(
        "the speed of %s under %s would read the derivative of a field, %s, so it is not "
        "derived" % (word, view.term, view.value),
        write="upwind(speed=f), f a function of face that returns the normal speed", site=site)


class _FaceFlux(SchemeValue):
    kinds = ("Flux",)

    def _reconstruction(self) -> None:
        if self.options.get("reconstruction") is not None:
            raise NotYetElaborated("%s(reconstruction=r)" % self.word, _RECONSTRUCTION,
                                   site=self.site)

    def _waves(self, view: TermView) -> None:
        if view.slope is None:
            if view.reads_derivative(view.value):
                _derivative_speed("%s()" % self.word, view, self.site)
            raise NotYetElaborated("the wave speeds of %s = %s, which this lot does not derive"
                                   % (view.term, view.value), _DERIVATIVES, site=self.site)


class _Upwind(_FaceFlux):
    def realize(self, view: TermView) -> Any:
        if self.options["speed"] is not None:
            raise NotYetElaborated("upwind(speed=f), a stated speed", _FACE, site=self.site)
        self._reconstruction()
        if view.unit_flux is None:
            if view.reads_derivative(view.value):
                _derivative_speed("upwind()", view, self.site)
            raise RealizationMismatch(
                "upwind() covers a flux F(1) u, linear in its scalar quantity u, and the flux "
                "of %s is %s" % (view.quantity, view.value),
                write="rusanov() or hll(), which take the wave speeds of the flux",
                site=self.site)
        return _first_order(view, "Rusanov")

    def describe(self) -> str:
        return ("FiniteVolume(variables=Conservative(u), reconstruction=FirstOrder(), "
                "riemann=Rusanov()), the upwind flux s u_L or s u_R of a flux F(1) u whose "
                "waves are s = F(1)")


class _Rusanov(_FaceFlux):
    def realize(self, view: TermView) -> Any:
        self._reconstruction()
        self._waves(view)
        return _first_order(view, "Rusanov")

    def describe(self) -> str:
        return ("FiniteVolume(variables=Conservative(u), reconstruction=FirstOrder(), "
                "riemann=Rusanov()), with the waves dF/du of the flux")


class _HLL(_FaceFlux):
    def realize(self, view: TermView) -> Any:
        self._reconstruction()
        self._waves(view)
        return _first_order(view, "HLL")

    def describe(self) -> str:
        return ("FiniteVolume(variables=Conservative(u), reconstruction=FirstOrder(), "
                "riemann=HLL()), with the waves dF/du of the flux")


class _TwoPoint(SchemeValue):
    kinds = ("DiffusiveFlux",)

    def realize(self, view: TermView) -> Any:
        if view.reads_quantity(view.coefficient):
            raise NotYetElaborated(
                "two_point() under %s, whose coefficient %s varies with the state"
                % (view.term, view.coefficient),
                "a later lot of N15 with the face means of 2.6.4 (G05)", site=self.site)
        from pops.numerics import Diffusion

        return Diffusion(flux=view.elaborated)

    def describe(self) -> str:
        return "Diffusion(flux=F^d), the two-point flux -k (u_R - u_L) / dx of b6"


def upwind(*, speed: Any = None, reconstruction: Any = None) -> SchemeValue:
    """``upwind(speed=None, reconstruction=None)``: s u_L where s > 0, else s u_R (2.6.9).

    Without ``speed`` the flux must be F(1) u, u its scalar quantity, and s = F(1)·n is derived;
    another flux is refused (``RealizationMismatch``, naming ``rusanov()`` and ``hll()``).
    """
    return _Upwind("upwind", speed=speed, reconstruction=reconstruction)


def rusanov(*, reconstruction: Any = None) -> SchemeValue:
    """``rusanov(reconstruction=None)``: the mean of the two fluxes less s (u_R - u_L) / 2, s the
    larger spectral radius of the two sides (R9)."""
    return _Rusanov("rusanov", reconstruction=reconstruction)


def hll(*, reconstruction: Any = None) -> SchemeValue:
    """``hll(reconstruction=None)``: the HLL flux with the slowest and fastest speeds (R9)."""
    return _HLL("hll", reconstruction=reconstruction)


def two_point() -> SchemeValue:
    """``two_point()``: F(u_f, g_f)·n with the face mean u_f and the two-point normal derivative
    g_f of each gradient variable (2.6.9, SD51)."""
    return _TwoPoint("two_point")


__all__ = ["hll", "rusanov", "two_point", "upwind"]
