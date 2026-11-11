"""The scheme words of the surface for a flux and a diffusive flux (spec 0.3, 2.6.9; A4, D5).

``upwind()``, ``rusanov()``, ``hll()`` and ``two_point()`` are the words that
``pops.Numerics(schemes={F: s})`` takes, each called with its options (SU5). Spec 0.3 writes them
with ``@pops.scheme`` over the entities ``face`` and ``term`` (2.6.4); until the library switches
to one authoring API (decision D3), each word is a ``SchemeValue`` whose ``realize`` gives the b6
realization of its key and checks the form of the key as ``term.expect`` would (SD87). A key of
another form is refused (``RealizationMismatch``), and a form that this lot does not realize is
``NotYetElaborated``. CF4 compares each word with the same scheme written with ``@pops.scheme``.

Lot 2 of N15 adds ``upwind(speed=f)``, f a plain function of ``face`` that returns the signed
normal speed (SU5), realized by ``pops.lib.numerics.fluxes.upwind``, and ``reconstruction=r`` of
a face flux, r a scheme of one's own that returns ``Reconstruct`` (``reconstruction.User``).

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
    check_plain_function,
)

_PRODUCT = "a later lot of N15 with the product supports of G09 (SD73)"
_DERIVATIVES = "N11 (expression closure, C1)"


def _first_order(view: TermView, face: str, recon: Any = None) -> Any:
    """The b6 finite volume of the key, with the native face ``face`` of b6 and the
    reconstruction ``recon`` (first order when ``None``)."""
    from pops.numerics import reconstruction, riemann, variables
    from pops.numerics.spatial import FiniteVolume

    return FiniteVolume(flux=view.elaborated, variables=variables.Conservative(view.state),
                        reconstruction=reconstruction.FirstOrder() if recon is None else recon,
                        riemann=getattr(riemann, face)())


def _derivative_speed(word: str, view: TermView, site: Any) -> None:
    raise AmbiguousElaboration(
        "the speed of %s under %s would read the derivative of a field, %s, so it is not "
        "derived" % (word, view.term, view.value),
        write="upwind(speed=f), f a function of face that returns the normal speed", site=site)


class _FaceFlux(SchemeValue):
    kinds = ("Flux",)

    def _reconstruction(self, view: TermView) -> Any:
        """The b6 reconstruction of ``reconstruction=r``, a scheme that returns Reconstruct."""
        recon = self.options.get("reconstruction")
        return None if recon is None else recon.reconstruction_of(view)

    def _recon_text(self) -> str:
        recon = self.options.get("reconstruction")
        return "FirstOrder()" if recon is None else \
            "reconstruction.User(the Reconstruct of %r)" % (recon,)

    def _waves(self, view: TermView) -> None:
        if view.slope is None:
            if view.reads_derivative(view.value):
                _derivative_speed("%s()" % self.word, view, self.site)
            raise NotYetElaborated("the wave speeds of %s = %s, which this lot does not derive"
                                   % (view.term, view.value), _DERIVATIVES, site=self.site)


class _Upwind(_FaceFlux):
    def realize(self, view: TermView) -> Any:
        recon = self._reconstruction(view)
        speed = self.options["speed"]
        if speed is not None:
            from pops.lib.numerics import fluxes

            return fluxes.upwind(view.elaborated, view.state,
                                 speed=view.face_function(speed, "upwind(speed=f)",
                                                          recon is not None),
                                 reconstruction=recon)
        if view.unit_flux is None:
            if view.reads_derivative(view.value):
                _derivative_speed("upwind()", view, self.site)
            raise RealizationMismatch(
                "upwind() covers a flux F(1) u, linear in its scalar quantity u, and the flux "
                "of %s is %s" % (view.quantity, view.value),
                write="rusanov() or hll(), which take the wave speeds of the flux",
                site=self.site)
        return _first_order(view, "Rusanov", recon)

    def describe(self) -> str:
        if self.options["speed"] is not None:
            return ("FiniteVolume(variables=Conservative(u), reconstruction=%s, riemann=riemann."
                    "User(body=s u_L where s > 0 else s u_R, stability=abs(s))), s = %s(face), "
                    "stated" % (self._recon_text(), getattr(self.options["speed"], "__name__",
                                                           "f")))
        return ("FiniteVolume(variables=Conservative(u), reconstruction=%s, "
                "riemann=Rusanov()), the upwind flux s u_L or s u_R of a flux F(1) u whose "
                "waves are s = F(1)" % self._recon_text())


class _Rusanov(_FaceFlux):
    def realize(self, view: TermView) -> Any:
        self._waves(view)
        return _first_order(view, "Rusanov", self._reconstruction(view))

    def describe(self) -> str:
        return ("FiniteVolume(variables=Conservative(u), reconstruction=%s, "
                "riemann=Rusanov()), with the waves dF/du of the flux" % self._recon_text())


class _HLL(_FaceFlux):
    def realize(self, view: TermView) -> Any:
        self._waves(view)
        return _first_order(view, "HLL", self._reconstruction(view))

    def describe(self) -> str:
        return ("FiniteVolume(variables=Conservative(u), reconstruction=%s, "
                "riemann=HLL()), with the waves dF/du of the flux" % self._recon_text())


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

    def linear_operator(self, view: TermView, *, weight: Any, layout: Any) -> _TwoPointLinear:
        """The linear map of the rate of the key, w div(k grad u), as the matrix-free apply of a
        solve reads it (the lot « solves », A1): on a uniform Cartesian periodic layout, the
        two-point flux of -k grad u with k constant is k (u_+ - 2 u + u_-) / dx^2 on each axis,
        b6's shared Laplacian scaled by k. ``weight`` is the weight of div(F^d) in the rate;
        ``layout`` holds ``uniform``, ``cartesian`` and ``periodic``. Each condition is checked on
        the elaborated key; another key is ``NotYetElaborated`` (G10 generates its operator)."""
        if view.reads_quantity(view.coefficient):
            raise NotYetElaborated(
                "two_point() under %s in a solve, whose coefficient %s varies with the state"
                % (view.term, view.coefficient), "G10, the operator generated from the term",
                site=self.site)
        if not (layout["uniform"] and layout["cartesian"] and layout["periodic"]):
            raise NotYetElaborated(
                "two_point() under %s in a solve on a layout that is not uniform, Cartesian and "
                "periodic" % view.term, "G10, the operator generated from the term",
                site=self.site)
        k = _constant(view.coefficient, view.term)
        return _TwoPointLinear(view.term, weight * k, str(view.coefficient))


class _TwoPointLinear:
    """The rate w div(k grad u) of a two-point diffusion as a linear map L(v) = kappa Lap_h(v),
    kappa = w k, with its claims (IR16): symmetric, and negative semidefinite when kappa >= 0."""

    __slots__ = ("term", "kappa", "spelled")

    def __init__(self, term: Any, kappa: Any, spelled: str) -> None:
        self.term = term
        self.kappa = kappa
        self.spelled = spelled

    def apply(self, P: Any, value: Any) -> Any:
        """``kappa * P.laplacian(v)`` on the apply ``P`` of a matrix-free operator."""
        return self.kappa * P.laplacian(P.scalar_field("laplacian"), value)

    def claims(self) -> Any:
        from pops.time._program.relations import LinearClaims

        positive = self.kappa >= 0
        return LinearClaims(
            symmetric=True, negative_semidefinite=positive,
            description="the two_point() rate of %s, kappa*Laplacian with kappa = %r"
                        % (self.term, self.kappa),
            proof=("kappa = %s = %r >= 0 and the shared Laplacian of a uniform periodic grid is "
                   "symmetric negative semidefinite" if positive else
                   "kappa = %s = %r is negative, so the map is not negative semidefinite")
                  % (self.spelled, self.kappa))


def _constant(value: Any, term: Any) -> Any:
    """The number of a coefficient written with numbers and model constants (Q4); a parameter is
    bound at bind, which the matrix-free apply of b6 cannot read (the lot « solves »)."""
    from fractions import Fraction

    from pops.surface._expr import Arith
    from pops.surface._geometry import MathConstant
    from pops.surface._model import ModelConstant, Parameter

    if isinstance(value, bool):
        raise TypeError("a diffusion coefficient is a number, got %r" % (value,))
    if isinstance(value, (int, float, Fraction)):
        return value
    if isinstance(value, MathConstant):
        return value.value
    if isinstance(value, ModelConstant):
        return value.value
    if isinstance(value, Parameter):
        raise NotYetElaborated(
            "two_point() under %s in a solve, whose coefficient reads the parameter %s: the "
            "matrix-free apply of the cg() bridge has no coefficient bound at bind; write "
            "model.constant(%r, value=...)" % (term, value.name, value.name),
            "the reference CG of N9 in pops.lib.solve with the operator generated by G10, "
            "which read a parameter as any value (MG9 bridge cg())")
    if isinstance(value, Arith):
        args = [_constant(arg, term) for arg in value.args]
        if value.op == "neg":
            return -args[0]
        left, right = args
        return {"+": lambda: left + right, "-": lambda: left - right, "*": lambda: left * right,
                "/": lambda: left / right, "**": lambda: left ** right}[value.op]()
    raise NotYetElaborated("two_point() in a solve with the coefficient %s" % (value,),
                           "G10, the operator generated from the term")


def _checked_reconstruction(word: str, reconstruction: Any) -> Any:
    if reconstruction is not None and not (isinstance(reconstruction, SchemeValue) and getattr(
            getattr(reconstruction, "scheme", None), "is_cell", False)):
        raise TypeError("%s(reconstruction=r) takes a scheme that returns Reconstruct, called "
                        "with its options, such as muscl(limiter=minmod); got %r"
                        % (word, reconstruction))
    return reconstruction


def upwind(*, speed: Any = None, reconstruction: Any = None) -> SchemeValue:
    """``upwind(speed=None, reconstruction=None)``: s u_L where s > 0, else s u_R (2.6.9).

    Without ``speed`` the flux must be F(1) u, u its scalar quantity, and s = F(1)·n is derived;
    another flux is refused (``RealizationMismatch``, naming ``rusanov()`` and ``hll()``). With
    ``speed=f``, s = f(face) is the signed normal speed, and nothing is checked against the flux;
    the face records |s| (SD44).
    """
    if isinstance(speed, dict):
        raise NotYetElaborated("upwind(speed={factor: f}), a speed per factor of a product "
                               "support", _PRODUCT)
    if speed is not None:
        check_plain_function(speed, "upwind(speed=f)")
    return _Upwind("upwind", speed=speed,
                   reconstruction=_checked_reconstruction("upwind", reconstruction))


def rusanov(*, reconstruction: Any = None) -> SchemeValue:
    """``rusanov(reconstruction=None)``: the mean of the two fluxes less s (u_R - u_L) / 2, s the
    larger spectral radius of the two sides (R9)."""
    return _Rusanov("rusanov", reconstruction=_checked_reconstruction("rusanov", reconstruction))


def hll(*, reconstruction: Any = None) -> SchemeValue:
    """``hll(reconstruction=None)``: the HLL flux with the slowest and fastest speeds (R9)."""
    return _HLL("hll", reconstruction=_checked_reconstruction("hll", reconstruction))


def two_point() -> SchemeValue:
    """``two_point()``: F(u_f, g_f)·n with the face mean u_f and the two-point normal derivative
    g_f of each gradient variable (2.6.9, SD51)."""
    return _TwoPoint("two_point")


__all__ = ["hll", "rusanov", "two_point", "upwind"]
