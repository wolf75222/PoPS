"""The solve algorithms of the library, written with ``@pops.solver`` (spec 0.3 2.6.9, SV9).

``cg(preconditioner=None, max_iterations=2000)`` is the reference conjugate gradient of the row
``cg`` of 2.6.9. A case passes the called word as the value of ``solvers={key: cg()}`` (SV4), and
the solve gives it its request (SV3). The native ``solve_cg`` is a bound substitution of this
reference (SV12, SV13) that runs only under ``pops.auto()`` until SV-T2 qualifies it at 64 ranks
(N9 lot 4, Q2).

**The relation solves of a traced method (N15 lot « solves », §5.5) run the bound native.** A
``pops.solve`` of a relation in a method body composes its operator with ``matrix_free_operator``
and solves it with b6's conjugate gradients, ``pops.solvers.CG``, under the request of the solve:
the passes that read an accepted step (the affine decomposition of the accepted transport, the
trial rows and the attempt reuse of N8, the partition and exchange passes) know that node and not
yet the outcome of the reference. ``cg`` therefore stays on the MG9 list of bridges, with this
narrower reason, until the N9 lot « La référence sous les solves de relation » runs the reference
there (N9 lot 4, step 2, ruling of 12/11 00:52). Conjugate gradients need a symmetric positive
definite operator: the relation solve derives it from the claims of its components (IR16) and
refuses it otherwise with ``PropertyNotDerived`` (K4).

The bridge also states the linear map of the rate that a relation solve reads at the unknown (the
matrix-free apply, ``linear_operator``), where N3 lot 1 put it beside ``cg()`` (D7): the
two-point diffusion of ``two_point()``, recognized by the normal form of its face, of a coefficient
that is a number or a ``model.constant``, on a uniform Cartesian periodic layout, is kappa times
b6's shared Laplacian. Every other term is refused with the lot that generates its operator (G10).

Every other word of 2.6.9 is importable and refuses its use with the lot that elaborates it.
"""
from __future__ import annotations

from typing import Any

from .krylov import cg

_LATER = "N9 (the references of SV9 and the selection of SV11)"
_G10 = "G10, the operator generated from the term"


def _native_cg(options: dict, request: Any) -> Any:
    """The b6 descriptor of conjugate gradients that a relation solve runs under ``cg(...)``, with
    the request of the solve (SV3, SV4): the bridge of MG9 named above."""
    from pops.solvers import CG

    max_iterations = options["max_iterations"]
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) \
            or max_iterations < 1:
        raise ValueError("cg(max_iterations=n) takes an integer n >= 1, got %r"
                         % (max_iterations,))
    rtol = 0.0 if request.rtol is None else float(request.rtol)
    atol = None if request.atol is None else float(request.atol)
    return CG(max_iter=max_iterations, rel_tol=rtol, abs_tol=atol)


cg.bind_relation_native(
    _native_cg, requires=("symmetric_positive_definite",),
    description="the native conjugate gradients of b6, pops.solvers.CG, which a relation solve of "
                "a traced method runs (a bridge of MG9 until the N9 lot « La référence sous les "
                "solves de relation » runs the reference there)")


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

    def claims(self) -> _TwoPointClaims:
        """The claims of the map (IR16), with their proof, as the solves of a method read them."""
        positive = bool(self.kappa >= 0)
        proof = ("kappa = %s = %r >= 0 and the shared Laplacian of a uniform periodic grid is "
                 "symmetric negative semidefinite" if positive else
                 "kappa = %s = %r is negative, so the map is not negative semidefinite") \
            % (self.spelled, self.kappa)
        return _TwoPointClaims(
            positive, proof, "the two_point() rate of %s, kappa*Laplacian with kappa = %r"
            % (self.term, self.kappa))


class _TwoPointClaims:
    """The claims of a two-point linear map (IR16): symmetric, negative semidefinite when
    kappa >= 0, with the proof and the description that ``pops.inspect`` reports."""

    __slots__ = ("symmetric", "negative_semidefinite", "proof", "description")

    def __init__(self, negative_semidefinite: bool, proof: str, description: str) -> None:
        self.symmetric = True
        self.negative_semidefinite = negative_semidefinite
        self.proof = proof
        self.description = description


def linear_operator(view: Any, *, scheme: Any, weight: Any, layout: Any) -> Any:
    """The linear map of the rate of the key of ``view`` under ``scheme``, w div(k grad u), as
    the matrix-free apply of a solve of this module reads it (the elaborator also takes it when
    ``solvers=`` states no solver, whose solve N8 then refuses with SV11): on a uniform Cartesian
    periodic layout, the two-point flux of -k grad u with k constant is
    k (u_+ - 2 u + u_-) / dx^2 on each axis, b6's shared Laplacian scaled by k. ``weight`` is the
    weight of div(F^d) in the rate; ``layout`` holds ``uniform``, ``cartesian`` and
    ``periodic``."""
    from pops.surface import NotYetElaborated

    site = getattr(scheme, "site", None)
    recognized = getattr(scheme, "is_two_point", None)
    if recognized is None or not recognized(view.term):
        raise NotYetElaborated(
            "the term %s under %r read at the unknown of a solve: cg() states no linear map "
            "of it; this lot solves the two_point() diffusion of a constant coefficient"
            % (view.term, scheme), _G10 + ", with the Newton solves of N9", site=view.site)
    if view.reads_quantity(view.coefficient):
        raise NotYetElaborated(
            "two_point() under %s in a solve, whose coefficient %s varies with the state"
            % (view.term, view.coefficient), _G10, site=site)
    if not (layout["uniform"] and layout["cartesian"] and layout["periodic"]):
        raise NotYetElaborated(
            "two_point() under %s in a solve on a layout that is not uniform, Cartesian and "
            "periodic" % view.term, _G10, site=site)
    parameters = view.parameters(view.coefficient)
    if parameters:
        raise NotYetElaborated(
            "two_point() under %s in a solve, whose coefficient reads the parameter %s: the "
            "matrix-free apply of the cg() bridge has no coefficient bound at bind; write "
            "model.constant(%r, value=...)" % (view.term, parameters[0], parameters[0]),
            "the reference CG of N9 in pops.lib.solve with the operator generated by G10, "
            "which read a parameter as any value (MG9 bridge cg())", site=site)
    k = view.number(view.coefficient)
    if k is None:
        raise NotYetElaborated("two_point() under %s in a solve with the coefficient %s"
                               % (view.term, view.coefficient), _G10, site=site)
    return _TwoPointLinear(view.term, weight * k, str(view.coefficient))


def _later(word: str) -> Any:
    def refused(*args: Any, **kwargs: Any) -> Any:
        from pops.surface import NotYetElaborated

        raise NotYetElaborated("pops.lib.solve.%s" % word, _LATER)
    refused.__name__ = word
    return refused


auto = _later("auto")
bicgstab = _later("bicgstab")
gmres = _later("gmres")
CGS2 = _later("CGS2")
richardson = _later("richardson")
jacobi = _later("jacobi")
newton = _later("newton")
dense_lu = _later("dense_lu")

__all__ = ["CGS2", "auto", "bicgstab", "cg", "dense_lu", "gmres", "jacobi", "newton",
           "richardson"]
