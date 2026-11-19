"""The claims of the rate of a term read at the unknown of a relation solve (IR16), beside
``cg()`` where N3 lot 1 put the map of ``two_point()`` (D7).

The operator of a relation solve evaluates the rates of its components at a State operand (N9 lot
« La référence sous les solves de relation », §3.4), so each term states what the solver needs to
know of its rate: symmetric, negative semidefinite, and linear and homogeneous in the state, with
the proof that ``pops.inspect`` reports. This module is infrastructure beside the words written with
the surface (SU-T1 lints those): it imports no module of PoPS, and reaches the refusal class of the
surface through importlib, as ``pops.lib.output`` does.
"""
from __future__ import annotations

from typing import Any

_G10 = "G10, the operator generated from the term"


def _not_yet(construct: str, later: str, site: Any = None) -> Exception:
    """The refusal of the surface for a construct that a later lot elaborates. The claims and the
    refusals of this module are infrastructure beside the words written with the surface (SU-T1
    lints those), so they reach the refusal class through importlib, as ``pops.lib.output`` does."""
    import importlib

    surface = importlib.import_module("pops.surface")
    return surface.NotYetElaborated(construct, later, site=site)


class _TwoPointClaims:
    """The claims of the rate w div(k grad u) of a two-point diffusion read at the unknown of a
    solve (IR16), kappa = w k: symmetric, negative semidefinite when kappa >= 0, and linear and
    homogeneous in the state (k reads no state, the layout is periodic), with the proof and the
    description that ``pops.inspect`` reports.

    A k that reads parameters is bound at bind, so its claim holds for every value the bind
    accepts, not for one: kappa >= 0 when the admissible sets of its parameters derive it
    (``pops.Positive()``, which the bind checks), and no claim when they derive no sign (§3.5 of
    the N9 lot « La référence sous les solves de relation »)."""

    __slots__ = ("symmetric", "negative_semidefinite", "homogeneous", "proof", "description")

    _OPERATOR = ("the two-point flux of a uniform periodic grid is symmetric negative "
                 "semidefinite, linear and homogeneous in the state")

    def __init__(self, term: Any, kappa: str, kappa_sign: int | None, why: str,
                 value: str | None = None) -> None:
        self.symmetric = True
        self.negative_semidefinite = kappa_sign is not None and kappa_sign >= 0
        self.homogeneous = True
        if kappa_sign is None:
            self.proof = ("the sign of kappa = %s is not derived %s, so the rate is not stated "
                          "negative semidefinite" % (kappa, why))
        elif kappa_sign >= 0:
            self.proof = "kappa = %s >= 0%s, and %s" % (kappa, why, self._OPERATOR)
        else:
            self.proof = "kappa = %s is negative%s, so the rate is not negative semidefinite" \
                % (kappa, why)
        self.description = "the two_point() rate of %s, kappa = %s" % (
            term, kappa if value is None else value)


def _kappa(weight: Any, coefficient: Any) -> str:
    """kappa = w k as ``pops.inspect`` spells it."""
    if weight == 1:
        return str(coefficient)
    if weight == -1:
        return "-(%s)" % (coefficient,)
    return "%s * (%s)" % (weight, coefficient)


def linear_claims(view: Any, *, scheme: Any, weight: Any, layout: Any) -> Any:
    """The claims of the rate of the key of ``view`` under ``scheme``, w div(k grad u), read at the
    unknown of a solve (the operator of the solve evaluates that rate at its operand, §3.4 of the
    N9 lot « La référence sous les solves de relation »): on a uniform Cartesian periodic layout,
    the two-point flux of -k grad u with k constant, or bound at bind (a parameter of the model),
    is linear and homogeneous in u, symmetric, and negative semidefinite when w k >= 0.
    ``weight`` is the weight of div(F^d) in the rate; ``layout`` holds ``uniform``, ``cartesian``
    and ``periodic``."""
    site = getattr(scheme, "site", None)
    recognized = getattr(scheme, "is_two_point", None)
    if recognized is None or not recognized(view.term):
        raise _not_yet(
            "the term %s under %r read at the unknown of a solve: its rate is not stated linear "
            "and homogeneous in the state; this lot solves the two_point() diffusion of a "
            "coefficient that reads no state" % (view.term, scheme), _G10 + ", with the Newton solves of N9",
            site=view.site)
    if view.reads_quantity(view.coefficient):
        raise _not_yet(
            "two_point() under %s in a solve, whose coefficient %s varies with the state"
            % (view.term, view.coefficient), _G10, site=site)
    if not (layout["uniform"] and layout["cartesian"] and layout["periodic"]):
        raise _not_yet(
            "two_point() under %s in a solve on a layout that is not uniform, Cartesian and "
            "periodic" % view.term, _G10, site=site)
    parameters = list(dict.fromkeys(view.parameters(view.coefficient)))
    if parameters:
        # §3.5 of the N9 lot « La référence sous les solves de relation »: k is bound at bind, so
        # the claim holds for every value that the bind accepts, from the admissible sets.
        derived = view.sign(view.coefficient)
        kappa = _kappa(weight, view.coefficient)
        if derived is None:
            return _TwoPointClaims(
                view.term, kappa, None,
                "from the admissible sets of %s (model.parameter(%r, ..., admissible="
                "pops.Positive()) gives a parameter its sign)" % (", ".join(parameters),
                                                                   parameters[0]))
        return _TwoPointClaims(view.term, kappa, ((weight > 0) - (weight < 0)) * derived,
                               " for every value of %s admissible in pops.Positive(), which the "
                               "bind checks" % ", ".join(parameters))
    k = view.number(view.coefficient)
    if k is None:
        raise _not_yet("two_point() under %s in a solve with the coefficient %s"
                               % (view.term, view.coefficient), _G10, site=site)
    kappa = weight * k
    return _TwoPointClaims(view.term, "%s = %r" % (view.coefficient, kappa),
                           (kappa > 0) - (kappa < 0), "", value=repr(kappa))


__all__ = ["linear_claims"]
