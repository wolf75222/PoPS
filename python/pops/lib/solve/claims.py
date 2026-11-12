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
    description that ``pops.inspect`` reports."""

    __slots__ = ("symmetric", "negative_semidefinite", "homogeneous", "proof", "description")

    def __init__(self, term: Any, kappa: Any, spelled: str) -> None:
        positive = bool(kappa >= 0)
        self.symmetric = True
        self.negative_semidefinite = positive
        self.homogeneous = True
        self.proof = ("kappa = %s = %r >= 0 and the two-point flux of a uniform periodic grid is "
                      "symmetric negative semidefinite, linear and homogeneous in the state"
                      if positive else
                      "kappa = %s = %r is negative, so the rate is not negative semidefinite") \
            % (spelled, kappa)
        self.description = "the two_point() rate of %s, kappa = %r" % (term, kappa)


def linear_claims(view: Any, *, scheme: Any, weight: Any, layout: Any) -> Any:
    """The claims of the rate of the key of ``view`` under ``scheme``, w div(k grad u), read at the
    unknown of a solve (the operator of the solve evaluates that rate at its operand, §3.4 of the
    N9 lot « La référence sous les solves de relation »): on a uniform Cartesian periodic layout,
    the two-point flux of -k grad u with k constant is linear and homogeneous in u, symmetric, and
    negative semidefinite when w k >= 0. ``weight`` is the weight of div(F^d) in the rate;
    ``layout`` holds ``uniform``, ``cartesian`` and ``periodic``."""
    site = getattr(scheme, "site", None)
    recognized = getattr(scheme, "is_two_point", None)
    if recognized is None or not recognized(view.term):
        raise _not_yet(
            "the term %s under %r read at the unknown of a solve: its rate is not stated linear "
            "and homogeneous in the state; this lot solves the two_point() diffusion of a "
            "constant coefficient" % (view.term, scheme), _G10 + ", with the Newton solves of N9",
            site=view.site)
    if view.reads_quantity(view.coefficient):
        raise _not_yet(
            "two_point() under %s in a solve, whose coefficient %s varies with the state"
            % (view.term, view.coefficient), _G10, site=site)
    if not (layout["uniform"] and layout["cartesian"] and layout["periodic"]):
        raise _not_yet(
            "two_point() under %s in a solve on a layout that is not uniform, Cartesian and "
            "periodic" % view.term, _G10, site=site)
    parameters = view.parameters(view.coefficient)
    if parameters:
        raise _not_yet(
            "two_point() under %s in a solve, whose coefficient reads the parameter %s: the claim "
            "of its rate from the admissibility of %s; write model.constant(%r, value=...)"
            % (view.term, parameters[0], parameters[0], parameters[0]),
            "the exit test of the N9 lot « La référence sous les solves de relation » (L1-d with "
            "nu a model.parameter, §3.5)", site=site)
    k = view.number(view.coefficient)
    if k is None:
        raise _not_yet("two_point() under %s in a solve with the coefficient %s"
                               % (view.term, view.coefficient), _G10, site=site)
    return _TwoPointClaims(view.term, weight * k, str(view.coefficient))


__all__ = ["linear_claims"]
