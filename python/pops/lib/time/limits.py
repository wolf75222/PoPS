"""``cfl_limit(flux, cfl=, max_dt=None)``: master's ``AdaptiveCFL`` as a function of ``cell`` (spec 0.3,
2.6.2, 2.6.9, SD44, SD70; N8 « dt= as a function of cell », design note D1).

``pops.Advance(method, dt=cfl_limit(advection, cfl=0.45, max_dt=1e-2))`` bounds each step by the
smallest value over the cells of

    f(cell) = (cfl * w) / s

with ``w`` the smallest width of the cell over the axes and ``s`` the largest spectral radius of
the fluxes at the cell value over the axes (``cell.speed(F, a)``, R9). A cell at rest bounds
nothing (2.6.9): its quotient is +infinity in IEEE 754, the value of E that the step rule admits,
since a constant of an expression is finite. With ``max_dt`` the value is
``smallest([where(s > 0, (cfl * w) / s, max_dt), max_dt])``: the cell at rest takes the cap, and
the quotient, an observed operation (C4), is not computed there. On a uniform layout this is master's ``cfl . h_min / max speed``
(``include/pops/runtime/system.hpp:872-874@bb9``), operation for operation; there is no speed
floor, unlike b6's ``max(s, 1e-30)``. ``cfl_limit([advection, drift], cfl=)`` keeps the smallest
of their limits.

The value is a plain function of ``cell`` on which ``cfl_limit`` sets ``step_fraction = cfl``: the
Courant budget of the step rule (C2), the fraction of the stable step that the face guards of each
attempt check, in (0, 1]. ``cfl > 1``, which master's ``AdaptiveCFL`` accepts, is refused: a
budget above one would loosen the guards of a fixed step (review rev 3, P1).

The rule reads the speeds of the transport fluxes it names; at elaboration, when the model is
complete, it refuses a model whose transport flux it leaves out, and a model with a diffusive
flux, whose limit it does not write: a function of cell written by hand writes both
(``combined_limit`` of L1). A flux whose waves are not derived is refused by ``cell.speed``.
"""
from __future__ import annotations

import math as _math
from typing import Any

from pops.surface.math import largest, smallest, where


def _kind(term: Any) -> Any:
    """The kind of a term of the model: ``"Flux"`` for a transport flux, ``"DiffusiveFlux"``."""
    return getattr(term, "kind", None)


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("cfl_limit(flux, %s=...) takes a positive number, got %r" % (name, value))
    value = float(value)
    if not _math.isfinite(value) or value <= 0.0:
        raise ValueError("cfl_limit(flux, %s=...) takes a positive finite number, got %r"
                         % (name, value))
    return value


def _fluxes(flux: Any) -> tuple:
    listed = tuple(flux) if isinstance(flux, (list, tuple)) else (flux,)
    if not listed:
        raise TypeError("cfl_limit(...) takes a flux of the model or a list of them")
    for item in listed:
        if _kind(item) != "Flux":
            raise TypeError("cfl_limit(...) takes transport fluxes of the model, model.flux(of=u, "
                            "value=...), got %r" % (item,))
    if len({id(item) for item in listed}) != len(listed):
        raise ValueError("cfl_limit(...) names each flux once")
    if len({id(item.model) for item in listed}) != 1:
        raise ValueError("cfl_limit(...) takes fluxes of one model")
    return listed


def _name(fluxes: tuple) -> str:
    names = [str(flux) for flux in fluxes]
    return names[0] if len(names) == 1 else "[%s]" % ", ".join(names)


def _check_model(fluxes: tuple, written: str) -> None:
    """The model reads no limit that the rule does not write (C5): every transport flux of its
    balances is named, and it has no diffusive flux."""
    model = fluxes[0].model
    named = {id(flux) for flux in fluxes}
    # a term in no balance is left to the coverage of the terms, which refuses it (T5)
    integrated = {id(term) for balance in model.balances for term in balance.terms}
    for term in model.terms:
        if id(term) not in integrated:
            continue
        if _kind(term) == "DiffusiveFlux":
            raise ValueError(
                "%s in a model with the diffusive flux %s, whose limit cfl_limit does not write: "
                "write dt= as a function of cell with both limits, as L1's combined_limit (the "
                "step rules of 5.11 with a diffusive limit are a later lot of N8)" % (written, term))
        if _kind(term) == "Flux" and id(term) not in named:
            raise ValueError(
                "%s leaves out the transport flux %s of the model: cfl_limit([...]) names every "
                "transport flux (the step rules of 5.11 per flux are a later lot of N8)"
                % (written, term))


def cfl_limit(flux: Any = None, *, cfl: Any = None, max_dt: Any = None, **earlier: Any) -> Any:
    """``cfl_limit(flux, cfl=, max_dt=None)``, or a list of fluxes (2.6.9): a function of cell."""
    if "number" in earlier:
        raise TypeError("cfl_limit(flux, cfl=...) takes master's keyword cfl=, not number= "
                        "(SD70)")
    if earlier:
        raise TypeError("cfl_limit has no option %s" % ", ".join(sorted(earlier)))
    if flux is None:
        raise TypeError("cfl_limit(flux, cfl=...) takes the flux whose speeds bound the step")
    if cfl is None:
        raise TypeError("cfl_limit(flux, cfl=...) needs cfl=, the Courant number")
    fluxes = _fluxes(flux)
    number = _positive(cfl, "cfl")
    if number > 1.0:
        raise ValueError(
            "cfl_limit(%s, cfl=%r): the Courant number of a step rule is in (0, 1], the budget "
            "that the face guards of its steps check; master's AdaptiveCFL accepted more"
            % (_name(fluxes), cfl))
    cap = None if max_dt is None else _positive(max_dt, "max_dt")
    written = "cfl_limit(%s, cfl=%r%s)" % (_name(fluxes), number,
                                          "" if cap is None else ", max_dt=%r" % cap)

    def limit(cell: Any) -> Any:
        _check_model(fluxes, written)
        axes = fluxes[0].quantity.space.support.axes
        speeds = [cell.speed(item, axis) for item in fluxes for axis in axes]
        widths = [cell.width(axis) for axis in axes]
        s = speeds[0] if len(speeds) == 1 else largest(speeds)
        w = widths[0] if len(widths) == 1 else smallest(widths)
        if cap is None:
            # a cell at rest bounds nothing: (cfl * w) / 0 is +infinity in IEEE 754, E itself,
            # which the step rule admits (2.6.9); pops.math has no infinite literal to write
            return (number * w) / s
        return smallest([where(s > 0, (number * w) / s, cap), cap])

    limit.__name__ = "cfl_limit"
    limit.__qualname__ = "cfl_limit"
    limit.step_fraction = number  # type: ignore[attr-defined]
    return limit


__all__ = ["cfl_limit"]
