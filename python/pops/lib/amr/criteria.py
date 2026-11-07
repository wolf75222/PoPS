"""Refinement-criterion reads written as functions of ``cell`` (spec 0.3, DM13b).

A function of ``cell`` given to ``Tag`` or ``Coarsen`` returns a Boolean expression of cell reads.
These helpers are such reads, written once in Python over the public ``cell`` surface: the compiled
Program generates their arithmetic like any other region.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pops.math import sqrt


def gradient_norm(cell: Any, quantity: Any, *, axes: Iterable[Any], stencil: Any) -> Any:
    """The l2 norm of the discrete gradient of ``quantity`` at ``cell``.

    ``quantity`` is a block state, or one typed component of it, read with ``cell.value``; or a
    function of ``cell``, such as a pressure written from the conserved components, evaluated at
    each cell of the stencil. ``stencil`` is the discrete gradient of the method that discretizes
    the state, the value of its ``amr_indicator_stencil(dimension=...)``: one row of ``offsets``
    and ``coefficients`` for each of ``axes``. The method owns the stencil, so there is no default.
    The derivative along each axis applies its row to the cells around ``cell`` and divides by the
    cell width; the terms and then the squared derivatives are added in order. This is the
    arithmetic of the ``gradient_above``/``gradient_below`` leaves of the tag-graph VM, so a
    criterion that compares the norm of a state with a threshold decides cell for cell as
    ``norm(grad(...))`` does under the stencil of the same method.
    """
    rows = getattr(stencil, "axes", None)
    if not isinstance(rows, tuple) or not rows or any(
            not hasattr(row, "offsets") or not hasattr(row, "coefficients") for row in rows):
        raise TypeError("gradient_norm requires the discrete gradient stencil of a method "
                        "(amr_indicator_stencil)")
    directions = tuple(axes)
    if not directions:
        raise ValueError("gradient_norm requires at least one axis")
    if len(directions) != len(rows):
        raise ValueError("gradient_norm requires one stencil row per axis")
    read: Callable[[Any], Any] = (
        quantity if callable(quantity) else lambda at: at.value(quantity))
    squared = None
    for axis, row in zip(directions, rows, strict=True):
        derivative = None
        for offset, coefficient in zip(row.offsets, row.coefficients, strict=True):
            term = coefficient * read(cell.shift(axis, offset))
            derivative = term if derivative is None else derivative + term
        derivative = derivative / cell.width(axis)
        square = derivative * derivative
        squared = square if squared is None else squared + square
    return sqrt(squared)


__all__ = ["gradient_norm"]
