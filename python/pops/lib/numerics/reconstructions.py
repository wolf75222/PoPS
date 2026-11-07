"""Per-axis reconstructions written in Python (spec 0.3 sections 4.3 and 4.4, IR25; N3).

``Reconstruct(rep, value)`` is the builder of section 4.3: ``value(c, a)`` gives the trace of
the cell ``c`` on its face on the positive side of the axis ``a``, and the trace on the face on
the negative side is the same body with its offsets along ``a`` mirrored. ``muscl(rep, limiter)``
is the MUSCL face value of the catalogue, written with it:

    u_c + limiter(u_c - u_(c-1), u_(c+1) - u_c) / 2

so that master's ``MUSCL(VanLeer())`` is ``muscl(Uh, vanleer)``. The body runs once, now, and
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

from collections.abc import Callable
from typing import Any


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
        formal_order = max(offsets, default=0) - min(offsets, default=0) + 1
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


__all__ = ["Reconstruct", "muscl"]
