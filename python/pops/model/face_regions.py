"""Face regions of an axis-wise face kernel (spec 0.3 IR14, IR17, IR20, IR21).

A face kernel computes one value on each face normal to an axis a, from the cells that
``Compose(Shift(o), Incidence(face(a), cell, side))`` reaches, and delivers it to the left
incident cell with weight -1 and to the right one with weight +1: the native assembly
net(cell) = J(cell) - J(cell + e_a) (D6). Its native face loop expects the class face incidence.

The builders of the numerics layer state their reads through this module: the typed maps live in
``pops._ir``, which the import graph keeps out of ``pops.numerics``. The analyses and the
classification of the regions stay in ``pops._ir.index_analysis``.
"""
from __future__ import annotations

from pops._ir.index_analysis import Read, Region
from pops._ir.index_maps import Compose, Incidence, Shift, cell, face


def face_regions(name, rank, reads_by_axis, *, basis="derived"):
    """One region per axis a, named ``name/axis a``, over the faces normal to a.

    ``reads_by_axis[a]`` holds ``(array, offset, side)`` triples: the read of ``array`` at the
    integer vector ``offset`` of length ``rank``, counted from the ``side`` incident cell of the
    face, ``"left"`` or ``"right"``. ``basis`` says whether a body gave the reads (``"derived"``)
    or a native catalogue declared them (``"declared"``, IR24).
    """
    regions = []
    for axis, reads in enumerate(reads_by_axis):
        regions.append(Region(
            "%s/axis %d" % (name, axis), face(axis), rank,
            tuple(Read(array, Compose(Shift(offset), Incidence(face(axis), cell, side)))
                  for array, offset, side in reads),
            ((Incidence(face(axis), cell, "left"), -1), (Incidence(face(axis), cell, "right"), 1)),
            expected="face_incidence", basis=basis))
    return tuple(regions)


__all__ = ["face_regions"]
