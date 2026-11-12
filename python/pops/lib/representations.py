"""Named representations (spec 0.3, D1, D1b, D2 and 2.6.9; N16 design note rev 3, section 3.7).

The core knows one representation construct, ``@pops.representation`` returning
``pops.Coefficients(...)`` or ``pops.Dual(R)``; a named representation is a word of this module,
written with that construct, with no privilege over a representation that a case writes (A4)::

    from pops.lib.representations import CellAverage, CellTotal

    numerics = pops.Numerics(represent={rho: CellAverage()}, schemes={advection: upwind()})

Every word is a function of the cell; its identity is the digest of the normal forms of its decode
and encode, of its carrier, size, rule and kind, never of its name (P10), so that the same
representation written in a case has the same identity (CF4,
``tests/python/unit/surface/test_representation_n16.py``).

========================= ========== ============================ ============ ===================
word                      on, size   decode                       pairing      encode
========================= ========== ============================ ============ ===================
CellAverage(quadrature=)  Cell, 1    coefficients[0]              quadrature   the projection
CellTotal()               Cell, 1    coefficients[0]/cell.measure none         the projection
PointValue()              Cell, 1    coefficients[0]              none         field at the centre
DualLoad(of=)             of ``of``  through the Riesz map of of  of ``of``    <field, phi_s>
FaceNormalIntegral()      Face, 1    lowest-order face element    GaussLeg(2)  face.integral
EdgeCirculation()         Edge, 1    lowest-order edge element    GaussLeg(2)  edge.integral
========================= ========== ============================ ============ ===================

N16 lot 1 elaborates ``CellAverage()`` on the route of ``pops.CellAverage()`` (SU-T3); the other
cell words are refused at ``pops.Numerics`` with the lot that elaborates them (G03 lot R-b for
``CellTotal()``, ``PointValue()`` and a stated rule, N16 lot 2 for ``DualLoad``), and the face and
edge words with G08, which fixes their meaning.
"""
from __future__ import annotations

import pops.surface as pops
from pops.lib.quadrature import GaussLegendre
from pops.surface.discrete import Cell, Edge, Face


@pops.representation
def CellAverage(cell, quadrature=None):
    """The cell average: one coefficient, constant on the cell. No rule unless one is stated, since
    none is needed; ``cell.load`` and a derivation of D3 then state it (index of defaults)."""
    def decode(coefficients, point):
        return coefficients[0]

    return pops.Coefficients(on=Cell, decode=decode, pairing=quadrature, kind="average")


@pops.representation
def CellTotal(cell):
    """The cell total: one coefficient, the integral of the field over the cell, so the field is
    the coefficient divided by the measure; its projection is the integral over the cell."""
    def decode(coefficients, point):
        return coefficients[0] / cell.measure

    return pops.Coefficients(on=Cell, decode=decode, kind="total")


@pops.representation
def PointValue(cell):
    """The value at the centre of the cell: one coefficient, constant on the cell, encoded by
    sampling the field at the centre rather than by the projection."""
    def decode(coefficients, point):
        return coefficients[0]

    def encode(cell, field):
        return field

    return pops.Coefficients(on=Cell, decode=decode, encode=encode, kind="sample")


@pops.representation
def DualLoad(cell, of=None):
    """The loads of a representation, by default of the cell average (D2)."""
    return pops.Dual(CellAverage() if of is None else of)


def _face_and_edge_reads(coefficients, point):
    """The decode of a face or edge element reads the coefficients of the faces or edges of the
    cell, which G08 gives (``face.value(B)``, ``edge.faces(axis)``); it is not traced before."""
    raise pops.NotYetElaborated("the decode of a face or edge carrier", "G08")


@pops.representation
def FaceNormalIntegral(cell):
    """The integral of the normal component over each face, lowest-order face element: component a
    interpolates linearly between the two faces normal to a, each coefficient divided by its face
    measure; encoded by ``face.integral(dot(field, face.normal))``. G08 elaborates it."""
    def encode(face, field):
        raise pops.NotYetElaborated("face.integral(dot(field, face.normal))", "G08")

    return pops.Coefficients(on=Face, decode=_face_and_edge_reads, pairing=GaussLegendre(2),
                             encode=encode, kind="face integral")


@pops.representation
def EdgeCirculation(cell):
    """The circulation of the tangential component along each edge, lowest-order edge element;
    encoded by ``edge.integral(dot(field, edge.tangent))``. G08 elaborates it."""
    def encode(edge, field):
        raise pops.NotYetElaborated("edge.integral(dot(field, edge.tangent))", "G08")

    return pops.Coefficients(on=Edge, decode=_face_and_edge_reads, pairing=GaussLegendre(2),
                             encode=encode, kind="edge circulation")


__all__ = ["CellAverage", "CellTotal", "DualLoad", "EdgeCirculation", "FaceNormalIntegral",
           "PointValue"]
