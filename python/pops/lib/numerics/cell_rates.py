"""The scheme word of a cell rate, written with ``@pops.scheme`` (spec 0.3, 2.6.9 row 991; A4, SU5,
CF4). N15, FaceFlux/CellRate lot FC-1 (ROM-98, note §3.5).

``pointwise()`` is what ``pops.Numerics(schemes={S: pointwise()})`` takes for a source S: the law
of the key at the cell value, ``CellRate(S(u_c))``, which the surface lowers to the b6 named source
of S selected from its rate and evaluated at each cell (``StateStorage()``). Its body is the one of
the gallery (``lib_schemes.py``, D8), byte for byte with its docstring and its signature: under a
definition, an inventory or a relation it would return ``CellValue`` or ``CellResidual``, which the
model refuses first in this lot (their keys belong to later lots), and ``gradient=`` reads
``cell.gradient``, which the lot of scheme gradients of N15 (GR) elaborates.
"""
import pops.surface as pops
from pops.surface.discrete import CellRate, CellResidual, CellValue, kinds

from .fluxes import _differentiated


@pops.scheme
def pointwise(cell, term, gradient=None):
    """The law S of the key at the cell values u_c of the quantities that it reads: a source or an interaction as
    the CellRate S(u_c), S being for an interaction the table of its deliveries; a definition or an inventory as
    the CellValue S(u_c), S being for an inventory the table of its densities (SD47); a model relation K == 0, such
    as a closure, as the CellResidual K(u_c), its unknowns being among the quantities that it reads and determined
    by the solve of its handle. An integral over a support that is only integrated over uses the nodes and weights
    of the quadrature rule that is the layout of that support (SD31). A law that reads grad(w) of an analytic
    definition, one that reads only coordinates, constants and parameters, takes it exactly; a law that reads the
    gradient of a discrete field takes it from the cell gradient scheme `gradient`, such as centred_gradient(),
    and is refused without it (AmbiguousElaboration, naming gradient=) (SD72)."""
    states = {r: cell.value(r) for r in term.quantities}
    if gradient is None:                    # Python data: only exact gradients of analytic definitions (SD72)
        value = term.law(states=states)
    else:
        value = term.law(states=states,
                         gradients={w: cell.gradient(w, scheme=gradient) for w in _differentiated(term)})
    kind = term.kind
    if kind is kinds.Definition or kind is kinds.Inventory:
        return CellValue(value)
    if kind is kinds.Relation:
        return CellResidual(value)
    return CellRate(value)


__all__ = ["pointwise"]
