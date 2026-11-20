"""The words of ``pops.discrete`` (spec 0.3, 2.6.4): the builders, the reductions, the speeds of a
flux and the words of the kinds and the forms, which a case that writes a scheme imports (SU13).

Lot 2 of N15 elaborates ``FaceFlux`` (without ``left=``, ``right=`` and ``volume=``) under a
``Flux`` key and ``Reconstruct`` as the ``reconstruction=`` of a face scheme, lot B ``Ghost`` under
a boundary policy (``boundary=``), the FaceFlux/CellRate lot FC-1 ``CellRate`` under a ``Source``
key, ``spectral_radius``,
``slowest_speed`` and ``fastest_speed`` of a flux at a trace, and ``kinds`` and ``forms``; N16
lot 1 adds the carriers ``Cell``, ``Node``, ``Face`` and ``Edge`` of ``pops.Coefficients(on=...)``,
of which it elaborates ``Cell`` (N16 lot 3 ``Node``, G08 ``Face`` and ``Edge``). Every
other builder and reduction of 2.6.4 is importable and refuses its use with ``NotYetElaborated``,
which names the lot that elaborates it. The module exports no entity: ``face``, ``cell``, ``term``
and ``axis`` exist only as parameters of a scheme (SU7). An unknown name is refused with
``SymbolNotInSurface``, whose message lists the words of the module (SU13).
"""
from typing import Any as _Any

from ._refusal import NotYetElaborated as _NotYetElaborated
from ._refusal import SymbolNotInSurface as _SymbolNotInSurface
from ._representation import Cell, Edge, Face, Node
from ._scheme import (
    CellRate, FaceFlux, Ghost, Reconstruct, fastest_speed, forms, kinds, slowest_speed,
    spectral_radius,
)
from ._scheme import _later_builder

_BUILDERS = ("FaceValue", "CellValue", "CellResidual", "BoundaryFlux",
             "EdgeFlux", "IntervalFaceFlux", "Transfer")
_REDUCTIONS = ("sum_faces", "sum_children", "sum_fibre", "sum_edges", "sum_cells",
               "largest_cells", "smallest_cells")
_REDUCTION_OWNER = ("a later lot of N15 with the cell schemes and their reductions (sources after "
                    "N11, boundaries after G04, hierarchies after G13)")

for _word in _BUILDERS:
    globals()[_word] = _later_builder(_word)


def _reduction(word: str) -> _Any:
    def reduction(*args: _Any, **kwargs: _Any) -> _Any:
        raise _NotYetElaborated("pops.discrete.%s" % word, _REDUCTION_OWNER).with_traceback(None)

    reduction.__name__ = reduction.__qualname__ = word
    reduction.__doc__ = "``%s``, a reduction of 2.6.4 that a later lot elaborates." % word
    return reduction


for _word in _REDUCTIONS:
    globals()[_word] = _reduction(_word)

__all__ = [  # pyright: ignore[reportUnsupportedDunderAll]
    "FaceFlux", "Reconstruct", "Ghost", "CellRate", *_BUILDERS, *_REDUCTIONS, "spectral_radius",
    "slowest_speed", "fastest_speed", "kinds", "forms",
    # the carriers of pops.Coefficients(on=...) (N16 lot 1, D1b)
    "Cell", "Node", "Face", "Edge"]


def __getattr__(name: str) -> _Any:
    """Refuse a name that pops.discrete does not list (SU7, SU13); export nothing."""
    if name.startswith("_"):
        raise AttributeError(name)
    raise _SymbolNotInSurface("pops.discrete has no word %s; its words are %s, and the entities "
                             "face, cell, term and axis are the parameters of a scheme (SU7)"
                             % (name, ", ".join(__all__)),
                             write="one of these words, or the entity as a parameter of the "
                                   "scheme that receives it")
