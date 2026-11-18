"""The words of 2.6.9 that a later lot elaborates: each is importable from ``pops.lib.solve`` and
refuses its use with the lot that elaborates it. Infrastructure beside the words written with the
surface: it imports no module of PoPS and reaches the refusal class through importlib."""
from __future__ import annotations

from typing import Any

_LATER = "N9 (the references of SV9 and the selection of SV11)"


def _not_yet(construct: str, later: str) -> Exception:
    """The refusal of the surface for a construct that a later lot elaborates."""
    import importlib

    surface = importlib.import_module("pops.surface")
    return surface.NotYetElaborated(construct, later)


def _later(word: str) -> Any:
    def refused(*args: Any, **kwargs: Any) -> Any:
        raise _not_yet("pops.lib.solve.%s" % word, _LATER)
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


__all__ = ["CGS2", "auto", "bicgstab", "dense_lu", "gmres", "jacobi", "newton", "richardson"]
