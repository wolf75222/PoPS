"""pops.fields.aux -- field-side aux declarations (Spec 5 sec.14.2.4).

A field problem can declare auxiliary fields it reads or derives: a :class:`StaticAux`
holds a fixed value supplied once, a :class:`DerivedAux` is computed from a PoPS
expression (in C++, not Python). Per-field halo policy belongs uniquely to
:class:`pops.mesh.AuxHalo`.

Inert descriptors; they compute nothing.
"""

from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor

class StaticAux(Descriptor):
    """A static auxiliary field named :paramref:`name`, holding a fixed :paramref:`value`."""

    category = "aux"

    def __init__(self, name: Any, value: Any = None) -> None:
        self._name = str(name)
        self.value = value

    @property
    def name(self) -> str:
        return self._name

    def options(self) -> dict:
        return {"name": self._name, "kind": "static", "value": self.value}


class DerivedAux(Descriptor):
    """An auxiliary field named :paramref:`name`, derived from a PoPS :paramref:`expression`.

    The expression is stored verbatim and evaluated in C++ by the runtime, not in Python.
    """

    category = "aux"

    def __init__(self, name: Any, expression: Any = None) -> None:
        self._name = str(name)
        self.expression = expression

    @property
    def name(self) -> str:
        return self._name

    def options(self) -> dict:
        return {
            "name": self._name,
            "kind": "derived",
            "expression": getattr(self.expression, "name", repr(self.expression))
            if self.expression is not None
            else None,
        }


__all__ = ["StaticAux", "DerivedAux"]
