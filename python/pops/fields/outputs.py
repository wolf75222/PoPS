"""pops.fields.outputs -- typed outputs of an elliptic field solve (Spec 5 sec.5.5 / sec.9).

A field solve produces the unknown itself and a set of DERIVED quantities computed from it (the
electric field ``E = -grad(phi)``, a stress / current, a diagnostic aux). Spec 5 makes each such
output a TYPED descriptor rather than a free ``{"E": "grad_phi"}`` string map, so a
:class:`FieldProblem` declares exactly which fields it exposes and how each is derived, before the
runtime materialises them.

* :class:`FieldOutput` -- the solved field itself surfaced under a name (``phi``);
* :class:`GradientOutput` -- the (negative) gradient of the solved field (``E = -grad(phi)``);
* :class:`DerivedField` -- an arbitrary field derived from the solve by a named recipe.

Inert descriptors: they record the name + recipe and answer the protocol; the runtime computes the
actual arrays. This is the typed counterpart of the flat ``physics.board_handles.FieldOutputs``
string map (which the facade ``solve_field`` shortcut still accepts).
"""
from __future__ import annotations

from typing import Any

from pops.descriptors import Descriptor
from pops.descriptors_report import CapabilitySet, RequirementSet

from ._references import reference_label, resolve_handle


class _Output(Descriptor):
    """Base of the field-output descriptors: a named quantity with a derivation recipe."""

    category = "field_output"
    #: The lowered recipe token the runtime reads (``"field"`` / ``"grad_phi"`` / ``"derived"``).
    recipe = "field"

    def __init__(self, name: Any, source: Any = None) -> None:
        if not isinstance(name, str) or not name:
            raise TypeError("field output name must be a non-empty string")
        if source is not None:
            from pops.model import Handle
            if not isinstance(source, Handle):
                raise TypeError(
                    "field output source must be a declaration Handle; names/strings are not "
                    "references")
        self._name = name
        self.source = source

    @property
    def name(self) -> str:
        return self._name

    def options(self) -> dict:
        return {"name": self._name, "recipe": self.recipe,
                "source": (reference_label(self.source, where="field output source")
                           if self.source is not None else None)}

    def requirements(self) -> Any:
        return RequirementSet({} if self.source is None else {
            "field": reference_label(self.source, where="field output source")})

    def resolve_references(self, resolver: Any) -> _Output:
        from copy import copy

        resolved = copy(self)
        if self.source is not None:
            resolved.source = resolve_handle(
                self.source, resolver, where="field output source")
        return resolved

    def declaration_references(self) -> tuple[Any, ...]:
        return () if self.source is None else (self.source,)


class FieldOutput(_Output):
    """Expose the solved field itself under :paramref:`name` (``FieldOutput("phi")``)."""

    recipe = "field"


class GradientOutput(_Output):
    """The negative gradient of the solved field: ``E = -grad(phi)`` (``GradientOutput("E", "phi")``).

    :paramref:`source` names the solved field to differentiate (the unknown of the field problem);
    :paramref:`name` is the exposed vector field. The runtime computes the centred gradient.
    """

    recipe = "grad_phi"

    def __init__(self, name: Any, source: Any) -> None:
        super().__init__(name, source=source)

    def capabilities(self) -> Any:
        return CapabilitySet({"vector": True, "derivative_order": 1})


class DerivedField(_Output):
    """A field derived from the solve by a named :paramref:`recipe` (``DerivedField("J", "ohm")``).

    Names an output the runtime computes with the given recipe from the solved field / declared
    inputs; the recipe token is carried inert (the numeric kernel is selected by the runtime).
    """

    def __init__(self, name: Any, recipe: Any, source: Any = None) -> None:
        super().__init__(name, source=source)
        self.recipe = str(recipe)

    def capabilities(self) -> Any:
        return CapabilitySet({"derived": True})


__all__ = ["FieldOutput", "GradientOutput", "DerivedField"]
