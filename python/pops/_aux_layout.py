"""Generic auxiliary-component layout derived from one resolved provider pack.

There is deliberately no process-wide prefix for potential, gradients, magnetic
fields, or temperatures.  An auxiliary channel is a compact ordered projection
of the exact :class:`pops.model.ProviderPack` selected for one compiled module.
The provider pack owns the qualified field identity and its slot; this small
value is only the detached lookup view consumed by authoring diagnostics.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


def require_aux_name(value: Any, *, where: str = "aux") -> str:
    """Return one valid local auxiliary identifier.

    Formula variables become C++ locals in generated kernels, so accepting a
    non-identifier here would defer a user error to a compiler diagnostic.
    """
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(
            "%s name must be a valid identifier (letters/digits/_, without a leading digit)"
            % where
        )
    return value


@dataclass(frozen=True, slots=True)
class AuxLayout:
    """Immutable compact ``name -> slot`` projection of one provider authority."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(require_aux_name(name, where="aux layout") for name in self.names)
        if len(set(names)) != len(names):
            raise ValueError("aux layout names must be unique")
        object.__setattr__(self, "names", names)

    @property
    def components(self) -> Mapping[str, int]:
        return MappingProxyType({name: slot for slot, name in enumerate(self.names)})

    @property
    def n_components(self) -> int:
        return len(self.names)

    def component_index(self, name: Any) -> int:
        checked = require_aux_name(name, where="aux field")
        try:
            return self.components[checked]
        except KeyError:
            raise ValueError("aux field %r is absent from this provider layout" % checked) from None


def aux_layout(names: Iterable[Any]) -> AuxLayout:
    """Build a compact detached layout from one ordered declaration sequence."""
    try:
        return AuxLayout(tuple(names))
    except TypeError:
        raise TypeError("aux layout names must be an iterable of identifiers") from None


def aux_component_index(name: Any, names: Iterable[Any]) -> int:
    """Look up ``name`` in one explicit ordered declaration sequence."""
    return aux_layout(names).component_index(name)


def aux_total_n_aux(names: Iterable[Any]) -> int:
    """Return the compact storage width of one explicit declaration sequence."""
    return aux_layout(names).n_components


__all__ = [
    "AuxLayout",
    "aux_component_index",
    "aux_layout",
    "aux_total_n_aux",
    "require_aux_name",
]
