"""Exact compile-time-rank auxiliary-channel layout.

This module is deliberately below ``pops.physics`` and ``pops.codegen`` so
authoring, code generation, runtime installation, and artifact inspection use
one dimension-qualified authority without importing each other.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

_CARTESIAN_AXES = ("x", "y", "z")
AUX_CANONICAL_NAMES = frozenset(
    {"phi", "grad_x", "grad_y", "grad_z", "B_z", "T_e"}
)
AUX_NAMED_MAX = 4


def _exact_dimension(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("aux layout dimension must be an exact integer")
    if value not in (1, 2, 3):
        raise ValueError("aux layout dimension must be 1, 2, or 3")
    return value


@dataclass(frozen=True, slots=True)
class AuxLayout:
    """One immutable mirror of ``pops::AuxComponentLayout<Dim>``."""

    dimension: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", _exact_dimension(self.dimension))

    @property
    def axes(self) -> tuple[str, ...]:
        return _CARTESIAN_AXES[: self.dimension]

    @property
    def canonical(self) -> Mapping[str, int]:
        values = {"phi": 0}
        values.update(
            ("grad_" + axis, 1 + index) for index, axis in enumerate(self.axes)
        )
        values["B_z"] = self.b_z_component
        values["T_e"] = self.t_e_component
        return MappingProxyType(values)

    @property
    def base_components(self) -> int:
        return 1 + self.dimension

    @property
    def b_z_component(self) -> int:
        return self.base_components

    @property
    def t_e_component(self) -> int:
        return self.b_z_component + 1

    @property
    def named_base(self) -> int:
        return self.t_e_component + 1

    @property
    def max_components(self) -> int:
        return self.named_base + AUX_NAMED_MAX

    def component_index(self, name: Any, named: Any = ()) -> int:
        canonical = self.canonical
        if name in canonical:
            return canonical[name]
        extra = tuple(named or ())
        if name in extra:
            return self.named_base + extra.index(name)
        if name in AUX_CANONICAL_NAMES:
            raise ValueError(
                "aux field %r is outside the %dD canonical layout %s"
                % (name, self.dimension, tuple(canonical))
            )
        raise ValueError(
            "aux field %r is neither canonical nor present in the model's named aux layout"
            % name
        )

    def required_components(self, canonical_names: Any, named: Any = ()) -> int:
        width = self.base_components
        for name in canonical_names:
            width = max(width, self.component_index(name) + 1)
        extra = tuple(named or ())
        if len(extra) > AUX_NAMED_MAX:
            raise ValueError(
                "aux layout supports at most %d model-named fields" % AUX_NAMED_MAX
            )
        if extra:
            width = max(width, self.named_base + len(extra))
        return width


def aux_layout(dimension: Any) -> AuxLayout:
    """Return the exact immutable auxiliary layout for one native rank."""
    return AuxLayout(_exact_dimension(dimension))


def aux_component_index(
    name: Any, aux_extra_names: Any = (), *, dimension: Any
) -> int:
    return aux_layout(dimension).component_index(name, aux_extra_names)


def aux_total_n_aux(
    aux_names: Any, aux_extra_names: Any, *, dimension: Any
) -> int:
    return aux_layout(dimension).required_components(aux_names, aux_extra_names)
