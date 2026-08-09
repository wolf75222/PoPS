"""Canonical Cartesian-axis maps shared by authoring and lowering.

The number of entries is the sole rank authority. Consumers iterate the returned map in canonical
axis order; no physics API selects separate one-, two-, or three-dimensional algorithms.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


CARTESIAN_AXIS_NAMES = ("x", "y", "z")


def canonical_axis_mapping(values: Any, *, where: str) -> dict[str, Any]:
    """Return one exact non-empty x[/y[/z]] mapping in canonical order."""
    if not isinstance(values, Mapping):
        raise TypeError("%s must map canonical Cartesian axis names" % where)
    keys = set(values)
    if any(not isinstance(name, str) for name in keys):
        raise TypeError("%s axis names must be strings" % where)
    expected = set(CARTESIAN_AXIS_NAMES[:len(keys)])
    if not keys or keys != expected:
        raise ValueError(
            "%s must define one canonical Cartesian axis prefix; got %s"
            % (where, sorted(keys))
        )
    return {name: values[name] for name in CARTESIAN_AXIS_NAMES if name in values}


def axis_name(direction: Any, names: Any, *, where: str) -> str:
    """Authenticate an integer ordinal or canonical name against one ranked axis tuple."""
    axes = tuple(names)
    if axes != CARTESIAN_AXIS_NAMES[:len(axes)] or not axes:
        raise ValueError("%s received a non-canonical axis authority" % where)
    if isinstance(direction, bool):
        raise TypeError("%s direction must be an axis ordinal or name" % where)
    if isinstance(direction, int):
        if direction not in range(len(axes)):
            raise ValueError("%s direction %r is outside the ranked axis set" % (where, direction))
        return axes[direction]
    if isinstance(direction, str):
        name = direction.lower()
        if name in axes:
            return name
    raise ValueError("%s direction %r is not one of %s" % (where, direction, axes))


def flattened_axis_values(values: Mapping[str, Any]) -> list[Any]:
    """Flatten a previously authenticated map without assuming its rank."""
    return [item for name in values for item in values[name]]


__all__ = ["CARTESIAN_AXIS_NAMES", "axis_name", "canonical_axis_mapping",
           "flattened_axis_values"]
