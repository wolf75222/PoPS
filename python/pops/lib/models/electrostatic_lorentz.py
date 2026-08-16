"""Exact-ranked Cartesian electrostatic-Lorentz source linearization.

The helper authors the local rotation generator on the momentum subspace,

``J[a,b] = sum_c epsilon[a,b,c] B[c]``.

Momentum indices come from the model's structured ``StateSchema`` and the magnetic input is one
explicit three-component provider vector.  There are no ``mx/my`` slots, reserved ``B_z`` name, or
planar runtime path.  In one dimension the projected generator is exactly zero, in two dimensions
only the axial component contributes, and in three dimensions the full cross-product matrix is
materialized.  Geometry-specific bases remain outside this Cartesian authoring helper.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pops.physics.aux import roles_for
from pops.physics.roles import StateSchema


LORENTZ_J_NAME = "electrostatic_lorentz_J"


def _levi_civita(first: int, second: int, third: int) -> int:
    if first == second or second == third or first == third:
        return 0
    return 1 if (first, second, third) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1


def _model_dimension(model: Any, explicit: Any) -> int:
    frame = getattr(model, "_frame", None)
    axes = None if frame is None else getattr(frame, "axes", None)
    frame_dimension = len(axes) if isinstance(axes, tuple) else None
    if explicit is None:
        if frame_dimension not in (1, 2, 3):
            raise TypeError(
                "author_electrostatic_lorentz requires an explicit dimension or typed frame")
        return frame_dimension
    if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit not in (1, 2, 3):
        raise ValueError("author_electrostatic_lorentz dimension must be 1, 2, or 3")
    if frame_dimension is not None and frame_dimension != explicit:
        raise ValueError(
            "author_electrostatic_lorentz dimension %d differs from frame dimension %d"
            % (explicit, frame_dimension))
    return explicit


def _schema(model: Any, dimension: int) -> StateSchema:
    authored = getattr(model, "_m", None)
    if authored is None:
        authored = getattr(getattr(model, "_dsl", None), "_m", model)
    names = getattr(authored, "cons_names", None)
    if names is None:
        names = getattr(authored, "_cons_names", None)
    if names is None:
        raise AttributeError(
            "author_electrostatic_lorentz cannot resolve the model conservative state")
    tokens = tuple(roles_for(names, getattr(authored, "cons_roles", None)))
    return StateSchema.resolve(
        tokens, dimension=dimension, where="electrostatic Lorentz state")


def author_electrostatic_lorentz(
    model: Any,
    *,
    magnetic_components: Sequence[str],
    dimension: int | None = None,
    name: str = LORENTZ_J_NAME,
) -> Any:
    """Author the exact Cartesian Lorentz generator on ``model``.

    ``magnetic_components`` must name the three provider components in Cartesian x/y/z order.
    Conservative momentum indices are resolved by ``momentum:<axis>`` roles independently of state
    ordering. ``dimension`` is required when the model has no typed frame and must equal a present
    frame's exact rank otherwise. The returned handle is produced by the model's generic
    local-linear-operator API.
    """
    if isinstance(magnetic_components, (str, bytes)):
        raise TypeError("magnetic_components must be an ordered three-component provider vector")
    components = tuple(magnetic_components)
    if (len(components) != 3 or any(not isinstance(item, str) or not item for item in components) or
            len(set(components)) != 3):
        raise ValueError(
            "magnetic_components must contain three unique non-empty Cartesian component names")

    schema = _schema(model, _model_dimension(model, dimension))
    magnetic = tuple(model.aux(component) for component in components)
    n_cons = len(schema.roles)
    momentum = tuple(schema.index("momentum:%d" % axis) for axis in range(schema.dimension))
    matrix: list[list[Any]] = [[0.0] * n_cons for _ in range(n_cons)]
    for row_axis, row in enumerate(momentum):
        for column_axis, column in enumerate(momentum):
            coefficient: Any = None
            for magnetic_axis in range(3):
                sign = _levi_civita(row_axis, column_axis, magnetic_axis)
                if sign == 0:
                    continue
                term = magnetic[magnetic_axis] if sign > 0 else -magnetic[magnetic_axis]
                coefficient = term if coefficient is None else coefficient + term
            matrix[row][column] = 0.0 if coefficient is None else coefficient

    local_operator = getattr(model, "local_linear_operator", None)
    if callable(local_operator):
        return local_operator(name, matrix=matrix)
    local_map = getattr(model, "local_linear_map", None)
    if callable(local_map):
        return local_map(name, matrix)
    raise TypeError(
        "author_electrostatic_lorentz requires a generic local-linear-operator authoring API")


__all__ = ["LORENTZ_J_NAME", "author_electrostatic_lorentz"]
