"""Canonical typed role fixtures for public physics.Model tests."""
from __future__ import annotations

from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.physics import Density, Energy, Momentum


FRAME = Rectangle("role_frame", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D())
X_AXIS, Y_AXIS = FRAME.axes


def planar_fluid_roles(
    density: str,
    momentum_x: str,
    momentum_y: str,
    *,
    energy: str | None = None,
) -> dict[str, object]:
    roles: dict[str, object] = {
        density: Density(),
        momentum_x: Momentum(axis=X_AXIS),
        momentum_y: Momentum(axis=Y_AXIS),
    }
    if energy is not None:
        roles[energy] = Energy()
    return roles


__all__ = ["FRAME", "X_AXIS", "Y_AXIS", "planar_fluid_roles"]
