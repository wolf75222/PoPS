"""Physical-model authoring through the single canonical :class:`Model` facade.

The implementation engines remain in their private modules for lowering existing operator
registries; they are not alternate authoring APIs. Generic composition happens through typed
operators and small protocols returned by ``Model``.
"""

from .board import Model
from .roles import (
    ComponentRole,
    Density,
    Energy,
    Momentum,
    Pressure,
    Scalar,
    Temperature,
    Velocity,
)

__all__ = [
    "Model", "ComponentRole", "Density", "Energy", "Momentum", "Pressure", "Scalar",
    "Temperature", "Velocity",
]
