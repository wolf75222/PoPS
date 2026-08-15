"""Public physical-role conveniences backed by the neutral model identity authority.

This module intentionally contains no parallel role implementation.  Existing imports remain
compatible while runtime and compiler layers can depend directly on :mod:`pops.model.identity`.
"""
from pops.model.identity import (
    Axial,
    ComponentRole,
    CouplingBlockContract,
    Custom,
    Density,
    Energy,
    Momentum,
    Pressure,
    RoleKey,
    Scalar,
    StateSchema,
    Temperature,
    Velocity,
    native_role_token,
    parse_role,
)

__all__ = [
    "Axial", "ComponentRole", "CouplingBlockContract", "Custom", "Density", "Energy", "Momentum",
    "Pressure", "Scalar", "Temperature", "Velocity", "RoleKey", "StateSchema",
    "native_role_token", "parse_role",
]
