"""Frozen role and bytecode vocabulary for generic coupled sources."""
from __future__ import annotations

from typing import Any


ROLE_TO_CANONICAL = {
    "Density": "density",
    "MomentumX": "momentum_x",
    "MomentumY": "momentum_y",
    "MomentumZ": "momentum_z",
    "Energy": "energy",
    "VelocityX": "velocity_x",
    "VelocityY": "velocity_y",
    "VelocityZ": "velocity_z",
    "Pressure": "pressure",
    "Temperature": "temperature",
    "Scalar": "scalar",
}


def role_canonical(role: Any) -> Any:
    """Canonical lowercase role accepted by the native boundary."""
    if role in ROLE_TO_CANONICAL:
        return ROLE_TO_CANONICAL[role]
    if role in ROLE_TO_CANONICAL.values():
        return role
    raise ValueError(
        "CoupledSource: unknown role %r (roles: %s)"
        % (role, ", ".join(sorted(ROLE_TO_CANONICAL))))


# Mirror of pops::CsOp and coupled_source_program.hpp capacities.
CS_PUSHREG = 0
CS_ADD = 1
CS_SUB = 2
CS_MUL = 3
CS_DIV = 4
CS_NEG = 5
CS_POW = 6
CS_SQRT = 7

CS_MAX_REG = 32
CS_MAX_TERMS = 16
CS_MAX_PROG = 256


__all__ = [
    "CS_ADD",
    "CS_DIV",
    "CS_MAX_PROG",
    "CS_MAX_REG",
    "CS_MAX_TERMS",
    "CS_MUL",
    "CS_NEG",
    "CS_POW",
    "CS_PUSHREG",
    "CS_SQRT",
    "CS_SUB",
    "role_canonical",
]
