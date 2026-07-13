"""Time-policy bricks and stable physical component roles.

Local IMEX/source-implicit policies remain descriptors because they select a per-block spatial/time
backend. Global solves are explicit :class:`pops.Program` graphs; reusable integration programs live
under :mod:`pops.lib.time` without introducing physics-specific time presets.
"""
from __future__ import annotations

from pops.runtime._bricks_time_imex import (  # noqa: F401
    IMEX,
    IMEXRK,
    SourceImplicit,
    SourceImplicitBE,
    _norm_implicit,
    _role_to_stable,
)


class Role:
    """Stable physical roles shared by descriptors and symbolic Program authoring."""

    Density = "density"
    MomentumX = "momentum_x"
    MomentumY = "momentum_y"
    MomentumZ = "momentum_z"
    Energy = "energy"
    VelocityX = "velocity_x"
    VelocityY = "velocity_y"
    VelocityZ = "velocity_z"
    Pressure = "pressure"
    Temperature = "temperature"
    Scalar = "scalar"


__all__ = [
    "IMEX", "IMEXRK", "SourceImplicit", "SourceImplicitBE", "Role",
    "_norm_implicit", "_role_to_stable",
]
