"""Private aggregate of descriptors consumed by native-engine internals.

The final Python interface authors physics, numerics and time through their dedicated packages.
Low-level engine adapters use one cycle-safe import point for native model, scheme, time-policy and
boundary descriptor values. This private module provides that point; :mod:`pops.runtime` never
re-exports it.
"""
from __future__ import annotations

from pops._bootstrap import abi_key  # noqa: F401
from pops.runtime._bricks_model import (  # noqa: F401
    BackgroundDensity,
    ChargeDensity,
    ChargeDensitySource,
    CompositeRhs,
    CompressibleFlux,
    DivEpsGrad,
    ElectricFieldFromPotential,
    EllipticModel,
    EllipticSolver,
    ExB,
    FluidState,
    GravityCoupling,
    GravityForce,
    IsothermalFlux,
    MagneticLorentzForce,
    Model,
    NoSource,
    PotentialForce,
    PotentialMagneticForce,
    Scalar,
    charge_density,
    composite_rhs,
    div_eps_grad,
    electric_field_from_potential,
    elliptic,
)
from pops.runtime._bricks_scheme import (  # noqa: F401
    Collision,
    Explicit,
    Ionization,
    Spatial,
    ThermalExchange,
)
from pops.runtime._bricks_time import (  # noqa: F401
    IMEX,
    IMEXRK,
    Role,
    SourceImplicit,
    SourceImplicitBE,
    _norm_implicit,
    _role_to_stable,
)
from pops.runtime._bricks_typed import Dirichlet, Neumann, Periodic  # noqa: F401
