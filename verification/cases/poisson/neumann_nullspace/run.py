"""Public 1-d homogeneous-Neumann Poisson authoring for PO-03.

Does not compile, bind, or launch a solver. Native elliptic execution is not
implemented in this worktree (no private solver).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import pops
from pops.domain import CartesianDomain
from pops.fields import (
    CellCenteredSecondOrder,
    ConstantNullspace,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
    MeanValueGauge,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Neumann
from pops.frames import Cartesian1D
from pops.math import laplacian
from pops.physics import Model
from pops.problem import Case
from pops.solvers.elliptic import GeometricMG
from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_exact = load_sibling_module(_CASE_DIR / "exact.py")

COMPATIBLE_RHS_ATOL = 1.0e-12


class AuthoringPending(RuntimeError):
    """Raised only if public elliptic validate/resolve cannot be completed."""


class IncompatibleRhs(ValueError):
    """Homogeneous Neumann Poisson requires a mean-free right-hand side."""


def require_compatible_rhs(rhs, volumes, *, atol: float = COMPATIBLE_RHS_ATOL) -> float:
    """Return the volume-weighted mean, or raise if it is not ~0."""
    field = np.asarray(rhs, dtype=np.float64)
    weights = np.asarray(volumes, dtype=np.float64)
    mean = float(np.average(field, weights=weights))
    if abs(mean) > atol:
        raise IncompatibleRhs(
            "homogeneous Neumann -Δφ=f with φ'=0 at the boundary requires "
            f"∫f=0; volume-weighted rhs mean is {mean}"
        )
    return mean


def incompatible_rhs_observation(
    rhs, volumes, *, atol: float = COMPATIBLE_RHS_ATOL
) -> dict:
    """Document whether rhs is compatible with the constant Neumann nullspace."""
    field = np.asarray(rhs, dtype=np.float64)
    weights = np.asarray(volumes, dtype=np.float64)
    mean = float(np.average(field, weights=weights))
    compatible = abs(mean) <= atol
    if compatible:
        reason = (
            "volume-weighted rhs mean is compatible with the constant "
            "Neumann nullspace"
        )
    else:
        reason = (
            "Homogeneous Neumann -Δφ=f with φ'=0 at the boundary requires "
            "∫f=0 (compatibility with the constant nullspace). A constant "
            f"rhs has mean {mean} and is incompatible."
        )
    return {"compatible": compatible, "mean": mean, "reason": reason}


def build_rhs_and_oracle(n_cells: int):
    """Return in-memory cell-center RHS and exact φ, E on a uniform 1-d grid."""
    centers, volumes = _exact.uniform_cell_grid(n_cells)
    rhs = _exact.rhs_exact(centers)
    require_compatible_rhs(rhs, volumes)
    return {
        "x": centers,
        "volumes": volumes,
        "phi": _exact.phi_exact(centers),
        "rhs": rhs,
        "e": _exact.e_exact(centers),
    }


def build_case(n_cells: int) -> Case:
    """Author a 1-d Neumann Poisson Case: -laplacian(phi) == rhs, GeometricMG."""
    del n_cells
    frame = CartesianDomain("po03-domain", (0.0,), (1.0,)).frame(Cartesian1D())
    model = Model("po03_neumann_nullspace", frame=frame)
    state = model.state("U", components=["rhs"])
    (rhs,) = state
    potential = model.field("potential")
    operator = model.field_operator(
        "poisson",
        unknown=potential,
        equation=(-laplacian(potential) == rhs),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential, sign=-1),
        ),
    )
    case = Case("po03-neumann-nullspace")
    case.block("electrostatic", model)
    case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Neumann(0.0)),),
            solver=GeometricMG(),
            nullspace=ConstantNullspace(),
            gauge=MeanValueGauge(0.0),
        ),
    )
    return case


def resolve_plan(n_cells: int):
    """Validate the public Case. Full resolve is pending a whole-system Program.

    ``pops.validate`` succeeds. ``pops.resolve`` requires a whole-system Program
    for Uniform layouts. This case does not invent a hyperbolic stepper or a
    private elliptic solver, so resolve stays documented as AuthoringPending.
    """
    pops.validate(build_case(n_cells))
    raise AuthoringPending(
        "PO-03 Case validates; resolve needs a whole-system Program "
        "(no invented time stepper or private elliptic solver)"
    )
