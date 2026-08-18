"""Public 1-d periodic Helmholtz authoring for CP-09 Debye screen.

The RHS is the manufactured cosine. ``build_case`` / ``resolve_plan``
author a public periodic screened-Poisson Case
(-laplacian(phi) + λ_D^{-2} phi == f). ``run_native`` is optional and
raises ``NativeUnavailable``. Does not call ROMEO.
"""
from __future__ import annotations

from pathlib import Path

import pops
from pops.domain import CartesianDomain
from pops.fields import (
    CellCenteredSecondOrder,
    FieldDiscretization,
    FieldOutput,
    GradientOutput,
)
from pops.fields.bcs import AllPhysicalBoundaries, BoundaryCondition, Periodic
from pops.frames import Cartesian1D
from pops.math import laplacian, unknown
from pops.physics import Model
from pops.problem import Case
from pops.solvers.elliptic import FFT
from verification.pops_verify.case_authoring import load_sibling_module

_CASE_DIR = Path(__file__).resolve().parent
N_CELLS = 32


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


class AuthoringPending(RuntimeError):
    """Raised only if public elliptic validate/resolve cannot be completed."""


def _exact_module():
    return load_sibling_module(_CASE_DIR / "exact.py")


def build_oracle(n_cells: int = N_CELLS, *, lambda_d=None, k=None):
    """Return in-memory Helmholtz source, φ, and E on a uniform 1-d grid."""
    exact = _exact_module()
    length = exact.LAMBDA_D if lambda_d is None else lambda_d
    mode = exact.K if k is None else k
    centers, volumes = exact.uniform_cell_grid(n_cells)
    return {
        "x": centers,
        "volumes": volumes,
        "f": exact.f_exact(centers, k=mode),
        "phi": exact.phi_exact(centers, lambda_d=length, k=mode),
        "e": exact.e_exact(centers, lambda_d=length, k=mode),
        "applied": exact.apply_helmholtz(
            exact.phi_exact(centers, lambda_d=length, k=mode),
            centers,
            lambda_d=length,
        ),
        "lambda_d": length,
        "k": float(mode),
    }


def build_case(n_cells: int = N_CELLS) -> Case:
    """Author a 1-d periodic Helmholtz Case: -Δφ + λ_D^{-2} φ == f, FFT, periodic."""
    del n_cells
    exact = _exact_module()
    kappa = exact.screening_coefficient()
    frame = CartesianDomain("cp09-domain", (0.0,), (1.0,)).frame(Cartesian1D())
    model = Model("cp09_debye_screen", frame=frame)
    state = model.state("U", components=["rhs"])
    (rhs,) = state
    potential = model.field("potential")
    phi = unknown(potential)
    operator = model.field_operator(
        "helmholtz",
        unknown=potential,
        equation=(-laplacian(phi) + kappa * phi == rhs),
        outputs=(
            FieldOutput("potential", potential),
            GradientOutput("electric_field", potential, sign=-1),
        ),
    )
    case = Case("cp09-debye-screen")
    case.block("electrostatic", model)
    case.field(
        operator,
        FieldDiscretization(
            method=CellCenteredSecondOrder(),
            boundaries=(BoundaryCondition(AllPhysicalBoundaries(), Periodic()),),
            solver=FFT(),
        ),
    )
    return case


def resolve_plan(n_cells: int = N_CELLS):
    """Validate the public Case. Full resolve is pending a whole-system Program."""
    pops.validate(build_case(n_cells))
    raise AuthoringPending(
        "CP-09 Case validates; resolve needs a whole-system Program "
        "(no invented time stepper or private elliptic solver)"
    )


def run_native(n_cells: int = N_CELLS, t_end: float = 0.0):
    """Optional native path. Raises NativeUnavailable without a compiler."""
    from tests.python.support.requirements import missing_compiler_requirement, repo_include

    del n_cells, t_end
    missing = missing_compiler_requirement(repo_include())
    if missing:
        raise NativeUnavailable(missing)
    raise NativeUnavailable("optional native CP-09 run not executed in this worktree")
