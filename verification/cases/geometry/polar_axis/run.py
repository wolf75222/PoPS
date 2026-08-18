"""GE-05 in-memory polar volumes. No compile, bind, or pops.run.

The in-memory oracle is the exact midpoint cell area r Δr Δθ. The public
polar System is not active; ``refuse_public_polar_runtime`` returns the
documented reason. ``run_native`` raises ``NativeUnavailable`` with that
same string. There is no public PolarMesh runtime.
"""
from __future__ import annotations

from pathlib import Path

from verification.pops_verify.case_authoring import load_sibling_module

_exact = load_sibling_module(Path(__file__).with_name("exact.py"))

R_IN = float(_exact.R_IN)
R_OUT = float(_exact.R_OUT)
N_R = int(_exact.N_R)
N_THETA = int(_exact.N_THETA)
POLAR_RUNTIME_REFUSAL = "public polar System not active"


class NativeUnavailable(RuntimeError):
    """Native compile/run is not available in this environment."""


def annulus_volume(
    r_in: float = R_IN,
    r_out: float = R_OUT,
    n_r: int = N_R,
    n_theta: int = N_THETA,
) -> float:
    """Return the discrete polar annulus volume via the sibling oracle."""
    return _exact.annulus_volume(r_in, r_out, n_r, n_theta)


def constant_state_integral(
    value=1.0,
    r_in: float = R_IN,
    r_out: float = R_OUT,
    n_r: int = N_R,
    n_theta: int = N_THETA,
) -> float:
    """Integrate a constant state over the polar annulus."""
    return _exact.constant_state_integral(value, r_in, r_out, n_r, n_theta)


def axis_cell_volume(dr, dtheta) -> float:
    """Return the documented regular axis-cell volume ½ (Δr)² Δθ."""
    return _exact.axis_cell_volume(dr, dtheta)


def refuse_public_polar_runtime() -> str:
    """Return the documented reason the public polar System is not active."""
    return POLAR_RUNTIME_REFUSAL


def run_native(n_r: int = N_R, n_theta: int = N_THETA):
    """Refuse native polar compile. Volumes stay on the in-memory oracle."""
    del n_r, n_theta
    raise NativeUnavailable(refuse_public_polar_runtime())
