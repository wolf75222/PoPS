"""Legacy board-shortcut lowering into typed elliptic solver descriptors."""
from __future__ import annotations

from typing import Any

from pops.descriptors import reject_string_selector


_LEGACY_SOLVER_TOKENS = {
    "geometric_mg": "GeometricMG",
    "fft": "FFT",
    "fft_spectral": "FFT",
}


def lower_field_solver(solver: Any) -> Any:
    """Turn a recognized board token into its typed solver before FieldProblem sees it."""
    if not isinstance(solver, str):
        return solver
    factory = _LEGACY_SOLVER_TOKENS.get(solver)
    if factory is None:
        reject_string_selector(
            solver,
            "solver",
            "pops.solvers.GeometricMG() / pops.solvers.FFT()",
        )
    from pops.solvers.elliptic import FFT, GeometricMG

    if factory == "FFT":
        return FFT(spectral=solver == "fft_spectral")
    return GeometricMG()


__all__ = ["lower_field_solver"]
