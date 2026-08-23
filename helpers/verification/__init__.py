"""Numerical post-processing shared by the verification benchmarks."""

from .sine_wave import (
    convergence_orders,
    direction_velocity,
    sine_diagnostics,
    sine_wave_cell_averages,
    weighted_error_norms,
)

__all__ = [
    "convergence_orders",
    "direction_velocity",
    "sine_diagnostics",
    "sine_wave_cell_averages",
    "weighted_error_norms",
]
