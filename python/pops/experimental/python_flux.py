"""PythonFlux : host (numpy) prototyping backend for the Flux interface.

NON-PRODUCTION / TESTS-ONLY: this module lives under :mod:`pops.experimental` and is not part
of the stable public API. It computes a numpy residual in Python (the PoPS "no public Python
numeric" rule keeps it off the public surface); use it only for residual prototyping in tests.
"""
from __future__ import annotations

from typing import Any


class PythonFlux:
    """PROTOTYPING backend (host, numpy) for the Flux interface: the user provides the physical
    flux and the wave speed in Python, and PythonFlux assembles the residual -div(F*) by finite
    volumes (Rusanov, order 1, periodic domain) over the whole array at once.

    OUT of the GPU/MPI hot path: this is a pure HOST path (numpy), it NEVER goes through a Kokkos
    kernel. For production (GPU/MPI), declare the physical flux on ``pops.Model`` and select its
    typed numerical realization through ``pops.numerics.FiniteVolume``. PythonFlux formalizes the
    test-only pattern for iterating on a novel flux without recompiling.

    NON-PRODUCTION / TESTS-ONLY: reachable as ``pops.experimental.PythonFlux`` for residual
    prototyping in tests; it is intentionally absent from the public ``pops`` surface.

    flux(U, dir) -> F: U and F are numpy (ncomp, n, n); dir = 0 (x) or 1 (y).
    max_wave_speed(U) -> float: bound for the Rusanov flux and the CFL.
    """

    def __init__(self, flux: Any, max_wave_speed: Any) -> None:
        self.flux = flux
        self.max_wave_speed = max_wave_speed

    def residual(self, U: Any, dx: Any, dy: Any = None) -> Any:
        """-div(F*) by Rusanov flux (order 1, periodic). U numpy (ncomp, n, n); returns dU/dt."""
        import numpy as np
        dy = dx if dy is None else dy
        a = float(self.max_wave_speed(U))
        res = np.zeros_like(U)
        for axis, h, d in ((2, dx, 0), (1, dy, 1)):  # x = axis 2, y = axis 1
            F = self.flux(U, d)
            UR = np.roll(U, -1, axis=axis)
            FR = np.roll(F, -1, axis=axis)
            face = 0.5 * (F + FR) - 0.5 * a * (UR - U)       # flux at the +d face of each cell
            res -= (face - np.roll(face, 1, axis=axis)) / h  # -div: (F_{i+1/2} - F_{i-1/2}) / h
        return res

    def cfl_dt(self, U: Any, h: Any, cfl: float = 0.4) -> float:
        """Stable time step: dt = cfl * h / max_wave_speed(U)."""
        return cfl * h / max(float(self.max_wave_speed(U)), 1e-30)
