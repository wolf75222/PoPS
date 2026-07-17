"""Vectorized host oracle used only to validate authored physical flux formulas."""
from __future__ import annotations

from typing import Any


class RusanovFiniteVolumeOracle:
    """Small NumPy reference, intentionally outside the installed :mod:`pops` package."""

    def __init__(self, flux: Any, max_wave_speed: Any) -> None:
        self._flux = flux
        self._max_wave_speed = max_wave_speed

    def residual(self, state: Any, dx: Any, dy: Any = None) -> Any:
        """Return the periodic first-order Rusanov residual for a test array."""
        import numpy as np

        dy = dx if dy is None else dy
        speed = float(self._max_wave_speed(state))
        residual = np.zeros_like(state)
        for axis, spacing, direction in ((2, dx, 0), (1, dy, 1)):
            flux = self._flux(state, direction)
            right_state = np.roll(state, -1, axis=axis)
            right_flux = np.roll(flux, -1, axis=axis)
            face = 0.5 * (flux + right_flux) - 0.5 * speed * (right_state - state)
            residual -= (face - np.roll(face, 1, axis=axis)) / spacing
        return residual

    def cfl_dt(self, state: Any, spacing: Any, cfl: float = 0.4) -> float:
        """Return the oracle's stable explicit step."""
        return cfl * spacing / max(float(self._max_wave_speed(state)), 1.0e-30)


def finite_volume_oracle(model: Any, aux: Any = None) -> RusanovFiniteVolumeOracle:
    """Adapt an interpreted authoring model to the tests-only Rusanov oracle."""
    fields = aux or {}
    return RusanovFiniteVolumeOracle(
        lambda state, direction: model.flux(state, fields, direction),
        lambda state: max(
            model.max_wave_speed(state, fields, 0),
            model.max_wave_speed(state, fields, 1),
        ),
    )
