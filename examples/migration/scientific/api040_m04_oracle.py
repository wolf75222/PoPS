"""Pure cell-mean oracle and Fourier/stencil checks for M04/W02."""
from __future__ import annotations

import numpy as np


def exact_cell_means(n: int, t: float, *, velocity: float = 1.,
                     diffusivity: float = .01, amplitude: float = .2) -> np.ndarray:
    """Integral mean of 1+A exp(-4π²Dt) cos(2π(x-at)) on each periodic cell."""
    if n <= 0 or t < 0 or diffusivity < 0:
        raise ValueError("n > 0, t >= 0, diffusivity >= 0 required")
    centers = (np.arange(n, dtype=float) + .5) / n
    return (1. + amplitude * np.sinc(1 / n) *
            np.exp(-4 * np.pi**2 * diffusivity * t) *
            np.cos(2 * np.pi * (centers - velocity * t)))


def frequencies(n: int, *, velocity: float = 1., diffusivity: float = .01,
                transverse_diffusivity: float | None = None) -> dict[str, float]:
    """Separate physical 1D and native diagonal-2D stability frequencies."""
    if transverse_diffusivity is None:
        transverse_diffusivity = diffusivity
    if n <= 0 or not all(np.isfinite(v) for v in
                         (velocity, diffusivity, transverse_diffusivity)) or min(
                             diffusivity, transverse_diffusivity) < 0:
        raise ValueError("finite velocity, n > 0 and nonnegative diffusivities required")
    advective = abs(velocity) * n
    diffusion_x = 2 * diffusivity * n * n
    diffusion_y = 2 * transverse_diffusivity * n * n
    return {"advection_x": advective, "diffusion_x": diffusion_x,
            "diffusion_y": diffusion_y, "physical_1d": advective + diffusion_x,
            "installed_2d": advective + diffusion_x + diffusion_y}


def impulse_weights_1d(c: float, r: float) -> dict[str, float]:
    """Forward-Euler upwind/central weights for c=a dt/h, r=D dt/h²."""
    return {"upstream": c + r, "center": 1 - c - 2 * r, "downstream": r}


def impulse_weights_2d(c: float, r: float) -> dict[str, float]:
    """Same x advection with isotropic x/y diffusion, for a 2D impulse."""
    return {"x_upstream": c + r, "center": 1 - c - 4 * r,
            "x_downstream": r, "y_lower": r, "y_upper": r}
