"""1-d Euler–Poisson manufactured solution (canonical electrostatic sign).

Potential and electron density (ions fixed, q_e = -e):

    phi = A cos(k x - omega t)
    n_e = n_i - (eps0 k^2 / e) phi

Poisson convention (plan §14 CP-01). Do not flip this sign in analyze:

    -eps0 * d^2 phi / dx^2 = e (n_i - n_e)

Equivalently D phi = rho_q with D = -eps0 d^2/dx^2 and rho_q = e(n_i - n_e).
The electric field is E = -d phi / dx.

Defaults: e = 1, eps0 = 1, n_i = 1, A = 1e-3, k = 2 pi, omega = 2 pi.
u_e and p_e are positive nontrivial traveling waves. Gamma-law electrons
use gamma = 5/3. Manufactured hyperbolic sources are S = dU/dt + dF/dx
minus the Lorentz coupling (q_e n E / m_e). Poisson is satisfied exactly;
no manufactured source is added to the elliptic residual.

Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

E_CHARGE = 1.0
EPS0 = 1.0
N_I = 1.0
A = 1.0e-3
K = 2.0 * np.pi
OMEGA = 2.0 * np.pi
U0 = 0.5
U1 = 0.1
P0 = 1.0
P1 = 0.05
GAMMA = 5.0 / 3.0
Q_E = -E_CHARGE
M_E = 1.0


def _as_samples(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def _phase(x, t) -> np.ndarray:
    return K * _as_samples(x) - OMEGA * float(t)


def phi_exact(x, t) -> np.ndarray:
    """Manufactured potential A cos(k x - omega t)."""
    return A * np.cos(_phase(x, t))


def e_exact(x, t) -> np.ndarray:
    """Electric field E = -d phi / dx = A k sin(k x - omega t)."""
    return A * K * np.sin(_phase(x, t))


def n_e_exact(x, t) -> np.ndarray:
    """Electron density from the closed Poisson identity."""
    return N_I - (EPS0 * K**2 / E_CHARGE) * phi_exact(x, t)


def u_e_exact(x, t) -> np.ndarray:
    """Positive nontrivial electron velocity."""
    return U0 + U1 * np.cos(_phase(x, t))


def p_e_exact(x, t) -> np.ndarray:
    """Positive nontrivial electron pressure."""
    return P0 + P1 * np.cos(_phase(x, t))


def fields_1d(x, t) -> dict:
    """Canonical 1-d fields. Density, velocity, and pressure stay positive."""
    return {
        "phi": phi_exact(x, t),
        "E": e_exact(x, t),
        "n_e": n_e_exact(x, t),
        "u_e": u_e_exact(x, t),
        "p_e": p_e_exact(x, t),
    }


def primitives_1d(x, t) -> np.ndarray:
    """Electron primitives W = (n_e, u_e, p_e). Shape (3, n)."""
    fields = fields_1d(x, t)
    return np.stack((fields["n_e"], fields["u_e"], fields["p_e"]))


def primitives_to_conserved_1d(primitives) -> np.ndarray:
    """Convert (n, u, p) to conserved (n, n u, energy)."""
    density, velocity, pressure = np.asarray(primitives, dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack((density, density * velocity, energy))


def conserved_1d(x, t) -> np.ndarray:
    """Conserved U = (n, n u, energy). Shape (3, n)."""
    return primitives_to_conserved_1d(primitives_1d(x, t))


def flux_1d(x, t) -> np.ndarray:
    """Euler flux F = (n u, n u^2 + p, u (energy + p)). Shape (3, n)."""
    density, velocity, pressure = primitives_1d(x, t)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack(
        (
            density * velocity,
            density * velocity * velocity + pressure,
            velocity * (energy + pressure),
        )
    )


def lorentz_source_1d(x, t) -> np.ndarray:
    """Coupling source (0, q_e n E / m_e, q_e n u E / m_e). Shape (3, n)."""
    fields = fields_1d(x, t)
    force = Q_E * fields["n_e"] * fields["E"] / M_E
    return np.stack((np.zeros_like(force), force, force * fields["u_e"]))


def poisson_residual_1d(x, t) -> np.ndarray:
    """Elliptic residual -eps0 phi_xx - e (n_i - n_e). Documented sign, no flip.

    ``n_e`` is defined from that identity, so the manufactured residual is
    the zero field. Analyze must not replace this with the flipped form
    ``+eps0 phi_xx - e(n_i-n_e)`` or ``n_e-n_i``.
    """
    del t
    return np.zeros_like(_as_samples(x))


def _primitive_derivatives_1d(x, t):
    """Analytic (d/dt, d/dx) of the manufactured primitives and field."""
    phase = _phase(x, t)
    sine = np.sin(phase)
    cosine = np.cos(phase)
    amplitude = EPS0 * K**2 / E_CHARGE
    return {
        "n_t": -amplitude * A * OMEGA * sine,
        "n_x": amplitude * A * K * sine,
        "u_t": U1 * OMEGA * sine,
        "u_x": -U1 * K * sine,
        "p_t": P1 * OMEGA * sine,
        "p_x": -P1 * K * sine,
    }


def sources_1d(x, t) -> np.ndarray:
    """Manufactured hyperbolic residual S = dU/dt + dF/dx - Lorentz. Shape (3, n)."""
    density, velocity, pressure = primitives_1d(x, t)
    deriv = _primitive_derivatives_1d(x, t)
    momentum_t = deriv["n_t"] * velocity + density * deriv["u_t"]
    momentum_x = deriv["n_x"] * velocity + density * deriv["u_x"]
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    energy_t = (
        deriv["p_t"] / (GAMMA - 1.0)
        + 0.5 * deriv["n_t"] * velocity * velocity
        + density * velocity * deriv["u_t"]
    )
    energy_x = (
        deriv["p_x"] / (GAMMA - 1.0)
        + 0.5 * deriv["n_x"] * velocity * velocity
        + density * velocity * deriv["u_x"]
    )
    flux_momentum_x = (
        deriv["n_x"] * velocity * velocity
        + 2.0 * density * velocity * deriv["u_x"]
        + deriv["p_x"]
    )
    flux_energy_x = deriv["u_x"] * (energy + pressure) + velocity * (
        energy_x + deriv["p_x"]
    )
    lorentz = lorentz_source_1d(x, t)
    return np.stack(
        (
            deriv["n_t"] + momentum_x,
            momentum_t + flux_momentum_x,
            energy_t + flux_energy_x,
        )
    ) - lorentz
