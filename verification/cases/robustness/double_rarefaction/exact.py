"""1-d double rarefaction / near-vacuum exact Riemann oracle.

Left (rho, u, p) = (1, -2, 0.4), right (1, 2, 0.4), gamma=1.4.
Diaphragm at x0=0.5 on the unit interval. Evaluation time t=0.15.

Star states and wave positions follow Toro, *Riemann Solvers and
Numerical Methods for Fluid Dynamics*, 3rd ed., Ch. 4:

    f_K(p) = (p-p_K) sqrt(A_K/(p+B_K))                 shock (p > p_K)
    f_K(p) = (2 c_K/(gamma-1)) ((p/p_K)^g1 - 1)        rarefaction
    f(p*)  = f_L(p*) + f_R(p*) + (u_R - u_L) = 0
    u*     = 1/2 (u_L+u_R) + 1/2 (f_R(p*) - f_L(p*))

Both sides rarefy into a low-density star (Toro Test 2 / 123 problem).
Does not import pops or read a PoPS output.
"""
from __future__ import annotations

import numpy as np

GAMMA = 1.4
RHO_L = 1.0
U_L = -2.0
P_L = 0.4
RHO_R = 1.0
U_R = 2.0
P_R = 0.4
X0 = 0.5
T_END = 0.15

_G1 = (GAMMA - 1.0) / (2.0 * GAMMA)
_G2 = (GAMMA + 1.0) / (2.0 * GAMMA)
_G3 = 2.0 * GAMMA / (GAMMA - 1.0)
_G4 = 2.0 / (GAMMA - 1.0)
_G5 = 2.0 / (GAMMA + 1.0)
_G6 = (GAMMA - 1.0) / (GAMMA + 1.0)
_G7 = (GAMMA - 1.0) / 2.0
_STAR_TOL = 1.0e-12
_STAR_ITERS = 50


def _as_samples(values) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=np.float64))


def _sound(density: float, pressure: float) -> float:
    return float(np.sqrt(GAMMA * pressure / density))


def _pressure_function(p_star: float, density: float, pressure: float, sound: float):
    if p_star > pressure:
        area = 2.0 / ((GAMMA + 1.0) * density)
        offset = _G6 * pressure
        root = np.sqrt(area / (p_star + offset))
        value = (p_star - pressure) * root
        deriv = (1.0 - 0.5 * (p_star - pressure) / (p_star + offset)) * root
        return float(value), float(deriv)
    ratio = p_star / pressure
    value = _G4 * sound * (ratio**_G1 - 1.0)
    deriv = (ratio ** (-_G2)) / (density * sound)
    return float(value), float(deriv)


def star_states() -> dict:
    """Exact star pressure, velocity, densities, and wave speeds.

    Newton iteration on Toro (4.5) with the two-rarefaction start (4.46).
    """
    sound_left = _sound(RHO_L, P_L)
    sound_right = _sound(RHO_R, P_R)
    guess = (
        (sound_left + sound_right - _G7 * (U_R - U_L))
        / (sound_left / P_L**_G1 + sound_right / P_R**_G1)
    ) ** _G3
    pressure = max(float(guess), _STAR_TOL)
    for _ in range(_STAR_ITERS):
        left, d_left = _pressure_function(pressure, RHO_L, P_L, sound_left)
        right, d_right = _pressure_function(pressure, RHO_R, P_R, sound_right)
        residual = left + right + (U_R - U_L)
        step = residual / (d_left + d_right)
        updated = max(pressure - step, _STAR_TOL)
        if abs(updated - pressure) <= _STAR_TOL * 0.5 * (updated + pressure):
            pressure = updated
            break
        pressure = updated
    left, _ = _pressure_function(pressure, RHO_L, P_L, sound_left)
    right, _ = _pressure_function(pressure, RHO_R, P_R, sound_right)
    velocity = 0.5 * (U_L + U_R) + 0.5 * (right - left)
    if pressure > P_L:
        density_left = RHO_L * (pressure / P_L * _G2 + _G1) / (pressure / P_L * _G1 + _G2)
        speed_left = U_L - sound_left * np.sqrt(_G2 * pressure / P_L + _G1)
        left_wave = "shock"
    else:
        density_left = RHO_L * (pressure / P_L) ** (1.0 / GAMMA)
        speed_left = None
        left_wave = "rarefaction"
    if pressure > P_R:
        density_right = RHO_R * (pressure / P_R * _G2 + _G1) / (pressure / P_R * _G1 + _G2)
        speed_right = U_R + sound_right * np.sqrt(_G2 * pressure / P_R + _G1)
        right_wave = "shock"
    else:
        density_right = RHO_R * (pressure / P_R) ** (1.0 / GAMMA)
        speed_right = None
        right_wave = "rarefaction"
    sound_star_left = _sound(density_left, pressure)
    sound_star_right = _sound(density_right, pressure)
    return {
        "p_star": float(pressure),
        "u_star": float(velocity),
        "rho_star_left": float(density_left),
        "rho_star_right": float(density_right),
        "c_left": float(sound_left),
        "c_right": float(sound_right),
        "c_star_left": float(sound_star_left),
        "c_star_right": float(sound_star_right),
        "left_wave": left_wave,
        "right_wave": right_wave,
        "shock_speed": None if speed_right is None else float(speed_right),
        "left_shock_speed": None if speed_left is None else float(speed_left),
        "rarefaction_head": float(U_L - sound_left),
        "rarefaction_tail": float(velocity - sound_star_left),
        "right_rarefaction_head": float(U_R + sound_right),
        "right_rarefaction_tail": float(velocity + sound_star_right),
    }


def wave_positions(t=T_END) -> dict:
    """Left/right rarefaction head/tail and contact positions at time t."""
    time = float(t)
    star = star_states()
    return {
        "rarefaction_head": X0 + star["rarefaction_head"] * time,
        "rarefaction_tail": X0 + star["rarefaction_tail"] * time,
        "contact": X0 + star["u_star"] * time,
        "right_rarefaction_tail": X0 + star["right_rarefaction_tail"] * time,
        "right_rarefaction_head": X0 + star["right_rarefaction_head"] * time,
        "shock": None if star["shock_speed"] is None else X0 + star["shock_speed"] * time,
    }


def _sample_self_similar(speed: float, star: dict) -> tuple[float, float, float]:
    if speed <= star["u_star"]:
        if star["left_wave"] == "shock":
            if speed <= star["left_shock_speed"]:
                return RHO_L, U_L, P_L
            return star["rho_star_left"], star["u_star"], star["p_star"]
        if speed <= star["rarefaction_head"]:
            return RHO_L, U_L, P_L
        if speed > star["rarefaction_tail"]:
            return star["rho_star_left"], star["u_star"], star["p_star"]
        density = RHO_L * (_G5 + _G6 / star["c_left"] * (U_L - speed)) ** _G4
        velocity = _G5 * (star["c_left"] + _G7 * U_L + speed)
        pressure = P_L * (_G5 + _G6 / star["c_left"] * (U_L - speed)) ** _G3
        return float(density), float(velocity), float(pressure)
    if star["right_wave"] == "shock":
        if speed >= star["shock_speed"]:
            return RHO_R, U_R, P_R
        return star["rho_star_right"], star["u_star"], star["p_star"]
    head = U_R + star["c_right"]
    tail = star["u_star"] + star["c_star_right"]
    if speed >= head:
        return RHO_R, U_R, P_R
    if speed <= tail:
        return star["rho_star_right"], star["u_star"], star["p_star"]
    density = RHO_R * (_G5 - _G6 / star["c_right"] * (U_R - speed)) ** _G4
    velocity = _G5 * (-star["c_right"] + _G7 * U_R + speed)
    pressure = P_R * (_G5 - _G6 / star["c_right"] * (U_R - speed)) ** _G3
    return float(density), float(velocity), float(pressure)


def primitives_1d(x, t) -> np.ndarray:
    """Exact double-rarefaction primitives W=(rho, u, p). Shape (3, n)."""
    samples = _as_samples(x)
    time = float(t)
    density = np.empty(samples.shape, dtype=np.float64)
    velocity = np.empty(samples.shape, dtype=np.float64)
    pressure = np.empty(samples.shape, dtype=np.float64)
    if time <= 0.0:
        left = samples < X0
        density[left] = RHO_L
        velocity[left] = U_L
        pressure[left] = P_L
        density[~left] = RHO_R
        velocity[~left] = U_R
        pressure[~left] = P_R
        return np.stack((density, velocity, pressure))
    star = star_states()
    for index, point in enumerate(samples):
        density[index], velocity[index], pressure[index] = _sample_self_similar(
            (float(point) - X0) / time, star
        )
    return np.stack((density, velocity, pressure))


def primitives_to_conserved_1d(primitives) -> np.ndarray:
    """Convert primitive (rho, u, p) to conserved (rho, rho u, E)."""
    density, velocity, pressure = np.asarray(primitives, dtype=np.float64)
    energy = pressure / (GAMMA - 1.0) + 0.5 * density * velocity * velocity
    return np.stack((density, density * velocity, energy))


def conserved_1d(x, t) -> np.ndarray:
    """Exact conserved U=(rho, rho u, E). Shape (3, n)."""
    return primitives_to_conserved_1d(primitives_1d(x, t))
