"""Independent SSPRK2/Rusanov boundary quadrature for the M03 Outflow problem.

This NumPy oracle consumes saved accepted states, never the native residual or
the change in total state. It includes the periodic transverse divergence when
predicting the first stage at each physical boundary. It is not an evolution
backend and is not imported by production PoPS.
"""
from __future__ import annotations

import numpy as np


def physical_flux(state, axis, gamma, p_inf):
    state = np.asarray(state, dtype=np.float64)
    if state.shape[0] != 4 or axis not in (0, 1):
        raise ValueError("M03 oracle requires four conservative components in 2D")
    if not (np.isfinite(state).all() and np.all(state[0] > 0)):
        raise ValueError("nonfinite or nonpositive density in boundary oracle")
    density, mx, my, energy = state
    internal = energy - (mx * mx + my * my) / (2 * density)
    pressure = (gamma - 1) * internal - gamma * p_inf
    if not (np.all(internal > 0) and np.all(pressure + p_inf > 0)):
        raise ValueError("inadmissible thermodynamic state in boundary oracle")
    velocity = state[axis + 1] / density
    sound = np.sqrt(gamma * (pressure + p_inf) / density)
    speed = np.maximum(np.abs(velocity - sound), np.abs(velocity + sound))
    flux = np.stack((state[axis + 1], mx * velocity,
                     my * velocity, (energy + pressure) * velocity))
    flux[axis + 1] += pressure
    if not np.isfinite(flux).all() or not np.isfinite(speed).all():
        raise ValueError("nonfinite boundary flux or speed")
    return flux, speed


def rusanov(left, right, axis, gamma, p_inf):
    left_flux, left_speed = physical_flux(left, axis, gamma, p_inf)
    right_flux, right_speed = physical_flux(right, axis, gamma, p_inf)
    return .5 * (left_flux + right_flux) - .5 * np.maximum(left_speed, right_speed) * (right - left)


def boundary_bands(state):
    state = np.asarray(state, dtype=np.float64)
    if state.ndim != 3 or state.shape[0] != 4 or state.shape[2] < 4:
        raise ValueError("expected a full (4, ny, nx) state, nx >= 4")
    return state[:, :, [0, 1, -2, -1]].copy()


def ssprk2_boundary_increment(bands, dt, dx, dy, gamma, p_inf):
    """Return the incoming conservative integral from the two x boundaries."""
    bands = np.asarray(bands, dtype=np.float64)
    if bands.ndim != 3 or bands.shape[0] != 4 or bands.shape[2] != 4:
        raise ValueError("boundary bands must be (4, ny, 4), ordered first/second/penultimate/last")
    if not all(np.isfinite(value) and value > 0 for value in (dt, dx, dy)):
        raise ValueError("positive finite dt and spacings are required")
    edges = bands[:, :, [0, 3]]
    exterior, _ = physical_flux(edges, 0, gamma, p_inf)
    left_inner = rusanov(bands[:, :, 0], bands[:, :, 1], 0, gamma, p_inf)
    right_inner = rusanov(bands[:, :, 2], bands[:, :, 3], 0, gamma, p_inf)
    x_rate = np.stack((exterior[:, :, 0] - left_inner,
                       right_inner - exterior[:, :, 1]), axis=-1) / dx
    transverse_upper = rusanov(edges, np.roll(edges, -1, axis=1), 1, gamma, p_inf)
    y_rate = (np.roll(transverse_upper, 1, axis=1) - transverse_upper) / dy
    stage_edges = edges + dt * (x_rate + y_rate)
    stage_exterior, _ = physical_flux(stage_edges, 0, gamma, p_inf)
    weighted = .5 * dt * (exterior + stage_exterior)
    return dy * np.sum(weighted[:, :, 0] - weighted[:, :, 1], axis=1)


def reopen_uniform_state(output, shape):
    """Assemble authenticated NPZ pieces with exact, complete coverage."""
    fields = output.manifest["datasets"]["fields"]
    if len(fields) != 1:
        raise ValueError("M03 trajectory must contain exactly one state")
    field = next(iter(fields.values()))
    if tuple(field["global_shape"]) != tuple(shape[1:]):
        raise ValueError("M03 trajectory has the wrong global shape")
    state = np.empty(shape, dtype=np.float64)
    coverage = np.zeros(shape[1:], dtype=np.uint8)
    for piece in field["pieces"]:
        selection = tuple(slice(a, b) for a, b in zip(piece["lower"], piece["upper"], strict=True))
        if np.any(coverage[selection]):
            raise ValueError("M03 trajectory pieces overlap")
        values = output.arrays[piece["name"]]
        if values.shape != state[(slice(None), *selection)].shape:
            raise ValueError("M03 trajectory component or piece shape differs")
        state[(slice(None), *selection)] = values
        coverage[selection] = 1
    if not np.all(coverage):
        raise ValueError("M03 trajectory is incomplete")
    return state
