"""Host-only finite-volume references for reception of M22/M24/M25.

These functions consume NumPy arrays after initialisation or native execution.
They never enter a PoPS Program and do not compile or run an alternative engine.
The spatial/time discretisations are independent implementations. M24/M25 share
the explicitly authored constitutive equations, not PoPS lowering or operators.
Arrays are ordered (component, y cell, x cell), on a periodic unit rectangle.
"""
from __future__ import annotations

import numpy as np


def _state(values, width):
    result = np.asarray(values, dtype=float)
    if result.ndim != 3 or result.shape[0] != width or min(result.shape[1:]) < 1:
        raise ValueError(f"expected ({width}, ny, nx) cell averages")
    if not np.all(np.isfinite(result)):
        raise ValueError("reference input must be finite")
    return result


def _vector(entries, like):
    return np.stack([np.broadcast_to(entry, like.shape[1:]) for entry in entries])


def _advance(initial, rhs, *, dt, steps):
    if not np.isfinite(dt) or dt <= 0 or type(steps) is not int or steps < 1:
        raise ValueError("positive finite dt and positive integer steps required")
    value = np.array(initial, dtype=float, copy=True)
    for _ in range(steps):
        first = value + dt*rhs(value)
        value = .5*value + .5*(first + dt*rhs(first))
    if not np.all(np.isfinite(value)):
        raise ValueError("reference evolution left its finite domain")
    return value


def _conservative_rhs(value, physical_flux_and_speed):
    result = np.zeros_like(value)
    for direction, storage_axis in ((0, 2), (1, 1)):
        flux, speed = physical_flux_and_speed(value, direction)
        right = np.roll(value, -1, axis=storage_axis)
        right_flux = np.roll(flux, -1, axis=storage_axis)
        bound = np.maximum(speed, np.roll(speed, -1, axis=storage_axis-1))
        face = .5*(flux + right_flux - bound[None]*(right-value))
        result -= value.shape[storage_axis]*(face-np.roll(face, 1, axis=storage_axis))
    return result


def _gas_flux(value, direction):
    rho, mx, my, total = value
    if np.any(rho <= 0):
        raise ValueError("Euler reference requires positive density")
    velocity = value[direction+1]/rho
    pressure = .4*(total-(mx*mx+my*my)/(2*rho))
    if np.any(pressure <= 0):
        raise ValueError("Euler reference requires positive internal energy")
    flux = np.stack((value[direction+1], mx*velocity+(pressure if direction == 0 else 0),
                     my*velocity+(pressure if direction == 1 else 0), (total+pressure)*velocity))
    return flux, np.abs(velocity)+np.sqrt(1.4*pressure/rho)


def _radiation_flux(value, direction):
    energy, fx, fy = value
    if np.any(energy <= 0):
        raise ValueError("M1 reference requires positive radiation energy")
    reduced_squared = (fx*fx+fy*fy)/(energy*energy)
    if np.any(reduced_squared > 1):
        raise ValueError("M1 reference requires |F| <= E")
    # Independent algebraic form of the Eddington factor.
    chi = (5-2*np.sqrt(4-3*reduced_squared))/3
    anisotropy = np.divide(3*chi-1, 2*reduced_squared,
        out=np.full_like(chi, .75), where=reduced_squared > 1e-14)
    def pressure(i):
        return (energy*(1-chi)/2 if i == direction else 0) + anisotropy*value[i+1]*value[direction+1]/energy
    return np.stack((value[direction+1], pressure(0), pressure(1))), np.ones_like(energy)


def rhs_m22(material, radiation, *, opacity=3.):
    """Both recipients of one lab-frame emission/momentum exchange."""
    material, radiation = _state(material, 4), _state(radiation, 3)
    if material.shape[1:] != radiation.shape[1:] or not np.isfinite(opacity) or opacity < 0:
        raise ValueError("co-located fields and finite opacity>=0 required")
    km = _conservative_rhs(material, _gas_flux)
    kr = _conservative_rhs(radiation, _radiation_flux)
    temperature = (material[3]-(material[1]**2+material[2]**2)/(2*material[0]))/material[0]
    emission = opacity*(temperature**4-radiation[0])
    drag = opacity*radiation[1:]
    km[1:3] += drag
    km[3] -= emission
    kr[0] += emission
    kr[1:] -= drag
    return km, kr


def advance_m22(material, radiation, *, opacity=3., dt=.0005, steps=4):
    """Return (material, radiation) after independent coupled SSPRK2 steps."""
    material, radiation = _state(material, 4), _state(radiation, 3)
    def rhs(combined):
        gas, rad = rhs_m22(combined[:4], combined[4:], opacity=opacity)
        return np.concatenate((gas, rad))
    combined = _advance(np.concatenate((material, radiation)), rhs, dt=dt, steps=steps)
    return combined[:4], combined[4:]


def _matrix_action(matrix, jump):
    # Entries may be scalar constants or cell arrays. No PoPS expression evaluator
    # is involved; this is the finite sum defining B(U) times a path tangent.
    return np.stack([sum(coefficient*jump[k] for k, coefficient in enumerate(row))
                     for row in matrix])


def _path_rhs(value, equations, speed_bound, *, path_points):
    if type(path_points) is not int or path_points < 1:
        raise ValueError("positive integer Gauss order required")
    nodes, weights = np.polynomial.legendre.leggauss(path_points)
    nodes, weights = (nodes+1)/2, weights/2
    fluxes, _matrices, source = equations(value)
    result = _vector(source, value)
    for direction, storage_axis in ((0, 2), (1, 1)):
        right = np.roll(value, -1, axis=storage_axis)
        jump = right-value
        flux = _vector(fluxes[direction], value)
        bound = np.broadcast_to(speed_bound(value, right, direction), value.shape[1:])
        face = .5*(flux+np.roll(flux, -1, axis=storage_axis)-bound[None]*jump)
        integral = np.zeros_like(value)
        for node, weight in zip(nodes, weights, strict=True):
            _fluxes, matrices, _source = equations(value+node*jump)
            integral += weight*_matrix_action(matrices[direction], jump)
        # Each interface delivers -I/2 to each of its two incident cells.
        result -= value.shape[storage_axis]*(face-np.roll(face, 1, axis=storage_axis)
                    + .5*(integral+np.roll(integral, 1, axis=storage_axis)))
    return result


def _bn_equations(value):
    from .models_M22_M25_M28 import bn_equations
    a, r1, r2 = value[:3]
    if np.any(a <= 0) or np.any(a >= 1) or np.any(r1 <= 0) or np.any(r2 <= 0):
        raise ValueError("BN reference requires two positive phases")
    flux, matrix, (_v1, _v2, p1, p2) = bn_equations(tuple(value))
    if np.any(p1 <= 0) or np.any(p2 <= 0):
        raise ValueError("BN reference requires positive phasic pressure")
    return (flux, (0.,)*7), (matrix, ((0.,)*7,)*7), (0.,)*7


def _bn_speed(left, right, direction):
    if direction == 1:
        return 0.
    bound = np.zeros_like(left[0])
    for density, momentum, energy, gamma in ((1, 3, 5, 1.4), (2, 4, 6, 1.6)):
        velocity = np.maximum(abs(left[momentum]/left[density]), abs(right[momentum]/right[density]))
        energy_per_mass = np.maximum(left[energy]/left[density], right[energy]/right[density])
        bound = np.maximum(bound, velocity+np.sqrt(gamma*(gamma-1)*energy_per_mass))
    return bound


def rhs_m24(value, *, path_points=2):
    return _path_rhs(_state(value, 7), _bn_equations, _bn_speed, path_points=path_points)


def advance_m24(initial, *, dt=.0005, steps=4, path_points=2):
    return _advance(_state(initial, 7), lambda value: rhs_m24(value, path_points=path_points), dt=dt, steps=steps)


def rhs_m25(value, *, tau_distortion=.1, tau_heat=.1, path_points=2):
    """Full17-component flux, two-sided B path and constitutive relaxation."""
    from .models_M22_M25_M28 import gpr_equations
    value = _state(value, 17)
    if not all(np.isfinite(tau) and tau > 0 for tau in (tau_distortion, tau_heat)):
        raise ValueError("GPR reference requires positive finite relaxation times")
    def equations(q):
        if np.any(q[0] <= 0):
            raise ValueError("GPR reference requires positive density")
        matrices = np.moveaxis(q[5:14].reshape(3, 3, *q.shape[1:]), (0, 1), (-2, -1))
        if np.any(np.linalg.det(matrices) <= 0):
            raise ValueError("GPR reference requires positive distortion determinant")
        fluxes, products, source, (temperature, _det, _elastic, _thermal) = gpr_equations(
            tuple(q), tau_distortion=tau_distortion, tau_heat=tau_heat)
        if np.any(temperature <= 0):
            raise ValueError("GPR reference requires positive temperature")
        return fluxes, products, source
    return _path_rhs(value, equations, lambda left, right, direction: 64., path_points=path_points)


def advance_m25(initial, *, dt=1e-5, steps=8, tau_distortion=.1, tau_heat=.1, path_points=2):
    return _advance(_state(initial, 17), lambda value: rhs_m25(value,
        tau_distortion=tau_distortion, tau_heat=tau_heat, path_points=path_points), dt=dt, steps=steps)


__all__ = ["rhs_m22", "advance_m22", "rhs_m24", "advance_m24", "rhs_m25", "advance_m25"]
