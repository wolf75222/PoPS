"""Host-only finite-volume references for reception of M22/M24/M25.

These functions consume NumPy arrays after initialisation or native execution.
They never enter a PoPS Program and do not compile or run an alternative engine.
The constitutive laws and spatial/time discretisations are independent NumPy
implementations. No authored model equation or PoPS operator is imported.
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
        first = value + dt * rhs(value)
        value = 0.5 * value + 0.5 * (first + dt * rhs(first))
    if not np.all(np.isfinite(value)):
        raise ValueError("reference evolution left its finite domain")
    return value


def _conservative_rhs(value, physical_flux_and_speed):
    result = np.zeros_like(value)
    for direction, storage_axis in ((0, 2), (1, 1)):
        flux, speed = physical_flux_and_speed(value, direction)
        right = np.roll(value, -1, axis=storage_axis)
        right_flux = np.roll(flux, -1, axis=storage_axis)
        bound = np.maximum(speed, np.roll(speed, -1, axis=storage_axis - 1))
        face = 0.5 * (flux + right_flux - bound[None] * (right - value))
        result -= value.shape[storage_axis] * (face - np.roll(face, 1, axis=storage_axis))
    return result


def _gas_flux(value, direction):
    rho, mx, my, total = value
    if np.any(rho <= 0):
        raise ValueError("Euler reference requires positive density")
    velocity = value[direction + 1] / rho
    pressure = 0.4 * (total - (mx * mx + my * my) / (2 * rho))
    if np.any(pressure <= 0):
        raise ValueError("Euler reference requires positive internal energy")
    flux = np.stack(
        (
            value[direction + 1],
            mx * velocity + (pressure if direction == 0 else 0),
            my * velocity + (pressure if direction == 1 else 0),
            (total + pressure) * velocity,
        )
    )
    return flux, np.abs(velocity) + np.sqrt(1.4 * pressure / rho)


def _radiation_flux(value, direction):
    energy, fx, fy = value
    if np.any(energy <= 0):
        raise ValueError("M1 reference requires positive radiation energy")
    reduced_squared = (fx * fx + fy * fy) / (energy * energy)
    if np.any(reduced_squared > 1):
        raise ValueError("M1 reference requires |F| <= E")
    # Independent algebraic form of the Eddington factor.
    chi = (5 - 2 * np.sqrt(4 - 3 * reduced_squared)) / 3
    anisotropy = np.divide(
        3 * chi - 1, 2 * reduced_squared, out=np.full_like(chi, 0.75), where=reduced_squared > 1e-14
    )

    def pressure(i):
        return (energy * (1 - chi) / 2 if i == direction else 0) + anisotropy * value[
            i + 1
        ] * value[direction + 1] / energy

    return np.stack((value[direction + 1], pressure(0), pressure(1))), np.ones_like(energy)


def rhs_m22(material, radiation, *, opacity=3.0):
    """Both recipients of one lab-frame emission/momentum exchange."""
    material, radiation = _state(material, 4), _state(radiation, 3)
    if material.shape[1:] != radiation.shape[1:] or not np.isfinite(opacity) or opacity < 0:
        raise ValueError("co-located fields and finite opacity>=0 required")
    km = _conservative_rhs(material, _gas_flux)
    kr = _conservative_rhs(radiation, _radiation_flux)
    temperature = (
        material[3] - (material[1] ** 2 + material[2] ** 2) / (2 * material[0])
    ) / material[0]
    emission = opacity * (temperature**4 - radiation[0])
    drag = opacity * radiation[1:]
    km[1:3] += drag
    km[3] -= emission
    kr[0] += emission
    kr[1:] -= drag
    return km, kr


def advance_m22(material, radiation, *, opacity=3.0, dt=0.0005, steps=4):
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
    if isinstance(matrix, np.ndarray):
        return np.einsum("ij...,j...->i...", matrix, jump)
    return np.stack(
        [sum(coefficient * jump[k] for k, coefficient in enumerate(row)) for row in matrix]
    )


def _path_rhs(value, equations, speed_bound, *, path_points):
    if type(path_points) is not int or path_points < 1:
        raise ValueError("positive integer Gauss order required")
    nodes, weights = np.polynomial.legendre.leggauss(path_points)
    nodes, weights = (nodes + 1) / 2, weights / 2
    fluxes, _matrices, source = equations(value)
    result = _vector(source, value)
    for direction, storage_axis in ((0, 2), (1, 1)):
        right = np.roll(value, -1, axis=storage_axis)
        jump = right - value
        flux = _vector(fluxes[direction], value)
        bound = np.broadcast_to(speed_bound(value, right, direction), value.shape[1:])
        face = 0.5 * (flux + np.roll(flux, -1, axis=storage_axis) - bound[None] * jump)
        integral = np.zeros_like(value)
        for node, weight in zip(nodes, weights, strict=True):
            _fluxes, matrices, _source = equations(value + node * jump)
            integral += weight * _matrix_action(matrices[direction], jump)
        # Each interface delivers -I/2 to each of its two incident cells.
        result -= value.shape[storage_axis] * (
            face
            - np.roll(face, 1, axis=storage_axis)
            + 0.5 * (integral + np.roll(integral, 1, axis=storage_axis))
        )
    return result


def _bn_constitutive(value):
    """S18, arXiv:1601.07345v4, equations (2),(7), ideal-gas variant.

    Work in two-phase primitive arrays rather than reuse the authored scalar
    expressions. The interface closure is p_I=p_1 and u_I=u_2.
    """
    fraction = np.stack((value[0], 1.0 - value[0]))
    mass, momentum, energy = value[1:3], value[3:5], value[5:7]
    if np.any(fraction <= 0) or np.any(mass <= 0):
        raise ValueError("BN reference requires two positive phases")
    velocity = momentum / mass
    density = mass / fraction
    internal = energy / mass - 0.5 * velocity**2
    pressure = np.array([0.4, 0.6])[:, None, None] * density * internal
    if np.any(pressure <= 0):
        raise ValueError("BN reference requires positive phasic pressure")
    return fraction, velocity, pressure


def _bn_equations(value):
    fraction, velocity, pressure = _bn_constitutive(value)
    fluxes = np.zeros((2, *value.shape))
    fluxes[0, 1:3] = value[3:5]
    fluxes[0, 3:5] = value[3:5] * velocity + fraction * pressure
    fluxes[0, 5:7] = (value[5:7] + fraction * pressure) * velocity
    matrices = np.zeros((2, 7, 7, *value.shape[1:]))
    matrices[0, 0, 0] = velocity[1]
    # Equal and opposite interface pressure forces and interface work.
    interface_force = np.array([-1.0, 1.0])[:, None, None] * pressure[0]
    matrices[0, 3:5, 0] = interface_force
    matrices[0, 5:7, 0] = interface_force * velocity[1]
    return fluxes, matrices, np.zeros_like(value)


def _bn_speed(left, right, direction):
    if direction == 1:
        return 0.0
    bound = np.zeros_like(left[0])
    for density, momentum, energy, gamma in ((1, 3, 5, 1.4), (2, 4, 6, 1.6)):
        velocity = np.maximum(
            abs(left[momentum] / left[density]), abs(right[momentum] / right[density])
        )
        energy_per_mass = np.maximum(left[energy] / left[density], right[energy] / right[density])
        bound = np.maximum(bound, velocity + np.sqrt(gamma * (gamma - 1) * energy_per_mass))
    return bound


def rhs_m24(value, *, path_points=2, pressure_relaxation=0.0, velocity_relaxation=0.0):
    value = _state(value, 7)
    if any(not np.isfinite(c) or c < 0 for c in (pressure_relaxation, velocity_relaxation)):
        raise ValueError("BN reference requires finite nonnegative relaxation coefficients")
    result = _path_rhs(value, _bn_equations, _bn_speed, path_points=path_points)
    _, (velocity_a, velocity_b), (pressure_a, pressure_b) = _bn_constitutive(value)
    amount = pressure_relaxation * (pressure_a - pressure_b)
    force = velocity_relaxation * (velocity_b - velocity_a)
    energy = -pressure_a * amount + velocity_b * force
    result[0] += amount
    result[3] += force
    result[4] -= force
    result[5] += energy
    result[6] -= energy
    return result


def advance_m24(
    initial, *, dt=0.0005, steps=4, path_points=2, pressure_relaxation=0.0, velocity_relaxation=0.0
):
    return _advance(
        _state(initial, 7),
        lambda value: rhs_m24(
            value,
            path_points=path_points,
            pressure_relaxation=pressure_relaxation,
            velocity_relaxation=velocity_relaxation,
        ),
        dt=dt,
        steps=steps,
    )


def _gpr_constitutive(value):
    """Independent tensor form of the non-electromagnetic GPR/HPR potential.

    See arXiv:1511.08995, equations (1),(2),(6),(8),(9),(11),(12), the
    thermomechanical subsystem underlying S19 (arXiv:1612.02093).
    gamma=1.4 and cv=cs=alpha=rho0=T0=1; E is total energy density.
    """
    rho = value[0]
    if np.any(rho <= 0):
        raise ValueError("GPR reference requires positive density")
    velocity = np.moveaxis(value[1:4] / rho, 0, -1)
    impulse = np.moveaxis(value[14:17] / rho, 0, -1)
    distortion = np.moveaxis(value[5:14].reshape(3, 3, *rho.shape), (0, 1), (-2, -1))
    determinant = np.linalg.det(distortion)
    if np.any(determinant <= 0):
        raise ValueError("GPR reference requires positive distortion determinant")
    gram = np.einsum("...ki,...kj->...ij", distortion, distortion)
    deviator = gram - np.trace(gram, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    elastic = 0.25 * np.einsum("...ij,...ij->...", deviator, deviator)
    thermal = 0.5 * np.einsum("...i,...i->...", impulse, impulse)
    kinetic = 0.5 * np.einsum("...i,...i->...", velocity, velocity)
    temperature = value[4] / rho - kinetic - elastic - thermal
    if np.any(temperature <= 0):
        raise ValueError("GPR reference requires positive temperature")
    gradient = distortion @ deviator
    stress = -rho[..., None, None] * (gram @ deviator)
    return dict(
        rho=rho,
        velocity=velocity,
        impulse=impulse,
        distortion=distortion,
        determinant=determinant,
        elastic=elastic,
        thermal=thermal,
        temperature=temperature,
        pressure=0.4 * rho * temperature,
        distortion_gradient=gradient,
        stress=stress,
    )


def _gpr_equations(value, *, tau_distortion=0.1, tau_heat=0.1):
    if not all(np.isfinite(tau) and tau > 0 for tau in (tau_distortion, tau_heat)):
        raise ValueError("GPR reference requires positive finite relaxation times")
    c = _gpr_constitutive(value)
    rho, velocity, impulse = c["rho"], c["velocity"], c["impulse"]
    distortion, temperature, pressure = c["distortion"], c["temperature"], c["pressure"]
    stress = c["stress"]
    fluxes = np.zeros((2, *value.shape))
    matrices = np.zeros((2, 17, 17, *rho.shape))
    source = np.zeros_like(value)
    relaxation = (
        -3
        * c["determinant"][..., None, None] ** (5 / 3)
        * c["distortion_gradient"]
        / tau_distortion
    )
    source[5:14] = np.moveaxis(relaxation.reshape(*rho.shape, 9), -1, 0)
    source[14:17] = np.moveaxis(-temperature[..., None] * impulse / tau_heat, -1, 0)
    advected_distortion = np.einsum("...ij,...j->...i", distortion, velocity)
    for axis in range(2):
        fluxes[axis, 0] = value[axis + 1]
        traction = -stress[..., :, axis].copy()
        traction[..., axis] += pressure
        momentum_flux = rho[..., None] * velocity * velocity[..., axis, None] + traction
        fluxes[axis, 1:4] = np.moveaxis(momentum_flux, -1, 0)
        fluxes[axis, 4] = (
            value[4] * velocity[..., axis]
            + np.einsum("...i,...i->...", velocity, traction)
            + temperature * impulse[..., axis]
        )
        distortion_flux = np.zeros_like(distortion)
        distortion_flux[..., :, axis] = advected_distortion
        fluxes[axis, 5:14] = np.moveaxis(distortion_flux.reshape(*rho.shape, 9), -1, 0)
        impulse_flux = rho[..., None] * impulse * velocity[..., axis, None]
        impulse_flux[..., axis] += temperature
        fluxes[axis, 14:17] = np.moveaxis(impulse_flux, -1, 0)
        curl_transport = velocity[..., axis, None, None] * np.eye(3)
        curl_transport[..., axis, :] -= velocity
        blocks = np.einsum("ij,...kl->...ikjl", np.eye(3), curl_transport)
        matrices[axis, 5:14, 5:14] = np.moveaxis(blocks.reshape(*rho.shape, 9, 9), (-2, -1), (0, 1))
    return fluxes, matrices, source


def rhs_m25(value, *, tau_distortion=0.1, tau_heat=0.1, path_points=2):
    """Full17-component flux, two-sided B path and constitutive relaxation."""
    value = _state(value, 17)

    def equations(q):
        return _gpr_equations(q, tau_distortion=tau_distortion, tau_heat=tau_heat)

    return _path_rhs(value, equations, lambda left, right, direction: 64.0, path_points=path_points)


def advance_m25(initial, *, dt=1e-5, steps=8, tau_distortion=0.1, tau_heat=0.1, path_points=2):
    return _advance(
        _state(initial, 17),
        lambda value: rhs_m25(
            value, tau_distortion=tau_distortion, tau_heat=tau_heat, path_points=path_points
        ),
        dt=dt,
        steps=steps,
    )


__all__ = ["rhs_m22", "advance_m22", "rhs_m24", "advance_m24", "rhs_m25", "advance_m25"]
