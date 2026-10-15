"""Independent NumPy FV reference; never used by the executed PoPS graph."""

from __future__ import annotations

import numpy as np


def coordinates(shape, bounds):
    axes = [
        a + (np.arange(n) + 0.5) * (b - a) / n for n, (a, b) in zip(shape, bounds, strict=True)
    ]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def initial(configuration_shape, velocity_shape, masses, extent=4.0):
    nx, nv = tuple(configuration_shape), tuple(velocity_shape)
    x = coordinates(nx, [(0.0, 1.0)] * len(nx))
    v = coordinates(nv, [(-extent, extent)] * len(nv))
    dv = np.prod([2 * extent / n for n in nv])
    populations = []
    for s, mass in enumerate(masses):
        mean = np.array([0.12 * (s + 1) * (-1) ** a for a in range(len(nv))])
        base = np.exp(-mass * np.sum((v - mean) ** 2, axis=-1) / (2 * (1 + 0.1 * s)))
        base /= dv * np.sum(base)
        modulation = 1 + sum(
            0.025 / len(nx) * np.cos(2 * np.pi * x[..., a] + 0.3 * s + 0.2 * a)
            for a in range(len(nx))
        )
        populations.append(
            modulation.reshape(nx + (1,) * len(nv)) * base.reshape((1,) * len(nx) + nv)
        )
    return np.stack(populations)


def kinetic_reference(
    *,
    model,
    configuration_shape,
    velocity_shape,
    masses,
    charges,
    frequency=0.2,
    dt=1e-5,
    steps=2,
    extent=4.0,
):
    nx, nv = tuple(configuration_shape), tuple(velocity_shape)
    dx = np.asarray([1 / n for n in nx])
    widths = np.asarray([2 * extent / n for n in nv])
    weight = np.prod(widths)
    count_v = int(np.prod(nv))
    xdim, vdim = len(nx), len(nv)
    coordinates_v = coordinates(nv, [(-extent, extent)] * vdim)
    flat_v = coordinates_v.reshape(-1, vdim)
    basis = np.vstack((np.ones(count_v), flat_v.T, np.sum(flat_v**2, axis=1)))
    gram = weight * basis @ basis.T
    gram_inverse = np.linalg.inv(gram)
    data = initial(nx, nv, masses, extent)
    first = data.copy()
    species_count = len(masses)

    def moment(f):
        return weight * np.einsum("kv,...v->k...", basis, f.reshape(nx + (count_v,)))

    def close_faces(flux, axis):
        # The finite velocity boundary is zero flux, not wrap-around transport.
        last = [slice(None)] * flux.ndim
        last[axis] = -1
        flux[tuple(last)] = 0
        return flux

    diagnostics = []
    for _ in range(steps):
        moments = [moment(f) for f in data]
        electric = []
        if model == "M19":
            rho = sum(q * m[0] for q, m in zip(charges, moments, strict=True)) - sum(charges)
            symbol = np.zeros(nx)
            for a, n in enumerate(nx):
                shape = [1] * xdim
                shape[a] = n
                symbol += (4 * np.sin(np.pi * np.fft.fftfreq(n)) ** 2 / dx[a] ** 2).reshape(shape)
            spectrum = np.fft.fftn(rho)
            inverse = np.divide(1.0, symbol, out=np.zeros_like(symbol), where=symbol != 0)
            potential = np.fft.ifftn(spectrum * inverse).real
            electric = [
                -(np.roll(potential, -1, a) - np.roll(potential, 1, a)) / (2 * dx[a])
                for a in range(xdim)
            ]
            diagnostics.append(
                {
                    "charge_mean": float(rho.mean()),
                    "field_amplitudes": [float(np.max(abs(e))) for e in electric],
                }
            )
        sources = []
        if model == "M19":
            for f, moment_values in zip(data, moments, strict=True):
                number = moment_values[0]
                mean = np.moveaxis(moment_values[1:-1] / number, 0, -1)
                theta = (moment_values[-1] / number - np.sum(mean**2, axis=-1)) / vdim
                exponent = np.sum(
                    (flat_v.reshape((1,) * xdim + (count_v, vdim)) - mean[..., None, :]) ** 2,
                    axis=-1,
                )
                sampled = (
                    number[..., None]
                    * np.exp(-exponent / (2 * theta[..., None]))
                    / (2 * np.pi * theta[..., None]) ** (vdim / 2)
                )
                defect = moment_values - weight * np.einsum("kv,...v->k...", basis, sampled)
                coefficients = np.einsum("kl,l...->k...", gram_inverse, defect)
                equilibrium = sampled + np.einsum("vk,k...->...v", basis.T, coefficients)
                sources.append(frequency * (equilibrium.reshape(nx + nv) - f))
        else:
            for s, f in enumerate(data):
                total = np.zeros_like(f)
                ns = moments[s][0]
                us = moments[s][1:-1] / ns
                ts = masses[s] * (moments[s][-1] / ns - np.sum(us**2, axis=0)) / vdim
                for t in range(species_count):
                    nt = moments[t][0]
                    ut = moments[t][1:-1] / nt
                    tt = masses[t] * (moments[t][-1] / nt - np.sum(ut**2, axis=0)) / vdim
                    rs, rt = masses[s] * ns, masses[t] * nt
                    mean = (rs * us + rt * ut) / (rs + rt)
                    temperature = (
                        ns * ts
                        + nt * tt
                        + rs * rt / (rs + rt) * np.sum((us - ut) ** 2, axis=0) / vdim
                    ) / (ns + nt)
                    for a in range(vdim):
                        axis = xdim + a
                        right = np.roll(f, -1, axis)
                        face_coordinate = coordinates_v[..., a] + widths[a] / 2
                        drift = face_coordinate.reshape((1,) * xdim + nv) - mean[a].reshape(
                            nx + (1,) * vdim
                        )
                        diffusion = temperature.reshape(nx + (1,) * vdim) / masses[s]
                        faces = frequency * (
                            drift * (f + right) / 2 + diffusion * (right - f) / widths[a]
                        )
                        close_faces(faces, axis)
                        total += (faces - np.roll(faces, 1, axis)) / widths[a]
                sources.append(total)
            # Joint, not pairwise-negative, correction of the finite-grid defects.
            constraints = np.zeros((species_count + vdim + 1, species_count * count_v))
            for s, mass in enumerate(masses):
                sl = slice(s * count_v, (s + 1) * count_v)
                constraints[s, sl] = 1
                constraints[species_count : species_count + vdim, sl] = mass * flat_v.T
                constraints[-1, sl] = 0.5 * mass * np.sum(flat_v**2, axis=1)
            raw = np.concatenate([a.reshape(nx + (count_v,)) for a in sources], axis=-1)
            error = weight * np.einsum("kv,...v->k...", constraints, raw)
            alpha = np.linalg.solve(
                weight * constraints @ constraints.T, error.reshape(len(error), -1)
            ).reshape(error.shape)
            raw -= np.einsum("vk,k...->...v", constraints.T, alpha)
            sources = [
                raw[..., s * count_v : (s + 1) * count_v].reshape(nx + nv)
                for s in range(species_count)
            ]
        derivative = []
        for s, f in enumerate(data):
            rhs = sources[s].copy()
            for a in range(xdim):
                speed = coordinates_v[..., a].reshape((1,) * xdim + nv)
                right = np.roll(f, -1, a)
                faces = (speed * (f + right) - abs(speed) * (right - f)) / 2
                rhs -= (faces - np.roll(faces, 1, a)) / dx[a]
            for a, electric_a in enumerate(electric):
                axis = xdim + a
                speed = charges[s] / masses[s] * electric_a.reshape(nx + (1,) * vdim)
                right = np.roll(f, -1, axis)
                faces = close_faces((speed * (f + right) - abs(speed) * (right - f)) / 2, axis)
                rhs -= (faces - np.roll(faces, 1, axis)) / widths[a]
            derivative.append(rhs)
        data = data + dt * np.stack(derivative)
    return first, data, diagnostics
