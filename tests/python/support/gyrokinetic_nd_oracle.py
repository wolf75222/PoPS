"""Independent dense spatial solve and NumPy gyrokinetic FV reference."""

from __future__ import annotations

import numpy as np

from .kinetic_nd_oracle import coordinates


def _gradient(value):
    return np.stack(
        [(np.roll(value, -1, a) - np.roll(value, 1, a)) * value.shape[a] / 2 for a in range(3)],
        axis=-1,
    )


def _curl(vector):
    derivatives = [_gradient(vector[..., a]) for a in range(3)]
    return np.stack(
        [
            derivatives[(a + 2) % 3][..., (a + 1) % 3] - derivatives[(a + 1) % 3][..., (a + 2) % 3]
            for a in range(3)
        ],
        axis=-1,
    )


def _incidence(shape):
    size = int(np.prod(shape))
    indices = np.arange(size).reshape(shape)
    matrices = []
    for a, n in enumerate(shape):
        rows = np.arange(size)
        right = np.roll(indices, -1, a).ravel()
        g = np.zeros((size, size))
        g[rows, rows] = -n
        g[rows, right] += n
        matrices.append(g)
    return matrices


def _operator(incidence, coefficient, direction):
    coefficient = coefficient.ravel()
    b = direction.reshape(-1, 3)
    matrix = np.zeros_like(incidence[0])
    for i in range(3):
        for j in range(3):
            metric = coefficient * ((1.0 if i == j else 0.0) - b[:, i] * b[:, j])
            matrix += incidence[i].T @ (metric[:, None] * incidence[j])
    return matrix


def _solve(matrix, rhs, *, gauge):
    size = rhs.size
    mean = float(rhs.mean()) if gauge else 0.0
    work = matrix + np.ones_like(matrix) / size if gauge else matrix
    solution = np.linalg.solve(work, (rhs - mean).ravel()).reshape(rhs.shape)
    residual = matrix @ solution.ravel() - rhs.ravel()
    return solution, float(residual @ residual), mean


def gyrokinetic_reference(
    *,
    configuration_shape=(3, 3, 3),
    velocity_shape=(4, 3),
    nonlinear_polarization=True,
    geometry_amplitude=0.1,
    mu0=0.2,
    collision_frequency=0.0,
    collision_projection=True,
    dt=1e-5,
    steps=2,
):
    shape, velocity_shape = tuple(configuration_shape), tuple(velocity_shape)
    masses, charges = (1.0, 2.0), (-1.0, 1.0)
    xyz = coordinates(shape, [(0.0, 1.0)] * 3)
    velocity = coordinates(velocity_shape, [(-4.0, 4.0), (0.0, 2.0)])
    v = velocity[..., 0]
    mu = velocity[..., 1]
    widths = (8 / velocity_shape[0], 2 / velocity_shape[1])
    weight = np.prod(widths)
    magnetic = np.stack(
        (
            1 + geometry_amplitude * np.sin(2 * np.pi * xyz[..., 1]),
            0.37 + geometry_amplitude * np.sin(2 * np.pi * xyz[..., 0]),
            np.zeros(shape),
        ),
        axis=-1,
    )
    magnitude = np.linalg.norm(magnetic, axis=-1)
    direction = magnetic / magnitude[..., None]
    curvature = _curl(direction)
    incidence = _incidence(shape)
    ampere = _operator(incidence, np.ones(shape), direction)

    def moment(value, mass, power):
        return 2 * np.pi / mass * weight * np.sum(value * v**power, axis=(-2, -1))

    initial = []
    for s, mass in enumerate(masses):
        temperature = 1 + 0.1 * s
        raw = np.exp(
            -mass * v**2 / (2 * temperature) - mu * magnitude[..., None, None] / temperature
        )
        normalization = moment(raw, mass, 0)
        number = (
            1
            + (-1) ** s
            * 0.01
            * sum(np.cos(2 * np.pi * xyz[..., a] + 0.2 * a) for a in range(3))
            / 3
        )
        initial.append(raw * (number / normalization)[..., None, None])
    initial = np.stack(initial)
    data = initial.copy()

    def divergence(value, speed, axis, *, centered=False):
        right, right_speed = np.roll(value, -1, axis), np.roll(speed, -1, axis)
        face_speed = (speed + right_speed) / 2
        faces = face_speed * (value + right) / 2
        if not centered:
            faces -= abs(face_speed) * (right - value) / 2
        if axis >= 3:
            selector = [slice(None)] * 5
            selector[axis] = -1
            faces[tuple(selector)] = 0
        width = 1 / shape[axis] if axis < 3 else widths[axis - 3]
        return (faces - np.roll(faces, 1, axis)) / width

    def collision(value, mass):
        number, momentum = moment(value, mass, 0), moment(value, mass, 1)
        second = (
            2
            * np.pi
            / mass
            * weight
            * np.sum(value * (v * v + 2 * mu * magnitude[..., None, None] / mass), axis=(-2, -1))
        )
        mean = momentum / number
        theta = (second / number - mean**2) / 3
        result = np.zeros_like(value)
        for a in (0, 1):
            axis = 3 + a
            right = np.roll(value, -1, axis)
            location = velocity[..., a] + widths[a] / 2
            drift = location - mean[..., None, None] if a == 0 else 2 * location
            diffusion = (
                theta[..., None, None]
                if a == 0
                else 2 * location * mass / magnitude[..., None, None] * theta[..., None, None]
            )
            faces = collision_frequency * (
                drift * (value + right) / 2 + diffusion * (right - value) / widths[a]
            )
            selector = [slice(None)] * 5
            selector[axis] = -1
            faces[tuple(selector)] = 0
            result += (faces - np.roll(faces, 1, axis)) / widths[a]
        before, after = [], []
        for index in np.ndindex(shape):
            basis = np.stack((np.ones_like(v), v, v**2 + 2 * mu * magnitude[index] / mass)).reshape(
                3, -1
            )
            raw = result[index].ravel()
            defect = weight * basis @ raw
            before.append(float(np.max(abs(defect))))
            if collision_projection:
                gram = weight * basis @ basis.T
                result[index] -= (basis.T @ np.linalg.solve(gram, defect)).reshape(velocity_shape)
            after.append(float(np.max(abs(weight * basis @ result[index].ravel()))))
        return result, max(before), max(after)

    records = []
    for _ in range(steps):
        numbers = [moment(value, mass, 0) for value, mass in zip(data, masses, strict=True)]
        charge = sum(q * n for q, n in zip(charges, numbers, strict=True))
        current = sum(
            q * moment(value, mass, 1) for q, value, mass in zip(charges, data, masses, strict=True)
        )
        coefficient = (
            sum(
                m * n if nonlinear_polarization else m * np.ones(shape)
                for m, n in zip(masses, numbers, strict=True)
            )
            / magnitude**2
        )
        polarization = _operator(incidence, coefficient, direction)
        phi, phi_residual, neutrality = _solve(polarization, charge, gauge=True)
        apar, apar_residual, mean_current = _solve(ampere, mu0 * current, gauge=True)
        grad_phi, grad_B = _gradient(phi), _gradient(magnitude)
        flutter = _curl(apar[..., None] * direction)
        forward = np.stack([(g @ phi.ravel()).reshape(shape) for g in incidence], axis=-1)
        perpendicular = forward - direction * np.sum(direction * forward, axis=-1)[..., None]
        flow_energy = np.sum(perpendicular**2, axis=-1) / magnitude**2
        grad_energy = _gradient(flow_energy)
        B = magnitude[..., None, None]
        unit = direction[..., None, None, :]
        expanded_v = v[..., None]
        star, dv = [], []
        characteristic_amplitudes = []
        collision_defects_before, collision_defects_after = [], []
        for s, (mass, q) in enumerate(zip(masses, charges, strict=True)):
            effective = (
                magnetic[..., None, None, :]
                + mass / q * expanded_v * curvature[..., None, None, :]
                + flutter[..., None, None, :]
            )
            force = mu[..., None] * grad_B[..., None, None, :] + q * grad_phi[..., None, None, :]
            if nonlinear_polarization:
                force = force - 0.5 * mass * grad_energy[..., None, None, :]
            velocity_x = effective * expanded_v / B[..., None] + np.cross(unit, force) / (
                q * B[..., None]
            )
            velocity_v = -np.sum(effective * force, axis=-1) / (mass * B)
            rate = -sum(divergence(data[s], velocity_x[..., a], a) for a in range(3))
            rate -= divergence(data[s], velocity_v, 3)
            if collision_frequency:
                collision_rate, defect_before, defect_after = collision(data[s], mass)
                rate += collision_rate
                collision_defects_before.append(defect_before)
                collision_defects_after.append(defect_after)
            star.append(rate)
            dv.append(divergence(data[s], np.ones_like(data[s]), 3, centered=True))
            characteristic_amplitudes.append(
                [float(np.max(abs(velocity_x[..., a]))) for a in range(3)]
                + [float(np.max(abs(velocity_v)))]
            )
        rhs = sum(
            mu0 * q * moment(value, mass, 1)
            for q, value, mass in zip(charges, star, masses, strict=True)
        )
        coefficient_a = -sum(
            mu0 * q * q / mass * moment(value, mass, 1)
            for q, value, mass in zip(charges, dv, masses, strict=True)
        )
        induction = ampere + np.diag(coefficient_a.ravel())
        adot, solver_residual, _ = _solve(induction, rhs, gauge=False)
        derivative = np.stack(
            [
                rate + q / mass * adot[..., None, None] * derivative_v
                for rate, q, mass, derivative_v in zip(star, charges, masses, dv, strict=True)
            ]
        )
        original = (ampere @ adot.ravel()).reshape(shape) - sum(
            mu0 * q * moment(value, mass, 1)
            for q, value, mass in zip(charges, derivative, masses, strict=True)
        )
        records.append(
            dict(
                phi=phi,
                apar=apar,
                adot=adot,
                phi_residual=phi_residual,
                apar_residual=apar_residual,
                neutrality=neutrality,
                mean_current=mean_current,
                derivative_residual=float(np.sum(original**2)),
                solver_residual=solver_residual,
                characteristic_amplitudes=characteristic_amplitudes,
                min_induction_mass=float(coefficient_a.min()),
                max_nonlinear_hamiltonian=float(abs(flow_energy).max()),
                max_curvature=float(abs(curvature).max()),
                max_flutter=float(abs(flutter).max()),
                collision_defects_before=collision_defects_before,
                collision_defects_after=collision_defects_after,
            )
        )
        data += dt * derivative
    return initial, data, records
