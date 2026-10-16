"""Independent dense incidence matrices for the periodic Hall-MHD CT profile."""

from __future__ import annotations

import numpy as np

NAMES = ("rho", "mx", "my", "mz", "energy", "bx", "by", "bz")


def hall_mhd_reference(*, cells=(6, 8), hall=0.15, ohmic=0.01, gamma=1.4, dt=1e-4, steps=3):
    shape = tuple(cells)
    n = int(np.prod(shape))
    eye = np.eye(n)
    ids = np.arange(n).reshape(shape)
    plus = []
    average = []
    for axis in range(2):
        shift = eye[np.roll(ids, -1, axis).ravel()]
        plus.append((shift - eye) * shape[axis])
        average.append((eye + shift) / 2)
    minus = [-matrix.T for matrix in plus]
    center = [(a + b) / 2 for a, b in zip(plus, minus, strict=True)]
    M = (average[1], average[0], average[0] @ average[1])
    x, y = np.meshgrid(
        (np.arange(shape[0]) + 0.5) / shape[0],
        (np.arange(shape[1]) + 0.5) / shape[1],
        indexing="ij",
    )
    x, y = x.ravel(), y.ravel()
    potential = 0.03 * np.sin(2 * np.pi * (x - 0.5 / shape[0])) * np.sin(
        2 * np.pi * (y - 0.5 / shape[1])
    ) + 0.01 * np.cos(4 * np.pi * (x - 0.5 / shape[0]) + 2 * np.pi * (y - 0.5 / shape[1]))
    state = np.zeros((8, n))
    state[0] = 1 + 0.05 * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    velocity = np.array(
        [
            0.1 * np.sin(2 * np.pi * y),
            0.1 * np.cos(2 * np.pi * x),
            0.05 * np.sin(2 * np.pi * (x + y)),
        ]
    )
    state[1:4] = state[0] * velocity
    state[5] = 1 + plus[1] @ potential
    state[6] = 0.3 - plus[0] @ potential
    state[7] = 0.4 + 0.08 * np.cos(2 * np.pi * (x + y))

    def magnetic(q):
        B = np.array([average[0] @ q[5], average[1] @ q[6], q[7]])
        energy = (average[0] @ (q[5] ** 2) + average[1] @ (q[6] ** 2) + q[7] ** 2) / 2
        J = np.array([minus[1] @ q[7], -minus[0] @ q[7], minus[0] @ q[6] - minus[1] @ q[5]])
        return B, energy, J

    _, magnetic_energy, _ = magnetic(state)
    pressure = 1 + 0.03 * np.cos(2 * np.pi * (x - y))
    state[4] = pressure / (gamma - 1) + state[0] * np.sum(velocity**2, axis=0) / 2 + magnetic_energy
    initial = state.copy()

    def divergence(q):
        return plus[0] @ q[5] + plus[1] @ q[6]

    def rhs(q):
        B, magnetic_energy, J = magnetic(q)
        u = q[1:4] / q[0]
        pressure = (gamma - 1) * (q[4] - np.sum(q[1:4] ** 2, axis=0) / (2 * q[0]) - magnetic_energy)
        Jc = np.array([M[a] @ J[a] for a in range(3)])
        ideal_c = -np.cross(u.T, B.T).T
        hall_c = hall * np.cross(Jc.T, B.T).T / q[0]
        ideal = np.array([M[a].T @ ideal_c[a] for a in range(3)])
        electric_hall = np.array([M[a].T @ hall_c[a] for a in range(3)])
        electric = ideal + electric_hall + ohmic * J
        Ec = np.array([M[a] @ electric[a] for a in range(3)])
        derivative = np.zeros_like(q)
        poynting = np.cross(Ec.T, B.T).T
        for axis in range(2):
            flux = np.zeros((5, n))
            flux[0] = q[axis + 1]
            for a in range(3):
                flux[a + 1] = q[a + 1] * u[axis] - B[a] * B[axis]
                if a == axis:
                    flux[a + 1] += pressure + np.sum(B**2, axis=0) / 2
            flux[4] = (q[4] - magnetic_energy + pressure) * u[axis] + poynting[axis]
            derivative[:5] -= flux @ center[axis].T
        derivative[5] = -plus[1] @ electric[2]
        derivative[6] = plus[0] @ electric[2]
        derivative[7] = plus[1] @ electric[0] - plus[0] @ electric[1]
        record = dict(
            hall_work=float(np.sum(J * electric_hall) / n),
            ohmic_heat=float(ohmic * np.sum(J * J) / n),
            field_work=float(np.sum(J * electric) / n),
            magnetic_rate=float(np.sum(q[5:] * derivative[5:]) / n),
            pressure_minimum=float(pressure.min()),
            divergence=float(abs(divergence(q)).max()),
            divergence_rate=float(abs(divergence(derivative)).max()),
            rhs=derivative.copy(),
        )
        return derivative, record

    records = []
    for _ in range(steps):
        k1, c1 = rhs(state)
        k2, c2 = rhs(state + dt * k1 / 2)
        k3, c3 = rhs(state + dt * k2 / 2)
        k4, c4 = rhs(state + dt * k3)
        old_magnetic = float(np.sum(state[5:] ** 2) / (2 * n))
        state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        new_magnetic = float(np.sum(state[5:] ** 2) / (2 * n))
        weighted_work = (
            dt
            * (c1["field_work"] + 2 * c2["field_work"] + 2 * c3["field_work"] + c4["field_work"])
            / 6
        )
        records.append(
            dict(
                stages=(c1, c2, c3, c4),
                divergence_after=float(abs(divergence(state)).max()),
                magnetic_energy_change=new_magnetic - old_magnetic,
                integrated_work=weighted_work,
                inventory_defect=float(abs(np.mean(state - initial, axis=1)).max()),
            )
        )
    return initial.reshape((8, *shape)), state.reshape((8, *shape)), records
