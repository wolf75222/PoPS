"""Independent discrete controls for the G09 Cartesian cuts. No PoPS import."""

from itertools import product

import numpy as np


CELLS = 8
SHIFT = CELLS // 4
TWIST_DT = 1 / 16
EULER_DT = 1 / 512
GAMMA = 1.4
P_INF = 0.2


def twisted_initial():
    values = np.ones((CELLS, CELLS), dtype=np.float64)
    values[0, -1] = 1.5
    values[0, 2] = 1.25
    return values


def twisted_step(values, *, shift=SHIFT):
    """Upwind velocity (−1, 0), array layout (x, y), one SSPRK2 step."""

    def rate(state):
        incoming = np.roll(state, -1, axis=0)
        incoming[-1] = np.roll(state[0], shift)
        return CELLS * (incoming - state)

    stage = values + TWIST_DT * rate(values)
    return 0.5 * (values + stage + TWIST_DT * rate(stage))


def euler_initial():
    """Cartesian array layout (component, y, x), quiet near all vertices."""
    size = CELLS
    rho = np.ones((size, size), dtype=np.float64)
    vx, vy = np.zeros_like(rho), np.zeros_like(rho)
    rho[0, 2], vx[0, 2], vy[0, 2] = 1.25, 0.1875, -0.3125
    rho[3, -1], vx[3, -1], vy[3, -1] = 0.875, 0.25, -0.09375
    rho[-1, 4], vx[-1, 4], vy[-1, 4] = 1.125, -0.09375, 0.21875
    energy = (1 + GAMMA * P_INF) / (GAMMA - 1) + rho * (vx**2 + vy**2) / 2
    return np.stack((rho, rho * vx, rho * vy, energy))


def outside(grid, x, y):
    return [
        face for face, outside_axis in (
            ("x_min", x < 0), ("x_max", x >= grid),
            ("y_min", y < 0), ("y_max", y >= grid),
        )
        if outside_axis
    ]


def quarter_index(n, x, y, face):
    """One gather. Rotation is the quarter-turn from sampled to ghost components."""
    if face == "x_min":
        return n - 1 - y, n + x, -1
    if face == "x_max":
        return n - 1 - y, x - n, -1
    if face == "y_min":
        return n + y, n - 1 - x, 1
    if face == "y_max":
        return y - n, n - 1 - x, 1
    raise ValueError(face)


def quarter_images(n, x, y):
    """Enumerate every authorized corner order, including the action on vector components."""
    faces = outside(n, x, y)
    if not faces:
        return {(x, y, 0)}
    images = set()
    for face in faces:
        src_x, src_y, rotation = quarter_index(n, x, y, face)
        assert len(outside(n, src_x, src_y)) < len(faces), (face, x, y, src_x, src_y)
        for fin_x, fin_y, after in quarter_images(n, src_x, src_y):
            images.add((fin_x, fin_y, (rotation + after) % 4))
    return images


def sample_quarter(state, x, y, *, transport):
    (source_x, source_y, action), = quarter_images(state.shape[-1], x, y)
    rho, mx, my, energy = state[:, source_y, source_x]
    if transport == "inverse":
        action = -action % 4
    if transport == "identity":
        action = 0
    mx, my = ((mx, my), (-my, mx), (-mx, -my), (my, -mx))[action]
    return np.array([rho, mx, my, energy])


def _normal_flux(state, axis):
    rho, mx, my, energy = state
    vx, vy = mx / rho, my / rho
    pressure = (GAMMA - 1) * (energy - (mx * vx + my * vy) / 2) - GAMMA * P_INF
    normal_speed = (vx, vy)[axis]
    momentum_x = mx * normal_speed + (pressure if axis == 0 else 0)
    momentum_y = my * normal_speed + (pressure if axis == 1 else 0)
    flux = np.array([rho * normal_speed, momentum_x, momentum_y, (energy + pressure) * normal_speed])
    radius = abs(normal_speed) + np.sqrt(GAMMA * (pressure + P_INF) / rho)
    return flux, radius


def _rusanov(left, right, axis):
    f_left, speed_left = _normal_flux(left, axis)
    f_right, speed_right = _normal_flux(right, axis)
    return 0.5 * (f_left + f_right) - 0.5 * max(speed_left, speed_right) * (right - left)


def euler_step(state, *, boundary="quarter", transport="direct"):
    if boundary not in {"quarter", "periodic", "slip"}:
        raise ValueError(boundary)
    if transport not in {"direct", "inverse", "identity"}:
        raise ValueError(transport)

    def sample(values, x, y):
        n = values.shape[-1]
        if boundary == "quarter":
            return sample_quarter(values, x, y, transport=transport)
        if boundary == "periodic":
            return values[:, y % n, x % n]
        reflect = lambda j: -1 - j if j < 0 else 2 * n - 1 - j if j >= n else j
        image = values[:, reflect(y), reflect(x)].copy()
        if not 0 <= x < n:
            image[1] *= -1
        if not 0 <= y < n:
            image[2] *= -1
        return image

    def rate(values):
        n = values.shape[-1]
        result = np.empty_like(values)
        for x, y in product(range(n), repeat=2):
            local = values[:, y, x]
            west = _rusanov(sample(values, x - 1, y), local, axis=0)
            east = _rusanov(local, sample(values, x + 1, y), axis=0)
            south = _rusanov(sample(values, x, y - 1), local, axis=1)
            north = _rusanov(local, sample(values, x, y + 1), axis=1)
            result[:, y, x] = n * ((west - east) + (south - north))
        return result

    stage = state + EULER_DT * rate(state)
    return 0.5 * (state + stage + EULER_DT * rate(stage))


def self_check():
    corners = [
        (x, y) for x, y in product(range(-2, CELLS + 2), repeat=2)
        if len(outside(CELLS, x, y)) == 2
    ]
    assert all(len(quarter_images(CELLS, *corner)) == 1 for corner in corners)
    (image_x, image_y, action), = quarter_images(CELLS, CELLS, CELLS - 1 - 2)
    assert (image_x, image_y, action) == (2, 0, 3)
    state = euler_initial()
    rho, mx, my, energy = state[:, 0, 2]
    np.testing.assert_array_equal(
        sample_quarter(state, CELLS, CELLS - 1 - 2, transport="direct"),
        (rho, my, -mx, energy),
    )

    twisted = twisted_initial()
    shifted = twisted_step(twisted)
    np.testing.assert_equal(shifted.sum(), twisted.sum())
    np.testing.assert_equal(shifted[-1, 1], 1.125)
    shift_gap = float(np.max(abs(shifted - twisted_step(twisted, shift=0))))
    assert shift_gap >= 0.125

    quarter = euler_step(state)
    mass_energy_defect = abs(quarter[[0, 3]].sum(axis=(1, 2)) -
                             state[[0, 3]].sum(axis=(1, 2)))
    assert np.max(mass_energy_defect) <= 4 * np.spacing(state[3].sum())
    gaps = {
        "untwisted": float(np.max(abs(quarter - euler_step(state, boundary="periodic")))),
        "slip": float(np.max(abs(quarter - euler_step(state, boundary="slip")))),
        "identity_components": float(np.max(abs(quarter - euler_step(state, transport="identity")))),
        "inverse_components": float(np.max(abs(quarter - euler_step(state, transport="inverse")))),
    }
    assert min(gaps.values()) > 1e-3
    print("twisted versus ordinary periodic gap", shift_gap)
    print("quarter-turn versus controls", gaps)
    print("scalar invariants mass/energy", mass_energy_defect.tolist())
    print("confluent corner cells, depths 1 and 2", len(corners))


if __name__ == "__main__":
    self_check()
