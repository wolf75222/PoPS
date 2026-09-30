"""Independent discrete mathematics; this file neither imports nor runs PoPS.

The manufactured relation has a product of different unknowns, cubic reaction,
fixed spatial captures, and nonsymmetric cross diffusion on a periodic grid.
The dense Newton reference is an oracle, not reception of native Newton/GMRES.
"""

from dataclasses import dataclass, replace

import numpy as np
import pytest


@dataclass(frozen=True)
class Witness:
    laplacian: np.ndarray
    diffusion: np.ndarray
    linear: np.ndarray
    cubic: np.ndarray
    product: np.ndarray
    square: np.ndarray
    capture: np.ndarray
    exact: np.ndarray
    rhs: np.ndarray

    def reaction(self, values):
        return (
            self.linear[:, None] * values
            + self.cubic[:, None] * values**3
            + self.capture * values * (self.product @ values)
            + self.square @ (values**2)
        )

    def residual(self, values):
        return self.reaction(values) - self.diffusion @ values @ self.laplacian.T - self.rhs

    def jacobian(self, values):
        width, cells = values.shape
        jac = -np.kron(self.diffusion, self.laplacian)
        for i in range(width):
            for j in range(width):
                pointwise = (
                    self.capture[i] * values[i] * self.product[i, j]
                    + 2 * self.square[i, j] * values[j]
                )
                if i == j:
                    pointwise = pointwise + (
                        self.linear[i]
                        + 3 * self.cubic[i] * values[i] ** 2
                        + self.capture[i] * (self.product @ values)[i]
                    )
                a, b = i * cells, j * cells
                jac[a : a + cells, b : b + cells] += np.diag(pointwise)
        return jac

    def permuted(self, order):
        return replace(
            self,
            diffusion=self.diffusion[np.ix_(order, order)],
            linear=self.linear[order],
            cubic=self.cubic[order],
            product=self.product[np.ix_(order, order)],
            square=self.square[np.ix_(order, order)],
            capture=self.capture[order],
            exact=self.exact[order],
            rhs=self.rhs[order],
        )


def witness(width=5):
    nx, ny = 5, 4
    dx, dy = 1 / nx, 1.7 / ny
    lap = np.zeros((nx * ny, nx * ny))
    for y in range(ny):
        for x in range(nx):
            row = y * nx + x
            for xx, yy, weight in (
                ((x - 1) % nx, y, dx**-2),
                ((x + 1) % nx, y, dx**-2),
                (x, (y - 1) % ny, dy**-2),
                (x, (y + 1) % ny, dy**-2),
            ):
                lap[row, yy * nx + xx] += weight
                lap[row, row] -= weight
    x, y = np.meshgrid((np.arange(nx) + 0.5) / nx, (np.arange(ny) + 0.5) / ny)
    exact = np.array(
        [
            0.15 + 0.055 * i + 0.035 * np.sin(2 * np.pi * x + 0.3 * i)
            + 0.025 * np.cos(2 * np.pi * y - 0.2 * i)
            for i in range(width)
        ]
    ).reshape(width, -1)
    capture = np.array(
        [1.1 + 0.2 * i + 0.15 * np.cos(2 * np.pi * (x + y)) for i in range(width)]
    ).reshape(width, -1)
    diffusion = np.diag(np.linspace(0.025, 0.041, width))
    product, square = np.zeros((width, width)), np.zeros((width, width))
    for i in range(width):
        diffusion[i, (i + 1) % width] = 0.003 + i * 0.0003
        product[i, (i + 1) % width] = 0.04 + i * 0.005
        square[i, (i - 1) % width] = 0.025 + i * 0.003
    data = Witness(
        lap, diffusion, np.linspace(1.3, 2.1, width), np.linspace(0.6, 1.0, width),
        product, square, capture, exact, np.zeros_like(exact),
    )
    return replace(data, rhs=data.residual(exact))


def dense_newton(data, seed, max_updates=12):
    """Explicit dense analytical Newton with Armijo; no inverse elimination."""
    values = seed.copy()
    norms = []
    for update in range(max_updates + 1):
        residual = data.residual(values)
        norm = np.linalg.norm(residual.ravel(), ord=np.inf)
        norms.append(norm)
        if norm < 2e-12:
            return values, norms
        if update == max_updates:
            raise RuntimeError("reference exhausted updates before original residual acceptance")
        direction = np.linalg.solve(data.jacobian(values), -residual.ravel()).reshape(values.shape)
        step = 1.0
        while step >= 2**-12:
            candidate = values + step * direction
            if np.linalg.norm(data.residual(candidate).ravel(), ord=np.inf) <= (1 - 1e-4 * step) * norm:
                values = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("reference line search refused original residual")
    raise AssertionError("unreachable")


@pytest.mark.parametrize("width", (2, 3, 5))
def test_original_product_reaction_diffusion_from_distinct_seeds(width):
    data = witness(width)
    assert not np.array_equal(data.diffusion, data.diffusion.T)
    np.testing.assert_allclose(data.laplacian @ np.ones(20), 0, atol=1e-13)
    np.testing.assert_allclose(data.residual(data.exact), 0, atol=1e-14)
    for seed in (0.7 * data.exact + 0.1, -0.2 - 0.3 * data.exact):
        saved_seed, saved_rhs, saved_capture = seed.copy(), data.rhs.copy(), data.capture.copy()
        actual, norms = dense_newton(data, seed)
        assert len(norms) > 2
        assert all(a > b for a, b in zip(norms, norms[1:], strict=False))
        np.testing.assert_allclose(actual, data.exact, atol=3e-12, rtol=0)
        assert np.max(np.abs(data.residual(actual))) < 2e-12
        np.testing.assert_array_equal(seed, saved_seed)
        np.testing.assert_array_equal(data.rhs, saved_rhs)
        np.testing.assert_array_equal(data.capture, saved_capture)


def test_explicit_central_difference_of_original_relation():
    data = witness()
    values = data.exact + 0.2
    direction = np.sin(np.arange(values.size) + 0.3).reshape(values.shape)
    exact_jvp = (data.jacobian(values) @ direction.ravel()).reshape(values.shape)
    errors = []
    for step in (1e-3, 5e-4):
        central = (data.residual(values + step * direction)
                   - data.residual(values - step * direction)) / (2 * step)
        expected_error = step**2 * data.cubic[:, None] * direction**3
        np.testing.assert_allclose(central - exact_jvp, expected_error, atol=3e-12, rtol=2e-5)
        errors.append(np.max(np.abs(central - exact_jvp)))
    assert abs(errors[0] / errors[1] - 4) < 1e-4
    step = 1e-7  # Explicit candidate policy, not an assumed universal derivative.
    central = (data.residual(values + step * direction)
               - data.residual(values - step * direction)) / (2 * step)
    np.testing.assert_allclose(central, exact_jvp, atol=2e-8, rtol=2e-8)
    forward = (data.residual(values + 1e-3 * direction) - data.residual(values)) / 1e-3
    assert np.max(np.abs(forward - exact_jvp)) > 1e-4


def test_permutation_routes_every_unknown_capture_seed_and_equation():
    data = witness()
    order = np.array([4, 1, 3, 0, 2])
    reordered = data.permuted(order)
    seed = -0.2 - 0.3 * data.exact
    np.testing.assert_allclose(reordered.residual(seed[order]), data.residual(seed)[order], atol=1e-14)
    values, _ = dense_newton(reordered, seed[order])
    np.testing.assert_allclose(values[np.argsort(order)], data.exact, atol=3e-12, rtol=0)
    indices = np.arange(seed.size).reshape(seed.shape)[order].ravel()
    np.testing.assert_allclose(
        reordered.jacobian(seed[order]), data.jacobian(seed)[np.ix_(indices, indices)], atol=1e-14,
    )


def test_original_residual_detects_linearized_reaction_and_wrong_routes():
    data = witness()
    seed = -0.2 - 0.3 * data.exact
    linear_matrix = np.kron(np.diag(data.linear), np.eye(20)) - np.kron(
        data.diffusion, data.laplacian,
    )
    frozen_nonlinearity = data.reaction(seed) - data.linear[:, None] * seed
    wrong = np.linalg.solve(linear_matrix, (data.rhs - frozen_nonlinearity).ravel()).reshape(seed.shape)
    assert np.max(np.abs(data.residual(wrong))) > 0.05
    assert np.max(np.abs(replace(data, diffusion=data.diffusion.T).residual(data.exact))) > 0.003
    assert np.max(np.abs(replace(data, capture=data.capture[::-1]).residual(data.exact))) > 0.001
    assert np.max(np.abs(replace(data, rhs=data.rhs[::-1]).residual(data.exact))) > 0.2
    damaged = data.exact.copy()
    damaged[-1] += 0.1
    # A component-zero-only observation can miss a tail defect at individual cells;
    # acceptance checks the original residual of every equation.
    assert np.max(np.abs(data.residual(damaged)[-1])) > 0.2


def test_exhausted_reference_does_not_publish_seed_or_fixed_data():
    data = witness()
    seed = -0.2 - 0.3 * data.exact
    old_seed, old_rhs, old_capture = seed.copy(), data.rhs.copy(), data.capture.copy()
    with pytest.raises(RuntimeError, match="exhausted updates"):
        dense_newton(data, seed, max_updates=0)
    np.testing.assert_array_equal(seed, old_seed)
    np.testing.assert_array_equal(data.rhs, old_rhs)
    np.testing.assert_array_equal(data.capture, old_capture)
