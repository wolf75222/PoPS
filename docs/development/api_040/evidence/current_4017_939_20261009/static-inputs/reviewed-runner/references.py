"""Independent cell-array finite-volume references; imports no PoPS implementation."""
import numpy as np


def centers(n):
    return np.meshgrid((np.arange(n) + 0.5) / n, (np.arange(n) + 0.5) / n, indexing="xy")


def cell_average(function, n):
    x, y = centers(n)
    nodes, weights = np.polynomial.legendre.leggauss(5)
    result = np.zeros_like(x)
    for a, wa in zip(nodes, weights, strict=True):
        for b, wb in zip(nodes, weights, strict=True):
            result += 0.25 * wa * wb * function(x + a / (2 * n), y + b / (2 * n))
    return result


def smooth(x, y):
    return 1 + x + y + 16 * x**2 * (1 - x)**2 * y**2 * (1 - y)**2


def smooth_source(x, y):
    # -div(diag(1+x/4, 2+y/2) grad(smooth)), expanded analytically.
    px, py = x**2 * (1 - x)**2, y**2 * (1 - y)**2
    dx, dy = 2 * x - 6 * x**2 + 4 * x**3, 2 * y - 6 * y**2 + 4 * y**3
    ddx, ddy = 2 - 12 * x + 12 * x**2, 2 - 12 * y + 12 * y**2
    return -0.75 - 16 * (((1 + 0.25 * x) * ddx + 0.25 * dx) * py
                         + ((2 + 0.5 * y) * ddy + 0.5 * dy) * px)


def inputs(kind, n=16):
    x, y = centers(n)
    if kind == "variable":
        initial = cell_average(lambda a, b: 2 + np.sin(2 * np.pi * a) * np.sin(2 * np.pi * b), n)
        s = np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
        a = 1 + 0.25 * np.sin(2 * np.pi * x)
        divergence = -8 * np.pi**2 * a * s + np.pi**2 * np.cos(2 * np.pi * x)**2 * np.sin(2 * np.pi * y)
        return initial, {"diffusivity": a, "source_factor": -1 - divergence / (2 + s)}
    if kind not in {"diagonal_linear", "diagonal_smooth"}:
        raise ValueError(kind)
    is_smooth = kind == "diagonal_smooth"
    initial = cell_average(smooth if is_smooth else lambda a, b: 1 + a + b, n)
    fields = {"a_x": 1 + 0.25 * x, "a_y": 2 + 0.5 * y,
              "forcing": cell_average(smooth_source, n) if is_smooth else np.full_like(x, -0.75)}
    return initial, fields


def rate(kind, state, fields):
    """Divergence of shared face fluxes plus the independently declared source.

    A face coefficient is the arithmetic mean of its adjacent cell inputs.
    Physical x-face values use the authored linear trace 1+x+y at half-cell distance;
    physical y fluxes use the authored outward conormals -2 and 2.5.
    """
    n = state.shape[0]
    assert state.shape == (n, n)
    if kind == "variable":
        coefficient = fields["diffusivity"]
        divergence = np.zeros_like(state)
        for axis in (0, 1):
            right = np.roll(state, -1, axis)
            shared_flux = 0.5 * (coefficient + np.roll(coefficient, -1, axis)) * (right - state) * n
            divergence += (shared_flux - np.roll(shared_flux, 1, axis)) * n
        return divergence + fields["source_factor"] * state
    if kind not in {"diagonal_linear", "diagonal_smooth"}:
        raise ValueError(kind)
    ax, ay = fields["a_x"], fields["a_y"]
    x_flux = np.empty((n, n + 1))
    y_flux = np.empty((n + 1, n))
    x_flux[:, 1:n] = 0.5 * (ax[:, :-1] + ax[:, 1:]) * (state[:, 1:] - state[:, :-1]) * n
    y_flux[1:n, :] = 0.5 * (ay[:-1, :] + ay[1:, :]) * (state[1:, :] - state[:-1, :]) * n
    _, y = centers(n)
    x_flux[:, 0] = (1.5 * ax[:, 0] - 0.5 * ax[:, 1]) * 2 * (state[:, 0] - (1 + y[:, 0])) * n
    x_flux[:, n] = (1.5 * ax[:, -1] - 0.5 * ax[:, -2]) * 2 * ((2 + y[:, -1]) - state[:, -1]) * n
    y_flux[0, :] = 2.0
    y_flux[n, :] = 2.5
    return (x_flux[:, 1:] - x_flux[:, :-1] + y_flux[1:, :] - y_flux[:-1, :]) * n + fields["forcing"]


def authored_update(kind, method, initial, fields, dt):
    first = initial + dt * rate(kind, initial, fields)
    if method == "euler":
        return first
    if method == "ssprk2":
        return 0.5 * initial + 0.5 * (first + dt * rate(kind, first, fields))
    raise ValueError(method)
