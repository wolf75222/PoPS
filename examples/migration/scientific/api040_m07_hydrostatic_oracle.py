"""Independent NumPy Audusse/FE/HLL reference; never a PoPS runtime substitute."""
from math import erf, sqrt, pi
import numpy as np

GRAVITY = 1.0
RESOLUTIONS = (40, 80, 160)
T_END = 1.0
EQUILIBRIUM_TOLERANCE = 1.e-12
ORDERS = (("h", "q", "z"), ("z", "h", "q"))


def equilibrium_averages(edges):
    """True averages, including analytic-equilibrium ghost cells beyond the ends."""
    edges = np.asarray(edges, dtype=float)
    primitive = np.array([erf(sqrt(50.) * x) for x in edges])
    z = .2 * sqrt(pi) / (2. * sqrt(50.)) * np.diff(primitive) / np.diff(edges)
    return np.array((1. - z, np.zeros_like(z), z))


def initial_averages(n):
    return equilibrium_averages(np.linspace(-1., 1., n + 1))


def face_contributions(left, right, *, source_factor=1.):
    """Canonical (h,q,z); return flux entering each incident-cell balance.

    source_factor=0 omits the topographic correction; 2 counts it twice.
    These deliberately wrong variants are rejection oracles only.
    """
    left, right = np.asarray(left), np.asarray(right)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite shallow-water state")
    if np.any(left[0] <= 0.) or np.any(right[0] <= 0.):
        raise ValueError("this frozen fully-wet witness requires positive cell depth")
    ul, ur = left[1] / left[0], right[1] / right[0]
    zstar = np.maximum(left[2], right[2])
    hl = np.maximum(0., left[0] + left[2] - zstar)
    hr = np.maximum(0., right[0] + right[2] - zstar)
    ql, qr = hl * ul, hr * ur
    fl = np.array((ql, ql * ul + .5 * GRAVITY * hl**2, np.zeros_like(hl)))
    fr = np.array((qr, qr * ur + .5 * GRAVITY * hr**2, np.zeros_like(hr)))
    sl = np.minimum(0., np.minimum(ul - np.sqrt(GRAVITY * hl),
                                  ur - np.sqrt(GRAVITY * hr)))
    sr = np.maximum(0., np.maximum(ul + np.sqrt(GRAVITY * hl),
                                  ur + np.sqrt(GRAVITY * hr)))
    jump = np.array((hr-hl, qr-ql, np.zeros_like(hl)))
    numerator = sr * fl - sl * fr + sl * sr * jump
    flux = np.divide(numerator, sr-sl, out=np.zeros_like(numerator), where=sr > sl)
    lower, upper = flux.copy(), flux.copy()
    lower[1] += source_factor * .5 * GRAVITY * (left[0]**2 - hl**2)
    upper[1] += source_factor * .5 * GRAVITY * (right[0]**2 - hr**2)
    return lower, upper, np.maximum(-sl, sr)


def rhs(state, *, source_factor=1.):
    n = state.shape[1]
    dx = 2. / n
    ghosts = equilibrium_averages(np.array([-1.-dx, -1., 1., 1.+dx]))
    padded = np.column_stack((ghosts[:, 0], state, ghosts[:, -1]))
    lower, upper, speed = face_contributions(padded[:, :-1], padded[:, 1:],
                                            source_factor=source_factor)
    return -(lower[:, 1:] - upper[:, :-1]) / dx, speed


def trajectory(n, *, order=ORDERS[0], source_factor=1.):
    if n not in RESOLUTIONS or set(order) != set(ORDERS[0]) or len(order) != 3:
        raise ValueError("unknown frozen M07 reception variant")
    state = initial_averages(n)
    before = state.copy()
    steps, dt = 5*n, 1./(5*n)
    max_courant = 0.
    for _ in range(steps):
        derivative, speed = rhs(state, source_factor=source_factor)
        max_courant = max(max_courant, float(dt/(2./n)*np.max(speed)))
        state = state + dt * derivative
    error = max(float(np.max(np.abs(state[0] + state[2] - 1.))),
                float(np.max(np.abs(state[1]))),
                float(np.max(np.abs(state[2] - before[2]))))
    permutation = [ORDERS[0].index(name) for name in order]
    return state[permutation], {"error": error, "max_courant": max_courant,
                                "steps": steps, "dt": dt, "time": steps*dt}
