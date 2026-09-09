"""Independent finite-volume reference for the four periodic composite IMEX witnesses.

This fixed problem uses conserved U, first-order upwind transport (0.7, -0.3), and
backward diffusion with arithmetic face coefficients. Covered coarse cells are restricted
from fine values; fine exterior values use monotonized-central coarse reconstruction.
Only complete coarse/fine interface face integrals are matched. No PoPS implementation,
solver, compiled artifact or runtime data enters this reference.
"""

import numpy as np

DT = 0.00025
KINDS = ("constant", "variable", "diagonal", "nonlinear_accumulation")


def periodic_cell_averages(n):
    phase = 2 * np.pi * (np.arange(n) + 0.5) / n
    sinc = np.sinc(1 / n)
    return 1 - 0.3 * sinc * np.cos(phase)[None, :] * (1 + 0.1 * sinc * np.cos(phase)[:, None])


def constant_uniform_fourier(n):
    """Independent diagonalization of all four upwind/BE steps on the uniform mesh."""
    phase = 2 * np.pi * np.fft.fftfreq(n)
    advection = (-0.7 * n * (1 - np.exp(-1j * phase))[None, :]
                 + 0.3 * n * (np.exp(1j * phase) - 1)[:, None])
    laplacian = -0.4 * n * n * (np.sin(phase / 2)[None, :] ** 2
                              + np.sin(phase / 2)[:, None] ** 2)
    multiplier = ((1 + DT * advection) / (1 - DT * laplacian)) ** 4
    return np.fft.ifft2(np.fft.fft2(periodic_cell_averages(n)) * multiplier).real


def _mc_slope(lower, center, upper):
    left, right = center - lower, upper - center
    centered = 0.5 * (upper - lower)
    magnitude = np.minimum(np.abs(centered), np.minimum(2 * np.abs(left), 2 * np.abs(right)))
    return np.where(left * right > 0, np.copysign(magnitude, centered), 0.0)


def _constitutive(u, kind):
    # Bounds are assertions about these positive profiles, never clipping or modified residuals.
    assert np.isfinite(u).all() and float(u.min()) > 0 and float(u.max()) < 1.4
    w = (np.sqrt(1 + 4 * u) - 1) / 2 if kind == "nonlinear_accumulation" else u
    if kind == "variable":
        ax = ay = 0.1 * (1 + 0.2 * u)
    elif kind == "diagonal":
        ax, ay = 0.1 * (1 + 0.2 * u), 0.07 * (1 + 0.1 * u)
    else:
        ax = ay = np.full_like(u, 0.1)
    assert min(float(ax.min()), float(ay.min())) > 0
    assert max(float(ax.max()), float(ay.max())) < 0.2
    return w, ax, ay


class _Operator:
    def __init__(self, n, kind, *, composite):
        assert type(n) is int and n >= 16 and n % 16 == 0 and kind in KINDS
        self.n, self.kind, self.composite = n, kind, composite
        self.lo, self.hi = 3 * n // 16, 13 * n // 16
        self.coarse_mask = np.ones((n, n), dtype=bool)
        self.coarse_mask[:, self.lo:self.hi] = False
        self.fine_mask = (~self.coarse_mask).repeat(2, 0).repeat(2, 1)
        self.nc = int(self.coarse_mask.sum())
        self.offset = np.tile(np.array([-0.25, 0.25]), n)
        counts = (np.r_[np.full(self.nc, n), np.full(int(self.fine_mask.sum()), 2 * n)]
                  if composite else np.full(n * n, n))
        self.volumes = counts.astype(float) ** -2
        # 0.2 bounds diffusivity only in this Jacobi preconditioner. The nonlinear residual
        # always evaluates the actual constitutive law and its current arithmetic face means.
        self.diagonal = 1 + 4 * DT * 0.2 * counts * counts

    def initial(self):
        if not self.composite:
            return periodic_cell_averages(self.n).ravel()
        return np.r_[periodic_cell_averages(self.n)[self.coarse_mask],
                     periodic_cell_averages(2 * self.n)[self.fine_mask]]

    def grids(self, u):
        n = self.n
        if not self.composite:
            return (u.reshape(n, n),)
        coarse = np.zeros((n, n))
        coarse[self.coarse_mask] = u[:self.nc]
        fine = np.zeros((2 * n, 2 * n))
        fine[self.fine_mask] = u[self.nc:]
        restricted = fine.reshape(n, 2, n, 2).mean(axis=(1, 3))
        coarse[~self.coarse_mask] = restricted[~self.coarse_mask]
        sx = _mc_slope(np.roll(coarse, 1, axis=1), coarse, np.roll(coarse, -1, axis=1))
        sy = _mc_slope(np.roll(coarse, 1, axis=0), coarse, np.roll(coarse, -1, axis=0))
        prolong = coarse.repeat(2, 0).repeat(2, 1)
        prolong += sx.repeat(2, 0).repeat(2, 1) * self.offset[None, :]
        prolong += sy.repeat(2, 0).repeat(2, 1) * self.offset[:, None]
        fine[~self.fine_mask] = prolong[~self.fine_mask]
        return coarse, fine

    def apply(self, u, *, diffusion):
        faces = []
        for level, grid in enumerate(self.grids(u)):
            n = self.n * 2 ** level
            if diffusion:
                w, ax, ay = _constitutive(grid, self.kind)
                fx = 0.5 * (ax + np.roll(ax, 1, axis=1)) * n * (w - np.roll(w, 1, axis=1))
                fy = 0.5 * (ay + np.roll(ay, 1, axis=0)) * n * (w - np.roll(w, 1, axis=0))
            else:
                fx, fy = 0.7 * np.roll(grid, 1, axis=1), -0.3 * grid
            faces.append((fx, fy))
        if self.composite:
            for x in (self.lo, self.hi):
                faces[0][0][:, x] = 0.5 * (faces[1][0][::2, 2 * x]
                                          + faces[1][0][1::2, 2 * x])
        rates = [(self.n * 2 ** level) * (1 if diffusion else -1)
                 * (np.roll(fx, -1, axis=1) - fx + np.roll(fy, -1, axis=0) - fy)
                 for level, (fx, fy) in enumerate(faces)]
        return (np.r_[rates[0][self.coarse_mask], rates[1][self.fine_mask]]
                if self.composite else rates[0].ravel())

    def solve(self, rhs):
        u = rhs.copy()
        for iteration in range(2000):
            residual = rhs - u + DT * self.apply(u, diffusion=True)
            norm = float(np.max(np.abs(residual)))
            if norm < 2e-14:
                return u, {"iterations": iteration, "residual_linf": norm,
                           "residual_l2": float(np.sqrt(self.volumes @ (residual * residual)))}
            u += 0.8 * residual / self.diagonal
        raise AssertionError("independent IMEX solve did not converge: %s/N%d/%g"
                             % (self.kind, self.n, norm))

    def trajectory(self):
        initial = self.initial()
        u = initial.copy()
        checks, reports = [], []
        for diffusion in (False, True):
            constant = float(np.max(np.abs(self.apply(np.ones_like(u), diffusion=diffusion))))
            mass_rate = float(self.volumes @ self.apply(u, diffusion=diffusion))
            assert constant == 0 and abs(mass_rate) < 1e-12
            checks.append({"diffusion": diffusion, "constant_defect": constant,
                           "mass_rate": mass_rate})
        for _ in range(4):
            rhs = u + DT * self.apply(u, diffusion=False)
            u, report = self.solve(rhs)
            drift = float(self.volumes @ (u - initial))
            assert abs(drift) < 5e-11
            report.update(mass_drift=drift, minimum=float(u.min()), maximum=float(u.max()))
            reports.append(report)
        return u, {"solves": reports, "operator_checks": checks}


def reference_pair(n, kind):
    """Return independent active composite and complete uniform final arrays plus diagnostics."""
    composite = _Operator(n, kind, composite=True)
    uniform = _Operator(2 * n, kind, composite=False)
    active, composite_checks = composite.trajectory()
    uniform_active, uniform_checks = uniform.trajectory()
    fine = uniform_active.reshape(2 * n, 2 * n)
    coarse = fine.reshape(n, 2, n, 2).mean(axis=(1, 3))
    difference = active - np.r_[coarse[composite.coarse_mask], fine[composite.fine_mask]]
    l2 = float(np.sqrt(composite.volumes @ (difference * difference)))
    fourier_error = None
    if kind == "constant":
        fourier_error = float(np.max(np.abs(fine - constant_uniform_fourier(2 * n))))
        assert fourier_error < 2e-11
    return {"composite_final": active, "uniform_final": fine,
            "coarse_mask": composite.coarse_mask, "fine_mask": composite.fine_mask,
            "diagnostics": {"n": n, "kind": kind, "l2": l2,
                            "composite": composite_checks, "uniform": uniform_checks,
                            "constant_uniform_fourier_linf": fourier_error}}
