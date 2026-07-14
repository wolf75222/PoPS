#!/usr/bin/env python3
"""Numerical oracle for the continuous-symbol FFT Poisson provider.

The final authoring selector is :class:`pops.solvers.elliptic.FFT`; this test derives every native
route token from that typed descriptor and exercises the native executor seam directly.  It keeps
the unique distinction between ``FFT()`` (the discrete five-point symbol) and
``FFT(spectral=True)`` (the continuous ``-(kx^2 + ky^2)`` symbol): a sinusoidal manufactured
solution is machine-accurate only on the spectral route.  The test also guards mean-zero gauge,
periodic-boundary requirements, and halo publication before a source reads ``grad(phi)``.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
from pops.solvers.elliptic import FFT, GeometricMG
import sys

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Dirichlet, Periodic
from pops.mesh.geometry import Disc
from pops.runtime._system import System

fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn()
        return ""
    except Exception as exc:  # noqa: BLE001 -- the error text is the contract under test
        return str(exc)


def _solver_token(descriptor):
    token = getattr(descriptor, "scheme", None)
    if not isinstance(token, str) or not token:
        raise TypeError("elliptic descriptor must expose its resolved native scheme")
    return token


def _model():
    return engine.Model(
        state=engine.FluidState("isothermal", cs2=0.5),
        transport=engine.IsothermalFlux(),
        source=engine.PotentialForce(charge=1.0),
        elliptic=engine.ChargeDensity(charge=1.0),
    )


def _system(n, *, periodic=True):
    sim = System(n=n, L=1.0, periodic=periodic)
    sim.add_equation(
        "ions",
        _model(),
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.Explicit(),
    )
    return sim


def solve_phi(n, solver, eps=1e-3):
    """Solve a zero-mean sinusoid and compare with the continuous analytic solution."""
    sim = _system(n)
    sim.set_poisson(
        rhs="charge_density", solver=_solver_token(solver), bc=Periodic())
    x = (np.arange(n) + 0.5) / n
    rho = eps * np.cos(2.0 * np.pi * x)[None, :] * np.ones((n, n))
    sim.set_state("ions", np.stack([rho, np.zeros_like(rho), np.zeros_like(rho)]))
    sim.solve_fields()
    phi = np.array(sim.potential())
    phi_exact = -(eps * np.cos(2.0 * np.pi * x) / (2.0 * np.pi) ** 2)[None, :] \
        * np.ones((n, n))
    scale = eps / (2.0 * np.pi) ** 2
    return np.abs(phi - phi_exact).max() / scale, abs(phi.mean()) / scale, phi


print("== (1) continuous spectral symbol matches the analytic mode ==")
e_sp, m_sp, _phi_sp = solve_phi(32, FFT(spectral=True))
chk(e_sp < 1e-12, "FFT(spectral=True) is machine-accurate on cos(2*pi*x)")
chk(m_sp < 1e-12, "the periodic potential uses a zero-mean gauge")

print("== (2) discrete FFT and GeometricMG retain the five-point O(h^2) result ==")
e_fft, _, phi_fft = solve_phi(32, FFT())
e_mg, _, phi_mg = solve_phi(32, GeometricMG())
chk(1e-4 < e_fft < 1e-2, "discrete FFT has the expected O(h^2) symbol error")
chk(1e-4 < e_mg < 1e-2, "GeometricMG has the expected O(h^2) stencil error")
d = np.abs(phi_fft - phi_mg).max() / (1e-3 / (2.0 * np.pi) ** 2)
chk(d < 1e-5, "discrete FFT and GeometricMG invert the same operator")
chk(e_sp < e_fft / 100, "the spectral route is numerically distinct from discrete FFT")

print("== (3) spectral exactness is independent of grid spacing ==")
e16, _, _ = solve_phi(16, FFT(spectral=True))
chk(e16 < 1e-12, "n=16 remains machine-accurate (no O(h^2) term)")

print("== (4) incompatible geometry and unknown routes fail loud ==")
sim = _system(32, periodic=False)
sim.set_poisson(
    rhs="charge_density",
    solver=_solver_token(FFT(spectral=True)),
    bc=Dirichlet(),
    wall=Disc(radius=0.4),
)
msg = err_msg(sim.solve_fields)
chk("fft_spectral" in msg and "wall" in msg,
    "spectral FFT refusal names the incompatible wall geometry")
sim2 = _system(32)
msg = err_msg(lambda: sim2.set_poisson(
    rhs="charge_density", solver="dct", bc=Periodic()))
chk("fft_spectral" in msg, "an unknown native route lists the spectral FFT alternative")

print("== (5) solved-potential halos are valid before source evaluation ==")


def rhs_with(solver):
    n = 32
    sim = _system(n)
    sim.set_poisson(
        rhs="charge_density", solver=_solver_token(solver), bc=Periodic())
    x = (np.arange(n) + 0.5) / n
    rho = (
        1e-3 * np.cos(2.0 * np.pi * x)[None, :] * np.ones((n, n))
        + 1e-3 * np.sin(2.0 * np.pi * x)[:, None] * np.ones((n, n))
    )
    sim.set_state("ions", np.stack([
        1.0 + rho, np.zeros_like(rho), np.zeros_like(rho)]))
    sim.solve_fields()
    return np.array(sim.eval_rhs("ions"))


reference = rhs_with(GeometricMG())
for solver, tolerance in ((FFT(), 1e-7), (FFT(spectral=True), 1e-4)):
    difference = np.abs(rhs_with(solver) - reference).max()
    chk(difference < tolerance,
        "%s publishes valid halos before grad(phi) is consumed" % solver.scheme)

print("FAILS =", fails)
sys.exit(1 if fails else 0)
