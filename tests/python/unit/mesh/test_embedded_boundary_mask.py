#!/usr/bin/env python3
"""Analytic embedded-boundary mask contract (inert by default).

The native runtime owns one exact-rank signed LevelSet and materializes its active mask.  This
script retains the useful checks: no geometry is bit-identical to Cartesian, a generic implicit
surface produces the exact 0/1 cell-centred mask, and transport mode requires that geometry.
"""
from tests.python.support.requirements import require_native_or_skip

from pops.numerics.variables import Conservative
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.riemann import Rusanov
from pops.mesh.masks import Staircase
from tests.python.support.explicit_program import install_forward_euler_program

import numpy as np

try:
    from pops.runtime._engine_descriptors import (
        BackgroundDensity, Explicit, FluidState, IsothermalFlux, Model, NoSource, Periodic, Spatial,
    )
    from pops.runtime._system import System  # ADC-545 advanced runtime seam
except ImportError as e:  # pragma: no cover - environnement sans build
    require_native_or_skip('module pops absent (PYTHONPATH ? build ?) : %s' % e)


fails = 0


def chk(cond, label):
    global fails
    print("  [%s] %s" % ("OK " if cond else "XX ", label))
    if not cond:
        fails += 1


def iso_model(cs2=1.0, alpha=3.0, n0=1.0):
    """Fluide isotherme natif, sans compilation C++; roles Density/MomentumX/MomentumY."""
    return Model(
        state=FluidState(kind="isothermal", cs2=cs2),
        transport=IsothermalFlux(),
        source=NoSource(),
        elliptic=BackgroundDensity(alpha=alpha, n0=n0),
    )


def ring(n, L, cx=0.5, cy=0.5):
    x = (np.arange(n) + 0.5) * (L / n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return 1.0 + 0.5 * np.exp(-(((X - cx * L) ** 2 + (Y - cy * L) ** 2) / (0.02 * L * L)))


# ---------------------------------------------------------------------------
# (a) inertie / bit-identite : masque tout actif par defaut, trajectoire inchangee
# ---------------------------------------------------------------------------

def _build(n, L):
    sim = System(n=n, L=L, periodicity=(True, True))
    sim.set_poisson(bc=Periodic())
    rho0 = ring(n, L)
    sim.add_equation("s", model=iso_model(n0=float(rho0.mean())),
                     spatial=Spatial(limiter=Minmod(), flux=Rusanov(),
                                     recon=Conservative()),
                     time=Explicit())
    # Vitesse initiale CONSTANTE non nulle : le transport advecte la bosse (test non trivial).
    sim.set_primitive_state("s", rho=rho0, u=0.7 + 0.0 * rho0, v=-0.4 + 0.0 * rho0)
    install_forward_euler_program(sim)
    return sim


def _install_half_space(sim, mode="staircase"):
    from pops.analytic import coordinates
    from pops.domain import CartesianDomain
    from pops.mesh.geometry import LevelSet
    from pops.runtime._analytic_expression_lowering import lower_analytic_components

    frame = CartesianDomain("test-eb-mask", (0.0, 0.0), (1.0, 1.0)).frame()
    level_set = LevelSet(coordinates(frame)[0] - 0.5)
    ((opcodes, literals),) = lower_analytic_components(
        (level_set.expression.to_data(),), frame_id=frame.canonical_id
    )
    sim._s._set_analytic_level_set(
        list(opcodes), list(literals), mode, 0.0, 0.0, 0.0
    )


def test_inert_default():
    n, L = 40, 1.0

    # No installed LevelSet means an all-active mask.
    sim = _build(n, L)
    mk = np.array(sim.embedded_boundary_mask())
    chk(mk.shape == (n, n),
        "(a) embedded_boundary_mask() has shape (ny, nx) = (%d, %d): got %r"
        % (n, n, tuple(mk.shape)))
    chk(np.all(mk == 1.0),
        "(a) masque TOUT ACTIF par defaut (tous 1.0, sous-domaine = domaine entier) : min = %g, "
        "max = %g" % (float(mk.min()), float(mk.max())))

    # Querying the default sidecar must not perturb an otherwise identical Cartesian trajectory.
    sim_a = _build(n, L)
    sim_b = _build(n, L)
    dt = 0.2 * (L / n) / np.hypot(0.7, 0.4)
    for _ in range(30):
        _ = sim_a.embedded_boundary_mask()
        sim_a.step(dt)
        sim_b.step(dt)
    ua = np.array(sim_a.density("s"))
    ub = np.array(sim_b.density("s"))
    max_diff = float(np.max(np.abs(ua - ub)))
    print("    [INERTIE] max|rho_avec_query - rho_sans_query| = %.3e (attendu 0)" % max_diff)
    chk(max_diff == 0.0,
        "(a) querying the default embedded-boundary mask does not change the trajectory "
        "(diff = 0) : invariant inerte par defaut")
    chk(np.max(np.abs(ua - 1.0)) > 1e-3,
        "(a) le transport a effectivement avance l'etat (test non trivial) : max dev = %.3e"
        % float(np.max(np.abs(ua - 1.0))))


# ---------------------------------------------------------------------------
# (b) a generic analytic LevelSet materializes the expected mask
# ---------------------------------------------------------------------------

def test_analytic_level_set_mask_matches_half_space():
    n, L = 48, 1.0
    sim = _build(n, L)
    _install_half_space(sim)
    mk = np.array(sim.embedded_boundary_mask())

    # phi = x - 1/2; active is the strict convention phi < 0.  x varies along columns.
    xc = (np.arange(n) + 0.5) * (L / n)
    yc = (np.arange(n) + 0.5) * (L / n)
    XI, _ = np.meshgrid(xc, yc, indexing="xy")
    ls = XI - 0.5
    expected = (ls < 0.0).astype(np.float64)

    chk(mk.shape == (n, n), "(b) embedded_boundary_mask() shape (ny, nx)")
    n_active = int(mk.sum())
    chk(0 < n_active < n * n,
        "(b) the analytic LevelSet partitions active and inactive cells: %d active out of %d"
        % (n_active, n * n))
    chk(np.array_equal(mk, expected),
        "(b) mask == indicator of phi=x-1/2 < 0: %d mismatched cells"
        % int(np.sum(mk != expected)))
    chk(set(np.unique(mk).tolist()) <= {0.0, 1.0},
        "(b) masque strictement 0/1 (valeurs uniques %r)" % np.unique(mk).tolist())


# ---------------------------------------------------------------------------
# (c) transport mode requires a prepared analytic geometry
# ---------------------------------------------------------------------------

def test_geometry_mode_requires_level_set():
    n, L = 32, 1.0
    sim = _build(n, L)
    raised = False
    try:
        sim.set_geometry_mode(Staircase())
    except Exception:
        raised = True
    chk(raised, "(c) staircase mode requires a prepared analytic LevelSet")


def main():
    print("(a) inertie / bit-identite (all-active default mask)")
    test_inert_default()
    print("(b) analytic LevelSet mask == cell-centred indicator")
    test_analytic_level_set_mask_matches_half_space()
    print("(c) geometry-mode guard")
    test_geometry_mode_requires_level_set()

    if fails == 0:
        print("test_embedded_boundary_mask: all checks passed")
    else:
        raise SystemExit("test_embedded_boundary_mask: %d check(s) failed" % fails)


if __name__ == "__main__":
    main()
