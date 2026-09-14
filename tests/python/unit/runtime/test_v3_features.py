#!/usr/bin/env python3
"""Vague 3 (solde des restes de genericite) : couverture facade.

  (A) CoupledSource.frequency : la 'CFL de couplage' declaree borne le pas
      (dt == cfl/mu, raison 'coupled_source:<nom>') -- System, packages compiles ; et un couplage
      REJETE ne laisse AUCUNE borne fantome (frequence enregistree apres validation, revue v3) ;
  (B) Newton sur AMR : le runtime spatial n'expose aucun moteur temporel ou rapport Newton cache ;
      l'execution non lineaire est couverte par le Program compile dans test_amr_newton_full ;
  (C) set_conservative_state MULTI-BLOCS : l'etat complet (avec quantite de mouvement) seede le
      grossier (la masse et la dynamique different du seed densite au repos) ;

The independent compiler-capability checks live in test_v3_compiled_capabilities.py.

Invariants par assert ; imprime "OK test_v3_features" en cas de succes.
"""

from pops.numerics.reconstruction.limiters import Minmod
import sys

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.physics.multispecies import CoupledSource
from pops.runtime._amr_package_lane import ensure_native_block_state_route
from pops.runtime._modelspec_compile import compile_modelspec_package
from pops.runtime._system import AmrSystem, System  # ADC-545 advanced runtime seam
from tests.python.support.explicit_program import (
    install_forward_euler_program,
)

# Five model packages and two time Programs compile on a cold runner.
# Keep this runtime group separate from the independent compiler-capability cases.
POPS_PROCESS_TIMEOUT = 900

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso_model(charge=1.0, *, n0=1.0, elliptic_alpha=None):
    alpha = charge if elliptic_alpha is None else elliptic_alpha
    return engine.Model(
        state=engine.FluidState("isothermal", cs2=0.5),
        transport=engine.IsothermalFlux(),
        source=engine.NoSource(),
        elliptic=engine.BackgroundDensity(alpha=alpha, n0=n0),
    )


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.4 * np.exp(-60.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


# --- (A) CoupledSource.frequency ---------------------------------------------------
print("== (A) CoupledSource.frequency : borne dt <= cfl/mu sur le macro-pas ==")
n = 16
rho16 = gaussian(n)
rho16_mean = float(rho16.mean())
sim = System(n=n, L=1.0, periodicity=(True, True))
sim.set_poisson(rhs="charge_density", solver="cartesian_cg", bc=Periodic())
# This density-exchange fixture sources the potential from the conserved total-density contrast.
# Both blocks use the same elliptic sign; only the explicitly installed CoupledSource acts.
sim._batch_native_packages = True
sim.add_equation(
    "a",
    iso_model(+1.0, n0=rho16_mean, elliptic_alpha=1.0),
    spatial=engine.Spatial(limiter=Minmod()),
)
sim.add_equation(
    "b",
    iso_model(-1.0, n0=rho16_mean, elliptic_alpha=1.0),
    spatial=engine.Spatial(limiter=Minmod()),
)
sim._commit_pending_native_packages()
sim._batch_native_packages = False
sim.set_density("a", rho16.ravel())
sim.set_density("b", rho16.ravel())
src = CoupledSource("friction").frequency(500.0)  # mu = 500 -> dt = 0.4/500 = 8e-4 << transport
na = src.block("a").role("density")
k = src.param("k", 1e-3)
src.add_pair("a", "b", role="density", expr=k * na)
sim.add_coupling(src.compile())
install_forward_euler_program(sim, coupled_sources=True)
dt = sim.step_cfl(0.4)
chk(abs(dt - 0.4 / 500.0) < 1e-15, f"dt = cfl/mu = 8e-4 ({dt:.3e})")
chk(
    sim.last_dt_bound() == "coupled_source:friction",
    f"borne active = coupled_source:friction (recu {sim.last_dt_bound()!r})",
)
# Pas de BORNE FANTOME (revue vague 3) : un couplage REJETE (role absent du bloc) ne doit laisser
# AUCUNE frequence enregistree -- sinon le pas serait bride par une physique inexistante.
ghost = CoupledSource("ghost").frequency(5000.0)  # 0.4/5000 = 8e-5 << 8e-4 si fantome
ng = ghost.block("a").role("density")
kg = ghost.param("kg", 1e-3)
ghost.add_pair("a", "b", role="energy", expr=kg * ng)  # isotherme : pas de role Energy -> rejet C++
try:
    sim.add_coupling(ghost.compile())
    chk(False, "couplage sur role absent aurait du lever")
except (RuntimeError, ValueError):
    pass
dt2 = sim.step_cfl(0.4)
chk(
    abs(dt2 - 0.4 / 500.0) < 1e-15 and sim.last_dt_bound() == "coupled_source:friction",
    f"couplage rejete = ZERO borne fantome (dt {dt2:.3e}, borne {sim.last_dt_bound()!r})",
)

# --- (B) le runtime spatial AMR ne possede aucun moteur Newton -----------------------
print("== (B) AMR : aucun Newton cache dans le runtime spatial ==")
amr_program_only = AmrSystem(n=16, L=1.0, periodicity=(True, True), regrid_every=0)
amr_program_only.set_temporal_relations([2], [1], ["integral_only"])
amr_program_only.add_equation(
    "e",
    iso_model(n0=rho16_mean),
    spatial=engine.Spatial(limiter=Minmod()),
    time=engine.Explicit(),
)
report = amr_program_only.newton_report()
chk(
    report["enabled"] is False,
    "Explicit AMR package publishes no Newton report until solve_implicit_source runs",
)

# --- (C) set_conservative_state multi-blocs ------------------------------------------
print("== (C) set_conservative_state multi-blocs : etat complet seede (avec derive) ==")
amr3 = AmrSystem(n=16, L=1.0, periodicity=(True, True), regrid_every=0)
amr3.set_temporal_relations([2], [1], ["integral_only"])
amr3.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
packages = {
    name: compile_modelspec_package(
        iso_model(charge, n0=rho16_mean), name=name, target="amr_system",
    )
    for name, charge in (("e1", +1.0), ("e2", -1.0))
}
for name, package in packages.items():
    ensure_native_block_state_route(amr3._s, name, package)
for name, package in packages.items():
    amr3.add_equation(name, package, spatial=engine.Spatial(limiter=Minmod()))
rho0 = rho16
u0 = 0.3 * np.ones((16, 16))
amr3.set_conservative_state("e1", np.stack([rho0, rho0 * u0, 0.0 * rho0]))
amr3.set_density("e2", rho0)
install_forward_euler_program(amr3)
amr3.mark_bound()
d_before = np.asarray(amr3.density("e1")).reshape(16, 16).copy()
amr3.step(2e-3)
d_after = np.asarray(amr3.density("e1")).reshape(16, 16)
chk(np.all(np.isfinite(d_after)), "multi-blocs + etat complet : pas fini")
# la derive u0=0.3 advecte la bosse -> la densite CHANGE des le 1er pas (un seed au repos ne
# bougerait quasiment pas en 1 pas) : preuve que la quantite de mouvement a ete seedee.
chk(
    float(np.max(np.abs(d_after - d_before))) > 1e-5,
    "la quantite de mouvement seedee advecte la densite (etat complet actif)",
)

if fails:
    print(f"FAIL test_v3_features : {fails} echec(s)")
    sys.exit(1)
print("OK test_v3_features")
