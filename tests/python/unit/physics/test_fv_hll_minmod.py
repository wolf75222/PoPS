#!/usr/bin/env python3
"""Test VISIBLE du chemin generique HLL + minmod sur un modele NON Euler.

engine.Spatial(limiter=Minmod(), riemann=HLL(), variables=Primitive()) doit etre accepte par
System ET AmrSystem des que le modele expose des vitesses d'onde signees (model.wave_speeds) --
ici le fluide ISOTHERME 3 variables (rho, rho_u, rho_v), qui n'est PAS Euler 4 variables : ce
modele expose les vitesses d'onde signees requises par HLL. Le controle HLLC utilise
la meme physique isotherme, mais sans declaration du fournisseur d'etat etoile HLLC.

Verifie aussi :
  - hll sur une advection scalaire (pas de wave_speeds) -> erreur EXPLICITE (pas de fallback) ;
  - hllc sur un modele isotherme sans fournisseur HLLC -> rejet explicite ;
  - quelques pas step_cfl : etat fini, masse conservee (domaine periodique).

Invariants par assert ; imprime "OK test_fv_hll_minmod" en cas de succes.
"""
from pops.numerics.riemann import HLL
from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.variables import Primitive
import sys

import numpy as np

from pops.physics import Density, Momentum, Velocity
from pops.physics._facade import Model
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.math import sqrt

import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.runtime._system import AmrSystem, System  # ADC-545 advanced runtime seam
from tests.python.support.explicit_program import install_forward_euler_program

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def iso_model(charge=1.0, cs2=0.5, n0=1.0):
    return engine.Model(state=engine.FluidState("isothermal", cs2=cs2),
                     transport=engine.IsothermalFlux(),
                     source=engine.NoSource(),
                     elliptic=engine.BackgroundDensity(alpha=charge, n0=n0))


def gaussian(n):
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="xy")
    return 1.0 + 0.5 * np.exp(-80.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))


# --- 1. System : hll + minmod + primitive sur isotherme 3 var (non Euler) -------
print("== System : FiniteVolume(minmod, hll, primitive) sur isotherme 3 var ==")
n = 32
rho0 = gaussian(n)
sim = System(n=n, L=1.0, periodicity=(True, True))
sim.add_equation("ions", iso_model(n0=float(rho0.mean())),
              spatial=engine.Spatial(limiter=Minmod(), flux=HLL(), recon=Primitive()),
              time=engine.Explicit())
sim.set_poisson(rhs="charge_density", solver="cartesian_cg", bc=Periodic())
sim.set_density("ions", rho0.ravel())
install_forward_euler_program(sim)
m0 = sim.mass("ions")
for _ in range(5):
    dt = sim.step_cfl(0.4)
    chk(np.isfinite(dt) and dt > 0, f"step_cfl rend un dt fini > 0 (dt={dt:.3e})")
rho = np.asarray(sim.density("ions"))
chk(np.all(np.isfinite(rho)), "densite finie apres 5 pas hll+minmod")
chk(abs(sim.mass("ions") - m0) < 1e-10 * abs(m0), "masse conservee (periodique)")

# --- 2. hll sans vitesses d'onde signees -> erreur explicite -------------------
print("== hll sans wave_speeds (advection scalaire) : rejet explicite ==")

scalar = Model("hll_missing_waves")
(scalar_rho,) = scalar.conservative_vars("n", roles=(Density(),))
scalar.flux(x=[0.3 * scalar_rho], y=[0.2 * scalar_rho])
scalar.eigenvalues(x=[0.3 + 0.0 * scalar_rho], y=[0.2 + 0.0 * scalar_rho])
scalar.primitive_vars(scalar_rho)
scalar.conservative_from([scalar_rho])
compiled_scalar = scalar.compile(
    backend="production", target="system", name="hll_missing_waves",
    consumer_owner_qid="tests.hll-minmod.scalar",
)
assert not compiled_scalar.has_wave_speeds
sim2 = System(n=16, L=1.0, periodicity=(True, True))
try:
    sim2.add_equation("e", compiled_scalar, spatial=engine.Spatial(limiter=Minmod(), flux=HLL()))
    chk(False, "hll sans wave_speeds aurait du lever")
except (ValueError, RuntimeError) as e:
    chk("wave_speeds" in str(e) or "signed wave speeds" in str(e), f"erreur explicite : {e}")

# --- 3. meme physique isotherme sans fournisseur HLLC -> rejet explicite -------
print("== hllc sans fournisseur d'etat etoile : rejet explicite ==")
x_axis, y_axis = Rectangle("hll-minmod-domain", (0.0, 0.0), (1.0, 1.0)).frame(Cartesian2D()).axes
isothermal = Model("hllc_missing_provider")
r, mx, my = isothermal.conservative_vars(
    "rho", "mx", "my", roles=(Density(), Momentum(axis=x_axis), Momentum(axis=y_axis)),
)
u = isothermal.primitive("u", mx / r)
v = isothermal.primitive("v", my / r)
p = isothermal.primitive("p", 0.5 * r)
isothermal.flux(x=[mx, mx * u + p, mx * v], y=[my, my * u, my * v + p])
c = sqrt(p / r)
isothermal.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
isothermal.primitive_vars(r, u, v, roles=(Density(), Velocity(axis=x_axis), Velocity(axis=y_axis)))
isothermal.conservative_from([r, r * u, r * v])
compiled_isothermal = isothermal.compile(
    backend="production", target="system", name="hllc_missing_provider",
    consumer_owner_qid="tests.hll-minmod.isothermal",
)
assert compiled_isothermal.has_wave_speeds
assert not compiled_isothermal.has_hllc
sim3 = System(n=16, L=1.0, periodicity=(True, True))
try:
    sim3.add_equation("ions", compiled_isothermal, spatial=engine.Spatial(limiter=Minmod(), flux=HLLC()))
    chk(False, "hllc sans fournisseur aurait du lever")
except (ValueError, RuntimeError) as e:
    chk("hllc" in str(e).lower(), f"erreur explicite : {e}")

# --- 4. AmrSystem : hll + minmod accepte (alignement de surface System/AMR) ------
print("== AmrSystem : add_equation(riemann='hll') accepte sur isotherme ==")
amr = AmrSystem(n=32, L=1.0, periodicity=(True, True), regrid_every=0)
amr.set_temporal_relations([2], [1], ["integral_only"])
amr.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
amr_rho0 = gaussian(32)
amr.add_equation("ions", iso_model(n0=float(amr_rho0.mean())),
              spatial=engine.Spatial(limiter=Minmod(), flux=HLL(), recon=Primitive()),
              time=engine.Explicit())
amr.set_density("ions", amr_rho0)
install_forward_euler_program(amr)
amr.mark_bound()
m0 = amr.mass("ions")
for _ in range(3):
    dt = amr.step_cfl(0.4)
    chk(np.isfinite(dt) and dt > 0, f"AMR step_cfl rend un dt fini > 0 (dt={dt:.3e})")
rho = np.asarray(amr.density("ions"))
chk(np.all(np.isfinite(rho)), "densite AMR finie apres 3 pas hll+minmod")

if fails:
    print(f"FAIL test_fv_hll_minmod : {fails} echec(s)")
    sys.exit(1)
print("OK test_fv_hll_minmod")
