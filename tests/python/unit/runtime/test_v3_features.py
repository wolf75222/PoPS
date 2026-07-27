#!/usr/bin/env python3
"""Vague 3 (solde des restes de genericite) : couverture facade.

  (A) CoupledSource.frequency : la 'CFL de couplage' declaree borne le pas
      (dt == cfl/mu, raison 'coupled_source:<nom>') -- System, sans compilateur ; et un couplage
      REJETE ne laisse AUCUNE borne fantome (frequence enregistree apres validation, revue v3) ;
  (B) Newton sur AMR : options non-defaut et diagnostics rejetes tant qu'aucune primitive implicite
      typee du Program ne les consomme ; aucun faux rapport ne reste expose par le runtime spatial ;
  (C) set_conservative_state MULTI-BLOCS : l'etat complet (avec quantite de mouvement) seede le
      grossier (la masse et la dynamique different du seed densite au repos) ;
  (D, compilateur) enable_hllc : riemann='hllc' accepte sur un modele DSL 3-var NON Euler via la
      capability emise ; rejete sans elle ; source_jacobian : parite de trajectoire avec les FD ;
      garde CODEGEN : source_jacobian sans source leve a compile() (pas seulement check()).

Invariants par assert ; imprime "OK test_v3_features" en cas de succes.
"""

from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction.limiters import Minmod
import os
import shutil
import sys
import tempfile

import numpy as np

import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Periodic
from pops.math import sqrt
from pops.physics._facade import Model
from pops.physics.multispecies import CoupledSource
from pops.runtime._system import AmrSystem, System  # ADC-545 advanced runtime seam
from tests.python.support.requirements import (
    missing_compiler_requirement,
    repo_include,
    require_native_or_skip,
)

fails = 0
INCLUDE = repo_include()


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
        source=engine.PotentialForce(charge=charge),
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
sim.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
# This density-exchange fixture sources the potential from the conserved total-density contrast.
# Both blocks therefore use the same elliptic sign while retaining their opposite force charges.
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
sim.set_density("a", rho16.ravel())
sim.set_density("b", rho16.ravel())
src = CoupledSource("friction").frequency(500.0)  # mu = 500 -> dt = 0.4/500 = 8e-4 << transport
na = src.block("a").role("density")
k = src.param("k", 1e-3)
src.add_pair("a", "b", role="density", expr=k * na)
sim.add_coupling(src.compile())
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

# --- (B) requetes Newton AMR : fail-closed ------------------------------------------
print("== (B) AMR : aucun Newton cache dans le runtime spatial ==")


def expect_amr_newton_rejection(label, time):
    system = AmrSystem(n=16, L=1.0, periodicity=(True, True), regrid_every=0)
    system.set_temporal_relations([2], [1], ["integral_only"])
    try:
        system.add_equation(
            "e",
            iso_model(n0=rho16_mean),
            spatial=engine.Spatial(limiter=Minmod()),
            time=time,
        )
    except (RuntimeError, ValueError) as error:
        chk(
            "Newton" in str(error) or "implicit" in str(error),
            f"{label} rejete explicitement",
        )
        return
    chk(False, f"{label} aurait du etre rejete")


expect_amr_newton_rejection(
    "options Newton non-defaut",
    engine.IMEX(newton_max_iters=5, newton_rel_tol=1e-10),
)
expect_amr_newton_rejection(
    "diagnostics Newton",
    engine.IMEX(newton_diagnostics=True),
)

amr_default_imex = AmrSystem(n=16, L=1.0, periodicity=(True, True), regrid_every=0)
amr_default_imex.set_temporal_relations([2], [1], ["integral_only"])
amr_default_imex.add_equation(
    "e",
    iso_model(n0=rho16_mean),
    spatial=engine.Spatial(limiter=Minmod()),
    time=engine.IMEX(),
)
chk(
    not hasattr(amr_default_imex, "newton_report"),
    "le runtime spatial AMR n'expose aucun faux rapport Newton",
)

# --- (C) set_conservative_state multi-blocs ------------------------------------------
print("== (C) set_conservative_state multi-blocs : etat complet seede (avec derive) ==")
amr3 = AmrSystem(n=16, L=1.0, periodicity=(True, True), regrid_every=0)
amr3.set_temporal_relations([2], [1], ["integral_only"])
amr3.set_poisson(rhs="charge_density", solver="geometric_mg", bc=Periodic())
amr3.set_refinement(1e30)
amr3.add_equation("e1", iso_model(+1.0, n0=rho16_mean), spatial=engine.Spatial(limiter=Minmod()))
amr3.add_equation("e2", iso_model(-1.0, n0=rho16_mean), spatial=engine.Spatial(limiter=Minmod()))
rho0 = rho16
u0 = 0.3 * np.ones((16, 16))
amr3.set_conservative_state("e1", np.stack([rho0, rho0 * u0, 0.0 * rho0]))
amr3.set_density("e2", rho0)
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
# --- (D) DSL : enable_hllc + source_jacobian (compilateur requis) ---------------------
missing = missing_compiler_requirement(INCLUDE)
if missing:
    if fails:
        print(f"FAIL test_v3_features : {fails} echec(s)")
        sys.exit(1)
    require_native_or_skip(f"(D) test_v3_features : {missing}")


def iso3_dsl(name, hllc=False, jac=False):
    m = Model(name)
    rho, mx, my = m.conservative_vars(
        "rho", "mx", "my", roles=["Density", "MomentumX", "MomentumY"]
    )
    cs2 = 0.5
    u = m.primitive("u", mx / rho)
    v = m.primitive("v", my / rho)
    m.primitive("p", cs2 * rho)
    c = sqrt(cs2)
    m.flux(x=[mx, mx * u + cs2 * rho, mx * v], y=[my, my * u, my * v + cs2 * rho])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    m.primitive_vars(rho, u, v)
    m.conservative_from([rho, rho * u, rho * v])
    m.elliptic_rhs(0.0 * rho)
    if hllc:
        m.enable_hllc()
    if jac:
        kk = 50.0
        m.source([0.0 * rho, -kk * mx, -kk * my])  # friction raide lineaire
        m.source_jacobian(
            [
                [0.0 * rho, 0.0 * rho, 0.0 * rho],
                [0.0 * rho, -kk + 0.0 * rho, 0.0 * rho],
                [0.0 * rho, 0.0 * rho, -kk + 0.0 * rho],
            ]
        )
    return m


tmp = tempfile.mkdtemp()
try:
    print("== (D1) enable_hllc : riemann='hllc' sur 3-var NON Euler ==")
    cm_h = iso3_dsl("iso3_hllc", hllc=True).compile(
        os.path.join(tmp, "iso3_hllc.so"), INCLUDE, backend="production"
    )
    chk(getattr(cm_h, "has_hllc", False), "CompiledModel.has_hllc = True (capability emise)")
    sh = System(n=24, L=1.0, periodicity=(True, True))
    sh.set_poisson()
    sh.add_equation(
        "f",
        model=cm_h,
        spatial=engine.Spatial(limiter=Minmod(), flux=HLLC()),
        time=engine.Explicit(),
    )
    z = np.zeros((24, 24))
    sh.set_primitive_state("f", rho=gaussian(24), u=z, v=z)
    for _ in range(5):
        sh.step_cfl(0.3)
    chk(
        np.all(np.isfinite(np.asarray(sh.density("f")))),
        "HLLC capability sur 3-var : 5 pas finis (contact-resolving hors Euler)",
    )
    cm_nh = iso3_dsl("iso3_nohllc").compile(
        os.path.join(tmp, "iso3_nohllc.so"), INCLUDE, backend="production"
    )
    try:
        s2 = System(n=16, L=1.0, periodicity=(True, True))
        s2.add_equation("f", model=cm_nh, spatial=engine.Spatial(limiter=Minmod(), flux=HLLC()))
        chk(False, "hllc sans capability sur 3-var aurait du lever")
    except (ValueError, RuntimeError) as e:
        chk("hllc" in str(e), f"rejet sans capability : {str(e)[:70]}")

    print("== (D2) source_jacobian : meme trajectoire que les differences finies ==")
    cm_j = iso3_dsl("iso3_jac", jac=True).compile(
        os.path.join(tmp, "iso3_jac.so"), INCLUDE, backend="production"
    )
    cm_f = iso3_dsl("iso3_fd", jac=True)
    cm_f._m._src_jac = None  # meme modele, SANS jacobien emis -> FD historiques
    cm_f = cm_f.compile(os.path.join(tmp, "iso3_fd.so"), INCLUDE, backend="production")

    def run_imex(cm):
        s = System(n=16, L=1.0, periodicity=(True, True))
        s.set_poisson()
        s.add_equation("f", model=cm, spatial=engine.Spatial(limiter=Minmod()), time=engine.IMEX())
        z16 = np.zeros((16, 16))
        s.set_primitive_state("f", rho=gaussian(16), u=0.2 + z16, v=z16)
        for _ in range(4):
            s.step(1e-3)
        return np.asarray(s.get_state("f"))

    uj, uf = run_imex(cm_j), run_imex(cm_f)
    chk(
        np.allclose(uj, uf, rtol=1e-9, atol=1e-11),
        f"jacobien analytique ~ FD (ecart max {float(np.max(np.abs(uj - uf))):.2e})",
    )
    chk(np.all(np.isfinite(uj)), "source raide (k*dt = 0.05*50) : etat fini")

    print("== (D3) garde CODEGEN : source_jacobian sans source -> erreur (pas de purge muette) ==")
    mg = iso3_dsl("iso3_guard", jac=True)
    mg._m._source = None  # jacobien declare, source retiree : compile() doit lever (pas check())
    try:
        mg.compile(os.path.join(tmp, "iso3_guard.so"), INCLUDE, backend="production")
        chk(False, "source_jacobian sans source aurait du lever au codegen")
    except ValueError as e:
        chk("source_jacobian" in str(e), f"codegen leve : {str(e)[:70]}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if fails:
    print(f"FAIL test_v3_features : {fails} echec(s)")
    sys.exit(1)
print("OK test_v3_features")
