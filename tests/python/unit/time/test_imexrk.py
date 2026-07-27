#!/usr/bin/env python3
"""Contrat fail-closed de l'ancien descripteur IMEX-RK ARS(2,2,2).

``engine.IMEXRK`` conserve l'identite et les validations de la famille historique. Depuis le cutover
Program-only, ce descripteur de bloc n'installe toutefois aucun schema temporel et ne peut plus
declencher le moteur C++ ARS(2,2,2). Un ``System.step`` doit exiger un vrai Program whole-system au
lieu de reconstruire silencieusement l'ancien integrateur.

La convergence d'ordre deux et la stabilite raide devront etre retablies dans un test numerique
distinct lorsque le Program public disposera d'une primitive ARS(2,2,2) exacte pour cette source
native. Ici on prouve uniquement les contrats executables actuels : identite du descripteur,
absence de fallback temporel, et rejets AMR/polaire/masque partiel.
"""
from pops.numerics.reconstruction.limiters import Minmod
import sys

import pops.runtime._engine_descriptors as engine
from pops.mesh import PolarMesh
from pops.runtime._system import AmrSystem, System  # ADC-545 advanced runtime seam

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def cyclotron_model(q):
    """Fluide isotherme + force magnetique q*(v x B). elliptic charge=0 -> Poisson trivial (phi=0),
    aucune force electrique : la dynamique d'un etat uniforme est la pure gyration cyclotron."""
    return engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                     transport=engine.IsothermalFlux(),
                     source=engine.MagneticLorentzForce(charge=q),
                     elliptic=engine.ChargeDensity(charge=0.0))


def build(time_policy):
    sim = System(n=8, L=1.0, periodicity=(True, True))
    sim.add_equation(
        "e",
        cyclotron_model(1.0),
        spatial=engine.Spatial(limiter=Minmod()),
        time=time_policy,
    )
    return sim


def expect_program_required(policy, label):
    sim = build(policy)
    try:
        sim.step(0.1)
    except RuntimeError as error:
        chk(
            "installed whole-system Program" in str(error),
            f"{label}: step refuse sans Program explicite",
        )
        return
    chk(False, f"{label}: un integrateur temporel cache a ete execute")


# --- (a) IDENTITE DU DESCRIPTEUR ---------------------------------------------------------
print("== (a) identite stable des descripteurs IMEX / IMEXRK ==")
chk(engine.IMEX().kind == "imex", "engine.IMEX.kind == 'imex' (backward-Euler local, defaut)")
chk(engine.IMEXRK().kind == "imexrk_ars222", "engine.IMEXRK.kind == 'imexrk_ars222' (famille distincte)")
chk(engine.IMEX().kind != engine.IMEXRK().kind, "kinds distincts -> chemins C++ distincts (pas un alias)")
chk(engine.IMEXRK().scheme == "ars222", "engine.IMEXRK.scheme == 'ars222'")

# --- (b) AUCUN MOTEUR TEMPOREL CACHE ----------------------------------------------------
print("== (b) les descripteurs de bloc n'avancent pas sans Program ==")
expect_program_required(engine.IMEXRK(), "IMEXRK ARS(2,2,2)")
expect_program_required(engine.IMEX(), "IMEX backward-Euler")

# --- (c) REJETS EXPLICITES (perimetre = System cartesien) -------------------------------
print("== (c) rejets explicites : AMR / polaire / masque partiel ==")

# (c1) AMR
amr = AmrSystem(n=16, L=1.0, periodicity=(True, True), regrid_every=0)
try:
    amr.add_equation("e", cyclotron_model(1.0), spatial=engine.Spatial(limiter=Minmod()),
                  time=engine.IMEXRK())
    chk(False, "AMR + IMEXRK aurait du lever")
except (RuntimeError, ValueError, TypeError) as e:
    chk("imexrk" in str(e).lower() or "imex-rk" in str(e).lower(), f"AMR rejet explicite : {e}")

# (c2) polaire (anneau) : la source raide implicite n'y est pas cablee
simp = System(mesh=PolarMesh(r_min=0.2, r_max=1.0, nr=16, ntheta=16))
try:
    simp.add_equation("e",
                   engine.Model(state=engine.Scalar(), transport=engine.ExB(B0=1.0),
                             source=engine.NoSource(), elliptic=engine.BackgroundDensity()),
                   spatial=engine.Spatial(), time=engine.IMEXRK())
    chk(False, "polaire + IMEXRK aurait du lever")
except (RuntimeError, ValueError, TypeError) as e:
    chk("imex" in str(e).lower(), f"polaire rejet explicite : {e}")

# (c3) masque IMEX partiel : la source IMEXRK est pleinement implicite -> rejet a l'ajout du bloc
sim_mask = System(n=8, L=1.0, periodicity=(True, True))
try:
    # engine.IMEXRK n'expose pas implicit_vars ; on force l'attribut pour exercer la garde C++.
    pol = engine.IMEXRK()
    pol.implicit_vars = ["rho_u"]
    sim_mask.add_equation("e", cyclotron_model(1.0), spatial=engine.Spatial(limiter=Minmod()),
                       time=pol)
    chk(False, "IMEXRK + implicit_vars aurait du lever")
except (RuntimeError, ValueError) as e:
    chk("imexrk" in str(e).lower() or "fully implicit" in str(e).lower(),
        f"masque partiel rejete : {e}")

if fails:
    print(f"FAIL test_imexrk : {fails} echec(s)")
    sys.exit(1)
print("OK test_imexrk")
