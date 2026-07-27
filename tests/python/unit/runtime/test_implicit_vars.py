#!/usr/bin/env python3
"""Contrat fail-closed du masque IMEX prive apres le cutover Program-only.

``engine.IMEX`` et ``engine.SourceImplicit`` conservent leurs champs d'auteur historiques afin de
valider les noms et roles au chargement d'un bloc natif. Ils ne constituent toutefois plus une
autorite temporelle : aucun ``System.advance`` ne peut deduire un backward-Euler, un masque partiel
ou un solveur Newton de ces descripteurs. Un vrai ``pops.Program`` portant la primitive implicite
doit etre installe ; cette primitive n'existe pas encore pour le masque partiel natif.

Le test verifie donc les seules promesses executables actuelles :
  1. normalisation stable des noms et roles portes par les descripteurs ;
  2. toutes les variantes IMEX refusent d'avancer sans Program, sans fallback temporel cache ;
  3. les noms/roles absents et un masque attache a ``Explicit`` restent rejetes clairement.
"""

import sys

import pops.runtime._engine_descriptors as engine
from pops.runtime._system import System  # ADC-545 advanced runtime seam

fails = 0


def chk(cond, label):
    global fails
    ok = "OK " if cond else "XX "
    print(f"  [{ok}] {label}")
    if not cond:
        fails += 1


def electron_model():
    return engine.Model(state=engine.FluidState("compressible", gamma=1.4),
                     transport=engine.CompressibleFlux(),
                     source=engine.PotentialForce(charge=-50.0),
                     elliptic=engine.ChargeDensity(charge=-1.0))


def system_with(policy):
    system = System(n=16, periodicity=(False, False))
    system.add_equation(
        "ne",
        electron_model(),
        spatial=engine.Spatial(minmod=True),
        time=policy,
    )
    return system


def expect_program_required(policy, label):
    system = system_with(policy)
    try:
        system.advance(0.002, 1)
    except RuntimeError as error:
        message = str(error)
        chk(
            "installed whole-system Program" in message,
            f"{label}: avance refusee sans Program explicite",
        )
        return
    chk(False, f"{label}: un moteur temporel cache a avance sans Program")


def add_policy(policy):
    s = System(n=16, periodicity=(False, False))
    s.add_equation("ne", electron_model(), spatial=engine.Spatial(minmod=True), time=policy)


# ---- 0. attributs portes par la politique (masque cote bloc, pas modele) -------
print("== 0. IMEX / SourceImplicit portent le masque implicite ==")
p = engine.IMEX(substeps=2, implicit_vars=["rho_u", "rho_v"])
chk(p.implicit_vars == ["rho_u", "rho_v"], "IMEX.implicit_vars stocke les noms")
chk(p.implicit_roles == [], "IMEX.implicit_roles vide par defaut")
pr = engine.IMEX(implicit_roles=["MomentumX", "MomentumY", "Energy"])
# normalisation PascalCase -> cle stable snake_case (cf. role_from_name C++)
chk(pr.implicit_roles == ["momentum_x", "momentum_y", "energy"],
    "IMEX.implicit_roles normalise PascalCase -> snake_case stable")
si = engine.SourceImplicit(implicit_vars=["E"])
chk(si.implicit_vars == ["E"], "SourceImplicit.implicit_vars stocke les noms")
chk(engine.IMEX().implicit_vars == [] and engine.IMEX().implicit_roles == [],
    "IMEX() sans masque : listes vides (defaut)")

# ---- 1. aucune politique de bloc ne remplace un Program -----------------------
print("== 1. les politiques IMEX de bloc ne sont pas des moteurs temporels ==")
expect_program_required(engine.IMEX(substeps=2), "IMEX sans masque")
expect_program_required(
    engine.IMEX(substeps=2, implicit_vars=["rho_u", "rho_v"]),
    "IMEX masque par noms",
)
expect_program_required(
    engine.IMEX(substeps=2, implicit_roles=["MomentumX", "MomentumY"]),
    "IMEX masque par roles",
)
expect_program_required(engine.SourceImplicit(substeps=2), "SourceImplicit")

# ---- 2. erreur claire si nom / role absent du bloc ----------------------------
print("== 2. erreur explicite sur un nom / role absent ==")
try:
    add_policy(engine.IMEX(implicit_vars=["rho_w"]))
    chk(False, "implicit_vars=['rho_w'] doit lever (nom absent)")
except Exception as e:
    msg = str(e)
    chk("rho_w" in msg and "implicit_vars" in msg,
        "nom absent -> erreur mentionnant 'rho_w' et 'implicit_vars'")

try:
    add_policy(engine.IMEX(implicit_roles=["Pressure"]))
    chk(False, "implicit_roles=['Pressure'] doit lever (role absent)")
except Exception as e:
    msg = str(e)
    chk("implicit_roles" in msg,
        "role absent -> erreur mentionnant 'implicit_roles'")

try:
    add_policy(engine.IMEX(implicit_roles=["NotARole"]))
    chk(False, "implicit_roles=['NotARole'] doit lever (role inconnu)")
except Exception as e:
    chk("implicit_roles" in str(e),
        "role inconnu -> erreur explicite")

# ---- 3. masque interdit hors IMEX (explicite) ---------------------------------
print("== 3. masque rejete sur une politique non-IMEX ==")
# ``Explicit`` ne porte pas de masque par construction. Pour exercer la validation native sans
# appeler ``_s`` directement, on attache volontairement un attribut de masque a la valeur de
# politique puis on passe par ``System.add_equation`` : le runtime doit refuser ce contrat incoherent
# (le masque n'a de sens qu'en IMEX).
try:
    s = System(n=16, periodicity=(False, False))
    explicit_with_mask = engine.Explicit()
    explicit_with_mask.implicit_vars = ["rho_u"]
    explicit_with_mask.implicit_roles = []
    s.add_equation(
        "ne", electron_model(),
        spatial=engine.Spatial(minmod=True),
        time=explicit_with_mask,
    )
    chk(False, "add_block(time='explicit', implicit_vars=['rho_u']) doit lever")
except Exception as e:
    chk("imex" in str(e).lower(),
        "masque sur time='explicit' -> erreur mentionnant imex")

# ---- Bilan -------------------------------------------------------------------
print()
n_chks = sum(1 for line in open(__file__) if line.strip().startswith("chk("))
if fails == 0:
    print("OK test_implicit_vars (%d assertions)" % n_chks)
else:
    print("ECHEC test_implicit_vars : %d assertion(s) en erreur" % fails)
    sys.exit(1)
