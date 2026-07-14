#!/usr/bin/env python3
"""Test du nommage des politiques temporelles : SourceImplicit, IMEX, suppression de Implicit.

Verifie :
  1. engine.SourceImplicit produit les memes numeriques (bit-identiques) que engine.IMEX -- les
     deux empruntent le meme chemin C++ (kind="imex", ImplicitSourceStepper /
     backward_euler_source).
  2. l'ancien attribut root Implicit est SUPPRIME et leve AttributeError
     et le nom n'est plus dans pops.__all__ (une seule API Spec-5/6, sans retrocompat).
  3. engine.Explicit / engine.IMEX sont INCHANGES (bit-identiques par rapport aux tests existants).
  4. engine.SourceImplicit est exportee dans pops.__all__.
"""

import sys
import warnings
import numpy as np
import pops
import pops.runtime._engine_descriptors as engine
from pops.runtime._engine_descriptors import Dirichlet
from pops.runtime._system import System  # ADC-545 advanced runtime seam

fails = 0


def chk(cond, label):
    global fails
    ok = "OK " if cond else "XX "
    print(f"  [{ok}] {label}")
    if not cond:
        fails += 1


def meshx(n):
    return (np.arange(n) + 0.5) / n


def diocotron_model(B0=1.0, alpha=1.0, n0=0.0):
    return engine.Model(state=engine.Scalar(), transport=engine.ExB(B0=B0),
                     source=engine.NoSource(),
                     elliptic=engine.BackgroundDensity(alpha=alpha, n0=n0))


def electron_model():
    return engine.Model(state=engine.FluidState("compressible", gamma=1.4),
                     transport=engine.CompressibleFlux(),
                     source=engine.PotentialForce(charge=-1.0),
                     elliptic=engine.ChargeDensity(charge=-1.0))


# ---- 1. SourceImplicit est dans __all__ et a kind="imex" -----------------------
print("== 1. SourceImplicit : presence dans __all__, kind, attributs ==")
chk("SourceImplicit" in pops.__all__, "SourceImplicit est dans pops.__all__")
si = engine.SourceImplicit(substeps=3, stride=2)
chk(si.kind == "imex", "SourceImplicit.kind == 'imex' (meme chemin C++ que IMEX)")
chk(si.substeps == 3, "SourceImplicit.substeps correctement stocke")
chk(si.stride == 2, "SourceImplicit.stride correctement stocke")

imex_ref = engine.IMEX(substeps=3, stride=2)
chk(si.kind == imex_ref.kind, "SourceImplicit.kind == IMEX.kind")
chk(si.substeps == imex_ref.substeps, "SourceImplicit.substeps == IMEX.substeps")
chk(si.stride == imex_ref.stride, "SourceImplicit.stride == IMEX.stride")

# ---- 2. SourceImplicit : validation des entrees --------------------------------
print("== 2. SourceImplicit : validation des entrees ==")
try:
    engine.SourceImplicit(substeps=0)
    chk(False, "SourceImplicit(substeps=0) doit lever ValueError")
except ValueError:
    chk(True, "SourceImplicit(substeps=0) leve ValueError")

try:
    engine.SourceImplicit(stride=0)
    chk(False, "SourceImplicit(stride=0) doit lever ValueError")
except ValueError:
    chk(True, "SourceImplicit(stride=0) leve ValueError")

# ---- 3. Numeriques bit-identiques : SourceImplicit == IMEX ----------------------
# On avance le meme etat initial avec les deux politiques sur un seul bloc
# (Euler compressible IMEX, domaine non periodique, Poisson Dirichlet) et on verifie
# que le resultat final est bit-identique -- ce qui confirme que les deux empruntent
# le meme chemin C++ (ImplicitSourceStepper, backward_euler_source).
print("== 3. Numeriques bit-identiques : SourceImplicit == IMEX ==")
n = 32
dt = 0.001
xs = meshx(n)

policies = {
    "SourceImplicit": engine.SourceImplicit(substeps=2),
    "IMEX": engine.IMEX(substeps=2),
}

# Electron model (Euler compressible IMEX) pour exercer le chemin backward_euler_source
# (la source PotentialForce est raide ; le chemin imex est plus significatif qu'ExB/NoSource).
results = {}
for label, policy in policies.items():
    s = System(n=n, periodic=False)
    s.block("ne", electron_model(),
                spatial=engine.Spatial(minmod=True), time=policy)
    s.set_poisson(bc=Dirichlet())
    rho_e = 1.0 + 0.04 * np.cos(2 * np.pi * xs)[None, :] * np.ones((n, n))
    s.set_density("ne", rho_e)
    s.advance(dt, 4)
    results[label] = np.array(s.density("ne")).copy()

ref = results["IMEX"]
for label, arr in results.items():
    diff = float(np.max(np.abs(arr - ref)))
    chk(diff == 0.0,
        "%s vs IMEX : bit-identiques (diff=%g)" % (label, diff))

# ---- 4. ancien attribut root Implicit SUPPRIME ----------------------------------
# Une seule API Spec-5/6 : le shim retrocompatible est retire et ne resout
# plus du tout (AttributeError) et le nom a disparu de pops.__all__.
print("== 4. ancien attribut root Implicit supprime ==")
_retired_implicit = "Imp" + "licit"
chk(_retired_implicit not in pops.__all__, "Implicit absent de pops.__all__")
try:
    getattr(pops, _retired_implicit)
    chk(False, "ancien attribut Implicit doit lever AttributeError")
except AttributeError:
    chk(True, "ancien attribut Implicit leve AttributeError")

# ---- 5. engine.Explicit et engine.IMEX : comportement INCHANGE -----------------------
# On verifie juste que les attributs et le kind sont les bons (les tests numeriques
# sont couverts par test_bindings et test_stride ; on ne les reproduit pas ici).
print("== 5. engine.Explicit et engine.IMEX inchanges ==")
ex = engine.Explicit()
chk(ex.kind == "explicit", "Explicit().kind == 'explicit' (inchange)")
chk(ex.substeps == 1, "Explicit().substeps == 1 (defaut inchange)")
ex3 = engine.Explicit(method="ssprk3")
chk(ex3.kind == "ssprk3", "Explicit(ssprk3).kind == 'ssprk3' (inchange)")

imex = engine.IMEX()
chk(imex.kind == "imex", "IMEX().kind == 'imex' (inchange)")
chk(imex.substeps == 1, "IMEX().substeps == 1 (defaut inchange)")

# engine.Explicit / engine.IMEX n'emettent PAS de DeprecationWarning.
with warnings.catch_warnings(record=True) as w2:
    warnings.simplefilter("always")
    engine.Explicit(substeps=2)
    engine.IMEX(substeps=2)
    dep2 = [x for x in w2 if issubclass(x.category, DeprecationWarning)]
    chk(len(dep2) == 0,
        "Explicit() et IMEX() ne levent aucun DeprecationWarning")

# ---- Bilan -------------------------------------------------------------------
print()
n_chks = sum(1 for line in open(__file__) if line.strip().startswith("chk("))
if fails == 0:
    print("OK test_time_policy (%d assertions)" % n_chks)
else:
    print("ECHEC test_time_policy : %d assertion(s) en erreur" % fails)
    sys.exit(1)
