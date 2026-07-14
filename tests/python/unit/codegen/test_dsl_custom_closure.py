#!/usr/bin/env python3
"""ADC-543 : une fermeture de moments PERSONNALISEE se branche par la facade generique
(pops.moments.CartesianVelocityMoments(order, closure=...)) SANS ajouter aucun fichier sous
pops.lib. La fermeture est authoring-only : elle est evaluee une seule fois, au build, sur les
moments standardises symboliques S (Expr DSL), et son resultat est plie dans l'AST du flux qui
descend en C++ -- aucun Python dans le chemin par cellule.

On verifie ici, sans compilateur :
 (1) une fermeture utilisateur locale (decoree @closure(2)) satisfait le protocole Closure
     et se branche par la facade ; le flux d'ordre 2 genere == miroir numpy ecrit a la main
     (meme algebre de de-standardisation que la reference hyqmom15) ;
 (2) une fermeture-OBJET (une classe implementant __call__) est acceptee de la meme facon
     (le protocole est structurel) et donne le meme flux ;
 (3) AUCUN fichier n'a ete ajoute sous python/pops/lib/models/moments (la fermeture vit dans le
     code utilisateur, pas dans la bibliotheque des modeles fournis).
Test pur Python : flux_value evalue l'AST symbolique en flottants, aucun _pops / Kokkos requis.
"""
import pathlib
import sys

import numpy as np

from pops.moments import CartesianVelocityMoments, Closure, closure, moment_indices

fails = 0


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


# --- an independent gaussian raw-moment oracle (Stein recurrence on RAW moments) --------------
RHO, UU, VV, C20v, C11v, C02v = 1.3, 0.4, -0.25, 0.9, 0.15, 0.6


def gauss_raw(p, q, u, v, c20, c11, c02, memo=None):
    if memo is None:
        memo = {}
    if p < 0 or q < 0:
        return 0.0
    if (p, q) == (0, 0):
        return 1.0
    if (p, q) not in memo:
        if p >= 1:
            memo[(p, q)] = (u * gauss_raw(p - 1, q, u, v, c20, c11, c02, memo)
                            + (p - 1) * c20 * gauss_raw(p - 2, q, u, v, c20, c11, c02, memo)
                            + q * c11 * gauss_raw(p - 1, q - 1, u, v, c20, c11, c02, memo))
        else:
            memo[(p, q)] = (v * gauss_raw(p, q - 1, u, v, c20, c11, c02, memo)
                            + (q - 1) * c02 * gauss_raw(p, q - 2, u, v, c20, c11, c02, memo))
    return memo[(p, q)]


U6 = np.array([RHO * gauss_raw(p, q, UU, VV, C20v, C11v, C02v) for (p, q) in moment_indices(2)])


def _hand_mirror():
    """The order-3 raw moments for the funky polynomial closure, standardization by hand.

    Same de-standardization algebra as the hyqmom15 cross-validation (test_dsl_moments step 3):
    S30 = 0.7*S11 + 0.2, S21 = S11^2 - 0.1, S12 = -0.4*S11, S03 = 1.1, then C = S sx^p sy^q and
    the inverse binomial m_pq = ... (written term by term).
    """
    u_, v_ = UU, VV
    sx_, sy_ = np.sqrt(C20v), np.sqrt(C02v)
    s11_ = C11v / (sx_ * sy_)
    c30 = (0.7 * s11_ + 0.2) * sx_**3
    c21 = (s11_ * s11_ - 0.1) * sx_**2 * sy_
    c12 = (-0.4 * s11_) * sx_ * sy_**2
    c03 = 1.1 * sy_**3
    m30 = u_**3 + 3 * u_ * C20v + c30
    m21 = u_ * u_ * v_ + v_ * C20v + 2 * u_ * C11v + c21
    m12 = u_ * v_ * v_ + u_ * C02v + 2 * v_ * C11v + c12
    m03 = v_**3 + 3 * v_ * C02v + c03
    fx = np.array([U6[1], U6[2], RHO * m30, U6[4], RHO * m21, RHO * m12])
    fy = np.array([U6[3], U6[4], RHO * m21, U6[5], RHO * m12, RHO * m03])
    return fx, fy


FX_REF, FY_REF = _hand_mirror()


print("== (1) fermeture utilisateur decoree @closure(2), branchee par la facade ==")


@closure(2)
def my_closure(S):  # noqa: N803  (S mirrors the engine variable name)
    return {"S30": 0.7 * S["S11"] + 0.2, "S21": S["S11"] * S["S11"] - 0.1,
            "S12": -0.4 * S["S11"], "S03": 1.1}


chk(isinstance(my_closure, Closure),
    "la fermeture utilisateur satisfait le protocole Closure (structurel)")
model = CartesianVelocityMoments(2, closure=my_closure).add_transport().build(name="custom_fn")
fx = np.asarray(model.flux_value(U6, {}, model.frame.axes[0])).ravel()
fy = np.asarray(model.flux_value(U6, {}, model.frame.axes[1])).ravel()
e1 = max(np.abs(fx - FX_REF).max(), np.abs(fy - FY_REF).max())
chk(e1 < 1e-13, f"flux facade (fermeture-fonction) == miroir manuel a la main ({e1:.2e})")

print("== (2) fermeture-OBJET (classe avec __call__) : meme protocole, meme flux ==")


class PolyClosure:
    """A user closure written as an object (implements the Closure __call__ contract)."""

    def __call__(self, S):  # noqa: N803
        return {"S30": 0.7 * S["S11"] + 0.2, "S21": S["S11"] * S["S11"] - 0.1,
                "S12": -0.4 * S["S11"], "S03": 1.1}


obj_closure = PolyClosure()
chk(isinstance(obj_closure, Closure),
    "la fermeture-objet satisfait le protocole Closure")
model_obj = CartesianVelocityMoments(2, closure=obj_closure).build(name="custom_obj")
fxo = np.asarray(model_obj.flux_value(U6, {}, model_obj.frame.axes[0])).ravel()
chk(np.abs(fxo - FX_REF).max() < 1e-13,
    "flux facade (fermeture-objet) == miroir manuel (meme AST que la fonction)")

print("== (3) aucun fichier ajoute sous pops.lib.models.moments ==")
# The custom closure lives in THIS test (user code); the provided-model library must be
# unchanged -- exactly HyQMOM15 / Gaussian, no custom.py.
repo_root = pathlib.Path(__file__).resolve().parents[4]
lib_moments = repo_root / "python" / "pops" / "lib" / "models" / "moments"
lib_files = sorted(p.name for p in lib_moments.glob("*.py"))
chk(lib_files == ["__init__.py", "gaussian.py", "hyqmom15.py"],
    f"pops.lib.models.moments inchange (aucun custom.py) : {lib_files}")

print("FAILS =", fails)
sys.exit(1 if fails else 0)
