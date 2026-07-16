#!/usr/bin/env python3
"""Generateur generique de modeles de moments 2D (pops.moments) : l'algebre binomiale
M -> C -> S -> fermeture -> C' -> M' est DERIVEE en boucles sur l'AST de la DSL ; seule la
fermeture (callable) est fournie. Reference de validation croisee : adc_cases/hyqmom15
(flux generes == goldens MATLAB a 7.7e-13, == modele manuel a 2.6e-13 -- hors de ce depot).

On verifie ici, sans dependance externe :
 (1) convention d'ordre : moment_names(4) == les 15 noms canoniques (contrat hyqmom15) ;
 (2) oracle d'Isserlis INDEPENDANT (recurrence de Stein sur moments BRUTS, u,v != 0 --
     autre chemin que la recurrence standardisee du module) : avec gaussian_closure(order),
     le flux d'un etat gaussien == moments bruts gaussiens decales, ordres 2, 3 et 4 ;
 (3) plomberie de de-standardisation : fermeture non triviale (polynome arbitraire de S)
     sur l'ordre 2, miroir numpy ecrit a la main dans le test ;
 (3b) standardisation d'ENTREE (p+q >= 3) : fermeture-copie consommant S30/S21/S12/S03 sur
     un etat ASYMETRIQUE (melange de 2 gaussiennes, moments centres impairs non nuls --
     un etat gaussien rendrait le test aveugle), miroir a la main ; attrape une inversion
     d'exposants sx^p sy^q (verifie par mutation) que (2)-(3) laissent passer ;
 (4) strategie de vitesses exacte : le descripteur final ExactSpeeds expose le choix et ses
     capacites natives, sans l'evaluateur host legacy ;
 (6) robust=True : etat quasi-vide fini la ou le chemin nu deborde ; quasi-identite sur
     etat sain ;
 (7) lorentz_sources : table d'ordre 2 derivee a la main, evaluee en flottants purs ;
     fermeture de la hierarchie (aucune reference hors variables transportees) ;
 (7b) maxwellian_moments / bgk_source : oracle d'Isserlis INDEPENDANT (recurrence de Stein
     sur moments BRUTS) ; point fixe (les moments d'une gaussienne == sa maxwellienne),
     source BGK nulle a l'equilibre, invariants collisionnels (M00/M10/M01) identiquement 0,
     gaussianisation d'un etat asymetrique (ordres <= 2 conserves, ordres >= 3 modifies) ;
 (8) gardes : order < 2 ; fermeture aux cles incompletes ;
 (9) [compilateur] lifecycle public Case -> compile -> bind -> run avec HLL : 10 pas finis,
     masse conservee.
S'auto-saute (exit 0) pour (9) sans compilateur C++ ou sans Kokkos (coeur Kokkos-only).
"""
import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.lib.time import ForwardEuler
from pops.layouts import Uniform
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, variables
from pops.numerics.riemann import FromJacobian, HLL, provider_of
from pops.numerics.spatial import FiniteVolume
from pops.time import FixedDt
import sys

import numpy as np

from pops.codegen.toolchain import _default_cxx
from pops.moments import (CartesianVelocityMoments, ExactSpeeds, bgk_source,
                          gaussian_closure, lorentz_sources, maxwellian_moments,
                          moment_indices, moment_names)
from tests.python.support.requirements import repo_include

fails = 0
INCLUDE = repo_include()
MOMENT_FRAME = Rectangle(
    "moment-domain", lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())


def chk(cond, label):
    global fails
    print(f"  [{'OK ' if cond else 'XX '}] {label}")
    if not cond:
        fails += 1


def err_msg(fn):
    try:
        fn()
        return ""
    except Exception as ex:  # noqa: BLE001
        return str(ex)


def gauss_raw(p, q, u, v, c20, c11, c02, memo=None):
    """Oracle : moment brut E[x^p y^q] d'une gaussienne (u, v, C) par la recurrence de
    Stein sur les moments BRUTS : m_pq = u m_{p-1,q} + (p-1) c20 m_{p-2,q} + q c11 m_{p-1,q-1}
    (et symetrique en y) -- independante de la recurrence STANDARDISEE de gaussian_closure."""
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


RHO, UU, VV, C20v, C11v, C02v = 1.3, 0.4, -0.25, 0.9, 0.15, 0.6


def gauss_state(order):
    return np.array([RHO * gauss_raw(p, q, UU, VV, C20v, C11v, C02v)
                     for (p, q) in moment_indices(order)])


def moment_model(name, order, closure, *, robust=False):
    """Build through the final typed moment facade with the oracle's exact guard mode."""
    return (CartesianVelocityMoments(order, closure=closure, robust=robust)
            .add_transport()
            .build(name=name, frame=MOMENT_FRAME))


print("== (1) convention d'ordre des variables ==")
chk(moment_names(4) == ["M00", "M10", "M20", "M30", "M40",
                        "M01", "M11", "M21", "M31",
                        "M02", "M12", "M22",
                        "M03", "M13",
                        "M04"],
    "moment_names(4) == 15 noms canoniques (contrat adc_cases/hyqmom15)")
chk(len(moment_indices(2)) == 6 and len(moment_indices(3)) == 10,
    "tailles 6 (ordre 2) et 10 (ordre 3)")

print("== (2) oracle d'Isserlis : gaussian_closure -> flux == moments gaussiens decales ==")
for order in (2, 3, 4):
    mg = moment_model("g%d" % order, order, gaussian_closure(order))
    U = gauss_state(order)
    emax = 0.0
    for d, shift in ((0, (1, 0)), (1, (0, 1))):
        F = np.asarray(mg.flux_value(U, {}, mg.frame.axes[d])).ravel()
        Fref = np.array([RHO * gauss_raw(p + shift[0], q + shift[1],
                                         UU, VV, C20v, C11v, C02v)
                         for (p, q) in moment_indices(order)])
        emax = max(emax, (np.abs(F - Fref) / np.maximum(np.abs(Fref), 1e-12)).max())
    chk(emax < 1e-12, f"ordre {order} ({len(U)} vars) : err rel max = {emax:.2e}")

print("== (3) fermeture non triviale : de-standardisation, miroir numpy a la main ==")


def funky_closure(S):
    return {"S30": 0.7 * S["S11"] + 0.2, "S21": S["S11"] * S["S11"] - 0.1,
            "S12": -0.4 * S["S11"], "S03": 1.1}


mf = moment_model("funky", 2, funky_closure)
U6 = gauss_state(2)
u_, v_ = UU, VV
sx_, sy_ = np.sqrt(C20v), np.sqrt(C02v)
s11_ = C11v / (sx_ * sy_)
C30_ = (0.7 * s11_ + 0.2) * sx_**3
C21_ = (s11_ * s11_ - 0.1) * sx_**2 * sy_
C12_ = (-0.4 * s11_) * sx_ * sy_**2
C03_ = 1.1 * sy_**3
# binomiale inverse a la main : m_30 = u^3 + 3 u C20 + C30, m_21 = u^2 v + v C20 + 2 u C11 + C21...
m30_ = u_**3 + 3 * u_ * C20v + C30_
m21_ = u_ * u_ * v_ + v_ * C20v + 2 * u_ * C11v + C21_
m12_ = u_ * v_ * v_ + u_ * C02v + 2 * v_ * C11v + C12_
m03_ = v_**3 + 3 * v_ * C02v + C03_
Fx = np.asarray(mf.flux_value(U6, {}, mf.frame.axes[0])).ravel()
Fy = np.asarray(mf.flux_value(U6, {}, mf.frame.axes[1])).ravel()
Fx_ref = np.array([U6[1], U6[2], RHO * m30_, U6[4], RHO * m21_, RHO * m12_])
Fy_ref = np.array([U6[3], U6[4], RHO * m21_, U6[5], RHO * m12_, RHO * m03_])
e3 = max(np.abs(Fx - Fx_ref).max(), np.abs(Fy - Fy_ref).max())
chk(e3 < 1e-13, f"flux ordre 2, fermeture polynomiale arbitraire : err max = {e3:.2e}")

print("== (3b) standardisation d'entree p+q >= 3 : fermeture-copie, etat asymetrique ==")
# Etat = melange de 2 gaussiennes (poids 0.4/0.6, vitesses opposees) : moments centres
# IMPAIRS non nuls. Indispensable : sur un etat gaussien C30=C21=C12=C03=0 et toute la
# couche S d'ordre 3 disparait des flux (c'est ce qui rendait (2)-(3) aveugles ici).
W1, W2 = 0.4, 0.6
G1 = (0.9, -0.3, 0.7, 0.10, 0.5)   # (u, v, C20, C11, C02) composante 1
G2 = (-0.5, 0.45, 1.2, -0.20, 0.8)
mix = {pq: W1 * gauss_raw(pq[0], pq[1], *G1) + W2 * gauss_raw(pq[0], pq[1], *G2)
       for pq in moment_indices(4)}
U10 = RHO * np.array([mix[pq] for pq in moment_indices(3)])


def copy_closure(S):
    # chaque cle d'ordre 4 recopie une cle d'ordre 3 DISTINCTE : tout exposant (p, q) de la
    # standardisation d'entree se retrouve, croise, dans la de-standardisation de sortie.
    return {"S40": S["S30"], "S31": S["S21"], "S22": S["S12"],
            "S13": S["S03"], "S04": S["S11"]}


mc = moment_model("probe_s", 3, copy_closure)
# miroir a la main : centres directs (formules classiques), standardisation, copie,
# de-standardisation, binomiale inverse d'ordre 4 (lineaire en C, ecrite terme a terme).
um, vm = mix[(1, 0)], mix[(0, 1)]
K20 = mix[(2, 0)] - um * um
K11 = mix[(1, 1)] - um * vm
K02 = mix[(0, 2)] - vm * vm
K30 = mix[(3, 0)] - 3 * um * mix[(2, 0)] + 2 * um**3
K21 = mix[(2, 1)] - 2 * um * mix[(1, 1)] - vm * mix[(2, 0)] + 2 * um * um * vm
K12 = mix[(1, 2)] - 2 * vm * mix[(1, 1)] - um * mix[(0, 2)] + 2 * um * vm * vm
K03 = mix[(0, 3)] - 3 * vm * mix[(0, 2)] + 2 * vm**3
chk(min(abs(K30), abs(K21), abs(K12), abs(K03)) > 1e-3,
    "garde de discrimination : etat reellement asymetrique (|C3*| > 1e-3)")
sxm, sym = np.sqrt(K20), np.sqrt(K02)
s11m = K11 / (sxm * sym)
s30m, s21m = K30 / sxm**3, K21 / (sxm**2 * sym)
s12m, s03m = K12 / (sxm * sym**2), K03 / sym**3
K40 = s30m * sxm**4
K31 = s21m * sxm**3 * sym
K22 = s12m * sxm**2 * sym**2
K13 = s03m * sxm * sym**3
K04 = s11m * sym**4
m40 = um**4 + 6 * um * um * K20 + 4 * um * K30 + K40
m31 = um**3 * vm + 3 * um * um * K11 + 3 * um * vm * K20 + 3 * um * K21 + vm * K30 + K31
m22 = (um * um * vm * vm + vm * vm * K20 + um * um * K02 + 4 * um * vm * K11
       + 2 * vm * K21 + 2 * um * K12 + K22)
m13 = um * vm**3 + 3 * vm * vm * K11 + 3 * um * vm * K02 + 3 * vm * K12 + um * K03 + K13
m04 = vm**4 + 6 * vm * vm * K02 + 4 * vm * K03 + K04
top_ref = {(4, 0): m40, (3, 1): m31, (2, 2): m22, (1, 3): m13, (0, 4): m04}
e3b = 0.0
for d, shift in ((0, (1, 0)), (1, (0, 1))):
    F = np.asarray(mc.flux_value(U10, {}, mc.frame.axes[d])).ravel()
    Fref = np.array([RHO * (top_ref[(p + shift[0], q + shift[1])]
                            if p + q == 3 else mix[(p + shift[0], q + shift[1])])
                     for (p, q) in moment_indices(3)])
    e3b = max(e3b, (np.abs(F - Fref) / np.maximum(np.abs(Fref), 1e-12)).max())
chk(e3b < 1e-12, f"flux fermeture-copie == miroir manuel sur etat asymetrique ({e3b:.2e})")

print("== (4) strategie de vitesses : descripteur final ==")
hierarchy = CartesianVelocityMoments(2, closure=gaussian_closure(2)).hierarchy()
chk(hierarchy.speeds.options()["kind"] == ExactSpeeds.EXACT_EIGENVALUES,
    "la hierarchie finale selectionne le spectre jacobien exact")
chk(hierarchy.speeds.capabilities().to_dict()["exact_speeds"] is True,
    "ExactSpeeds declare la capacite native de vitesses signees")

print("== (6) robust : planchers lisses ==")
mr = moment_model("g2rob", 2, gaussian_closure(2), robust=True)
m0 = moment_model("g2raw", 2, gaussian_closure(2), robust=False)
Uvac = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
with np.errstate(all="ignore"):
    Fvac_r = np.asarray(mr.flux_value(Uvac, {}, mr.frame.axes[0])).ravel()
    Fvac_0 = np.asarray(m0.flux_value(Uvac, {}, m0.frame.axes[0])).ravel()
chk(np.isfinite(Fvac_r).all(), "robust : flux FINI sur etat vide (M00 = 0)")
chk(not np.isfinite(Fvac_0).all(), "nu : flux NON fini sur etat vide (0/0, par construction)")
U = gauss_state(2)
Fh_r = np.asarray(mr.flux_value(U, {}, mr.frame.axes[0])).ravel()
Fh_0 = np.asarray(m0.flux_value(U, {}, m0.frame.axes[0])).ravel()
eh = (np.abs(Fh_r - Fh_0) / np.maximum(np.abs(Fh_0), 1e-12)).max()
chk(eh < 1e-10, f"robust == nu sur etat sain a 1e-10 ({eh:.2e})")

print("== (7) lorentz_sources : table ordre 2 a la main, en flottants purs ==")
Mf = {pq: float(k + 2) * (0.5 + 0.1 * k) for k, pq in enumerate(moment_indices(2))}
qm, oc, ex, ey = 1.7, -0.6, 0.3, 0.9
src = lorentz_sources(Mf, ex, ey, qm, oc)
expected = [
    0.0,
    qm * ex * Mf[(0, 0)] + oc * Mf[(0, 1)],
    qm * 2 * ex * Mf[(1, 0)] + oc * 2 * Mf[(1, 1)],
    qm * ey * Mf[(0, 0)] - oc * Mf[(1, 0)],
    qm * (ex * Mf[(0, 1)] + ey * Mf[(1, 0)]) + oc * (Mf[(0, 2)] - Mf[(2, 0)]),
    qm * 2 * ey * Mf[(0, 1)] - oc * 2 * Mf[(1, 1)],
]
e7 = max(abs(a - b) for a, b in zip(src, expected, strict=True))
chk(e7 < 1e-14, f"6 termes == table manuelle (err {e7:.1e})")
chk(len(lorentz_sources({pq: 1.0 for pq in moment_indices(4)}, ex, ey, qm, oc)) == 15,
    "ordre 4 : hierarchie fermee (15 termes, aucune cle hors variables transportees)")

print("== (7b) maxwellian_moments / bgk_source : point fixe et gaussianisation ==")
# Point fixe : les moments BRUTS d'une gaussienne (oracle de Stein, gauss_raw) SONT ceux de
# sa maxwellienne -- maxwellian_moments doit les rendre a l'identique, et la source BGK doit
# s'annuler. On le verifie aux ordres 2, 3 et 4 en flottants purs (pas de compilateur).
for order in (2, 3, 4):
    idxo = moment_indices(order)
    Mg = {pq: RHO * gauss_raw(pq[0], pq[1], UU, VV, C20v, C11v, C02v) for pq in idxo}
    meq = maxwellian_moments(Mg)
    efp = max(abs(meq[k] - Mg[pq]) for k, pq in enumerate(idxo))
    chk(efp < 1e-12, f"ordre {order} : maxwellian_moments(gaussienne) == elle-meme ({efp:.1e})")
    s = bgk_source(Mg, 7.0)
    chk(max(abs(float(x)) for x in s) < 1e-12,
        f"ordre {order} : source BGK nulle a l'equilibre ({max(abs(float(x)) for x in s):.1e})")
    inv = [s[k] for k, pq in enumerate(idxo) if pq in ((0, 0), (1, 0), (0, 1))]
    chk(all(float(x) == 0.0 for x in inv),
        f"ordre {order} : invariants collisionnels M00/M10/M01 identiquement 0")

# Gaussianisation : sur un etat ASYMETRIQUE (le melange de 2 gaussiennes de (3b), moments
# centres impairs non nuls), la maxwellienne CONSERVE masse/moyenne/covariance (ordres <= 2)
# et MODIFIE les ordres >= 3 (elle annule les moments centres impairs). Etat gaussien =
# aveugle ici (M_eq == M partout), d'ou le melange.
Mne = {pq: RHO * mix[pq] for pq in moment_indices(4)}
meq = maxwellian_moments(Mne)
idx4 = moment_indices(4)
elow = max(abs(meq[k] - Mne[pq]) for k, pq in enumerate(idx4) if pq[0] + pq[1] <= 2)
dhi = max(abs(meq[k] - Mne[pq]) for k, pq in enumerate(idx4) if pq[0] + pq[1] >= 3)
chk(elow < 1e-12, f"melange asymetrique : ordres <= 2 (masse/moyenne/cov) conserves ({elow:.1e})")
chk(dhi > 1e-3, f"melange asymetrique : ordres >= 3 gaussianises (ecart {dhi:.2e} > 1e-3)")
sne = bgk_source(Mne, 3.0)
chk(all(float(sne[k]) == 0.0 for k, pq in enumerate(idx4)
        if pq in ((0, 0), (1, 0), (0, 1))),
    "melange asymetrique : invariants collisionnels toujours 0 (masse/qdm conservees)")

print("== (8) gardes ==")
msg = err_msg(lambda: CartesianVelocityMoments(1, closure=gaussian_closure(1)))
chk("must be an int >= 2" in msg, f"order=1 refuse avec le contrat exact ({msg[:42]}...)")
msg = err_msg(lambda: moment_model("bad2", 2, lambda S: {"S30": 0.0}))
chk("S30" in msg, f"fermeture incomplete refusee ({msg[:42]}...)")

cxx = _default_cxx(None)
if not cxx:
    print("pas de compilateur C++ : test (9) saute")
    print("FAILS =", fails)
    sys.exit(1 if fails else 0)

print("== (9) Case -> validate -> resolve -> compile -> bind -> run (HLL) ==")
try:
    model = moment_model("g2sys", 2, gaussian_closure(2))
    state = model.states["U"]
    flux = model.fluxes["transport"]
    rate = model.operators["transport"]

    # The model emits signed speeds from the exact Jacobian eigenvalue route.  Pin the
    # discretization to that same typed provider so HLL cannot silently fall back to a
    # Rusanov majorant or another speed source.
    provider = provider_of(model)
    chk(provider is not None and provider.kind == "jacobian",
        "le modele declare des vitesses signees derivees du jacobien")
    requested_speeds = FromJacobian(eig=provider.options()["eig"])
    chk(requested_speeds.options() == provider.options(),
        "HLL consomme exactement le fournisseur de vitesses du modele")

    case = pops.Case("g2sys-native-case")
    block = case.block("mom", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(
        rate,
        FiniteVolume(
            flux=flux,
            variables=variables.Conservative(state),
            reconstruction=reconstruction.FirstOrder(),
            riemann=HLL(waves=requested_speeds),
        ),
    )
    case.numerics(numerics, block=block)
    program = ForwardEuler(block[state], rate=rate)
    program.step_strategy(FixedDt(5.0e-4))
    case.program(program)
    layout = Uniform(
        CartesianGrid(
            frame=model.frame,
            cells=(16, 16),
            periodic=PeriodicAxes(model.frame.axes),
        )
    )
    resolved = pops.resolve(
        pops.validate(case),
        layout=layout,
        backend=Production(),
        compile_options={"include": INCLUDE, "cxx": cxx},
    )
    artifact = pops.compile(resolved)
    artifact.verify()
    chk(len(artifact.blocks) == 1, "lifecycle final : un composant de bloc compile")
except RuntimeError as ex:
    if "Kokkos" in str(ex):
        print("Kokkos introuvable (POPS_KOKKOS_ROOT) : test (9) saute --", str(ex)[:60])
        print("FAILS =", fails)
        sys.exit(1 if fails else 0)
    raise
n = 16
x = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(x, x, indexing="ij")
pert = 1.0 + 0.1 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
U0 = gauss_state(2)[:, None, None] * pert[None, :, :]
sim = pops.bind(artifact, initial_state={"mom": U0})
report = pops.run(sim, t_end=5.0e-3, max_steps=10)
chk(report.accepted_steps == 10, "lifecycle final : exactement 10 pas acceptes")
out = np.array(sim.state_global("mom"))
chk(np.isfinite(out).all(), "10 pas HLL : etat fini")
dm = abs(out[0].sum() - U0[0].sum()) / abs(U0[0].sum())
chk(dm < 1e-12, f"masse conservee ({dm:.2e})")

print("FAILS =", fails)
sys.exit(1 if fails else 0)
