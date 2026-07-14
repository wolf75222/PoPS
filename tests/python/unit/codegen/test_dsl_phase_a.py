"""DSL Phase A : l'API utilisateur stable (Model facade + Param + CompiledModel + add_equation +
FiniteVolume + run). PUR-PYTHON au-dessus de HyperbolicModel : aucune numerique nouvelle. cf.
docs/DSL_MODEL_DESIGN.md.

Deux niveaux :
(1) PUR-PYTHON (aucun compilateur requis) : Param nomme + runtime supporte (P7-b), flux vs eval_flux distincts,
    primitive_vars kwargs (layout ordonne, rho conservatif rejoint le layout sans etre redefini),
    FiniteVolume(riemann=), et les erreurs explicites (backend/target inconnus, hllc sans pression,
    remplacement de noms interdit sur un package natif).
(2) BOUT EN BOUT (saute si pas de compilateur / en-tetes) : compile(backend="production") ->
    CompiledModel -> add_equation -> chemin natif add_native_block. Le seam bas niveau refuse
    volontairement ``run`` hors ``pops.bind`` ; le test avance donc explicitement par ``step_cfl``.
"""
from pops.numerics.riemann import HLLC
from pops.numerics.reconstruction.limiters import Minmod
from pops.numerics.variables import Primitive
from pops.numerics.reconstruction import WENO5
import os
import shutil
import tempfile

import numpy as np

import pops
import pops.runtime._engine_descriptors as engine
from pops.codegen.loader import CompiledModel
from pops.math import sqrt
from pops.physics._facade import Model
from pops.params import ConstParam, RuntimeParam

from tests.python.support.initial_states import euler_bubble_state
from tests.python.support.requirements import repo_include
from pops.runtime._system import System  # ADC-545 advanced runtime seam
INCLUDE = repo_include()
GAMMA = 1.6667


def build_euler(name="euler_pa"):
    """Euler 2D ecrit via la FACADE Model (kwargs primitive_vars, param gamma nomme)."""
    m = Model(name)
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.value(m.param(ConstParam("gamma", GAMMA)))                       # Param NOMME, inline au codegen + set_gamma
    u = rhou / rho
    v = rhov / rho
    p = (g - 1.0) * (E - 0.5 * rho * (u * u + v * v))
    H = (E + p) / rho
    c = sqrt(g * p / rho)
    m.flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
           y=[rhov, rhov * u, rhov * v + p, rho * H * v])   # DECLARATEUR
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    # KWARGS : rho conservatif rejoint le layout ; renvoie les Var PRIMITIVES (rho/u/v/p comme
    # locales primitives), a utiliser dans conservative_from (qui exprime cons EN FONCTION des prim).
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)
    m.conservative_from([prho, prho * pu, prho * pv,
                         pp / (g - 1.0) + 0.5 * prho * (pu * pu + pv * pv)])
    # ADC-590 : riemann='hllc' generique exige la capability EMISE (plus de fallback Euler implicite).
    m.enable_hllc()
    return m


def build_euler_predef(name="euler_predef"):
    """IDENTIQUE a build_euler, mais u/v/p sont des Var PRIMITIVES deja definies (m.primitive(...))
    passees en SELF-REFERENCE a primitive_vars(rho=rho, u=u, v=v, p=p). C'est le style cible avec des
    Var pre-definies : sans le garde-fou self-ref, u=u redefinirait la primitive en `const Real u = u;`
    (auto-init -> NaN). Doit produire le MEME modele que build_euler (formes equivalentes)."""
    m = Model(name)
    rho, rhou, rhov, E = m.conservative_vars(
        "rho", "rho_u", "rho_v", "E",
        roles=["Density", "MomentumX", "MomentumY", "Energy"])
    g = m.value(m.param(ConstParam("gamma", GAMMA)))
    u = m.primitive("u", rhou / rho)                  # Var PRIMITIVE deja definie
    v = m.primitive("v", rhov / rho)
    p = m.primitive("p", (g - 1.0) * (E - 0.5 * rho * (u * u + v * v)))
    H = (E + p) / rho
    c = sqrt(g * p / rho)
    m.flux(x=[rhou, rhou * u + p, rhou * v, rho * H * u],
           y=[rhov, rhov * u, rhov * v + p, rho * H * v])
    m.eigenvalues(x=[u - c, u, u + c], y=[v - c, v, v + c])
    prho, pu, pv, pp = m.primitive_vars(rho=rho, u=u, v=v, p=p)   # u=u : Var primitive self-ref
    m.conservative_from([prho, prho * pu, prho * pv,
                         pp / (g - 1.0) + 0.5 * prho * (pu * pu + pv * pv)])
    m.enable_hllc()  # ADC-590 : meme capability que build_euler (les deux formes restent le MEME modele)
    return m


def initial_state(n):
    return euler_bubble_state(n, GAMMA)


def expect_raises(exc, fn, label):
    try:
        fn()
    except exc:
        print("OK  %s : %s levee" % (label, exc.__name__))
        return
    raise AssertionError("%s : %s attendue, non levee" % (label, exc.__name__))


def advance_low_level(system, *, t_end, cfl):
    """Advance an explicitly assembled low-level engine after pinning the public run guard."""
    expect_raises(
        RuntimeError,
        lambda: system.run(
            t_end=t_end, max_steps=100000, strategy=pops.time.AdaptiveCFL(cfl)),
        "run bas niveau hors transaction pops.bind refuse",
    )
    nsteps = 0
    while system.time() < t_end:
        system.step_cfl(cfl)
        nsteps += 1
    return nsteps


def pure_python_checks():
    # Declarations explicites + identites de handles ; runtime supporte (P7-b).
    m = build_euler()
    g = m.params["gamma"]
    assert isinstance(g, ConstParam) and g.name == "gamma" and abs(g.value - GAMMA) < 1e-12
    # P7-b : les parametres runtime sont desormais implementes (cf. test_dsl_runtime_params). L'ancienne
    # assertion "runtime rejete -> NotImplementedError" etait perimee depuis l'arrivee de la feature et
    # echouait en silence (CI auto-decouverte avalant l'echec, cf. ADC-104).
    kp = m.param(RuntimeParam("kappa", default=1.0))
    assert kp.param_kind == "runtime" and kp.local_id == "kappa"
    assert m.value(kp) is not kp
    print("OK  declarations explicites + handles distincts des Expr")

    # flux declarateur vs eval_flux evaluateur : noms distincts, methodes distinctes
    assert m.flux is not m.eval_flux, "flux et eval_flux doivent etre distincts"
    assert m._m._flux.get("x"), "m.flux(...) a bien declare le flux x"
    print("OK  m.flux (declarateur) != m.eval_flux (evaluateur)")

    # primitive_vars kwargs : layout ordonne [rho,u,v,p] ; rho (conservatif) PAS redefini en primitive
    assert m.prim_state == ["rho", "u", "v", "p"], "layout primitif kwargs : %r" % m.prim_state
    assert "rho" not in m._m.prim_defs, "rho conservatif ne doit pas etre redefini comme primitive"
    assert "u" in m._m.prim_defs and "p" in m._m.prim_defs, "u/p definis comme primitives"
    print("OK  primitive_vars kwargs : layout ordonne, rho conservatif rejoint le layout")

    # FiniteVolume : riemann (PAS flux) -> Spatial.flux ; variables -> recon
    fv = engine.Spatial(limiter=Minmod(), flux=HLLC(), recon=Primitive())
    assert fv.flux == "hllc" and fv.limiter == "minmod" and fv.recon == "primitive", \
        "FiniteVolume(riemann=) -> Spatial.flux"
    print("OK  FiniteVolume(limiter=, riemann=, variables=) remappe sur Spatial")

    # compile : backend et target inconnus sont rejetes AVANT toute compilation.
    expect_raises(ValueError, lambda: m.compile("x.so", INCLUDE, backend="bogus"),
                  "backend inconnu")
    expect_raises(ValueError,
                  lambda: m.compile("x.so", INCLUDE, backend="production", target="bogus"),
                  "target inconnu")

    # add_equation : erreurs sur un CompiledModel FACTICE (pas de .so reel necessaire, les gardes
    # levent AVANT la frontiere C++).
    sys = System(n=16, periodic=True)
    fake = CompiledModel(so_path="/inexistant.so", backend="production", adder="add_native_block",
                         cons_names=["rho", "rho_u", "rho_v", "E"],
                         cons_roles=["Density", "MomentumX", "MomentumY", "Energy"],
                         prim_names=["rho", "u", "v"],  # PAS de 'p' -> hllc/roe doit lever
                         n_vars=4, gamma=GAMMA, n_aux=3, params={}, caps={},
                         abi_key="k", model_hash="h", cxx="c++", std="c++20")
    # WENO5 est accepte par le package natif : il passe la garde Python et echoue seulement au dlopen
    # du package factice.
    expect_raises(RuntimeError, lambda: sys.add_equation("g", fake,
                  spatial=engine.Spatial(limiter=WENO5())),
                  "weno5 production : accepte (echec au dlopen)")
    expect_raises(ValueError, lambda: sys.add_equation("g", fake,
                  spatial=engine.Spatial(flux=HLLC())), "hllc sans pression")
    expect_raises(ValueError, lambda: sys.add_equation("g", fake, names=["x"]),
                  "names= sur production natif")
    print("OK  add_equation production : weno5 accepte, hllc sans p et names= rejetes")


def end_to_end_checks():
    n = 32
    tmp = tempfile.mkdtemp()
    try:
        m = build_euler("euler_production")
        cm = m.compile(os.path.join(tmp, "m_production.so"), INCLUDE, backend="production")
        assert isinstance(cm, CompiledModel), "compile -> CompiledModel"
        assert cm.backend == "production" and cm.adder == "add_native_block"
        assert cm.n_vars == 4 and abs((cm.gamma or 0) - GAMMA) < 1e-12
        assert cm.abi_key and cm.model_hash, "abi_key + model_hash presents"
        assert "gamma" in cm.params, "params porte le Param gamma"
        print("OK  production : compile -> CompiledModel(add_native_block)")

        s = System(n=n, periodic=True)
        s.add_equation("gas", cm, spatial=engine.Spatial(limiter=Minmod(), flux=HLLC(),
                                                           recon=Primitive()))
        s.set_poisson(rhs="charge_density", solver="geometric_mg")
        s.set_state("gas", initial_state(n))
        nsteps = advance_low_level(s, t_end=0.02, cfl=0.4)
        assert nsteps > 0, "run a avance"
        final = np.array(s.get_state("gas"))
        assert np.all(np.isfinite(final)), "production : etat fini"
        print("OK  production : add_equation + run(%d pas) -> etat fini" % nsteps)

        # Garde-fou self-ref kwargs (style cible avec Var pre-definies) : u/v/p definies par m.primitive
        # puis passees en primitive_vars(rho=rho, u=u, v=v, p=p). Doit (a) ne PAS produire de NaN
        # (sans le fix, u=u -> `Real u = u;` auto-init) et (b) donner le MEME modele que la forme expr.
        mp = build_euler_predef("euler_predef")
        cmp_ = mp.compile(os.path.join(tmp, "m_predef.so"), INCLUDE, backend="production")
        sp = System(n=n, periodic=True)
        sp.add_equation("gas", cmp_, spatial=engine.Spatial(limiter=Minmod(), flux=HLLC(),
                                                              recon=Primitive()))
        sp.set_poisson(rhs="charge_density", solver="geometric_mg")
        sp.set_state("gas", initial_state(n))
        advance_low_level(sp, t_end=0.02, cfl=0.4)
        pf = np.array(sp.get_state("gas"))
        assert np.all(np.isfinite(pf)), "primitive_vars kwargs (Var pre-definies) : etat fini, pas de NaN"
        dp = float(np.max(np.abs(pf - final)))
        assert dp < 1e-10, "primitive_vars(u=u) Var pre-definie == forme expr (meme modele), dmax=%.3e" % dp
        print("OK  primitive_vars kwargs Var pre-definies : pas de NaN, == forme expr (dmax=%.3e)" % dp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def modelspec_substeps_check():
    """substeps= doit etre forwarde pour un ModelSpec (pas seulement pour un CompiledModel) : la
    branche ModelSpec d'add_equation appelle _s.add_block DIRECTEMENT avec nsub (pas self.add_block,
    qui retomberait sur time.substeps et IGNORERAIT l'override). Verifie via un espion sur _s.add_block."""
    s = System(n=16, periodic=True)
    spec = engine.Model(state=engine.FluidState("isothermal", cs2=1.0), transport=engine.IsothermalFlux(),
                     source=engine.NoSource(), elliptic=engine.ChargeDensity(charge=-1.0))
    calls = []

    class _Spy:
        def add_block(self, *a):
            calls.append(a)

    s._s = _Spy()
    # _s.add_block positional : (name, model, limiter, flux, recon, time_kind, substeps, evolve)
    s.add_equation("ions", spec, time=engine.Explicit(), substeps=10)
    assert calls, "add_equation(ModelSpec) doit appeler _s.add_block"
    assert calls[0][6] == 10, "substeps= ignore pour ModelSpec : recu %r" % (calls[0][6],)
    calls.clear()
    s.add_equation("ions2", spec, time=engine.Explicit(substeps=3))   # defaut = time.substeps
    assert calls[0][6] == 3, "defaut substeps != time.substeps : recu %r" % (calls[0][6],)
    print("OK  substeps= override forwarde pour ModelSpec (10) ; defaut = time.substeps (3)")


def predef_primitive_selfref_check():
    """primitive_vars(rho=rho, u=u, v=v, p=p) avec u/v/p des Var PRIMITIVES deja definies (m.primitive).
    Le garde-fou self-ref ne doit PAS redefinir u en `u = u` (auto-init NaN) : prim_defs garde la
    formule d'origine (rho_u/rho), pas un renvoi a soi. Pur-Python (aucun compilateur requis)."""
    m = build_euler_predef("euler_predef_pp")
    pd = m._m.prim_defs
    for nm in ("u", "v", "p"):
        assert nm in pd, "primitive '%s' absente de prim_defs" % nm
        assert pd[nm].to_cpp() != nm, \
            "primitive '%s' auto-initialisee (self-ref kwargs mal gere : `%s = %s;`)" % (nm, nm, nm)
    assert "rho_u" in pd["u"].deps(), "primitive 'u' doit garder sa formule (depend de rho_u)"
    print("OK  primitive_vars kwargs Var pre-definies : pas d'auto-init (formules prim_defs preservees)")


def main():
    pure_python_checks()
    predef_primitive_selfref_check()
    modelspec_substeps_check()
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx or not os.path.isdir(INCLUDE):
        print("skip  bout-en-bout (compilateur ou en-tetes pops absents)")
    else:
        end_to_end_checks()
    print("test_dsl_phase_a : tout est vert")


if __name__ == "__main__":
    main()
