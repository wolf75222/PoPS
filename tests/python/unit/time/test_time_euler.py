#!/usr/bin/env python3
"""Methode temporelle explicite "euler" (ForwardEuler, ordre 1) -- ADC-174.

Motivation : fidelite aux references au premier ordre (RIEMOM2D : split dimensionnel
additif + Euler == Euler non-splitte, algebriquement Mx+My-M = M+dt(Lx+Ly)) ; le seul
ecart de schema d'un replay vs ces references est l'etage 2 de ssprk2. ``euler`` reste un
schema public explicite ; ssprk2 reste le defaut (no-default-change verifie ici).

On verifie :
 (1) facade : Explicit() -> kind 'explicit' (defaut intact) ; Explicit(method='euler') ->
     kind 'euler' ; methode inconnue rejetee avec la liste a jour ;
 (2) IDENTITE DE SHU-OSHER, bit-exacte : un pas ssprk2 == 0.5 U0 + 0.5 euler(euler(U0)) --
     plus fort qu'un test d'ordre : prouve que 'euler' est EXACTEMENT l'operateur d'etage
     de ssprk2 (memes rhs, memes ghosts), donc ordre 1 par construction ;
 (3) no-default-change : un pas Explicit() == un pas Explicit(method='ssprk2') bit-exact ;
 (4) garde de discrimination : euler != ssprk2 sur le meme pas (le test (2) ne compare pas
     deux choses egales par accident) ;
 (5) AmrSystem : time='euler' porte par le chemin Forward-Euler AMR existant, avec sa relation
     temporelle explicite -- pas de rabattement silencieux vers SSPRK2 ;
 (6) [compilateur] le package natif backend='production' (add_native_block, gabarit
     add_compiled_model -> make_block) porte euler avec la meme identite de Shu-Osher.

Modele natif pur transport (isotherme sans source ni Poisson) pour (1)-(5) : aucun
compilateur requis ; (6) s'auto-saute sans compilateur ou sans Kokkos.
"""
from pops.numerics.reconstruction import FirstOrder
from pops.numerics.riemann import Rusanov
import sys
import tempfile

import numpy as np
import pops
from pops.codegen import Production
from pops.domain import Rectangle
from pops.frames import Cartesian2D
from pops.layouts import Uniform
from pops.math import ddt, div
from pops.mesh import CartesianGrid, PeriodicAxes
from pops.numerics import DiscretizationPlan, reconstruction, riemann, variables
from pops.numerics.spatial import FiniteVolume
from pops.physics import Model as BoardModel
from pops.numerics.terms import Flux as FinalFlux, DefaultSource as FinalDefaultSource

import pops.runtime._engine_descriptors as engine
from pops.codegen.toolchain import _default_cxx
from pops.physics._facade import Model
from pops.runtime._system import AmrSystem, System  # ADC-545 advanced runtime seam
from tests.python.support.requirements import repo_include

fails = 0


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


def transport_model():
    # NoSource + fond discret neutralisant : l'avance est un PUR transport, condition de
    # l'identite de Shu-Osher du test (2), même si le runtime resout le champ periodique.
    return engine.Model(state=engine.FluidState("isothermal", cs2=0.5),
                     transport=engine.IsothermalFlux(),
                     source=engine.NoSource(),
                     elliptic=engine.BackgroundDensity(alpha=1.0, n0=1.0))


def make_sim(method):
    n = 24
    sim = System(n=n, L=1.0, periodic=True)
    sim.add_equation("ions", transport_model(),
                  spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                  time=engine.Explicit(method=method))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_state("ions", np.stack([rho, 0.4 * rho, -0.2 * rho]))
    return sim


print("== (1) facade ==")
chk(engine.Explicit().kind == "explicit", "Explicit() -> kind 'explicit' (defaut intact)")
chk(engine.Explicit(method="euler").kind == "euler", "Explicit(method='euler') -> kind 'euler'")
msg = err_msg(lambda: engine.Explicit(method="rk4"))
chk("'euler'" in msg, f"methode inconnue rejetee, liste a jour ({msg[:48]}...)")

print("== (2) identite de Shu-Osher : ssprk2 == 0.5 U0 + 0.5 euler(euler(.)) ==")
dt = 2e-3
s2 = make_sim("ssprk2")
se = make_sim("euler")
U0 = np.array(se.get_state("ions"))
s2.step(dt)
se.step(dt)
se.step(dt)
ref = 0.5 * U0 + 0.5 * np.array(se.get_state("ions"))
got = np.array(s2.get_state("ions"))
bit = np.array_equal(got, ref)
emax = np.abs(got - ref).max()
chk(bit or emax < 1e-15, f"identite bit-exacte (array_equal={bit}, err max {emax:.1e})")
if not bit:
    print("      NOTE : pas bit-exact mais < 1e-15 -- ordre des flottants de lincomb")

print("== (3) no-default-change : defaut == ssprk2 ==")
sd = make_sim("ssprk2")
s_def = System(n=24, L=1.0, periodic=True)
s_def.add_equation("ions", transport_model(),
                   spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                   time=engine.Explicit())
s_def.set_state("ions", np.array(sd.get_state("ions")))
sd.step(dt)
s_def.step(dt)
chk(np.array_equal(np.array(sd.get_state("ions")), np.array(s_def.get_state("ions"))),
    "Explicit() et Explicit(method='ssprk2') bit-identiques")

print("== (4) garde de discrimination ==")
s2b = make_sim("ssprk2")
seb = make_sim("euler")
s2b.step(dt)
seb.step(dt)
d = np.abs(np.array(s2b.get_state("ions")) - np.array(seb.get_state("ions"))).max()
chk(d > 1e-8, f"euler != ssprk2 sur un pas (ecart max {d:.2e})")

def make_amr_sim(method):
    n = 24
    sim = AmrSystem(n=n, L=1.0, periodic=True, regrid_every=0)
    sim.set_temporal_relations([2], [1], ["integral_only"])
    sim.add_equation(
        "ions",
        transport_model(),
        spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
        time=engine.Explicit(method=method),
    )
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    rho = 1.0 + 0.3 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    sim.set_density("ions", rho.ravel())
    return sim, rho


print("== (5) AMR : Euler explicite execute avec relation temporelle ==")
amr_euler, amr_rho0 = make_amr_sim("euler")
amr_ssprk2, _ = make_amr_sim("ssprk2")
amr_euler.step(dt)
amr_euler_once = np.asarray(amr_euler.density("ions")).reshape(amr_rho0.shape)
amr_ssprk2.step(dt)
amr_ssprk2_once = np.asarray(amr_ssprk2.density("ions")).reshape(amr_rho0.shape)
chk(
    amr_euler.macro_step() == 1 and abs(amr_euler.time() - dt) < 1e-15,
    "AmrSystem Euler avance exactement un macro-pas",
)
chk(
    np.all(np.isfinite(amr_euler_once)),
    "AmrSystem Euler publie un etat fini apres son premier pas",
)
chk(
    float(np.max(np.abs(amr_euler_once - amr_ssprk2_once))) > 1e-8,
    "AmrSystem Euler ne se rabat pas silencieusement sur SSPRK2",
)
amr_euler.step(dt)
amr_euler_twice = np.asarray(amr_euler.density("ions")).reshape(amr_rho0.shape)
chk(
    np.all(np.isfinite(amr_euler_twice))
    and float(np.max(np.abs(amr_euler_twice - amr_rho0))) > 1e-8,
    "deux pas Euler AMR produisent un transport non trivial et fini",
)
amr_ssprk2_reference = 0.5 * amr_rho0 + 0.5 * amr_euler_twice
amr_relation_error = float(np.max(np.abs(amr_ssprk2_once - amr_ssprk2_reference)))
chk(
    np.array_equal(amr_ssprk2_once, amr_ssprk2_reference)
    or amr_relation_error < 1e-15,
    "identite de Shu-Osher AMR : SSPRK2 == 0.5 U0 + 0.5 Euler(Euler(U0)) "
    f"(err max {amr_relation_error:.1e})",
)

cxx = _default_cxx(None)
if not cxx:
    print("pas de compilateur C++ : test (6) saute")
    print("FAILS =", fails)
    sys.exit(1 if fails else 0)

print("== (6) package natif production : Euler explicite porte ==")
INCLUDE = repo_include()


def adv_model():
    m = Model("eulprod")
    q1, q2 = m.conservative_vars("q1", "q2")
    m.flux(x=[1.5 * q1, -0.7 * q2], y=[-0.7 * q1, 1.5 * q2])
    z = 0.0 * q1  # les listes d'eigenvalues attendent des Expr (pas des flottants nus)
    m.eigenvalues(x=[z - 1.5, z + 1.5], y=[z - 1.5, z + 1.5])
    m.primitive_vars(q1, q2)
    m.conservative_from([q1, q2])
    return m


def _public_adv_artifact(name="eulprod"):
    """Compile the production component through the final typed Case lifecycle."""
    frame = Rectangle("%s-domain" % name, lower=(0.0, 0.0), upper=(1.0, 1.0)).frame(Cartesian2D())
    x_axis, y_axis = frame.axes
    model = BoardModel(name, frame=frame)
    state = model.state("U", components=("q1", "q2"))
    q1, q2 = state
    flux = model.flux(
        "transport", frame=frame, state=state,
        components={x_axis: (1.5 * q1, -0.7 * q2),
                     y_axis: (-0.7 * q1, 1.5 * q2)},
        waves={x_axis: (1.5 + 0.0 * q1,) * 2,
               y_axis: (1.5 + 0.0 * q1,) * 2},
    )
    source = model.source("zero", on=state, value=(0.0 * q1, 0.0 * q2))
    rate = model.rate("explicit_rhs", equation=ddt(state) == -div(flux) + source)
    case = pops.Case("%s-case" % name)
    block = case.block("q", model)
    numerics = DiscretizationPlan()
    numerics.rates.add(rate, FiniteVolume(
        flux=flux, variables=variables.Conservative(state),
        reconstruction=reconstruction.FirstOrder(), riemann=riemann.Rusanov()))
    case.numerics(numerics, block=block)
    program = __import__("pops").time.Program("%s-program" % name)
    temporal = program.state(block[state])
    rhs = program.rhs(state=temporal.n, terms=[FinalFlux(), FinalDefaultSource()])
    program.commit(temporal.next, program.value(
        "U1", temporal.n + program.dt * rhs, at=temporal.next.point))
    case.program(program)
    layout = Uniform(CartesianGrid(frame=frame, cells=(16, 16), periodic=PeriodicAxes(frame.axes)))
    resolved = pops.resolve(pops.validate(case), layout=layout, backend=Production(),
                            compile_options={"include": str(__import__("pathlib").Path(__file__).resolve().parents[4] / "include")})
    return pops.compile(resolved).blocks[0].model


tmp = tempfile.mkdtemp(prefix="pops_euler_")
try:
    prod = _public_adv_artifact()
except RuntimeError as ex:
    if "Kokkos" in str(ex):
        print("Kokkos introuvable : test (6) saute --", str(ex)[:60])
        print("FAILS =", fails)
        sys.exit(1 if fails else 0)
    raise


def make_prod_sim(method):
    n = 16
    sim = System(n=n, L=1.0, periodic=True)
    sim.add_equation("q", model=prod,
                     spatial=engine.Spatial(limiter=FirstOrder(), flux=Rusanov()),
                     time=engine.Explicit(method=method))
    x = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    sim.set_state("q", np.stack([1.0 + 0.3 * np.sin(2 * np.pi * X),
                                 1.0 + 0.2 * np.cos(2 * np.pi * Y)]))
    return sim


p2 = make_prod_sim("ssprk2")
pe = make_prod_sim("euler")
U0p = np.array(pe.get_state("q"))
p2.step(dt)
pe.step(dt)
pe.step(dt)
refp = 0.5 * U0p + 0.5 * np.array(pe.get_state("q"))
gotp = np.array(p2.get_state("q"))
ep = np.abs(gotp - refp).max()
chk(np.array_equal(gotp, refp) or ep < 1e-15,
    f"production : identite de Shu-Osher (err max {ep:.1e})")

print("FAILS =", fails)
sys.exit(1 if fails else 0)
